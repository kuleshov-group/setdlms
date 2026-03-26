import datetime
import importlib
import inspect
import json
import logging
import os
import re
from typing import Any, Optional

import hydra
import mauve
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
)
from transformers.modeling_outputs import ModelOutput

from scripts.utils import (
    count_parameters,
    format_number,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.denoiser.ar import AR, ARConfig
from src.denoiser.diffusion import BD3LM, MDLM, SEDD, BD3LMConfig, MDLMConfig
from src.noise_schedule.noise_schedules import LinearNoise
from src.utils import fsspec_exists, fsspec_mkdirs

log = logging.getLogger(__name__)


def patch_mauve_for_modernbert() -> None:
    """
    Robust monkey patch for mauve so arbitrary HF encoder models
    (including answerdotai/ModernBERT-large) can be used as featurizers.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Import the real installed modules, not package attributes.
    mauve_utils = importlib.import_module("mauve.utils")
    mauve_compute_mod = importlib.import_module("mauve.compute_mauve")

    def _get_device_from_arg(device_id):
        if (
            device_id is not None
            and torch.cuda.is_available()
            and isinstance(device_id, int)
            and 0 <= device_id < torch.cuda.device_count()
        ):
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")

    def _get_tokenizer(model_name="gpt2"):
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # MAUVE later pads batches, so ensure pad_token exists.
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.cls_token is not None:
                tok.pad_token = tok.cls_token
            elif tok.sep_token is not None:
                tok.pad_token = tok.sep_token
            else:
                # last resort
                tok.add_special_tokens({"pad_token": "[PAD]"})
        return tok

    def _get_model(model_name, tokenizer, device_id):
        device = _get_device_from_arg(device_id)

        kwargs = {"trust_remote_code": True}
        if getattr(tokenizer, "pad_token_id", None) is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id

        model = AutoModel.from_pretrained(model_name, **kwargs)
        model = model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def _featurize_tokens_from_model(
        model,
        tokenized_texts,
        batch_size,
        name="",
        verbose=False,
    ):
        """
        For encoder-only models like ModernBERT, use attention-mask-aware
        mean pooling over the last hidden state.
        For decoder-only models, keep last non-pad token pooling.
        """
        device = next(model.parameters()).device
        feats = []

        model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
        is_encoder_only = model_type in {
            "bert",
            "roberta",
            "distilbert",
            "albert",
            "deberta",
            "deberta-v2",
            "modernbert",
        }

        for start in range(0, len(tokenized_texts), batch_size):
            chunk = [t.view(-1) for t in tokenized_texts[start : start + batch_size]]
            sent_lens = [len(x) for x in chunk]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                chunk,
                batch_first=True,
                padding_value=0,
            ).to(device)

            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.ones(n, dtype=torch.long) for n in sent_lens],
                batch_first=True,
                padding_value=0,
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

            last_hidden = outputs.last_hidden_state  # (B, T, H)

            if is_encoder_only:
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(
                    1.0
                )
            else:
                last_idx = attention_mask.sum(dim=1) - 1
                pooled = last_hidden[
                    torch.arange(last_hidden.size(0), device=device),
                    last_idx,
                ]

            feats.append(pooled.cpu())

        return torch.cat(feats, dim=0)

    # Patch mauve.utils
    mauve_utils.get_device_from_arg = _get_device_from_arg
    mauve_utils.get_tokenizer = _get_tokenizer
    mauve_utils.get_model = _get_model
    mauve_utils.featurize_tokens_from_model = _featurize_tokens_from_model

    # Patch mauve.compute_mauve module globals
    mauve_compute_mod.get_device_from_arg = _get_device_from_arg
    mauve_compute_mod.get_tokenizer = _get_tokenizer
    mauve_compute_mod.get_model = _get_model
    mauve_compute_mod.featurize_tokens_from_model = _featurize_tokens_from_model

    # Patch package-level exports too, just in case
    if hasattr(mauve, "get_tokenizer"):
        mauve.get_tokenizer = _get_tokenizer
    if hasattr(mauve, "get_model"):
        mauve.get_model = _get_model
    if hasattr(mauve, "featurize_tokens_from_model"):
        mauve.featurize_tokens_from_model = _featurize_tokens_from_model

    # Reset MAUVE caches if present
    for mod in (mauve_compute_mod, mauve_utils):
        for attr in ("MODEL", "TOKENIZER", "MODEL_NAME"):
            if hasattr(mod, attr):
                setattr(mod, attr, None)

    # Hard verification so failure is obvious immediately.
    src = inspect.getsource(mauve_utils.get_tokenizer)
    assert "AutoTokenizer.from_pretrained" in src, (
        "MAUVE patch did not apply: mauve.utils.get_tokenizer is still original"
    )


# patch_mauve_for_modernbert()

THROUGHPUT_WARMUP = 0


def _summarize_numeric_list(values: list) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(len(arr)),
    }


def build_generation_metrics_dict(
    tputs: list,
    parallelism_factors: list,
    lengths: list,
    entropies: list,
) -> dict[str, Any]:
    """Aggregate generation stats into a flat JSON-serializable dict.

    Same shape as seq2seq_eval metrics.
    """
    tp = _summarize_numeric_list(tputs)
    pf = _summarize_numeric_list(parallelism_factors)
    ln = _summarize_numeric_list(lengths)
    ent = _summarize_numeric_list(entropies)
    out: dict[str, Any] = {
        "throughput_tok_per_s_mean": tp["mean"],
        "throughput_tok_per_s_std": tp["std"],
        "throughput_tok_per_s_count": tp["count"],
        "parallelism_factor_mean": pf["mean"],
        "parallelism_factor_std": pf["std"],
        "parallelism_factor_count": pf["count"],
        "sequence_length_tokens_mean": ln["mean"],
        "sequence_length_tokens_std": ln["std"],
        "sequence_length_tokens_count": ln["count"],
        "entropy_nats_mean": ent["mean"],
        "entropy_nats_std": ent["std"],
        "entropy_nats_count": ent["count"],
    }
    return out


def _use_adlm_compatible_mauve(cfg: DictConfig) -> bool:
    return bool(getattr(cfg, "adlm_compatible_mauve", False))


def _should_reject_generated_sample(
    cfg: DictConfig,
    outputs: torch.LongTensor,
    entropy: list[float],
) -> bool:
    if _use_adlm_compatible_mauve(cfg):
        return False
    return entropy[0] < 4 or outputs.shape[1] <= 50


def _configure_tokenizer_for_adlm_reference(tokenizer: Any) -> Any:
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    if isinstance(tokenizer, (GPT2TokenizerFast, GPT2Tokenizer)):
        import tokenizers

        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )
    return tokenizer


def _group_adlm_style_texts(
    tokenized_examples: dict[str, list[list[int]]],
    block_size: int,
    bos_token_id: int,
    eos_token_id: int,
) -> dict[str, list[list[int]]]:
    concatenated_examples = []
    for ids in tokenized_examples["input_ids"]:
        concatenated_examples.extend(ids)
    new_block_size = block_size - 2
    total_length = (len(concatenated_examples) // new_block_size) * new_block_size

    input_ids = []
    attention_mask = []
    for start in range(0, total_length, new_block_size):
        input_ids.append(
            [bos_token_id]
            + concatenated_examples[start : start + new_block_size]
            + [eos_token_id]
        )
        attention_mask.append([1] * block_size)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_reference_from_adlm_openwebtext_valid(
    cfg: DictConfig,
    tokenizer: Any,
    num_samples: int,
) -> list[str]:
    """Mirror ADLM's OpenWebText-valid MAUVE reference construction."""
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "ADLM-compatible MAUVE requires the `datasets` package to load "
            "OpenWebText validation references."
        ) from exc

    tokenizer = _configure_tokenizer_for_adlm_reference(tokenizer)
    block_size = int(getattr(cfg, "max_length", None) or getattr(cfg, "max_new_tokens", 1024) + 1)
    if block_size <= 0:
        block_size = 1024

    seed = int(getattr(cfg, "adlm_compatible_mauve_valid_seed", getattr(cfg, "seed", 0)))
    eval_batch_size = int(getattr(cfg, "batch_size", 1))
    eval_batch_size = int(getattr(cfg, "adlm_compatible_mauve_eval_batch_size", eval_batch_size))

    raw_dataset = datasets.load_dataset(
        "openwebtext",
        split="train[-100000:]",
    )

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokenized = tokenizer(
            batch["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {
            "input_ids": [
                ids + [tokenizer.eos_token_id] for ids in tokenized["input_ids"]
            ]
        }

    tokenized_dataset = raw_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing ADLM-compatible MAUVE references",
    )
    grouped_dataset = tokenized_dataset.map(
        lambda batch: _group_adlm_style_texts(
            batch,
            block_size=block_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        batched=True,
        desc="Grouping ADLM-compatible MAUVE references",
    )
    grouped_dataset = grouped_dataset.with_format("torch")

    generator = torch.Generator().manual_seed(seed)
    valid_loader = torch.utils.data.DataLoader(
        grouped_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        generator=generator,
    )

    references: list[str] = []
    num_batches = (num_samples + eval_batch_size - 1) // eval_batch_size
    for _ in range(num_batches):
        batch = next(iter(valid_loader))
        input_ids = batch["input_ids"]
        references.extend(tokenizer.batch_decode(input_ids))
    return references[:num_samples]


def gather_results(results, world_size):
    if world_size == 1:
        return results
    # Each GPU has local 'results' (any pickle-able object)
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore

    return all_results


def setup_ddp() -> int:
    """Sets up torch.distributed and selects GPU.

    Returns:
        (int) local_rank
    """
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def _try_load_pretrained_model(cfg: DictConfig) -> Any:
    """Try checkpoint dir, then CausalLM, then MaskedLM (last without extra kwargs)."""
    overrides = getattr(cfg, "model_config_overrides", {})
    revision = getattr(cfg, "pretrained_model_revision", None)
    path = cfg.pretrained_model_name_or_path
    try:
        return load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=path,
            load_ema_weights=cfg.load_ema_weights,
            ckpt_file=cfg.ckpt_file,
            **overrides,
        )
    except Exception:
        loaders = (
            lambda: AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                revision=revision,
                **overrides,
            ),
            lambda: AutoModelForMaskedLM.from_pretrained(
                path,
                trust_remote_code=True,
                revision=revision,
                **overrides,
            ),
            lambda: AutoModelForMaskedLM.from_pretrained(
                path,
                trust_remote_code=True,
                revision=revision,
            ),
        )
        for load in loaders:
            try:
                return load()
            except Exception:
                continue
        return None


def _dit_legacy_backbone_template() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "model",
        "backbone",
        "dit_legacy.yaml",
    )


def _load_backbone_state_into_denoiser(
    denoiser: Any,
    hf_model: Optional[Any],
    pretrained_path: str,
) -> None:
    """Fill denoiser.backbone from an HF wrapper or a raw checkpoint path."""
    if hf_model is not None:
        denoiser.backbone = hf_model.backbone
        return
    state_dict = torch.load(
        pretrained_path,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    for key in list(state_dict.keys()):
        new_key = key
        if "backbone." in new_key:
            new_key = new_key.replace("backbone.", "")
        if "_orig_mod." in new_key:
            new_key = new_key.replace("_orig_mod.", "")
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)
    state_dict.pop("sampling_eps_min", None)
    state_dict.pop("sampling_eps_max", None)
    denoiser.backbone.load_state_dict(state_dict)


def _build_legacy_denoiser(
    cfg: DictConfig,
    tokenizer: Any,
    hf_model: Optional[Any],
) -> Any:
    """Build MDLM, SEDD, AR, or BD3LM and load backbone weights (legacy checkpoints)."""
    backbone_config = OmegaConf.load(_dit_legacy_backbone_template())
    length = getattr(cfg, "length", 1024)
    backbone_config.length = length
    backbone_config.vocab_size = len(tokenizer)
    backbone_config.block_size = getattr(cfg, "block_size", None)
    backbone_config.pretrained_model_name_or_path = getattr(
        cfg, "pretrained_model_name_or_path", None
    )
    backbone_config.num_layers = 12
    backbone_config.n_heads = 12
    backbone_config.hidden_size = 768

    name = backbone_config.pretrained_model_name_or_path or ""
    if "-ar-" in name:
        backbone_config.adaln = False
        backbone_config.causal_attention = True
        backbone_config.attn_backend = "flash_attn"
    elif "mdlm-" in name:
        backbone_config.adaln = True
    else:
        backbone_config.adaln = True

    if not isinstance(backbone_config, DictConfig):
        backbone_config = OmegaConf.create(
            OmegaConf.to_container(backbone_config, resolve=False)
        )

    bc = OmegaConf.to_container(backbone_config, resolve=True)

    def _mdlm_like_denoiser(ctor):
        model_config = MDLMConfig(length=length)
        model_config.backbone_config = bc
        model_config.keep_clean_bos = True
        model_config.mask_token_id = tokenizer.mask_token_id
        model_config.vocab_size = len(tokenizer)
        return ctor(model_config, tokenizer=tokenizer)

    if "mdlm-" in name:
        denoiser = _mdlm_like_denoiser(MDLM)
    elif "sedd-" in name:
        denoiser = _mdlm_like_denoiser(SEDD)
    elif "ar-" in name:
        model_config = ARConfig(length=length, backbone_config=backbone_config)
        model_config.backbone_config = bc
        model_config.keep_clean_bos = True
        model_config.mask_token_id = tokenizer.mask_token_id
        model_config.vocab_size = len(tokenizer)
        denoiser = AR(model_config, tokenizer=tokenizer)
    else:
        model_config = BD3LMConfig(
            length=length,
            backbone_config=backbone_config,
            block_size=cfg.block_size,
        )
        model_config.backbone_config = bc
        model_config.keep_clean_bos = True
        model_config.mask_token_id = tokenizer.mask_token_id
        model_config.vocab_size = len(tokenizer)
        denoiser = BD3LM(model_config, tokenizer=tokenizer)

    _load_backbone_state_into_denoiser(
        denoiser,
        hf_model,
        cfg.pretrained_model_name_or_path,
    )
    denoiser.noise_schedule = LinearNoise()
    return denoiser


def generate_samples(
    cfg: DictConfig, device: str, local_rank: int
) -> tuple[list[str], Optional[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.pretrained_model_name_or_path
    )
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    # Load model: checkpoint / HF, then optionally wrap in a denoiser (legacy layouts).
    model = _try_load_pretrained_model(cfg)
    if model is None or not hasattr(model, "generate"):
        model = _build_legacy_denoiser(cfg, tokenizer, model)

    if getattr(cfg, "compile_backbone", False):
        print("Compiling model backbone")
        model.backbone = torch.compile(
            model.backbone, dynamic=False, mode="max-autotune-no-cudagraphs"
        )

    model = model.to(device)
    if local_rank == 0:
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
    model.eval()
    gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
    if model.tokenizer.bos_token_id is None:
        if model.tokenizer.eos_token_id is None:
            model.tokenizer.bos_token = model.tokenizer.cls_token
            model.tokenizer.eos_token = model.tokenizer.cls_token
        else:
            model.tokenizer.bos_token = model.tokenizer.eos_token

    # set stopping criteria for non-throughput run
    if not getattr(cfg, "throughput_run", False):
        bos_token_pattern = re.escape(model.tokenizer.bos_token)
        gen_kwargs["stopping_criteria"][0].pattern = rf"{bos_token_pattern}"

    # Iterate through the dataset and sample
    generated_samples = []
    tputs = []
    parallelism_factors = []
    lengths = []
    entropies = []
    # divide num_samples by world size, if rank is 0, use the remainder
    num_samples = cfg.num_samples
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        new_max_samples = int(num_samples // world_size)
        if dist.get_rank() == 0:
            new_max_samples += int(num_samples % world_size)
        num_samples = new_max_samples

    if dist.is_available() and dist.is_initialized():
        dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1

    pbar = tqdm(range(num_samples), desc="Generating")

    for ind, i in enumerate(pbar):
        input_ids = torch.tensor([model.tokenizer.bos_token_id])[None, :].to(
            model.device
        )
        # Generate samples
        with torch.no_grad():
            while True:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                generation_output = model.generate(
                    inputs=input_ids,
                    disable_pbar=True,
                    tokenizer=tokenizer,
                    **gen_kwargs,
                )
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_s = start_event.elapsed_time(end_event) / 1000
                if isinstance(generation_output, ModelOutput):
                    outputs = generation_output.sequences
                    parallelism_factor = generation_output.get(
                        "parallelism_factor", -1.0
                    )
                    if parallelism_factor is None:
                        parallelism_factor = -1.0
                else:
                    outputs = generation_output
                    parallelism_factor = -1.0
                length = outputs.numel() - input_ids.numel()
                entropy = _compute_entropy(
                    outputs, model.tokenizer.mask_token_id, model.tokenizer.pad_token_id
                )
                if (
                    gen_kwargs["stopping_criteria"] is not None
                    and hasattr(gen_kwargs["stopping_criteria"][0], "truncate_idx")
                    and gen_kwargs["stopping_criteria"][0].truncate_idx is not None
                ):
                    truncate_idx = gen_kwargs["stopping_criteria"][0].truncate_idx[0]
                    if truncate_idx is not None:
                        outputs = outputs[:, : min(truncate_idx, outputs.shape[1])]
                if _should_reject_generated_sample(cfg, outputs, entropy):
                    continue
                break

            print("final length:", outputs.shape[1])

            if ind % 100 == 0:
                print(tokenizer.decode(outputs[0]))

            if i >= THROUGHPUT_WARMUP:
                tputs.append(length / elapsed_time_s)
                parallelism_factors.append(parallelism_factor)
                lengths.append(outputs.shape[1])
                entropies.extend(entropy)
            output_text = model.tokenizer.decode(outputs[0])
            generated_samples.append(output_text)

            pbar.set_postfix(
                tput=f"{np.mean(tputs):.2f} +/- {np.std(tputs):.2f}",
                parallel=(
                    f"{np.mean(parallelism_factors):.2f} "
                    f"+/- {np.std(parallelism_factors):.2f}"
                ),
            )

    # gather samples across devices
    generated_samples = gather_results(generated_samples, world_size)
    tputs = gather_results(tputs, world_size)
    parallelism_factors = gather_results(parallelism_factors, world_size)
    lengths = gather_results(lengths, world_size)
    entropies = gather_results(entropies, world_size)
    gen_metrics: Optional[dict[str, Any]] = None
    if local_rank == 0:
        print(
            f"TPUT (tok/s) over {len(tputs)} samples: "
            f"{np.mean(tputs)} +/- {np.std(tputs)}"
        )
        print(
            f"Parallelism factor over {len(parallelism_factors)} samples: "
            f"{np.mean(parallelism_factors)} +/- {np.std(parallelism_factors)}"
        )
        print(
            f"Lengths over {len(lengths)} samples: "
            f"{np.mean(lengths)} +/- {np.std(lengths)}"
        )
        print(
            f"Entropies over {len(entropies)} samples: "
            f"{np.mean(entropies)} +/- {np.std(entropies)}"
        )
        gen_metrics = build_generation_metrics_dict(
            tputs, parallelism_factors, lengths, entropies
        )
        if not fsspec_exists(cfg.generated_samples_output_path):
            fsspec_mkdirs(cfg.generated_samples_output_path)
        with open(
            f"{cfg.generated_samples_output_path}/generated_samples.json", "w"
        ) as f:
            json.dump(
                generated_samples,
                f,  # type: ignore
                indent=2,
            )

    return generated_samples, gen_metrics


def _compute_entropy(
    x: torch.LongTensor, mask_token_id: int, pad_token_id: int
) -> torch.Tensor:
    """
    x: (B, L)
    returns: (B,) entropy per sequence (nats)
    """
    B, L = x.shape

    entropies = [0] * B

    for i in range(B):
        xi = x[i]

        # drop mask + padding tokens
        xi = xi[(xi != mask_token_id) & (xi != pad_token_id)]

        if xi.numel() == 0:
            entropies[i] = 0.0
            continue

        _, counts = torch.unique(xi, return_counts=True, sorted=False)
        p = counts.float() / counts.sum()
        entropies[i] = torch.special.entr(p).sum().item()

    return entropies


def _load_text_corpus(path: str) -> list[str]:
    """
    Load a reference corpus for MAUVE.

    Supported formats:
      - .json   : list[str] or list[{"text": ...}]
      - .jsonl  : one string or one {"text": ...} per line
      - .txt    : one sample per line
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, got {type(data)}")
        out = []
        for x in data:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) and "text" in x:
                out.append(x["text"])
            else:
                raise ValueError(
                    f"Unsupported JSON entry type in {path}: {type(x)}. "
                    "Expected str or {'text': ...}."
                )
        return [x for x in out if len(x) > 0]

    if path.endswith(".jsonl"):
        out = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    out.append(obj)
                elif isinstance(obj, dict) and "text" in obj:
                    out.append(obj["text"])
                else:
                    raise ValueError(
                        f"Unsupported JSONL entry in {path}: {type(obj)}. "
                        "Expected str or {'text': ...}."
                    )
        return [x for x in out if len(x) > 0]

    if path.endswith(".txt"):
        with open(path, "r") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    raise ValueError(
        f"Unsupported corpus format for {path}. Use .json, .jsonl, or .txt."
    )


def _device_to_mauve_device_id(device: str) -> int:
    if device == "cpu":
        return -1
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    if device == "cuda":
        return 0
    return -1


def _load_reference_from_dataset(cfg, tokenizer, num_samples: int) -> list[str]:
    """
    Load reference samples from a dataset config (e.g. owt_eval_gpt2).
    Takes the first num_samples and decodes input_ids to text.
    """
    ref_cfg = getattr(cfg, "mauve_reference_dataset", None)
    if ref_cfg is None:
        raise ValueError("mauve_reference_dataset is not set in config")
    # Load more than needed in case some decode to empty; limit_size caps total
    dataset = hydra.utils.instantiate(
        ref_cfg,
        limit_size=num_samples * 2,  # buffer for empty samples
    )
    texts = []
    for i in range(len(dataset)):
        if len(texts) >= num_samples:
            break
        row = dataset[i]
        ids = row["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if text.strip():
            texts.append(text)
    return texts[:num_samples]


def compute_mauve_metrics(
    cfg, samples, device="cuda", tokenizer=None
) -> dict[str, Any] | None:
    """
    Compute MAUVE against a human/reference corpus using ModernBERT-large
    features by default.

    Notes:
      - Runs on a single rank only.
      - Reference: cfg.mauve_reference_dataset (dataset config) or
    """
    ref_dataset_cfg = getattr(cfg, "mauve_reference_dataset", None)
    if (not _use_adlm_compatible_mauve(cfg)) and ref_dataset_cfg is None:
        return None

    if tokenizer is None:
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    if _use_adlm_compatible_mauve(cfg):
        reference_samples = _load_reference_from_adlm_openwebtext_valid(
            cfg, tokenizer, len(samples)
        )
    else:
        num_ref = cfg.get("mauve_reference_num_samples", 5000)
        reference_samples = _load_reference_from_dataset(cfg, tokenizer, num_ref)
    if len(samples) == 0:
        raise ValueError("No generated samples available for MAUVE.")

    n = min(len(reference_samples), len(samples))
    reference_samples = reference_samples[:n]
    generated_samples = samples[:n]

    mauve_kwargs = {
        "p_text": reference_samples,
        "q_text": generated_samples,
        "device_id": _device_to_mauve_device_id(device),
        "seed": int(getattr(cfg, "seed", 0)),
        "verbose": bool(getattr(cfg, "mauve_verbose", False)),
    }
    if _use_adlm_compatible_mauve(cfg):
        mauve_kwargs["max_text_length"] = int(
            getattr(cfg, "adlm_compatible_mauve_max_text_length", 1024)
        )

    out = mauve.compute_mauve(
        **mauve_kwargs,
    )
    return {
        "mauve": float(out.mauve),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    local_rank = setup_ddp()
    set_seed(cfg.seed + local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(cfg.generated_samples_output_path):
        if local_rank == 0:
            os.makedirs(cfg.generated_samples_output_path, exist_ok=True)
    gen_metrics: Optional[dict[str, Any]] = None
    if not getattr(cfg, "eval_only", False):
        samples, gen_metrics = generate_samples(cfg, device, local_rank)
    else:
        # read from file
        with open(
            f"{cfg.generated_samples_output_path}/generated_samples.json", "r"
        ) as f:
            samples = json.load(f)

    # MAUVE is computed once on rank 0 using the full generated corpus.
    mauve_ref_dataset = getattr(cfg, "mauve_reference_dataset", None)
    mauve_stats: Optional[dict[str, Any]] = None
    if (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    ) and mauve_ref_dataset is not None:
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)
        mauve_stats = compute_mauve_metrics(
            cfg, samples, device=device, tokenizer=tokenizer
        )
        if mauve_stats is not None:
            print(f"MAUVE: {mauve_stats['mauve']}")

    if local_rank == 0:
        metrics: dict[str, Any] = {}
        if gen_metrics is not None:
            metrics.update(gen_metrics)
        if mauve_stats is not None:
            metrics.update(mauve_stats)
        if metrics:
            if not fsspec_exists(cfg.generated_samples_output_path):
                fsspec_mkdirs(cfg.generated_samples_output_path)
            with open(f"{cfg.generated_samples_output_path}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
