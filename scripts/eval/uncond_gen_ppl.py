import datetime
import hashlib
import json
import logging
import math
import os
import random
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
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
)
from transformers.modeling_outputs import ModelOutput

from scripts.eval.model_loading import (
    configure_rank_local_torchinductor_cache,
    load_eval_model,
)
from scripts.utils import (
    count_parameters,
    format_number,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs

log = logging.getLogger(__name__)


THROUGHPUT_WARMUP = 0


def _rank_invariant_generation_enabled() -> bool:
    return str(os.environ.get("LM_EVAL_RANK_INVARIANT_SEED", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _stable_generation_seed(base_seed: int, *parts: object) -> int:
    payload = [str(int(base_seed))]
    payload.extend(str(part) for part in parts)
    digest = hashlib.blake2b(
        "\\0".join(payload).encode("utf-8", errors="surrogatepass"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


def _seed_generation_for_example(seed: int) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def _write_json_atomic(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _generation_checkpoint_path(cfg: DictConfig, local_rank: int) -> str:
    checkpoint_path = getattr(cfg, "generation_checkpoint_path", None)
    if checkpoint_path not in (None, "", "null", "None"):
        return str(checkpoint_path)
    return os.path.join(
        str(cfg.generated_samples_output_path),
        f"generation_checkpoint.rank{local_rank}.pt",
    )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _load_generation_checkpoint(
    cfg: DictConfig, local_rank: int
) -> Optional[dict[str, Any]]:
    if not bool(getattr(cfg, "resume_generation_checkpoint", False)):
        return None
    checkpoint_path = _generation_checkpoint_path(cfg, local_rank)
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _restore_rng_state(checkpoint["rng_state"])
    print(
        f"Resumed generation checkpoint for rank {local_rank}: "
        f"{len(checkpoint.get('generated_samples', []))} samples from {checkpoint_path}"
    )
    return checkpoint


def _save_generation_checkpoint(
    cfg: DictConfig,
    local_rank: int,
    state: dict[str, Any],
) -> None:
    if not bool(getattr(cfg, "generation_checkpoint", False)):
        return
    output_path = str(cfg.generated_samples_output_path)
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = _generation_checkpoint_path(cfg, local_rank)
    tmp_path = f"{checkpoint_path}.tmp"
    torch.save({**state, "rng_state": _capture_rng_state()}, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    partial_path = os.path.join(
        output_path, f"generated_samples.rank{local_rank}.partial.json"
    )
    _write_json_atomic(partial_path, state["generated_samples"])
    if local_rank == 0 and (
        (not dist.is_available())
        or (not dist.is_initialized())
        or dist.get_world_size() == 1
    ):
        _write_json_atomic(
            os.path.join(output_path, "generated_samples.partial.json"),
            state["generated_samples"],
        )

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
    latencies: list,
    parallelism_factors: list,
    lengths: list,
    entropies: list,
) -> dict[str, Any]:
    """Aggregate generation stats into a flat JSON-serializable dict.

    Same shape as seq2seq_eval metrics.
    """
    tp = _summarize_numeric_list(tputs)
    lat = _summarize_numeric_list(latencies)
    pf = _summarize_numeric_list(parallelism_factors)
    ln = _summarize_numeric_list(lengths)
    ent = _summarize_numeric_list(entropies)
    output_lengths = []
    if tputs and latencies:
        output_lengths = [
            float(throughput) * float(latency)
            for throughput, latency in zip(tputs, latencies)
        ]
    out_len = _summarize_numeric_list(output_lengths)
    out: dict[str, Any] = {
        "throughput_tok_per_s_mean": tp["mean"],
        "throughput_tok_per_s_std": tp["std"],
        "throughput_tok_per_s_count": tp["count"],
        "latency_s_mean": lat["mean"],
        "latency_s_std": lat["std"],
        "latency_s_count": lat["count"],
        "output_length_from_tput_latency_mean": out_len["mean"],
        "output_length_from_tput_latency_std": out_len["std"],
        "output_length_from_tput_latency_count": out_len["count"],
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


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "auto"}:
            return None
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


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
    block_size = int(
        getattr(cfg, "max_length", None) or getattr(cfg, "max_new_tokens", 1024) + 1
    )
    if block_size <= 0:
        block_size = 1024

    seed = int(
        getattr(cfg, "adlm_compatible_mauve_valid_seed", getattr(cfg, "seed", 0))
    )
    eval_batch_size = int(getattr(cfg, "batch_size", 1))
    eval_batch_size = int(
        getattr(cfg, "adlm_compatible_mauve_eval_batch_size", eval_batch_size)
    )

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


def generate_samples(
    cfg: DictConfig, device: str, local_rank: int
) -> tuple[list[str], Optional[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.pretrained_model_name_or_path
    )
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    model = load_eval_model(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        pretrained_model_revision=getattr(cfg, "pretrained_model_revision", None),
        load_ema_weights=cfg.load_ema_weights,
        ckpt_file=cfg.ckpt_file,
        model_config_overrides=getattr(cfg, "model_config_overrides", {}),
        force_legacy_if_no_generate=True,
    )

    is_setdlm = model.__class__.__name__ == "SetDLM"
    compile_requested = bool(getattr(cfg, "compile_backbone", False))
    compile_supported = bool(hasattr(model, "backbone"))
    compile_mode = getattr(cfg, "compile_mode", None)
    if compile_mode in ("", "none", "None", "null", "default"):
        compile_mode = None
    if compile_mode is None and not is_setdlm:
        compile_mode = "max-autotune-no-cudagraphs"
    compile_dynamic = _optional_bool(getattr(cfg, "compile_dynamic", None))
    if compile_dynamic is None:
        compile_dynamic = True if is_setdlm else False
    if compile_requested:
        if not compile_supported:
            print(
                "COMPILE_BACKBONE=true requested, but this model has no backbone; "
                "running uncompiled."
            )
        else:
            cache_dir = configure_rank_local_torchinductor_cache()
            if cache_dir:
                print(f"Using rank-local TorchInductor cache: {cache_dir}")
            compile_kwargs = {"dynamic": compile_dynamic}
            if compile_mode is not None:
                compile_kwargs["mode"] = compile_mode
            print(f"Compiling model backbone with torch.compile({compile_kwargs})")
            model.backbone = torch.compile(model.backbone, **compile_kwargs)
    print(
        "Compile settings: "
        f"requested={compile_requested}, supported={compile_supported}, "
        f"mode={compile_mode or 'default'}, dynamic={compile_dynamic}"
    )

    model = model.to(device)
    if local_rank == 0:
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
    model.eval()
    gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
    instantiated_generation_config = gen_kwargs.get("generation_config")
    cfg_max_window_size = OmegaConf.select(cfg, "generation_config.max_window_size")
    if cfg_max_window_size is not None:
        gen_kwargs["max_window_size"] = cfg_max_window_size
        if instantiated_generation_config is not None:
            instantiated_generation_config.max_window_size = cfg_max_window_size
    if model.tokenizer.bos_token_id is None:
        if model.tokenizer.eos_token_id is None:
            model.tokenizer.bos_token = model.tokenizer.cls_token
            model.tokenizer.eos_token = model.tokenizer.cls_token
        else:
            model.tokenizer.bos_token = model.tokenizer.eos_token

    # Throughput runs should not stop early on content-based criteria.
    if getattr(cfg, "throughput_run", False):
        gen_kwargs["stopping_criteria"] = None
    else:
        bos_token_pattern = re.escape(model.tokenizer.bos_token)
        gen_kwargs["stopping_criteria"][0].pattern = rf"{bos_token_pattern}"

    # Iterate through the dataset and sample
    generated_samples = []
    tputs = []
    latencies = []
    parallelism_factors = []
    lengths = []
    entropies = []
    measured_parallelism_factors = []
    measured_lengths = []
    measured_entropies = []
    throughput_run = bool(getattr(cfg, "throughput_run", False))
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    throughput_warmup = int(getattr(cfg, "throughput_warmup", THROUGHPUT_WARMUP))
    throughput_num_measurements = int(
        getattr(
            cfg,
            "throughput_num_measurements",
            getattr(cfg, "throughput_samples_per_rank", 200),
        )
    )
    throughput_global_measurements = bool(
        getattr(cfg, "throughput_global_measurements", False)
    )
    local_measurement_target = int(getattr(cfg, "throughput_samples_per_rank", 200))
    if throughput_global_measurements and world_size > 1:
        local_measurement_target = math.ceil(throughput_num_measurements / world_size)
    # For normal runs, shard the requested sample count across ranks.
    # For throughput runs, warm up per rank, then gather measured examples
    # across ranks and trim to the requested global sample count.
    if throughput_run:
        num_samples = throughput_warmup + local_measurement_target
    else:
        num_samples = cfg.num_samples
        if dist.is_available() and dist.is_initialized():
            new_max_samples = int(num_samples // world_size)
            if dist.get_rank() == 0:
                new_max_samples += int(num_samples % world_size)
            num_samples = new_max_samples

    checkpoint = _load_generation_checkpoint(cfg, local_rank)
    if checkpoint is not None:
        generated_samples = list(checkpoint.get("generated_samples", []))
        tputs = list(checkpoint.get("tputs", []))
        latencies = list(checkpoint.get("latencies", []))
        parallelism_factors = list(checkpoint.get("parallelism_factors", []))
        lengths = list(checkpoint.get("lengths", []))
        entropies = list(checkpoint.get("entropies", []))
        measured_parallelism_factors = list(
            checkpoint.get("measured_parallelism_factors", [])
        )
        measured_lengths = list(checkpoint.get("measured_lengths", []))
        measured_entropies = list(checkpoint.get("measured_entropies", []))
    start_index = len(generated_samples)
    if start_index > num_samples:
        raise ValueError(
            f"Checkpoint has {start_index} samples, but this run only requests "
            f"{num_samples} samples on rank {local_rank}."
        )

    pbar = tqdm(
        range(start_index, num_samples),
        desc="Generating",
        disable=(local_rank != 0),
        initial=start_index,
        total=num_samples,
    )
    checkpoint_interval = max(
        1, int(getattr(cfg, "generation_checkpoint_interval", 1))
    )

    for i in pbar:
        input_ids = torch.tensor([model.tokenizer.bos_token_id])[None, :].to(
            model.device
        )
        # Generate samples
        with torch.no_grad():
            while True:
                if _rank_invariant_generation_enabled():
                    example_seed = _stable_generation_seed(
                        int(os.environ.get("LM_EVAL_BASE_SEED", cfg.seed)),
                        i,
                    )
                    _seed_generation_for_example(example_seed)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                if gen_kwargs["stopping_criteria"] is not None:
                    reset_stopping_criteria = getattr(
                        gen_kwargs["stopping_criteria"], "reset", None
                    )
                    if callable(reset_stopping_criteria):
                        reset_stopping_criteria()
                generation_output = model.generate(
                    inputs=input_ids,
                    disable_pbar=False,
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
                    (not throughput_run)
                    and gen_kwargs["stopping_criteria"] is not None
                    and hasattr(gen_kwargs["stopping_criteria"][0], "truncate_idx")
                    and gen_kwargs["stopping_criteria"][0].truncate_idx is not None
                ):
                    truncate_idx = gen_kwargs["stopping_criteria"][0].truncate_idx[0]
                    if truncate_idx is not None:
                        outputs = outputs[:, : min(truncate_idx, outputs.shape[1])]
                if (not throughput_run) and _should_reject_generated_sample(
                    cfg, outputs, entropy
                ):
                    continue
                break

            print("final length:", outputs.shape[1])

            if i % 100 == 0:
                print(tokenizer.decode(outputs[0]))

            if i >= throughput_warmup:
                tputs.append(length / elapsed_time_s)
                latencies.append(elapsed_time_s)
                measured_parallelism_factors.append(parallelism_factor)
                measured_lengths.append(outputs.shape[1])
                measured_entropies.extend(entropy)
            parallelism_factors.append(parallelism_factor)
            lengths.append(outputs.shape[1])
            entropies.extend(entropy)
            output_text = model.tokenizer.decode(outputs[0])
            generated_samples.append(output_text)

            if (
                bool(getattr(cfg, "generation_checkpoint", False))
                and (
                    len(generated_samples) % checkpoint_interval == 0
                    or len(generated_samples) == num_samples
                )
            ):
                _save_generation_checkpoint(
                    cfg,
                    local_rank,
                    {
                        "rank": int(local_rank),
                        "target_num_samples": int(num_samples),
                        "generated_samples": generated_samples,
                        "tputs": tputs,
                        "latencies": latencies,
                        "parallelism_factors": parallelism_factors,
                        "lengths": lengths,
                        "entropies": entropies,
                        "measured_parallelism_factors": measured_parallelism_factors,
                        "measured_lengths": measured_lengths,
                        "measured_entropies": measured_entropies,
                    },
                )

            if local_rank == 0:
                postfix = {
                    "parallel": (
                        f"{np.mean(parallelism_factors):.2f} "
                        f"+/- {np.std(parallelism_factors):.2f}"
                    )
                }
                if tputs:
                    postfix["tput"] = f"{np.mean(tputs):.2f} +/- {np.std(tputs):.2f}"
                pbar.set_postfix(postfix)

    # gather samples across devices
    generated_samples = gather_results(generated_samples, world_size)
    tputs = gather_results(tputs, world_size)
    latencies = gather_results(latencies, world_size)
    parallelism_factors = gather_results(parallelism_factors, world_size)
    lengths = gather_results(lengths, world_size)
    entropies = gather_results(entropies, world_size)
    measured_parallelism_factors = gather_results(
        measured_parallelism_factors, world_size
    )
    measured_lengths = gather_results(measured_lengths, world_size)
    measured_entropies = gather_results(measured_entropies, world_size)
    gen_metrics: Optional[dict[str, Any]] = None
    if local_rank == 0:
        if throughput_run:
            tputs = tputs[:throughput_num_measurements]
            latencies = latencies[:throughput_num_measurements]
            measured_parallelism_factors = measured_parallelism_factors[
                :throughput_num_measurements
            ]
            measured_lengths = measured_lengths[:throughput_num_measurements]
            measured_entropies = measured_entropies[:throughput_num_measurements]
        if tputs:
            tput_prefix = "TPUT"
            print(
                f"{tput_prefix} (tok/s) over {len(tputs)} samples: "
                f"{np.mean(tputs)} +/- {np.std(tputs)}"
            )
        if latencies:
            print(
                f"Latency (s) over {len(latencies)} samples: "
                f"{np.mean(latencies)} +/- {np.std(latencies)}"
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
        metrics_parallelism = (
            measured_parallelism_factors if throughput_run else parallelism_factors
        )
        metrics_lengths = measured_lengths if throughput_run else lengths
        metrics_entropies = measured_entropies if throughput_run else entropies
        gen_metrics = build_generation_metrics_dict(
            tputs, latencies, metrics_parallelism, metrics_lengths, metrics_entropies
        )
        if not fsspec_exists(cfg.generated_samples_output_path):
            fsspec_mkdirs(cfg.generated_samples_output_path)
        if throughput_run:
            output_lengths_from_timing = [
                float(throughput) * float(latency)
                for throughput, latency in zip(tputs, latencies)
            ]
            generation_config = getattr(cfg, "generation_config", {})
            nfe = getattr(generation_config, "num_steps", None)
            throughput_summary = {
                "throughput_mean": float(np.mean(tputs)) if tputs else None,
                "throughput_std": float(np.std(tputs)) if tputs else None,
                "throughput_all": [float(x) for x in tputs],
                "latency_mean": float(np.mean(latencies)) if latencies else None,
                "latency_std": float(np.std(latencies)) if latencies else None,
                "latency_all": [float(x) for x in latencies],
                "output_length_from_tput_latency_mean": (
                    float(np.mean(output_lengths_from_timing))
                    if output_lengths_from_timing
                    else None
                ),
                "output_length_from_tput_latency_std": (
                    float(np.std(output_lengths_from_timing))
                    if output_lengths_from_timing
                    else None
                ),
                "parallelism_factor_mean": (
                    float(np.mean(measured_parallelism_factors))
                    if measured_parallelism_factors
                    else None
                ),
                "parallelism_factor_std": (
                    float(np.std(measured_parallelism_factors))
                    if measured_parallelism_factors
                    else None
                ),
                "parallelism_factor_all": [
                    float(x) for x in measured_parallelism_factors
                ],
                "sequence_length_tokens_mean": (
                    float(np.mean(measured_lengths)) if measured_lengths else None
                ),
                "sequence_length_tokens_std": (
                    float(np.std(measured_lengths)) if measured_lengths else None
                ),
                "warmup_examples_per_rank": throughput_warmup,
                "measured_examples": len(tputs),
                "requested_measured_examples": throughput_num_measurements,
                "local_measurement_target": local_measurement_target,
                "throughput_global_measurements": throughput_global_measurements,
                "world_size": world_size,
                "nfe": int(nfe) if nfe is not None else None,
                "compile_backbone": compile_requested,
                "compile_supported": compile_supported,
                "compile_mode": compile_mode or "default",
                "compile_dynamic": compile_dynamic,
                "confidence_based_noising": bool(
                    getattr(generation_config, "confidence_based_noising", False)
                ),
                "confidence_threshold": float(
                    getattr(generation_config, "confidence_threshold", 1e6)
                ),
                "setdlm_fast_inference": bool(
                    getattr(generation_config, "setdlm_fast_inference", False)
                ),
                "setdlm_dynamic_active_logits": bool(
                    getattr(generation_config, "setdlm_dynamic_active_logits", False)
                ),
                "setdlm_deterministic_sampler_fastpath": bool(
                    getattr(
                        generation_config,
                        "setdlm_deterministic_sampler_fastpath",
                        False,
                    )
                ),
                "setdlm_vectorized_repetition_penalty": bool(
                    getattr(
                        generation_config,
                        "setdlm_vectorized_repetition_penalty",
                        False,
                    )
                ),
                "setdlm_dynamic_tensor_attention_mask": bool(
                    getattr(
                        generation_config,
                        "setdlm_dynamic_tensor_attention_mask",
                        False,
                    )
                ),
                "setdlm_dynamic_full_window_fastpath": bool(
                    getattr(
                        generation_config,
                        "setdlm_dynamic_full_window_fastpath",
                        False,
                    )
                ),
            }
            with open(
                f"{cfg.generated_samples_output_path}/throughput-all.json", "w"
            ) as f:
                json.dump(throughput_summary, f, indent=2)
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
    os.environ.setdefault("LM_EVAL_BASE_SEED", str(int(cfg.seed)))
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
        not getattr(cfg, "throughput_run", False)
        and not getattr(cfg, "skip_mauve", False)
        and (
            (not dist.is_available())
            or (not dist.is_initialized())
            or dist.get_rank() == 0
        )
        and mauve_ref_dataset is not None
    ):
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
