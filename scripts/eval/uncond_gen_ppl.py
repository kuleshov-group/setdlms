import datetime
import json
import os
import sys
import logging
from typing import Any

import evaluate
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from streaming import StreamingDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.generation import StopStringCriteria
from src.noise_schedule.noise_schedules import LinearNoise
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import ModelOutput
import re
from omegaconf import DictConfig, OmegaConf
from scripts.utils import maybe_add_missing_special_tokens
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from src.denoiser.ar import AR, ARConfig
from src.denoiser.diffusion import BD3LM, BD3LMConfig
from src.denoiser.diffusion import SetDLM, SEDD
from src.denoiser.diffusion import MDLM, MDLMConfig
log = logging.getLogger(__name__)


from scripts.utils import (
    count_parameters,
    format_number,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs

THROUGHPUT_WARMUP = 0
MAX_SAMPLES = 100


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


def generate_samples(cfg: DictConfig, device: str, local_rank: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.pretrained_model_name_or_path)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    # Load model
    try:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
            load_ema_weights=cfg.load_ema_weights,
            ckpt_file=cfg.ckpt_file,
            **getattr(cfg, "model_config_overrides", {}),
        )
    except:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path,
                trust_remote_code=True,
                revision=getattr(cfg, "pretrained_model_revision", None),
                **getattr(cfg, "model_config_overrides", {}),
            )
        except:  # Model not compatible with CausalLM
            try:
                model = AutoModelForMaskedLM.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=getattr(cfg, "pretrained_model_revision", None),
                    **getattr(cfg, "model_config_overrides", {}),
                )
            except:
                try:
                    model = AutoModelForMaskedLM.from_pretrained(
                        cfg.pretrained_model_name_or_path,
                        trust_remote_code=True,
                        revision=getattr(cfg, "pretrained_model_revision", None),
                    )
                except:
                    model = None

    # HACK for legacy codebase compatibility
    if model is None or not hasattr(model, "generate"):
        
        # Create dit backbone config
        # Load the dit config template and update with actual values
        dit_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "model", "backbone", "dit_legacy.yaml"
        )
        backbone_config = OmegaConf.load(dit_config_path)
        
        # Update backbone config with necessary parameters (resolving the template values)
        length = getattr(cfg, "length", 1024)
        backbone_config.length = length
        backbone_config.vocab_size = len(tokenizer)
        backbone_config.block_size = getattr(cfg, "block_size", None)
        backbone_config.pretrained_model_name_or_path = getattr(cfg, "pretrained_model_name_or_path", None)
        backbone_config.num_layers = 12
        backbone_config.n_heads = 12
        backbone_config.hidden_size = 768

        if "-ar-" in backbone_config.pretrained_model_name_or_path:
            backbone_config.adaln = False
            backbone_config.causal_attention = True
            backbone_config.attn_backend = "flash_attn"
        elif "mdlm-" in backbone_config.pretrained_model_name_or_path:
            # backbone_config.attn_backend = "flash_attn"
            backbone_config.adaln = True
        else:
            backbone_config.adaln = True
    
        # Ensure it's a DictConfig
        if not isinstance(backbone_config, DictConfig):
            backbone_config = OmegaConf.create(OmegaConf.to_container(backbone_config, resolve=False))
        if "mdlm-" in backbone_config.pretrained_model_name_or_path:
            model_config = MDLMConfig(
                length=length,
            )
            model_config.backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = MDLM(
                model_config,
                tokenizer=tokenizer,
            )
        elif "sedd-" in backbone_config.pretrained_model_name_or_path:
            model_config = MDLMConfig(
                length=length,
            )
            model_config.backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = SEDD(
                model_config,
                tokenizer=tokenizer,
            )
        elif "ar-" in backbone_config.pretrained_model_name_or_path:
            model_config = ARConfig(
                length=length,
                backbone_config=backbone_config,
            )
            model_config.backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = AR(
                model_config,
                tokenizer=tokenizer,
            )
        else:
            model_config = BD3LMConfig(
                length=length,
                backbone_config=backbone_config,
                block_size=cfg.block_size,
            )
            model_config.backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = BD3LM(
                model_config,
                tokenizer=tokenizer,
            )
        if model is not None:
            denoiser.backbone = model.backbone
        else:
            state_dict = torch.load(
                cfg.pretrained_model_name_or_path,
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

        model = denoiser.to(device)
        model.noise_schedule = LinearNoise()

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
    # divide MAX_SAMPLES by world size, if rank is 0, use the remainder
    MAX_SAMPLES = 5000
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        new_max_samples = int(MAX_SAMPLES // world_size)
        if dist.get_rank() == 0:
            new_max_samples += int(MAX_SAMPLES % world_size)
        MAX_SAMPLES = new_max_samples
    pbar = tqdm(range(MAX_SAMPLES), desc="Generating")
    for ind, i in enumerate(pbar):
        input_ids = torch.tensor([model.tokenizer.bos_token_id])[None, :].to(model.device)
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
                    parallelism_factor = generation_output.get("parallelism_factor", -1.0)
                    if parallelism_factor is None:
                        parallelism_factor = -1.0
                else:
                    outputs = generation_output
                    parallelism_factor = -1.0
                length = outputs.numel() - input_ids.numel()
                entropy = _compute_entropy(outputs, model.tokenizer.mask_token_id, model.tokenizer.pad_token_id)
                if gen_kwargs["stopping_criteria"] is not None and hasattr(gen_kwargs["stopping_criteria"][0], "truncate_idx") and gen_kwargs["stopping_criteria"][0].truncate_idx is not None:
                    truncate_idx = gen_kwargs["stopping_criteria"][0].truncate_idx[0]
                    if truncate_idx is not None:
                        outputs = outputs[:, :min(truncate_idx, outputs.shape[1])]
                if outputs.shape[1] <= 4: # too short samples
                    continue
                if entropy < 3: # degenereate samples
                    continue
                break
            print(f"Length: {length}")
            print("final length:", outputs.shape[1])

            if ind % 1 == 0:
                print(tokenizer.decode(outputs[0]))

            if i >= THROUGHPUT_WARMUP:
                tputs.append(length / elapsed_time_s)
                parallelism_factors.append(parallelism_factor)
                lengths.append(outputs.shape[1])
            # postprocess
            output_text = model.tokenizer.decode(outputs[0])
            # print(output_text)
            # remove all text after the second <|endoftext|>
            # output_text = model.tokenizer.bos_token + "".join(output_text.split(model.tokenizer.bos_token)[1:2])
            # if length > 1024:
            generated_samples.append(output_text)
        pbar.set_postfix(tput=f"{np.mean(tputs):.2f} +/- {np.std(tputs):.2f}", parallel=f"{np.mean(parallelism_factors):.2f} +/- {np.std(parallelism_factors):.2f}")
    # gather samples across devices
    generated_samples = gather_results(generated_samples, dist.get_world_size())
    tputs = gather_results(tputs, dist.get_world_size())
    parallelism_factors = gather_results(parallelism_factors, dist.get_world_size())
    lengths = gather_results(lengths, dist.get_world_size())
    if local_rank == 0:
        print(f"TPUT (tok/s) over {len(tputs)} samples: {np.mean(tputs)} +/- {np.std(tputs)}")
        print(f"Parallelism factor over {len(parallelism_factors)} samples: {np.mean(parallelism_factors)} +/- {np.std(parallelism_factors)}")
        print(f"Lengths over {len(lengths)} samples: {np.mean(lengths)} +/- {np.std(lengths)}")
        with open(f"{cfg.generated_samples_output_path}/generated_samples.json", "w") as f:
            json.dump(
                generated_samples,
                f,  # type: ignore
                indent=2,
            )
        
    return generated_samples

def _compute_entropy(x: torch.LongTensor, mask_token_id: int, pad_token_id: int) -> torch.Tensor:
    """
    x: (B, L)
    returns: (B,) entropy per sequence (nats)
    """
    B, L = x.shape
    device = x.device

    entropies = torch.zeros(B, device=device, dtype=torch.float32)

    for i in range(B):
        xi = x[i]

        # drop mask + padding tokens
        xi = xi[(xi != mask_token_id) & (xi != pad_token_id)]

        if xi.numel() == 0:
            entropies[i] = 0.0
            continue

        _, counts = torch.unique(xi, return_counts=True, sorted=False)
        p = counts.float() / counts.sum()
        entropies[i] = torch.special.entr(p).sum()

    return entropies


@torch.no_grad()
def _accumulate_nll_sliding_window(
    eval_model: torch.nn.Module,
    input_ids: torch.LongTensor,        # (B, L) padded
    attention_mask: torch.LongTensor,   # (B, L) 1 for valid, 0 for pad
    *,
    context_size: int,
    stride: int,
    eos_token_id: int | None,
    device: str,
) -> torch.Tensor:
    """
    Computes token-level NLL over full sequences using a sliding window
    (like your reference code), ignoring padding and optionally ignoring EOS.

    Returns:
      nlls (B, L) tensor of nlls for each token
    """
    B, L = input_ids.shape

    # Accumulate per-token nll into a buffer to avoid double counting across windows
    # (same pattern as your example).
    nll_accum = torch.zeros((B, L), device=device, dtype=torch.float32)
    valid_accum = torch.zeros((B, L), device=device, dtype=torch.float32)

    # How many windows? Ensure at least 1.
    num_strides = max(1, math.ceil((L - context_size + stride) / stride))

    for w in range(num_strides):
        if w == 0:
            start = 0
            end = min(context_size, L)
        else:
            start = w * stride
            end = min(start + context_size, L)

        if start >= L:
            break

        chunk_ids = input_ids[:, start:end]
        chunk_attn = attention_mask[:, start:end]

        # Forward
        logits = eval_model(input_ids=chunk_ids, attention_mask=chunk_attn).logits  # (B, T, V)

        # Token NLL for positions 1..T-1 within this chunk
        # (predict token t given tokens <t)
        logits = logits[:, :-1, :]                  # (B, T-1, V)
        labels = chunk_ids[:, 1:]                   # (B, T-1)
        mask = chunk_attn[:, 1:].to(torch.float32)  # (B, T-1)

        # Optionally exclude EOS tokens from likelihood (matches your ref snippet).
        if eos_token_id is not None:
            mask = mask * (labels != eos_token_id).to(torch.float32)

        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        nll = (-token_logp) * mask  # (B, T-1)

        if w == 0:
            # First window: write the entire context range (positions start+1 .. end-1)
            # to global positions (start+1 .. end-1) == (1 .. end-1)
            nll_accum[:, start + 1:end] += nll
            valid_accum[:, start + 1:end] += mask
        else:
            # Subsequent windows: only write the *last stride* worth of tokens to avoid overlaps
            update_start = max(start + 1, end - stride)
            update_window = end - update_start
            if update_window <= 0:
                continue
            # The corresponding slice in nll is the last `update_window` tokens of the chunk labels,
            # which align to global positions [update_start .. end)
            nll_accum[:, update_start:end] += nll[:, -update_window:]
            valid_accum[:, update_start:end] += mask[:, -update_window:]
    # Convert to per-token values (typically valid_accum is 0/1; division is defensive)
    valid = (valid_accum > 0).to(torch.float32)
    return nll_accum[torch.where(valid > 0)]



def compute_metrics(cfg, samples, device="cuda") -> float:
    eval_model_name = getattr(cfg, "eval_model_name", "gpt2-large")

    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_model_name,
        trust_remote_code=True,
        revision=getattr(cfg, "pretrained_model_revision", None),
    ).to(device)

    eval_model.eval()

    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    # Sliding-window likelihood settings (defaults mirror your snippet)
    stride = int(getattr(cfg, "eval_stride", 512))
    # context_size: prefer cfg override, else model max positions if available, else fall back to tokenized length
    context_size = 1024

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # shard deterministically: each rank evaluates a disjoint subset
    local_samples = samples[rank::world_size]

    # tokenize properly (rank-local)
    encodings = eval_tokenizer(
        local_samples,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    if context_size <= 0:
        context_size = int(input_ids.shape[1])
        context_size = min(context_size, int(input_ids.shape[1]))

    eval_dataloader = DataLoader(
        list(zip(input_ids, attention_mask)),
        batch_size=min(16, len(input_ids)),
        shuffle=False,
    )

    # We aggregate globally without gathering all per-token values:
    #   nll: token-level (ignoring padding and the first token)
    #   entropy: sequence-level (one per sample)
    nlls = torch.tensor([], device=device, dtype=torch.float64)
    entropies = torch.tensor([], device=device, dtype=torch.float64)


    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids, attention_mask = [x.to(device) for x in batch]

            # Sliding-window token likelihood (handles long sequences)
            nlls_batch = _accumulate_nll_sliding_window(
                eval_model,
                input_ids,
                attention_mask,
                context_size=context_size,
                stride=stride,
                eos_token_id=eval_tokenizer.eos_token_id,
                device=device,
            )
            nlls = torch.cat([nlls, nlls_batch], dim=0)

            # sequence-level entropy (one per sample in batch)
            entropy = _compute_entropy(
                input_ids,
                eval_tokenizer.mask_token_id,
                eval_tokenizer.pad_token_id,
            ).to(torch.float64)  # (B,)
            entropies = torch.cat([entropies, entropy], dim=0)
            pbar.set_postfix(nll_mean=nlls.mean().item(), ent_mean=entropies.mean().item())

    # gather nlls and entropies across devices
    nlls = gather_results(nlls.detach().cpu().numpy(), dist.get_world_size())
    entropies = gather_results(entropies.detach().cpu().numpy(), dist.get_world_size())
 
    nll_mean = np.mean(nlls)
    nll_std = np.std(nlls)
    ent_mean = np.mean(entropies)
    ent_std = np.std(entropies)

    return {
        "nll_mean": nll_mean,
        "nll_std": nll_std,
        "ppl": np.exp(nll_mean),
        "entropy_mean": ent_mean,
        "entropy_std": ent_std,
        "num_samples_local": len(local_samples),
        "num_samples_global": len(nlls),
        "num_tokens_global": len(nlls),
        "eval_context_size": int(context_size),
        "eval_stride": int(stride),
    }

@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    local_rank = setup_ddp()
    set_seed(cfg.seed + local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(cfg.generated_samples_output_path):
        if local_rank == 0:
            os.makedirs(cfg.generated_samples_output_path, exist_ok=True)
    if not getattr(cfg, "eval_only", False):
        samples = generate_samples(cfg, device, local_rank)
    else:
        # read from file
        with open(f"{cfg.generated_samples_output_path}/generated_samples.json", "r") as f:
            samples = json.load(f)
    
    if hasattr(cfg, "eval_model_name"):
        stats = compute_metrics(cfg, samples, device=device)
        # only rank0 prints aggregated metrics
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[DDP eval] teacher={cfg.eval_model_name} "
                  f"samples={stats['num_samples_global']} tokens={stats['num_tokens_global']}")
            print(f"Avg gen NLL under {cfg.eval_model_name}: {stats['nll_mean']} +/- {stats['nll_std']}")
            print(f"Avg gen PPL under {cfg.eval_model_name}: {stats['ppl']}")
            print(f"Avg gen entropy under {cfg.eval_model_name}: {stats['entropy_mean']} +/- {stats['entropy_std']}")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    register_useful_resolvers()
    main()
