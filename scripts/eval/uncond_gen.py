import datetime
import json
import os
import sys

import evaluate
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.generation import StopStringCriteria

from scripts.utils import (
    count_parameters,
    format_number,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs

THROUGHPUT_SAMPLES = 0
THROUGHPUT_WARMUP = 0
MAX_SAMPLES = 15


def gather_results(results, world_size):
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


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    # local_rank = setup_ddp()
    local_rank = 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load model
    try:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
            load_ema_weights=cfg.load_ema_weights,
            ckpt_file=cfg.ckpt_file,
            **getattr(cfg, "model_config_overrides", {}),
        )
    except FileNotFoundError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path,
                trust_remote_code=True,
                revision=getattr(cfg, "pretrained_model_revision", None),
                **getattr(cfg, "model_config_overrides", {}),
            )
        except ValueError:  # Model not compatible with CausalLM
            model = AutoModelForMaskedLM.from_pretrained(
                cfg.pretrained_model_name_or_path,
                trust_remote_code=True,
                revision=getattr(cfg, "pretrained_model_revision", None),
                **getattr(cfg, "model_config_overrides", {}),
            )
    model = model.to(device)
    if local_rank == 0:
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
    model.eval()
    gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
    stop_tokens = None
    if getattr(gen_kwargs, "stopping_criteria", None) is not None:
        for sc in gen_kwargs["stopping_criteria"]:
            if isinstance(sc, StopStringCriteria):
                stop_tokens = list(sc.stop_strings)
                break

    # Iterate through the dataset and sample
    generated_samples = []
    all_intm_samples = []
    for i in range(25):
        
        input_ids = torch.tensor([tokenizer.bos_token_id])[None, :].to(model.device)
        # Generate samples
        with torch.no_grad():
            sample, intm_samples, caching_wall_clocks, decoding_wall_clocks = model.generate(
                inputs=input_ids,
                disable_pbar=(local_rank != 0),
                tokenizer=tokenizer,  # For debugging: prints intermediate generation
                save_intermediate_samples=True,
                **gen_kwargs,
            )
            print(tokenizer.decode(sample[0]))
            caching_wall_clock = np.mean(caching_wall_clocks)
            decoding_wall_clock = np.mean(decoding_wall_clocks)
            print('total wall', sum(decoding_wall_clocks) + sum(caching_wall_clocks))

        all_intm_samples.append({
            "intermediate_samples": intm_samples,
            "caching_wall_clock": caching_wall_clock,
            "decoding_wall_clock": decoding_wall_clock,
        })
        if i > MAX_SAMPLES:
            break
    with open(f"output/intermediate_samples.json", "w") as f:
        json.dump(
            all_intm_samples,
            f,  # type: ignore
            indent=2,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
