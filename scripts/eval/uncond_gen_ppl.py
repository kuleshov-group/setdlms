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
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import ModelOutput
import re
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
MAX_SAMPLES = 200


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
    
    if model.tokenizer.bos_token_id is None:
        if model.tokenizer.eos_token_id is None:
            model.tokenizer.bos_token = model.tokenizer.cls_token
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
    pbar = tqdm(range(MAX_SAMPLES), desc="Generating")
    for i in pbar:
        input_ids = torch.tensor([model.tokenizer.bos_token_id])[None, :].to(model.device)
        # Generate samples
        with torch.no_grad():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            generation_output = model.generate(
                inputs=input_ids,
                disable_pbar=True,
                # tokenizer=tokenizer,  # For debugging: prints intermediate generation
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

            # DEBUG
            print(model.tokenizer.decode(outputs[0]))
            # TODO: CHECK IF OUTPUT CONTAINS PROMPT
            if i >= THROUGHPUT_WARMUP:
                tputs.append((outputs.numel() - input_ids.numel()) / elapsed_time_s)
                parallelism_factors.append(parallelism_factor)
            generated_samples.append(outputs)
        pbar.set_postfix(tput=np.mean(tputs), parallel=np.mean(parallelism_factors))
    # gather samples across devices
    generated_samples = gather_results(generated_samples, dist.get_world_size())
    tputs = gather_results(tputs, dist.get_world_size())
    if local_rank == 0:
        print(f"TPUT (tok/s) over {len(tputs)} samples: {np.mean(tputs)} +/- {np.std(tputs)}")
    with open(f"output/generated_samples.json", "w") as f:
        json.dump(
            generated_samples,
            f,  # type: ignore
            indent=2,
        )
        
    return generated_samples

def compute_metrics(cfg, samples) -> None:
    eval_model_name = getattr(cfg, "eval_model_name", "gpt2-large")
    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_model_name,
        trust_remote_code=True,
        revision=getattr(cfg, "pretrained_model_revision", None),
        **getattr(cfg, "model_config_overrides", {}),
    )
    eval_model = eval_model.to(device)
    eval_model.eval()
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)

    # create an eval dataloader from the samples
    eval_dataset = StreamingDataset(
        data_source=samples,
        transform=lambda x: x,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
    
    all_nlls = []
    # compute nlls under the eval model
    for batch in eval_dataloader:
        with torch.no_grad():
            generation_output = eval_model(inputs=batch,)
            nll = -generation_output.likelihoods[:, 1:] # exclude token before eos
            all_nlls.append(nll)
    all_nlls = torch.cat(all_nlls, dim=0)
    return all_nlls.mean().item()

@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    samples = generate_samples(cfg, device, local_rank)

    if hasattr(cfg, "eval_model_name"):
        gen_nll = compute_metrics(cfg, samples)
        print(f"Avg gen NLL under {cfg.eval_model_name}: {gen_nll}")
        print(f"Avg gen PPL under {cfg.eval_model_name}: {torch.exp(gen_nll)}")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    register_useful_resolvers()
    main()
