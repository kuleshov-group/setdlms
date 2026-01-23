import datetime
import json
import os
import sys
import logging

import evaluate
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from streaming import StreamingDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.generation import StopStringCriteria
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

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

THROUGHPUT_SAMPLES = 0
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


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    # local_rank = setup_ddp()
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.cls_token

    eval_dataset = hydra.utils.instantiate(
        cfg.task.eval_dataset, tokenizer=tokenizer, max_length=cfg.max_length
    )

    collator = hydra.utils.instantiate(
        cfg.task.collator,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, sampler=eval_sampler, num_workers=0, pin_memory=True, collate_fn=collator)
    # Load model
    try:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
            load_ema_weights=cfg.load_ema_weights,
            ckpt_file=cfg.ckpt_file,
            verbose=True,
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
    # ckpt_desired="/share/kuleshov/ma2238/textdiffusion/checkpoints/lm1b_wrap_pretrain/checkpoints/last.ckpt"
    # ckpt_desired="/share/kuleshov/ma2238/textdiffusion/checkpoints/ablation_bs16_loglinear_final/last-v1.ckpt"
    # state_dict = torch.load(ckpt_desired, weights_only=False, map_location=device)
    # # rm backbone. prefix
    # state_dict["state_dict"] = {k.replace("backbone.", ""): v for k, v in state_dict["state_dict"].items()}
    # state_dict["state_dict"].pop("sampling_eps_min")
    # state_dict["state_dict"].pop("sampling_eps_max")
    # model.backbone.load_state_dict(state_dict["state_dict"])

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
    pbar = tqdm(eval_dataloader, desc="Generating")
    for i, batch in enumerate(pbar):
        input_ids = torch.tensor([tokenizer.bos_token_id])[None, :].to(model.device).repeat(batch["input_ids"].shape[0], 1)
        # Generate samples
        with torch.no_grad():
            generation_output = model.generate(
                inputs=input_ids,
                disable_pbar=True,
                # tokenizer=tokenizer,  # For debugging: prints intermediate generation
                eval_ground_truth=batch["input_ids"].to(model.device),
                **gen_kwargs,
            )
            nll = -generation_output.likelihoods[:, 1:] # exclude token before eos
            if i == 0:
                all_nlls = nll
            else:
                all_nlls = torch.cat([all_nlls, nll], dim=0)
        pbar.set_postfix(NLL=all_nlls.mean().item(), PPL=torch.exp(all_nlls.mean()).item())

    # report avg nll/ppl across devices
    all_nlls_list = all_nlls.detach().float().cpu().flatten().tolist()
    all_nlls_list = gather_results(all_nlls_list, dist.get_world_size())
    if local_rank == 0:
        avg = float(np.mean(all_nlls_list))
        print(f"Avg NLL: {avg}")
        print(f"Avg PPL: {np.exp(avg)}")
    
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
