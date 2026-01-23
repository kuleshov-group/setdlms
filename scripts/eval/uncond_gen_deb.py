import datetime
import os
import logging
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

log = logging.getLogger(__name__)


from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs


def generate_with_diffusion(model, eval_ground_truth, keep_clean_bos=False):
    device = eval_ground_truth.device
    B, L = eval_ground_truth.shape

    mask_id = getattr(model, "mask_token_id", None)
    lls = torch.zeros((B, L), device=device, dtype=torch.float32)

    xt = model.tokenizer.mask_token_id * torch.ones((B, L), device=device, dtype=torch.long)
    if keep_clean_bos:
        xt[:, 0] = eval_ground_truth[:, 0] # bos

    block_id = 0
    block_size = 4

    while True:
        masked = (xt == mask_id)
        if not masked.any():
            break
        # import ipdb ; ipdb.set_trace()
        denoiser_inputs, _ = model._prepare_inputs_inference(input_ids=xt)

        backbone_output = model._backbone_forward(denoiser_inputs,)
        backbone_output = {k: v for k, v in backbone_output.items()}
        logits = backbone_output.pop("logits")

        logits[..., mask_id] = model.neg_infinity
        logits = logits - torch.logsumexp(
            logits, dim=-1, keepdim=True
        )

        true_logp = logits.gather(-1, eval_ground_truth[..., None]).squeeze(-1)

        # eligible positions: currently masked and within the current block
        eligible = (masked.clone() & (torch.arange(L).to(device) < (block_id + 1) * block_size))

        score = torch.where(
            eligible,
            true_logp,
            torch.full_like(true_logp, -float("inf")),
        )
        idx = score.argmax(dim=-1)  # (B,)

        # record chosen token
        chosen = true_logp.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)
        lls.scatter_(1, idx.unsqueeze(1), chosen.unsqueeze(1))

        # reveal ground truth token
        tok = eval_ground_truth.gather(1, idx.unsqueeze(1))  # (B,1)
        xt.scatter_(1, idx.unsqueeze(1), tok)

        # advance block
        prefix = xt[:, :(block_id + 1) * block_size]
        if not (prefix == mask_id).any():
            block_id += 1
    if keep_clean_bos:
        lls = lls[:, 1:]
    return xt, -lls

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
    model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
        load_ema_weights=cfg.load_ema_weights,
        ckpt_file=cfg.ckpt_file,
        **getattr(cfg, "model_config_overrides", {}),
    )
    
    ckpt_desired = "/share/kuleshov/ma2238/textdiffusion/checkpoints/lm1b_wrap_pretrain/checkpoints/last.ckpt"
    state_dict = torch.load(ckpt_desired, weights_only=False, map_location=device)
    # rm backbone. prefix
    state_dict["state_dict"] = {k.replace("backbone.", ""): v for k, v in state_dict["state_dict"].items()}
    model.backbone.load_state_dict(state_dict["state_dict"])
    
    model = model.to(device)
    model.eval()
    
    pbar = tqdm(eval_dataloader, desc="Generating")
    for i, batch in enumerate(pbar):
        with torch.no_grad():
            _, nlls = generate_with_diffusion(
                model=model,
                eval_ground_truth=batch["input_ids"].to(model.device),
            )
            if i == 0:
                all_nlls = nlls
            else:
                all_nlls = torch.cat([all_nlls, nlls], dim=0)
        pbar.set_postfix(NLL=all_nlls.mean().item(), PPL=torch.exp(all_nlls.mean()).item())

    # report avg nll/ppl across devices
    all_nlls_list = all_nlls.detach().float().cpu().flatten().tolist()
    all_nlls_list = gather_results(all_nlls_list, dist.get_world_size())
    if local_rank == 0:
        avg = float(np.mean(all_nlls_list))
        print(f"Avg NLL: {avg}")
        print(f"Avg PPL: {np.exp(avg)}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
