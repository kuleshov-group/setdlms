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
from omegaconf import DictConfig, OmegaConf
from scripts.utils import maybe_add_missing_special_tokens
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
MAX_SAMPLES = 2


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
    maybe_add_missing_special_tokens(tokenizer)
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

    # HACK FOR MDLM/BD3LM HF MODELS
    if model is None or not hasattr(model, "generate"):
        # from src.denoiser.diffusion import MDLM, MDLMConfig
        from src.denoiser.diffusion import BD3LM, BD3LMConfig
        from src.noise_schedule.noise_schedules import LinearNoise
        
        # Create dit backbone config
        # Load the dit config template and update with actual values
        dit_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "model", "backbone", "dit.yaml"
        )
        backbone_config = OmegaConf.load(dit_config_path)
        
        # Update backbone config with necessary parameters (resolving the template values)
        backbone_config.length = cfg.max_length
        backbone_config.vocab_size = len(tokenizer)
        backbone_config.block_size = getattr(cfg, "block_size", None)
        backbone_config.pretrained_model_name_or_path = getattr(cfg, "pretrained_model_name_or_path", None)
        backbone_config.num_layers = 12
        backbone_config.n_heads = 12
        backbone_config.hidden_size = 768
        backbone_config.adaln = True
        
        # Ensure it's a DictConfig
        if not isinstance(backbone_config, DictConfig):
            backbone_config = OmegaConf.create(OmegaConf.to_container(backbone_config, resolve=False))
        
        # mdlm_config = MDLMConfig(
        #     length=length,
        #     backbone_config=backbone_config,
        # )
        mdlm_config = BD3LMConfig(
            length=backbone_config.length,
            backbone_config=backbone_config,
            block_size=cfg.block_size,
        )
        mdlm_config.mask_token_id = tokenizer.mask_token_id
        mdlm_config.vocab_size = len(tokenizer)
        # model_ = MDLM(
        #     mdlm_config,
        #     tokenizer=tokenizer,
        # )
        model_ = BD3LM(
            mdlm_config,
            tokenizer=tokenizer,
        )
        if model is not None:
            state_dict = model.state_dict()
        else:
            state_dict = torch.load(cfg.pretrained_model_name_or_path, weights_only=False)
            state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key
            if "backbone." in key:
                new_key = key.replace("backbone.", "")
            if "_orig_mod." in new_key:
                new_key = new_key.replace("_orig_mod.", "")
            new_state_dict[new_key] = state_dict[key]
        if "sampling_eps_min" in new_state_dict:
            new_state_dict.pop("sampling_eps_min")
        if "sampling_eps_max" in new_state_dict:
            new_state_dict.pop("sampling_eps_max")
        model_.backbone.load_state_dict(new_state_dict)
        model = model_.to(device)
        model.noise_schedule = LinearNoise()

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

            # print(model.tokenizer.decode(outputs[0]))
            if i >= THROUGHPUT_WARMUP:
                tputs.append((outputs.numel() - input_ids.numel()) / elapsed_time_s)
                parallelism_factors.append(parallelism_factor)
            generated_samples.append(model.tokenizer.decode(outputs[0]))
        pbar.set_postfix(tput=np.mean(tputs), parallel=np.mean(parallelism_factors))
    # gather samples across devices
    generated_samples = gather_results(generated_samples, dist.get_world_size())
    tputs = gather_results(tputs, dist.get_world_size())
    if local_rank == 0:
        print(f"TPUT (tok/s) over {len(tputs)} samples: {np.mean(tputs)} +/- {np.std(tputs)}")
    with open(f"{cfg.generated_samples_output_path}/generated_samples.json", "w") as f:
        json.dump(
            generated_samples,
            f,  # type: ignore
            indent=2,
        )
        
    return generated_samples

def compute_metrics(cfg, samples, device="cuda") -> None:
    eval_model_name = getattr(cfg, "eval_model_name", "gpt2-large")
    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_model_name,
        trust_remote_code=True,
        revision=getattr(cfg, "pretrained_model_revision", None),
    )
    eval_model = eval_model.to(device)
    eval_model.eval()
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    samples = [eval_tokenizer.encode(sample) for sample in samples]

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
    if not os.path.exists(cfg.generated_samples_output_path):
        os.makedirs(cfg.generated_samples_output_path)
    if not getattr(cfg, "eval_only", False):
        samples = generate_samples(cfg, device, local_rank)
    else:
        # read from file
        with open(f"{cfg.generated_samples_output_path}/generated_samples.json", "r") as f:
            samples = json.load(f)

    if hasattr(cfg, "eval_model_name"):
        gen_nll = compute_metrics(cfg, samples)
        print(f"Avg gen NLL under {cfg.eval_model_name}: {gen_nll}")
        print(f"Avg gen PPL under {cfg.eval_model_name}: {torch.exp(gen_nll)}")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    register_useful_resolvers()
    main()
