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
from torch.utils.data import DataLoader, DistributedSampler
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

THROUGHPUT_SAMPLES = 100
THROUGHPUT_WARMUP = 100


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
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load the dataset
    dataset = hydra.utils.instantiate(
        cfg.task.dataset,
        tokenizer=tokenizer,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True
    )

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
        except:  # Model not compatible with CausalLM
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
    tputs = []
    for elem_id, elem in tqdm(
        enumerate(dataloader),
        desc="Generating",
        total=len(dataloader),
        disable=(local_rank != 0),
    ):
        if getattr(cfg, "throughput_run", False) and elem_id >= (
            THROUGHPUT_SAMPLES + THROUGHPUT_WARMUP
        ):
            if not fsspec_exists(cfg.output_path):
                fsspec_mkdirs(cfg.output_path)
            tputs_path = f"{cfg.output_path}/throughput-rank{local_rank}"
            with open(f"{tputs_path}.json", "w") as f:
                json.dump(
                    {
                        "throughput_mean": np.mean(tputs),
                        "throughput_std": np.std(tputs),
                        "throughput_all": tputs,
                    },
                    f,  # type: ignore
                    indent=2,
                )
            if dist.is_initialized():
                dist.destroy_process_group()
            sys.exit(0)
        if local_rank == 0:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event, end_event = None, None
        input_ids = elem["input_ids"].to(device)  # type: ignore
        if dataset.target_prompt_text is not None:
            prompt_ids = (
                torch.tensor(tokenizer.encode(dataset.target_prompt_text.strip()))
                .to(input_ids.dtype)
                .to(input_ids.device)
                .unsqueeze(0)
            )
            input_ids = torch.cat((input_ids, prompt_ids), dim=-1)
        # Generate samples
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                disable_pbar=(local_rank != 0),
                # tokenizer=tokenizer,  # For debugging: prints intermediate generation
                **gen_kwargs,
            )
            if local_rank == 0:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_s = start_event.elapsed_time(end_event) / 1000
                if elem_id >= THROUGHPUT_WARMUP:
                    tputs.append((outputs.numel() - input_ids.numel()) / elapsed_time_s)
        outputs = outputs[:, input_ids.shape[-1] :]
        # Decode the generated samples
        outputs = tokenizer.decode(outputs[0])
        # Post-process:
        outputs = outputs.replace(" .", ".")
        if stop_tokens is not None:
            for st in stop_tokens:
                outputs = outputs.split(st)[0]
        decoded_samples = outputs.strip()
        if local_rank == 0:
            print("Input:", tokenizer.decode(input_ids[0]))
            print("Output:", decoded_samples)
            if elem_id >= THROUGHPUT_WARMUP:
                print(f"Thput (tok/s): {np.mean(tputs):0.2f} +/- {np.std(tputs):0.2f}")
        generated_samples.append(decoded_samples)

    # Compute metrics
    references = dataset.target_references
    local_indices = list(sampler)[: len(generated_samples)]
    references = [references[i] for i in local_indices]
    world_size = dist.get_world_size()
    generated_samples = gather_results(generated_samples, world_size)
    references = gather_results(references, world_size)
    if local_rank == 0:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("sacrebleu")
        meteor = evaluate.load("meteor")
        rouge_scores = rouge.compute(
            predictions=generated_samples, references=references
        )
        bleu_score = bleu.compute(
            predictions=generated_samples, references=[[ref] for ref in references]
        )
        meteor_score = meteor.compute(
            predictions=generated_samples, references=references
        )

        # Display results
        print("\n=== Evaluation Metrics ===\n")
        print("| Metric  | Value   |")
        print("|---------|---------|")
        print(f"| ROUGE-1 | {rouge_scores['rouge1']:>7.4f} |")
        print(f"| ROUGE-2 | {rouge_scores['rouge2']:>7.4f} |")
        print(f"| ROUGE-L | {rouge_scores['rougeL']:>7.4f} |")
        print(f"| BLEU    | {bleu_score['score']:>7.4f} |")
        print(f"| METEOR  | {meteor_score['meteor']:>7.4f} |")

        res_for_json = [
            {"ground_truth": references[i], "result": generated_samples[i]}
            for i in range(len(generated_samples))
        ]

        if not fsspec_exists(cfg.output_path):
            fsspec_mkdirs(cfg.output_path)
        with open(f"{cfg.output_path}/all_ranks.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        with open(f"{cfg.output_path}/metrics.json", "w") as f:
            json.dump(
                {
                    "ROUGE-1": rouge_scores["rouge1"],
                    "ROUGE-2": rouge_scores["rouge2"],
                    "ROUGE-L": rouge_scores["rougeL"],
                    "BLEU": bleu_score["score"],
                    "METEOR": meteor_score["meteor"],
                },
                f,  # type: ignore
                indent=2,
            )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
