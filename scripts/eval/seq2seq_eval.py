import datetime
import json
import os
import sys
from collections import OrderedDict

import evaluate
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers.modeling_outputs import ModelOutput

from scripts.eval.model_loading import load_eval_model, normalize_model_config_overrides
from scripts.utils import (
    count_parameters,
    format_number,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.denoiser.ar import AR
from src.noise_schedule.noise_schedules import LinearNoise
from src.utils import fsspec_exists, fsspec_mkdirs

THROUGHPUT_WARMUP = 50


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
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load model
    model_config_overrides = normalize_model_config_overrides(
        getattr(cfg, "model_config_overrides", None)
    )

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
    local_indices = list(sampler)

    model = load_eval_model(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        pretrained_model_revision=getattr(cfg, "pretrained_model_revision", None),
        load_ema_weights=cfg.load_ema_weights,
        ckpt_file=cfg.ckpt_file,
        model_config_overrides=model_config_overrides,
        force_legacy_if_no_generate=True,
    )

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
    gen_kwargs["generation_config"].pad_token_id = tokenizer.pad_token_id
    stop_tokens = ["<|endoftext|>"]

    # Iterate through the dataset and sample
    example_ids = []
    generated_samples = []
    tputs = []
    parallelism_factors = []
    latencies = []
    pbar = tqdm(
        dataloader, desc="Generating", total=len(dataloader), disable=(local_rank != 0)
    )
    for elem_id, elem in enumerate(pbar):
        ex_id = int(local_indices[elem_id])
        example_ids.append(ex_id)
        if getattr(cfg, "throughput_run", False) and elem_id >= THROUGHPUT_WARMUP:
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
        input_ids = elem["input_ids"].to(device)  # type: ignore
        if getattr(dataset, "target_prompt_text", None) is not None:
            prompt_ids = (
                torch.tensor(tokenizer.encode(dataset.target_prompt_text.strip()))
                .to(input_ids.dtype)
                .to(input_ids.device)
                .unsqueeze(0)
            )
            input_ids = torch.cat((input_ids, prompt_ids), dim=-1)
        # Generate samples
        with torch.no_grad():
            # if this is an ar model, only pass the left context to the model.
            if isinstance(model, AR) and (input_ids == tokenizer.mask_token_id).any():
                gen_kwargs.update(
                    {
                        "max_new_tokens": (input_ids == tokenizer.mask_token_id).sum(),
                    }
                )
                first_mask_index = (input_ids == tokenizer.mask_token_id).nonzero(
                    as_tuple=True
                )[1][0]
                input_ids = input_ids[:, :first_mask_index]
            if local_rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_event, end_event = None, None
            generation_outputs = model.generate(
                inputs=input_ids,
                disable_pbar=True,
                tokenizer=tokenizer,
                **gen_kwargs,
            )
            if local_rank == 0:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_s = start_event.elapsed_time(end_event) / 1000
            if isinstance(generation_outputs, ModelOutput):
                outputs = generation_outputs.sequences
                parallelism_factor = generation_outputs.get("parallelism_factor", -1.0)
                if parallelism_factor is None:
                    parallelism_factor = -1.0
            else:
                outputs = generation_outputs
                parallelism_factor = -1.0
            if (
                tokenizer.mask_token_id is not None
                and tokenizer.mask_token_id in elem["input_ids"]
            ):
                if isinstance(model, AR):
                    outputs = outputs[0][first_mask_index:]
                else:
                    outputs = outputs[0][
                        elem["input_ids"][0] == tokenizer.mask_token_id
                    ]
            else:
                outputs = outputs[:, input_ids.shape[-1] :].squeeze(0)
            parallelism_factors.append(parallelism_factor)
            if local_rank == 0:
                if elem_id >= THROUGHPUT_WARMUP:
                    tputs.append(outputs.numel() / elapsed_time_s)
                    latencies.append(elapsed_time_s)
        pbar.set_postfix(
            tput=np.mean(tputs),
            parallel=np.mean(parallelism_factors),
            latency=np.mean(latencies),
        )
        # Decode the generated samples
        outputs = tokenizer.decode(outputs)
        # Post-process:
        for st in stop_tokens:
            outputs = outputs.split(st)[0]
        decoded_samples = outputs.strip()
        if elem_id % 100 == 0:
            print("Input:", tokenizer.decode(elem["input_ids"][0]))
            print("Output:", decoded_samples)
            print("Output length:", len(tokenizer(decoded_samples)["input_ids"]))
            print("Ground truth:", dataset.target_references[ex_id])
            print(
                f"Parallelism factor: {np.mean(parallelism_factors):0.2f} "
                f"+/- {np.std(parallelism_factors):0.2f}"
            )
            if elem_id >= THROUGHPUT_WARMUP:
                print(f"Thput (tok/s): {np.mean(tputs):0.2f} +/- {np.std(tputs):0.2f}")
                print(
                    f"Latency (s): {np.mean(latencies):0.2f} "
                    f"+/- {np.std(latencies):0.2f}"
                )
                lat_ms = np.array(latencies) * 1000
                print(f"Latency (ms): {np.mean(lat_ms):.2f} +/- {np.std(lat_ms):.2f}")
        generated_samples.append(decoded_samples)

    # Compute metrics

    references = [dataset.target_references[i] for i in example_ids]

    # Gather from all ranks
    example_ids = gather_results(example_ids, world_size)

    generated_samples = gather_results(generated_samples, world_size)
    references = gather_results(references, world_size)
    parallelism_factors = gather_results(parallelism_factors, world_size)
    throughputs = gather_results(tputs, world_size)
    latencies = gather_results(latencies, world_size)
    if local_rank == 0:
        # Deduplicate padded/duplicated samples introduced by
        # DistributedSampler (drop_last=False), and restore deterministic
        # ordering by example id.
        by_id = OrderedDict()
        for i, pred, ref in zip(example_ids, generated_samples, references):
            if i not in by_id:
                by_id[i] = (pred, ref)
        ordered_ids = sorted(by_id.keys())
        generated_samples = [by_id[i][0] for i in ordered_ids]
        references = [by_id[i][1] for i in ordered_ids]

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
        print(
            f"Parallelism factor: {np.mean(parallelism_factors):0.2f} "
            f"+/- {np.std(parallelism_factors):0.2f}"
        )
        print(
            f"Thput (tok/s): {np.mean(throughputs):0.2f} +/- {np.std(throughputs):0.2f}"
        )
        print(f"Latency (s): {np.mean(latencies):0.2f} +/- {np.std(latencies):0.2f}")
        lat_ms = np.array(latencies) * 1000
        print(f"Latency (ms): {np.mean(lat_ms):.2f} +/- {np.std(lat_ms):.2f}")
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
