import datetime
import hashlib
import itertools
import json
import math
import os
import re
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
from src.utils import fsspec_exists, fsspec_mkdirs

THROUGHPUT_WARMUP = 50
THROUGHPUT_NUM_MEASUREMENTS = 200


def strip_generated_target_prompt(text: str, target_prompt_text: str | None) -> str:
    if not target_prompt_text:
        return text
    prompt = target_prompt_text.rstrip()
    stripped = text.lstrip()
    if stripped.startswith(prompt):
        return stripped[len(prompt) :].lstrip()
    return text


def gather_results(results, world_size):
    # Each GPU has local 'results' (any pickle-able object)
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore

    return all_results


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


def setup_ddp() -> int:
    """Sets up torch.distributed and selects GPU.

    Returns:
        (int) local_rank
    """
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def _find_subsequence(sequence: torch.Tensor, subsequence: torch.Tensor, start: int) -> int:
    if subsequence.numel() == 0:
        return start
    max_start = sequence.numel() - subsequence.numel()
    for idx in range(start, max_start + 1):
        if torch.equal(sequence[idx : idx + subsequence.numel()], subsequence):
            return idx
    return -1


def extract_infill_output_ids(
    outputs: torch.Tensor,
    input_ids: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    output_ids = outputs[0] if outputs.dim() == 2 else outputs
    source_ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    source_ids = source_ids.to(output_ids.device)
    input_mask = source_ids == mask_token_id

    if output_ids.shape[-1] == source_ids.shape[-1]:
        return output_ids[input_mask]

    mask_positions = input_mask.nonzero(as_tuple=False).flatten()
    if mask_positions.numel() == 0:
        return output_ids

    # Confidence-threshold decoding can leave some masks unresolved; generate()
    # removes those masks, so recover the compacted middle infill span by
    # aligning the unchanged context around the contiguous ROCStories mask span.
    start = int(mask_positions[0].item())
    end = int(mask_positions[-1].item()) + 1
    prefix = source_ids[:start]
    suffix = source_ids[end:]

    cursor = 0
    if prefix.numel() > 0 and output_ids.numel() >= prefix.numel():
        if torch.equal(output_ids[: prefix.numel()], prefix):
            cursor = int(prefix.numel())

    suffix_start = _find_subsequence(output_ids, suffix, cursor)
    if suffix_start >= 0:
        return output_ids[cursor:suffix_start]

    return output_ids[cursor:]


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    os.environ.setdefault("LM_EVAL_BASE_SEED", str(int(cfg.seed)))
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

    is_setdlm = model.__class__.__name__ == "SetDLM"
    if is_setdlm:
        model._setdlm_profile_decode = bool(
            getattr(cfg, "setdlm_profile_decode", False)
        )
        profile_path = getattr(cfg, "setdlm_decode_profile_path", None)
        if model._setdlm_profile_decode and profile_path is None:
            profile_path = f"{cfg.output_path}/setdlm_decode_profile_rank{local_rank}.jsonl"
        model._setdlm_decode_profile_path = profile_path
    setdlm_max_autotune_compile = False
    if getattr(cfg, "compile_backbone", False):
        # SetDLM inference calls the backbone on small, variable-length decode
        # windows. Max-autotune specializes each new window shape and can spend
        # minutes autotuning invalid Triton GEMMs on A6000. Keep dynamic default
        # compilation for SetDLM, but use static buckets for explicit max-autotune.
        compile_mode = getattr(cfg, "compile_mode", None)
        if compile_mode is None:
            compile_mode = "default" if is_setdlm else "max-autotune-no-cudagraphs"
        setdlm_max_autotune_compile = is_setdlm and compile_mode in (
            "max-autotune",
            "max-autotune-no-cudagraphs",
        )
        compile_dynamic = getattr(cfg, "compile_dynamic", None)
        if compile_dynamic is None:
            compile_dynamic = is_setdlm and not setdlm_max_autotune_compile
        else:
            compile_dynamic = bool(compile_dynamic)
        compile_kwargs = {"dynamic": compile_dynamic}
        if compile_mode not in (None, "", "default"):
            compile_kwargs["mode"] = compile_mode
        if is_setdlm:
            static_compile_cache = bool(getattr(cfg, "setdlm_static_compile_cache", False))
            clone_compile_cache_cfg = getattr(
                cfg, "setdlm_clone_compile_cache", None
            )
            if clone_compile_cache_cfg is None:
                use_clone_compile_cache = (
                    compile_mode == "reduce-overhead"
                    and not static_compile_cache
                )
            else:
                use_clone_compile_cache = bool(clone_compile_cache_cfg)
            model._setdlm_static_compile_cache = static_compile_cache
            model._setdlm_clone_compile_cache = use_clone_compile_cache
            print(
                "SetDLM compile cache "
                f"clone={use_clone_compile_cache}, static={static_compile_cache}"
            )
        print(f"Compiling model backbone with torch.compile({compile_kwargs})")
        model.backbone = torch.compile(model.backbone, **compile_kwargs)

    model = model.to(device)
    if local_rank == 0:
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
    model.eval()
    gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
    cnndm_generate_target_prompt = bool(
        getattr(cfg, "cnndm_generate_target_prompt", False)
    )
    gen_kwargs["generation_config"].pad_token_id = tokenizer.pad_token_id
    stop_tokens = ["<|endoftext|>"]

    # Iterate through the dataset and sample
    example_ids = []
    generated_samples = []
    tputs = []
    throughput_warmup = int(getattr(cfg, "throughput_warmup", THROUGHPUT_WARMUP))
    throughput_num_measurements = int(
        getattr(cfg, "throughput_num_measurements", THROUGHPUT_NUM_MEASUREMENTS)
    )
    throughput_global_measurements = bool(
        getattr(cfg, "throughput_global_measurements", False)
    )
    local_throughput_num_measurements = throughput_num_measurements
    if throughput_global_measurements and world_size > 1:
        local_throughput_num_measurements = math.ceil(
            throughput_num_measurements / world_size
        )
    parallelism_factors = []
    latencies = []
    throughput_run = bool(getattr(cfg, "throughput_run", False))
    max_eval_samples = getattr(cfg, "max_eval_samples", None)
    if max_eval_samples in ("", "null", "None"):
        max_eval_samples = None
    if max_eval_samples is not None:
        max_eval_samples = int(max_eval_samples)
    local_iteration_target = None
    if throughput_run:
        local_iteration_target = throughput_warmup + local_throughput_num_measurements
    elif max_eval_samples is not None:
        local_iteration_target = min(
            len(dataloader), math.ceil(max_eval_samples / max(world_size, 1))
        )
    if throughput_run and local_iteration_target > len(dataloader):
        pbar_iter = itertools.islice(
            itertools.cycle(enumerate(dataloader)), local_iteration_target
        )
        pbar_total = local_iteration_target
    elif local_iteration_target is not None:
        pbar_iter = itertools.islice(enumerate(dataloader), local_iteration_target)
        pbar_total = local_iteration_target
    else:
        pbar_iter = enumerate(dataloader)
        pbar_total = len(dataloader)
    pbar = tqdm(
        pbar_iter, desc="Generating", total=pbar_total, disable=(local_rank != 0)
    )
    for elem_id, (local_elem_id, elem) in enumerate(pbar):
        ex_id = int(local_indices[local_elem_id])
        example_ids.append(ex_id)
        input_ids = elem["input_ids"].to(device)  # type: ignore
        target_prompt_text = None
        if getattr(dataset, "target_prompt_text", None) is not None:
            # Drop only trailing whitespace so the prompt boundary matches training
            # tokenization, where that whitespace can merge into the next target token.
            target_prompt_text = dataset.target_prompt_text.rstrip()
            if not cnndm_generate_target_prompt:
                prompt_ids = (
                    torch.tensor(tokenizer.encode(target_prompt_text))
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
            if _rank_invariant_generation_enabled():
                example_seed = _stable_generation_seed(
                    int(os.environ.get("LM_EVAL_BASE_SEED", cfg.seed)),
                    ex_id,
                    elem.get("source_ids", ""),
                    elem.get("target_ids", ""),
                )
                _seed_generation_for_example(example_seed)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            generation_config = gen_kwargs.get("generation_config")
            if is_setdlm and generation_config is not None:
                if (
                    bool(getattr(generation_config, "setdlm_decode_diagnostic_log", False))
                    or bool(getattr(generation_config, "setdlm_decode_order_trace", False))
                    or bool(getattr(generation_config, "setdlm_decode_snapshot_log", False))
                ):
                    generation_config.setdlm_decode_diagnostic_example_id = int(ex_id)
            generation_outputs = model.generate(
                inputs=input_ids,
                disable_pbar=True,
                tokenizer=tokenizer,
                **gen_kwargs,
            )
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
            full_outputs = outputs
            prompt_slice_start = None
            prompt_boundary_ids = []
            if (
                tokenizer.mask_token_id is not None
                and tokenizer.mask_token_id in elem["input_ids"]
            ):
                if isinstance(model, AR):
                    outputs = outputs[0][first_mask_index:]
                else:
                    outputs = extract_infill_output_ids(
                        outputs, elem["input_ids"], tokenizer.mask_token_id
                    )
            else:
                prompt_slice_start = input_ids.shape[-1]
                if elem_id % 100 == 0:
                    full_output_ids = (
                        full_outputs[0] if full_outputs.dim() == 2 else full_outputs
                    )
                    prompt_boundary_ids = (
                        full_output_ids[
                            max(0, prompt_slice_start - 8) : prompt_slice_start + 20
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                outputs = outputs[:, prompt_slice_start:].squeeze(0)
            raw_output_ids = outputs.detach().cpu().tolist()
            if isinstance(raw_output_ids, int):
                raw_output_ids = [raw_output_ids]
            parallelism_factors.append(parallelism_factor)
            local_should_stop_throughput_run = False
            if elem_id >= throughput_warmup:
                tputs.append(outputs.numel() / elapsed_time_s)
                latencies.append(elapsed_time_s)
                local_should_stop_throughput_run = (
                    getattr(cfg, "throughput_run", False)
                    and len(tputs) >= local_throughput_num_measurements
                )
            if world_size > 1:
                stop_tensor = torch.tensor(
                    [int(local_should_stop_throughput_run)], device=device
                )
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MIN)
                should_stop_throughput_run = bool(stop_tensor.item())
            else:
                should_stop_throughput_run = local_should_stop_throughput_run
            if should_stop_throughput_run:
                if local_rank == 0 and not fsspec_exists(cfg.output_path):
                    fsspec_mkdirs(cfg.output_path)
                if dist.is_initialized():
                    dist.barrier()
                tputs_path = f"{cfg.output_path}/throughput-rank{local_rank}"
                with open(f"{tputs_path}.json", "w") as f:
                    json.dump(
                        {
                            "throughput_mean": np.mean(tputs),
                            "throughput_std": np.std(tputs),
                            "throughput_all": tputs,
                            "latency_mean": np.mean(latencies),
                            "latency_std": np.std(latencies),
                            "latency_all": latencies,
                            "warmup_examples": throughput_warmup,
                            "measured_examples": len(tputs),
                            "requested_measured_examples": throughput_num_measurements,
                            "local_measurement_target": local_throughput_num_measurements,
                            "throughput_global_measurements": throughput_global_measurements,
                            "world_size": world_size,
                        },
                        f,  # type: ignore
                        indent=2,
                    )
                gathered_tputs = [None for _ in range(world_size)]
                gathered_latencies = [None for _ in range(world_size)]
                gathered_parallelism = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_tputs, tputs)
                dist.all_gather_object(gathered_latencies, latencies)
                dist.all_gather_object(gathered_parallelism, parallelism_factors)
                if local_rank == 0:
                    all_tputs = [x for part in gathered_tputs for x in part]
                    all_latencies = [x for part in gathered_latencies for x in part]
                    all_parallelism = [x for part in gathered_parallelism for x in part]
                    measured_throughputs = all_tputs[:throughput_num_measurements]
                    measured_latencies = all_latencies[:throughput_num_measurements]
                    measured_parallelism = all_parallelism[:throughput_num_measurements]
                    with open(f"{cfg.output_path}/throughput-all.json", "w") as f:
                        json.dump(
                            {
                                "throughput_mean": np.mean(measured_throughputs),
                                "throughput_std": np.std(measured_throughputs),
                                "throughput_all": measured_throughputs,
                                "latency_mean": np.mean(measured_latencies),
                                "latency_std": np.std(measured_latencies),
                                "latency_all": measured_latencies,
                                "parallelism_factor_mean": (
                                    np.mean(measured_parallelism)
                                    if measured_parallelism
                                    else None
                                ),
                                "parallelism_factor_std": (
                                    np.std(measured_parallelism)
                                    if measured_parallelism
                                    else None
                                ),
                                "parallelism_factor_all": measured_parallelism,
                                "warmup_examples_per_rank": throughput_warmup,
                                "measured_examples": len(measured_throughputs),
                                "requested_measured_examples": throughput_num_measurements,
                                "local_measurement_target": local_throughput_num_measurements,
                                "throughput_global_measurements": throughput_global_measurements,
                                "world_size": world_size,
                            },
                            f,  # type: ignore
                            indent=2,
                        )
                if dist.is_initialized():
                    dist.destroy_process_group()
                sys.exit(0)
        pbar.set_postfix(
            tput=np.mean(tputs),
            parallel=np.mean(parallelism_factors),
            latency=np.mean(latencies),
        )
        # Decode the generated samples
        decoded_outputs_before_postprocess = tokenizer.decode(outputs)
        outputs = decoded_outputs_before_postprocess
        # Post-process:
        for st in stop_tokens:
            outputs = outputs.split(st)[0]
        if cnndm_generate_target_prompt:
            outputs = strip_generated_target_prompt(outputs, target_prompt_text)
        decoded_samples = outputs.strip()
        if bool(getattr(cfg, "cnndm_diagnostic_log", False)):
            raw_eos_offsets = []
            first_eos_offset = None
            decoded_raw_after_first_eos = ""
            if tokenizer.eos_token_id is not None:
                raw_eos_offsets = [
                    offset
                    for offset, token_id in enumerate(raw_output_ids)
                    if token_id == tokenizer.eos_token_id
                ]
                if raw_eos_offsets:
                    first_eos_offset = raw_eos_offsets[0]
                    decoded_raw_after_first_eos = tokenizer.decode(
                        raw_output_ids[first_eos_offset + 1 :]
                    )[:320]
            print(
                "Seq2Seq CNN/DM diagnostic: "
                + json.dumps(
                    {
                        "event": "example",
                        "example_id": int(ex_id),
                        "model_class": model.__class__.__name__,
                        "input_length": int(input_ids.shape[-1]),
                        "prompt_slice_start": (
                            None if prompt_slice_start is None else int(prompt_slice_start)
                        ),
                        "raw_output_numel": len(raw_output_ids),
                        "raw_eos_offsets": raw_eos_offsets,
                        "raw_eos_count": len(raw_eos_offsets),
                        "first_generated_eos_offset": first_eos_offset,
                        "decoded_raw_after_first_eos": decoded_raw_after_first_eos,
                        "generated_word_length": len(
                            re.findall(r"\b\w+(?:['-]\w+)?\b", decoded_samples)
                        ),
                        "decoded_prefix": decoded_samples[:160],
                        "use_first_hitting_order_in_decode": bool(
                            getattr(
                                gen_kwargs.get("generation_config"),
                                "use_first_hitting_order_in_decode",
                                False,
                            )
                        ),
                        "cnndm_generate_target_prompt": bool(
                            cnndm_generate_target_prompt
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if elem_id % 100 == 0:
            raw_output_head = raw_output_ids[:20]
            raw_output_head_tokens = [
                repr(tokenizer.decode([token_id])) for token_id in raw_output_head
            ]
            prompt_boundary_tokens = [
                repr(tokenizer.decode([token_id])) for token_id in prompt_boundary_ids
            ]
            print("Input:", tokenizer.decode(elem["input_ids"][0]))
            print("Raw output numel:", len(raw_output_ids))
            print("Raw output first 20 ids:", raw_output_head)
            print("Raw output first 20 tokens:", raw_output_head_tokens)
            print(
                "Raw output first token is eos:",
                bool(raw_output_ids and raw_output_ids[0] == tokenizer.eos_token_id),
            )
            print("Raw output eos count:", raw_output_ids.count(tokenizer.eos_token_id))
            if tokenizer.mask_token_id is not None:
                print(
                    "Raw output mask count:",
                    raw_output_ids.count(tokenizer.mask_token_id),
                )
            if prompt_slice_start is not None:
                print("Prompt slice start:", int(prompt_slice_start))
                print("Full output shape:", tuple(full_outputs.shape))
                print("Prompt boundary ids [-8:+20]:", prompt_boundary_ids)
                print("Prompt boundary tokens [-8:+20]:", prompt_boundary_tokens)
            print(
                "Decoded before postprocess:",
                repr(decoded_outputs_before_postprocess[:200]),
            )
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
        if getattr(cfg, "throughput_run", False):
            measured_throughputs = throughputs[:throughput_num_measurements]
            measured_latencies = latencies[:throughput_num_measurements]
            measured_parallelism = parallelism_factors[:throughput_num_measurements]
            with open(f"{cfg.output_path}/throughput-all.json", "w") as f:
                json.dump(
                    {
                        "throughput_mean": np.mean(measured_throughputs),
                        "throughput_std": np.std(measured_throughputs),
                        "throughput_all": measured_throughputs,
                        "latency_mean": np.mean(measured_latencies),
                        "latency_std": np.std(measured_latencies),
                        "latency_all": measured_latencies,
                        "parallelism_factor_mean": (
                            np.mean(measured_parallelism)
                            if measured_parallelism
                            else None
                        ),
                        "parallelism_factor_std": (
                            np.std(measured_parallelism)
                            if measured_parallelism
                            else None
                        ),
                        "parallelism_factor_all": measured_parallelism,
                        "warmup_examples_per_rank": throughput_warmup,
                        "measured_examples": len(measured_throughputs),
                        "requested_measured_examples": throughput_num_measurements,
                        "local_measurement_target": local_throughput_num_measurements,
                        "throughput_global_measurements": throughput_global_measurements,
                        "world_size": world_size,
                    },
                    f,  # type: ignore
                    indent=2,
                )
        with open(f"{cfg.output_path}/all_ranks.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        metrics_for_json = {
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "BLEU": bleu_score["score"],
            "METEOR": meteor_score["meteor"],
            "parallelism_factor_mean": float(np.mean(parallelism_factors))
            if parallelism_factors
            else None,
            "parallelism_factor_std": float(np.std(parallelism_factors))
            if parallelism_factors
            else None,
            "throughput_tok_per_s_mean": float(np.mean(throughputs))
            if throughputs
            else None,
            "throughput_tok_per_s_std": float(np.std(throughputs))
            if throughputs
            else None,
            "latency_s_mean": float(np.mean(latencies)) if latencies else None,
            "latency_s_std": float(np.std(latencies)) if latencies else None,
            "throughput_measured_examples": len(throughputs),
            "throughput_warmup_examples_per_rank": throughput_warmup,
            "throughput_global_measurements": throughput_global_measurements,
            "world_size": world_size,
        }
        with open(f"{cfg.output_path}/metrics.json", "w") as f:
            json.dump(
                metrics_for_json,
                f,  # type: ignore
                indent=2,
            )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
