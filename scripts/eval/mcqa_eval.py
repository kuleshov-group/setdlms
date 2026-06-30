import datetime
import json
import math
import os
from collections import OrderedDict
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from scripts.eval.model_loading import load_eval_model, normalize_model_config_overrides
from scripts.utils import (
    count_parameters,
    format_number,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.denoiser.ar import AR
from src.denoiser.base import Denoiser
from src.utils import fsspec_exists, fsspec_mkdirs


def gather_results(results, world_size):
    if world_size == 1:
        return results
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore[arg-type]
    return all_results


def setup_ddp() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=120),
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def build_hellaswag_prompt(example: dict[str, Any]) -> tuple[str, list[str], int]:
    activity_label = normalize_text(example.get("activity_label", ""))
    ctx_a = normalize_text(example.get("ctx_a", ""))
    ctx_b = capitalize_first(normalize_text(example.get("ctx_b", "")))
    context = " ".join(part for part in [ctx_a, ctx_b] if part)
    prompt_lines = []
    if activity_label:
        prompt_lines.append(f"Activity: {activity_label}")
    prompt_lines.append(f"Context: {context}")
    prompt_lines.append("Continuation:")
    options = [normalize_text(option) for option in example["endings"]]
    return "\n".join(prompt_lines), options, int(example["label"])


def build_piqa_prompt(example: dict[str, Any]) -> tuple[str, list[str], int]:
    goal = normalize_text(example["goal"])
    prompt = (
        f"Question: Which option best accomplishes the following goal?\n{goal}\nAnswer:"
    )
    options = [normalize_text(example["sol1"]), normalize_text(example["sol2"])]
    return prompt, options, int(example["label"])


def build_siqa_prompt(example: dict[str, Any]) -> tuple[str, list[str], int]:
    context = normalize_text(example["context"])
    question = normalize_text(example["question"])
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    label = int(str(example["label"]).strip()) - 1
    options = [
        normalize_text(example["answerA"]),
        normalize_text(example["answerB"]),
        normalize_text(example["answerC"]),
    ]
    return prompt, options, label


PROMPT_BUILDERS = {
    "hellaswag": build_hellaswag_prompt,
    "piqa": build_piqa_prompt,
    "siqa": build_siqa_prompt,
}


def load_benchmark_dataset(
    benchmark_name: str,
    benchmark_cfg: DictConfig,
) -> Dataset:
    load_errors = []
    subset_name = getattr(benchmark_cfg, "subset_name", None)
    for dataset_name in benchmark_cfg.dataset_names:
        try:
            return load_dataset(
                dataset_name,
                subset_name,
                split=benchmark_cfg.split,
                trust_remote_code=True,
            )
        except Exception as exc:
            load_errors.append(f"{dataset_name}: {exc}")
    raise RuntimeError(
        f"Failed to load dataset for benchmark '{benchmark_name}'. "
        + " Tried: "
        + "; ".join(load_errors)
    )


def resolve_benchmark_max_examples(cfg: DictConfig) -> int | None:
    explicit = getattr(getattr(cfg, "task", None), "max_examples", None)
    if explicit is not None:
        return int(explicit)
    if bool(getattr(getattr(cfg, "task", None), "test_mode", False)):
        return int(getattr(cfg.task, "test_num_examples_per_benchmark", 32))
    return None


def maybe_subsample_benchmark_dataset(
    dataset: Dataset,
    *,
    cfg: DictConfig,
    max_examples: int | None,
) -> Dataset:
    if max_examples is None:
        return dataset
    if bool(getattr(cfg.task, "test_mode", False)) and bool(
        getattr(cfg.task, "test_shuffle", True)
    ):
        dataset = dataset.shuffle(seed=int(cfg.seed))
    return dataset.select(range(min(int(max_examples), len(dataset))))


def load_mcqa_model(cfg: DictConfig, tokenizer, device: torch.device):
    model_config_overrides = normalize_model_config_overrides(
        getattr(cfg, "model_config_overrides", None)
    )
    model = load_eval_model(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        pretrained_model_revision=getattr(cfg, "pretrained_model_revision", None),
        load_ema_weights=cfg.task.load_ema_weights,
        ckpt_file=cfg.task.ckpt_file,
        model_config_overrides=model_config_overrides,
        force_legacy_if_no_generate=False,
    )
    return model


def resolve_max_length(cfg: DictConfig, model, tokenizer) -> int:
    candidates = [
        getattr(cfg, "max_length", None),
        getattr(getattr(model, "config", None), "length", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int) and candidate > 0 and candidate < 1_000_000:
            return candidate
    raise ValueError("Could not resolve a finite max sequence length for MCQA eval.")


def encode_prompt_and_options(
    tokenizer,
    prompt: str,
    options: list[str],
    max_length: int,
):
    encoded_options = []
    for option in options:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        option_text = option if prompt.endswith((" ", "\n", "\t")) else f" {option}"
        option_ids = tokenizer(option_text, add_special_tokens=False)["input_ids"]
        if len(option_ids) == 0:
            raise ValueError(f"Empty tokenization for MCQA option: {option!r}")
        if len(option_ids) >= max_length:
            raise ValueError(
                "Answer option is longer than the maximum supported sequence length."
            )
        available_prompt_tokens = max_length - len(option_ids)
        truncated_prompt_ids = prompt_ids[-available_prompt_tokens:]
        full_ids = truncated_prompt_ids + option_ids
        context_mask = [1] * len(truncated_prompt_ids) + [0] * len(option_ids)
        encoded_options.append(
            {
                "input_ids": full_ids,
                "context_mask": context_mask,
                "answer_token_count": len(option_ids),
                "prompt_truncated": len(truncated_prompt_ids) != len(prompt_ids),
            }
        )
    return encoded_options


def pad_option_batch(
    encoded_options,
    pad_token_id: int,
    device: torch.device,
    target_length: int | None = None,
):
    max_len = max(len(option["input_ids"]) for option in encoded_options)
    if target_length is not None:
        if target_length < max_len:
            raise ValueError(
                "MCQA pad target cannot be shorter than the longest encoded option."
            )
        max_len = target_length
    input_ids = []
    attention_mask = []
    context_mask = []
    answer_token_counts = []
    truncation_flags = []
    for option in encoded_options:
        pad_len = max_len - len(option["input_ids"])
        input_ids.append(option["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(option["input_ids"]) + [0] * pad_len)
        context_mask.append(option["context_mask"] + [1] * pad_len)
        answer_token_counts.append(option["answer_token_count"])
        truncation_flags.append(option["prompt_truncated"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "context_mask": torch.tensor(context_mask, dtype=torch.long, device=device),
        "answer_token_counts": torch.tensor(
            answer_token_counts, dtype=torch.long, device=device
        ),
        "prompt_truncated": truncation_flags,
    }


def sample_mcqa_t(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    sampling_eps: float,
    restricted_t_range: list[float] | tuple[float, float] | None,
    block_size: int | None,
):
    num_blocks = math.ceil(seq_len / block_size) if block_size else 1
    if block_size:
        eps_t = torch.rand(1, num_blocks, device=device)
    else:
        eps_t = torch.rand(1, device=device)
    t = (1 - sampling_eps) * eps_t + sampling_eps
    if restricted_t_range is not None:
        low, high = restricted_t_range
        t = (low - high) * t + high
    if block_size:
        t = t.repeat_interleave(block_size, dim=1)[:, :seq_len]
    return t.expand(batch_size, *t.shape[1:])


class MCQAScorer:
    def __init__(self, cfg: DictConfig, model, tokenizer, device: torch.device):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = resolve_max_length(cfg, model, tokenizer)
        self.model_context_length = getattr(
            getattr(model, "config", None), "length", None
        )
        self.pad_token_id = tokenizer.pad_token_id
        self.normalize_by_length = cfg.task.normalize_by_answer_length
        self.num_importance_samples = cfg.task.num_importance_samples
        self.sampling_eps = cfg.task.sampling_eps
        self.restricted_t_range = getattr(cfg.task, "restricted_t_range", None)
        if (
            self.model_context_length is not None
            and self.max_length > self.model_context_length
        ):
            raise ValueError(
                "MCQA eval max_length exceeds the model context length: "
                f"{self.max_length} > {self.model_context_length}."
            )
        self.model.eval()

    def _is_denoiser(self) -> bool:
        return isinstance(self.model, Denoiser)

    def _is_causal(self) -> bool:
        if isinstance(self.model, AR):
            return True
        if self._is_denoiser():
            return False
        model_name = self.model.__class__.__name__
        if "MaskedLM" in model_name:
            return False
        if "CausalLM" in model_name or "LMHeadModel" in model_name:
            return True
        return bool(getattr(getattr(self.model, "config", None), "is_decoder", False))

    def _denoiser_block_size(self) -> int | None:
        eval_block_size = getattr(
            getattr(self.model, "config", None), "eval_block_size", None
        )
        if eval_block_size is not None:
            return eval_block_size
        model_block_size = getattr(
            getattr(self.model, "config", None), "block_size", None
        )
        if model_block_size is not None:
            return model_block_size
        cfg_block_size = getattr(self.cfg, "block_size", None)
        return cfg_block_size if cfg_block_size is not None else None

    def _batch_pad_length(self) -> int | None:
        if self._is_denoiser():
            return self.model_context_length or self.max_length
        return None

    @torch.no_grad()
    def _score_causal_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> list[dict[str, float]]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if logits.shape[1] == batch["input_ids"].shape[1]:
            # Standard causal LM logits predict the next token at each position.
            log_probs = logits[:, :-1, :].log_softmax(dim=-1)
        elif logits.shape[1] == batch["input_ids"].shape[1] - 1:
            # AR denoiser wrappers already left-shift inputs internally.
            log_probs = logits.log_softmax(dim=-1)
        else:
            raise ValueError(
                "Unexpected causal MCQA logits length. "
                f"Got logits len {logits.shape[1]} for input len "
                f"{batch['input_ids'].shape[1]}."
            )
        targets = batch["input_ids"][:, 1 : 1 + log_probs.shape[1]]
        target_log_probs = torch.gather(
            log_probs, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        answer_mask = (
            batch["attention_mask"][:, 1 : 1 + log_probs.shape[1]]
            * (1 - batch["context_mask"][:, 1 : 1 + log_probs.shape[1]])
        ).to(target_log_probs.dtype)
        total_log_probs = (target_log_probs * answer_mask).sum(dim=-1)
        token_counts = answer_mask.sum(dim=-1).clamp_min(1)
        avg_log_probs = total_log_probs / token_counts
        return [
            {
                "score": avg_log_probs[i].item()
                if self.normalize_by_length
                else total_log_probs[i].item(),
                "total_logprob": total_log_probs[i].item(),
                "avg_logprob": avg_log_probs[i].item(),
                "answer_token_count": int(token_counts[i].item()),
            }
            for i in range(batch["input_ids"].shape[0])
        ]

    @torch.no_grad()
    def _score_denoiser_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> list[dict[str, float]]:
        batch_size, seq_len = batch["input_ids"].shape
        block_size = self._denoiser_block_size()
        total_nll = torch.zeros(batch_size, device=self.device, dtype=torch.float64)
        for _ in range(self.num_importance_samples):
            t = sample_mcqa_t(
                batch_size=batch_size,
                seq_len=seq_len,
                device=self.device,
                sampling_eps=self.sampling_eps,
                restricted_t_range=self.restricted_t_range,
                block_size=block_size,
            )
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                context_mask=batch["context_mask"],
                t=t,
                compute_loss=True,
            )
            total_nll += outputs.nlls.sum(dim=-1).to(torch.float64)
        mean_nll = total_nll / self.num_importance_samples
        token_counts = batch["answer_token_counts"].clamp_min(1).to(torch.float64)
        avg_log_probs = -mean_nll / token_counts
        total_log_probs = -mean_nll
        return [
            {
                "score": avg_log_probs[i].item()
                if self.normalize_by_length
                else total_log_probs[i].item(),
                "total_logprob": total_log_probs[i].item(),
                "avg_logprob": avg_log_probs[i].item(),
                "answer_token_count": int(batch["answer_token_counts"][i].item()),
            }
            for i in range(batch_size)
        ]

    @torch.no_grad()
    def _score_masked_lm_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> list[dict[str, float]]:
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Generic masked-LM MCQA scoring requires a tokenizer mask token."
            )
        scores = []
        for option_index in range(batch["input_ids"].shape[0]):
            input_ids = batch["input_ids"][option_index]
            attention_mask = batch["attention_mask"][option_index]
            answer_positions = (
                ((attention_mask == 1) & (batch["context_mask"][option_index] == 0))
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            variants = input_ids.unsqueeze(0).repeat(answer_positions.numel(), 1)
            variants[torch.arange(answer_positions.numel()), answer_positions] = (
                self.tokenizer.mask_token_id
            )
            outputs = self.model(
                input_ids=variants,
                attention_mask=attention_mask.unsqueeze(0).expand_as(variants),
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            log_probs = logits.log_softmax(dim=-1)
            target_tokens = input_ids[answer_positions]
            token_log_probs = log_probs[
                torch.arange(answer_positions.numel(), device=self.device),
                answer_positions,
                target_tokens,
            ]
            total_log_prob = token_log_probs.sum().item()
            avg_log_prob = total_log_prob / max(answer_positions.numel(), 1)
            scores.append(
                {
                    "score": avg_log_prob
                    if self.normalize_by_length
                    else total_log_prob,
                    "total_logprob": total_log_prob,
                    "avg_logprob": avg_log_prob,
                    "answer_token_count": int(answer_positions.numel()),
                }
            )
        return scores

    def score_options(
        self, prompt: str, options: list[str]
    ) -> tuple[list[dict[str, float]], bool]:
        encoded_options = encode_prompt_and_options(
            tokenizer=self.tokenizer,
            prompt=prompt,
            options=options,
            max_length=self.max_length,
        )
        batch = pad_option_batch(
            encoded_options=encoded_options,
            pad_token_id=self.pad_token_id,
            device=self.device,
            target_length=self._batch_pad_length(),
        )
        if self._is_causal():
            scores = self._score_causal_batch(batch)
        elif self._is_denoiser():
            scores = self._score_denoiser_batch(batch)
        else:
            scores = self._score_masked_lm_batch(batch)
        return scores, any(batch["prompt_truncated"])


def format_metrics_table(metrics_by_task: OrderedDict, mean_accuracy: float) -> str:
    lines = [
        "| Benchmark | Accuracy | Examples |",
        "|-----------|----------|----------|",
    ]
    for benchmark_name, metrics in metrics_by_task.items():
        lines.append(
            f"| {benchmark_name} | {metrics['accuracy']:.4f} | "
            f"{metrics['num_examples']} |"
        )
    lines.append(f"| mean | {mean_accuracy:.4f} | - |")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    rank, world_size, device = setup_ddp()

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    model = load_mcqa_model(cfg=cfg, tokenizer=tokenizer, device=device)
    if getattr(cfg, "compile_backbone", False) and hasattr(model, "backbone"):
        print("Compiling model backbone")
        model.backbone = torch.compile(
            model.backbone, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    if rank == 0:
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")

    scorer = MCQAScorer(cfg=cfg, model=model, tokenizer=tokenizer, device=device)

    if rank == 0 and bool(getattr(cfg.task, "test_mode", False)):
        print(
            "MCQA test_mode enabled: evaluating "
            f"{int(getattr(cfg.task, 'test_num_examples_per_benchmark', 32))} "
            "examples per benchmark"
            + (" (shuffled)." if bool(getattr(cfg.task, "test_shuffle", True)) else ".")
        )

    local_results = []
    for benchmark_name in cfg.task.benchmarks:
        benchmark_cfg = cfg.task.benchmark_configs[benchmark_name]
        dataset = load_benchmark_dataset(benchmark_name, benchmark_cfg)
        max_examples = resolve_benchmark_max_examples(cfg)
        dataset = maybe_subsample_benchmark_dataset(
            dataset,
            cfg=cfg,
            max_examples=max_examples,
        )
        local_indices = list(range(rank, len(dataset), world_size))
        pbar = tqdm(
            local_indices,
            desc=f"Evaluating {benchmark_name}",
            disable=(rank != 0),
        )
        for dataset_index in pbar:
            raw_example = dataset[dataset_index]
            prompt, options, gold_label = PROMPT_BUILDERS[benchmark_name](raw_example)
            option_scores, was_truncated = scorer.score_options(
                prompt=prompt, options=options
            )
            predicted_label = int(
                max(
                    range(len(option_scores)),
                    key=lambda idx: option_scores[idx]["score"],
                )
            )
            local_results.append(
                {
                    "benchmark": benchmark_name,
                    "example_id": str(
                        raw_example.get("id", f"{benchmark_name}-{dataset_index}")
                    ),
                    "dataset_index": dataset_index,
                    "prompt": prompt,
                    "answer_options": options,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "correct": int(predicted_label == gold_label),
                    "per_option_scores": option_scores,
                    "prompt_truncated": was_truncated,
                }
            )

    all_results = gather_results(local_results, world_size)

    if rank == 0:
        deduped_results = OrderedDict()
        for result in sorted(
            all_results, key=lambda row: (row["benchmark"], row["dataset_index"])
        ):
            deduped_results[(result["benchmark"], result["dataset_index"])] = result
        final_results = list(deduped_results.values())

        metrics_by_task = OrderedDict()
        selected_benchmarks = list(cfg.task.benchmarks)
        for benchmark_name in selected_benchmarks:
            benchmark_results = [
                row for row in final_results if row["benchmark"] == benchmark_name
            ]
            accuracy = (
                float(np.mean([row["correct"] for row in benchmark_results]))
                if benchmark_results
                else 0.0
            )
            metrics_by_task[benchmark_name] = {
                "accuracy": accuracy,
                "num_examples": len(benchmark_results),
            }
        mean_accuracy = float(
            np.mean([metrics["accuracy"] for metrics in metrics_by_task.values()])
        )

        output_path = cfg.output_path
        if not fsspec_exists(output_path):
            fsspec_mkdirs(output_path)
        metrics_table = format_metrics_table(metrics_by_task, mean_accuracy)
        print("\n=== MCQA Metrics ===\n")
        print(metrics_table)

        predictions_path = os.path.join(output_path, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(final_results, f, indent=2)
        metrics_path = os.path.join(output_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "metrics_by_task": metrics_by_task,
                    "mean_accuracy": mean_accuracy,
                    "score_normalization": (
                        "average_logprob_per_answer_token"
                        if cfg.task.normalize_by_answer_length
                        else "total_logprob"
                    ),
                    "max_examples": cfg.task.max_examples,
                    "test_mode": bool(getattr(cfg.task, "test_mode", False)),
                    "test_num_examples_per_benchmark": int(
                        getattr(cfg.task, "test_num_examples_per_benchmark", 32)
                    ),
                    "test_shuffle": bool(getattr(cfg.task, "test_shuffle", True)),
                    "num_importance_samples": cfg.task.num_importance_samples,
                    "sampling_eps": cfg.task.sampling_eps,
                    "restricted_t_range": cfg.task.restricted_t_range,
                    "num_predictions": len(final_results),
                },
                f,
                indent=2,
            )
        metrics_txt_path = os.path.join(output_path, "metrics.txt")
        with open(metrics_txt_path, "w") as f:
            f.write(metrics_table)
            f.write("\n\n")
            f.write(
                "Score normalization: average log-prob per answer token.\n"
                if cfg.task.normalize_by_answer_length
                else "Score normalization: total log-prob.\n"
            )
            if isinstance(model, Denoiser):
                f.write(
                    "Denoiser likelihood estimates use the repo's stochastic loss "
                    f"estimator averaged over {cfg.task.num_importance_samples} "
                    "sampled timesteps per option.\n"
                )
            elif not scorer._is_causal():
                f.write(
                    "Generic masked LM fallback uses pseudo-log-likelihood over answer "
                    "tokens by masking one answer token at a time.\n"
                )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
