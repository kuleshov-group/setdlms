"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import re
import sys
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import accelerate
import hydra
import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import InitProcessGroupKwargs
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import ModelOutput

from datasets import Dataset
from scripts.eval.model_loading import (
    configure_rank_local_torchinductor_cache,
    load_eval_model,
    normalize_model_config_overrides,
)
from scripts.utils import (
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs


def gather_results(results, world_size):
    # Each GPU has local 'results' (any pickle-able object)
    if world_size == 1:
        return results
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore

    return all_results


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = float(np.mean(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_summary(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "auto"}:
            return None
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


class LMEvalHarnessModel(LM):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        generated_samples_output_path: str,
        tokenizer: PreTrainedTokenizer,
        pretrained_model_revision: str | None = None,
        load_ema_weights: bool = False,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        gen_kwargs: Any | None = None,
        accelerator: accelerate.Accelerator | None = None,
        throughput_run: bool = False,
        throughput_samples: int = 100,
        throughput_warmup: int = 100,
        throughput_global_measurements: bool = False,
        stop_on_im_end: bool | str = False,
        compile_backbone: bool = False,
        compile_mode: str | None = None,
        compile_dynamic: bool | None = None,
        model_config_overrides: dict[str, Any] | None = None,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Path to ckpt dir or HF model repo.
            generated_samples_output_path (str): Path to generated samples dir.
            tokenizer (str): Tokenizer name or path.
            pretrained_model_revision (Optional[str]): Revision (e.g., commit id)
                passed to `.from_pretrained` model instantiation.
            load_ema_weights (bool): Whether to load ema weights (for local ckpts).
            ckpt_file (str): Name of ckpt file (for local ckpts).
            gen_kwargs (dict): Generator kwargs.
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
            throughput_run (bool): Whether to run the evaluation throughput.
            model_config_overrides (dict[str, Any]): Model config overrides.
        """
        super().__init__()
        self.generated_samples_output_path = generated_samples_output_path
        if not fsspec_exists(self.generated_samples_output_path):
            fsspec_mkdirs(self.generated_samples_output_path)
        self.accelerator = accelerator
        if self.accelerator is not None:
            device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rank = 0
            self._world_size = 1
        self.device = torch.device(f"{device}")

        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        model_config_overrides = normalize_model_config_overrides(
            model_config_overrides
        )
        self.model = load_eval_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=self.tokenizer,
            device=self.device,
            pretrained_model_revision=pretrained_model_revision,
            load_ema_weights=load_ema_weights,
            ckpt_file=ckpt_file,
            model_config_overrides=model_config_overrides,
        )
        is_setdlm = self.model.__class__.__name__ == "SetDLM"
        self.compile_backbone = bool(compile_backbone)
        self.compile_supported = bool(hasattr(self.model, "backbone"))
        if compile_mode in ("", "none", "None", "null", "default"):
            compile_mode = None
        if compile_mode is None and not is_setdlm:
            compile_mode = "max-autotune-no-cudagraphs"
        compile_dynamic_value = _optional_bool(compile_dynamic)
        if compile_dynamic_value is None:
            compile_dynamic_value = True if is_setdlm else False
        self.compile_mode = compile_mode or "default"
        self.compile_dynamic = compile_dynamic_value
        if self.compile_backbone:
            if not self.compile_supported:
                print(
                    "COMPILE_BACKBONE=true requested, but this model has no "
                    "backbone; running uncompiled."
                )
            else:
                cache_dir = configure_rank_local_torchinductor_cache()
                if cache_dir:
                    print(f"Using rank-local TorchInductor cache: {cache_dir}")
                compile_kwargs = {"dynamic": compile_dynamic_value}
                if compile_mode is not None:
                    compile_kwargs["mode"] = compile_mode
                print(f"Compiling model backbone with torch.compile({compile_kwargs})")
                self.model.backbone = torch.compile(self.model.backbone, **compile_kwargs)
        print(
            "Compile settings: "
            f"requested={self.compile_backbone}, "
            f"supported={self.compile_supported}, "
            f"mode={self.compile_mode}, dynamic={self.compile_dynamic}"
        )
        self.model.eval()
        self.gen_kwargs = gen_kwargs
        self.throughput_run = throughput_run
        self.throughput_warmup = throughput_warmup
        self.throughput_samples = throughput_samples
        self.throughput_global_measurements = throughput_global_measurements
        self.stop_on_im_end = bool(_optional_bool(stop_on_im_end))

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    def _generation_config_value(self, name: str, default: Any = None) -> Any:
        gen_kwargs = self.gen_kwargs or {}
        try:
            generation_config = gen_kwargs.get("generation_config")
        except AttributeError:
            generation_config = getattr(gen_kwargs, "generation_config", None)
        if generation_config is None:
            return default
        if isinstance(generation_config, dict):
            return generation_config.get(name, default)
        return getattr(generation_config, name, default)

    def _write_throughput_outputs(
        self,
        tputs: list[float],
        latencies: list[float],
        response_lengths: list[float],
        parallelism_factors: list[float],
        non_ar_tokens_per_step_values: list[float],
        inf_budgets: list[float],
        local_measurement_target: int,
    ) -> None:
        rank_summary = {
            "throughput_mean": _numeric_summary(tputs)[0],
            "throughput_std": _numeric_summary(tputs)[1],
            "throughput_all": tputs,
            "latency_mean": _numeric_summary(latencies)[0],
            "latency_std": _numeric_summary(latencies)[1],
            "latency_all": latencies,
            "warmup_examples_per_rank": self.throughput_warmup,
            "measured_examples": len(tputs),
            "requested_measured_examples": self.throughput_samples,
            "local_measurement_target": local_measurement_target,
            "throughput_global_measurements": self.throughput_global_measurements,
            "world_size": self.world_size,
        }
        with open(
            f"{self.generated_samples_output_path}/throughput-rank{self.rank}.json",
            "w",
        ) as f:
            json.dump(rank_summary, f, indent=2)

        all_tputs = gather_results(tputs, self.world_size)
        all_latencies = gather_results(latencies, self.world_size)
        all_response_lengths = gather_results(response_lengths, self.world_size)
        all_parallelism_factors = gather_results(parallelism_factors, self.world_size)
        all_non_ar_tokens_per_step_values = gather_results(
            non_ar_tokens_per_step_values, self.world_size
        )
        all_inf_budgets = gather_results(inf_budgets, self.world_size)

        if self.throughput_global_measurements:
            limit = self.throughput_samples
            all_tputs = all_tputs[:limit]
            all_latencies = all_latencies[:limit]
            all_response_lengths = all_response_lengths[:limit]
            all_parallelism_factors = all_parallelism_factors[:limit]
            all_non_ar_tokens_per_step_values = all_non_ar_tokens_per_step_values[
                :limit
            ]
            all_inf_budgets = all_inf_budgets[:limit]

        output_lengths_from_timing = [
            float(throughput) * float(latency)
            for throughput, latency in zip(all_tputs, all_latencies)
        ]
        valid_parallelism_factors = [
            float(x) for x in all_parallelism_factors if x is not None and x >= 0
        ]
        nfe = self._generation_config_value("num_steps")
        tput_mean, tput_std = _numeric_summary(all_tputs)
        latency_mean, latency_std = _numeric_summary(all_latencies)
        out_len_mean, out_len_std = _numeric_summary(output_lengths_from_timing)
        resp_len_mean, resp_len_std = _numeric_summary(all_response_lengths)
        pf_mean, pf_std = _numeric_summary(valid_parallelism_factors)
        non_ar_mean, non_ar_std = _numeric_summary(all_non_ar_tokens_per_step_values)
        inf_budget_mean, inf_budget_std = _numeric_summary(all_inf_budgets)
        zero_frac = (
            float(np.mean([1.0 if length <= 0 else 0.0 for length in all_response_lengths]))
            if all_response_lengths
            else None
        )
        if self.rank == 0:
            summary = {
                "throughput_mean": tput_mean,
                "throughput_std": tput_std,
                "throughput_all": [float(x) for x in all_tputs],
                "latency_mean": latency_mean,
                "latency_std": latency_std,
                "latency_all": [float(x) for x in all_latencies],
                "output_length_from_tput_latency_mean": out_len_mean,
                "output_length_from_tput_latency_std": out_len_std,
                "response_length_mean": resp_len_mean,
                "response_length_std": resp_len_std,
                "response_length_all": [float(x) for x in all_response_lengths],
                "zero_frac": zero_frac,
                "parallelism_factor_mean": pf_mean,
                "parallelism_factor_std": pf_std,
                "parallelism_factor_all": valid_parallelism_factors,
                "non_ar_tokens_per_step_mean": non_ar_mean,
                "non_ar_tokens_per_step_std": non_ar_std,
                "inf_budget_mean": inf_budget_mean,
                "inf_budget_std": inf_budget_std,
                "warmup_examples_per_rank": self.throughput_warmup,
                "measured_examples": len(all_tputs),
                "requested_measured_examples": self.throughput_samples,
                "local_measurement_target": local_measurement_target,
                "throughput_global_measurements": self.throughput_global_measurements,
                "world_size": self.world_size,
                "nfe": int(nfe) if nfe is not None else None,
                "confidence_threshold": self._generation_config_value(
                    "confidence_threshold"
                ),
                "confidence_based_noising": self._generation_config_value(
                    "confidence_based_noising"
                ),
                "confidence_margin_based_noising": self._generation_config_value(
                    "confidence_margin_based_noising"
                ),
                "compile_backbone": self.compile_backbone,
                "compile_supported": self.compile_supported,
                "compile_mode": self.compile_mode,
                "compile_dynamic": self.compile_dynamic,
                "setdlm_fast_inference": self._generation_config_value(
                    "setdlm_fast_inference"
                ),
                "setdlm_dynamic_active_logits": self._generation_config_value(
                    "setdlm_dynamic_active_logits"
                ),
                "setdlm_deterministic_sampler_fastpath": self._generation_config_value(
                    "setdlm_deterministic_sampler_fastpath"
                ),
                "setdlm_vectorized_repetition_penalty": self._generation_config_value(
                    "setdlm_vectorized_repetition_penalty"
                ),
                "setdlm_dynamic_tensor_attention_mask": self._generation_config_value(
                    "setdlm_dynamic_tensor_attention_mask"
                ),
                "setdlm_dynamic_full_window_fastpath": self._generation_config_value(
                    "setdlm_dynamic_full_window_fastpath"
                ),
            }
            with open(
                f"{self.generated_samples_output_path}/throughput-all.json", "w"
            ) as f:
                json.dump(summary, f, indent=2)

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ):
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return self.tokenizer.apply_chat_template(
                chat_history,
                enable_thinking=True,
                **chat_template_kwargs,
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            return self.tokenizer.apply_chat_template(
                chat_history,
                **chat_template_kwargs,
            )

    def generate_until(self, requests, **generation_kwargs):
        def _tokenize(
            e,
            prefix_text: str | None = (
                "Please reason step by step, and put your "
                + "final answer within $\\boxed{}$. "
            ),
        ):
            ctx = e["prefix"]
            if self.tokenizer.chat_template is not None:
                # Extract question part (before "Answer:" if it exists)
                if "\nAnswer:" in ctx:
                    question_part = ctx.split("\nAnswer:")[0]
                else:
                    question_part = ctx

                # Remove "Question: " prefix if present
                if "Question: " in question_part:
                    question_text = prefix_text + question_part.split("Question: ")[1]
                else:
                    question_text = question_part

                messages = [
                    {"role": "user", "content": question_text},
                ]
                ctx = self.apply_chat_template(messages)
            else:
                ctx = re.sub(
                    r"^####\s*(\d+)\s*$",
                    r"$\\boxed{\1}$" + self.tokenizer.eos_token,
                    ctx,
                    flags=re.MULTILINE,
                )
                ctx = ctx.replace("Question: ", prefix_text)
                ctx = ctx.replace("\nAnswer:", f"{self.tokenizer.eos_token}Answer:")
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        res = []
        res_for_json = []
        correct, total = 0, 0
        tputs = []
        response_lengths = []
        parallelism_factors = []
        non_ar_tokens_per_step_values = []
        inf_budgets = []
        latencies = []
        measured_parallelism_factors = []
        measured_non_ar_tokens_per_step_values = []
        measured_inf_budgets = []
        if self.throughput_global_measurements:
            local_measurement_target = int(
                np.ceil(self.throughput_samples / max(self.world_size, 1))
            )
        else:
            local_measurement_target = self.throughput_samples
        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            if self.throughput_run and len(tputs) >= local_measurement_target:
                self._write_throughput_outputs(
                    tputs=tputs,
                    latencies=latencies,
                    response_lengths=response_lengths,
                    parallelism_factors=measured_parallelism_factors,
                    non_ar_tokens_per_step_values=measured_non_ar_tokens_per_step_values,
                    inf_budgets=measured_inf_budgets,
                    local_measurement_target=local_measurement_target,
                )
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(0)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            generation_outputs = self.model.generate(
                inputs=elem["prefix"][None, ...].to(self.device),
                tokenizer=self.tokenizer,
                **self.gen_kwargs,
            )
            if isinstance(generation_outputs, ModelOutput):
                sample = generation_outputs.sequences
                parallelism_factor = generation_outputs.get("parallelism_factor", -1.0)
                if parallelism_factor is None:
                    parallelism_factor = -1.0
                non_ar_tokens_per_step = generation_outputs.get(
                    "non_ar_tokens_per_step"
                )
                inf_budget = generation_outputs.get("inf_budget")
            else:
                sample = generation_outputs
                parallelism_factor = -1.0
                non_ar_tokens_per_step = None
                inf_budget = None
            parallelism_factor_value = _as_float(parallelism_factor)
            parallelism_factors.append(
                parallelism_factor_value if parallelism_factor_value is not None else -1.0
            )
            if non_ar_tokens_per_step is not None:
                non_ar_tokens_per_step_value = _as_float(non_ar_tokens_per_step)
                if non_ar_tokens_per_step_value is not None:
                    non_ar_tokens_per_step_values.append(non_ar_tokens_per_step_value)
            if inf_budget is not None:
                inf_budget_value = _as_float(inf_budget)
                if inf_budget_value is not None:
                    inf_budgets.append(inf_budget_value)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000
            tput = (sample.numel() - elem["prefix"].numel()) / elapsed_time_s
            response_length = sample.numel() - elem["prefix"].numel()
            if i >= self.throughput_warmup:
                tputs.append(float(tput))
                latencies.append(float(elapsed_time_s))
                response_lengths.append(float(response_length))
                if parallelism_factor_value is not None:
                    measured_parallelism_factors.append(parallelism_factor_value)
                if non_ar_tokens_per_step is not None:
                    non_ar_tokens_per_step_value = _as_float(non_ar_tokens_per_step)
                    if non_ar_tokens_per_step_value is not None:
                        measured_non_ar_tokens_per_step_values.append(
                            non_ar_tokens_per_step_value
                        )
                if inf_budget is not None:
                    inf_budget_value = _as_float(inf_budget)
                    if inf_budget_value is not None:
                        measured_inf_budgets.append(inf_budget_value)
            result = self.tokenizer.decode(sample[0, len(elem["prefix"]) :])
            stop_strings = list(elem["target"]["until"]) + ["<|eot_id|>"]
            if self.stop_on_im_end:
                stop_strings.append("<|im_end|>")
            if self.tokenizer.eos_token is not None:
                stop_strings.append(self.tokenizer.eos_token)
            for until in stop_strings:
                if until:
                    result = result.split(until)[0]
            predicted_ans = None
            if "boxed{" in result:
                predicted_ans = result.split("boxed{")[1].split("}")[0]
                result = result.split("boxed{")[0] + "#### " + predicted_ans
                result = result.replace("$\\", "")
            if self.rank == 0:
                print("=" * 20)
                print("prefix: ", elem["prefix_text"], result)
                print("(Ground truth): ", requests[i].doc["answer"])
                print("=" * 20, end="\n\n")
                print(
                    f"Parallelism factor: {np.mean(parallelism_factors):0.2f} "
                    f"+/- {np.std(parallelism_factors):0.2f}"
                )
                if non_ar_tokens_per_step_values:
                    print(
                        "Fully parallel non-AR tokens per step: "
                        f"{np.mean(non_ar_tokens_per_step_values):0.2f} "
                        f"+/- {np.std(non_ar_tokens_per_step_values):0.2f}"
                    )
                if inf_budget is not None:
                    print(
                        f"Inference prediction budget: {np.mean(inf_budgets):0.2f} "
                        f"+/- {np.std(inf_budgets):0.2f}"
                    )
            res.append(result)

            # log accuracy
            ground_truth_ans = requests[i].doc["answer"].split("### ")[1]
            if predicted_ans is not None and ground_truth_ans == predicted_ans:
                correct += 1
            total += 1

            res_for_json.append(
                {
                    "prefix": elem["prefix_text"],
                    "result": result,
                }
            )
            if self.rank == 0:
                print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")
                if i >= self.throughput_warmup:
                    print(
                        f"Thput (tok/s): {np.mean(tputs):0.2f} "
                        f"+/- {np.std(tputs):0.2f}"
                    )
                    print(
                        f"Response length: {np.mean(response_lengths):0.2f} "
                        f"+/- {np.std(response_lengths):0.2f}"
                    )
                else:
                    print(f"Thput (tok/s): {tput:0.2f}")
                    print(f"Response length: {response_length:0.2f}")
        samples_path = f"{self.generated_samples_output_path}/rank{self.rank}"
        with open(f"{samples_path}.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        print(f"RANK {self.rank} completed!")
        parallelism_factors = gather_results(parallelism_factors, self.world_size)
        non_ar_tokens_per_step_values = gather_results(
            non_ar_tokens_per_step_values, self.world_size
        )
        throughputs = gather_results(tputs, self.world_size)
        if inf_budgets:
            inf_budgets = gather_results(inf_budgets, self.world_size)
        response_lengths = gather_results(response_lengths, self.world_size)
        if self.rank == 0:
            print("=" * 20)
            print("Metrics aggregated across ranks:")
            print(
                f"Parallelism factor: {np.mean(parallelism_factors):0.2f} "
                f"+/- {np.std(parallelism_factors):0.2f}"
            )
            if non_ar_tokens_per_step_values:
                print(
                    "Fully parallel non-AR tokens per step: "
                    f"{np.mean(non_ar_tokens_per_step_values):0.2f} "
                    f"+/- {np.std(non_ar_tokens_per_step_values):0.2f}"
                )
            print(
                f"Thput (tok/s): {np.mean(throughputs):0.2f} "
                f"+/- {np.std(throughputs):0.2f}"
            )
            print(
                f"Response length: {np.mean(response_lengths):0.2f} "
                f"+/- {np.std(response_lengths):0.2f}"
            )
            if inf_budgets:
                print(
                    f"Inference prediction budget: {np.mean(inf_budgets):0.2f} "
                    f"+/- {np.std(inf_budgets):0.2f}"
                )
        return res


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    ipg = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = accelerate.Accelerator(kwargs_handlers=[ipg])
    accelerator = accelerator if accelerator.num_processes > 1 else None
    set_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.task.model, accelerator=accelerator)
    results = hydra.utils.call(cfg.task, model=model)
    if results is not None and (
        accelerator is None or accelerator.local_process_index == 0
    ):
        samples = results.pop("samples")
        evaluation_tracker = EvaluationTracker(output_path=cfg.output_path)
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=samples[task_name]
            )
        print(make_table(results))
        metrics_f = f"{cfg.task.model.generated_samples_output_path}/metrics.txt"
        with open(metrics_f, "w") as f:
            f.write(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    register_useful_resolvers()
    main()
