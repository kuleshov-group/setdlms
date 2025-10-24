"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import re
import sys
from typing import Any, List, Tuple

import accelerate
import hydra
import numpy as np
import torch
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
)

from datasets import Dataset
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs


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

        model_config_overrides = (
            {} if model_config_overrides is None else model_config_overrides
        )
        if fsspec_exists(os.path.join(pretrained_model_name_or_path, "config.yaml")):
            model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=pretrained_model_name_or_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
                device=self.device,
                **model_config_overrides,
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
            except:  # Model not compatible with CausalLM
                model = AutoModelForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.gen_kwargs = gen_kwargs
        self.throughput_run = throughput_run
        self.throughput_warmup = throughput_warmup
        self.throughput_samples = throughput_samples

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

    def generate_until(self, requests, **generation_kwargs):
        # TODO: Move this to utils file / perhaps use chat template
        def _tokenize(
            e,
            prefix_text: str | None = (
                f"{self.tokenizer.bos_token}Please reason step by step, and put your "
                + "final answer within $\\boxed{}$. "
            ),
        ):
            ctx = e["prefix"]
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
        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            if (
                self.throughput_run
                and i >= self.throughput_samples + self.throughput_warmup
            ):
                tputs_path = (
                    f"{self.generated_samples_output_path}/throughput-rank{self.rank}"
                )
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
                sys.exit(0)
            if self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_event, end_event = None, None
            sample = self.model.generate(
                inputs=elem["prefix"][None, ...].to(self.device),
                disable_pbar=(self.rank != 0),
                # tokenizer=self.tokenizer,  # Uncomment for debugging
                **self.gen_kwargs,
            )
            if self.rank == 0:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_s = start_event.elapsed_time(end_event) / 1000
                tput = (sample.numel() - elem["prefix"].numel()) / elapsed_time_s
                if i >= self.throughput_warmup:
                    tputs.append(tput)
            result = self.tokenizer.decode(sample[0, len(elem["prefix"]) :])
            for until in elem["target"]["until"] + [
                "<|eot_id|>",
                self.tokenizer.eos_token,
            ]:
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
            # torch.cuda.empty_cache()
            if self.rank == 0:
                print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")
                if i >= self.throughput_warmup:
                    print(
                        f"Thput (tok/s): {np.mean(tputs):0.2f} +/- {np.std(tputs):0.2f}"
                    )
                else:
                    print(f"Thput (tok/s): {tput:0.2f}")
        samples_path = f"{self.generated_samples_output_path}/rank{self.rank}"
        with open(f"{samples_path}.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        print(f"RANK {self.rank} completed!")
        return res


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    accelerator = accelerate.Accelerator()
    accelerator = accelerate.Accelerator() if accelerator.num_processes > 1 else None
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
