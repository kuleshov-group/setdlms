"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import re
from typing import Any, List, Tuple

import accelerate
import hydra
import torch
from lm_eval.api.model import LM
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    StoppingCriteria,
)

from datasets import Dataset
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    print_and_save_config,
    register_useful_resolvers,
    set_seed,
)


class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, pattern):
        self.tokenizer = tokenizer
        self.pattern = pattern

    def __call__(
        self, input_ids: torch.LongTensor, scores: None | torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids.numel() == 0:
            return False
        matches = re.findall(self.pattern, self.tokenizer.decode(input_ids[0]))
        if len(matches) > 1:
            return True
        return False


class LMEvalHarnessModel(LM):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        generated_samples_output_path: str,
        tokenizer: PreTrainedTokenizer,
        load_ema_weights: bool = False,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        gen_kwargs: Any | None = None,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Path to ckpt dir or HF model repo.
            generated_samples_output_path (str): Path to generated samples dir.
            tokenizer (str): Tokenizer name or path.
            load_ema_weights (bool): Whether to load ema weights (for local ckpts).
            ckpt_file (str): Name of ckpt file (for local ckpts).
            gen_kwargs (dict): Generator kwargs.
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
        """
        super().__init__()
        self.generated_samples_output_path = (
            f"{generated_samples_output_path}/lm_eval_harness_output"
        )
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
            device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rank = 0
            self._world_size = 1
        self.device = torch.device(f"{device}")

        try:
            model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=pretrained_model_name_or_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
            )
        except FileNotFoundError:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
            )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.gen_kwargs = gen_kwargs

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
        def _tokenize(
            e,
            prefix_text: str | None = (
                "<|im_end|>Please reason step by step, and put your "
                + "final answer within $\\boxed{}$. "
            ),
        ):
            ctx = (prefix_text if prefix_text is not None else "") + e["prefix"]
            # TODO: Hacks to make data look like training set
            ctx = ctx.replace("Question: ", "")
            ctx = ctx.replace("\nAnswer:", "<|im_end|>Answer:")
            n_spaces = len(ctx) - len(ctx)
            if n_spaces > 0:
                ctx = ctx[:-n_spaces]
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
        # total_len = [len(x["prefix"]) + max_new_tokens for x in ds]
        # assert max(total_len) <= self.model.max_length, (
        #     "Input length(s) exceeds max_length"
        # )
        res = []
        res_for_json = []
        correct, total = 0, 0
        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            sample = self.model.generate(
                inputs=elem["prefix"][None, ...].to(self.device),
                disable_pbar=(self.rank != 0),
                # tokenizer=self.tokenizer,
                **self.gen_kwargs,
            )
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
            torch.cuda.empty_cache()
            if self.rank == 0:
                print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")

        if not os.path.exists(self.generated_samples_output_path):
            os.mkdir(self.generated_samples_output_path)
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
    print_and_save_config(cfg, resolve=True, save_cfg=False)
    set_seed(cfg.seed)
    # TODO: Do something with results:
    #  - compute overall accuracy
    #  - pretty print results table
    #  - save to json
    #  -.
    # results = (
    hydra.utils.call(cfg.task)


if __name__ == "__main__":
    register_useful_resolvers()
    main()
