"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import random
import re
from typing import List, Tuple

import accelerate
import numpy as np
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
)
from src.datasets.tokenize_on_demand import GSM8KDataset
from src.sampler import SamplerConfig


class BoxedStoppingCriteria(StoppingCriteria):
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

class LengthStoppingCriteria(StoppingCriteria):
    def __init__(self, max_length):
        self.max_length = max_length
    def __call__(
        self, input_ids: torch.LongTensor, scores: None | torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids.shape[-1] >= self.max_length:
            return True
        else:
            return False

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("lm_eval_harness_model")
class LMEvalHarness(LM):
    def __init__(
        self,
        # Model args
        max_cont_len: int = 128,
        model_path: str = "",
        tokenizer_name_or_path: str = "",
        data_split: str = "test",
        device: str = "cuda",
        load_ema_weights: bool = True,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        # Sampler args
        num_samples: int = 1,
        batch_size: int = 1,
        num_steps: int = 1000,
        min_t: float = 1e-5,
        top_p: float = 0.9,
        pad_context: bool = False,
        greedy: bool = False,
        use_x0_pred: bool = False,
        first_hitting: bool = False,
        low_confidence_remasking: bool = False,
        disable_cache: bool = False,
        kv_caching: bool = False,
        max_length: int | None = None,  # Default to model config, if None
        block_size: int | None = None,  # Default to model config, if None
        shift_logits: bool | None = None,  # Default to model config, if None
    ):
        """
        Args:
            max_cont_len (int): Max length of continuation tokens.
            model_path (str): Checkpoint path.
            tokenizer_name_or_path (str): Tokenizer name or path.
        """
        super().__init__()
        self.max_cont_length = max_cont_len
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        self.device = torch.device(device)
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self.tokenizer = maybe_add_missing_special_tokens(
            AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, trust_remote_code=True
            )
        )
        self.hf_model = False
        try:
            self.model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=model_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
            )
        except FileNotFoundError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.hf_model = True
        self.model.eval()
        self.model = self.model.to(self.device)
        self.mask_token_id = None
        self.mask_token_id = getattr(
            self.model, "mask_token_id", getattr(self.tokenizer, "mask_token_id", None)
        )
        self.sampler_config = SamplerConfig(
            num_samples=num_samples,
            batch_size=int(batch_size),
            num_steps=num_steps,
            min_t=min_t,
            top_p=top_p,
            pad_context=pad_context,
            greedy=greedy,
            use_x0_pred=use_x0_pred,
            first_hitting=first_hitting,
            low_confidence_remasking=low_confidence_remasking,
            disable_cache=disable_cache,
            kv_caching=kv_caching,
            max_length=max_length
            if max_length is not None
            else self.model.config.length,
            block_size=block_size
            if block_size is not None
            else self.model.config.block_size,
            shift_logits=shift_logits
            if shift_logits is not None
            else self.model.config.shift_logits,
        )
        self.model.sampler_config = self.sampler_config

        self.train_dataset = GSM8KDataset(
            tokenizer=self.tokenizer,
            split=data_split,
            max_seq_len=self.model.config.length,
        )

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
        del requests

        def _tokenize(
            e,
            prefix_text: str | None = (
                "Please reason step by step, and put your "
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

        boxed_stopping_criteria = BoxedStoppingCriteria(
            tokenizer=self.tokenizer, pattern=r"\\boxed\{.*?\}"
        )

        correct, total = 0, 0
        throughputs = []
        for elem in tqdm(self.train_dataset, desc="Generating"):
            context_mask = elem["context_mask"]
            context_mask[(context_mask == 0).nonzero()[:1]] = 1
            length_stopping_criteria = LengthStoppingCriteria(max_length=context_mask.sum() + self.max_cont_length)
            stopping_criteria = StoppingCriteriaList([boxed_stopping_criteria, length_stopping_criteria])

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            if not self.hf_model:
                sample, _ = self.model.generate(
                    max_length=context_mask.sum() + self.max_cont_length,
                    context=elem["input_ids"][context_mask.bool()][None, ...].to(
                        self.device
                    ),
                    device=self.device,
                    stopping_criteria=stopping_criteria,
                    # tokenizer=self.tokenizer,
                )
            else:
                sample = self.model.generate(
                    input_ids=elem["input_ids"][None, ...].to(self.device),
                    max_length=len(elem["input_ids"]) + self.max_cont_length,
                    num_return_sequences=1,
                    stopping_criteria=stopping_criteria,
                )
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000
            throughputs.append(sample.numel() / elapsed_time_s)

            result = self.tokenizer.decode(sample[0, len(elem["input_ids"]) :])
            question = elem["input_ids"][elem["context_mask"].bool()]
            ground_truth = self.tokenizer.decode(
                elem["input_ids"][~elem["context_mask"].bool()]
            )
            print("=" * 20)
            print("Question: ", question)
            print("\nAnswer: ", result)
            print("\n(Ground truth): ", ground_truth)
            print("=" * 20, end="\n\n")

            # log accuracy
            ground_truth_ans = ground_truth.split("\\boxed{")[1].split("}")[0]
            if "\\boxed{" in result:
                predicted_ans = result.split("\\boxed{")[1].split("}")[0]
                if ground_truth_ans == predicted_ans:
                    correct += 1
            total += 1

            # res_for_json.append(
            #     {
            #         "prefix": elem["prefix_text"],
            #         "result": result,
            #     }
            # )
            torch.cuda.empty_cache()
            print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")
            print(f"Throughput (tok/s): {np.mean(throughputs)} +/- {np.std(throughputs)}")
        # with open(self.model.config.eval.generated_samples_path, "w") as f:
        #     json.dump(
        #         res_for_json,
        #         f,  # type: ignore
        #         indent=2,
        #     )
        return []


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
