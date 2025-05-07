"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import random
from typing import List, Tuple

import accelerate
import numpy as np
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from datasets import Dataset
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
)
from src.sampler import SamplerConfig


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
        device: str = "cuda",
        load_ema_weights: bool = True,
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
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        self.device = torch.device(device)
        self.tokenizer = maybe_add_missing_special_tokens(
            AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, trust_remote_code=True
            )
        )
        try:
            self.model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=model_path,
                load_ema_weights=load_ema_weights,
            ).to(self.device)
        except FileNotFoundError:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            ).to(self.device)
        self.model.eval()
        assert (
            self.model.mask_token_id is not None
            or self.tokenizer.mask_token_id is not None
        ), "Mask token id must be set in either the model or tokenizer."
        self.mask_token_id = getattr(
            self.model, "mask_token_id", self.tokenizer.mask_token_id
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

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    def generate_until(self, requests, **generation_kwargs):
        def _tokenize(
            e,
            prefix_text: str | None = (
                "Please reason step by step, and put your "
                + "final answer within $\\boxed{}$. "
            ),
        ):
            ctx = (prefix_text if prefix_text is not None else "") + e["prefix"]
            n_spaces = len(ctx) - len(ctx.rstrip())
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
        total_len = [len(x["prefix"]) + self.max_cont_length for x in ds]
        assert max(total_len) <= self.sampler_config.max_length, (
            "Input length(s) exceeds max_length"
        )

        res = []
        res_for_json = []
        for elem in tqdm(ds, desc="Generating"):
            sample, _ = self.model.generate(
                max_length=len(elem["prefix"]) + self.max_cont_length,
                context=elem["prefix"][None, ...].to(self.device),
                device=self.device,
                # tokenizer=self.tokenizer,  # For debugging
            )
            result = self.tokenizer.decode(sample[0, len(elem["prefix"]) :])
            for until in elem["target"]["until"] + [
                "<|eot_id|>",
                self.tokenizer.eos_token,
            ]:
                result = result.split(until)[0]
            print("=" * 20)
            print("prefix: ", elem["prefix_text"], result)
            print("=" * 20, end="\n\n")
            res.append(result)
            res_for_json.append(
                {
                    "prefix": elem["prefix_text"],
                    "result": result,
                }
            )
            torch.cuda.empty_cache()
        with open(self.model.config.eval.generated_samples_path, "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        return res


if __name__ == "__main__":
    cli_evaluate()
