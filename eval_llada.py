"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import random

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from datasets import Dataset


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(
        categorical_probs.dtype
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        max_length=4096,
        generated_samples_path="",
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.0,
        sampling_steps=512,
        mask_length=512,
        block_size=32,
        remasking="low_confidence",
        device="cuda",
        sampler="",
        remdm_number=0,
    ):
        """
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        """
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.generated_samples_path = generated_samples_path
        self.sampler = sampler
        self.remdm_number = remdm_number

        self.cfg = cfg
        self.sampling_steps = sampling_steps
        self.mask_length = mask_length
        self.block_size = block_size
        self.remasking = remasking
        print(self.generated_samples_path)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, prompt_index.sum(), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    @torch.no_grad()
    def llada_conf_sample(self, prompt):
        xt = torch.full(
            (1, prompt.shape[1] + self.mask_length), self.mask_id, dtype=torch.long
        ).to(self.model.device)
        xt[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = xt != self.mask_id
        prompt_len = prompt_index.sum(1).item()

        assert self.mask_length % self.block_size == 0
        num_blocks = self.mask_length // self.block_size

        assert self.sampling_steps % num_blocks == 0
        steps = self.sampling_steps // num_blocks

        assert self.mask_length % self.sampling_steps == 0

        for num_block in range(num_blocks):
            for i in range(steps):
                mask_index = xt == self.mask_id
                logits = self.model(xt).logits
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = _sample_categorical(p)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l

                x0_p[:, prompt_len + (num_block + 1) * self.block_size :] = -np.inf
                x0 = torch.where(mask_index, x0, xt)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=int(self.mask_length / self.sampling_steps)
                    )
                    transfer_index[j, select_index] = True
                xt[transfer_index] = x0[transfer_index]
            if torch.sum(xt == self.tokenizer.eos_token_id) > 0:
                return xt

        return xt

    @torch.no_grad()
    def llada_confremdm_sample(self, prompt):
        xt = torch.full(
            (1, prompt.shape[1] + self.mask_length), self.mask_id, dtype=torch.long
        ).to(self.model.device)
        xt[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = xt != self.mask_id
        prompt_len = prompt_index.sum(1).item()

        assert self.mask_length % self.block_size == 0
        num_blocks = self.mask_length // self.block_size

        assert self.sampling_steps % num_blocks == 0
        steps = self.sampling_steps // num_blocks

        assert self.mask_length % self.sampling_steps == 0

        for num_block in range(num_blocks):
            conf_cache = torch.ones_like(xt, dtype=torch.float64) * np.inf
            for i in range(steps):
                mask_index = xt == self.mask_id
                logits = self.model(xt).logits
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = _sample_categorical(p)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l

                x0_p[:, prompt_len + (num_block + 1) * self.block_size :] = -np.inf
                x0 = torch.where(mask_index, x0, xt)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                if i < steps - 1:
                    transfer_length = (
                        int(self.mask_length / self.sampling_steps) + 1
                    )  # why
                else:
                    transfer_length = int(self.mask_length / self.sampling_steps)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=transfer_length)
                    transfer_index[j, select_index] = True
                xt[transfer_index] = x0[transfer_index]
                conf_cache[transfer_index] = confidence[transfer_index]

                if i < steps - 1:
                    remask_index = torch.zeros_like(
                        xt, dtype=torch.bool, device=xt.device
                    )
                    _, mask_indices = torch.topk(
                        conf_cache, k=1, largest=False, dim=1
                    )  # pick lowest confidence token in each batch
                    remask_index[0, mask_indices] = True
                    conf_cache[remask_index] = np.inf
                    xt[remask_index] = self.mask_id

            if torch.sum(xt == self.tokenizer.eos_token_id) > 0:
                return xt

        return xt

    @torch.no_grad()
    def llada_randomremdmloop_sample(self, prompt):
        xt = torch.full(
            (1, prompt.shape[1] + self.mask_length), self.mask_id, dtype=torch.long
        ).to(self.model.device)
        xt[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = xt != self.mask_id
        prompt_len = prompt_index.sum(1).item()

        assert self.mask_length % self.block_size == 0
        num_blocks = self.mask_length // self.block_size

        assert self.sampling_steps % num_blocks == 0
        steps = self.sampling_steps // num_blocks

        assert self.mask_length % self.sampling_steps == 0

        for num_block in range(num_blocks):
            remask_thres = int(self.block_size / 8 * 7)
            for i in range(2 * steps):
                mask_index = xt == self.mask_id
                if i >= remask_thres and i < remask_thres + steps:
                    remask_index = torch.zeros_like(
                        xt, dtype=torch.bool, device=xt.device
                    )
                    mask_conf = torch.rand((xt.shape[0], xt.shape[1]), device=x0.device)
                    mask_conf[:, : prompt.shape[1]] = -np.inf
                    mask_conf[
                        :, prompt_len + (num_block + 1) * self.block_size :
                    ] = -np.inf
                    mask_conf = torch.where(mask_index, -np.inf, mask_conf)
                    _, mask_indices = torch.topk(mask_conf, k=self.remdm_number, dim=1)
                    remask_index[0, mask_indices] = True
                    xt[remask_index] = self.mask_id
                logits = self.model(xt).logits
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = _sample_categorical(p)
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                x0_p[:, prompt_len + (num_block + 1) * self.block_size :] = -np.inf
                x0 = torch.where(mask_index, x0, xt)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                if i >= remask_thres and i < remask_thres + steps:
                    transfer_length = self.remdm_number
                else:
                    transfer_length = int(self.mask_length / self.sampling_steps)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=transfer_length)
                    transfer_index[j, select_index] = True
                xt[transfer_index] = x0[transfer_index]
            if torch.sum(xt == self.tokenizer.eos_token_id) > 0:
                return xt

        return xt

    @torch.no_grad()
    def llada_confremdmloop_sample(self, prompt):
        xt = torch.full(
            (1, prompt.shape[1] + self.mask_length), self.mask_id, dtype=torch.long
        ).to(self.model.device)
        xt[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = xt != self.mask_id
        prompt_len = prompt_index.sum(1).item()

        assert self.mask_length % self.block_size == 0
        num_blocks = self.mask_length // self.block_size

        assert self.sampling_steps % num_blocks == 0
        steps = self.sampling_steps // num_blocks

        assert self.mask_length % self.sampling_steps == 0

        for num_block in range(num_blocks):
            conf_cache = torch.ones_like(xt, dtype=torch.float64) * np.inf
            remask_thres = int(self.block_size / 8 * 7)
            for i in range(2 * steps):
                if i >= remask_thres and i < remask_thres + steps:
                    remask_index = torch.zeros_like(
                        xt, dtype=torch.bool, device=xt.device
                    )
                    _, mask_indices = torch.topk(
                        conf_cache, k=self.remdm_number, largest=False, dim=1
                    )
                    remask_index[0, mask_indices] = True
                    conf_cache[remask_index] = np.inf
                    xt[remask_index] = self.mask_id
                mask_index = xt == self.mask_id
                logits = self.model(xt).logits
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = _sample_categorical(p)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l

                x0_p[:, prompt_len + (num_block + 1) * self.block_size :] = -np.inf
                x0 = torch.where(mask_index, x0, xt)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                if i >= remask_thres and i < remask_thres + steps:
                    transfer_length = self.remdm_number
                else:
                    transfer_length = int(self.mask_length / self.sampling_steps)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=transfer_length)
                    transfer_index[j, select_index] = True
                xt[transfer_index] = x0[transfer_index]
                conf_cache[transfer_index] = confidence[transfer_index]
            if torch.sum(xt == self.tokenizer.eos_token_id) > 0:
                return xt

        return xt

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [
            {"question": req.args[0], "until": req.args[1]["until"]} for req in requests
        ]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out, out_for_json = [], []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"] + ["<|eot_id|>", self.tokenizer.eos_token]

            if self.sampler == "llada_conf":
                generated_answer = self.llada_conf_sample(prompt)
            elif self.sampler == "llada_conf_remdm":
                generated_answer = self.llada_confremdm_sample(prompt)
            elif self.sampler == "llada_random_remdm_loop":
                generated_answer = self.llada_randomremdmloop_sample(prompt)
            elif self.sampler == "llada_conf_remdm_loop":
                generated_answer = self.llada_confremdmloop_sample(prompt)

            generated_answer = self.tokenizer.decode(
                generated_answer[0][prompt.shape[1] :], skip_special_tokens=False
            )
            # print(elem['question_text'] + generated_answer)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            # print(elem['question_text'] + generated_answer)
            out.append(generated_answer)
            out_for_json.append(
                {
                    "prefix": elem["question_text"],
                    "result": generated_answer,
                }
            )

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        with open(
            os.path.join(self.generated_samples_path, str(self._rank) + ".json"), "w"
        ) as f:
            json.dump(out_for_json, f, indent=2)

        return out


if __name__ == "__main__":
    cli_evaluate()
