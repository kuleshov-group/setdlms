import random
import re
from typing import Any, Dict, Literal

import torch
from torch.utils.data.dataset import Dataset
from src.datasets.preprocessed_dataset import load_preprocessed_dataset
from transformers import PreTrainedTokenizer
from src.utils import fsspec_exists
import csv

from datasets import load_dataset
from scripts.utils import maybe_add_missing_special_tokens

_QUESTION_PREFIX = (
    "Please reason step by step, and put your final answer within $\\boxed{}$. "
)
_SUMMARY_PREFIX = "Please summarize the following text: "
_TRANSLATION_PREFIX = "Translate the following text from {source} to {target}: "


class GSM8KDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "test"],
        max_length: int,
        dataset_path: str = "openai/gsm8k",
        config_name: Literal["main", "socratic"] = "main",
        padding: bool = False,
        add_special_tokens: bool = True,
        source_prompt_text: str | None = _QUESTION_PREFIX,
        target_prompt_text: str | None = "Answer: ",
        source_key: str = "question",
        target_key: str = "answer",
        num_shot: int = 0,
        use_chat_template: bool = False,
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.dataset = load_dataset(
            dataset_path, config_name, split=split, trust_remote_code=True
        )
        self.max_length = max_length
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.source_prompt_text = source_prompt_text
        self.target_prompt_text = target_prompt_text
        self.source_key = source_key
        self.target_key = target_key
        self.num_shot = num_shot
        self.use_chat_template = use_chat_template
        self._arange = range(len(self.dataset))

        # Check if tokenizer has chat template
        if self.use_chat_template and not hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            raise ValueError(
                "use_chat_template=True but tokenizer does not support chat templates"
            )

    def __len__(self):
        return len(self.dataset)

    def _few_shot_idxs(self, exclude: int):
        candidates = [x for x in self._arange if x != exclude]
        if self.split == "train":
            return random.sample(candidates, self.num_shot)
        return [(exclude + n) % len(self.dataset) for n in range(self.num_shot)]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.use_chat_template:
            # Use chat template format
            messages = []

            # Add few-shot examples if needed
            if self.num_shot > 0:
                example_shots = [self.dataset[fsi] for fsi in self._few_shot_idxs(idx)]
                for shot in example_shots:
                    user_content = (
                        self.source_prompt_text if self.source_prompt_text else ""
                    ) + shot[self.source_key]  # type: ignore
                    assistant_content = (
                        self.target_prompt_text if self.target_prompt_text else ""
                    ) + re.sub(  # type: ignore
                        r"^####\s*(\d+)\s*$",
                        r"$\\boxed{\1}$",
                        shot[self.target_key],
                        flags=re.MULTILINE,
                    )
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": assistant_content})

            # Add current example
            user_content = (
                self.source_prompt_text if self.source_prompt_text else ""
            ) + example[self.source_key]  # type: ignore
            messages.append({"role": "user", "content": user_content})
            # Format source with chat template (up to user message, with generation prompt)
            source = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assistant_content = (
                self.target_prompt_text if self.target_prompt_text else ""
            ) + re.sub(  # type: ignore
                r"^####\s*(\d+)\s*$",
                r"$\\boxed{\1}$",
                example[self.target_key],
                flags=re.MULTILINE,
            )
            # Get the full formatted conversation including assistant response
            full_messages = messages + [
                {"role": "assistant", "content": assistant_content}
            ]
            full_formatted = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            # Extract just the assistant's response part
            # The target is everything after the source in the full formatted string
            if full_formatted.startswith(source):
                target = full_formatted[len(source) :]
            else:
                # Fallback: if templates don't align, just use assistant content
                # Some templates might add extra tokens, so we tokenize to get the difference
                source_tokens = self.tokenizer.encode(source, add_special_tokens=False)
                full_tokens = self.tokenizer.encode(
                    full_formatted, add_special_tokens=False
                )
                if len(full_tokens) > len(source_tokens):
                    target_tokens = full_tokens[len(source_tokens) :]
                    target = self.tokenizer.decode(
                        target_tokens, skip_special_tokens=False
                    )
                else:
                    target = assistant_content
        else:
            sp = (self.tokenizer.bos_token if self.add_special_tokens else "") + (
                self.source_prompt_text if self.source_prompt_text is not None else ""
            )
            tp = self.target_prompt_text if self.target_prompt_text is not None else ""
            if self.num_shot > 0:
                example_shots = [self.dataset[fsi] for fsi in self._few_shot_idxs(idx)]
                source = "\n".join(
                    [
                        sp
                        + i[self.source_key]  # type: ignore
                        + (self.tokenizer.eos_token if self.add_special_tokens else "")
                        + tp
                        + re.sub(  # type: ignore
                            r"^####\s*(\d+)\s*$",
                            r"$\\boxed{\1}$",
                            i[self.target_key],
                            flags=re.MULTILINE,
                        )
                        + (self.tokenizer.eos_token if self.add_special_tokens else "")
                        for i in example_shots
                    ]
                )
            else:
                source = ""
            source = (
                source
                + sp
                + example[self.source_key]  # type: ignore
                + (self.tokenizer.eos_token if self.add_special_tokens else "")
            )
            target = (
                tp
                + re.sub(  # type: ignore
                    r"^####\s*(\d+)\s*$",
                    r"$\\boxed{\1}$",
                    example[self.target_key],
                    flags=re.MULTILINE,
                )
                + (self.tokenizer.eos_token if self.add_special_tokens else "")
            )

        qa_tokenized = self.tokenizer.batch_encode_plus(
            [source, target],
            max_length=self.max_length // 2,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, above
            truncation=True,
        )

        input_ids = torch.cat(
            [torch.LongTensor(t) for t in qa_tokenized["input_ids"]], dim=-1
        )
        attention_mask = torch.cat(
            [torch.LongTensor(a) for a in qa_tokenized["attention_mask"]], dim=-1
        )
        context_mask = torch.cat(
            (
                torch.LongTensor(qa_tokenized["attention_mask"][0]),
                torch.zeros_like(torch.LongTensor(qa_tokenized["input_ids"][1])),
            ),
            dim=-1,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
            "index": idx,
        }

class CNNDailyMailDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "validation", "test"],
        max_source_length: int,
        max_target_length: int,
        dataset_path: str = "abisee/cnn_dailymail",
        config_name: Literal["1.0.0", "2.0.0", "3.0.0"] = "3.0.0",
        padding: bool = False,
        add_special_tokens: bool = True,
        source_prompt_text: str | None = None,
        target_prompt_text: str | None = "Summary: ",
        source_key: str = "article",
        target_key: str = "highlights",
        separate_input_output: bool = False,
        truncate: bool = True,
        use_chat_template: bool = False,
        cache_path: str = "",
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.cache_path = cache_path
        if self.cache_path and fsspec_exists(self.cache_path):
            self.dataset = load_preprocessed_dataset(self.cache_path)
        else:
            self.dataset = load_dataset(
                dataset_path, config_name, split=split, trust_remote_code=True
            )
            
            def _to_text(x):
                if isinstance(x, list):
                    return "\n".join(map(str, x))
                return x

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.source_prompt_text = source_prompt_text
        self.target_prompt_text = target_prompt_text
        self.separate_input_output = separate_input_output
        self.source_key = source_key
        self.target_key = target_key
        self.truncate = truncate
        self.use_chat_template = use_chat_template

        # Check if tokenizer has chat template
        if self.use_chat_template and not hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            raise ValueError(
                "use_chat_template=True but tokenizer does not support chat templates"
            )

    @property
    def target_references(self) -> list[str]:
        """Helper method to retrieve list of ground truth labels for downstream eval."""
        return self.dataset[self.target_key]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        if self.use_chat_template:
            messages = [
                {
                    "role": "user",
                    "content": (self.source_prompt_text or "")
                    + example[self.source_key],
                }
            ]
            source = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assistant_content = (self.target_prompt_text or "") + example[
                self.target_key
            ]
            full_messages = messages + [
                {"role": "assistant", "content": assistant_content}
            ]
            full_formatted = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if full_formatted.startswith(source):
                target = full_formatted[len(source) :]
            else:
                # Fallback: tokenize to get difference
                source_tokens = self.tokenizer.encode(source, add_special_tokens=False)
                full_tokens = self.tokenizer.encode(
                    full_formatted, add_special_tokens=False
                )
                if len(full_tokens) > len(source_tokens):
                    target_tokens = full_tokens[len(source_tokens) :]
                    target = self.tokenizer.decode(
                        target_tokens, skip_special_tokens=False
                    )
                else:
                    target = assistant_content
        else:
            source = example[self.source_key]
            target = example[self.target_key]
            if self.source_prompt_text is not None:
                source = self.source_prompt_text + source  # type: ignore
            if self.target_prompt_text is not None:
                target = self.target_prompt_text + target  # type: ignore
            if self.add_special_tokens:
                source = self.tokenizer.bos_token + source + self.tokenizer.eos_token
                target = target + self.tokenizer.eos_token
        source_ids = self.tokenizer(source, add_special_tokens=self.add_special_tokens, max_length=self.max_source_length, truncation=self.truncate)
        target_ids = self.tokenizer(target, add_special_tokens=self.add_special_tokens, max_length=self.max_target_length, truncation=self.truncate)
        # inject eos token if it was cut off
        if (source_ids["input_ids"][-1] != self.tokenizer.eos_token_id):
            source_ids["input_ids"][-1] = self.tokenizer.eos_token_id
        if (target_ids["input_ids"][-1] != self.tokenizer.eos_token_id):
            target_ids["input_ids"][-1] = self.tokenizer.eos_token_id
        if self.separate_input_output:
            input_ids = torch.LongTensor(source_ids["input_ids"])
            attention_mask = torch.LongTensor(source_ids["attention_mask"])
            context_mask = torch.LongTensor(source_ids["attention_mask"])
            output_ids = torch.LongTensor(target_ids["input_ids"])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "context_mask": context_mask,
                "output_ids": output_ids,
            }
        else:
            input_ids = torch.cat(
                [torch.LongTensor(source_ids["input_ids"]), torch.LongTensor(target_ids["input_ids"])], dim=-1
            )
            attention_mask = torch.cat(
                [torch.LongTensor(source_ids["attention_mask"]), torch.LongTensor(target_ids["attention_mask"])],
                dim=-1,
            )
            context_mask = torch.cat(
                (
                    torch.LongTensor(source_ids["attention_mask"]),
                    torch.zeros_like(
                        torch.LongTensor(target_ids["input_ids"])
                    ),
                ),
                dim=-1,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "context_mask": context_mask,
            }


class ROCStoriesDataset(Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "validation", "test"],
        max_length: int,
        dataset_path: str = "eval_datasets/cloze_test_val__spring2016.csv",
        padding: bool = False,
        num_target_sentences: int = 1,
        max_samples: int | None = None,
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.tokenizer = tokenizer
        self.num_target_sentences = num_target_sentences
        assert self.num_target_sentences in [1, 3], "Only 1 or 3 target sentences are supported"
        self.max_length = max_length
        self.padding = padding
        self.dataset = []
        with open(dataset_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
                self.dataset.append(sents)
                if max_samples is not None and len(self.dataset) >= max_samples:
                    break

    @property
    def target_references(self) -> list[str]:
        """Helper method to retrieve list of ground truth labels for downstream eval."""
        if self.num_target_sentences == 1:
            return [story[2] for story in self.dataset]
        elif self.num_target_sentences == 3:
            return [story[1] + " " + story[2] + " " + story[3] for story in self.dataset]
        else:
            raise ValueError(f"Invalid number of target sentences: {self.num_target_sentences}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        story = self.dataset[idx]
        if self.num_target_sentences == 1:
            prompt = story[0] + " " + story[1]
            suffix = " " + story[3] + " " + story[4]
            middle = story[2]
        else:
            prompt = story[0]
            suffix = " " + story[4]
            middle = story[1] + " " + story[2] + " " + story[3]

        prefix_ids = self.tokenizer(prompt).input_ids
        suffix_ids = self.tokenizer(suffix).input_ids
        middle_ids = self.tokenizer(middle).input_ids

        input_ids = [self.tokenizer.bos_token_id] +\
            prefix_ids + [self.tokenizer.mask_token_id] * len(middle_ids) +\
            suffix_ids + [self.tokenizer.eos_token_id]
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.ones_like(input_ids)

        if self.padding:
            input_ids = torch.nn.functional.pad(input_ids, (0, self.max_length - len(input_ids)), value=self.tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, self.max_length - len(attention_mask)), value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }