from typing import Any, Dict, Literal

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from datasets import load_dataset

_QUESTION_PREFIX = (
    "Please reason step by step, " + "and put your final answer within $\\boxed{}$. "
)
_SUMMARY_PREFIX = "Please summarize the following text: "
_TRANSLATION_PREFIX = "Translate the following text from {source} to {target}: "


class GSM8KDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "test"],
        max_seq_len: int,
        dataset_path: str = "openai/gsm8k",
        config_name: Literal["main", "socratic"] = "main",
        padding: bool = False,
        add_special_tokens: bool = True,
        question_prompt_text: str | None = _QUESTION_PREFIX,
        answer_prompt_text: str | None = "Answer: ",
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            dataset_path, config_name, split=split, trust_remote_code=True
        )
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.question_prompt_text = question_prompt_text
        self.answer_prompt_text = answer_prompt_text

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _postprocess_box_answer(
        answer: str, prefix: str = "$\\boxed{", suffix: str = "}$"
    ):
        """
        Post-processes the answer for the desired format.
        Args:
            answer (str): The answer string to be post-processed.
        Returns:
            str: The post-processed answer string.
        """
        answer = answer.replace("#### ", prefix) + suffix
        return answer

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.question_prompt_text is not None:
            example["question"] = self.question_prompt_text + example["question"]
        if self.answer_prompt_text is not None:
            example["answer"] = self.answer_prompt_text + example["answer"]
        example["answer"] = self._postprocess_box_answer(example["answer"])
        if self.add_special_tokens:
            example["question"] = (
                self.tokenizer.bos_token
                + example["question"]
                + self.tokenizer.eos_token
            )
            example["answer"] = example["answer"] + self.tokenizer.eos_token

        qa_tokenized = self.tokenizer.batch_encode_plus(
            [example["question"], example["answer"]],
            max_length=self.max_seq_len // 2,
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
        }


class HendrycksMathDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "test"],
        max_seq_len: int,
        dataset_path: str = "EleutherAI/hendrycks_math",
        padding: bool = False,
        add_special_tokens: bool = True,
        question_prompt_text: str | None = _QUESTION_PREFIX,
        answer_prompt_text: str | None = "Answer: ",
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_path, split=split, trust_remote_code=True)
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.question_prompt_text = question_prompt_text
        self.answer_prompt_text = answer_prompt_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.question_prompt_text is not None:
            example["problem"] = self.question_prompt_text + example["problem"]
        if self.answer_prompt_text is not None:
            example["solution"] = self.answer_prompt_text + example["solution"]
        if self.add_special_tokens:
            example["problem"] = (
                self.tokenizer.bos_token + example["problem"] + self.tokenizer.eos_token
            )
            example["solution"] = example["solution"] + self.tokenizer.eos_token

        qa_tokenized = self.tokenizer.batch_encode_plus(
            [example["problem"], example["solution"]],
            max_length=self.max_seq_len // 2,
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
        }


class CNNDailyMailDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "validation", "test"],
        max_seq_len: int,
        dataset_path: str = "abisee/cnn_dailymail",
        config_name: Literal["1.0.0", "2.0.0", "3.0.0"] = "3.0.0",
        padding: bool = False,
        add_special_tokens: bool = True,
        article_prompt_text: str | None = _SUMMARY_PREFIX,
        summary_prompt_text: str | None = "Summary: ",
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            dataset_path, config_name, split=split, trust_remote_code=True
        )
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.article_prompt_text = article_prompt_text
        self.summary_prompt_text = summary_prompt_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.article_prompt_text is not None:
            example["article"] = self.article_prompt_text + example["article"]
        if self.summary_prompt_text is not None:
            example["highlights"] = self.summary_prompt_text + example["highlights"]
        if self.add_special_tokens:
            example["article"] = (
                self.tokenizer.bos_token + example["article"] + self.tokenizer.eos_token
            )
            example["highlights"] = example["highlights"] + self.tokenizer.eos_token

        seq2seq_tokenized = self.tokenizer.batch_encode_plus(
            [example["article"], example["highlights"]],
            max_length=self.max_seq_len // 2,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, above
            truncation=True,
        )

        input_ids = torch.cat(
            [torch.LongTensor(t) for t in seq2seq_tokenized["input_ids"]], dim=-1
        )
        attention_mask = torch.cat(
            [torch.LongTensor(a) for a in seq2seq_tokenized["attention_mask"]], dim=-1
        )
        context_mask = torch.cat(
            (
                torch.LongTensor(seq2seq_tokenized["attention_mask"][0]),
                torch.zeros_like(torch.LongTensor(seq2seq_tokenized["input_ids"][1])),
            ),
            dim=-1,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
        }


class WMTDataset(Dataset):
    _LANGUAGE = {
        "cs": "Czech",
        "en": "English",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "validation", "test"],
        max_seq_len: int,
        dataset_path: str = "wmt/wmt14",
        subset: str = "de-en",
        padding: bool = False,
        add_special_tokens: bool = True,
        source_prompt_text: str | None = _TRANSLATION_PREFIX,
        target_prompt_text: str | None = "Translation: ",
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            dataset_path, subset, split=split, trust_remote_code=True
        )
        self.source = subset.split("-")[0]
        self.target = subset.split("-")[1]
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.source_prompt_text = source_prompt_text.format(
            source=self._LANGUAGE[subset.split("-")[0]],
            target=self._LANGUAGE[subset.split("-")[1]],
        )
        self.target_prompt_text = target_prompt_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        source = example["translation"][self.source]
        target = example["translation"][self.target]
        if self.source_prompt_text is not None:
            source = self.source_prompt_text + source
        if self.target_prompt_text is not None:
            target = self.target_prompt_text + target
        if self.add_special_tokens:
            source = self.tokenizer.bos_token + source + self.tokenizer.eos_token
            target = target + self.tokenizer.eos_token

        seq2seq_tokenized = self.tokenizer.batch_encode_plus(
            [source, target],
            max_length=self.max_seq_len // 2,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, above
            truncation=True,
        )

        input_ids = torch.cat(
            [torch.LongTensor(t) for t in seq2seq_tokenized["input_ids"]], dim=-1
        )
        attention_mask = torch.cat(
            [torch.LongTensor(a) for a in seq2seq_tokenized["attention_mask"]], dim=-1
        )
        context_mask = torch.cat(
            (
                torch.LongTensor(seq2seq_tokenized["attention_mask"][0]),
                torch.zeros_like(torch.LongTensor(seq2seq_tokenized["input_ids"][1])),
            ),
            dim=-1,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
        }
