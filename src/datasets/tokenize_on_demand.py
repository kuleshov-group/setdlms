from typing import Any, Dict, Literal

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from datasets import load_dataset


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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        qa_tokenized = self.tokenizer.batch_encode_plus(
            [example["question"], example["answer"]],
            max_length=self.max_seq_len,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, below
            truncation=True,
        )
        if self.add_special_tokens:
            qa_tokenized["input_ids"] = [
                [self.tokenizer.bos_token_id] + t + [self.tokenizer.eos_token_id]
                for t in qa_tokenized["input_ids"]
            ]
            qa_tokenized["attention_mask"] = [
                [1, 1] + t for t in qa_tokenized["attention_mask"]
            ]
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
