import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch
    from omegaconf import OmegaConf
    from torch import nn

    from scripts.eval.mcqa_eval import (
        MCQAScorer,
        load_mcqa_model,
        pad_option_batch,
        resolve_max_length,
    )
except ModuleNotFoundError:
    torch = None
    nn = None
    OmegaConf = None
    load_mcqa_model = None
    pad_option_batch = None
    resolve_max_length = None


def _make_cfg(
    *,
    max_length: int | None = None,
    model_config_overrides: dict | None = None,
):
    assert OmegaConf is not None
    return OmegaConf.create(
        {
            "pretrained_model_name_or_path": "plain-backbone-name",
            "pretrained_model_revision": None,
            "max_length": max_length,
            "model_config_overrides": model_config_overrides or {},
            "task": {
                "load_ema_weights": False,
                "ckpt_file": "best-rank0.pt",
            },
        }
    )


@unittest.skipIf(
    torch is None or pad_option_batch is None,
    "MCQA eval dependencies are unavailable",
)
class PadOptionBatchTests(unittest.TestCase):
    def test_uses_longest_option_by_default(self):
        batch = pad_option_batch(
            encoded_options=[
                {
                    "input_ids": [10, 11, 12],
                    "context_mask": [1, 1, 0],
                    "answer_token_count": 1,
                    "prompt_truncated": False,
                },
                {
                    "input_ids": [20, 21],
                    "context_mask": [1, 0],
                    "answer_token_count": 1,
                    "prompt_truncated": True,
                },
            ],
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertTrue(
            torch.equal(batch["attention_mask"][1], torch.tensor([1, 1, 0]))
        )
        self.assertTrue(torch.equal(batch["context_mask"][1], torch.tensor([1, 0, 1])))

    def test_can_pad_to_fixed_model_context_length(self):
        batch = pad_option_batch(
            encoded_options=[
                {
                    "input_ids": [1, 2, 3],
                    "context_mask": [1, 1, 0],
                    "answer_token_count": 1,
                    "prompt_truncated": False,
                },
                {
                    "input_ids": [4, 5],
                    "context_mask": [1, 0],
                    "answer_token_count": 1,
                    "prompt_truncated": False,
                },
            ],
            pad_token_id=99,
            device=torch.device("cpu"),
            target_length=8,
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertTrue(
            torch.equal(
                batch["input_ids"][0],
                torch.tensor([1, 2, 3, 99, 99, 99, 99, 99]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["attention_mask"][0],
                torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["context_mask"][0],
                torch.tensor([1, 1, 0, 1, 1, 1, 1, 1]),
            )
        )

    def test_rejects_target_shorter_than_encoded_option(self):
        with self.assertRaisesRegex(ValueError, "pad target cannot be shorter"):
            pad_option_batch(
                encoded_options=[
                    {
                        "input_ids": [1, 2, 3],
                        "context_mask": [1, 1, 0],
                        "answer_token_count": 1,
                        "prompt_truncated": False,
                    }
                ],
                pad_token_id=0,
                device=torch.device("cpu"),
                target_length=2,
            )


@unittest.skipIf(
    torch is None
    or OmegaConf is None
    or load_mcqa_model is None
    or resolve_max_length is None,
    "MCQA eval dependencies are unavailable",
)
class MCQAGuardrailTests(unittest.TestCase):
    def test_plain_causal_path_is_unchanged(self):
        cfg = _make_cfg(
            model_config_overrides={"length": 128},
        )
        plain_model = SimpleNamespace(config=SimpleNamespace(length=128))

        with patch(
            "scripts.eval.mcqa_eval.load_eval_model", return_value=plain_model
        ) as load_model_mock:
            loaded_model = load_mcqa_model(
                cfg,
                tokenizer=SimpleNamespace(),
                device=torch.device("cpu"),
            )

        self.assertIs(loaded_model, plain_model)
        self.assertEqual(
            load_model_mock.call_args.kwargs["model_config_overrides"],
            {"length": 128},
        )
        self.assertEqual(
            resolve_max_length(
                cfg,
                SimpleNamespace(config=SimpleNamespace(length=None)),
                SimpleNamespace(model_max_length=512),
            ),
            512,
        )


@unittest.skipIf(
    torch is None or OmegaConf is None or MCQAScorer is None,
    "MCQA eval dependencies are unavailable",
)
class MCQACausalScoringTests(unittest.TestCase):
    def test_score_causal_batch_accepts_pre_shifted_ar_logits(self):
        class _FakeARModel:
            def __init__(self):
                self.config = SimpleNamespace(length=4)

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask):
                logits = torch.tensor(
                    [
                        [
                            [0.0, 10.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0, 0.0],
                            [0.0, 0.0, 0.0, 10.0],
                        ]
                    ],
                    dtype=torch.float32,
                )
                return SimpleNamespace(logits=logits)

        cfg = OmegaConf.create(
            {
                "max_length": 4,
                "task": {
                    "normalize_by_answer_length": True,
                    "num_importance_samples": 1,
                    "sampling_eps": 1e-3,
                    "restricted_t_range": None,
                },
            }
        )
        scorer = MCQAScorer(
            cfg=cfg,
            model=_FakeARModel(),
            tokenizer=SimpleNamespace(pad_token_id=0, model_max_length=4),
            device=torch.device("cpu"),
        )
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "context_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            "answer_token_counts": torch.tensor([2], dtype=torch.long),
            "prompt_truncated": [False],
        }

        scores = scorer._score_causal_batch(batch)

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["answer_token_count"], 2)
        self.assertGreater(scores[0]["avg_logprob"], -0.01)

    def test_denoiser_block_size_falls_back_to_eval_cfg(self):
        cfg = OmegaConf.create(
            {
                "max_length": 32,
                "block_size": 1024,
                "task": {
                    "normalize_by_answer_length": True,
                    "num_importance_samples": 1,
                    "sampling_eps": 1e-3,
                    "restricted_t_range": None,
                },
            }
        )
        model = SimpleNamespace(config=SimpleNamespace(length=32, block_size=None, eval_block_size=None))
        scorer = MCQAScorer(
            cfg=cfg,
            model=model,
            tokenizer=SimpleNamespace(pad_token_id=0, model_max_length=32),
            device=torch.device("cpu"),
        )

        self.assertEqual(scorer._denoiser_block_size(), 1024)


if __name__ == "__main__":
    unittest.main()
