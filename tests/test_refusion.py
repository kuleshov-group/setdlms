from __future__ import annotations

import os
import re
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    from src.denoiser.base import DenoiserInput
    from src.denoiser.refusion import (
        IGNORE_INDEX,
        ReFusion,
        ReFusionConfig,
        ReFusionDynamicCache,
        ReFusionGenerationConfig,
    )
else:
    DenoiserInput = None
    ReFusion = None
    ReFusionConfig = None
    ReFusionDynamicCache = None
    ReFusionGenerationConfig = None
    IGNORE_INDEX = -100

try:
    from scripts.eval import model_loading
except ModuleNotFoundError:
    model_loading = None


def _make_bare_refusion(**overrides) -> ReFusion:
    assert ReFusion is not None
    assert nn is not None
    model = object.__new__(ReFusion)
    nn.Module.__init__(model)
    config = {
        "ignore_index": IGNORE_INDEX,
        "slot_size_set": [2],
        "training_eps": 0.1,
        "length": 6,
    }
    config.update(overrides)
    model.config = SimpleNamespace(**config)
    model.mask_token_id = 99
    model.eos_token_id = 98
    model.training = True
    return model


class _DummyTokenizer:
    def __init__(
        self,
        mask_token_id: int | None = 99,
        vocab_size: int = 128,
        eos_token_id: int | None = 98,
        bos_token_id: int | None = 1,
        pad_token_id: int | None = 0,
    ):
        self.bos_token = None if bos_token_id is None else f"tok_{bos_token_id}"
        self.eos_token = None if eos_token_id is None else f"tok_{eos_token_id}"
        self.pad_token = None if pad_token_id is None else f"tok_{pad_token_id}"
        self.mask_token = None if mask_token_id is None else f"tok_{mask_token_id}"
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self._vocab_size = vocab_size
        self._added_vocab = {
            token: idx
            for token, idx in (
                (self.bos_token, bos_token_id),
                (self.eos_token, eos_token_id),
                (self.pad_token, pad_token_id),
                (self.mask_token, mask_token_id),
            )
            if token is not None and idx is not None
        }

    def __len__(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)

    def add_special_tokens(self, special_tokens_dict: dict[str, str]) -> int:
        added = 0
        for token_attr, token_text in special_tokens_dict.items():
            token_id_attr = f"{token_attr}_id"
            token_id = self._added_vocab.get(token_text)
            if token_id is None:
                token_id = self._vocab_size
                self._added_vocab[token_text] = token_id
                self._vocab_size += 1
                added += 1
            setattr(self, token_attr, token_text)
            setattr(self, token_id_attr, token_id)
        return added


def _make_plain_causal_lm(
    *,
    mask_token_id: int | None = 99,
    eos_token_id: int | None = 98,
    bos_token_id: int | None = 1,
    pad_token_id: int | None = 0,
    vocab_size: int = 128,
):
    model = SimpleNamespace()
    model.config = SimpleNamespace(
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
    )
    model.generate = MagicMock()
    def _resize_token_embeddings(new_size: int):
        model.config.vocab_size = new_size
        return None
    model.resize_token_embeddings = MagicMock(side_effect=_resize_token_embeddings)
    model.to = MagicMock(side_effect=lambda device: model)
    return model


class _LegacyBackboneWithoutConfig:
    def __init__(self, *, attn_backend: str):
        self.attn_backend = attn_backend
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        return None


class _FakeLegacyAR:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.backbone = _LegacyBackboneWithoutConfig(
            attn_backend=config.backbone_config["attn_backend"]
        )
        self.noise_schedule = None

    def to(self, device):
        self.device = device
        return self


class _UpstreamRefusionLikeCausalLM:
    def __init__(
        self,
        *,
        mask_token_id: int | None = 99,
        eos_token_id: int | None = 98,
        bos_token_id: int | None = 1,
        pad_token_id: int | None = 0,
        vocab_size: int = 128,
        max_position_embeddings: int = 4096,
    ):
        self.config = SimpleNamespace(
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
        )
        self.generate = MagicMock()
        self.to = MagicMock(side_effect=lambda device: self)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        prompt_lengths=None,
        past_key_values=None,
        labels=None,
        **kwargs,
    ):
        raise AssertionError("This test double should not be executed.")


def _make_cache(batch_size: int, seq_len: int) -> ReFusionDynamicCache:
    assert ReFusionDynamicCache is not None
    cache = ReFusionDynamicCache()
    if seq_len <= 0:
        return cache
    cache.key_cache = [torch.zeros((batch_size, 1, seq_len, 1), dtype=torch.float32)]
    cache.value_cache = [torch.zeros((batch_size, 1, seq_len, 1), dtype=torch.float32)]
    return cache


def _make_output(
    batch_tokens: list[list[int]],
    *,
    cache_seq_len: int | None,
    position_scores: dict[tuple[int, int], dict[int, float]] | None = None,
    vocab_size: int = 128,
):
    assert torch is not None
    seq_len = len(batch_tokens[0])
    logits = torch.full(
        (len(batch_tokens), seq_len, vocab_size),
        -20.0,
        dtype=torch.float32,
    )
    for batch_idx, tokens in enumerate(batch_tokens):
        for position_idx, token_id in enumerate(tokens):
            logits[batch_idx, position_idx, token_id] = 20.0
    for (batch_idx, position_idx), scores in (position_scores or {}).items():
        logits[batch_idx, position_idx].fill_(-20.0)
        for token_id, score in scores.items():
            logits[batch_idx, position_idx, token_id] = score
    cache = None if cache_seq_len is None else _make_cache(len(batch_tokens), cache_seq_len)
    return SimpleNamespace(logits=logits, past_key_values=cache)


class _QueuedBackboneStep:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, input_ids, position_ids, past_key_values, use_cache):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "past_seq_len": past_key_values.get_seq_length(),
                "use_cache": use_cache,
            }
        )
        if not self.outputs:
            raise AssertionError("Unexpected extra backbone generation step.")
        return self.outputs.pop(0)


def _make_prefix_only_generation_model(slot_size: int, serial_num_blocks: int) -> ReFusion:
    model = _make_bare_refusion(length=64)
    desired_token = 7
    outputs = []
    for _ in range(serial_num_blocks):
        outputs.extend(
            [
                _make_output(
                    [[0] + [desired_token] * slot_size],
                    cache_seq_len=64,
                ),
                _make_output(
                    [[desired_token] * slot_size],
                    cache_seq_len=64,
                ),
            ]
        )
    model._backbone_generate_step = _QueuedBackboneStep(outputs)
    return model


@unittest.skipIf(torch is None, "torch is not installed in this validation environment")
class ReFusionTests(unittest.TestCase):
    def test_refusion_config_validates_slot_sizes(self):
        with self.assertRaises(ValueError):
            ReFusionConfig(slot_size_set=[])
        with self.assertRaises(ValueError):
            ReFusionGenerationConfig(slot_size=0)

    def test_prepare_inputs_matches_upstream_slot_partitioning(self):
        model = _make_bare_refusion()
        input_ids = torch.tensor([[1, 2, 10, 11, 12, 13]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.long)

        with (
            patch("src.denoiser.refusion.random.choice", return_value=2),
            patch(
                "src.denoiser.refusion.random.random",
                side_effect=[0.5, 0.2, 0.8],
            ),
            patch("src.denoiser.refusion.random.shuffle", side_effect=lambda items: None),
        ):
            denoiser_inputs = model._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
            )

        self.assertTrue(
            torch.equal(
                denoiser_inputs.xt,
                torch.tensor([[1, 2, 12, 13, 99, 99]], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                denoiser_inputs.x0,
                torch.tensor([[-100, -100, 13, -100, 10, 11]], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                denoiser_inputs.valid_tokens.bool(),
                torch.tensor([[False, False, False, False, True, True]]),
            )
        )
        self.assertTrue(
            torch.equal(
                denoiser_inputs.tokens_mask,
                torch.tensor([[0, 0, 1, 0, 1, 1]], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                denoiser_inputs.backbone_kwargs["position_ids"],
                torch.tensor([[0, 1, 4, 5, 2, 3]], dtype=torch.long),
            )
        )
        self.assertAlmostEqual(float(denoiser_inputs.t[0, 0].item()), 0.55)
        self.assertEqual(int(denoiser_inputs.alpha_t[0, 0].item()), 4)

    def test_prepare_inputs_rejects_non_prefix_context_mask(self):
        model = _make_bare_refusion()
        input_ids = torch.tensor([[1, 2, 10, 11]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            model._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
            )

    def test_prepare_inputs_requires_explicit_prompt_structure(self):
        model = _make_bare_refusion()
        input_ids = torch.tensor([[1, 2, 10, 11]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with self.assertRaisesRegex(ValueError, "explicit prompt structure"):
            model._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=None,
            )

    def test_prepare_inputs_drops_empty_answer_rows_like_upstream(self):
        model = _make_bare_refusion()
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 0, 0],
                [1, 2, 10, 11, 12, 13],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        context_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        valid_only_input_ids = input_ids[1:].clone()
        valid_only_attention_mask = attention_mask[1:].clone()
        valid_only_context_mask = context_mask[1:].clone()

        with (
            patch("src.denoiser.refusion.random.choice", return_value=2),
            patch(
                "src.denoiser.refusion.random.random",
                side_effect=[0.5, 0.2, 0.8],
            ),
            patch("src.denoiser.refusion.random.shuffle", side_effect=lambda items: None),
        ):
            mixed_batch_inputs = model._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
            )
        with (
            patch("src.denoiser.refusion.random.choice", return_value=2),
            patch(
                "src.denoiser.refusion.random.random",
                side_effect=[0.5, 0.2, 0.8],
            ),
            patch("src.denoiser.refusion.random.shuffle", side_effect=lambda items: None),
        ):
            valid_only_inputs = model._prepare_inputs(
                input_ids=valid_only_input_ids,
                attention_mask=valid_only_attention_mask,
                context_mask=valid_only_context_mask,
            )

        self.assertEqual(mixed_batch_inputs.xt.shape[0], 1)
        self.assertTrue(torch.equal(mixed_batch_inputs.xt, valid_only_inputs.xt))
        self.assertTrue(torch.equal(mixed_batch_inputs.x0, valid_only_inputs.x0))
        self.assertTrue(
            torch.equal(mixed_batch_inputs.tokens_mask, valid_only_inputs.tokens_mask)
        )
        self.assertTrue(
            torch.equal(
                mixed_batch_inputs.backbone_kwargs["position_ids"],
                valid_only_inputs.backbone_kwargs["position_ids"],
            )
        )

    def test_prepare_inputs_rejects_all_empty_answer_rows(self):
        model = _make_bare_refusion()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.ones_like(input_ids)

        with self.assertRaisesRegex(ValueError, "no answer tokens"):
            model._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
            )

    def test_compute_loss_matches_upstream_hybrid_objective(self):
        model = _make_bare_refusion()
        correct_probs = torch.tensor([0.7, 0.8, 0.6], dtype=torch.float32)
        log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.2, 0.1, 0.7],
                        [1.0, 0.0, 0.0],
                        [0.1, 0.8, 0.1],
                        [0.6, 0.2, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        )
        denoiser_inputs = DenoiserInput(
            xt=torch.tensor([[1, 2, 12, 13, 99, 99]], dtype=torch.long),
            x0=torch.tensor([[-100, -100, 2, -100, 1, 0]], dtype=torch.long),
            valid_tokens=torch.tensor([[0, 0, 0, 0, 1, 1]], dtype=torch.float32),
            tokens_mask=torch.tensor([[0, 0, 1, 0, 1, 1]], dtype=torch.float32),
            t=torch.full((1, 6), 0.5, dtype=torch.float32),
            alpha_t=torch.full((1, 6), 4.0, dtype=torch.float32),
        )

        output = model._compute_loss(log_probs, denoiser_inputs)
        expected_ar_loss = -torch.log(correct_probs[0])
        expected_mdm_loss = (
            (-torch.log(correct_probs[1]) / 0.5) / 4
            + (-torch.log(correct_probs[2]) / 0.5) / 4
        )
        expected_loss = expected_ar_loss + expected_mdm_loss

        self.assertTrue(torch.allclose(output.loss, expected_loss))
        self.assertTrue(
            torch.allclose(
                output.nlls[0, [2, 4, 5]],
                torch.tensor(
                    [
                        -torch.log(correct_probs[0]),
                        (-torch.log(correct_probs[1]) / 0.5) / 4,
                        (-torch.log(correct_probs[2]) / 0.5) / 4,
                    ]
                ),
            )
        )

    def test_compute_loss_all_masked_slots_is_mdm_only(self):
        model = _make_bare_refusion()
        log_probs = torch.log(
            torch.tensor(
                [[[0.2, 0.8], [0.75, 0.25]]],
                dtype=torch.float32,
            )
        )
        denoiser_inputs = DenoiserInput(
            xt=torch.tensor([[99, 99]], dtype=torch.long),
            x0=torch.tensor([[1, 0]], dtype=torch.long),
            valid_tokens=torch.ones((1, 2), dtype=torch.float32),
            tokens_mask=torch.ones((1, 2), dtype=torch.float32),
            t=torch.full((1, 2), 0.5, dtype=torch.float32),
            alpha_t=torch.full((1, 2), 2.0, dtype=torch.float32),
        )

        output = model._compute_loss(log_probs, denoiser_inputs)
        expected_loss = ((-torch.log(torch.tensor(0.8)) / 0.5) / 2) + (
            (-torch.log(torch.tensor(0.75)) / 0.5) / 2
        )

        self.assertTrue(torch.allclose(output.other_loss_terms["ar_loss"], torch.zeros(())))
        self.assertTrue(torch.allclose(output.loss, expected_loss))

    def test_compute_loss_all_unmasked_slots_is_ar_only(self):
        model = _make_bare_refusion()
        log_probs = torch.log(
            torch.tensor(
                [[[0.2, 0.8], [0.75, 0.25]]],
                dtype=torch.float32,
            )
        )
        denoiser_inputs = DenoiserInput(
            xt=torch.tensor([[1, 0]], dtype=torch.long),
            x0=torch.tensor([[1, 0]], dtype=torch.long),
            valid_tokens=torch.zeros((1, 2), dtype=torch.float32),
            tokens_mask=torch.ones((1, 2), dtype=torch.float32),
            t=torch.full((1, 2), 0.5, dtype=torch.float32),
            alpha_t=torch.full((1, 2), 2.0, dtype=torch.float32),
        )

        output = model._compute_loss(log_probs, denoiser_inputs)
        expected_loss = (-torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.75))) / 2

        self.assertTrue(torch.allclose(output.other_loss_terms["mdm_loss"], torch.zeros(())))
        self.assertTrue(torch.allclose(output.loss, expected_loss))

    def test_refusion_dynamic_cache_updates_and_selects_batches(self):
        cache = ReFusionDynamicCache()
        cache.key_cache = [torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2)]
        cache.value_cache = [
            torch.arange(12, 24, dtype=torch.float32).reshape(2, 1, 3, 2)
        ]

        cache.full_update(
            (
                (
                    torch.full((2, 1, 1, 2), 100.0),
                    torch.full((2, 1, 1, 2), 200.0),
                ),
            )
        )
        self.assertEqual(tuple(cache.key_cache[0].shape), (2, 1, 4, 2))
        self.assertTrue(torch.equal(cache.key_cache[0][:, :, -1, :], torch.full((2, 1, 2), 100.0)))

        cache.batch_repeat_interleave(2)
        self.assertEqual(tuple(cache.key_cache[0].shape), (4, 1, 4, 2))

        cache.batch_select_indices(torch.tensor([3, 1], dtype=torch.long))
        self.assertEqual(tuple(cache.key_cache[0].shape), (2, 1, 4, 2))

        cache.select_partial(torch.tensor([0, 2], dtype=torch.long))
        self.assertEqual(tuple(cache.key_cache[0].shape), (2, 1, 2, 2))

        cache.batch_select_minibatch(1)
        self.assertEqual(tuple(cache.key_cache[0].shape), (1, 1, 2, 2))

    def test_prepare_inputs_inference_recomputes_positions_after_cache_crop(self):
        model = _make_bare_refusion(length=4)
        cache = _make_cache(batch_size=1, seq_len=4)

        denoiser_inputs, _ = model._prepare_inputs_inference(
            input_ids=torch.tensor([[5, 6]], dtype=torch.long),
            cache={"past_key_values": cache},
        )

        self.assertTrue(
            torch.equal(
                denoiser_inputs.backbone_kwargs["position_ids"],
                torch.tensor([[2, 3]], dtype=torch.long),
            )
        )

    def test_generate_rejects_lengths_that_change_upstream_block_partition(self):
        model = _make_prefix_only_generation_model(slot_size=8, serial_num_blocks=2)

        with self.assertRaisesRegex(ValueError, "slot_size \\* serial_num_blocks"):
            model.generate(
                inputs=torch.tensor([[1]], dtype=torch.long),
                generation_config=ReFusionGenerationConfig(
                    slot_size=8,
                    serial_num_blocks=2,
                    max_new_tokens=8,
                ),
            )

    def test_generate_returns_upstream_padded_length_without_trimming(self):
        model = _make_prefix_only_generation_model(slot_size=8, serial_num_blocks=2)

        outputs = model.generate(
            inputs=torch.tensor([[1]], dtype=torch.long),
            generation_config=ReFusionGenerationConfig(
                slot_size=8,
                serial_num_blocks=2,
                max_new_tokens=10,
            ),
        )

        self.assertEqual(tuple(outputs.shape), (1, 17))
        self.assertTrue(
            torch.equal(outputs[:, 1:], torch.full((1, 16), 7, dtype=torch.long))
        )

    def test_generate_uses_temperature_even_when_do_sample_is_false(self):
        model = _make_prefix_only_generation_model(slot_size=1, serial_num_blocks=1)

        with patch.object(
            ReFusion,
            "add_gumbel_noise",
            side_effect=lambda logits, temperature: logits,
        ) as add_gumbel_noise_mock:
            model.generate(
                inputs=torch.tensor([[1]], dtype=torch.long),
                generation_config=ReFusionGenerationConfig(
                    slot_size=1,
                    serial_num_blocks=1,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.7,
                ),
            )

        self.assertEqual(add_gumbel_noise_mock.call_args.kwargs["temperature"], 0.7)

    def test_refusion_generation_config_yaml_defaults_to_deterministic_temperature(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "generation",
            "refusion_generation_config.yaml",
        )

        with open(config_path, encoding="utf-8") as handle:
            config_text = handle.read()

        match = re.search(r"^temperature:\s*([0-9.]+)\s*(?:#.*)?$", config_text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(float(match.group(1)), 0.0)

    def test_generate_stops_on_eos_in_confident_prefix(self):
        model = _make_bare_refusion(length=32)
        queued_step = _QueuedBackboneStep(
            [
                _make_output(
                    [[0, 98, 98, 98, 98]],
                    cache_seq_len=32,
                ),
                _make_output(
                    [[98, 98, 98, 98]],
                    cache_seq_len=32,
                ),
            ]
        )
        model._backbone_generate_step = queued_step

        outputs = model.generate(
            inputs=torch.tensor([[1]], dtype=torch.long),
            generation_config=ReFusionGenerationConfig(
                slot_size=4,
                serial_num_blocks=1,
                max_new_tokens=4,
            ),
        )

        self.assertTrue(torch.equal(outputs, torch.tensor([[1, 98]], dtype=torch.long)))

    def test_generate_stops_on_eos_during_speculative_verification(self):
        model = _make_bare_refusion(length=32)
        queued_step = _QueuedBackboneStep(
            [
                _make_output(
                    [[0, 7, 8, 7, 8]],
                    cache_seq_len=32,
                ),
                _make_output(
                    [[7, 7, 7, 7]],
                    cache_seq_len=32,
                    position_scores={
                        (0, 0): {7: 4.0, 8: 0.0},
                        (0, 1): {7: 4.0},
                        (0, 2): {7: 4.0, 8: 0.0},
                    },
                ),
                _make_output(
                    [[0, 98], [0, 8]],
                    cache_seq_len=None,
                ),
                _make_output(
                    [[7, 98], [7, 8]],
                    cache_seq_len=32,
                    position_scores={
                        (0, 0): {7: 2.0, 98: 2.0},
                        (1, 0): {7: 2.0, 8: 2.0},
                    },
                ),
            ]
        )
        model._backbone_generate_step = queued_step

        outputs = model.generate(
            inputs=torch.tensor([[1]], dtype=torch.long),
            generation_config=ReFusionGenerationConfig(
                slot_size=2,
                serial_num_blocks=1,
                max_new_tokens=4,
                token_threshold=0.4,
            ),
        )

        self.assertTrue(torch.equal(outputs, torch.tensor([[1, 7, 98]], dtype=torch.long)))
        self.assertEqual(len(queued_step.outputs), 0)

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_wraps_explicit_refusion_requests_without_path_heuristic(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        raw_hf_model = _make_plain_causal_lm(mask_token_id=17)
        wrapped_refusion = _make_bare_refusion(length=32)
        wrapped_refusion.config.mask_token_id = 17
        wrapped_refusion.mask_token_id = 17

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(raw_hf_model, "causal_lm"),
            ),
            patch.object(
                model_loading,
                "_load_legacy_denoiser",
                return_value=wrapped_refusion,
            ) as load_legacy_mock,
        ):
            loaded_model = model_loading.load_eval_model(
                pretrained_model_name_or_path="plain-backbone-name",
                tokenizer=tokenizer,
                device="cpu",
                model_config_overrides={"model_type": "refusion", "length": 32},
            )

        self.assertIs(loaded_model, wrapped_refusion)
        self.assertIsInstance(loaded_model, ReFusion)
        self.assertIs(loaded_model.generate.__func__, ReFusion.generate)
        self.assertIsNot(loaded_model.generate.__func__, raw_hf_model.generate)
        self.assertIs(load_legacy_mock.call_args.kwargs["model"], raw_hf_model)
        self.assertEqual(
            load_legacy_mock.call_args.kwargs["requested_model_type"],
            ReFusionConfig.model_type,
        )

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_rejects_missing_refusion_mask_token(self):
        tokenizer = _DummyTokenizer(mask_token_id=None)
        raw_hf_model = _make_plain_causal_lm(mask_token_id=17)

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(raw_hf_model, "causal_lm"),
            ),
        ):
            loaded_model = model_loading.load_eval_model(
                pretrained_model_name_or_path="plain-backbone-name",
                tokenizer=tokenizer,
                device="cpu",
                model_config_overrides={"model_type": "refusion", "length": 32},
            )

        self.assertIsInstance(loaded_model, ReFusion)
        self.assertEqual(tokenizer.mask_token, "<|mask|>")
        self.assertIsNotNone(tokenizer.mask_token_id)

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_legacy_denoiser_applies_backbone_overrides_without_hf_config(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)

        with (
            patch.object(model_loading, "AR", _FakeLegacyAR),
            patch(
                "scripts.eval.model_loading.torch.load",
                return_value={"state_dict": {}},
            ),
        ):
            loaded_model = model_loading._load_legacy_denoiser(
                pretrained_model_name_or_path="plain-ar-checkpoint",
                tokenizer=tokenizer,
                device="cpu",
                model_config_overrides={
                    "backbone_config": {"attn_backend": "sdpa"},
                },
            )

        self.assertEqual(loaded_model.config.backbone_config["attn_backend"], "sdpa")
        self.assertEqual(loaded_model.backbone.attn_backend, "sdpa")
        self.assertEqual(loaded_model.backbone.loaded_state_dict, {})

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_configure_refusion_special_tokens_resizes_raw_backbone_for_added_mask(self):
        tokenizer = _DummyTokenizer(mask_token_id=None, vocab_size=128)
        raw_hf_model = _make_plain_causal_lm(mask_token_id=None, vocab_size=128)

        model_loading._configure_refusion_special_token_ids(raw_hf_model, tokenizer)

        self.assertEqual(tokenizer.mask_token, "<|mask|>")
        self.assertEqual(raw_hf_model.config.mask_token_id, tokenizer.mask_token_id)
        raw_hf_model.resize_token_embeddings.assert_called_once_with(len(tokenizer))

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_does_not_auto_wrap_plain_causal_lm_with_mask_token(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        raw_hf_model = _make_plain_causal_lm(mask_token_id=17)

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(raw_hf_model, "causal_lm"),
            ),
            patch.object(model_loading, "_load_legacy_denoiser") as load_legacy_mock,
        ):
            loaded_model = model_loading.load_eval_model(
                pretrained_model_name_or_path="official-upstream-checkpoint",
                tokenizer=tokenizer,
                device="cpu",
            )

        self.assertIs(loaded_model, raw_hf_model)
        load_legacy_mock.assert_not_called()
        raw_hf_model.to.assert_called_once_with("cpu")

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_auto_wraps_upstream_refusion_like_causal_lm(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        upstream_like_model = _UpstreamRefusionLikeCausalLM(
            mask_token_id=17,
            max_position_embeddings=8192,
        )
        wrapped_refusion = _make_bare_refusion(length=8192)
        wrapped_refusion.config.mask_token_id = 17
        wrapped_refusion.mask_token_id = 17

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(upstream_like_model, "causal_lm"),
            ),
            patch.object(
                model_loading,
                "_load_legacy_denoiser",
                return_value=wrapped_refusion,
            ) as load_legacy_mock,
        ):
            loaded_model = model_loading.load_eval_model(
                pretrained_model_name_or_path="official-upstream-checkpoint",
                tokenizer=tokenizer,
                device="cpu",
            )

        self.assertIs(loaded_model, wrapped_refusion)
        self.assertEqual(load_legacy_mock.call_args.kwargs["requested_model_type"], "refusion")
        self.assertEqual(load_legacy_mock.call_args.kwargs["model_config_overrides"]["length"], 8192)
        self.assertIs(load_legacy_mock.call_args.kwargs["model"], upstream_like_model)

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_accepts_refusion_checkpoint_without_path_heuristic(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        wrapped_refusion = _make_bare_refusion(length=32)
        wrapped_refusion.config.mask_token_id = 17
        wrapped_refusion.mask_token_id = 17

        with (
            patch.object(model_loading, "fsspec_exists", return_value=True),
            patch.object(
                model_loading,
                "load_model_from_ckpt_dir_path",
                return_value=wrapped_refusion,
            ),
        ):
            loaded_model = model_loading.load_eval_model(
                pretrained_model_name_or_path="/tmp/plain-checkpoint",
                tokenizer=tokenizer,
                device="cpu",
            )

        self.assertIs(loaded_model, wrapped_refusion)
        self.assertIsInstance(loaded_model, ReFusion)

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_rejects_non_refusion_checkpoint_for_explicit_request(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)

        with (
            patch.object(model_loading, "fsspec_exists", return_value=True),
            patch.object(
                model_loading,
                "load_model_from_ckpt_dir_path",
                return_value=SimpleNamespace(generate=MagicMock()),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "Explicit ReFusion request could not be satisfied"
            ):
                model_loading.load_eval_model(
                    pretrained_model_name_or_path="/tmp/plain-checkpoint",
                    tokenizer=tokenizer,
                    device="cpu",
                    model_config_overrides={"model_type": "refusion"},
                )

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_rejects_silent_refusion_fallback_failure(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        raw_hf_model = _make_plain_causal_lm(mask_token_id=17)

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(raw_hf_model, "causal_lm"),
            ),
            patch.object(
                model_loading,
                "_load_legacy_denoiser",
                return_value=SimpleNamespace(generate=MagicMock()),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "Explicit ReFusion request could not be satisfied"
            ):
                model_loading.load_eval_model(
                    pretrained_model_name_or_path="plain-backbone-name",
                    tokenizer=tokenizer,
                    device="cpu",
                    model_config_overrides={"model_type": "refusion", "length": 32},
                )

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_rejects_refusion_rebuild_without_explicit_length(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        raw_hf_model = _make_plain_causal_lm(mask_token_id=17)

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(raw_hf_model, "causal_lm"),
            ),
            patch.object(model_loading, "_load_legacy_denoiser") as load_legacy_mock,
        ):
            with self.assertRaisesRegex(
                ValueError, "implicit default length"
            ):
                model_loading.load_eval_model(
                    pretrained_model_name_or_path="plain-backbone-name",
                    tokenizer=tokenizer,
                    device="cpu",
                    model_config_overrides={"model_type": "refusion"},
                )

        load_legacy_mock.assert_not_called()

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_rejects_masked_lm_backbones_for_explicit_refusion(self):
        tokenizer = _DummyTokenizer(mask_token_id=17)
        masked_lm = _make_plain_causal_lm(mask_token_id=17)

        with (
            patch.object(model_loading, "fsspec_exists", return_value=False),
            patch.object(
                model_loading,
                "_load_hf_model",
                return_value=(masked_lm, "masked_lm"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "causal LM backbone"):
                model_loading.load_eval_model(
                    pretrained_model_name_or_path="masked-lm-checkpoint",
                    tokenizer=tokenizer,
                    device="cpu",
                    model_config_overrides={"model_type": "refusion", "length": 32},
                )

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_load_eval_model_rejects_refusion_special_token_mismatch(self):
        tokenizer = _DummyTokenizer(mask_token_id=17, eos_token_id=42)
        wrapped_refusion = _make_bare_refusion(length=32)
        wrapped_refusion.config.mask_token_id = 17
        wrapped_refusion.mask_token_id = 17
        wrapped_refusion.config.eos_token_id = 99
        wrapped_refusion.eos_token_id = 99

        with (
            patch.object(model_loading, "fsspec_exists", return_value=True),
            patch.object(
                model_loading,
                "load_model_from_ckpt_dir_path",
                return_value=wrapped_refusion,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "eos_token_id"):
                model_loading.load_eval_model(
                    pretrained_model_name_or_path="/tmp/refusion-checkpoint",
                    tokenizer=tokenizer,
                    device="cpu",
                )

    @unittest.skipIf(
        model_loading is None,
        "model_loading dependencies are unavailable in this environment",
    )
    def test_loader_wires_eos_from_tokenizer_for_generation_without_caller_eos(self):
        tokenizer = _DummyTokenizer(mask_token_id=17, eos_token_id=42)
        model = _make_bare_refusion(length=32)
        model.mask_token_id = None
        model.eos_token_id = None
        model.bos_token_id = None
        model.pad_token_id = None
        model.config.mask_token_id = None
        model.config.eos_token_id = None
        model.config.bos_token_id = None
        model.config.pad_token_id = None
        model_loading._configure_refusion_special_token_ids(model, tokenizer)
        model._backbone_generate_step = _QueuedBackboneStep(
            [
                _make_output(
                    [[0, 42, 42, 42, 42]],
                    cache_seq_len=32,
                ),
                _make_output(
                    [[42, 42, 42, 42]],
                    cache_seq_len=32,
                ),
            ]
        )

        outputs = model.generate(
            inputs=torch.tensor([[1]], dtype=torch.long),
            generation_config=ReFusionGenerationConfig(
                slot_size=4,
                serial_num_blocks=1,
                max_new_tokens=4,
                eos_token_id=None,
            ),
        )

        self.assertEqual(model.config.eos_token_id, 42)
        self.assertEqual(model.eos_token_id, 42)
        self.assertTrue(torch.equal(outputs, torch.tensor([[1, 42]], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
