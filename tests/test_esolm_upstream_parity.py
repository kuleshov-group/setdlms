from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.backbone.automodel import AutoModelFromPreTrained
from src.backbone.dit import DIT, EmbeddingLayer
from src.denoiser.base import DenoiserInput
from src.denoiser.diffusion import EsoLM, SetDiffusionGenerationConfig
from src.noise_schedule.noise_schedules import EsoLogLinearNoise


def _make_bare_esolm(**overrides) -> EsoLM:
    model = object.__new__(EsoLM)
    nn.Module.__init__(model)
    config = {
        "alpha_0": 1.0,
        "length": 4,
        "diffusion_shuffle": False,
        "diffusion_attn_mode": "causal",
        "sequential_shuffle": False,
        "sequential_attn_mode": "causal",
        "loss_type": "elbo",
        "num_iw_orders": 0,
        "batch_split": 1.0,
    }
    config.update(overrides)
    model.config = SimpleNamespace(**config)
    model.mask_token_id = 7
    model.neg_infinity = -1e6
    model.alpha_0 = float(model.config.alpha_0)
    model.num_tokens = int(model.config.length)
    model.noise_schedule = EsoLogLinearNoise(alpha_0=model.alpha_0)
    return model


def _reference_sort_indices(
    indices: torch.LongTensor,
    mask_token_id: int,
    shuffle: bool,
    keep_masks_unshuffled: bool = False,
) -> torch.LongTensor:
    masked = indices == mask_token_id
    if shuffle:
        offsets = torch.rand(indices.shape, device=indices.device) * 0.9
        if keep_masks_unshuffled:
            offsets = offsets.clone()
            offsets[masked] = torch.linspace(
                0, 1, int(masked.sum().item()), device=indices.device
            )
    else:
        offsets = torch.linspace(0, 0.9, indices.shape[1], device=indices.device)[
            None, :
        ]
        offsets = offsets.expand_as(indices)
    return (masked.to(offsets.dtype) + offsets).argsort(dim=-1, descending=False)


def _reference_tokens_unmasked_per_step(
    alpha_0: float, num_tokens: int, num_steps: int, eps: float = 1e-3
) -> list[int]:
    remaining_tokens = num_tokens
    num_tokens_to_unmask = []
    dt = 1 / num_steps
    for t in torch.linspace(1.0, dt, steps=num_steps).tolist():
        alpha_t = alpha_0 * (1 - (1 - eps) * t)
        alpha_s = alpha_0 * (1 - (1 - eps) * (t - dt))
        n_unmask = int(
            torch.binomial(
                torch.tensor(float(remaining_tokens)),
                torch.tensor(float((alpha_s - alpha_t) / (1 - alpha_t))),
            ).item()
        )
        if n_unmask != 0:
            num_tokens_to_unmask.append(n_unmask)
            remaining_tokens -= n_unmask
    if remaining_tokens != 0 and alpha_0 == 1:
        num_tokens_to_unmask.append(remaining_tokens)
    return num_tokens_to_unmask


class DummyRotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, position_ids):
        batch_size, seq_len = position_ids.shape
        cos = torch.ones(batch_size, seq_len, 3, 1, self.dim)
        sin = torch.zeros_like(cos)
        return cos, sin


class DummySigmaMap(nn.Module):
    def forward(self, sigma):
        return sigma[:, None]


class RecordingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, object]] = []
        self.reset_calls = 0

    def reset_kv_cache(self):
        self.reset_calls += 1

    def forward(
        self,
        x,
        rotary_cos_sin,
        c=None,
        mask=None,
        kv_cache=False,
        num_clean=None,
        num_clean_and_mask=None,
        **kwargs,
    ):
        del c, kwargs
        self.calls.append(
            {
                "seq_len": x.shape[1],
                "mask_shape": None if mask is None else tuple(mask.shape),
                "kv_cache": kv_cache,
                "num_clean": num_clean,
                "num_clean_and_mask": num_clean_and_mask,
                "rotary_shape": tuple(rotary_cos_sin[0].shape),
            }
        )
        return x


class DummyOutputLayer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, c):
        del c
        batch_size, seq_len, _ = x.shape
        logits = torch.arange(seq_len * self.vocab_size, dtype=torch.float32)
        logits = logits.view(1, seq_len, self.vocab_size)
        return logits.expand(batch_size, -1, -1).clone()


class DummyCache:
    def __init__(self):
        self.length = 0
        self.crop_calls: list[int] = []

    def crop(self, keep_length: int):
        self.crop_calls.append(keep_length)
        self.length = keep_length


class DummyHFModel(nn.Module):
    def __init__(self, vocab_size: int = 11):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        del kwargs
        cache = past_key_values if past_key_values is not None else DummyCache()
        if use_cache:
            cache.length += input_ids.shape[1]
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            -100.0,
            dtype=torch.float32,
        )
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1])[None, :].expand_as(input_ids)
        logits.scatter_(-1, (position_ids % self.vocab_size).unsqueeze(-1), 0.0)
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None
                if attention_mask is None
                else attention_mask.clone(),
                "position_ids": None if position_ids is None else position_ids.clone(),
                "use_cache": use_cache,
                "cache_length": cache.length,
            }
        )
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)


def _make_recording_dit(vocab_size: int = 11) -> tuple[DIT, RecordingBlock]:
    backbone = object.__new__(DIT)
    nn.Module.__init__(backbone)
    block = RecordingBlock()
    backbone.causal = False
    backbone.adaLN = True
    backbone.vocab_embed = EmbeddingLayer(dim=4, vocab_dim=vocab_size)
    backbone.sigma_map = DummySigmaMap()
    backbone.rotary_emb = DummyRotary(dim=4)
    backbone.blocks = nn.ModuleList([block])
    backbone.output_layer = DummyOutputLayer(vocab_size=vocab_size)
    backbone.rotary_cos_sin_sorted = None
    return backbone, block


class EsoLMUpstreamParityTests(unittest.TestCase):
    def test_prepare_sequential_inputs_uses_zero_sigma_and_backbone_packing(self):
        model = _make_bare_esolm(alpha_0=0.5, length=4)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.zeros_like(input_ids)

        torch.manual_seed(0)
        denoiser_inputs = model._prepare_sequential_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
        )

        self.assertTrue(torch.equal(denoiser_inputs.xt.shape, input_ids.shape))
        self.assertTrue(torch.equal(denoiser_inputs.x0.shape, input_ids.shape))
        self.assertTrue(torch.all(denoiser_inputs.backbone_kwargs["sigma"] == 0))
        self.assertTrue(torch.equal(denoiser_inputs.attention_mask, attention_mask))

    def test_prepare_diffusion_inputs_keeps_alpha_per_example_after_sorting(self):
        model = _make_bare_esolm(alpha_0=0.8, length=4, diffusion_shuffle=False)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.zeros_like(input_ids)
        t = torch.tensor([0.25, 0.75])

        torch.manual_seed(0)
        denoiser_inputs = model._prepare_diffusion_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            t=t,
        )

        expected_alpha_t, expected_alpha_t_prime = model.noise_schedule(t)
        expected_alpha_t = expected_alpha_t[:, None]
        expected_alpha_t_prime = expected_alpha_t_prime[:, None]

        self.assertEqual(denoiser_inputs.alpha_t.shape, (2, 1))
        self.assertEqual(denoiser_inputs.alpha_t_prime.shape, (2, 1))
        self.assertTrue(torch.allclose(denoiser_inputs.alpha_t, expected_alpha_t))
        self.assertTrue(
            torch.allclose(denoiser_inputs.alpha_t_prime, expected_alpha_t_prime)
        )
        self.assertTrue(
            torch.equal(
                denoiser_inputs.x0,
                torch.gather(
                    input_ids,
                    dim=-1,
                    index=denoiser_inputs.backbone_kwargs["sort_index"],
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                denoiser_inputs.alpha_t.expand_as(input_ids),
                expected_alpha_t.expand_as(input_ids),
            )
        )
        self.assertTrue(
            torch.allclose(
                denoiser_inputs.backbone_kwargs["sigma"],
                model._sigma_from_alpha_t(expected_alpha_t.squeeze(-1)),
            )
        )

    def test_sort_indices_match_upstream_formula(self):
        model = _make_bare_esolm()
        indices = torch.tensor([[3, model.mask_token_id, 1, model.mask_token_id]])

        torch.manual_seed(123)
        observed = model._sort_indices(
            indices, shuffle=True, keep_masks_unshuffled=True
        )
        torch.manual_seed(123)
        expected = _reference_sort_indices(
            indices=indices,
            mask_token_id=model.mask_token_id,
            shuffle=True,
            keep_masks_unshuffled=True,
        )

        self.assertTrue(torch.equal(observed, expected))

    def test_diffusion_and_sequential_masks_match_upstream_relations(self):
        model = _make_bare_esolm()
        attention_mask = torch.ones(1, 4, dtype=torch.long)
        diffusion_mask = model._build_diffusion_attention_mask(
            attention_mask=attention_mask,
            cutoffs=torch.tensor([2]),
        )
        diffusion_mask = diffusion_mask[0, 0] == 0
        expected_diffusion = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(diffusion_mask, expected_diffusion))

        prefix_mask = model._sequential_prefix_mask(
            seq_len=3,
            cutoffs=torch.tensor([1]),
            device=torch.device("cpu"),
        )[0]
        expected_prefix = torch.tensor(
            [
                [True, False, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [True, False, False, True, False, False],
                [True, True, False, True, True, False],
                [True, True, True, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(prefix_mask, expected_prefix))

    def test_hf_backbone_kwargs_use_dense_masks_and_sorted_positions(self):
        model = _make_bare_esolm(
            alpha_0=0.8,
            length=4,
            diffusion_shuffle=False,
            sequential_shuffle=False,
        )

        diffusion_inputs = DenoiserInput(
            xt=torch.tensor([[1, model.mask_token_id, 3, model.mask_token_id]]),
            x0=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            backbone_kwargs={
                "sort_index": torch.tensor([[2, 0, 3, 1]]),
                "diffusion_attn_mode": "causal",
                "mask_token_id": model.mask_token_id,
                "sigma": torch.zeros(1),
            },
        )
        diffusion_kwargs, diffusion_output_length = model._build_hf_esolm_backbone_kwargs(
            denoiser_inputs=diffusion_inputs,
            merged_kwargs=dict(diffusion_inputs.backbone_kwargs),
        )
        expected_diffusion_mask = model._build_diffusion_attention_mask(
            attention_mask=diffusion_inputs.attention_mask,
            cutoffs=torch.sum(diffusion_inputs.xt != model.mask_token_id, dim=1),
            attn_mode="causal",
        )
        self.assertIsNone(diffusion_output_length)
        self.assertTrue(
            torch.equal(diffusion_kwargs["position_ids"], diffusion_inputs.backbone_kwargs["sort_index"])
        )
        self.assertTrue(torch.equal(diffusion_kwargs["attention_mask"], expected_diffusion_mask))
        self.assertNotIn("sigma", diffusion_kwargs)

        sequential_inputs = DenoiserInput(
            xt=torch.tensor([[1, model.mask_token_id, 3, 4, 1, 2, 3, 4]]),
            x0=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            backbone_kwargs={
                "sort_index": torch.tensor([[2, 0, 3, 1]]),
                "sequential_input": True,
                "sequential_attn_mode": "causal",
                "mask_token_id": model.mask_token_id,
                "sigma": torch.zeros(1),
            },
        )
        sequential_kwargs, sequential_output_length = model._build_hf_esolm_backbone_kwargs(
            denoiser_inputs=sequential_inputs,
            merged_kwargs=dict(sequential_inputs.backbone_kwargs),
        )
        expected_sequential_mask = model._build_sequential_attention_mask(
            attention_mask=sequential_inputs.attention_mask,
            xt=sequential_inputs.xt,
            mask_token_id=model.mask_token_id,
            attn_mode="causal",
        )
        self.assertEqual(sequential_output_length, 4)
        self.assertTrue(
            torch.equal(
                sequential_kwargs["position_ids"],
                torch.cat(
                    [
                        sequential_inputs.backbone_kwargs["sort_index"],
                        sequential_inputs.backbone_kwargs["sort_index"],
                    ],
                    dim=1,
                ),
            )
        )
        self.assertTrue(
            torch.equal(sequential_kwargs["attention_mask"], expected_sequential_mask)
        )
        self.assertNotIn("sigma", sequential_kwargs)

    def test_per_token_loss_weights_match_upstream(self):
        model = _make_bare_esolm(loss_type="elbo")
        x0 = torch.tensor([[1, 2, 3]])
        diffusion_inputs = DenoiserInput(
            xt=torch.zeros_like(x0),
            x0=x0,
            tokens_mask=torch.tensor([[1.0, 0.0, 1.0]]),
            valid_tokens=torch.tensor([[1.0, 1.0, 1.0]]),
            alpha_t=torch.tensor([[0.4, 0.2, 0.6]]),
            alpha_t_prime=torch.tensor([[-0.5, -0.5, -0.5]]),
        )
        diffusion_log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.1, 0.7, 0.1, 0.1, 0.0],
                        [0.1, 0.2, 0.6, 0.1, 0.0],
                        [0.1, 0.1, 0.1, 0.7, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
        )

        model.training = True
        model.config.loss_type = "low_var"
        low_var_loss, low_var_nlls = model._compute_diffusion_loss(
            diffusion_log_probs, diffusion_inputs
        )
        expected_low_var = torch.tensor(
            [[-diffusion_log_probs[0, 0, 1], 0.0, -diffusion_log_probs[0, 2, 3]]]
        )
        self.assertTrue(torch.allclose(low_var_nlls, expected_low_var))
        self.assertTrue(
            torch.allclose(low_var_loss, expected_low_var.sum() / diffusion_inputs.valid_tokens.sum())
        )

        model.config.loss_type = "elbo"
        elbo_loss, elbo_nlls = model._compute_diffusion_loss(
            diffusion_log_probs, diffusion_inputs
        )
        coeff = -(diffusion_inputs.alpha_t_prime / (1 - diffusion_inputs.alpha_t))
        expected_elbo = expected_low_var * coeff
        self.assertTrue(torch.allclose(elbo_nlls, expected_elbo))
        self.assertTrue(
            torch.allclose(elbo_loss, expected_elbo.sum() / diffusion_inputs.valid_tokens.sum())
        )

        sequential_inputs = DenoiserInput(
            xt=torch.zeros_like(x0),
            x0=x0,
            tokens_mask=torch.tensor([[0.0, 1.0, 1.0]]),
            valid_tokens=torch.tensor([[1.0, 1.0, 1.0]]),
        )
        sequential_log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.1, 0.7, 0.1, 0.1, 0.0],
                        [0.1, 0.1, 0.7, 0.1, 0.0],
                        [0.1, 0.1, 0.1, 0.7, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
        )
        sequential_loss, sequential_nlls = model._compute_sequential_loss(
            sequential_log_probs, sequential_inputs
        )
        expected_sequential = torch.tensor(
            [[0.0, -sequential_log_probs[0, 1, 2], -sequential_log_probs[0, 2, 3]]]
        )
        self.assertTrue(torch.allclose(sequential_nlls, expected_sequential))
        self.assertTrue(
            torch.allclose(
                sequential_loss,
                expected_sequential.sum() / sequential_inputs.valid_tokens.sum(),
            )
        )

    def test_tokens_unmasked_per_step_match_upstream_formula(self):
        model = _make_bare_esolm(alpha_0=1.0, length=8)
        torch.manual_seed(7)
        observed = model._tokens_unmasked_per_step(num_steps=4)
        torch.manual_seed(7)
        expected = _reference_tokens_unmasked_per_step(
            alpha_0=model.alpha_0,
            num_tokens=model.num_tokens,
            num_steps=4,
        )
        self.assertEqual(observed, expected)

    def test_dit_forward_builds_sequential_features_inside_backbone(self):
        backbone, block = _make_recording_dit()
        zt = torch.tensor([[7, 7, 1, 7]])
        x0 = torch.tensor([[4, 5, 6, 8]])
        sort_idx = torch.tensor([[2, 0, 3, 1]])

        output = DIT.forward(
            backbone,
            input_ids=zt,
            attention_mask=torch.ones_like(zt),
            sigma=torch.zeros(1),
            sort_index=sort_idx,
            x0=x0,
            mask_token_id=7,
        )

        self.assertEqual(output.logits.shape, (1, 4, 11))
        self.assertEqual(block.calls[0]["seq_len"], 8)
        self.assertEqual(block.calls[0]["mask_shape"], (1, 1, 8, 8))

    def test_dit_forward_sample_threads_upstream_cache_metadata(self):
        backbone, block = _make_recording_dit()
        zt = torch.tensor([[9, 8, 7, 6]])
        sort_idx = torch.tensor([[2, 0, 3, 1]])

        output = DIT.forward_sample(
            backbone,
            zt=zt,
            sort_idx=sort_idx,
            last_k_start=1,
            curr_k_start=3,
            curr_k_end=4,
        )

        self.assertEqual(output.logits.shape, (1, 1, 11))
        self.assertTrue(block.calls[0]["kv_cache"])
        self.assertEqual(block.calls[0]["num_clean"], 2)
        self.assertEqual(block.calls[0]["num_clean_and_mask"], 3)
        self.assertEqual(block.calls[0]["rotary_shape"], (1, 3, 3, 1, 4))

    def test_generate_samples_preserves_upstream_ordering_and_resets_internal_caches(self):
        class DummyNoise(EsoLogLinearNoise):
            def compute_first_hitting_times(self, *args, **kwargs):
                raise AssertionError(
                    "EsoLM sampling should not call first_hitting_times"
                )

        class DummyBackbone(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = vocab_size
                self.reset_kv_cache_calls = 0
                self.reset_sorted_rotary_cache_calls = 0

            def reset_kv_cache(self):
                self.reset_kv_cache_calls += 1

            def reset_sorted_rotary_cache(self):
                self.reset_sorted_rotary_cache_calls += 1

            def forward_sample(
                self,
                zt,
                sort_idx,
                last_k_start=None,
                curr_k_start=None,
                curr_k_end=None,
            ):
                del zt, last_k_start
                num_samples = sort_idx.shape[0]
                k = curr_k_end - curr_k_start
                logits = torch.full(
                    (num_samples, k, self.vocab_size), -100.0, dtype=torch.float32
                )
                next_tokens = sort_idx[:, curr_k_start:curr_k_end] % self.vocab_size
                logits.scatter_(-1, next_tokens.unsqueeze(-1), 0.0)
                return CausalLMOutputWithPast(logits=logits, past_key_values=None)

        model = _make_bare_esolm(alpha_0=1.0, length=4)
        model.noise_schedule = DummyNoise(alpha_0=1.0)
        model.backbone = DummyBackbone(vocab_size=8)

        generation_config = SetDiffusionGenerationConfig(
            num_steps=4,
            block_size=4,
            use_cache=True,
            do_sample=False,
        )
        samples, nfe, _ = model.generate_samples(
            num_samples=1,
            generation_config=generation_config,
        )

        self.assertTrue(torch.equal(samples, torch.tensor([[0, 1, 2, 3]])))
        self.assertEqual(nfe, 4.0)
        self.assertEqual(model.backbone.reset_kv_cache_calls, 2)
        self.assertEqual(model.backbone.reset_sorted_rotary_cache_calls, 2)

    def test_generate_samples_rejects_generic_backbone_fallback(self):
        class GenericBackbone(nn.Module):
            def forward(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("Should not hit the generic backbone forward path.")

        model = _make_bare_esolm(alpha_0=1.0, length=4)
        model.backbone = GenericBackbone()

        generation_config = SetDiffusionGenerationConfig(
            num_steps=4,
            block_size=4,
            use_cache=True,
            do_sample=False,
        )
        with self.assertRaises(NotImplementedError):
            model.generate_samples(num_samples=1, generation_config=generation_config)

    def test_automodel_native_esolm_forward_builds_sorted_positions_and_dense_masks(self):
        backbone = object.__new__(AutoModelFromPreTrained)
        nn.Module.__init__(backbone)
        backbone.use_causal_mask = False
        backbone.is_esolm_backbone = True
        backbone._esolm_past_key_values = None
        backbone.model = DummyHFModel(vocab_size=16)

        input_ids = torch.tensor([[7, 7, 1, 7, 4, 5, 6, 8]])
        attention_mask = torch.ones_like(input_ids)
        sort_idx = torch.tensor([[2, 0, 3, 1]])

        output = backbone.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sort_index=sort_idx,
            sequential_input=True,
            sequential_attn_mode="causal",
            mask_token_id=7,
            sigma=torch.zeros(1),
        )

        self.assertEqual(output.logits.shape, (1, 4, 16))
        call = backbone.model.calls[0]
        self.assertEqual(tuple(call["position_ids"].shape), (1, 8))
        self.assertTrue(
            torch.equal(
                call["position_ids"],
                torch.cat([sort_idx, sort_idx], dim=1),
            )
        )
        self.assertEqual(tuple(call["attention_mask"].shape), (1, 1, 8, 8))

    def test_automodel_forward_sample_replays_last_clean_block_before_sampling(self):
        backbone = object.__new__(AutoModelFromPreTrained)
        nn.Module.__init__(backbone)
        backbone.use_causal_mask = False
        backbone.is_esolm_backbone = True
        backbone._esolm_past_key_values = None
        backbone.model = DummyHFModel(vocab_size=16)

        zt = torch.tensor([[9, 9, 9, 9]])
        sort_idx = torch.tensor([[2, 0, 3, 1]])

        step1 = backbone.forward_sample(
            zt=zt,
            sort_idx=sort_idx,
            last_k_start=0,
            curr_k_start=0,
            curr_k_end=1,
        )
        self.assertEqual(step1.logits.shape, (1, 1, 16))

        step2 = backbone.forward_sample(
            zt=zt,
            sort_idx=sort_idx,
            last_k_start=0,
            curr_k_start=1,
            curr_k_end=3,
        )
        self.assertEqual(step2.logits.shape, (1, 2, 16))
        self.assertEqual(tuple(backbone.model.calls[1]["input_ids"].shape), (1, 3))

        step3 = backbone.forward_sample(
            zt=zt,
            sort_idx=sort_idx,
            last_k_start=1,
            curr_k_start=3,
            curr_k_end=4,
        )
        self.assertEqual(step3.logits.shape, (1, 1, 16))
        self.assertEqual(tuple(backbone.model.calls[2]["input_ids"].shape), (1, 3))
        self.assertEqual(backbone._esolm_past_key_values.crop_calls[-1], 1)

    def test_forward_supports_num_iw_orders(self):
        model = _make_bare_esolm(alpha_0=0.5, length=4)
        orders = deque(
            [
                torch.tensor([-4.0, -8.0]),
                torch.tensor([-2.0, -6.0]),
            ]
        )
        model._any_order_ar_loss = lambda x0: orders.popleft()  # type: ignore[method-assign]

        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        output = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            context_mask=torch.zeros_like(input_ids),
            num_iw_orders=2,
        )

        expected_logp = torch.logsumexp(
            torch.stack([torch.tensor([-4.0, -8.0]), torch.tensor([-2.0, -6.0])], dim=1),
            dim=1,
        ) - torch.log(torch.tensor(2.0))
        expected_nll_per_seq = -expected_logp
        expected_loss = expected_nll_per_seq.sum() / (input_ids.shape[0] * input_ids.shape[1])

        self.assertTrue(torch.allclose(output.loss, expected_loss))
        self.assertTrue(
            torch.allclose(
                output.nlls,
                (expected_nll_per_seq / input_ids.shape[1])[:, None].expand_as(input_ids.float()),
            )
        )

    def test_forward_rejects_empty_branch_from_batch_split(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.zeros_like(input_ids)

        for batch_split, missing_branch in ((0.0, "diffusion"), (1.0, "sequential")):
            with self.subTest(batch_split=batch_split, missing_branch=missing_branch):
                model = _make_bare_esolm(
                    alpha_0=0.5,
                    batch_split=batch_split,
                    length=4,
                )
                with self.assertRaisesRegex(ValueError, missing_branch):
                    model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        context_mask=context_mask,
                    )


if __name__ == "__main__":
    unittest.main()
