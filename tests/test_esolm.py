from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.backbone.dit import DDiTBlock, DIT, EmbeddingLayer
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


class EsoLMRegressionTests(unittest.TestCase):
    def test_loglinear_matches_upstream_formula(self):
        schedule = EsoLogLinearNoise(alpha_0=1.0)
        t = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32)
        alpha_t, dalpha_t = schedule(t)

        expected_alpha = 1.0 - (1 - schedule.eps) * t
        expected_dalpha = torch.full_like(t, -(1 - schedule.eps))

        self.assertTrue(torch.allclose(alpha_t, expected_alpha))
        self.assertTrue(torch.allclose(dalpha_t, expected_dalpha))

    def test_esolm_loglinear_first_hitting_times_fail_fast(self):
        schedule = EsoLogLinearNoise(alpha_0=1.0)
        with self.assertRaises(NotImplementedError):
            schedule.compute_first_hitting_times(1, 4, torch.device("cpu"))

    def test_sort_indices_matches_upstream_formula(self):
        model = _make_bare_esolm()
        indices = torch.tensor([[3, model.mask_token_id, 1, model.mask_token_id]])

        torch.manual_seed(123)
        observed = model._sort_indices(
            indices, shuffle=True, keep_masks_unshuffled=True
        )
        torch.manual_seed(123)
        expected = _reference_sort_indices(
            indices,
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

    def test_diffusion_loss_matches_low_var_and_elbo_branches(self):
        model = _make_bare_esolm(loss_type="elbo")
        x0 = torch.tensor([[1, 2]])
        tokens_mask = torch.tensor([[1.0, 0.0]])
        valid_tokens = torch.tensor([[1.0, 1.0]])
        alpha_t = torch.tensor([[0.4, 0.2]])
        alpha_t_prime = torch.tensor([[-0.5, -0.5]])
        log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.1, 0.7, 0.1, 0.1],
                        [0.1, 0.2, 0.6, 0.1],
                    ]
                ],
                dtype=torch.float32,
            )
        )
        inputs = DenoiserInput(
            xt=torch.zeros_like(x0),
            x0=x0,
            tokens_mask=tokens_mask,
            valid_tokens=valid_tokens,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
        )

        model.training = True
        model.config.loss_type = "low_var"
        low_var_loss, low_var_nlls = model._compute_diffusion_loss(log_probs, inputs)
        expected_low_var = -log_probs[0, 0, 1]
        self.assertTrue(
            torch.allclose(low_var_nlls, torch.tensor([[expected_low_var, 0.0]]))
        )
        self.assertTrue(torch.allclose(low_var_loss, expected_low_var / valid_tokens.sum()))

        model.config.loss_type = "elbo"
        elbo_loss, elbo_nlls = model._compute_diffusion_loss(log_probs, inputs)
        coeff = -(alpha_t_prime[0, 0] / (1 - alpha_t[0, 0]))
        expected_elbo = expected_low_var * coeff
        self.assertTrue(
            torch.allclose(elbo_nlls, torch.tensor([[expected_elbo, 0.0]]))
        )
        self.assertTrue(torch.allclose(elbo_loss, expected_elbo / valid_tokens.sum()))

    def test_sequential_reconstruction_loss_masks_only_masked_tokens(self):
        model = _make_bare_esolm()
        x0 = torch.tensor([[1, 2, 3]])
        tokens_mask = torch.tensor([[0.0, 1.0, 1.0]])
        valid_tokens = torch.tensor([[1.0, 1.0, 1.0]])
        log_probs = torch.log(
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
        inputs = DenoiserInput(
            xt=torch.zeros_like(x0),
            x0=x0,
            tokens_mask=tokens_mask,
            valid_tokens=valid_tokens,
        )

        loss, nlls = model._compute_sequential_loss(log_probs, inputs)
        expected = torch.tensor(
            [[0.0, -log_probs[0, 1, 2], -log_probs[0, 2, 3]]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(nlls, expected))
        self.assertTrue(torch.allclose(loss, expected.sum() / valid_tokens.sum()))

    def test_sampling_token_count_schedule_matches_upstream_formula(self):
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

    def test_generate_samples_uses_sorted_order_and_skips_first_hitting(self):
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
                self.reset_calls = 0

            def reset_kv_cache(self):
                self.reset_kv_cache_calls += 1

            def reset_sorted_rotary_cache(self):
                self.reset_calls += 1

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
        self.assertEqual(model.backbone.reset_calls, 2)

    def test_generate_supports_prompt_conditioned_sampling(self):
        class DummyBackbone(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = vocab_size
                self.reset_kv_cache_calls = 0
                self.reset_sorted_rotary_cache_calls = 0
                self.calls: list[dict[str, int]] = []

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
                del zt
                assert last_k_start is not None
                assert curr_k_start is not None
                assert curr_k_end is not None
                self.calls.append(
                    {
                        "last_k_start": last_k_start,
                        "curr_k_start": curr_k_start,
                        "curr_k_end": curr_k_end,
                    }
                )
                num_samples = sort_idx.shape[0]
                k = curr_k_end - curr_k_start
                logits = torch.full(
                    (num_samples, k, self.vocab_size), -100.0, dtype=torch.float32
                )
                if k > 0:
                    next_tokens = sort_idx[:, curr_k_start:curr_k_end] % self.vocab_size
                    logits.scatter_(-1, next_tokens.unsqueeze(-1), 0.0)
                return CausalLMOutputWithPast(logits=logits, past_key_values=None)

        model = _make_bare_esolm(alpha_0=1.0, length=6)
        model.backbone = DummyBackbone(vocab_size=16)
        prompt = torch.tensor([[10, 11, 12]])
        generation_config = SetDiffusionGenerationConfig(
            num_steps=2,
            block_size=6,
            use_cache=True,
            do_sample=False,
            subcontext_len=2,
        )

        outputs = model.generate(
            inputs=prompt,
            generation_config=generation_config,
            max_new_tokens=2,
        )

        self.assertTrue(torch.equal(outputs, torch.tensor([[10, 11, 12, 3, 4]])))
        self.assertEqual(model.backbone.reset_kv_cache_calls, 2)
        self.assertEqual(model.backbone.reset_sorted_rotary_cache_calls, 2)
        self.assertEqual(
            model.backbone.calls,
            [
                {"last_k_start": 0, "curr_k_start": 3, "curr_k_end": 3},
                {"last_k_start": 3, "curr_k_start": 3, "curr_k_end": 4},
                {"last_k_start": 3, "curr_k_start": 4, "curr_k_end": 5},
            ],
        )

    def test_generate_allows_prompt_plus_new_tokens_to_exceed_config_length(self):
        class DummyBackbone(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = vocab_size

            def reset_kv_cache(self):
                return None

            def reset_sorted_rotary_cache(self):
                return None

            def forward_sample(
                self,
                zt,
                sort_idx,
                last_k_start=None,
                curr_k_start=None,
                curr_k_end=None,
            ):
                del zt, last_k_start
                assert curr_k_start is not None
                assert curr_k_end is not None
                num_samples = sort_idx.shape[0]
                k = curr_k_end - curr_k_start
                logits = torch.full(
                    (num_samples, k, self.vocab_size), -100.0, dtype=torch.float32
                )
                if k > 0:
                    next_tokens = sort_idx[:, curr_k_start:curr_k_end] % self.vocab_size
                    logits.scatter_(-1, next_tokens.unsqueeze(-1), 0.0)
                return CausalLMOutputWithPast(logits=logits, past_key_values=None)

        model = _make_bare_esolm(alpha_0=1.0, length=4)
        model.backbone = DummyBackbone(vocab_size=16)

        outputs = model.generate(
            inputs=torch.tensor([[10, 11, 12]]),
            generation_config=SetDiffusionGenerationConfig(
                num_steps=4,
                block_size=4,
                use_cache=True,
                do_sample=False,
            ),
            max_new_tokens=4,
        )

        self.assertTrue(torch.equal(outputs, torch.tensor([[10, 11, 12, 3, 4, 5, 6]])))

    def test_generate_applies_stopping_criteria_after_prompt_sampling(self):
        class DummyBackbone(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = vocab_size

            def reset_kv_cache(self):
                return None

            def reset_sorted_rotary_cache(self):
                return None

            def forward_sample(
                self,
                zt,
                sort_idx,
                last_k_start=None,
                curr_k_start=None,
                curr_k_end=None,
            ):
                del zt, last_k_start
                assert curr_k_start is not None
                assert curr_k_end is not None
                num_samples = sort_idx.shape[0]
                k = curr_k_end - curr_k_start
                logits = torch.full(
                    (num_samples, k, self.vocab_size), -100.0, dtype=torch.float32
                )
                if k > 0:
                    next_tokens = sort_idx[:, curr_k_start:curr_k_end] % self.vocab_size
                    logits.scatter_(-1, next_tokens.unsqueeze(-1), 0.0)
                return CausalLMOutputWithPast(logits=logits, past_key_values=None)

        class LastTokenStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                del scores, kwargs
                return input_ids[:, -1] == 3

        model = _make_bare_esolm(alpha_0=1.0, length=6)
        model.backbone = DummyBackbone(vocab_size=16)
        prompt = torch.tensor([[10, 11, 12]])
        generation_config = SetDiffusionGenerationConfig(
            num_steps=2,
            block_size=6,
            use_cache=True,
            do_sample=False,
            subcontext_len=2,
        )

        outputs = model.generate(
            inputs=prompt,
            generation_config=generation_config,
            max_new_tokens=2,
            stopping_criteria=StoppingCriteriaList([LastTokenStop()]),
        )

        self.assertTrue(torch.equal(outputs, torch.tensor([[10, 11, 12, 3]])))

    def test_prepare_diffusion_inputs_keeps_context_prefix_fixed(self):
        model = _make_bare_esolm(alpha_0=1.0, length=6, diffusion_shuffle=True)
        input_ids = torch.tensor([[10, 11, 12, 20, 21, 22]])
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

        torch.manual_seed(0)
        denoiser_inputs = model._prepare_diffusion_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            t=torch.tensor([1.0]),
        )

        self.assertTrue(torch.equal(denoiser_inputs.x0[0, :3], input_ids[0, :3]))
        self.assertTrue(torch.equal(denoiser_inputs.context_mask[0, :3], context_mask[0, :3]))

    def test_prepare_sequential_inputs_keeps_context_prefix_fixed_when_shuffling(self):
        model = _make_bare_esolm(alpha_0=0.5, length=6, sequential_shuffle=True)
        input_ids = torch.tensor([[10, 11, 12, 20, 21, 22]])
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

        torch.manual_seed(0)
        denoiser_inputs = model._prepare_sequential_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
        )

        self.assertTrue(torch.equal(denoiser_inputs.x0[0, :3], input_ids[0, :3]))
        self.assertTrue(torch.equal(denoiser_inputs.context_mask[0, :3], context_mask[0, :3]))

    def test_ddit_block_prefilled_cache_is_visible_without_clean_replay(self):
        block = object.__new__(DDiTBlock)
        nn.Module.__init__(block)
        block.k_cache = torch.tensor([[[[1.0]], [[2.0]]]])
        block.v_cache = torch.tensor([[[[3.0]], [[4.0]]]])
        block.num_clean_cached = 2

        k = torch.tensor([[[[5.0]]]])
        v = torch.tensor([[[[6.0]]]])
        combined_k, combined_v = DDiTBlock._process_and_update_kv(
            block,
            k=k,
            v=v,
            num_clean=0,
        )

        self.assertTrue(
            torch.equal(combined_k, torch.tensor([[[[1.0]], [[2.0]], [[5.0]]]]))
        )
        self.assertTrue(
            torch.equal(combined_v, torch.tensor([[[[3.0]], [[4.0]], [[6.0]]]]))
        )
        self.assertEqual(block.num_clean_cached, 2)

    def test_ddit_block_expands_cache_for_prompt_plus_generation(self):
        block = object.__new__(DDiTBlock)
        nn.Module.__init__(block)
        block.n = 4
        block.n_heads = 1
        block.attn_qkv = nn.Linear(1, 3, bias=False)
        block.reset_kv_cache()

        prompt_k = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1)
        prompt_v = prompt_k + 10
        DDiTBlock._process_and_update_kv(block, k=prompt_k, v=prompt_v, num_clean=3)

        gen_k = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1) + 20
        gen_v = gen_k + 10
        combined_k, combined_v = DDiTBlock._process_and_update_kv(
            block,
            k=gen_k,
            v=gen_v,
            num_clean=1,
        )

        self.assertGreaterEqual(block.k_cache.shape[1], 7)
        self.assertEqual(block.num_clean_cached, 4)
        self.assertTrue(torch.equal(combined_k, torch.cat([prompt_k, gen_k], dim=1)))
        self.assertTrue(torch.equal(combined_v, torch.cat([prompt_v, gen_v], dim=1)))

    def test_dit_forward_sample_threads_clean_token_metadata(self):
        backbone = object.__new__(DIT)
        nn.Module.__init__(backbone)
        calls = []

        class DummySigmaMap(nn.Module):
            def forward(self, sigma):
                return sigma[:, None]

        class RecordingBlock(nn.Module):
            def reset_kv_cache(self):
                return None

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
                del c, mask, kwargs
                calls.append(
                    {
                        "seq_len": x.shape[1],
                        "rotary_shape": tuple(rotary_cos_sin[0].shape),
                        "kv_cache": kv_cache,
                        "num_clean": num_clean,
                        "num_clean_and_mask": num_clean_and_mask,
                    }
                )
                return x

        class DummyRotary(nn.Module):
            def forward(self, position_ids):
                batch_size, seq_len = position_ids.shape
                cos = torch.ones(batch_size, seq_len, 3, 1, 4)
                sin = torch.zeros_like(cos)
                return cos, sin

        class DummyOutputLayer(nn.Module):
            def forward(self, x, c):
                del c
                logits = torch.arange(x.shape[1] * 5, dtype=torch.float32)
                logits = logits.view(1, x.shape[1], 5)
                return logits

        backbone.causal = False
        backbone.adaLN = True
        backbone.vocab_embed = EmbeddingLayer(dim=4, vocab_dim=16)
        backbone.sigma_map = DummySigmaMap()
        backbone.rotary_emb = DummyRotary()
        backbone.blocks = nn.ModuleList([RecordingBlock()])
        backbone.output_layer = DummyOutputLayer()
        backbone.rotary_cos_sin_sorted = None

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

        self.assertEqual(output.logits.shape, (1, 1, 5))
        self.assertTrue(calls[0]["kv_cache"])
        self.assertEqual(calls[0]["num_clean"], 2)
        self.assertEqual(calls[0]["num_clean_and_mask"], 3)
        self.assertEqual(calls[0]["rotary_shape"], (1, 3, 3, 1, 4))


if __name__ == "__main__":
    unittest.main()
