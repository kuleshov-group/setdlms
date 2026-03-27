from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from src.denoiser.base import DenoiserInput
from src.denoiser.diffusion import DiffusionGenerationConfig, MDLM, MDLMConfig


def _make_bare_mdlm(**overrides) -> MDLM:
    model = object.__new__(MDLM)
    nn.Module.__init__(model)
    config = {
        "keep_clean_bos": False,
        "papl_alpha": 0.0,
        "papl_tau": 1.0,
        "train_on_nelbo": False,
        "mdlm_loss_scale": False,
        "length": 4,
        "block_size": 4,
    }
    config.update(overrides)
    model.config = SimpleNamespace(**config)
    model.mask_token_id = 4
    model.neg_infinity = -1e12
    model.training = True
    return model


def _papl_reference_loss(
    log_probs: torch.Tensor,
    x0: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_log_probs = log_probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    target_nll = -target_log_probs
    masked_nll = target_nll * mask

    detached_scores = (target_log_probs.detach() / tau).masked_fill(
        ~mask.bool(), float("-inf")
    )
    planner_weights = torch.zeros_like(target_log_probs)
    has_masked = mask.bool().any(dim=-1)
    if has_masked.any():
        planner_weights[has_masked] = F.softmax(detached_scores[has_masked], dim=-1)
    planner_weights = torch.where(mask.bool(), planner_weights, 0.0)

    n_masked = mask.sum(dim=-1).clamp_min(1).float()
    base_weight = n_masked.reciprocal().unsqueeze(-1)
    weights = base_weight * (1.0 + alpha * planner_weights)
    loss = (weights * masked_nll).sum(dim=-1).mean()
    return loss, planner_weights


class MDLMPAPLTests(unittest.TestCase):
    def test_mdlm_config_validates_papl_parameters(self):
        with self.assertRaises(ValueError):
            MDLMConfig(papl_alpha=-0.1)
        with self.assertRaises(ValueError):
            MDLMConfig(papl_tau=0.0)

    def test_papl_loss_matches_reference_formula(self):
        model = _make_bare_mdlm(papl_alpha=1.5, papl_tau=0.7, block_size=1)
        log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.6, 0.3, 0.1, 0.0, 0.0],
                        [0.1, 0.7, 0.2, 0.0, 0.0],
                        [0.2, 0.2, 0.6, 0.0, 0.0],
                        [0.1, 0.2, 0.3, 0.4, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        )
        x0 = torch.tensor([[0, 1, 2, 3]])
        xt = torch.tensor([[model.mask_token_id, 1, model.mask_token_id, 3]])
        tokens_mask = torch.ones_like(x0, dtype=torch.float32)
        inputs = DenoiserInput(
            xt=xt,
            x0=x0,
            tokens_mask=tokens_mask,
            alpha_t=torch.ones_like(tokens_mask),
            alpha_t_prime=torch.zeros_like(tokens_mask),
        )

        output = model._compute_loss(log_probs, inputs)
        expected_loss, planner_weights = _papl_reference_loss(
            log_probs=log_probs,
            x0=x0,
            mask=(xt == model.mask_token_id) & tokens_mask.bool(),
            alpha=model.config.papl_alpha,
            tau=model.config.papl_tau,
        )

        self.assertTrue(torch.allclose(output.loss, expected_loss))
        self.assertTrue(torch.allclose(output.nlls, -log_probs.gather(-1, x0[..., None]).squeeze(-1)))
        self.assertTrue(
            torch.allclose(
                output.other_loss_terms["papl_avg_n_masked"], torch.tensor(2.0)
            )
        )
        expected_entropy = -(planner_weights * planner_weights.clamp_min(1e-8).log()).sum(
            dim=-1
        ).mean()
        self.assertTrue(
            torch.allclose(
                output.other_loss_terms["papl_avg_planner_entropy"], expected_entropy
            )
        )

    def test_papl_uses_detached_planner_weights(self):
        model = _make_bare_mdlm(papl_alpha=1.0, papl_tau=0.5)
        logits = torch.tensor(
            [
                [
                    [1.2, 0.2, -0.1, -1.0, -5.0],
                    [0.1, 1.1, 0.2, -0.4, -5.0],
                    [0.3, -0.5, 1.4, 0.0, -5.0],
                ]
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        x0 = torch.tensor([[0, 1, 2]])
        xt = torch.tensor([[model.mask_token_id, model.mask_token_id, 2]])
        tokens_mask = torch.ones_like(x0, dtype=torch.float32)

        inputs = DenoiserInput(
            xt=xt,
            x0=x0,
            tokens_mask=tokens_mask,
            alpha_t=torch.ones_like(tokens_mask),
            alpha_t_prime=torch.zeros_like(tokens_mask),
        )
        output = model._compute_loss(F.log_softmax(logits, dim=-1), inputs)
        output.loss.backward()
        observed_grad = logits.grad.clone()

        logits_ref = logits.detach().clone().requires_grad_(True)
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        expected_loss, _ = _papl_reference_loss(
            log_probs=log_probs_ref,
            x0=x0,
            mask=(xt == model.mask_token_id) & tokens_mask.bool(),
            alpha=model.config.papl_alpha,
            tau=model.config.papl_tau,
        )
        expected_loss.backward()

        self.assertTrue(torch.allclose(observed_grad, logits_ref.grad, atol=1e-6))

    def test_papl_alpha_zero_matches_current_training_loss(self):
        model = _make_bare_mdlm(papl_alpha=0.0)
        x0 = torch.tensor([[0, 1, 2, 3]])
        xt = torch.tensor([[model.mask_token_id, 1, model.mask_token_id, 3]])
        tokens_mask = torch.ones_like(x0, dtype=torch.float32)
        log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.7, 0.1, 0.1, 0.1, 0.0],
                        [0.1, 0.7, 0.1, 0.1, 0.0],
                        [0.1, 0.1, 0.7, 0.1, 0.0],
                        [0.1, 0.1, 0.1, 0.7, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        )
        inputs = DenoiserInput(
            xt=xt,
            x0=x0,
            tokens_mask=tokens_mask,
            alpha_t=torch.ones_like(tokens_mask),
            alpha_t_prime=torch.zeros_like(tokens_mask),
        )

        output = model._compute_loss(log_probs, inputs)
        target_log_probs = log_probs.gather(-1, x0[..., None]).squeeze(-1)
        expected = -(
            target_log_probs * tokens_mask
        ).sum(dim=-1) / (xt == model.mask_token_id).sum(dim=-1)

        self.assertTrue(torch.allclose(output.loss, expected.mean()))
        self.assertNotIn("papl_enabled", output.other_loss_terms)
        self.assertNotIn("papl_avg_n_masked", output.other_loss_terms)
        self.assertNotIn("papl_avg_planner_entropy", output.other_loss_terms)
        self.assertNotIn("papl_avg_correct_prob_on_masked", output.other_loss_terms)

    def test_papl_ignores_unmasked_and_filtered_masked_positions(self):
        model = _make_bare_mdlm(papl_alpha=2.0, papl_tau=1.0)
        x0 = torch.tensor([[0, 1, 2, 3]])
        xt = torch.tensor([[model.mask_token_id, 1, model.mask_token_id, model.mask_token_id]])
        tokens_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.8, 0.1, 0.05, 0.05, 0.0],
                        [0.01, 0.01, 0.01, 0.97, 0.0],
                        [0.01, 0.01, 0.01, 0.97, 0.0],
                        [0.2, 0.2, 0.2, 0.4, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        )
        inputs = DenoiserInput(
            xt=xt,
            x0=x0,
            tokens_mask=tokens_mask,
            alpha_t=torch.ones_like(tokens_mask),
            alpha_t_prime=torch.zeros_like(tokens_mask),
        )

        output = model._compute_loss(log_probs, inputs)
        masked_positions = (xt == model.mask_token_id) & tokens_mask.bool()
        expected_loss, _ = _papl_reference_loss(
            log_probs=log_probs,
            x0=x0,
            mask=masked_positions,
            alpha=model.config.papl_alpha,
            tau=model.config.papl_tau,
        )

        self.assertTrue(torch.allclose(output.loss, expected_loss))
        self.assertTrue(torch.equal(masked_positions.int(), torch.tensor([[1, 0, 0, 1]])))

    def test_keep_clean_bos_eval_path_is_unchanged_with_papl_enabled(self):
        x0 = torch.tensor([[0, 1, 2]])
        xt = torch.tensor([[0, 1, 2]])
        tokens_mask = torch.ones_like(x0, dtype=torch.float32)
        alpha_t = torch.full_like(tokens_mask, 0.25)
        alpha_t_prime = torch.full_like(tokens_mask, -0.5)
        log_probs = torch.log(
            torch.tensor(
                [[[0.7, 0.2, 0.1, 0.0, 0.0], [0.1, 0.7, 0.2, 0.0, 0.0], [0.1, 0.2, 0.7, 0.0, 0.0]]],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        )
        inputs = DenoiserInput(
            xt=xt.clone(),
            x0=x0.clone(),
            tokens_mask=tokens_mask.clone(),
            alpha_t=alpha_t.clone(),
            alpha_t_prime=alpha_t_prime.clone(),
        )

        disabled = _make_bare_mdlm(keep_clean_bos=True, papl_alpha=0.0)
        enabled = _make_bare_mdlm(keep_clean_bos=True, papl_alpha=1.0)
        disabled.training = False
        enabled.training = False

        out_disabled = disabled._compute_loss(
            log_probs.clone(),
            DenoiserInput(
                xt=inputs.xt.clone(),
                x0=inputs.x0.clone(),
                tokens_mask=inputs.tokens_mask.clone(),
                alpha_t=inputs.alpha_t.clone(),
                alpha_t_prime=inputs.alpha_t_prime.clone(),
            ),
        )
        out_enabled = enabled._compute_loss(
            log_probs.clone(),
            DenoiserInput(
                xt=inputs.xt.clone(),
                x0=inputs.x0.clone(),
                tokens_mask=inputs.tokens_mask.clone(),
                alpha_t=inputs.alpha_t.clone(),
                alpha_t_prime=inputs.alpha_t_prime.clone(),
            ),
        )

        self.assertTrue(torch.allclose(out_disabled.loss, out_enabled.loss))
        self.assertTrue(torch.allclose(out_disabled.nlls, out_enabled.nlls))

    def test_nlls_and_masked_tokens_remain_metric_compatible(self):
        model = _make_bare_mdlm(papl_alpha=1.0, block_size=1)
        x0 = torch.tensor([[0, 1, 2, 3]])
        xt = torch.tensor([[model.mask_token_id, 1, model.mask_token_id, 3]])
        tokens_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        log_probs = torch.log(
            torch.tensor(
                [
                    [
                        [0.7, 0.1, 0.1, 0.1, 0.0],
                        [0.1, 0.7, 0.1, 0.1, 0.0],
                        [0.1, 0.1, 0.7, 0.1, 0.0],
                        [0.1, 0.1, 0.1, 0.7, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        )
        inputs = DenoiserInput(
            xt=xt,
            x0=x0,
            tokens_mask=tokens_mask,
            alpha_t=torch.ones_like(tokens_mask),
            alpha_t_prime=torch.zeros_like(tokens_mask),
        )

        output = model._compute_loss(log_probs, inputs)
        target_log_probs = log_probs.gather(-1, x0[..., None]).squeeze(-1)

        self.assertTrue(
            torch.allclose(output.nlls, -target_log_probs * tokens_mask)
        )
        self.assertTrue(
            torch.equal(
                output.other_loss_terms["masked_tokens"],
                (xt == model.mask_token_id).int(),
            )
        )

    def test_generate_output_is_unchanged_when_papl_is_enabled(self):
        def make_model(papl_alpha: float) -> MDLM:
            model = _make_bare_mdlm(
                papl_alpha=papl_alpha,
                length=4,
                block_size=2,
            )
            model._compute_sampling_lengths = lambda generation_config, input_length, max_new_tokens=None, max_length=None: (
                2,
                2,
            )
            model._compute_max_blocks_and_pad_input = (
                lambda inputs, generation_config, max_new_tokens, block_size, is_infill_task, mdlm_inference: (
                    inputs,
                    1,
                    None,
                )
            )
            model._sample_prior = (
                lambda inputs, batch_size, generation_config, max_new_tokens, block_size, is_infill_task, device: torch.full(
                    (batch_size, max_new_tokens),
                    fill_value=model.mask_token_id,
                    dtype=torch.long,
                    device=torch.device(device),
                )
            )
            model._sample_generation_timesteps = (
                lambda generation_config, max_length=None, device=None, dtype=torch.float64, first_hitting_times=None: torch.tensor(
                    [1.0], device=device
                )
            )
            model._prepare_inputs_inference = (
                lambda input_ids=None, attention_mask=None, context=None, context_mask=None, cache=None, **backbone_kwargs: (
                    DenoiserInput(xt=input_ids.clone()),
                    {} if cache is None else cache,
                )
            )

            def _generate_unconditional(
                self,
                generation_config,
                t,
                next_t,
                denoiser_inputs=None,
                cache=None,
                running_generation=None,
                inputs_offset=0,
                logits_processor=None,
                sample_indices=None,
                input_indices=None,
                return_updated_cache=False,
                cache_len=None,
                window_size=0,
                block_size=0,
                **kwargs,
            ):
                del (
                    generation_config,
                    t,
                    next_t,
                    running_generation,
                    inputs_offset,
                    logits_processor,
                    return_updated_cache,
                    cache_len,
                    window_size,
                    block_size,
                    kwargs,
                )
                xs = torch.tensor([[1, 2]], device=denoiser_inputs.xt.device)
                return xs, {} if cache is None else cache

            model._generate_unconditional = MethodType(_generate_unconditional, model)
            return model

        generation_config = DiffusionGenerationConfig(
            num_steps=1,
            block_size=2,
            use_cache=False,
            do_sample=False,
        )
        inputs = torch.zeros((1, 0), dtype=torch.long)
        disabled = make_model(0.0)
        enabled = make_model(2.0)

        samples_disabled = disabled.generate(
            inputs=inputs,
            generation_config=generation_config,
            batch_size=1,
            device="cpu",
            disable_pbar=True,
        )
        samples_enabled = enabled.generate(
            inputs=inputs,
            generation_config=generation_config,
            batch_size=1,
            device="cpu",
            disable_pbar=True,
        )

        self.assertTrue(torch.equal(samples_disabled, samples_enabled))
        self.assertTrue(torch.equal(samples_enabled, torch.tensor([[1, 2]])))


if __name__ == "__main__":
    unittest.main()
