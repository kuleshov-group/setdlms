# ReFusion Patch Instructions

This document translates the ReFusion audit into a concrete patch plan.

Scope:
- Preserve the local denoiser/backbone split.
- Preserve the local HF backbone abstraction.
- Match official ReFusion semantics where they are algorithmically relevant.
- Do not copy the upstream Qwen wrapper wholesale.

Primary upstream references:
- `generate.py`
- `train.py`
- `eval.py`
- `qwen3/modeling_qwen3_refusion.py`
- `qwen3/diffusion_cache_utils.py`

## Patch order

1. Fix generation-length semantics first.
2. Fix eval/loading so ReFusion cannot silently degrade to plain AR generation.
3. Tighten the mask-token contract.
4. Decide whether to match upstream empty-answer handling exactly.
5. Add tests for the failure modes above.

## 1. Fix generation-length divisibility

Files:
- `src/denoiser/refusion.py`
- `tests/test_refusion.py`

Problem:
- The current code pads `max_new_tokens` to `lcm(slot_size, serial_num_blocks)`.
- The decode loop later reshapes each serial block into `(-1, slot_size)`.
- That reshape requires each per-block length to be divisible by `slot_size`.
- Therefore `padded_new_tokens` must be divisible by `slot_size * serial_num_blocks`, not just the LCM.

Required change:
- In `ReFusion.generate`, replace the current padding rule with one of these two options:
  1. Strict faithfulness option: require `max_new_tokens % (slot_size * serial_num_blocks) == 0` and raise a clear error otherwise.
  2. Local-API convenience option: pad to the next multiple of `slot_size * serial_num_blocks`, then trim back to `requested_new_tokens` at the end.

Recommendation:
- Use option 2, because the local `generate()` API exposes arbitrary `max_new_tokens`.
- Update the method comment so it states the real invariant.

Tests to add:
- A test where `slot_size=8`, `serial_num_blocks=2`, and `max_new_tokens=8` no longer crashes.
- A test where the returned sequence is trimmed back to the requested token count.

## 2. Prevent silent fallback to non-ReFusion eval behavior

Files:
- `scripts/eval/model_loading.py`
- optionally `tests/test_refusion.py` or a new eval-loading test file

Problem:
- `load_eval_model()` currently accepts any successfully loaded HF causal LM.
- For ReFusion checkpoints or model names, that can bypass the local `ReFusion` wrapper entirely.
- That silently changes generation semantics from ReFusion decode to plain autoregressive decode.

Required change:
- When the requested model is ReFusion, always return a `ReFusion` wrapper.
- Treat `"refusion"` in the checkpoint/model path as a hard routing signal.
- If the user points to an HF backbone, wrap that backbone in `ReFusion`.
- If the required ReFusion config inputs are missing, raise a clear error instead of falling back.

Recommended logic:
- In `load_eval_model()`, detect `is_refusion = "refusion" in pretrained_model_name_or_path.lower()` once.
- If `is_refusion`:
  - Prefer checkpoint-dir loading if `config.yaml` exists.
  - Otherwise load the HF backbone, then call `_load_legacy_denoiser(..., model=model, ...)`.
  - Do not return the raw HF backbone directly.

Tests to add:
- A test that a ReFusion-named path returns a `ReFusion` instance even when `_load_hf_model()` succeeds.
- A test that generation on the loaded object resolves to `ReFusion.generate`, not plain backbone generation.

## 3. Enforce the ReFusion mask-token contract

Files:
- `scripts/eval/model_loading.py`
- optionally `configs/model/refusion.yaml`
- optionally `scripts/utils.py`

Problem:
- Official ReFusion training/eval assumes a dedicated mask token (`<|mask|>` in the upstream setup).
- The local repo currently relies on generic tokenizer fallback behavior.
- That is acceptable for generic denoisers, but it is ambiguous for ReFusion reproducibility.

Required change:
- Add an explicit validation path for ReFusion:
  - `tokenizer.mask_token_id` must exist.
  - The model config must receive that mask token explicitly.
- If exact upstream reproduction is required, document that the tokenizer should expose the official ReFusion mask token rather than an arbitrary fallback token.

Recommendation:
- Keep the local tokenizer abstraction.
- Add a ReFusion-specific warning or assertion in loading code if `mask_token_id` is missing.
- Do not silently proceed with `None`.

Optional hardening:
- Add a config flag such as `require_explicit_mask_token: true` for ReFusion models.

## 4. Decide whether to match upstream empty-answer handling exactly

Files:
- `src/denoiser/refusion.py`
- tests

Problem:
- Upstream `forward_process()` drops samples with zero answer slots.
- The local `_prepare_inputs()` preserves them and assigns all-ignored labels.
- That changes effective batch weighting for the MDM term.

Required decision:
- If the goal is exact upstream faithfulness, drop empty-answer rows before stacking.
- If the goal is robustness to generic local datasets, keep the current behavior but document it as an intentional deviation.

If matching upstream:
- Filter empty-answer rows out of all processed tensors before stacking.
- Ensure `_compute_loss()` still handles fully masked or fully unmasked edge cases safely.

Tests to add:
- A test that zero-answer examples are either dropped or explicitly preserved, depending on the chosen policy.
- The chosen policy should be documented in the test name.

## 5. Keep cache behavior aligned with the local decode path

Files:
- `src/denoiser/refusion.py`
- tests

Problem:
- The local cache helper is sufficient for the current decode loop, but it is not a full upstream API clone.

Required change:
- No major algorithmic rewrite is needed.
- Add either:
  - a compatibility alias named `select_partial`, or
  - a short comment that the local helper intentionally exposes only the subset needed by `ReFusion.generate`.

Tests to add:
- Cache batch-repeat and batch-select coverage around the speculative verification branch.
- A generation test that exercises EOS in the speculative branch and confirms cache selection/cropping still produces the expected output length.

## 6. Test matrix to add before calling the patch complete

At minimum:
- preprocessing: shuffled unmasked slots, masked slots appended in original order, position IDs preserved
- loss: mixed AR/MDM example, all-masked-slot example, all-unmasked-slot example
- generation: incompatible `max_new_tokens` length, EOS in confident prefix, EOS during speculative verification
- loading: ReFusion model path cannot return a raw HF causal LM
- mask token: ReFusion loading fails clearly if no mask token is available

## 7. Non-goals

Do not treat these as required for this patch:
- matching upstream training dataset construction
- matching upstream model size
- matching timestep-conditioning behavior
- copying the upstream Qwen-specific wrapper structure directly

## Deliverables

A complete patch should include:
- code fixes for generation-length handling
- eval/loading routing fixes
- explicit ReFusion mask-token validation
- a documented decision on empty-answer samples
- new regression tests for generation, loading, and cache behavior
