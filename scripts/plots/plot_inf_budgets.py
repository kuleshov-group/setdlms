from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.multiprocessing as mp
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizerFast
from transformers.cache_utils import Cache, DynamicCache

from scripts.utils import maybe_add_missing_special_tokens
from src.denoiser.diffusion import AnyOrderBD3LM, BD3LMConfig, DiffusionGenerationConfig, SetDiffusionGenerationConfig
from src.noise_schedule.noise_schedules import EaseOutPowerNoise


class DummyBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, device: str | torch.device):
        super().__init__()
        self._logits = torch.randn(vocab_size, device=device)[None, None, :]

    def forward(self, x, **kwargs):
        return {"logits": self._logits.repeat(x.shape[0], x.shape[1], 1), "past_key_values": DynamicCache()}


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    return maybe_add_missing_special_tokens(tokenizer)


def _split_num_trials(num_trials: int, n_parts: int) -> list[int]:
    if n_parts <= 0:
        raise ValueError(f"n_parts must be > 0, got {n_parts}")
    base = num_trials // n_parts
    rem = num_trials % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _generate_inf_budgets_worker(
    *,
    device_idx: int,
    L: int,
    num_steps: int,
    num_trials: int,
    desired_block_size: int,
    queue: mp.Queue,
) -> None:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but a CUDA worker was launched.")
        torch.cuda.set_device(device_idx)
        device = f"cuda:{device_idx}"

        tokenizer = build_tiny_tokenizer()
        backbone = DummyBackbone(vocab_size=int(tokenizer.mask_token_id) + 1, device=device)

        config = BD3LMConfig(
            block_size=L,
            eval_block_size=L,
            tokenizer_name="gpt2",
            length=L,
        )
        noise = EaseOutPowerNoise(
            block_size=L,
            desired_block_size=desired_block_size,
            max_block_size=L,
            length=L,
            k=1.0 if desired_block_size in {1, L} else None,
        )

        model = AnyOrderBD3LM(config=config)
        model.bos_token_id = tokenizer.pad_token_id
        model.mask_token_id = tokenizer.mask_token_id
        model.noise_schedule = noise
        model.backbone = backbone
        model.to(device)

        generation_config = SetDiffusionGenerationConfig(
            max_length=L,
            use_cache=True,
            compute_inf_budget=True,
            align_inputs_to_blocks=False,
            block_size=L,
            max_window_size=L,
            do_sample=True,
            num_steps=num_steps,
            min_t=1e-5,
            first_hitting=False,
        )

        inf_budgets: list[float] = []
        it = range(num_trials)
        if device_idx == 0:
            it = tqdm(it, total=num_trials, desc=f"Generating (dbs={desired_block_size}) [rank0]")

        for _ in it:
            generation_outputs = model.generate(
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
            inf_budgets.append(float(generation_outputs.inf_budget))
            if device_idx == 0:
                it.set_postfix(inf_budget=float(generation_outputs.inf_budget), mean=float(np.mean(inf_budgets)))

        queue.put(("ok", device_idx, inf_budgets))
    except Exception as e:
        queue.put(("err", device_idx, f"{type(e).__name__}: {e}"))


def run_for_block_size(
    *,
    L: int,
    num_steps: int,
    num_trials: int,
    desired_block_size: int,
    tokenizer: PreTrainedTokenizerFast,
    backbone: torch.nn.Module,
) -> tuple[float, float, float]:
    config = BD3LMConfig(
        block_size=L,
        eval_block_size=L,
        tokenizer_name="gpt2",
        length=L,
    )
    noise = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=desired_block_size,
        max_block_size=L,
        length=L,
        k=1.0 if desired_block_size in {1, L} else None,
    )

    model = AnyOrderBD3LM(config=config)
    model.bos_token_id = tokenizer.pad_token_id
    model.mask_token_id = tokenizer.mask_token_id
    model.noise_schedule = noise
    model.backbone = backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(model.noise_schedule.b, model.noise_schedule.k)

    generation_config = SetDiffusionGenerationConfig(
        max_length=L,
        use_cache=True,
        compute_inf_budget=True,
        align_inputs_to_blocks=False,
        block_size=L,
        max_window_size=L,
        do_sample=True,
        num_steps=num_steps,
        min_t=1e-5,
        first_hitting=False,
    )
    inf_budgets: list[float] = []
    pbar = tqdm(range(num_trials), desc=f"Generating (dbs={desired_block_size})")
    for _ in pbar:
        generation_outputs = model.generate(
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        inf_budget = float(generation_outputs.inf_budget)
        inf_budgets.append(inf_budget)
        pbar.set_postfix(inf_budget=inf_budget, mean=float(np.mean(inf_budgets)))

    theoretical = float(noise.compute_inf_budget())
    empirical_mean = float(np.mean(inf_budgets))
    empirical_std = float(np.std(inf_budgets))
    return theoretical, empirical_mean, empirical_std


def run_for_block_size_multi_gpu(
    *,
    L: int,
    num_steps: int,
    num_trials: int,
    desired_block_size: int,
) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU path requested but CUDA is not available.")
    n_devices = torch.cuda.device_count()
    if n_devices <= 1:
        raise RuntimeError(f"Multi-GPU path requested but only {n_devices} CUDA device(s) found.")

    # Theoretical budget depends only on the schedule parameters.
    noise = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=desired_block_size,
        max_block_size=L,
        length=L,
        k=1.0 if desired_block_size in {1, L} else None,
    )
    theoretical = float(noise.compute_inf_budget())

    trial_splits = _split_num_trials(num_trials, n_devices)
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    procs: list[mp.Process] = []
    for device_idx, n_trials in enumerate(trial_splits):
        if n_trials <= 0:
            continue
        p = ctx.Process(
            target=_generate_inf_budgets_worker,
            kwargs=dict(
                device_idx=device_idx,
                L=L,
                num_steps=num_steps,
                num_trials=n_trials,
                desired_block_size=desired_block_size,
                queue=queue,
            ),
        )
        p.start()
        procs.append(p)

    inf_budgets: list[float] = []
    remaining_ok = len(procs)
    try:
        while remaining_ok > 0:
            status, device_idx, payload = queue.get()
            if status != "ok":
                raise RuntimeError(f"Worker cuda:{device_idx} failed: {payload}")

            inf_budgets.extend(payload)
            remaining_ok -= 1
    finally:
        for p in procs:
            p.join()

    empirical_mean = float(np.mean(inf_budgets))
    empirical_std = float(np.std(inf_budgets))
    return theoretical, empirical_mean, empirical_std


def main() -> None:
    L = 512
    num_steps = 2048
    desired_block_sizes = [4, 8, 16, 32]
    test_run = False
    num_trials = 3 if test_run else 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tiny_tokenizer()
    backbone = DummyBackbone(vocab_size=int(tokenizer.mask_token_id) + 1, device=device)

    theoretical_inf_budgets: list[float] = []
    empirical_inf_budgets_mean: list[float] = []
    empirical_inf_budgets_std: list[float] = []

    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi_gpu:
        print(f"Using multi-GPU generation across {torch.cuda.device_count()} CUDA devices.")

    for desired_block_size in desired_block_sizes:
        if use_multi_gpu:
            theoretical, mean, std = run_for_block_size_multi_gpu(
                L=L,
                num_steps=num_steps,
                num_trials=num_trials,
                desired_block_size=desired_block_size,
            )
        else:
            theoretical, mean, std = run_for_block_size(
                L=L,
                num_steps=num_steps,
                num_trials=num_trials,
                desired_block_size=desired_block_size,
                tokenizer=tokenizer,
                backbone=backbone,
            )
        theoretical_inf_budgets.append(theoretical)
        empirical_inf_budgets_mean.append(mean)
        empirical_inf_budgets_std.append(std)
        print(f"dbs={desired_block_size:>4} | empirical: {mean:.4f} +/- {std:.4f} | theoretical: {theoretical:.4f}")

    x_labels = [str(b) for b in desired_block_sizes]
    fig = go.Figure()
    fig.add_bar(
        x=x_labels,
        y=empirical_inf_budgets_mean,
        error_y=dict(type="data", array=empirical_inf_budgets_std, visible=True),
        name=f"Simulated (T={num_steps}, N={num_trials})",
    )
    fig.add_scatter(
        x=x_labels,
        y=theoretical_inf_budgets,
        mode="lines+markers",
        name="Theoretical",
    )
    fig.update_layout(
        title="",
        xaxis_title="Matched block size",
        yaxis_title="Inference Prediction Budget",
        template="plotly_white",
        width=600,
        height=500,
        xaxis=dict(type="category", title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=16)),
        legend=dict(
            font=dict(size=16),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
            xref="paper",
            yref="paper",
        ),
        font=dict(size=16),
        margin=dict(l=60, r=20, t=0, b=60),
    )

    block_sizes_str = "_".join(x_labels)
    suffix = "png" if test_run else "pdf"
    out_path = Path(f"inf_budgets_L{L}_desired_block_sizes_{block_sizes_str}.{suffix}")
    fig.write_image(out_path)
    print(f"Saved figure to {out_path.resolve()}")


if __name__ == "__main__":
    main()