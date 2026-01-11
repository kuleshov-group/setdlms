from src.denoiser.diffusion import AnyOrderBD3LM, BD3LMConfig
from src.noise_schedule.noise_schedules import EaseOutPowerNoise, LinearNoise
from transformers import AutoTokenizer, GenerationConfig
import torch
import numpy as np
from scripts.utils import maybe_add_missing_special_tokens
from tqdm import tqdm
# tokenizer_name = "bert-base-uncased"
# tokenizer = maybe_add_missing_special_tokens(AutoTokenizer.from_pretrained(tokenizer_name))

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

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

tokenizer = maybe_add_missing_special_tokens(tokenizer)
backbone = torch.nn.Module()

device = "cuda" if torch.cuda.is_available() else "cpu"
rand_weights = torch.randn(tokenizer.mask_token_id+1)[None, None, :].to(device)

def dummy_backbone_forward(x, **kwargs):
    return {
        "logits": rand_weights,
        "past_key_values": {},
    }

backbone.forward = dummy_backbone_forward

L = 32

test_run = False
num_trials = 3 if test_run else 500
num_steps = 128
desired_block_sizes = [4, 8, 16, 32]

theoretical_inf_budgets = []
empirical_inf_budgets_mean = []
empirical_inf_budgets_std = []
for desired_block_size in desired_block_sizes:
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
        k=1.0 if desired_block_size in {1,L} else None,
    )
    # noise = LinearNoise(
    #     block_size=desired_block_size,
    #     length=L,
    # )
    theoretical_inf_budget = noise.compute_inf_budget()
    theoretical_inf_budgets.append(theoretical_inf_budget)
    model = AnyOrderBD3LM(config=config)
    model.bos_token_id = tokenizer.pad_token_id
    model.mask_token_id = tokenizer.mask_token_id
    model.noise_schedule = noise
    model.backbone = backbone

    generation_config = GenerationConfig(
        max_length=L,
        use_cache=True,
        compute_inf_budget=True,
        align_inputs_to_blocks=False,
        block_size=L,
        max_window_size=L,
        do_sample=True,
        num_steps=num_steps,
        min_t=1e-5,
    )

    inf_budgets = []
    pbar = tqdm(range(num_trials), desc="Generating")
    for trial in pbar:
        generation_outputs = model.generate(generation_config=generation_config, return_dict_in_generate=True)
        inf_budget = generation_outputs.inf_budget
        inf_budgets.append(inf_budget)
        pbar.set_postfix(inf_budget=inf_budget)
        print(f"Average inf budget: {np.mean(inf_budgets)} +/- {np.std(inf_budgets)}")
    empirical_inf_budgets_mean.append(np.mean(inf_budgets))
    empirical_inf_budgets_std.append(np.std(inf_budgets))

import plotly.graph_objects as go
from pathlib import Path

x_labels = [str(b) for b in desired_block_sizes]

fig = go.Figure()

fig.add_bar(
    x=x_labels,
    y=empirical_inf_budgets_mean,
    error_y=dict(
        type="data",
        array=empirical_inf_budgets_std,
        visible=True,
    ),
    name=f"Simulated (T={num_steps}, N={num_trials})",
)

fig.add_scatter(
    x=x_labels,
    y=theoretical_inf_budgets,
    mode="lines+markers",
    name="Theoretical",
)

fig.update_layout(
    # title="Inference Prediction Budgets for Soft Block Diffusion",
    title="",
    xaxis_title=f"Matched block size",
    yaxis_title="Inference Prediction Budget",
    template="plotly_white",
    width=600,
    height=500,
    xaxis=dict(
        type="category",
        title_font=dict(size=18),
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=16),
    ),
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
    font=dict(size=16),  # global fallback
)
fig.update_layout(
    margin=dict(
        l=60,   # left: just enough for y-axis label
        r=20,   # right: minimal
        t=0,   # top: fits centered legend
        b=60,   # bottom: fits x-axis label
    )
)

block_sizes_str = "_".join(x_labels)
if test_run:
    out_path = Path(f"inf_budgets_L{L}_desired_block_sizes_{block_sizes_str}.png")
else:
    out_path = Path(f"inf_budgets_L{L}_desired_block_sizes_{block_sizes_str}.pdf")

fig.write_image(out_path)
print(f"Saved figure to {out_path.resolve()}")