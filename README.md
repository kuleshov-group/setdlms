# Set Diffusion: Interpolating Token Orderings between Autoregression and Diffusion for Fast and Flexible Decoding

[![deploy](https://img.shields.io/badge/Paper_📃-green)](https://arxiv.org/abs/2510.22852)
[![deploy](https://img.shields.io/badge/Blog_📝%20%20-8A2BE2)](https://m-arriola.com/setdlms)
[![deploy](https://img.shields.io/badge/HuggingFace_🤗%20-E2D2%20-orange)](https://huggingface.co/collections/kuleshov-group/setdlms)

## 0. Setup

### Setup environment

Install conda:

```bash
# For conda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
bash miniconda.sh -b -p /opt/conda
```

Create the locked conda environment:

```bash
conda env create -f requirements.yaml
conda activate dllm-dev
```

The conda file installs Python, pip, the pinned dependencies in
[`requirements-lock.txt`](requirements-lock.txt), and this package in editable mode.
To validate a fresh environment, run:

```bash
python -m pip check
python - <<'PY'
import torch
import transformers
import src

print(torch.__version__)
print(transformers.__version__)
PY
```

For a pip-only install, use the pinned direct requirements:

```bash
python -m pip install -r requirements.txt -e .
```

Use [`requirements-lock.txt`](requirements-lock.txt) when byte-for-byte dependency
reproduction is required. Regenerate the lockfile only as an intentional release step.

Activate an existing environment with:

```bash
conda activate dllm-dev
```

We also include a [`setup_env.sh`](./setup_env.sh) script for runtime shell variables on
compute nodes. Run it from the repository root after the environment has been created:

```bash
source setup_env.sh
```

Credentials are intentionally opt-in. If you want the setup script to source private
W&B or Hugging Face settings, point `DLLM_PRIVATE_ENV` at a local, untracked shell file:

```shell
export WANDB__SERVICE_WAIT=600
export WANDB_ENTITY="<WANDB_ENTITY>"
export WANDB_API_KEY="<WANDB_API_KEY>"
export HUGGINGFACE_TOKEN="<HF_TOKEN>"
```

Then run `DLLM_PRIVATE_ENV=/path/to/private_env.sh source setup_env.sh`.

- WandB token can be found [here](https://wandb.ai/authorize).
- Hugging Face token can be created [here](https://huggingface.co/settings/tokens).

## 1. Code Organization
1. [`bash_scripts`](bash_scripts): These shell scripts can be used to reproduce the
experiments from our work.
2. [`configs`](configs): We use Hydra config files to organize experiments.
   1. [`config.yaml`](configs/config.yaml): Entry point for launching
   training experiments.
   2. [`eval_config.yaml`](configs/eval_config.yaml): Entry point for
   evaluations.
3. [`scripts`](scripts): The main training and evaluation scripts
   1. [`scripts/composer_scripts/train_discrete_denoiser.py`](scripts/composer_scripts/train_discrete_denoiser.py):
   This script is the main training entry point.
   2. [`scripts/eval`](scripts/eval): These scripts run evaluation for the
   translation, summarization, and math reasoning datasets, as well as any likelihood
   evaluation.
4. [`src`](src):
   1. [`src/denoiser`](src/denoiser): During training, denoisers take in "noisy" inputs
   and predict clean signals.
   At inference, starting from a purely noisy signal, through iterative denoising, these
   classes produce samples that resemble data.
      1. `AR`: We can view autoregressive models within this paradigm.
      Noise is applied by masking tokens one at a time from right-to-left.
      Denoising is done one token at a time, left-to-right.
      2. `Diffusion`: We implement masked diffusion models:
         - `MDLM`: Standard masked diffusion.
         - `BD3LM`: Block diffusion models.
         - `SetDLM`: Set diffusion models.
   2. [`src/backbone`](src/backbone): These are the underlying neural networks that take
   in noisy inputs and produce logits.
   Each denoiser is parameterized by a backbone.
   The denoiser can optionally post-process the logit outputs of the backbone to
   produce log-probs over the clean sequence.

## 2. Reproducing Experiments
The shell scripts provided in [`bash_scripts`](bash_scripts) can be used to reproduce
the training and evaluations from our work.
- For training, the files follow a convention where the dataset and denoiser class are
specified.
For example, to train SetDLM on the GSM8K dataset, use
[`run_train_setdlm_gsm8k.sh`](bash_scripts/run_train_setdlm_gsm8k.sh).
- Once models have been trained or downloaded, the provided evaluation scripts can be used
to reproduce the reported metrics from our work. For example, to evaluate GSM8K models,
use [`run_lm_eval_harness.sh`](bash_scripts/run_lm_eval_harness.sh). The task-specific
wrappers below document the supported evaluation entry points. Plotting utilities are kept
out of this release repo; generate plots from exported metrics/TSV artifacts downstream.

Evaluation scripts resolve checkpoints through [`bash_scripts/eval_model_paths.sh`](bash_scripts/eval_model_paths.sh).
For known paper checkpoints, the resolver prefers the Hugging Face model id, falls back
to the corresponding local checkpoint path if the Hub model is unavailable, and errors
before launching evaluation if neither is available. Select a known checkpoint with
`EVAL_MODEL_KEY`, an HF id, or a local checkpoint path in `MODEL_PATH`:

```bash
# Prefer kuleshov-group/setdlm-gsm8k-smax32, fall back to the matching local run dir.
EVAL_MODEL_KEY=gsm8k:setdlm-d16 bash bash_scripts/run_lm_eval_harness.sh

# Explicit local paths are also resolved to their HF ids when available.
MODEL_PATH=/share/kuleshov/ma2238/runs/dllm-dev/<run-dir> bash bash_scripts/run_lm_eval_harness.sh

# Force local checkpoints, or skip the live Hub availability check when needed.
EVAL_MODEL_PREFER_LOCAL=true EVAL_MODEL_KEY=cnndm:setdlm-d8 bash bash_scripts/run_seq2seq_eval_cnndm.sh
EVAL_MODEL_SKIP_HF_CHECK=true EVAL_MODEL_KEY=gsm8k:setdlm-d16 bash bash_scripts/run_lm_eval_harness.sh
```

Dataset configs read from `DLLM_DATA_DIR` and default to `data/`. Set
`DLLM_DATA_DIR=/path/to/datasets` when cached datasets live elsewhere. Evaluation scripts
write outputs under `outputs/` by default and accept checkpoint-related overrides such as
`MODEL_PATH`, `EVAL_MODEL_KEY`, `CKPT_FILE` or `CKPT`, and `USE_EMA`. `LM1B_MODEL_KEY`
and `LM1B_MODEL_PATH` are accepted by the LM1B likelihood wrapper.

Evaluation scripts are provided for the following tasks:
- Text summarization: [`run_seq2seq_eval_cnndm.sh`](bash_scripts/run_seq2seq_eval_cnndm.sh)
- Mathematical reasoning: [`run_lm_eval_harness.sh`](bash_scripts/run_lm_eval_harness.sh), [`run_likelihood_eval_gsm8k.sh`](bash_scripts/run_likelihood_eval_gsm8k.sh)
- Likelihood estimation: [`run_likelihood_eval_owt.sh`](bash_scripts/run_likelihood_eval_owt.sh), [`run_likelihood_eval_lm1b.sh`](bash_scripts/run_likelihood_eval_lm1b.sh)
- Multiple-choice commonsense benchmarks (trained on OpenWebText): [`run_mcqa_eval_owt.sh`](bash_scripts/run_mcqa_eval_owt.sh)
- Infilling (trained on OpenWebText): [`run_seq2seq_eval_infill_nlp.sh`](bash_scripts/run_seq2seq_eval_infill_nlp.sh)
- Unconditional generation (trained on OpenWebText): [`run_uncond_gen_ppl_owt.sh`](bash_scripts/run_uncond_gen_ppl_owt.sh)

For full experiment sweeps, [`scripts/eval/repro_suite_runner.py`](scripts/eval/repro_suite_runner.py)
builds the evaluation matrix and uses the same HF-first checkpoint resolver, so generated
commands and expected output paths agree with the shell wrappers.

The MCQA evaluation flow uses Hydra overrides in the same style as the other eval
frameworks, for example `+eval/mcqa@task=all` or `+eval/mcqa@task=hellaswag`.
It evaluates the validation splits of HellaSwag, PIQA, and Social IQa with
continuation scoring rather than greedy generation, writes per-example predictions
and option scores to `predictions.json`, and saves aggregate accuracies plus a
summary table to `metrics.json` / `metrics.txt`. By default, answer options are
ranked by average log-probability per answer token to reduce length bias.

## 3. HuggingFace Integration
Evaluation checkpoints are mapped in [`scripts/push_hf_models.py`](scripts/push_hf_models.py).
The script converts the old notebook workflow into a reproducible command-line tool and
covers the 32 local checkpoint-backed models used by the evaluation matrix across GSM8K,
CNN/DM, OpenWebText, and LM1B.

```bash
# List every checkpoint that would be pushed, with HF repo id and local path.
python scripts/push_hf_models.py

# Filter the list, or resolve a key/path exactly as the eval scripts do.
python scripts/push_hf_models.py --only gsm8k
python scripts/push_hf_models.py --resolve gsm8k:setdlm-d16

# Push to Hugging Face. Repos are private by default; add --public for public repos.
python scripts/push_hf_models.py --yes
```

The GSM8K SetDLM release ids are:
- `kuleshov-group/setdlm-gsm8k-smax8`
- `kuleshov-group/setdlm-gsm8k-smax16`
- `kuleshov-group/setdlm-gsm8k-smax32`

The corresponding OWT, LM1B, and CNN/DM ids are:
- `kuleshov-group/owt-setdlm-smax8`, `kuleshov-group/owt-setdlm-smax16`, `kuleshov-group/owt-setdlm-smax32`
- `kuleshov-group/lm1b-setdlm-smax8`, `kuleshov-group/lm1b-setdlm-smax16`, `kuleshov-group/lm1b-setdlm-smax32`
- `kuleshov-group/cnndm-setdlm-smax8`, `kuleshov-group/cnndm-setdlm-smax16`, `kuleshov-group/cnndm-setdlm-smax32`

Other evaluated checkpoints use the `kuleshov-group/<dataset>-<model>` naming
scheme, for example `kuleshov-group/owt-bd3lm-s16`. The resolver accepts either
these HF ids or compact keys such as `cnndm:setdlm-d8`, `owt:bd3lm-s16`, and
`lm1b:ar`. For exact GSM8K SetDLM Pareto reproduction, use the repo evaluation
loader/scripts rather than plain `AutoModel.from_pretrained`, because the loader
normalizes legacy checkpoint config and eval-time SetDLM noise/cache-order settings.

## Citation
```
@article{arriola2026setdiffusion,
  title={Set Diffusion: Interpolating Token Orderings between Autoregression and Diffusion for Fast and Flexible Decoding},
  author={Marianne Arriola and Volodymyr Kuleshov},
  journal={arXix},
  year={2026},
  url={...}
}
```
