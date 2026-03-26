# Set Diffusion: Interpolating Token Orderings between Autoregression and Diffusion for Fast and Flexible Decoding

[![deploy](https://img.shields.io/badge/Paper_📃-green)](https://arxiv.org/abs/2510.22852)
[![deploy](https://img.shields.io/badge/Blog_📝%20%20-8A2BE2)](https://m-arriola.com/e2d2)
[![deploy](https://img.shields.io/badge/HuggingFace_🤗%20-E2D2%20-orange)](https://huggingface.co/collections/kuleshov-group/e2d2)


## 0. Setup

### Setup environment

Install conda:

```bash
# For conda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
bash miniconda.sh -b -p /opt/conda
```

Setup a conda environment and install dependencies using:

Activate the environment:

```bash
conda activate dllm-dev
```

We also include a [`setup_env.sh`](./setup_env.sh) script that can be used to set up the
environment on a new machine.
Run the script using:
```bash
source setup_env.sh
```

You can also include this snippet in shell / slurm scripts to set up the environment on
a compute node.

In this script, we set up WandB and HuggingFace tokens by sourcing a script which is
expected to be in the `/home/<YOUR_USER_NAME>/` directory.
Copy the contents below into a shell script `/home/<YOUR_USER_NAME>/setup_discdiff.sh`
and replace the placeholder tokens with your own:
```shell
# W&B / HF Setup
export WANDB__SERVICE_WAIT=600
export _WANDB_STARTUP_DEBUG="true"
export WANDB_ENTITY="kuleshov-group"
export WANDB_API_KEY="<WANDB_API_KEY>"
echo "Logging into W&B as '${WANDB_ENTITY}'."

# HF Setup
export HUGGINGFACE_TOKEN="<HF_TOKEN>"
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```
- WandB token can be found [here](https://wandb.ai/authorize).
- HuggingFace token can be setup [here](https://huggingface.co/settings/tokens).

## 1. Code Organization
1. [`bash_scripts`](bash_scripts): These shells scripts can be used to reproduce the
experiments from our work.
2. [`configs`](configs): We utilize hydra config files to organize experiments.
   1. [`config.yaml`](configs/config.yaml) This config is the entry point for launching
   training experiments.
   2. [`eval_config.yaml`](configs/eval_config.yaml) This config is the entry point for
   evaluations.
3. [`scripts`](scripts): The main training and evaluation scripts
   1. [`scripts/composer_scripts/train_discrete_denoiser.py`](scripts/composer_scripts/train_discrete_denoiser.py):
   This script is the main training entry point.
   2. [`scripts/evals`](scripts/eval): These scripts run the evaluation for the
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
   2. [`src/backbone`](src/backbone): These are the underlying neural networks the take
   in noisy inputs and produce logits.
   Each denoiser is parameterized by a backbone.
   The denoiser can optionally, post-process the logit outputs of the backbone to
   produce log-probs over the clean sequence.

## 2. Reproducing Experiments
The shell scripts provided in [`bash_scripts`](bash_scripts) can be used to reproduce
the training and evaluations from our work.
- For training, the files follow a convention where the dataset and denoiser class are
specified.
For example, to train SetDLM on the GSM8K dataset, use
[`run_train_setdlm_gsm8k.sh`](bash_scripts/run_train_setdlm_gsm8k.sh).
- Once models have been trained, the provided evaluation scripts can be used to reproduce
tables and figures from our work.
For example, to evaluate models trained on the GSM8K dataset, use
[`run_lm_eval_harness.sh`](bash_scripts/run_lm_eval_harness.sh) (or the `*_tput` / `*_intm` variants).
In that file, and similar ones for other evaluations, specify the path to the saved
checkpoints, and uncomment the relevant section for a given denoiser class.
We also provide scripts that will produce the generation throughput numbers we report.
These files contain a `_tput` at the end of the script name.

Below are the evaluation scripts provided for various tasks:
- Text summarization: [`run_seq2seq_eval_cnndm.sh`](bash_scripts/run_seq2seq_eval_cnndm.sh), [`run_seq2seq_eval_cnndm_tput.sh`](bash_scripts/run_seq2seq_eval_cnndm_tput.sh)
- Mathematical reasoning: [`run_lm_eval_harness.sh`](bash_scripts/run_lm_eval_harness.sh), [`run_lm_eval_harness_tput.sh`](bash_scripts/run_lm_eval_harness_tput.sh), [`run_likelihood_eval_gsm8k.sh`](bash_scripts/run_likelihood_eval_gsm8k.sh)
- Likelihood estimation [`run_likelihood_eval_owt.sh`](bash_scripts/run_likelihood_eval_owt.sh), [`run_likelihood_eval_lm1b.sh`](bash_scripts/run_likelihood_eval_lm1b.sh)
- Infilling (trained on OpenWebText): [`run_seq2seq_eval_infill_nlp.sh`](bash_scripts/run_seq2seq_eval_infill_nlp.sh)
- Unconditional generation (trained on OpenWebText): [`run_uncond_gen_ppl_owt.sh`](bash_scripts/run_uncond_gen_ppl_owt.sh)

## 3. HuggingFace Integration
We release the following SetDLMs (s ≤ 8, 16, 32) on HuggingFace:
- 1.7B SetDLMs for text summarization (distilled on GSM8K from Qwen3 1.7B):
[`kuleshov-group/..`](..)
- 80M SetDLMs for text summarization (trained on CNN/DM from scratch):
[`kuleshov-group/..`](..)
- 110M SetDLMs for likelihood estimation / infilling (trained on OWT from scratch):
[`kuleshov-group/..`](..)

## Citation
```
@inproceedings{
arriola2026setdiffusion,
title={Set Diffusion: Interpolating Token Orderings between Autoregression and Diffusion for Fast and Flexible Decoding},
author={Marianne Arriola and Volodymyr Kuleshov},
booktitle={arXiv},
year={2026},
url={..}
}
```
