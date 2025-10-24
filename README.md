# dllm-dev
Internal repo for iteration on Diffusion LLMs


## 0. Setup

### Provision hardware

If necessary, provision accelerator-enabled VMs with [SkyPilot](https://docs.skypilot.co/en/latest/).

For Lambda, e.g., this is all it takes to create a single A100 node for development:

```bash
pip install skypilot[lambda]
sky launch --cluster dllm --gpus A100
ssh dllm # sky creates ssh configs for you
```

SkyPilot can also provision clusters, setup environments, manage task execution and some
other useful stuff.
See [docs/skypilot.md](docs/skypilot.md) for more details.

### Setup environment

Install mamba or conda (mamba is far faster):

```bash
# For mamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#umamba-install
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# For conda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
bash miniconda.sh -b -p /opt/conda
```

Setup a conda environment and install dependencies using:

```bash
micromamba env create -y -f requirements.yaml --channel-priority flexible
```

Activate the environment:

```bash
conda activate dllm-dev
# OR micromamba activate dllm-dev
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

### Contributing to the repo
We will try to use GitHub issues to track bugs, features, and todos.
To contribute to the repo, please create a new issue and assign it to yourself.
Then [create a new branch from the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-a-branch-for-an-issue)
and open a pull request.


We use [pre-commit](https://pre-commit.com/) to run linters and formatters on the code.
To install the pre-commit hooks, run:

```bash
pre-commit install
```
On every `git commit`,
the pre-commit hooks will run automatically and report any issues / automatic fixes that
were applied.

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
         - `E2D2`: Our encoder-decoder implementation.
   2. [`src/backbone`](src/backbone): These are the underlying neural networks the take
   in noisy inputs and produce logits.
   Each denoiser is parameterized by a backbone.
   The denoiser can optionally, post-process the logit outputs of the backbone to
   produce log-probs over the clean sequence.
