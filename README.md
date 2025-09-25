# dllm-dev
Internal repo for iteration on Diffusion LLMs


## 0. Setup

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

In this script, we setup WandB and HuggingFace tokens by sourcing a script which is
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
We will try to use github issues to track bugs, features, and todos.
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

## 1. Training

### tl;dr
Run the composer training script using:
```bash
composer -n <num_devices> scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=<run_name> \
  tokenizer.pretrained_model_name_or_path=<pretrained_tokenizer> \
  dataset@train_dataset=<train_dataset> \
  dataset@eval_dataset=<eval_dataset> \
  model/backbone@model.config.backbone_config=<backbone>
```
Filling in the relevant variables.

---

Experiment configs are setup
using [Hydra](https://hydra.cc/docs/intro/) and can be found in the
[`configs`](./configs) directory.

The main config is [`configs/config.yaml`](./configs/config.yaml).
Here we use `hydra` defaults list to setup the experiment / run, but all of these can be
changed using `hydra`'s command line overrides.
Any parameters
set to `???` need to be filled in by the user via the command-line overrides.

We leverage `hydra`'s useful yaml syntax to setup the config, e.g., using `@` to
move parameters to different levels of the config hierarchy.

As an example,
if you want to change the `backbone` for the denoising model,
to use some pre-trained HuggingFace model,
you can do so with the following command line overrides:
```bash
model/backbone@model.config.backbone_config=automodel_for_masked_lm \
pretrained_model_name_or_path=bert-base-uncased
```
This will set the model backbone to the one defined in
[`automodel_for_masked_lm.yaml`](configs/model/backbone/automodel_for_masked_lm.yaml)
which will use `hydra.utils.instantiate` tools to initialize the backbone and the `@`
syntax will move this parameter to `config.model.config.backbone_config`
and set the `pretrained_model_name_or_path` to `bert-base-uncased`.

Another example,
if you want to remove something from the defaults list, use `~` syntax, e.g.:
```bash
~composer.trainer.parallelism_config
```
This will remove the `parallelism_config` from the defaults defined in the composer
config [`configs/composer/default_composer`](./configs/composer/default_composer.yaml).

## Tour of the codebase
TODO: Fill his in
