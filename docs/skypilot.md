## SkyPilot

These examples show how to use [SkyPilot](https://docs.skypilot.co/en/latest/) to deploy development nodes or clusters.  

All examples assume [Lambda](https://lambda.ai/) as the cloud provider, but many others are possible: https://docs.skypilot.co/en/latest/overview.html#cloud-vms.

### Contents

- [Deploying a development node](#deploying-a-development-node)
- [Configure a development node](#configure-a-development-node)
- [Run training on a single node](#run-training-on-a-single-node)
- [Miscellaneous](#miscellaneous)

### Deploying a development node

This is the simplest use case where you just want a node with a single GPU.  You can pick any directory, but typically a GitHub repo root, and do this:

```bash
# Configure Lambda access: https://docs.skypilot.co/en/latest/getting-started/installation.html#lambda-cloud
# Go here https://cloud.lambda.ai/api-keys/cloud-api and create an API key
mkdir -p ~/.lambda_cloud
echo "api_key = <your_api_key_here>" > ~/.lambda_cloud/lambda_keys

# Pin uvicorn version bound to avoid: https://github.com/skypilot-org/skypilot/issues/7287
pip install skypilot[lambda] uvicorn==0.35.0

# Deploy a single A100 node
sky launch --infra lambda --cluster dllm --gpus A100 --disk-size 100
```

Then you can ssh to the node via `ssh dllm`, or whatever "cluster name" you chose.  If your cluster is more than one node, the rest will be named like `dllm-worker1`, `dllm-worker2`, etc. as hosts in your SSH config.

See also:

- [SkyPilot Installation](https://docs.skypilot.co/en/v0.9.3/getting-started/installation.html)
- [SkyPilot CLI](https://docs.skypilot.co/en/latest/getting-started/cli.html)
- [GPUs and Accelerators](https://docs.skypilot.co/en/v0.9.3/compute/gpus.html)

### Configure a development node

The above is example is fine, except it doesn't take care of creating an environment, setting env vars, etc.  Here's an example of how to at least create an environment:

```bash
cat << 'EOF' > cluster.sky.yaml
resources:
  cloud: lambda
  # Use cheaper of A100s or H100s, whatever is available
  accelerators: ["A100:8", "H100:8"]
  disk: 100 # GB
setup: |
  # Create dllm-dev conda environment
  conda env create -f requirements.yaml
EOF
```

Then you can launch the cluster with:

```bash
sky launch --cluster dllm cluster.sky.yaml

# Once that's done:
ssh dllm
conda deactivate # deactivate base conda env skypilot creates
conda activate dllm-dev # activate dllm-dev env
```

### Run training on a single node

Building on the example above in [Configure a development node](#configure-a-development-node), you can now run training with some extra initial cluster setup:

```bash
cat << 'EOF' > cluster.sky.yaml
# Use these env vars fromyour LOCAL machine so you only ever
# have to set them in one place
envs:
  HUGGING_FACE_HUB_TOKEN: null
  WANDB_API_KEY: null

# This determines the local dir for /home/ubuntu/sky_workdir,
# which is where `setup` runs below
workdir: .  

setup: |

  # Create dllm-dev conda environment
  conda create -f requirements.yaml

  # Create local env for future jobs and ssh sessions
  > ~/.env # Clear first
  
  # Add tokens/secrets from client env
  for var in HUGGING_FACE_HUB_TOKEN WANDB_API_KEY; do
    declare -n ref=$var
    echo "$var=$ref" >> ~/.env
  done

  # Add other project-specific configuration
  cat << EOF >> ~/.env
  WANDB__SERVICE_WAIT=600
  _WANDB_STARTUP_DEBUG=true
  WANDB_ENTITY=kuleshov-group
  HF_HOME=${PWD}/.hf_cache
  PYTHONPATH=${PWD}:${PWD}/.hf_cache/modules
  HYDRA_FULL_ERROR=1
  NCCL_P2P_LEVEL=NVL
  EOF

  # Source ~/.env on login
  if ! grep -q "set -a; source ~/.env; set +a" ~/.bashrc; then
    echo "set -a; source ~/.env; set +a" >> ~/.bashrc
  fi
EOF
```

Notably, this will take your W&B and HuggingFace tokens from your local machine and set them in the cluster so that you don't ever have to copy these around manually in repo-specific files (usually what you want).  They will be set if you ssh in to a cluster node as well.  Then, training can be run this way:

```bash
# Launch the single-node cluster
sky launch --cluster dllm cluster.sky.yaml

# Define the task to run
cat << 'EOF' > task.sky.yaml
workdir: .
run: |
  set -exo pipefail

  conda deactivate
  conda activate dllm-dev

  python -c "import torch; assert torch.cuda.is_available()"

  export RUN_DIR=outputs
  export DATA_DIR=data
  export NUM_VISIBLE_DEVICES=8
  bash bash_scripts/run_train_e2d2_wmt_lambda.sh
EOF

# Run the task
sky exec -c dllm task.sky.yaml
```

See also:

- [SkyPilot YAML Spec](https://docs.skypilot.co/en/v0.9.3/reference/yaml-spec.html)
- [Run job on existing cluster](https://docs.skypilot.co/en/v0.5.0/getting-started/quickstart.html#execute-a-task-on-an-existing-cluster)

### FAQ

How do I shut a cluster down?

```bash
sky down dllm
```

---

How do I reset SkyPilot state? 

This is very important when jumping around between projects or when using different versions of SkyPilot installed locally.  There is no good reason not to do this frequently, aside from certain cloud providers that require global sky configurations in `~/.sky/config.yaml` (e.g. [kubernetes](https://docs.skypilot.co/en/latest/reference/kubernetes/kubernetes-getting-started.html#launching-your-first-task)).

```bash
sky api stop; [ -d ~/.sky ] && rm -rf ~/.sky
```

---

How do I check if Lambda is setup correctly?

```bash
sky check lambda
```

---

How do I cancel a task that is running?

Normally ctrl-c disconnects streaming logs without cancel a task.  To do that manually, use:

```bash
# Show all running jobs
> sky queue
Fetching and parsing job queue...
Fetching job queue for: dllm

Job queue of current user on cluster dllm
ID  NAME  USER    SUBMITTED    STARTED      DURATION  RESOURCES   STATUS     LOG                                        GIT COMMIT
10  -     eczech  11 mins ago  11 mins ago  11m 19s   1x[CPU:1+]  RUNNING    ~/sky_logs/sky-2025-09-22-19-51-11-954378  e610162171a5eaaff76d7c6b31074f4ac9fbadf8
9   -     eczech  18 mins ago  18 mins ago  4m 36s    1x[CPU:1+]  FAILED     ~/sky_logs/sky-2025-09-22-19-44-00-948488  e610162171a5eaaff76d7c6b31074f4ac9fbadf8
8   -     eczech  2 hrs ago    2 hrs ago    9m 59s    1x[CPU:1+]  FAILED     ~/sky_logs/sky-2025-09-22-17-12-50-492436  e610162171a5eaaff76d7c6b31074f4ac9fbadf8
7   -     eczech  2 hrs ago    2 hrs ago    < 1s      1x[CPU:1+]  SUCCEEDED  ~/sky_logs/sky-2025-09-22-17-11-42-124874  e610162171a5eaaff76d7c6b31074f4ac9fbadf8

# Cancel the job you want to finish
> sky cancel dllm 10
```

Confusingly, there are [Cluster Jobs](https://docs.skypilot.co/en/v0.9.3/reference/job-queue.html) and then there are [Managed Jobs](https://docs.skypilot.co/en/v0.9.3/examples/managed-jobs.html) in SkyPilot.  The example above shows how to cancel a "Cluster Job".  The "Managed Jobs" are not purely user-defined like the "Cluster Jobs" and SkyPilot tries to add features around them for pipelining jobs together and handling preemptions.  There is a special `sky jobs` command for the "Managed Jobs".

---

How do I show all GPUs of a certain type available on Lambda?

```
sky show-gpus A100 --infra lambda
GPU   QTY  CLOUD   INSTANCE_TYPE     DEVICE_MEM  vCPUs  HOST_MEM  HOURLY_PRICE  HOURLY_SPOT_PRICE  REGION
A100  1.0  Lambda  gpu_1x_a100       40GB        30     200GB     $ 1.290       -                  europe-central-1
A100  1.0  Lambda  gpu_1x_a100_sxm4  40GB        30     200GB     $ 1.290       -                  europe-central-1
A100  2.0  Lambda  gpu_2x_a100       40GB        60     400GB     $ 2.580       -                  europe-central-1
A100  4.0  Lambda  gpu_4x_a100       40GB        120    800GB     $ 5.160       -                  europe-central-1
A100  8.0  Lambda  gpu_8x_a100       40GB        124    1800GB    $ 10.320      -                  europe-central-1

GPU        QTY  CLOUD   INSTANCE_TYPE          DEVICE_MEM  vCPUs  HOST_MEM  HOURLY_PRICE  HOURLY_SPOT_PRICE  REGION
A100-80GB  8.0  Lambda  gpu_8x_a100_80gb_sxm4  80GB        240    1800GB    $ 14.320      -                  europe-central-1
```