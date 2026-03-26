import logging
import os
import pathlib
import shutil
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import numpy as np
import torch
import wandb
from composer import TimeUnit
from composer.callbacks import CheckpointSaver
from composer.core import Callback, State, Time, Timestamp
from composer.loggers import Logger
from composer.utils import PartialFilePath, dist, get_save_filename

from src.utils import (
    fsspec_exists,
    save_pretrained_or_push_to_hub,
    snapshot_repo_to_tmp_dir,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)
__all__ = [
    "HuggingFaceCompatibleCheckpointing",
    "SaveBestCheckpointing",
    "LogGradientNorms",
    "LogGradientVariance",
]


class HuggingFaceCompatibleCheckpointing(CheckpointSaver):
    """A checkpoint callback that saves models in a manner in which one can
    `AutoModel.from_pretrained(<ckpt_path>)`.

    """

    def __init__(
        self,
        disable_hf: bool = False,
        save_local: bool = True,
        save_to_hub: bool = False,
        hub_repo_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_to_hub = save_to_hub and not disable_hf
        self.hub_repo_id = hub_repo_id
        if self.save_to_hub and hub_repo_id is None:
            raise ValueError("Saving to hub requires a hub repo id be provided.")
        self.save_local = save_local and not disable_hf
        self.disable_hf = disable_hf or not (self.save_to_hub or self.save_local)
        self.project_root = ""
        self.hf_filename = PartialFilePath(
            f"HF_{self.filename.filename.split('.pt')[0]}", self.filename.folder
        )
        if self.latest_filename is not None:
            self.latest_hf_filename = PartialFilePath(
                f"HF_{self.latest_filename.filename.split('.pt')[0]}",
                self.latest_filename.folder,
            )
        self.saved_hf_checkpoints: list[str] = []
        self.all_saved_hf_checkpoints_to_timestamp: dict[str, Timestamp] = {}
        # TODO: Leads to OSError device is busy when using tmpdir in /share/kuleshov

    def fit_start(self, state: State, logger: Logger) -> None:
        super().fit_start(state, logger)
        if dist.get_global_rank() == 0 and not self.disable_hf:
            self.project_root = snapshot_repo_to_tmp_dir(tmp_dir_exists_ok=True)
            log.info(f"Created tmp repo for HF checkpointing at {self.project_root}")
        dist.barrier()  # Holds all the ranks until repo snapshot is done

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()

        all_hf_checkpoints = []
        for (
            save_filename,
            timestamp,
        ) in self.all_saved_hf_checkpoints_to_timestamp.items():
            all_hf_checkpoints.append((save_filename, timestamp.state_dict()))

        state_dict["all_saved_hf_checkpoints_to_timestamp"] = all_hf_checkpoints
        return state_dict

    def load_state_dict(self, state: dict[str, Any]):
        super().load_state_dict(state)
        if "all_saved_hf_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state in state[
                "all_saved_hf_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_hf_checkpoints_to_timestamp[save_filename] = (
                    load_timestamp
                )

    def _save_checkpoint(self, state: State, logger: Logger) -> None:
        """
        Copied / adapted from composer.callbacks.CheckpointSaver._save_checkpoint
            for HF compatibility.
        """
        # TODO: Check that HF saving works with state.fsdp_sharded_state_dict_enabled
        #  (or if we can ignore this scenario).
        # TODO: Do we need to implement HF uploading for remote uploading too?
        if self.disable_hf:
            super()._save_checkpoint(state, logger)  # Perform standard checkpointing
            return

        hf_filename_with_placeholders = self.hf_filename.format(
            state, keep_placeholders=True
        )
        save_hf_filename = get_save_filename(state, hf_filename_with_placeholders)
        self.all_saved_hf_checkpoints_to_timestamp[save_hf_filename] = state.timestamp

        # Adapting `checkpoint.save_checkpoint / ._save_checkpoint` for HF
        saved_hf_path = None
        if dist.get_global_rank() == 0:
            if self.save_local:
                save_pretrained_or_push_to_hub(
                    model=(
                        state.model.module.model
                        if hasattr(state.model, "module")
                        else state.model.model
                    ),
                    tokenizer=(
                        state.model.module.tokenizer
                        if hasattr(state.model, "module")
                        else state.model.tokenizer
                    ),
                    repo_id=save_hf_filename,
                    local=True,
                    project_root=self.project_root,
                )
                saved_hf_path = save_hf_filename
                log.debug(f"HF checkpoint locally saved to {saved_hf_path}")
            if self.save_to_hub:
                metrics_str = "Train metrics:\n\t" + "\n\t".join(
                    [
                        f"{k}={v.item():0.4f}"
                        for k, v in state.train_metric_values.items()
                    ]
                )
                if hasattr(state, "eval_metric_values"):
                    metrics_str += "\n\nVal metrics:\n\t" + "\n\t".join(
                        [
                            f"{k}={v.item():0.4f}"
                            for k, v in state.eval_metric_values.items()
                        ]
                    )
                commit_message = (
                    f"Checkpoint @ Epoch {state.timestamp.epoch.value}, "
                    f"Batch {state.timestamp.batch.value}\n\n"
                    f"{metrics_str}\n\n"
                    f"Timestamp:\n"
                    f"\titeration={state.timestamp.iteration.value}\n"
                    f"\tepoch={state.timestamp.epoch.value}\n"
                    f"\tbatch={state.timestamp.batch.value}\n"
                    f"\tsample={state.timestamp.sample.value}\n"
                    f"\ttoken={state.timestamp.token.value}\n"
                    f"\tepoch_in_iteration={state.timestamp.epoch_in_iteration.value}\n"
                    f"\ttoken_in_iteration={state.timestamp.token_in_iteration.value}\n"
                    f"\tbatch_in_epoch={state.timestamp.batch_in_epoch.value}\n"
                    f"\tsample_in_epoch={state.timestamp.sample_in_epoch.value}\n"
                    f"\ttoken_in_epoch={state.timestamp.token_in_epoch.value}"
                )
                save_pretrained_or_push_to_hub(
                    model=(
                        state.model.module.model
                        if hasattr(state.model, "module")
                        else state.model.model
                    ),
                    tokenizer=(
                        state.model.module.tokenizer
                        if hasattr(state.model, "module")
                        else state.model.tokenizer
                    ),
                    repo_id=self.hub_repo_id,
                    local=False,
                    project_root=self.project_root,
                    commit_message=commit_message,
                )
            log.debug(f"HF checkpoint pushed to {self.hub_repo_id}")

        if not saved_hf_path:  # not all ranks save
            super()._save_checkpoint(state, logger)  # Perform standard checkpointing
            return

        self.rank_saves_symlinks = (
            dist.get_global_rank() == 0 or not state.fsdp_sharded_state_dict_enabled
        )
        if self.latest_hf_filename is not None and self.num_checkpoints_to_keep != 0:
            symlink = self.latest_hf_filename.format(state)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            # Sharded checkpoints for torch >2.0 use directories not files for
            # load_paths
            if state.fsdp_sharded_state_dict_enabled:
                src_path = str(pathlib.Path(saved_hf_path).parent)
            else:
                src_path = saved_hf_path
            if self.rank_saves_symlinks:
                os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)
        self.saved_hf_checkpoints.append(saved_hf_path)

        if self.num_checkpoints_to_keep >= 0:
            # Adapting `super().__rotate_checkpoints` for HF
            while len(self.saved_hf_checkpoints) > self.num_checkpoints_to_keep:
                checkpoint_to_delete = self.saved_hf_checkpoints.pop(0)
                prefix_dir = str(pathlib.Path(checkpoint_to_delete).parent)
                if not state.fsdp_sharded_state_dict_enabled:
                    shutil.rmtree(checkpoint_to_delete)
                else:
                    if dist.get_global_rank() == 0:
                        shutil.rmtree(prefix_dir)
        super()._save_checkpoint(state, logger)  # Perform standard checkpointing

    def close(self, state: State, logger: Logger) -> None:
        """Clean up tmp repo snapshot"""
        if dist.get_global_rank() == 0:
            # Only clean up if project_root was initialized (not empty string)
            if self.project_root and fsspec_exists(self.project_root):
                shutil.rmtree(self.project_root)
        dist.barrier()
        super().close(state, logger)


class SaveBestCheckpointing(HuggingFaceCompatibleCheckpointing):
    """Save the best checkpoint based on a metric."""

    def __init__(
        self,
        metric_to_monitor: str,
        mode: Literal["min", "max"] = "min",
        disable_hf: bool = False,
        save_local: bool = True,
        save_to_hub: bool = False,
        hub_repo_id: str | None = None,
        start: str = "0.0dur",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            disable_hf, save_local, save_to_hub, hub_repo_id, *args, **kwargs
        )

        self.metric_to_monitor = metric_to_monitor
        self.train_or_eval = metric_to_monitor.split("/")[0]
        self.metric_name = "/".join(metric_to_monitor.split("/")[1:])
        self.mode = mode
        self.best_value = None
        self.start = Time.from_timestring(start)
        self.started = False

        self.latest_filename = None
        self.latest_hf_filename = None

    @staticmethod
    def _validate_metric_name(metric_to_monitor: str) -> None:
        invalid_name = False
        split_name = metric_to_monitor.split("/")
        if len(split_name) < 2:
            invalid_name = True
        if split_name[0] not in ["train", "eval"]:
            invalid_name = True

        if invalid_name:
            raise ValueError(
                f"Invalid metric name {metric_to_monitor}. "
                "Expected format is <train|eval>/<metric_name>."
            )

    @property
    def _metric_dict(self) -> dict[str, Any]:
        return {
            "metric_to_monitor": self.metric_to_monitor,
            "mode": self.mode,
            "best_value": self.best_value,
        }

    def _should_start(self, state: State) -> bool:
        if self.start.unit == TimeUnit.DURATION:
            current_time = state.get_elapsed_duration()
            if current_time is not None:
                should_start = self.start <= current_time
            else:
                should_start = False
        else:
            current_time = state.timestamp.get(self.start.unit).value
            should_start = self.start.value <= current_time

        return should_start

    def _trigger_save(self, metric_value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return metric_value < self.best_value
        return metric_value > self.best_value  # self.mode == "max"

    def eval_end(self, state: State, logger: Logger) -> None:
        metrics = (
            state.train_metric_values
            if self.train_or_eval == "train"
            else state.eval_metric_values
        )
        if self.metric_name not in [k.lower() for k in metrics.keys()]:
            raise ValueError(
                f"Metric {self.metric_name} not found in metrics {metrics.keys()}"
            )
        metrics = {
            key.lower(): (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in metrics.items()
        }
        metric_value = metrics[self.metric_name]
        if self._trigger_save(metric_value):
            filename_with_placeholders = self.filename.format(
                state, keep_placeholders=True
            )
            save_filename = get_save_filename(state, filename_with_placeholders)
            log.info(
                f"Best {self.metric_name} attained: {metric_value}. "
                f"Saving checkpoint to {save_filename}."
            )
            self.best_value = metrics[self.metric_name]
            self._save_checkpoint(state, logger)

    def batch_checkpoint(self, state: State, logger: Logger):
        pass  # Force no saving

    def epoch_checkpoint(self, state: State, logger: Logger):
        pass

    def iteration_checkpoint(self, state: State, logger: Logger):
        pass

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()

        # Add best checkpoint info to state_dict
        all_checkpoints = []
        for save_filename, timestamp in self.all_saved_checkpoints_to_timestamp.items():
            all_checkpoints.append(
                (save_filename, timestamp.state_dict(), self._metric_dict)
            )

        state_dict["all_saved_checkpoints_to_timestamp"] = all_checkpoints
        return state_dict

    def load_state_dict(self, state: dict[str, Any]):
        if "all_saved_hf_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state in state[
                "all_saved_hf_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_hf_checkpoints_to_timestamp[save_filename] = (
                    load_timestamp
                )
        if "all_saved_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state, metrics_dict in state[
                "all_saved_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_checkpoints_to_timestamp[save_filename] = load_timestamp
                self.best_value = metrics_dict["best_value"]  # restore best_value


class LogGradientNorms(Callback):
    """Log gradient norms of model parameters."""

    def __init__(
        self,
        log_frequency: int = 1,
        include_embedding_params: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.log_frequency = log_frequency
        self.include_embedding_params = include_embedding_params
        self.step_count = 0

    def _is_embedding_param(self, param_name: str) -> bool:
        """Check if a parameter is an embedding parameter."""
        return "embed" in param_name.lower()

    def _get_model(self, state: State):
        """Get the model, handling wrapped models."""
        if hasattr(state.model, "module"):
            return state.model.module.model
        return state.model.model

    def after_backward(self, state: State, logger: Logger) -> None:
        """Log gradient norms after backward pass."""
        self.step_count += 1
        if self.step_count % self.log_frequency != 0:
            return

        model = self._get_model(state)
        metrics = {}

        total_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None and (
                self.include_embedding_params or not self._is_embedding_param(name)
            ):
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        # Log total gradient norm across all non-embedding parameters
        total_norm = total_norm ** (1.0 / 2)
        if self.include_embedding_params:
            metrics["grad_stats/norm"] = total_norm
        else:
            metrics["grad_stats/norm_non_embedding"] = total_norm

        if metrics:
            logger.log_metrics(metrics)


class LogGradientVariance(Callback):
    """Log variance over multiple gradient updates."""

    def __init__(
        self,
        accumulation_steps: int = 10,
        log_frequency: int = 1,
        include_embedding_params: bool = False,
        log_per_param_variance: bool = False,
        log_outlier_params: bool = False,
        outlier_top_k: int = 10,
        outlier_threshold_multiplier: float = 2.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps
        self.log_frequency = log_frequency
        self.include_embedding_params = include_embedding_params
        self.log_per_param_variance = log_per_param_variance
        self.log_outlier_params = log_outlier_params
        self.outlier_top_k = outlier_top_k
        self.outlier_threshold_multiplier = outlier_threshold_multiplier
        self.step_count = 0
        self.accumulated_grads: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _is_embedding_param(param_name: str) -> bool:
        """Check if a parameter is an embedding parameter."""
        return "embed" in param_name.lower()

    @staticmethod
    def _get_model(state: State):
        """Get the model, handling wrapped models."""
        if hasattr(state.model, "module"):
            return state.model.module.model
        return state.model.model

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """Accumulate gradients and log variance across steps."""
        self.step_count += 1
        if self.step_count % self.log_frequency != 0:
            return

        model = self._get_model(state)

        # Collect current gradients
        current_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None and (
                self.include_embedding_params or not self._is_embedding_param(name)
            ):
                current_grads[name] = param.grad.data.clone().detach().view(-1)

        # Accumulate gradients
        self.accumulated_grads.append(current_grads)

        # Check if we've accumulated enough steps
        if len(self.accumulated_grads) >= self.accumulation_steps:
            metrics = self._compute_variance_across_steps()
            if metrics:
                logger.log_metrics(metrics)
                # Also log per-parameter variances directly to wandb if enabled
                if self.log_per_param_variance and wandb.run is not None:
                    per_param_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k.startswith("grad_stats/variance_per_param/")
                    }
                    if per_param_metrics:
                        wandb.log(per_param_metrics)
            self.accumulated_grads.clear()

    def _compute_variance_across_steps(self) -> dict[str, float]:
        """Compute variance of gradients across accumulated steps."""
        if len(self.accumulated_grads) == 0:
            return {}
        metrics = {}
        param_names = set(self.accumulated_grads[0].keys())
        all_gradients = torch.stack(
            [
                torch.cat([step_grads[name] for name in param_names], dim=0)
                for step_grads in self.accumulated_grads
            ],
            dim=0,
        )

        total_variance = torch.norm(
            all_gradients - all_gradients.mean(dim=0), p=2, dim=1
        ).pow(2)
        total_variance = total_variance.sum() / (all_gradients.shape[0] - 1)

        if self.include_embedding_params:
            metrics["grad_stats/variance"] = total_variance.item()
        else:
            metrics["grad_stats/variance_non_embedding"] = total_variance.item()

        # Optionally compute and log per-parameter variances
        if self.log_per_param_variance:
            for param_name in param_names:
                # Stack gradients for this parameter across all steps
                param_gradients = torch.stack(
                    [step_grads[param_name] for step_grads in self.accumulated_grads],
                    dim=0,
                )
                # Compute variance: mean squared deviation from mean
                param_mean = param_gradients.mean(dim=0)
                param_variance = (
                    torch.norm(param_gradients - param_mean, p=2, dim=1)
                    .pow(2)
                    .mean()
                    .item()
                )

                # Sanitize parameter name for metric key (replace dots and slashes)
                sanitized_name = param_name.replace(".", "/").replace("\\", "/")
                metrics[f"grad_stats/variance_per_param/{sanitized_name}"] = (
                    param_variance
                )

        # Optionally identify and log outlier parameters contributing to high variance
        if self.log_outlier_params:
            param_contributions = {}
            # Compute each parameter's contribution to total variance
            for param_name in param_names:
                # Stack gradients for this parameter across all steps
                param_gradients = torch.stack(
                    [step_grads[param_name] for step_grads in self.accumulated_grads],
                    dim=0,
                )
                # Compute variance contribution: norm of deviations from mean
                param_mean = param_gradients.mean(dim=0)
                param_variance = torch.norm(
                    param_gradients - param_mean, p=2, dim=1
                ).pow(2)
                # Contribution is the sum of squared norms across steps
                contribution = param_variance.sum().item() / (
                    all_gradients.shape[0] - 1
                )
                param_contributions[param_name] = contribution

            if param_contributions:
                # Sort by contribution (descending)
                sorted_contributions = sorted(
                    param_contributions.items(), key=lambda x: x[1], reverse=True
                )

                # Compute statistics for outlier detection
                contributions_array = np.array(list(param_contributions.values()))
                mean_contribution = np.mean(contributions_array)
                std_contribution = np.std(contributions_array)
                threshold = (
                    mean_contribution
                    + self.outlier_threshold_multiplier * std_contribution
                )

                # Get top K outliers
                top_k_outliers = sorted_contributions[: self.outlier_top_k]

                # Log outlier parameters
                for rank, (param_name, contribution) in enumerate(top_k_outliers, 1):
                    sanitized_name = param_name.replace(".", "/").replace("\\", "/")
                    # Log contribution value
                    metrics[f"grad_stats/outlier_contribution/{sanitized_name}"] = (
                        contribution
                    )
                    # Log percentage contribution relative to total variance
                    if total_variance.item() > 0:
                        pct_contribution = (
                            contribution / total_variance.item()
                        ) * 100.0
                        metrics[
                            f"grad_stats/outlier_contribution_pct/{sanitized_name}"
                        ] = pct_contribution

                # Log summary statistics
                metrics["grad_stats/outlier_mean_contribution"] = mean_contribution
                metrics["grad_stats/outlier_threshold"] = threshold

                # Log how many parameters exceed threshold
                n_outliers = sum(
                    1
                    for _, contrib in param_contributions.items()
                    if contrib > threshold
                )
                metrics["grad_stats/n_outliers"] = n_outliers

        return metrics
