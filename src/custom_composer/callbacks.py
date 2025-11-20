import gc
import glob
import ipdb
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from composer.callbacks import CheckpointSaver
from composer.core import Callback, State, Time, Timestamp
from composer.loggers import Logger
from composer.utils import PartialFilePath, dist, get_save_filename

from src.utils import (
    fsspec_exists,
    save_pretrained_or_push_to_hub,
    snapshot_repo_to_tmp_dir,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "ERROR"))
log = logging.getLogger(__name__)
__all__ = [
    "DataloaderSpeedMonitor",
    "LogGradientNorms",
    "MaskingPatternLossAnalysis",
    "PermutationOrderLossAnalysis",
    "PermutationToMaskingPatternAnalysis",
]


class DataloaderSpeedMonitor(Callback):
    """Measure how long it takes to return a batch from the dataloader.

    Copied from:
        https://github.com/AnswerDotAI/ModernBERT/blob/main/src/callbacks/dataloader_speed.py
        Copyright 2024 onwards Answer.AI, LightOn, and contributors
        License: Apache-2.0
    """  # noqa: E501

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_time = time.time_ns()

    def after_dataloader(self, state: State, logger: Logger) -> None:
        self.batch_serve_time = time.time_ns() - self.batch_start_time
        logger.log_metrics(
            {
                "throughput/batch_serve_time_ns": self.batch_serve_time,
                "throughput/batch_serve_time_ms": self.batch_serve_time / 1e6,
            }
        )


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
                    model=state.model.module.model
                    if hasattr(state.model, "module")
                    else state.model.model,
                    tokenizer=state.model.module.tokenizer
                    if hasattr(state.model, "module")
                    else state.model.tokenizer,
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
                    model=state.model.module.model
                    if hasattr(state.model, "module")
                    else state.model.model,
                    tokenizer=state.model.module.tokenizer
                    if hasattr(state.model, "module")
                    else state.model.tokenizer,
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


class LogProfilingTraceToWandb(Callback):
    """Log profiling trace to wandb.

    See: https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
    """  # noqa: E501

    def __init__(
        self,
        composer_prof_folder: str,
        torch_prof_folder: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.composer_prof_folder = composer_prof_folder
        self.torch_prof_folder = torch_prof_folder

    def fit_end(self, state: State, logger: Logger) -> None:
        if dist.get_global_rank() == 0:
            profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
            for trace_file in glob.glob(f"{self.composer_prof_folder}/*.json"):
                profile_art.add_file(trace_file, os.path.basename(trace_file))
            for trace_file in glob.glob(f"{self.torch_prof_folder}/*.pt.trace.json"):
                profile_art.add_file(trace_file, os.path.basename(trace_file))
            wandb.run.log_artifact(profile_art).wait()
        dist.barrier()


class LogSampledTimestep(Callback):
    def batch_end(self, state: State, logger: Logger) -> None:
        sampled_t = state.batch.t
        logger.log_metrics(
            {
                f"sampled_t/{state.dataloader_label}/mean": sampled_t.mean().item(),
                f"sampled_t/{state.dataloader_label}/std": sampled_t.std().item(),
                f"sampled_t/{state.dataloader_label}/max": sampled_t.max().item(),
                f"sampled_t/{state.dataloader_label}/min": sampled_t.min().item(),
            },
        )

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        sampled_t = state.batch["t"]
        logger.log_metrics(
            {
                f"sampled_t/{state.dataloader_label}/mean": sampled_t.mean().item(),
                f"sampled_t/{state.dataloader_label}/std": sampled_t.std().item(),
                f"sampled_t/{state.dataloader_label}/max": sampled_t.max().item(),
                f"sampled_t/{state.dataloader_label}/min": sampled_t.min().item(),
            },
        )


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
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps
        self.log_frequency = log_frequency
        self.include_embedding_params = include_embedding_params
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
        return metrics

class WarmupWithFrozenEncoder(Callback):
    def __init__(
        self,
        num_warmup_steps: str,
    ):
        super().__init__()
        self.num_warmup_steps = Time.from_timestring(num_warmup_steps)
        self.encoder_is_frozen = False

    def fit_start(self, state: State, logger: Logger) -> None:
        if hasattr(state.model, "module"):
            if not hasattr(state.model.module.model.backbone, "freeze_encoder"):
                raise NotImplementedError(
                    "Model backbone does not have freeze_encoder implemented."
                )
            state.model.module.model.backbone.freeze_encoder()
        else:
            if not hasattr(state.model.model.backbone, "freeze_encoder"):
                raise NotImplementedError(
                    "Model backbone does not have freeze_encoder implemented."
                )
            state.model.model.backbone.freeze_encoder()
        self.encoder_is_frozen = True

    def batch_end(self, state: State, logger: Logger) -> None:
        current_time = state.timestamp.get(self.num_warmup_steps.unit).value
        if self.encoder_is_frozen and self.num_warmup_steps.value >= current_time:
            if hasattr(state.model, "module"):
                state.model.module.model.backbone.unfreeze_encoder()
            else:
                state.model.model.backbone.unfreeze_encoder()
            self.encoder_is_frozen = False


class MaskingPatternLossAnalysis(Callback):
    """Callback to analyze loss by masking pattern within blocks.
    
    For each datapoint, records the loss for every single masking pattern
    (within each block). At the end of validation, reports the average loss
    for each masking pattern across all examples and blocks.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Store all loss values for each masking pattern
        self.pattern_stats: dict[tuple, list[float]] = {}

    def _get_model(self, state: State):
        """Get the model, handling wrapped models."""
        if hasattr(state.model, "module"):
            return state.model.module.model
        return state.model.model

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        """Record masking patterns and losses for each block."""
        # Only run during validation
        if state.dataloader_label == "train":
            return
        
        # Get model
        model = self._get_model(state)
        
        # Check if model has block_size config
        if not hasattr(model.config, "block_size") and not hasattr(
            model.config, "eval_block_size"
        ):
            log.warning(
                "Model does not have block_size config. Skipping masking pattern analysis."
            )
            return
        
        block_size = getattr(
            model.config, "eval_block_size", getattr(model.config, "block_size", None)
        )
        if block_size is None:
            log.warning("Could not determine block_size. Skipping masking pattern analysis.")
            return
        
        # Get batch data
        batch = state.batch
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return
        
        # Get model outputs
        outputs = state.outputs
        if outputs is None:
            return
        
        # Extract nlls from outputs (per-token losses)
        nlls = getattr(outputs, "other_loss_terms", {}).get("log_p_theta", None)
        if nlls is None:
            log.warning("Could not find nlls in outputs. Skipping this batch.")
            return
        
        # Get attention mask and tokens_mask
        attention_mask = batch.get("attention_mask")
        
        # Get context_mask from batch
        context_mask = batch.get("context_mask")
        
        # Get mask_token_id
        mask_token_id = model.mask_token_id

        masked_tokens = getattr(outputs, "other_loss_terms", {}).get("masked_tokens")
        if masked_tokens is None:
            log.warning("Could not find masked_tokens in outputs. Skipping this batch.")
            return
        
        # Check if model is BD3LM or similar that concatenates x0 and xt
        from src.denoiser.diffusion import BD3LM, E2D2, AnyOrderBD3LM
        
        is_bd3lm = isinstance(model, (BD3LM, E2D2))
        if isinstance(model, AnyOrderBD3LM):
            is_bd3lm = False
        
        if is_bd3lm:
            batch_size, full_seq_len = masked_tokens.shape
            seq_len = full_seq_len // 2
            masked_tokens_xt = masked_tokens[:, seq_len:]
            
            # Adjust context_mask if needed
            if context_mask is not None and context_mask.shape[1] > seq_len:
                context_mask = context_mask[:, seq_len:]
        else:
            # For MDLM models, masked_tokens has shape (B, L)
            masked_tokens_xt = masked_tokens
            seq_len = masked_tokens.shape[1]
        

        if context_mask is not None and context_mask.shape[1] != seq_len:
            context_mask = context_mask[:, :seq_len]
        
        # Extract patterns per block
        batch_size = masked_tokens_xt.shape[0]
        n_blocks = seq_len // block_size
        
        for b in range(batch_size):
            for block_idx in range(n_blocks):
                start_idx = block_idx * block_size
                end_idx = start_idx + block_size
                
                block_pattern = masked_tokens_xt[b, start_idx:end_idx]
                
                # Get block masks
                block_tokens_mask = attention_mask[b, start_idx:end_idx]
                context_tokens_mask = context_mask[b, start_idx:end_idx]
                if (block_tokens_mask == 0).any() or (context_tokens_mask == 1).any(): # Padding tokens, skip
                    continue

                # Convert to tuple for hashing (move to CPU and convert to avoid keeping GPU tensors)
                block_pattern_cpu = block_pattern.cpu()
                pattern = tuple(block_pattern_cpu.int().tolist())


                if sum(pattern) == 0: # No masked tokens, skip
                    continue
                
                # Extract corresponding loss for this block
                block_nlls = nlls[b, start_idx:end_idx]
                block_avg_loss = (block_nlls.mean() * (block_size / block_pattern.sum())).item()

                if pattern not in self.pattern_stats:
                    self.pattern_stats[pattern] = []
                self.pattern_stats[pattern].append(block_avg_loss)

    def eval_end(self, state: State, logger: Logger) -> None:
        """Report average loss per masking pattern."""
        # Gather data from all ranks for distributed evaluation
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                # Gather pattern_stats from all ranks
                gathered_pattern_stats = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_pattern_stats, self.pattern_stats)

                # Combine statistics from all ranks
                combined_pattern_stats = {}
                for rank_stats in gathered_pattern_stats:
                    for pattern, loss_values in rank_stats.items():
                        if pattern not in combined_pattern_stats:
                            combined_pattern_stats[pattern] = []
                        combined_pattern_stats[pattern].extend(loss_values)
                
                self.pattern_stats = combined_pattern_stats
                dist.barrier()
        
        if not self.pattern_stats:
            log.warning("No masking patterns recorded. Skipping report.")
            return

        # Combine _all_ losses from all ranks / patterns
        combined_nll_stats = []
        for pattern, loss_values in self.pattern_stats.items():
            combined_nll_stats.extend(loss_values)
        combined_nll_stats = np.array(combined_nll_stats)
        log.info(f"Final NLL: {combined_nll_stats.mean():.4f}±{combined_nll_stats.std():.4f}")

        # Compute average loss and std from collected values
        pattern_avg_losses = {}
        pattern_std_losses = {}
        pattern_counts = {}
        for pattern, loss_values in self.pattern_stats.items():
            loss_array = np.array(loss_values)
            pattern_avg_losses[pattern] = float(np.mean(loss_array))
            if len(loss_values) > 1:
                pattern_std_losses[pattern] = float(np.std(loss_array, ddof=1))
            else:
                pattern_std_losses[pattern] = 0.0
            pattern_counts[pattern] = len(loss_values)
        
        # Sort patterns by number of masked tokens (ascending: least masking to most masking)
        # Then by position of 0's (more 0's on the left ranked higher)
        sorted_patterns = sorted(
            pattern_avg_losses.items(),
            key=lambda x: (sum(x[0]), x[0]),  # First by number of masked tokens, then by pattern (0's on left rank higher)
        )
        
        # Log metrics for each pattern
        # Convert pattern tuple to string for logging
        metrics = {}
        for pattern, avg_loss in sorted_patterns:
            # Convert pattern to a readable string
            # Pattern is a tuple of 0s and 1s, where 1 = masked
            pattern_str = "".join(str(int(x)) for x in pattern)
            count = pattern_counts[pattern]
            std_loss = pattern_std_losses[pattern]
            metrics[f"masking_pattern_loss/{pattern_str}"] = avg_loss
            metrics[f"masking_pattern_loss_std/{pattern_str}"] = std_loss
            metrics[f"masking_pattern_count/{pattern_str}"] = count
        
        logger.log_metrics(metrics)
        
        # Create bar chart with error bars and log summary (only on rank 0)
        if dist.get_global_rank() == 0:
            self._plot_masking_patterns(
                pattern_avg_losses, pattern_std_losses, sorted_patterns, pattern_counts, logger
            )
            self._plot_masking_pattern_frequencies(
                sorted_patterns, pattern_counts, logger
            )
            
            # Also log a summary
            log.info("=" * 80)
            log.info("Masking Pattern Loss Analysis Summary")
            log.info("=" * 80)
            log.info(f"Total unique patterns: {len(pattern_avg_losses)}")
            log.info(f"Total blocks analyzed: {sum(pattern_counts.values())}")
            log.info("\nTop 10 patterns by average loss:")
            for i, (pattern, avg_loss) in enumerate(sorted_patterns[:10], 1):
                pattern_str = "".join(str(int(x)) for x in pattern)
                count = pattern_counts[pattern]
                std_loss = pattern_std_losses[pattern]
                log.info(
                    f"  {i}. Pattern {pattern_str}: "
                    f"avg_loss={avg_loss:.4f}±{std_loss:.4f}, count={count}"
                )
            log.info("\nBottom 10 patterns by average loss:")
            for i, (pattern, avg_loss) in enumerate(sorted_patterns[-10:], 1):
                pattern_str = "".join(str(int(x)) for x in pattern)
                count = pattern_counts[pattern]
                std_loss = pattern_std_losses[pattern]
                log.info(
                    f"  {i}. Pattern {pattern_str}: "
                    f"avg_loss={avg_loss:.4f}±{std_loss:.4f}, count={count}"
                )
            log.info("=" * 80)
        # Reset for next eval
        self.pattern_stats.clear()

    def _plot_masking_patterns(
        self,
        pattern_avg_losses: dict[tuple, float],
        pattern_std_losses: dict[tuple, float],
        sorted_patterns: list[tuple[tuple, float]],
        pattern_counts: dict[tuple, int],
        logger: Logger,
    ) -> None:
        """Create and log a bar chart of masking pattern losses with error bars."""
        # Prepare data for plotting
        pattern_strings = []
        avg_losses = []
        std_losses = []
        counts = []
        
        for pattern, avg_loss in sorted_patterns:
            pattern_str = "".join(str(int(x)) for x in pattern)
            pattern_strings.append(pattern_str)
            avg_losses.append(avg_loss)
            std_losses.append(pattern_std_losses[pattern])
            counts.append(pattern_counts[pattern])
        
        # Compute overall mean and std across all patterns
        total_count = sum(pattern_counts.values())
        if total_count > 0:
            overall_mean = sum(
                avg_loss * pattern_counts[pattern]
                for pattern, avg_loss in pattern_avg_losses.items()
            ) / total_count
            
            # Compute overall std from all values
            all_losses = []
            for pattern, loss_values in self.pattern_stats.items():
                all_losses.extend(loss_values)
            if len(all_losses) > 1:
                overall_std = float(np.std(all_losses, ddof=1))
            else:
                overall_std = 0.0
        else:
            overall_mean = 0.0
            overall_std = 0.0
        
        # Create figure with dynamic sizing
        n_patterns = len(pattern_strings)
        fig_width = max(12, min(n_patterns * 0.5, 30))  # Cap width at 30 inches
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Create bar chart with error bars
        x_pos = np.arange(n_patterns)
        bars = ax.bar(
            x_pos,
            avg_losses,
            yerr=std_losses,
            capsize=5,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Customize plot
        ax.set_xlabel("Masking Pattern (1=masked, 0=unmasked)", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title(
            f"Val. NLL by Masking Pattern\n"
            f"(NLL averaged over all patterns: {overall_mean:.4f}±{overall_std:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        
        # Adjust font size and rotation based on number of patterns
        if n_patterns > 20:
            label_fontsize = 8
            rotation = 90
        elif n_patterns > 10:
            label_fontsize = 10
            rotation = 60
        else:
            label_fontsize = 12
            rotation = 45
        
        ax.set_xticklabels(pattern_strings, rotation=rotation, ha="right", fontsize=label_fontsize)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add horizontal line for overall mean
        ax.axhline(
            y=overall_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean NLL over all patterns"
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add bar height annotations on bars
        for i, (bar, avg_loss) in enumerate(zip(bars, avg_losses)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std_losses[i] + 0.01 * max(avg_losses),
                f"{avg_loss:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        plt.tight_layout()
        plt.savefig("masking_pattern_loss_analysis.png")
        
        # Close figure to free memory
        plt.close(fig)

    def _plot_masking_pattern_frequencies(
        self,
        sorted_patterns: list[tuple[tuple, float]],
        pattern_counts: dict[tuple, int],
        logger: Logger,
    ) -> None:
        """Create and log a bar chart of masking pattern frequencies."""
        # Prepare data for plotting
        pattern_strings = []
        counts = []
        
        for pattern, _ in sorted_patterns:
            pattern_str = "".join(str(int(x)) for x in pattern)
            pattern_strings.append(pattern_str)
            counts.append(pattern_counts[pattern])
        
        # Compute total count and mean frequency
        total_count = sum(pattern_counts.values())
        if total_count > 0:
            mean_frequency = total_count / len(pattern_counts)
        else:
            mean_frequency = 0.0
        
        # Create figure with dynamic sizing
        n_patterns = len(pattern_strings)
        fig_width = max(12, min(n_patterns * 0.5, 30))  # Cap width at 30 inches
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Create bar chart
        x_pos = np.arange(n_patterns)
        bars = ax.bar(
            x_pos,
            counts,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Customize plot
        ax.set_xlabel("Masking Pattern (1=masked, 0=unmasked)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Masking Pattern Frequency\n"
            f"(Total blocks: {total_count}, Mean frequency: {mean_frequency:.1f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        
        # Adjust font size and rotation based on number of patterns
        if n_patterns > 20:
            label_fontsize = 8
            rotation = 90
        elif n_patterns > 10:
            label_fontsize = 10
            rotation = 60
        else:
            label_fontsize = 12
            rotation = 45
        
        ax.set_xticklabels(pattern_strings, rotation=rotation, ha="right", fontsize=label_fontsize)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add horizontal line for mean frequency
        ax.axhline(
            y=mean_frequency,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean frequency"
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add bar height annotations on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(counts),
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        plt.tight_layout()
        plt.savefig("masking_pattern_frequency_analysis.png")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"masking_pattern_frequency_analysis": wandb.Image("masking_pattern_frequency_analysis.png")})
        else:
            logger.log_images({"masking_pattern_frequency_analysis": "masking_pattern_frequency_analysis.png"})
        
        # Close figure to free memory
        plt.close(fig)


class PermutationOrderLossAnalysis(Callback):
    """Callback to analyze loss by permutation order within blocks for AnyOrderBD3LM.
    
    For each datapoint, records the loss for every single permutation order
    (within each block). At the end of validation, reports the average loss
    for each permutation order across all examples and blocks.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Store all loss values for each permutation order
        self.permutation_stats: dict[tuple, list[float]] = {}

    def _get_model(self, state: State):
        """Get the model, handling wrapped models."""
        if hasattr(state.model, "module"):
            return state.model.module.model
        return state.model.model

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        """Record permutation orders and losses for each block."""
        # Only run during validation
        if state.dataloader_label == "train":
            return
        
        # Get model
        model = self._get_model(state)
        
        # Check if model is AnyOrderBD3LM
        from src.denoiser.diffusion import AnyOrderBD3LM
        
        if not isinstance(model, AnyOrderBD3LM):
            return  # Silently skip if not AnyOrderBD3LM
        
        # Check if model has block_size config
        if not hasattr(model.config, "block_size") and not hasattr(
            model.config, "eval_block_size"
        ):
            log.warning(
                "Model does not have block_size config. Skipping permutation order analysis."
            )
            return
        
        block_size = getattr(
            model.config, "eval_block_size", getattr(model.config, "block_size", None)
        )
        if block_size is None:
            log.warning("Could not determine block_size. Skipping permutation order analysis.")
            return
        
        # Get batch data
        batch = state.batch
        input_ids = batch.get("input_ids")
        context_mask = batch.get("context_mask")
        if input_ids is None:
            return
        
        # Get model outputs
        outputs = state.outputs
        if outputs is None:
            return
        
        # Extract nlls from outputs (per-token losses)
        nlls = getattr(outputs, "other_loss_terms", {}).get("log_p_theta", None)
        if nlls is None:
            log.warning("Could not find nlls in outputs. Skipping this batch.")
            return
        
        attention_mask = batch.get("attention_mask")
        
        # Get permutation_order from other_loss_terms
        # Shape is (B, n_blocks, block_size) where each element is the relative position
        permutation_order = getattr(outputs, "other_loss_terms", {}).get("permutation_order")
        if permutation_order is None:
            log.warning("Could not find permutation_order in outputs. Skipping this batch.")
            return
        
        # For AnyOrderBD3LM, nlls has shape (B, L) where L is sequence length
        seq_len = nlls.shape[1]
        batch_size = nlls.shape[0]
        n_blocks = seq_len // block_size
        
        for b in range(batch_size):
            for block_idx in range(n_blocks):
                start_idx = block_idx * block_size
                end_idx = start_idx + block_size
                
                # Extract permutation order for this block
                # permutation_order shape: (B, n_blocks, block_size)
                block_permutation = permutation_order[b, start_idx:end_idx] - start_idx
                
                # Check if block is valid (not all padding)
                block_tokens_mask = attention_mask[b, start_idx:end_idx]
                context_tokens_mask = context_mask[b, start_idx:end_idx]
                if (block_tokens_mask == 0).any() or (context_tokens_mask == 1).any(): # Padding tokens, skip
                    continue
                
                # Convert to tuple for hashing (move to CPU and convert to avoid keeping GPU tensors)
                block_permutation_cpu = block_permutation.cpu()
                perm_tuple = tuple(block_permutation_cpu.int().tolist())
                
                # Extract corresponding loss for this block
                block_nlls = nlls[b, start_idx:end_idx]
                block_avg_loss = block_nlls.mean().item()
                
                if perm_tuple not in self.permutation_stats:
                    self.permutation_stats[perm_tuple] = []
                self.permutation_stats[perm_tuple].append(block_avg_loss)

    def eval_end(self, state: State, logger: Logger) -> None:
        """Report average loss per permutation order."""
        # Gather data from all ranks for distributed evaluation
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                # Gather permutation_stats from all ranks
                gathered_permutation_stats = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_permutation_stats, self.permutation_stats)
                
                # Combine statistics from all ranks
                combined_permutation_stats = {}
                for rank_stats in gathered_permutation_stats:
                    for perm_tuple, loss_values in rank_stats.items():
                        if perm_tuple not in combined_permutation_stats:
                            combined_permutation_stats[perm_tuple] = []
                        combined_permutation_stats[perm_tuple].extend(loss_values)
                
                # Clear gathered data to free memory
                del gathered_permutation_stats
                self.permutation_stats = combined_permutation_stats
                
                # Synchronize all ranks before proceeding
                dist.barrier()
        
        if not self.permutation_stats:
            log.warning("No permutation orders recorded. Skipping report.")
            return

        
        # Combine _all_ losses from all ranks / patterns
        combined_nll_stats = []
        for perm_tuple, loss_values in self.permutation_stats.items():
            combined_nll_stats.extend(loss_values)
        combined_nll_stats = np.array(combined_nll_stats)
        log.info(f"Final NLL: {combined_nll_stats.mean():.4f}±{combined_nll_stats.std():.4f}")
        
        # Compute average loss and std from collected values
        permutation_avg_losses = {}
        permutation_std_losses = {}
        permutation_counts = {}
        for perm_tuple, loss_values in self.permutation_stats.items():
            loss_array = np.array(loss_values)
            permutation_avg_losses[perm_tuple] = float(np.mean(loss_array))
            if len(loss_values) > 1:
                permutation_std_losses[perm_tuple] = float(np.std(loss_array, ddof=1))
            else:
                permutation_std_losses[perm_tuple] = 0.0
            permutation_counts[perm_tuple] = len(loss_values)
        
        # Sort permutation orders by lexicographic order (for consistent ordering)
        sorted_permutations = sorted(
            permutation_avg_losses.items(),
            key=lambda x: x[0],  # Sort by permutation tuple
        )
        
        # Log metrics for each permutation order
        metrics = {}
        for perm_tuple, avg_loss in sorted_permutations:
            # Convert permutation tuple to a readable string (1-indexed)
            perm_str = ",".join(str(int(x) + 1) for x in perm_tuple)
            count = permutation_counts[perm_tuple]
            std_loss = permutation_std_losses[perm_tuple]
            metrics[f"permutation_order_loss/{perm_str}"] = avg_loss
            metrics[f"permutation_order_loss_std/{perm_str}"] = std_loss
            metrics[f"permutation_order_count/{perm_str}"] = count
        
        logger.log_metrics(metrics)
        
        # Create bar chart with error bars and log summary (only on rank 0)
        if dist.get_global_rank() == 0:
            self._plot_permutation_orders(
                permutation_avg_losses, permutation_std_losses, sorted_permutations, permutation_counts, logger
            )
            self._plot_permutation_frequencies(
                sorted_permutations, permutation_counts, logger
            )
            
            # Also log a summary
            log.info("=" * 80)
            log.info("Permutation Order Loss Analysis Summary")
            log.info("=" * 80)
            log.info(f"Total unique permutation orders: {len(permutation_avg_losses)}")
            log.info(f"Total blocks analyzed: {sum(permutation_counts.values())}")
            log.info("\nTop 10 permutation orders by average loss:")
            for i, (perm_tuple, avg_loss) in enumerate(sorted_permutations[:10], 1):
                perm_str = ",".join(str(int(x) + 1) for x in perm_tuple)  # 1-indexed
                count = permutation_counts[perm_tuple]
                std_loss = permutation_std_losses[perm_tuple]
                log.info(
                    f"  {i}. Permutation {perm_str}: "
                    f"avg_loss={avg_loss:.4f}±{std_loss:.4f}, count={count}"
                )
            log.info("\nBottom 10 permutation orders by average loss:")
            for i, (perm_tuple, avg_loss) in enumerate(sorted_permutations[-10:], 1):
                perm_str = ",".join(str(int(x) + 1) for x in perm_tuple)  # 1-indexed
                count = permutation_counts[perm_tuple]
                std_loss = permutation_std_losses[perm_tuple]
                log.info(
                    f"  {i}. Permutation {perm_str}: "
                    f"avg_loss={avg_loss:.4f}±{std_loss:.4f}, count={count}"
                )
            log.info("=" * 80)
        
        # Reset for next eval
        self.permutation_stats.clear()

    def _plot_permutation_orders(
        self,
        permutation_avg_losses: dict[tuple, float],
        permutation_std_losses: dict[tuple, float],
        sorted_permutations: list[tuple[tuple, float]],
        permutation_counts: dict[tuple, int],
        logger: Logger,
    ) -> None:
        """Create and log a bar chart of permutation order losses with error bars."""
        # Prepare data for plotting
        permutation_strings = []
        avg_losses = []
        std_losses = []
        counts = []
        
        for perm_tuple, avg_loss in sorted_permutations:
            perm_str = ",".join(str(int(x) + 1) for x in perm_tuple)  # 1-indexed
            permutation_strings.append(perm_str)
            avg_losses.append(avg_loss)
            std_losses.append(permutation_std_losses[perm_tuple])
            counts.append(permutation_counts[perm_tuple])
        
        # Compute overall mean and std across all permutation orders
        # Use weighted average based on counts
        total_count = sum(permutation_counts.values())
        if total_count > 0:
            overall_mean = sum(
                avg_loss * permutation_counts[perm_tuple]
                for perm_tuple, avg_loss in permutation_avg_losses.items()
            ) / total_count
            
            # Compute overall std from all values
            all_losses = []
            for perm_tuple, loss_values in self.permutation_stats.items():
                all_losses.extend(loss_values)
            if len(all_losses) > 1:
                overall_std = float(np.std(all_losses, ddof=1))
            else:
                overall_std = 0.0
        else:
            overall_mean = 0.0
            overall_std = 0.0
        
        # Create figure with dynamic sizing
        n_permutations = len(permutation_strings)
        fig_width = max(12, min(n_permutations * 0.5, 30))  # Cap width at 30 inches
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Create bar chart with error bars
        x_pos = np.arange(n_permutations)
        bars = ax.bar(
            x_pos,
            avg_losses,
            yerr=std_losses,
            capsize=5,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Customize plot
        ax.set_xlabel("Permutation Order", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title(
            f"Val. NLL by Permutation Order\n"
            f"(NLL averaged over all orders: {overall_mean:.4f}±{overall_std:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        
        # Adjust font size and rotation based on number of permutations
        if n_permutations > 20:
            label_fontsize = 8
            rotation = 90
        elif n_permutations > 10:
            label_fontsize = 10
            rotation = 60
        else:
            label_fontsize = 12
            rotation = 45
        
        ax.set_xticklabels(permutation_strings, rotation=rotation, ha="right", fontsize=label_fontsize)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add horizontal line for overall mean
        ax.axhline(
            y=overall_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean NLL over all orders"
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add bar height annotations on bars
        for i, (bar, avg_loss) in enumerate(zip(bars, avg_losses)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std_losses[i] + 0.01 * max(avg_losses),
                f"{avg_loss:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        plt.tight_layout()
        plt.savefig("permutation_order_loss_analysis.png")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"permutation_order_loss_analysis": wandb.Image("permutation_order_loss_analysis.png")})
        else:
            logger.log_images({"permutation_order_loss_analysis": "permutation_order_loss_analysis.png"})
        
        # Close figure to free memory
        plt.close(fig)

    def _plot_permutation_frequencies(
        self,
        sorted_permutations: list[tuple[tuple, float]],
        permutation_counts: dict[tuple, int],
        logger: Logger,
    ) -> None:
        """Create and log a bar chart of permutation order frequencies."""
        # Prepare data for plotting
        permutation_strings = []
        counts = []
        
        for perm_tuple, _ in sorted_permutations:
            perm_str = ",".join(str(int(x) + 1) for x in perm_tuple)  # 1-indexed
            permutation_strings.append(perm_str)
            counts.append(permutation_counts[perm_tuple])
        
        # Compute total count and mean frequency
        total_count = sum(permutation_counts.values())
        if total_count > 0:
            mean_frequency = total_count / len(permutation_counts)
        else:
            mean_frequency = 0.0
        
        # Create figure with dynamic sizing
        n_permutations = len(permutation_strings)
        fig_width = max(12, min(n_permutations * 0.5, 30))  # Cap width at 30 inches
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Create bar chart
        x_pos = np.arange(n_permutations)
        bars = ax.bar(
            x_pos,
            counts,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Customize plot
        ax.set_xlabel("Permutation Order", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Permutation Order Frequency\n"
            f"(Total blocks: {total_count}, Mean frequency: {mean_frequency:.1f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        
        # Adjust font size and rotation based on number of permutations
        if n_permutations > 20:
            label_fontsize = 8
            rotation = 90
        elif n_permutations > 10:
            label_fontsize = 10
            rotation = 60
        else:
            label_fontsize = 12
            rotation = 45
        
        ax.set_xticklabels(permutation_strings, rotation=rotation, ha="right", fontsize=label_fontsize)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add horizontal line for mean frequency
        ax.axhline(
            y=mean_frequency,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean frequency"
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add bar height annotations on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(counts),
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        plt.tight_layout()
        plt.savefig("permutation_order_frequency_analysis.png")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"permutation_order_frequency_analysis": wandb.Image("permutation_order_frequency_analysis.png")})
        else:
            logger.log_images({"permutation_order_frequency_analysis": "permutation_order_frequency_analysis.png"})
        
        # Close figure to free memory
        plt.close(fig)


class PermutationToMaskingPatternAnalysis(Callback):
    """Callback to analyze loss by masking pattern derived from permutation orders.
    
    For AnyOrderBD3LM, each permutation order (e.g., 1234) corresponds to a sequence
    of masking patterns. This callback maps per-token NLLs to their corresponding
    masking patterns based on the permutation order, then reports aggregated statistics.
    
    Example: Permutation 1234 uses masking patterns:
    - Position 1: 1111 (all masked)
    - Position 2: 0111 (first unmasked, rest masked)
    - Position 3: 0011 (first two unmasked, rest masked)
    - Position 4: 0001 (first three unmasked, last masked)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Use incremental statistics to avoid OOM
        # For each masking pattern: (count, mean, M2) for Welford's algorithm
        self.pattern_stats: dict[tuple, tuple[int, float, float]] = {}

    def _get_model(self, state: State):
        """Get the model, handling wrapped models."""
        if hasattr(state.model, "module"):
            return state.model.module.model
        return state.model.model

    def _permutation_to_masking_patterns(
        self, perm_tuple: tuple[int, ...], block_size: int
    ) -> list[tuple[int, ...]]:
        """Convert a permutation order to a sequence of masking patterns.
        
        Args:
            perm_tuple: Permutation order (0-indexed positions)
            block_size: Size of the block
            
        Returns:
            List of masking patterns, one for each position in the permutation
        """
        # Create inverse permutation: for each position, which step it's predicted at
        inverse_perm = [0] * block_size
        for step, pos in enumerate(perm_tuple):
            inverse_perm[pos] = step
        
        # For each step in the permutation, create the masking pattern
        patterns = []
        for step in range(block_size):
            # At step 'step', tokens predicted at steps 0 to step-1 are unmasked
            # All others are masked
            pattern = tuple(
                0 if inverse_perm[pos] < step else 1
                for pos in range(block_size)
            )
            patterns.append(pattern)
        
        return patterns

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        """Record masking patterns and losses for each block based on permutation order."""
        # Only run during validation
        if state.dataloader_label == "train":
            return
        
        # Get model
        model = self._get_model(state)
        
        # Check if model is AnyOrderBD3LM
        from src.denoiser.diffusion import AnyOrderBD3LM
        
        if not isinstance(model, AnyOrderBD3LM):
            return  # Silently skip if not AnyOrderBD3LM
        
        # Check if model has block_size config
        if not hasattr(model.config, "block_size") and not hasattr(
            model.config, "eval_block_size"
        ):
            log.warning(
                "Model does not have block_size config. Skipping permutation-to-masking analysis."
            )
            return
        
        block_size = getattr(
            model.config, "eval_block_size", getattr(model.config, "block_size", None)
        )
        if block_size is None:
            log.warning("Could not determine block_size. Skipping permutation-to-masking analysis.")
            return
        
        # Get batch data
        batch = state.batch
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return
        
        # Get model outputs
        outputs = state.outputs
        if outputs is None:
            return
        
        # Extract nlls from outputs (per-token losses)
        nlls = getattr(outputs, "other_loss_terms", {}).get("log_p_theta", None)
        if nlls is None:
            log.warning("Could not find nlls in outputs. Skipping this batch.")
            return
        
        # Get attention mask and tokens_mask
        attention_mask = batch.get("attention_mask")
        
        # Get context_mask from batch
        context_mask = batch.get("context_mask")
        
        # Get permutation_order from other_loss_terms
        permutation_order = getattr(outputs, "other_loss_terms", {}).get("permutation_order")
        if permutation_order is None:
            log.warning("Could not find permutation_order in outputs. Skipping this batch.")
            return
        
        # For AnyOrderBD3LM, nlls has shape (B, L) where L is sequence length
        seq_len = nlls.shape[1]
        batch_size = nlls.shape[0]
        n_blocks = seq_len // block_size
        
        
        if context_mask is not None and context_mask.shape[1] != seq_len:
            context_mask = context_mask[:, :seq_len]
        
        for b in range(batch_size):
            for block_idx in range(n_blocks):
                start_idx = block_idx * block_size
                end_idx = start_idx + block_size
                
                # Extract permutation order for this block
                block_permutation = permutation_order[b, start_idx:end_idx] - start_idx
                # Convert to tuple for hashing (move to CPU and convert to avoid keeping GPU tensors)
                block_permutation_cpu = block_permutation.cpu()
                perm_tuple = tuple(block_permutation_cpu.int().tolist())
                
                # Check if block is valid (not all padding)
                block_tokens_mask = attention_mask[b, start_idx:end_idx]
                context_tokens_mask = context_mask[b, start_idx:end_idx]
                if (block_tokens_mask == 0).any() or (context_tokens_mask == 1).any(): # Padding tokens, skip
                    continue
                
                # Convert permutation to masking patterns
                masking_patterns = self._permutation_to_masking_patterns(perm_tuple, block_size)
                
                # Extract per-token losses for this block
                block_nlls = nlls[b, start_idx:end_idx]
                
                # For each token position, assign its NLL to the corresponding masking pattern
                # The masking pattern for position i is determined by when it's predicted in the permutation
                inverse_perm = [0] * block_size
                for step, pos in enumerate(perm_tuple):
                    inverse_perm[pos] = step
                
                for token_pos in range(block_size):
                    # Get the step at which this token is predicted
                    prediction_step = inverse_perm[token_pos]
                    
                    # Get the masking pattern for this step
                    pattern = masking_patterns[prediction_step]
                    
                    # Get the NLL for this token
                    token_nll = block_nlls[token_pos].item()
                    
                    if pattern not in self.pattern_stats:
                        self.pattern_stats[pattern] = []
                    self.pattern_stats[pattern].append(token_nll)

    def eval_end(self, state: State, logger: Logger) -> None:
        """Report average loss per masking pattern."""
        # Gather data from all ranks for distributed evaluation
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                # Gather pattern_stats from all ranks
                gathered_pattern_stats = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_pattern_stats, self.pattern_stats)
                
                # Combine statistics from all ranks
                combined_pattern_stats = {}
                for rank_stats in gathered_pattern_stats:
                    for pattern, loss_values in rank_stats.items():
                        if pattern not in combined_pattern_stats:
                            combined_pattern_stats[pattern] = []
                        combined_pattern_stats[pattern].extend(loss_values)
                
                self.pattern_stats = combined_pattern_stats
                dist.barrier()
        
        if not self.pattern_stats:
            log.warning("No masking patterns recorded. Skipping report.")
            return

        
        # Combine _all_ losses from all ranks / patterns
        combined_nll_stats = []
        for pattern, loss_values in self.pattern_stats.items():
            combined_nll_stats.extend(loss_values)
        combined_nll_stats = np.array(combined_nll_stats)
        log.info(f"Final NLL: {combined_nll_stats.mean():.4f}±{combined_nll_stats.std():.4f}")
        
        # Compute average loss and std from collected values
        pattern_avg_losses = {}
        pattern_std_losses = {}
        pattern_counts = {}
        for pattern, loss_values in self.pattern_stats.items():
            loss_array = np.array(loss_values)
            pattern_avg_losses[pattern] = float(np.mean(loss_array))
            if len(loss_values) > 1:
                pattern_std_losses[pattern] = float(np.std(loss_array, ddof=1))
            else:
                pattern_std_losses[pattern] = 0.0
            pattern_counts[pattern] = len(loss_values)
        
        # Sort patterns by number of masked tokens (ascending: least masking to most masking)
        # Then by position of 0's (more 0's on the left ranked higher)
        sorted_patterns = sorted(
            pattern_avg_losses.items(),
            key=lambda x: (sum(x[0]), x[0]),  # First by number of masked tokens, then by pattern (0's on left rank higher)
        )
        
        # Log metrics for each pattern
        metrics = {}
        for pattern, avg_loss in sorted_patterns:
            pattern_str = "".join(str(int(x)) for x in pattern)
            count = pattern_counts[pattern]
            std_loss = pattern_std_losses[pattern]
            metrics[f"permutation_derived_pattern_loss/{pattern_str}"] = avg_loss
            metrics[f"permutation_derived_pattern_loss_std/{pattern_str}"] = std_loss
            metrics[f"permutation_derived_pattern_count/{pattern_str}"] = count
        
        logger.log_metrics(metrics)
        
        # Create bar chart with error bars and log summary (only on rank 0)
        if dist.get_global_rank() == 0:
            self._plot_patterns(
                pattern_avg_losses, pattern_std_losses, sorted_patterns, pattern_counts, logger
            )
            self._plot_pattern_frequencies(
                sorted_patterns, pattern_counts, logger
            )
            
            # Also log a summary
            log.info("=" * 80)
            log.info("Permutation-Derived Masking Pattern Loss Analysis Summary")
            log.info("=" * 80)
            log.info(f"Total unique patterns: {len(pattern_avg_losses)}")
            log.info(f"Total tokens analyzed: {sum(pattern_counts.values())}")
            log.info("\nTop 10 patterns by average loss:")
            for i, (pattern, avg_loss) in enumerate(sorted_patterns[:10], 1):
                pattern_str = "".join(str(int(x)) for x in pattern)
                count = pattern_counts[pattern]
                std_loss = pattern_std_losses[pattern]
                log.info(
                    f"  {i}. Pattern {pattern_str}: "
                    f"avg_loss={avg_loss:.4f}±{std_loss:.4f}, count={count}"
                )
            log.info("\nBottom 10 patterns by average loss:")
            for i, (pattern, avg_loss) in enumerate(sorted_patterns[-10:], 1):
                pattern_str = "".join(str(int(x)) for x in pattern)
                count = pattern_counts[pattern]
                std_loss = pattern_std_losses[pattern]
                log.info(
                    f"  {i}. Pattern {pattern_str}: "
                    f"avg_loss={avg_loss:.4f}±{std_loss:.4f}, count={count}"
                )
            log.info("=" * 80)
        
        # Reset for next eval
        self.pattern_stats.clear()

    def _plot_patterns(
        self,
        pattern_avg_losses: dict[tuple, float],
        pattern_std_losses: dict[tuple, float],
        sorted_patterns: list[tuple[tuple, float]],
        pattern_counts: dict[tuple, int],
        logger: Logger,
    ) -> None:
        """Create and log a bar chart of masking pattern losses with error bars."""
        # Prepare data for plotting
        pattern_strings = []
        avg_losses = []
        std_losses = []
        counts = []
        
        for pattern, avg_loss in sorted_patterns:
            pattern_str = "".join(str(int(x)) for x in pattern)
            pattern_strings.append(pattern_str)
            avg_losses.append(avg_loss)
            std_losses.append(pattern_std_losses[pattern])
            counts.append(pattern_counts[pattern])
        
        # Compute overall mean and std
        total_count = sum(pattern_counts.values())
        if total_count > 0:
            overall_mean = sum(
                avg_loss * pattern_counts[pattern]
                for pattern, avg_loss in pattern_avg_losses.items()
            ) / total_count
            
            # Compute overall std from all values
            all_losses = []
            for pattern, loss_values in self.pattern_stats.items():
                all_losses.extend(loss_values)
            if len(all_losses) > 1:
                overall_std = float(np.std(all_losses, ddof=1))
            else:
                overall_std = 0.0
        else:
            overall_mean = 0.0
            overall_std = 0.0
        
        # Create figure with dynamic sizing
        n_patterns = len(pattern_strings)
        fig_width = max(12, min(n_patterns * 0.5, 30))
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Create bar chart with error bars
        x_pos = np.arange(n_patterns)
        bars = ax.bar(
            x_pos,
            avg_losses,
            yerr=std_losses,
            capsize=5,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Customize plot
        ax.set_xlabel("Masking Pattern (1=masked, 0=unmasked)", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title(
            f"Val. NLL by Masking Pattern (from Permutations)\n"
            f"(NLL averaged over all patterns: {overall_mean:.4f}±{overall_std:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        
        # Adjust font size and rotation based on number of patterns
        if n_patterns > 20:
            label_fontsize = 8
            rotation = 90
        elif n_patterns > 10:
            label_fontsize = 10
            rotation = 60
        else:
            label_fontsize = 12
            rotation = 45
        
        ax.set_xticklabels(pattern_strings, rotation=rotation, ha="right", fontsize=label_fontsize)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add horizontal line for overall mean
        ax.axhline(
            y=overall_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Mean NLL over all patterns",
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add bar height annotations on bars
        for i, (bar, avg_loss) in enumerate(zip(bars, avg_losses)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std_losses[i] + 0.01 * max(avg_losses),
                f"{avg_loss:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        plt.tight_layout()
        plt.savefig("permutation_derived_pattern_loss_analysis.png")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"permutation_derived_pattern_loss_analysis": wandb.Image("permutation_derived_pattern_loss_analysis.png")})
        else:
            logger.log_images({"permutation_derived_pattern_loss_analysis": "permutation_derived_pattern_loss_analysis.png"})
        
        # Close figure to free memory
        plt.close(fig)

    def _plot_pattern_frequencies(
        self,
        sorted_patterns: list[tuple[tuple, float]],
        pattern_counts: dict[tuple, int],
        logger: Logger,
    ) -> None:
        """Create and log a bar chart of masking pattern frequencies (from permutations)."""
        # Prepare data for plotting
        pattern_strings = []
        counts = []
        
        for pattern, _ in sorted_patterns:
            pattern_str = "".join(str(int(x)) for x in pattern)
            pattern_strings.append(pattern_str)
            counts.append(pattern_counts[pattern])
        
        # Compute total count and mean frequency
        total_count = sum(pattern_counts.values())
        if total_count > 0:
            mean_frequency = total_count / len(pattern_counts)
        else:
            mean_frequency = 0.0
        
        # Create figure with dynamic sizing
        n_patterns = len(pattern_strings)
        fig_width = max(12, min(n_patterns * 0.5, 30))
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # Create bar chart
        x_pos = np.arange(n_patterns)
        bars = ax.bar(
            x_pos,
            counts,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Customize plot
        ax.set_xlabel("Masking Pattern (1=masked, 0=unmasked)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Masking Pattern Frequency (from Permutations)\n"
            f"(Total tokens: {total_count}, Mean frequency: {mean_frequency:.1f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        
        # Adjust font size and rotation based on number of patterns
        if n_patterns > 20:
            label_fontsize = 8
            rotation = 90
        elif n_patterns > 10:
            label_fontsize = 10
            rotation = 60
        else:
            label_fontsize = 12
            rotation = 45
        
        ax.set_xticklabels(pattern_strings, rotation=rotation, ha="right", fontsize=label_fontsize)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add horizontal line for mean frequency
        ax.axhline(
            y=mean_frequency,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Mean frequency",
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add bar height annotations on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(counts),
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        plt.tight_layout()
        plt.savefig("permutation_derived_pattern_frequency_analysis.png")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"permutation_derived_pattern_frequency_analysis": wandb.Image("permutation_derived_pattern_frequency_analysis.png")})
        else:
            logger.log_images({"permutation_derived_pattern_frequency_analysis": "permutation_derived_pattern_frequency_analysis.png"})
        
        # Close figure to free memory
        plt.close(fig)
