import glob
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Literal

import torch
import wandb
from composer.callbacks import CheckpointSaver
from composer.core import Callback, State, Timestamp
from composer.loggers import Logger
from composer.utils import PartialFilePath, dist, get_save_filename

from src.utils import fsspec_exists, push_to_hub, snapshot_repo_to_tmp_dir

log = logging.getLogger(__name__)
__all__ = ["DataloaderSpeedMonitor"]


class DataloaderSpeedMonitor(Callback):
    """Measure how long it takes to return a batch from the dataloader.

    Copied from:
        https://github.com/AnswerDotAI/ModernBERT/blob/main/src/callbacks/dataloader_speed.py
        Copyright 2024 onwards Answe-r.AI, LightOn, and contributors
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

    def __init__(self, disable_hf: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.disable_hf = disable_hf
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
        if dist.get_global_rank() == 0:
            push_to_hub(
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
        else:
            saved_hf_path = None
        log.debug(f"HF checkpoint locally saved to {saved_hf_path}")

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
            if fsspec_exists(self.project_root):
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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(disable_hf, *args, **kwargs)

        self.metric_to_monitor = metric_to_monitor
        self.train_or_eval = metric_to_monitor.split("/")[0]
        self.metric_name = "/".join(metric_to_monitor.split("/")[1:])
        self.mode = mode
        self.best_value = None

        self.latest_filename = None
        self.latest_hf_filename = None
        self.num_checkpoints_to_keep = 1

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
