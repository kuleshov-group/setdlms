import logging
import os
import pathlib
import shutil
import time
from typing import Any

from composer.callbacks import CheckpointSaver
from composer.core import Callback, State, Timestamp
from composer.loggers import Logger, WandBLogger
from composer.utils import PartialFilePath, dist, get_save_filename

from src.utils import push_to_hub, snapshot_repo

log = logging.getLogger(__name__)
__all__ = ["DataloaderSpeedMonitor"]


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
        if dist.get_global_rank() == 0:
            run_id = None
            for logger in logger.destinations:
                if isinstance(logger, WandBLogger):
                    run_id = logger.run_url
                    break
            self.project_root = snapshot_repo(run_id=run_id)
        dist.barrier()  # Holds all the ranks until repo snapshot is done

    def _save_checkpoint(self, state: State, logger: Logger) -> None:
        """
        Copied / adapted from composer.callbacks.CheckpointSaver._save_checkpoint
            for HF compatibility.
        """
        # TODO: Check that HF saving works with state.fsdp_sharded_state_dict_enabled
        #  (or if we can ignore this scenario).
        # TODO: Do we need to implement HF uploading for remote uploading too?

        super()._save_checkpoint(state, logger)  # Perform standard checkpointing

        hf_filename_with_placeholders = self.hf_filename.format(
            state, keep_placeholders=True
        )
        save_hf_filename = get_save_filename(state, hf_filename_with_placeholders)
        self.all_saved_hf_checkpoints_to_timestamp[save_hf_filename] = state.timestamp

        # Adapting `checkpoint.save_checkpoint / ._save_checkpoint` for HF
        if dist.get_global_rank() == 0:
            push_to_hub(
                model=state.model.model,
                tokenizer=state.model.tokenizer,
                repo_id=save_hf_filename,
                local=True,
                project_root=self.project_root,
            )
            saved_hf_path = save_hf_filename
        else:
            saved_hf_path = None
        log.debug(f"HF checkpoint locally saved to {saved_hf_path}")

        if not saved_hf_path:  # not all ranks save
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
