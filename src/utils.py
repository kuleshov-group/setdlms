import hashlib
import inspect
import logging
import os
import re
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from types import MethodType

import fsspec
from huggingface_hub import HfApi, file_exists, repo_exists
from transformers import PreTrainedModel, PreTrainedTokenizer

log = logging.getLogger(__name__)


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def snapshot_repo_to_tmp_dir(
    run_id: str | None = None,
    tmp_dir_exists_ok: bool = False,
) -> str:
    """Snapshot a repo to a local (tmp) directory.

    Args:
        run_id (optional: str): Run ID (e.g., wandb uuid), to be used in creating hash
            for the local (tmp) directory
            If None, timestamp is used.
        tmp_dir_exists_ok (bool): Whether to throw an error (False) if tmp dir exists
            already or re-use existing (True).
    """

    def _snapshot_files(src_path: Path, dest_path: Path, ignore: list[str]) -> None:
        """Helper method that recursively copies files from src_path to dest_path.
        Ignores files matching the patterns in ignore (list).
        """
        if any([re.search(ignore_file, str(src_path)) for ignore_file in ignore]):
            return
        if os.path.isdir(src_path):
            dest_path.mkdir(parents=True, exist_ok=True)
            for sp in fsspec_listdir(src_path):
                _snapshot_files(
                    src_path / sp, dest_path / Path(sp).resolve().name, ignore
                )
        if os.path.isdir(src_path):
            return
        # else: src_path is a file
        shutil.copy2(src_path, dest_path)

    # Get .gitignore list
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / ".gitignore", "r", encoding="utf-8") as gf:
        ignore_list = [line.strip() for line in gf.readlines()]
    ignore_list.extend(
        [ignore_file[:-1] for ignore_file in ignore_list if ignore_file.endswith("/")]
    )

    # Construct a unique ID for temporary directory
    root = gettempdir()
    log.debug(root)
    hash_key = hashlib.blake2s(
        (run_id if run_id is not None else str(int(time.time()))).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    tmp_dir = os.path.join(root, f"tmp{hash_key}")
    if fsspec_exists(tmp_dir):
        if tmp_dir_exists_ok:
            return tmp_dir
        else:
            raise ValueError(
                f"Cannot create snapshot. Temporary directory {tmp_dir} already exists."
                " Please remove it or use a different run_id."
            )
    fsspec_mkdirs(tmp_dir)
    log.debug(tmp_dir)
    _snapshot_files(project_root, Path(tmp_dir).resolve(), ignore_list)
    log.debug(f"Snapshot saved to {tmp_dir}")
    return tmp_dir


def _flatten_and_copy(src_path: Path, dest_path: Path, ignore: list[str]) -> None:
    """Copy file contents and flatten relative imports.

    Ignores __init__.py files.
    Recursively applies to directories and flattens file names from `/` to `_`
    """

    def _copy_file_contents_and_flatten_relative_imports(src, dest):
        """Helper method that copies file contents and flattens relative imports.

        All instances of `src.` are replaced with `.` and all `.` (after the first one)
            in relative imports are replaced with `_`.
        """
        with open(src, "r", encoding="utf-8") as f:
            lines = f.readlines()
        modified_lines = []
        for line in lines:
            # Match lines starting with "import ."
            if re.match(r"^\s*import\s+\.(\S+)\s*$", line):
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"import \.([\w.]+)",
                    lambda m: f"import {m.group(1).replace('.', '_')}",
                    line,
                )
            # Match lines starting with "from ."
            elif re.match(r"^\s*from\s+\.(\S+)\s+import", line):
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"from \.([\w.]+)",
                    lambda m: f"from {m.group(1).replace('.', '_')}",
                    line,
                )
            # Match lines starting with "import src."
            elif re.match(r"^\s*import\s+src\.(\S+)\s*$", line):
                # Replace 'import src.' with 'import .'
                modified_line = re.sub(r"^\s*import\s+src\.", "import .", line)
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"^import \.([\w.]+)",
                    lambda m: f"import .{m.group(1).replace('.', '_')}",
                    modified_line,
                )
            # Match lines starting with "from src."
            elif re.match(r"^\s*from\s+src\.(\S+)\s+import", line):
                # Replace 'from src.' with 'from .'
                modified_line = re.sub(r"^\s*from\s+src\.", "from .", line)
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"^from \.([\w.]+)",
                    lambda m: f"from .{m.group(1).replace('.', '_')}",
                    modified_line,
                )
            else:
                modified_line = line
            modified_lines.append(modified_line.encode("utf-8"))
        with open(
            dest,
            "wb",
        ) as f:
            f.writelines(modified_lines)

    if any([re.search(ignore_file, str(src_path)) for ignore_file in ignore]):
        log.debug("Skipping:", src_path)
        return
    if os.path.isdir(src_path):
        for sp in fsspec_listdir(src_path):
            _flatten_and_copy(
                src_path / sp,
                Path(f"{str(dest_path)}_{Path(sp).resolve().name}"),
                ignore,
            )
    if os.path.isdir(src_path):
        return
    # Copy contents; replace `.` in relative imports with `_` and remove `src.` prefix
    # (e.g. `src.backbone.dit` -> `.backbone_dit`)
    log.debug(f"Copying {src_path} to {dest_path}")
    _copy_file_contents_and_flatten_relative_imports(src_path, dest_path)


def _state_dict_no_buffers(self, *args, **kwargs):
    # Copy original state_dict
    filtered_state_dict = {
        k: v for k, v in super(self.__class__, self).state_dict(*args, **kwargs).items()
    }
    # Skip explicitly listed parameter names
    skip_params_for_push = getattr(self, "skip_params_for_push", [])
    for skip_name in skip_params_for_push:
        filtered_state_dict.pop(skip_name, None)

    return filtered_state_dict


def save_pretrained_or_push_to_hub(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    repo_id: str = "kuleshov-group/dllm-dev",
    commit_message: str = "Add model and code",
    local: bool = False,
    private: bool = True,
    project_root: str | None = None,
) -> None:
    """Push / Save model and code to hub / local directory.

    Enables model loading using `AutoModel.from_pretrained` paradigm.

    Args:
        model (PreTrainedModel): Model to push / save.
        tokenizer (PreTrainedTokenizer): Tokenizer to push / save.
        repo_id (str) Repository ID on Hugging Face Hub / Local directory.
        commit_message (str): Commit message.
        local (bool): If True, push to local directory instead of Hugging Face Hub.
        private (bool): Whether remote hub repo is private.
        project_root (optional: str): Path to the project root directory. If None, uses
            the parent of __file__ path.
            Use this parameter if, for example, pushing from a tmp copy of the repo.
    """
    # Register config and model classes
    model.config.auto_map = model.config.auto_map
    model_cls_path = (  # e.g.:
        inspect.getfile(model.__class__)  # <project_path>/src/denoiser/diffusion.py
        .split(str(Path(__file__).resolve().parent.parent))[-1]
        .replace("/", ".")  # .src.denoiser.diffusion.py
        .split(".py")[0][1:]  # src.denoiser.diffusion
    )
    exec(f"from {model_cls_path} import {type(model).__name__}")
    exec(f"from {model_cls_path} import {type(model.config).__name__}")
    exec(f"{type(model.config).__name__}.register_for_auto_class()")
    for automodel in ["AutoModel", "AutoModelForCausalLM", "AutoModelForMaskedLM"]:
        if automodel in model.config.__class__.auto_map.keys():
            exec(f'{type(model).__name__}.register_for_auto_class("{automodel}")')

    # Update model config paths to remove `src` and flatten (replace `.` with `_`)
    # in `_target_` (e.g. `src.backbone.dit` -> `backbone_dit`)
    if re.match(r"^src\.", model.config.backbone_config["_target_"]):
        model.config.backbone_config["_target_"] = re.sub(
            r"^([\w.]+)\.",
            lambda m: f"{m.group(1).replace('.', '_')}.",
            re.sub(r"^src\.", "", model.config.backbone_config["_target_"]),
        )
    if re.match(r"^src\.", model.config.noise_config["_target_"]):
        model.config.noise_config["_target_"] = re.sub(
            r"^([\w.]+)\.",
            lambda m: f"{m.group(1).replace('.', '_')}.",
            re.sub(r"^src\.", "", model.config.noise_config["_target_"]),
        )
    log.debug("Updated model.config:")
    log.debug(model.config)

    # Set up destination
    tmp_dir = TemporaryDirectory() if not local else None
    dest_path = Path(repo_id) if local else Path(tmp_dir.name)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Temporarily override state_dict() to remove buffers
    model.state_dict = MethodType(_state_dict_no_buffers, model)
    # Save/push model and tokenizer
    log.debug(f"{'Saving' if local else 'Pushing'} model to {repo_id}")
    if local:
        model.save_pretrained(dest_path, safe_serialization=False)
        tokenizer.save_pretrained(dest_path)
    else:
        if not repo_id:
            raise ValueError("Argument `repo_id` is required for push_to_hub.")
        if not repo_exists(repo_id) or not file_exists(repo_id, "tokenizer.json"):
            tokenizer.push_to_hub(
                repo_id, private=private, commit_message="Upload tokenizer"
            )

        model.push_to_hub(
            repo_id,
            private=private,
            commit_message="Update pytorch.bin; " + commit_message,
            safe_serialization=False,
        )

    # Copy source files
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root).resolve()
    with open(project_root / ".gitignore", "r", encoding="utf-8") as gf:
        ignore = [line.strip() for line in gf.readlines()]
    ignore.extend(
        [ignore_file[:-1] for ignore_file in ignore if ignore_file.endswith("/")]
    )
    ignore.append("__init__.py")
    model_file_path = inspect.getfile(model.__class__).split(
        str(Path(__file__).resolve().parent.parent)
    )[-1][1:]
    paths_to_copy = {
        project_root / ".gitignore": ".gitignore",
        project_root / "src/denoiser/base.py": "denoiser_base.py",
        project_root / model_file_path: model_file_path.split("/")[-1],
        project_root / "src/backbone": "backbone",
        project_root / "src/noise_schedule": "noise_schedule",
    }
    for src_path, dest_name in paths_to_copy.items():
        dest = dest_path / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        _flatten_and_copy(src_path, dest, ignore)
    # Add __init__.py
    (dest_path / "__init__.py").touch()
    # Upload to hub if not local
    if not local:
        api = HfApi()
        log.debug(f"Creating repo or fetching URL for {repo_id}")
        url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        log.debug(f"Found repo at {url}")

        log.debug(f"Uploading files to {repo_id} from {dest_path}")
        commit_info = api.upload_folder(
            folder_path=dest_path,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        log.debug(f"Commit info: {commit_info}")
        log.debug(f"Removing temporary directory {tmp_dir.name}")
        tmp_dir.cleanup()
    log.debug("Done")
