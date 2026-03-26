import logging
import os

import hydra
import torch
import torch.distributed as torch_dist
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
)
from src.denoiser.ar import AR, ARConfig
from src.denoiser.diffusion import BD3LM, MDLM, SEDD, BD3LMConfig, MDLMConfig
from src.noise_schedule.noise_schedules import LinearNoise
from src.utils import fsspec_exists

log = logging.getLogger(__name__)


class RepeatDataloader:
    """Repeat an existing dataloader for k full passes."""

    def __init__(self, dataloader, k: int):
        self.dataloader = dataloader
        self.k = k

    def __iter__(self):
        for _ in range(self.k):
            yield from iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader) * self.k


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    reproducibility.seed_all(cfg.seed)
    reproducibility.configure_deterministic_mode()
    local_rank = dist.get_local_rank()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Setup distributed
    if not dist.is_initialized():
        log.info("Initializing dist")
        dist.initialize_dist(timeout=600)
    log.info("All nodes connected")

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load model
    model_config_overrides = getattr(cfg, "model_config_overrides", None) or {}
    # Convert to plain dict if it's a DictConfig to ensure proper merging
    if isinstance(model_config_overrides, DictConfig):
        model_config_overrides = OmegaConf.to_container(
            model_config_overrides, resolve=True
        )
    if fsspec_exists(os.path.join(cfg.pretrained_model_name_or_path, "config.yaml")):
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
            load_ema_weights=cfg.task.load_ema_weights,
            ckpt_file=cfg.task.ckpt_file,
            verbose=True,
            **model_config_overrides,
        )
    else:
        pretrained_kwargs = {
            "trust_remote_code": True,
            "revision": getattr(cfg, "pretrained_model_revision", None),
        }
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is not None:
            pretrained_kwargs["token"] = hf_token
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path,
                **pretrained_kwargs,
            )
        except Exception:  # Model not compatible with CausalLM
            try:
                model = AutoModelForMaskedLM.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    **pretrained_kwargs,
                )
            except Exception:
                model = None

    # HACK for legacy codebase compatibility
    if model is None or not hasattr(model, "generate"):
        # Create dit backbone config
        # Load the dit config template and update with actual values
        dit_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "model",
            "backbone",
            "dit_legacy.yaml",
        )
        backbone_config = OmegaConf.load(dit_config_path)

        # Update backbone config with necessary parameters
        # (resolving the template values)
        length = getattr(cfg, "length", 1024)
        backbone_config.length = length
        backbone_config.vocab_size = len(tokenizer)
        backbone_config.block_size = getattr(cfg, "block_size", None)
        backbone_config.pretrained_model_name_or_path = getattr(
            cfg, "pretrained_model_name_or_path", None
        )
        backbone_config.num_layers = 12
        backbone_config.n_heads = 12
        backbone_config.hidden_size = 768

        if "-ar-" in backbone_config.pretrained_model_name_or_path:
            backbone_config.adaln = False
            backbone_config.causal_attention = True
            backbone_config.attn_backend = "flash_attn"
        elif "mdlm-" in backbone_config.pretrained_model_name_or_path:
            # backbone_config.attn_backend = "flash_attn"
            backbone_config.adaln = True
        else:
            backbone_config.adaln = True

        # Ensure it's a DictConfig
        if not isinstance(backbone_config, DictConfig):
            backbone_config = OmegaConf.create(
                OmegaConf.to_container(backbone_config, resolve=False)
            )
        if "mdlm-" in backbone_config.pretrained_model_name_or_path:
            model_config = MDLMConfig(
                length=length,
            )
            model_config.backbone_config = OmegaConf.to_container(
                backbone_config, resolve=True
            )
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            for k, v in model_config_overrides.items():
                setattr(model_config, k, v)
            denoiser = MDLM(
                model_config,
                tokenizer=tokenizer,
            )
        elif "sedd-" in backbone_config.pretrained_model_name_or_path:
            model_config = MDLMConfig(
                length=length,
            )
            model_config.backbone_config = OmegaConf.to_container(
                backbone_config, resolve=True
            )
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = SEDD(
                model_config,
                tokenizer=tokenizer,
            )
        elif "ar-" in backbone_config.pretrained_model_name_or_path:
            model_config = ARConfig(
                length=length,
                backbone_config=backbone_config,
            )
            model_config.backbone_config = OmegaConf.to_container(
                backbone_config, resolve=True
            )
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = AR(
                model_config,
                tokenizer=tokenizer,
            )
        else:
            model_config = BD3LMConfig(
                length=length,
                backbone_config=backbone_config,
                block_size=cfg.block_size,
            )
            model_config.backbone_config = OmegaConf.to_container(
                backbone_config, resolve=True
            )
            model_config.keep_clean_bos = True
            model_config.mask_token_id = tokenizer.mask_token_id
            model_config.vocab_size = len(tokenizer)
            denoiser = BD3LM(
                model_config,
                tokenizer=tokenizer,
            )
        if model is not None:
            denoiser.backbone = model.backbone
        else:
            state_dict = torch.load(
                cfg.pretrained_model_name_or_path,
                map_location="cpu",
                weights_only=False,
            )["state_dict"]

            for key in list(state_dict.keys()):
                new_key = key
                if "backbone." in new_key:
                    new_key = new_key.replace("backbone.", "")
                if "_orig_mod." in new_key:
                    new_key = new_key.replace("_orig_mod.", "")

                if new_key != key:
                    state_dict[new_key] = state_dict.pop(key)

            state_dict.pop("sampling_eps_min", None)
            state_dict.pop("sampling_eps_max", None)

            denoiser.backbone.load_state_dict(state_dict)

        model = denoiser.to(device)
        model.noise_schedule = LinearNoise()

    if getattr(cfg, "compile_backbone", False):
        log.info("Compiling model backbone")
        model.backbone = torch.compile(
            model.backbone, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(cfg.task.metrics).values()),
    )

    print(f"Running likelihood eval for {cfg.task.eval_dataset}")
    eval_dataset = hydra.utils.instantiate(
        cfg.task.eval_dataset, tokenizer=tokenizer, max_length=model.config.length
    )

    collator = hydra.utils.instantiate(
        cfg.task.collator,
        rank=dist.get_global_rank(),
        world_size=dist.get_world_size(),
        tokenizer=tokenizer,
        max_length=model.config.length,
    )
    eval_sampler = dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)

    eval_dataloader = hydra.utils.instantiate(
        cfg.task.eval_dataloader,
        _convert_="partial",
        dataset=eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
    )
    num_importance_samples = getattr(cfg, "num_importance_samples", 1)
    if num_importance_samples > 1:
        eval_dataloader = RepeatDataloader(eval_dataloader, num_importance_samples)

    if hasattr(cfg, "composer") and hasattr(cfg.composer, "callbacks"):
        callbacks = hydra.utils.instantiate(cfg.composer.callbacks)
        callbacks = list(callbacks.values())
    else:
        callbacks = []

    trainer = hydra.utils.instantiate(
        cfg.task.trainer,
        _convert_="all",
        model=model,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
    )
    trainer.eval()
    print(
        "\nEval Metrics:\n\t"
        + "\n\t".join(
            [
                f"{k}: {v.item():0.4f}"
                for k, v in trainer.state.eval_metric_values.items()
            ]
        )
    )

    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
