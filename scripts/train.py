#!/usr/bin/env python
"""Train a floorplan diffusion model on ResPlan data.

Usage::

    uv run python scripts/train.py --config configs/resplan_housediff.yaml
    uv run python scripts/train.py --config configs/my.yaml --training.lr 0.0005
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.floorplan_diffusion.data.dataset import CORNER_IDX_DIMS, ROOM_IDX_DIMS
from src.floorplan_diffusion.models.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from src.floorplan_diffusion.models.respace import SpacedDiffusion, space_timesteps
from src.floorplan_diffusion.models.transformer import TransformerModel
from src.floorplan_diffusion.training.data_module import ResPlanDataModule
from src.floorplan_diffusion.training.lightning_module import FloorplanDiffusionModule

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file. Exits if the file is missing or empty."""
    path = Path(config_path)
    if not path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        logger.error("Config file is empty: %s", path)
        sys.exit(1)
    for section in ("model", "diffusion", "training", "data"):
        if section not in cfg:
            logger.error("Config missing required section '%s'", section)
            sys.exit(1)
    return cfg


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted CLI overrides like ``--training.lr 0.001``."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        if i + 1 < len(overrides):
            value = overrides[i + 1]
        else:
            break
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Auto-cast value.
        existing = d.get(parts[-1])
        if isinstance(existing, bool):
            d[parts[-1]] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            d[parts[-1]] = int(value)
        elif isinstance(existing, float):
            d[parts[-1]] = float(value)
        else:
            d[parts[-1]] = value
        i += 2
    return cfg


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_model(cfg: dict) -> TransformerModel:
    """Instantiate the Transformer model from config."""
    analog_bit = cfg["data"]["analog_bit"]
    num_coords = 16 if analog_bit else 2
    in_channels = num_coords + (2 * 8 if not analog_bit else 0)
    condition_channels = 25 + CORNER_IDX_DIMS + ROOM_IDX_DIMS  # room_type + corner_idx + room_idx
    out_channels = num_coords

    return TransformerModel(
        in_channels=in_channels,
        condition_channels=condition_channels,
        model_channels=cfg["model"]["num_channels"],
        out_channels=out_channels,
        dataset="rplan",
        use_checkpoint=False,
        use_unet=False,
        analog_bit=analog_bit,
    )


def create_diffusion(cfg: dict) -> SpacedDiffusion:
    """Instantiate the diffusion process from config."""
    diff_cfg = cfg["diffusion"]
    steps = diff_cfg["steps"]
    betas = get_named_beta_schedule(diff_cfg["noise_schedule"], steps)

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for training."""
    parser = argparse.ArgumentParser(description="Train floorplan diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/resplan_housediff.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args, remaining = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config with CLI overrides.
    cfg = load_config(args.config)
    if remaining:
        cfg = apply_cli_overrides(cfg, remaining)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    logger.info("Config: %s", cfg)

    # --- Data ---
    data_module = ResPlanDataModule(
        pickle_path=data_cfg["pickle_path"],
        cache_dir=data_cfg["cache_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_fraction=data_cfg["val_fraction"],
    )

    # --- Model + Diffusion ---
    model = create_model(cfg)
    diffusion = create_diffusion(cfg)

    lit_module = FloorplanDiffusionModule(
        model=model,
        diffusion=diffusion,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        ema_rate=train_cfg["ema_rate"],
        lr_decay_steps=train_cfg["lr_decay_steps"],
        analog_bit=data_cfg["analog_bit"],
    )

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath="models/checkpoints",
            filename="floorplan-{step}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            every_n_train_steps=train_cfg["save_interval"],
            save_last=True,
        ),
    ]

    # --- Logger ---
    csv_logger = CSVLogger(save_dir="logs", name="floorplan_diffusion")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_steps=train_cfg["max_steps"],
        val_check_interval=train_cfg["val_check_interval"],
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=train_cfg["log_interval"],
        gradient_clip_val=1.0,
        precision="16-mixed" if data_cfg["fp16"] else "32-true",
        accelerator="auto",
        devices="auto",
    )

    # --- Train ---
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=args.resume)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
