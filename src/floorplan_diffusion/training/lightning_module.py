"""PyTorch Lightning module wrapping the Transformer + GaussianDiffusion pipeline.

Handles training/validation steps, EMA weight maintenance, LR scheduling,
and metric logging.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch

from ..models.nn import update_ema
from ..models.resample import UniformSampler

logger = logging.getLogger(__name__)


class FloorplanDiffusionModule(pl.LightningModule):
    """Lightning wrapper for HouseDiffusion-style training.

    Args:
        model: The ``TransformerModel`` instance.
        diffusion: A ``SpacedDiffusion`` (or ``GaussianDiffusion``) instance.
        lr: Learning rate for AdamW.
        weight_decay: AdamW weight decay coefficient.
        warmup_steps: Number of linear warmup steps before applying the decay schedule.
        ema_rate: Exponential moving average decay rate for model parameters.
        lr_decay_steps: Apply a 10x LR decay every this many global steps.
            Set to 0 to disable step-decay scheduling.
        analog_bit: Whether the model operates in analog (True) or binary (False) mode.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: Any,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 2000,
        ema_rate: float = 0.9999,
        lr_decay_steps: int = 100_000,
        analog_bit: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.ema_rate = ema_rate
        self.lr_decay_steps = lr_decay_steps
        self.analog_bit = analog_bit

        self.sampler = UniformSampler(diffusion)
        
        # EMA parameters — a deep copy of model params stored as buffers.
        self._ema_params: list[torch.Tensor] = [
            p.clone().detach() for p in self.model.parameters()
        ]

        self.save_hyperparameters(ignore=["model", "diffusion"])

    # Move params to gpu NEEDFIX
    def on_train_start(self) -> None:
        """Called by Lightning when training begins (after moving to GPU)."""
        self._ema_params = [p.to(self.device) for p in self._ema_params]
        
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Run one training step: sample timestep, compute diffusion loss."""
        x_start, cond = batch
        x_start = x_start.float()
        cond = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in cond.items()}

        t, weights = self.sampler.sample(x_start.shape[0], x_start.device)
        terms = self.diffusion.training_losses(
            self.model, x_start, t, cond, analog_bit=self.analog_bit
        )

        loss = (terms["loss"] * weights).mean()
        self.log("train/loss", loss, prog_bar=True)

        # Log sub-losses when available.
        if "mse_dec" in terms:
            self.log("train/mse_dec", terms["mse_dec"].mean())
        if "mse_bin" in terms:
            self.log("train/mse_bin", terms["mse_bin"].mean())

        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update EMA parameters after each training step."""
        update_ema(self._ema_params, list(self.model.parameters()), rate=self.ema_rate)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Compute validation loss (same as training, no augmentation)."""
        x_start, cond = batch
        x_start = x_start.float()
        cond = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in cond.items()}

        t, weights = self.sampler.sample(x_start.shape[0], x_start.device)
        terms = self.diffusion.training_losses(
            self.model, x_start, t, cond, analog_bit=self.analog_bit
        )

        loss = (terms["loss"] * weights).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW optimizer with linear warmup and step-decay LR scheduling."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        if self.lr_decay_steps <= 0 and self.warmup_steps <= 0:
            return {"optimizer": optimizer}

        warmup = max(1, self.warmup_steps)
        decay_steps = self.lr_decay_steps

        def lr_lambda(step: int) -> float:
            # Linear warmup from 0 to 1 over warmup steps.
            if step < warmup:
                return step / warmup
            # 10x decay every lr_decay_steps (if enabled).
            if decay_steps > 0:
                return 0.1 ** (step // decay_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ------------------------------------------------------------------
    # EMA utilities
    # ------------------------------------------------------------------

    def ema_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a state dict with EMA parameters mapped to model keys.

        Useful for loading EMA weights into the model for sampling/eval.
        """
        state = {}
        for (name, _), ema_p in zip(self.model.named_parameters(), self._ema_params):
            state[name] = ema_p.clone()
        return state

    def load_ema_weights(self) -> None:
        """Copy EMA parameters into the model (for inference)."""
        for param, ema_p in zip(self.model.parameters(), self._ema_params):
            param.data.copy_(ema_p.data)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist EMA params alongside the regular checkpoint."""
        checkpoint["ema_params"] = [p.cpu().clone() for p in self._ema_params]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore EMA params from checkpoint."""
        if "ema_params" in checkpoint:
            self._ema_params = [p.to(self.device) for p in checkpoint["ema_params"]]
