"""Model and diffusion instantiation + reverse-diffusion sampling.

Provides factory functions for building the Transformer + SpacedDiffusion
pair and running the reverse process to generate floorplan samples.
"""

from __future__ import annotations

import torch

from floorplan_diffusion.data.dataset import CORNER_IDX_DIMS, MAX_NUM_POINTS
from floorplan_diffusion.models.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from floorplan_diffusion.models.respace import SpacedDiffusion, space_timesteps
from floorplan_diffusion.models.transformer import TransformerModel


def create_model_and_diffusion(
    analog_bit: bool = False,
    num_channels: int = 512,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
) -> tuple[TransformerModel, SpacedDiffusion]:
    """Instantiate model and diffusion matching the training config.

    Args:
        analog_bit: Use analog-bit coordinate encoding.
        num_channels: Transformer hidden dimension.
        diffusion_steps: Total diffusion timesteps.
        noise_schedule: Beta schedule name (e.g. ``"cosine"``).

    Returns:
        ``(model, diffusion)`` tuple ready for checkpoint loading.
    """
    num_coords = 16 if analog_bit else 2
    in_channels = num_coords + (2 * 8 if not analog_bit else 0)

    model = TransformerModel(
        in_channels=in_channels,
        # room_type(25) + corner_idx(CORNER_IDX_DIMS) + room_idx(32); must match
        # the dataset feature layout (see ResPlanDataset.__getitem__).
        condition_channels=25 + CORNER_IDX_DIMS + 32,
        model_channels=num_channels,
        out_channels=num_coords,
        dataset="rplan",
        use_checkpoint=False,
        use_unet=False,
        analog_bit=analog_bit,
    )

    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, [diffusion_steps]),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
    )

    return model, diffusion


@torch.no_grad()
def generate_samples(
    model: TransformerModel,
    diffusion: SpacedDiffusion,
    cond_batch: dict[str, torch.Tensor],
    analog_bit: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Run the reverse diffusion process to generate floorplan samples.

    Args:
        model: Trained Transformer model (in eval mode).
        diffusion: The diffusion process.
        cond_batch: Dict of conditioning tensors (batched).
        analog_bit: Analog vs binary mode.
        device: Torch device.

    Returns:
        Generated samples, shape ``[batch, 2, MAX_NUM_POINTS]``.
    """
    model.eval()
    batch_size = cond_batch["room_types"].shape[0]
    shape = (batch_size, 2, MAX_NUM_POINTS)

    # Move conditioning to device.
    model_kwargs = {f"syn_{k}": v.float().to(device) for k, v in cond_batch.items()}

    sample_stack = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=True,
        model_kwargs=model_kwargs,
        analog_bit=analog_bit,
        device=device,
    )
    # p_sample_loop returns [num_final_steps, batch, 2, MAX_NUM_POINTS].
    # Take the last timestep as the final sample.
    return sample_stack[-1]
