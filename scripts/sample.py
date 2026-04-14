#!/usr/bin/env python
"""Generate and visualise floorplan samples from a trained checkpoint.

Usage::

    uv run python scripts/sample.py --checkpoint models/checkpoints/last.ckpt
    uv run python scripts/sample.py --checkpoint last.ckpt --num_samples 16 --output_dir outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.floorplan_diffusion.data.dataset import ROOM_TYPE_TO_INT, ResPlanDataset
from src.floorplan_diffusion.models.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from src.floorplan_diffusion.models.respace import SpacedDiffusion, space_timesteps
from src.floorplan_diffusion.models.transformer import TransformerModel
from src.floorplan_diffusion.training.lightning_module import FloorplanDiffusionModule

logger = logging.getLogger(__name__)

# Room-type int → colour for rendering.
ROOM_COLORS: dict[int, str] = {
    1: "#EE4D4D",   # living
    2: "#C67C7B",   # bedroom
    3: "#FFD274",   # kitchen
    4: "#BEBEBE",   # bathroom
    10: "#1F849B",  # balcony
}

INT_TO_ROOM_NAME: dict[int, str] = {v: k for k, v in ROOM_TYPE_TO_INT.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def points_to_room_polygons(
    points: np.ndarray,
    room_types: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    """Group generated points into per-room polygons.

    Args:
        points: ``[100, 2]`` array of (x, y) coordinates in [-1, 1].
        room_types: ``[100, 25]`` one-hot room type per point.
        room_indices: ``[100, 32]`` one-hot room index per point.
        padding_mask: ``[100]`` — 0 for real, 1 for padding.

    Returns:
        List of ``(polygon_coords, room_type_int)`` tuples.
    """
    rooms: dict[int, list[tuple[float, float]]] = {}
    room_type_map: dict[int, int] = {}

    for i in range(points.shape[0]):
        if padding_mask[i] > 0.5:
            continue
        ridx = int(np.argmax(room_indices[i]))
        rtype = int(np.argmax(room_types[i]))
        rooms.setdefault(ridx, []).append((points[i, 0], points[i, 1]))
        room_type_map[ridx] = rtype

    result = []
    for ridx in sorted(rooms.keys()):
        coords = np.array(rooms[ridx])
        result.append((coords, room_type_map[ridx]))
    return result


def render_floorplan(
    room_polygons: list[tuple[np.ndarray, int]],
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Render room polygons on a matplotlib axes.

    Args:
        room_polygons: Output of :func:`points_to_room_polygons`.
        title: Optional title for the plot.
        ax: Optional axes to draw on.

    Returns:
        The matplotlib axes with the rendered floorplan.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    for coords, rtype in room_polygons:
        color = ROOM_COLORS.get(rtype, "#888888")
        label = INT_TO_ROOM_NAME.get(rtype, f"type_{rtype}")
        if len(coords) >= 3:
            poly = plt.Polygon(coords, closed=True, facecolor=color,
                               edgecolor="black", linewidth=1.0, alpha=0.7,
                               label=label)
            ax.add_patch(poly)
        # Draw corner markers.
        ax.scatter(coords[:, 0], coords[:, 1], s=10, c="black", zorder=5)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def compute_graph_accuracy_from_points(
    points: np.ndarray,
    door_mask: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
    adjacency_threshold: float = 0.15,
) -> float:
    """Estimate graph accuracy using generated point coordinates.

    For each pair of rooms that should be connected (door_mask=0 between
    them), check whether the minimum distance between any of their generated
    points is below *adjacency_threshold* (in normalised [-1, 1] space).

    Args:
        points: ``[100, 2]`` generated coordinates.
        door_mask: ``[100, 100]`` attention mask (0 = connected).
        room_indices: ``[100, 32]`` one-hot room index per point.
        padding_mask: ``[100]`` — 0 for real, 1 for padding.
        adjacency_threshold: Max distance to consider rooms adjacent.

    Returns:
        Fraction of expected edges that are spatially satisfied (0–1).
    """
    # Group points by room index.
    ridx_to_pts: dict[int, np.ndarray] = {}
    for i in range(padding_mask.shape[0]):
        if padding_mask[i] > 0.5:
            continue
        ridx = int(np.argmax(room_indices[i]))
        ridx_to_pts.setdefault(ridx, []).append(points[i])
    for k in ridx_to_pts:
        ridx_to_pts[k] = np.array(ridx_to_pts[k])

    room_ids = sorted(ridx_to_pts.keys())
    if len(room_ids) < 2:
        return 1.0

    # Find expected edges from door_mask.
    expected_edges: list[tuple[int, int]] = []
    for a_i, ra in enumerate(room_ids):
        for b_i, rb in enumerate(room_ids):
            if b_i <= a_i:
                continue
            # Pick a representative point from each room.
            pa = next(
                i for i in range(100)
                if padding_mask[i] < 0.5 and int(np.argmax(room_indices[i])) == ra
            )
            pb = next(
                i for i in range(100)
                if padding_mask[i] < 0.5 and int(np.argmax(room_indices[i])) == rb
            )
            if door_mask[pa, pb] < 0.5:
                expected_edges.append((ra, rb))

    if not expected_edges:
        return 1.0

    satisfied = 0
    for ra, rb in expected_edges:
        pts_a = ridx_to_pts[ra]
        pts_b = ridx_to_pts[rb]
        # Minimum pairwise distance.
        diffs = pts_a[:, None, :] - pts_b[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        if dists.min() < adjacency_threshold:
            satisfied += 1

    return satisfied / len(expected_edges)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def create_model_and_diffusion(
    analog_bit: bool = False,
    num_channels: int = 512,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
) -> tuple[TransformerModel, SpacedDiffusion]:
    """Instantiate model and diffusion matching the training config."""
    num_coords = 16 if analog_bit else 2
    in_channels = num_coords + (2 * 8 if not analog_bit else 0)

    model = TransformerModel(
        in_channels=in_channels,
        condition_channels=89,
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
        Generated samples, shape ``[batch, 2, 100]``.
    """
    model.eval()
    batch_size = cond_batch["room_types"].shape[0]
    shape = (batch_size, 2, 100)

    # Move conditioning to device.
    model_kwargs = {k: v.float().to(device) for k, v in cond_batch.items()}

    sample_stack = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=True,
        model_kwargs=model_kwargs,
        analog_bit=analog_bit,
        device=device,
    )
    # p_sample_loop returns [num_final_steps, batch, 2, 100].
    # Take the last timestep as the final sample.
    return sample_stack[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for sampling."""
    parser = argparse.ArgumentParser(description="Sample floorplans from a trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--pickle_path", type=str, default="data/raw/ResPlan.pkl",
        help="Path to ResPlan pickle (for conditioning data)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="data/processed",
        help="Cache directory for processed tensors",
    )
    parser.add_argument("--num_samples", type=int, default=8, help="How many to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for sampling")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument(
        "--analog_bit", action="store_true", default=False,
        help="Use analog mode (default: binary)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # --- Load model from checkpoint ---
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, diffusion = create_model_and_diffusion(analog_bit=args.analog_bit)

    lit_module = FloorplanDiffusionModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        diffusion=diffusion,
        map_location=device,
    )
    lit_module.load_ema_weights()
    model = lit_module.model.to(device)
    model.eval()
    logger.info("Model loaded with EMA weights.")

    # --- Load conditioning data from val split ---
    logger.info("Loading dataset for conditioning...")
    dataset = ResPlanDataset(
        pickle_path=args.pickle_path,
        cache_dir=args.cache_dir,
        set_name="eval",
    )
    n_total = len(dataset)
    num_samples = min(args.num_samples, n_total)
    logger.info("Dataset has %d samples, generating %d.", n_total, num_samples)

    # --- Generate samples ---
    all_accuracies: list[float] = []
    sample_idx = 0

    while sample_idx < num_samples:
        batch_end = min(sample_idx + args.batch_size, num_samples)
        batch_indices = list(range(sample_idx, batch_end))

        # Collate a batch of conditioning.
        batch_items = [dataset[i] for i in batch_indices]
        cond_batch = {}
        for key in batch_items[0][1]:
            cond_batch[key] = torch.stack(
                [torch.from_numpy(np.array(item[1][key])) for item in batch_items]
            )

        # Generate.
        logger.info("Generating batch %d–%d ...", sample_idx, batch_end - 1)
        generated = generate_samples(
            model, diffusion, cond_batch,
            analog_bit=args.analog_bit, device=device,
        )
        generated_np = generated.cpu().numpy()  # [batch, 2, 100]

        # --- Render and evaluate each sample ---
        for b in range(len(batch_indices)):
            idx = batch_indices[b]
            gen_points = generated_np[b].T  # [100, 2]
            gt_points = batch_items[b][0].T  # [100, 2]
            cond = batch_items[b][1]

            room_types = cond["room_types"]
            room_indices = cond["room_indices"]
            padding_mask = cond["src_key_padding_mask"]
            door_mask = cond["door_mask"]

            # Build polygons.
            gen_polys = points_to_room_polygons(
                gen_points, room_types, room_indices, padding_mask,
            )
            gt_polys = points_to_room_polygons(
                gt_points, room_types, room_indices, padding_mask,
            )

            # Graph accuracy.
            accuracy = compute_graph_accuracy_from_points(
                gen_points, door_mask, room_indices, padding_mask,
            )
            all_accuracies.append(accuracy)

            # Render side-by-side: ground truth vs generated.
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            render_floorplan(gt_polys, title="Ground Truth", ax=axes[0])
            render_floorplan(gen_polys, title=f"Generated (adj={accuracy:.2f})", ax=axes[1])

            # De-duplicate legend labels.
            handles, labels = axes[1].get_legend_handles_labels()
            seen = set()
            unique = [
                (h, lab) for h, lab in zip(handles, labels)
                if lab not in seen and not seen.add(lab)
            ]
            if unique:
                fig.legend(*zip(*unique), loc="lower center", ncol=len(unique), fontsize=9)

            fig.tight_layout()
            fig.savefig(output_dir / f"sample_{idx:04d}.png", dpi=150)
            plt.close(fig)
            logger.info(
                "  [%d] graph_accuracy=%.2f  saved to %s",
                idx, accuracy, output_dir / f"sample_{idx:04d}.png",
            )

        sample_idx = batch_end

    # --- Summary ---
    if all_accuracies:
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)
        logger.info(
            "Graph accuracy: %.4f ± %.4f  (n=%d)",
            mean_acc, std_acc, len(all_accuracies),
        )
    logger.info("Sampling complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
