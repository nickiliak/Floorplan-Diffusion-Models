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
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from floorplan_diffusion.data.dataset import MAX_NUM_POINTS, ROOM_TYPE_TO_INT, ResPlanDataset
from floorplan_diffusion.evaluation.render import (
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    derive_walls,
    points_to_room_polygons,
)
from floorplan_diffusion.models.sampling import create_model_and_diffusion, generate_samples
from floorplan_diffusion.training.lightning_module import FloorplanDiffusionModule

logger = logging.getLogger(__name__)

INT_TO_ROOM_NAME: dict[int, str] = {v: k for k, v in ROOM_TYPE_TO_INT.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


DOOR_INTS: frozenset[int] = frozenset({ROOM_TYPE_TO_INT["door"], ROOM_TYPE_TO_INT["front_door"]})


def _shapely_to_pathpatch(geom, **kwargs) -> PathPatch | None:
    """Convert a Shapely (Multi)Polygon (with holes) to a matplotlib patch.

    Interior rings become holes via the path's even-odd/winding fill, so a
    wall lattice keeps each room's interior empty.
    """
    from shapely.geometry import Polygon as _Polygon

    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    vertices: list[np.ndarray] = []
    codes: list[int] = []
    for poly in polys:
        if not isinstance(poly, _Polygon) or poly.is_empty:
            continue
        for ring in (poly.exterior, *poly.interiors):
            ring_coords = np.asarray(ring.coords)
            if len(ring_coords) < 3:
                continue
            vertices.append(ring_coords)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(ring_coords) - 1))
    if not vertices:
        return None
    return PathPatch(MplPath(np.concatenate(vertices), codes), **kwargs)


def render_floorplan(
    room_polygons: list[tuple[np.ndarray, int]],
    title: str = "",
    ax: plt.Axes | None = None,
    draw_walls: bool = True,
    wall_thickness: float = 0.025,
) -> plt.Axes:
    """Render room polygons in the ResPlan notebook style.

    Matches ``resplan_utils.plot_plan`` (see
    ``notebooks/01_resplan_exploration.ipynb``): solid ``CATEGORY_COLORS`` fills,
    thin black edges, equal aspect, no axis or corner markers. When
    *draw_walls* is set, a yellow wall lattice synthesized from the room
    boundaries (:func:`derive_walls`) is overlaid between the room fills and the
    doors — derived identically for ground-truth and generated polygons.

    Args:
        room_polygons: Output of :func:`points_to_room_polygons`.
        title: Optional title for the plot.
        ax: Optional axes to draw on.
        draw_walls: Overlay the synthesized wall lattice.
        wall_thickness: Wall width in normalized [-1, 1] units.

    Returns:
        The matplotlib axes with the rendered floorplan.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Layer back-to-front like plot_plan so doors/balconies sit on top of rooms.
    order = {name: i for i, name in enumerate(CATEGORY_ORDER)}
    ordered = sorted(
        room_polygons,
        key=lambda r: order.get(INT_TO_ROOM_NAME.get(r[1], ""), len(order)),
    )

    def _draw(coords: np.ndarray, rtype: int) -> None:
        name = INT_TO_ROOM_NAME.get(rtype, f"type_{rtype}")
        color = CATEGORY_COLORS.get(name, "#000000")
        ax.add_patch(
            plt.Polygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                label=name.replace("_", " "),
            )
        )

    # 1. Room fills (doors deferred so they sit on top of the walls).
    for coords, rtype in ordered:
        if len(coords) >= 3 and rtype not in DOOR_INTS:
            _draw(coords, rtype)

    # 2. Synthesized walls, derived from the same polygons for GT and generated.
    if draw_walls:
        walls = derive_walls(room_polygons, wall_thickness=wall_thickness)
        if walls is not None:
            patch = _shapely_to_pathpatch(
                walls,
                facecolor=CATEGORY_COLORS["wall"],
                edgecolor="black",
                linewidth=0.3,
                label="wall",
            )
            if patch is not None:
                ax.add_patch(patch)

    # 3. Doors on top of the walls (so openings stay visible).
    for coords, rtype in ordered:
        if len(coords) >= 3 and rtype in DOOR_INTS:
            _draw(coords, rtype)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
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
        points: ``[MAX_NUM_POINTS, 2]`` generated coordinates.
        door_mask: ``[MAX_NUM_POINTS, MAX_NUM_POINTS]`` attention mask (0 = connected).
        room_indices: ``[MAX_NUM_POINTS, ROOM_IDX_DIMS]`` one-hot room index per point.
        padding_mask: ``[MAX_NUM_POINTS]`` — 0 for real, 1 for padding.
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
                i
                for i in range(MAX_NUM_POINTS)
                if padding_mask[i] < 0.5 and int(np.argmax(room_indices[i])) == ra
            )
            pb = next(
                i
                for i in range(MAX_NUM_POINTS)
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
        dists = np.sqrt((diffs**2).sum(axis=-1))
        if dists.min() < adjacency_threshold:
            satisfied += 1

    return satisfied / len(expected_edges)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for sampling."""
    parser = argparse.ArgumentParser(description="Sample floorplans from a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        default="data/raw/ResPlan.pkl",
        help="Path to ResPlan pickle (for conditioning data)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/processed",
        help="Cache directory for processed tensors",
    )
    parser.add_argument("--num_samples", type=int, default=8, help="How many to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for sampling")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument(
        "--analog_bit",
        action="store_true",
        default=False,
        help="Use analog mode (default: binary)",
    )
    parser.add_argument(
        "--no_walls",
        action="store_true",
        default=False,
        help="Disable the synthesized wall lattice overlay",
    )
    parser.add_argument(
        "--wall_thickness",
        type=float,
        default=0.025,
        help="Wall width in normalized [-1, 1] units (default: 0.025)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ckpt_stem = Path(args.checkpoint).stem
    output_dir = Path(args.output_dir) / ckpt_stem
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
            model,
            diffusion,
            cond_batch,
            analog_bit=args.analog_bit,
            device=device,
        )
        generated_np = generated.cpu().numpy()  # [batch, 2, MAX_NUM_POINTS]

        # --- Render and evaluate each sample ---
        for b in range(len(batch_indices)):
            idx = batch_indices[b]
            gen_points = generated_np[b].T  # [MAX_NUM_POINTS, 2]
            gt_points = batch_items[b][0].T  # [MAX_NUM_POINTS, 2]
            cond = batch_items[b][1]

            room_types = cond["room_types"]
            room_indices = cond["room_indices"]
            padding_mask = cond["src_key_padding_mask"]
            door_mask = cond["door_mask"]

            # Build polygons.
            gen_polys = points_to_room_polygons(
                gen_points,
                room_types,
                room_indices,
                padding_mask,
            )
            gt_polys = points_to_room_polygons(
                gt_points,
                room_types,
                room_indices,
                padding_mask,
            )

            # Graph accuracy.
            accuracy = compute_graph_accuracy_from_points(
                gen_points,
                door_mask,
                room_indices,
                padding_mask,
            )
            all_accuracies.append(accuracy)

            # Render side-by-side: ground truth vs generated.
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            render_floorplan(
                gt_polys,
                title="Ground Truth",
                ax=axes[0],
                draw_walls=not args.no_walls,
                wall_thickness=args.wall_thickness,
            )
            render_floorplan(
                gen_polys,
                title=f"Generated (adj={accuracy:.2f})",
                ax=axes[1],
                draw_walls=not args.no_walls,
                wall_thickness=args.wall_thickness,
            )

            # De-duplicate legend labels.
            handles, labels = axes[1].get_legend_handles_labels()
            seen = set()
            unique = [
                (h, lab) for h, lab in zip(handles, labels) if lab not in seen and not seen.add(lab)
            ]
            if unique:
                fig.legend(*zip(*unique), loc="lower center", ncol=len(unique), fontsize=9)

            fig.tight_layout()
            fig.savefig(output_dir / f"sample_{idx:04d}.png", dpi=150)
            plt.close(fig)
            logger.info(
                "  [%d] graph_accuracy=%.2f  saved to %s",
                idx,
                accuracy,
                output_dir / f"sample_{idx:04d}.png",
            )

        sample_idx = batch_end

    # --- Summary ---
    if all_accuracies:
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)
        logger.info(
            "Graph accuracy: %.4f ± %.4f  (n=%d)",
            mean_acc,
            std_acc,
            len(all_accuracies),
        )
    logger.info("Sampling complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
