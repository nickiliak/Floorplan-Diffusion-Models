"""Multi-run benchmark orchestration for HouseDiffusion paper metrics.

Coordinates multiple independent sampling runs, renders PNGs, computes
FID and Compatibility, and aggregates results with mean +/- std.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from floorplan_diffusion.evaluation.compatibility import estimate_graph_errors
from floorplan_diffusion.evaluation.fid import compute_fid
from floorplan_diffusion.evaluation.render import (
    points_to_room_polygons,
    render_batch_to_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    fid: float
    compatibility: float  # mean edge errors per sample
    num_samples: int


@dataclass
class AggregatedResults:
    """Aggregated results across multiple runs."""

    fid_mean: float
    fid_std: float
    compatibility_mean: float
    compatibility_std: float
    fid_floor: float
    num_runs: int
    per_run: list[BenchmarkResult] = field(default_factory=list)


def _symlink_half_dirs(gt_dir: Path) -> tuple[Path, Path]:
    """Split rendered GT PNGs into two disjoint half-directories via symlinks.

    Used to measure the FID floor: FID between two identically-distributed
    halves of the ground truth is pure finite-sample + rendering noise.

    Args:
        gt_dir: Directory of rendered ground-truth PNGs.

    Returns:
        ``(half_a, half_b)`` directory paths (even/odd indexed files).
    """
    half_a = gt_dir.parent / f"{gt_dir.name}_half_a"
    half_b = gt_dir.parent / f"{gt_dir.name}_half_b"
    files = sorted(gt_dir.glob("*.png"))
    for dst_dir, half in ((half_a, files[0::2]), (half_b, files[1::2])):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in half:
            link = dst_dir / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    return half_a, half_b


def run_benchmark(
    checkpoint_path: Path,
    num_runs: int = 5,
    num_samples: int = 1000,
    batch_size: int = 16,
    output_dir: Path = Path("outputs/benchmark"),
    device: str = "cuda",
    pickle_path: Path = Path("data/raw/ResPlan.pkl"),
    cache_dir: Path = Path("data/processed"),
) -> AggregatedResults:
    """Run the full HouseDiffusion benchmark suite.

    Performs *num_runs* independent sampling passes, computing FID and
    Compatibility for each, then aggregates with mean +/- std.

    Args:
        checkpoint_path: Path to a Lightning ``.ckpt`` file.
        num_runs: Number of independent sampling runs.
        num_samples: Samples per run.
        batch_size: Batch size for reverse diffusion.
        output_dir: Root directory for rendered PNGs.
        device: Torch device string.

    Returns:
        Aggregated benchmark results.
    """
    from floorplan_diffusion.data.dataset import ResPlanDataset
    from floorplan_diffusion.models.sampling import create_model_and_diffusion, generate_samples
    from floorplan_diffusion.training.lightning_module import FloorplanDiffusionModule

    logger.info("Loading model from %s", checkpoint_path)
    model, diffusion = create_model_and_diffusion()
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    lit_module = FloorplanDiffusionModule.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        diffusion=diffusion,
        map_location=torch_device,
    )
    lit_module.load_ema_weights()
    model = lit_module.model.to(torch_device)
    model.eval()

    logger.info("Loading validation dataset...")
    dataset = ResPlanDataset(
        pickle_path=pickle_path,
        cache_dir=cache_dir,
        set_name="eval",
    )
    n_total = len(dataset)
    actual_samples = min(num_samples, n_total)
    if actual_samples < num_samples:
        logger.warning(
            "Dataset has only %d samples, using %d instead of requested %d",
            n_total,
            actual_samples,
            num_samples,
        )

    fid_device = device if torch.cuda.is_available() else "cpu"

    # --- Ground truth: identical across runs, so build + render it once ---
    logger.info("Rendering %d ground-truth PNGs...", actual_samples)
    all_gt_polys: list[list[tuple[np.ndarray, int]]] = []
    for i in range(actual_samples):
        arr, cond = dataset[i]
        all_gt_polys.append(
            points_to_room_polygons(
                arr.T,
                cond["room_types"],
                cond["room_indices"],
                cond["src_key_padding_mask"],
            )
        )
    gt_dir = output_dir / "gt"
    render_batch_to_dir(all_gt_polys, gt_dir)

    # --- FID floor: GT half vs GT half measures finite-sample + render noise.
    # The halves are n/2 images each, and FID is biased high at smaller n, so
    # this floor is conservative relative to the n-vs-n model FID below.
    half_a, half_b = _symlink_half_dirs(gt_dir)
    fid_floor = compute_fid(half_a, half_b, batch_size=64, device=fid_device)
    logger.info("FID floor (GT vs GT, %d images per half): %.2f", actual_samples // 2, fid_floor)

    per_run_results: list[BenchmarkResult] = []

    for run_idx in range(num_runs):
        logger.info("=== Run %d/%d ===", run_idx + 1, num_runs)
        pred_dir = output_dir / f"run_{run_idx}" / "pred"

        all_pred_polys: list[list[tuple[np.ndarray, int]]] = []
        all_graph_errors: list[int] = []

        sample_idx = 0
        while sample_idx < actual_samples:
            batch_end = min(sample_idx + batch_size, actual_samples)
            batch_indices = list(range(sample_idx, batch_end))

            # Collate conditioning batch.
            batch_items = [dataset[i] for i in batch_indices]
            cond_batch: dict[str, torch.Tensor] = {}
            for key in batch_items[0][1]:
                cond_batch[key] = torch.stack(
                    [torch.from_numpy(np.array(item[1][key])) for item in batch_items]
                )

            # Generate samples.
            generated = generate_samples(
                model,
                diffusion,
                cond_batch,
                device=torch_device,
            )
            generated_np = generated.cpu().numpy()  # [batch, 2, MAX_NUM_POINTS]

            for b in range(len(batch_indices)):
                gen_points = generated_np[b].T  # [MAX_NUM_POINTS, 2]
                cond = batch_items[b][1]

                room_types = cond["room_types"]
                room_indices = cond["room_indices"]
                padding_mask = cond["src_key_padding_mask"]
                door_mask = cond["door_mask"]

                gen_polys = points_to_room_polygons(
                    gen_points,
                    room_types,
                    room_indices,
                    padding_mask,
                )
                all_pred_polys.append(gen_polys)

                # Graph compatibility errors.
                errors = estimate_graph_errors(
                    gen_polys,
                    room_types,
                    room_indices,
                    padding_mask,
                    door_mask,
                )
                all_graph_errors.append(errors)

            sample_idx = batch_end

        # Render PNGs for FID.
        logger.info("Rendering %d predicted PNGs...", len(all_pred_polys))
        render_batch_to_dir(all_pred_polys, pred_dir)

        # Compute FID against the shared GT renders.
        fid_score = compute_fid(gt_dir, pred_dir, batch_size=64, device=fid_device)
        mean_errors = float(np.mean(all_graph_errors))

        logger.info("Run %d: FID=%.2f, Compatibility=%.2f", run_idx, fid_score, mean_errors)
        per_run_results.append(
            BenchmarkResult(fid=fid_score, compatibility=mean_errors, num_samples=actual_samples)
        )

    # Aggregate.
    fids = np.array([r.fid for r in per_run_results])
    compats = np.array([r.compatibility for r in per_run_results])

    return AggregatedResults(
        fid_mean=float(fids.mean()),
        fid_std=float(fids.std()),
        compatibility_mean=float(compats.mean()),
        compatibility_std=float(compats.std()),
        fid_floor=float(fid_floor),
        num_runs=num_runs,
        per_run=per_run_results,
    )
