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
    num_runs: int
    per_run: list[BenchmarkResult] = field(default_factory=list)


def run_benchmark(
    checkpoint_path: Path,
    num_runs: int = 5,
    num_samples: int = 1000,
    batch_size: int = 16,
    output_dir: Path = Path("outputs/benchmark"),
    device: str = "cuda",
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
    dataset = ResPlanDataset(set_name="eval")
    n_total = len(dataset)
    actual_samples = min(num_samples, n_total)
    if actual_samples < num_samples:
        logger.warning(
            "Dataset has only %d samples, using %d instead of requested %d",
            n_total,
            actual_samples,
            num_samples,
        )

    per_run_results: list[BenchmarkResult] = []

    for run_idx in range(num_runs):
        logger.info("=== Run %d/%d ===", run_idx + 1, num_runs)
        run_dir = output_dir / f"run_{run_idx}"
        gt_dir = run_dir / "gt"
        pred_dir = run_dir / "pred"

        all_gt_polys: list[list[tuple[np.ndarray, int]]] = []
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
            generated_np = generated.cpu().numpy()  # [batch, 2, 100]

            for b in range(len(batch_indices)):
                gen_points = generated_np[b].T  # [100, 2]
                gt_points = batch_items[b][0].T  # [100, 2]
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
                gt_polys = points_to_room_polygons(
                    gt_points,
                    room_types,
                    room_indices,
                    padding_mask,
                )

                all_gt_polys.append(gt_polys)
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
        logger.info("Rendering %d GT and predicted PNGs...", len(all_gt_polys))
        render_batch_to_dir(all_gt_polys, gt_dir)
        render_batch_to_dir(all_pred_polys, pred_dir)

        # Compute FID.
        fid_device = device if torch.cuda.is_available() else "cpu"
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
        num_runs=num_runs,
        per_run=per_run_results,
    )
