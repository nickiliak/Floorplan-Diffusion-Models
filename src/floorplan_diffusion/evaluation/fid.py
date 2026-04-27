"""FID (Frechet Inception Distance) wrapper around pytorch-fid."""

from __future__ import annotations

from pathlib import Path


def compute_fid(
    gt_dir: Path,
    pred_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
    dims: int = 2048,
) -> float:
    """Compute FID between two directories of PNG images.

    Args:
        gt_dir: Directory of ground-truth rendered PNGs.
        pred_dir: Directory of predicted rendered PNGs.
        batch_size: InceptionV3 batch size.
        device: Torch device string.
        dims: InceptionV3 feature dimensionality (2048 = pool3).

    Returns:
        FID score (lower is better).
    """
    from pytorch_fid.fid_score import calculate_fid_given_paths

    return calculate_fid_given_paths(
        [str(gt_dir), str(pred_dir)], batch_size, device, dims
    )
