#!/usr/bin/env python
"""Run HouseDiffusion paper benchmarks (FID, Compatibility, multi-run).

Usage::

    uv run python scripts/evaluate.py --checkpoint models/checkpoints/last.ckpt
    uv run python scripts/evaluate.py --checkpoint last.ckpt --num_runs 5 --num_samples 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from floorplan_diffusion.evaluation.benchmark import run_benchmark


def main() -> None:
    """Entry point for running HouseDiffusion benchmarks."""
    parser = argparse.ArgumentParser(description="Run HouseDiffusion benchmarks")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_benchmark(
        checkpoint_path=args.checkpoint,
        num_runs=args.num_runs,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )

    print(f"\n{'=' * 60}")
    print(f"Benchmark Results ({results.num_runs} runs, {results.per_run[0].num_samples} samples)")
    print(f"{'=' * 60}")
    print(f"FID:           {results.fid_mean:.2f} +/- {results.fid_std:.2f}")
    print(f"Compatibility: {results.compatibility_mean:.2f} +/- {results.compatibility_std:.2f}")
    print(f"{'=' * 60}")

    for i, run in enumerate(results.per_run):
        print(f"  Run {i}: FID={run.fid:.2f}, Compatibility={run.compatibility:.2f}")


if __name__ == "__main__":
    main()
