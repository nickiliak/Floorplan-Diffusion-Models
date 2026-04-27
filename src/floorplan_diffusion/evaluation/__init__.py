"""Evaluation metrics for floorplan diffusion models.

Implements the HouseDiffusion paper benchmarks:
- FID (Frechet Inception Distance) via rendered 256x256 PNGs
- Compatibility (graph edge accuracy via polygon IoU reconstruction)
- Multi-run aggregation (5 passes, mean +/- std)
"""

from floorplan_diffusion.evaluation.compatibility import estimate_graph_errors
from floorplan_diffusion.evaluation.fid import compute_fid
from floorplan_diffusion.evaluation.render import render_floorplan_png

__all__ = ["compute_fid", "estimate_graph_errors", "render_floorplan_png"]
