---
description: Guidance for evaluation and visualization workflows
---

# Evaluation

When working on evaluation and visualization:

1. **Metrics**: FID (via pytorch-fid), room count accuracy, adjacency accuracy
2. **Visualization**: Render floorplans using `src/floorplan_diffusion/evaluation/visualize.py`
3. **Figures**: Save to `reports/figures/`
4. **Ground truth**: Compare against data in `data/processed/`
5. **Run evaluation**: `uv run python scripts/evaluate.py --checkpoint models/latest.pt`
6. **Sampling**: `task sample` or `uv run python scripts/sample.py --config configs/resplan_housediff.yaml`
