---
description: Guidance for model training workflows
---

# Training

When working on model training:

1. **Training entrypoint**: `scripts/train.py`
2. **Config files**: `configs/*.yaml` — YAML-based experiment configuration
3. **Model wrappers**: Import HouseDiffusion components from the submodule via
   `src/floorplan_diffusion/models/house_diffusion.py`
4. **Experiment tracking**: Use wandb (configured in config YAML)
5. **Checkpoints**: Save to `models/` (gitignored, never committed)
6. **Debugging**: Use a `--debug` flag to limit to a small data subset
7. **Run training**: `task train` or `uv run python scripts/train.py --config configs/resplan_housediff.yaml`
8. **Multi-GPU**: Use PyTorch DDP (not MPI as in original HouseDiffusion)
