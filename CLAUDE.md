# CLAUDE.md — Floorplan Diffusion Models

## Project Overview
DTU special course (team of 3): Recreate HouseDiffusion using the ResPlan dataset
(17k floorplans) instead of RPLAN. Later, implement additional state-of-the-art
diffusion model architectures for improved generation quality.

## Architecture
- `src/floorplan_diffusion/` — Main Python package (all new code goes here)
- `external/house_diffusion/` — Git submodule (original HouseDiffusion, READ-ONLY)
- `external/ResPlan/` — Git submodule (ResPlan dataset tools, READ-ONLY)
- `scripts/` — CLI entrypoints for training, sampling, evaluation
- `configs/` — YAML experiment configurations
- `notebooks/` — Exploration and analysis notebooks
- `data/` — Raw, interim, and processed data (gitignored, not committed)
- `models/` — Saved checkpoints (gitignored)

## Key Data Flow
1. ResPlan pickle (Shapely polygons + NetworkX graphs) in `data/raw/ResPlan.pkl`,
   extracted from `external/ResPlan/ResPlan.zip` via `task data:download`.
2. `src/floorplan_diffusion/data/dataset.py` (`ResPlanDataset`) reads the pickle
   directly, converts each plan to HouseDiffusion tensors, and caches the result
   as a compressed `.npz` in `data/processed/`. There is NO separate convert step.
3. Diffusion model training via `scripts/train.py` (the data module instantiates
   `ResPlanDataset`, which auto-converts + caches on first run).

## Tensor / .npz Format (conversion target)
`ResPlanDataset` emits a fixed-width array per plan (see `NUM_COLUMNS`, currently
the `c158p192` schema: 192 points, 64 room-idx dims) plus masks, cached as
`resplan_{set}_{pickle_hash}_{schema_tag}_{split_tag}.npz`. Coordinates normalized
to [-1, 1]. Train/eval is a deterministic seeded split over raw pickle indices
(`SPLIT_SEED`); each split caches only its own subset, so `set_name="eval"` is
guaranteed held-out from training.
Reference: `external/house_diffusion/house_diffusion/rplanhg_datasets.py`

## ResPlan Data Format (conversion source)
- Rooms: Shapely Polygons keyed by type (living, bedroom, bathroom, kitchen, balcony)
- Architectural elements: wall (Polygon), door/window/front_door (LineString)
- Graph: NetworkX with nodes=rooms, edges=adjacency/door/window relationships
- Reference: `external/ResPlan/resplan_utils.py`

## Commands
- `task setup` — Install deps, init submodules
- `task data:download` — Extract `ResPlan.pkl` into `data/raw/`
- `task train` — Train model with default config
- `task test` — Run tests
- `task lint` — Run ruff linting
- `task format` — Auto-format code

## Code Conventions
- Linter/formatter: ruff (line-length 100, Python 3.11)
- Type hints on all function signatures
- Docstrings on public functions (Google style)
- Tests for data pipeline components in `tests/`

## Important Constraints
- Do NOT modify files in `external/` — submodules are read-only references
- All new code goes in `src/floorplan_diffusion/` or `scripts/`
- Large files (data, checkpoints) are gitignored — never commit them
- Config changes for experiments go in `configs/` as new YAML files
- Use `uv run` to execute Python scripts (ensures correct environment)

## Git / Commits
- Do NOT add Claude attribution to commits or PRs. No `Co-Authored-By: Claude`
  trailer and no "Generated with Claude Code" lines in commit messages or PR bodies.
