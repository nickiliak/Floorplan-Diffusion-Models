# CLAUDE.md — Floorplan Diffusion Models

## Project Overview
DTU special course (team of 3): Recreate HouseDiffusion using the ResPlan dataset
(17k floorplans) instead of RPLAN. Later, implement additional state-of-the-art
diffusion model architectures for improved generation quality.

## Architecture
- `src/floorplan_diffusion/` — Main Python package (all new code goes here)
- `external/house_diffusion/` — Git submodule (original HouseDiffusion, READ-ONLY)
- `external/ResPlan/` — Git submodule (ResPlan dataset tools, READ-ONLY)
- `scripts/` — CLI entrypoints for data conversion, training, sampling, evaluation
- `configs/` — YAML experiment configurations
- `notebooks/` — Exploration and analysis notebooks
- `data/` — Raw, interim, and processed data (gitignored, not committed)
- `models/` — Saved checkpoints (gitignored)

## Key Data Flow
1. ResPlan pickle (Shapely polygons + NetworkX graphs) in `data/raw/`
2. `scripts/convert_resplan.py` → JSON files in `data/processed/`
3. `src/floorplan_diffusion/data/dataset.py` → PyTorch tensors
4. Diffusion model training via `scripts/train.py`

## HouseDiffusion JSON Format (conversion target)
Each JSON file contains:
- `boxes`: Room bounding boxes as `[x0, y0, x1, y1]` (normalized)
- `edges`: Edge line segments as `[x0, y0, x1, y1]`
- `room_type`: Integer array (0-24 encoding)
- `ed_rm`: Edge-to-room index mapping

Room types are one-hot encoded to 25 dims. Coordinates normalized to [-1, 1].
Reference: `external/house_diffusion/house_diffusion/rplanhg_datasets.py`

## ResPlan Data Format (conversion source)
- Rooms: Shapely Polygons keyed by type (living, bedroom, bathroom, kitchen, balcony)
- Architectural elements: wall (Polygon), door/window/front_door (LineString)
- Graph: NetworkX with nodes=rooms, edges=adjacency/door/window relationships
- Reference: `external/ResPlan/resplan_utils.py`

## Commands
- `task setup` — Install deps, init submodules, install pre-commit hooks
- `task data:convert` — Run ResPlan → HouseDiffusion conversion
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
