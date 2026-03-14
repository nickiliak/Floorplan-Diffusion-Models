# Floorplan Diffusion Models

Diffusion models for floorplan data generation — DTU Special Course.

## Overview

This project recreates [HouseDiffusion](https://github.com/aminshabani/house_diffusion) using the modern [ResPlan](https://github.com/m-agour/ResPlan) dataset (17,000 residential floorplans) instead of RPLAN. The goal is to bridge these two works and later explore state-of-the-art diffusion architectures for improved generation quality.

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Task](https://taskfile.dev/) (task runner)
- Git with submodule support

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd Floorplan-Diffusion-Models

# Install everything (deps + submodules + pre-commit hooks)
task setup
```

## Quick Start

```bash
# Convert ResPlan data to HouseDiffusion format
task data:convert

# Train the model
task train

# Generate samples
task sample

# Run tests
task test
```

## Project Structure

```
src/floorplan_diffusion/    # Main Python package
external/                   # Git submodules (read-only references)
  ├── house_diffusion/      # Original HouseDiffusion
  └── ResPlan/              # ResPlan dataset tools
scripts/                    # CLI entrypoints
configs/                    # Experiment configs (YAML)
notebooks/                  # Exploration notebooks
data/                       # Data directory (gitignored)
models/                     # Checkpoints (gitignored)
```

## References

- [HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising](https://arxiv.org/abs/2211.13287)
- [ResPlan: 17,000 Residential Floor Plans Dataset](https://github.com/m-agour/ResPlan)

## Team

DTU Special Course — Diffusion Models for Floorplan Data Generation
