#!/bin/sh
### LSF Queue Options
#BSUB -q gpuv100
#BSUB -J train_house_diffusion_res
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 18GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

set -e # Exit on error

echo "--------------------------------------------------"
echo "Job ID: $LSB_JOBID | Node: $(hostname) | Date: $(date)"
echo "--------------------------------------------------"

# Environment Setup
export PATH="$HOME/.local/bin:$PATH"
cd ~/Floorplan-Diffusion-Models || { echo "Project directory not found"; exit 1; }

# Module Loading & Verification
module load cuda/12.1
nvidia-smi

# Dependency Sync and Validation
echo ">>> Syncing environment with uv..."
uv sync

echo ">>> Validating PyTorch CUDA..."
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || {
    echo "CRITICAL: CUDA unavailable in Python environment."
    exit 1
}

## --- Execution: Part A ---
echo ">>> Initiate training of ddpm"
uv run --no-sync python -m scripts.train --config configs/resplan_housediff_def.yaml