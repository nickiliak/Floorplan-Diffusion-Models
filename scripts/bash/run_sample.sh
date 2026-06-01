#!/bin/sh
### LSF Queue Options
#BSUB -q gpuv100
#BSUB -J sample_house_diffusion_res
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
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

## --- Execution ---
echo ">>> Generating samples"
uv run python -m scripts.sample --checkpoint models/checkpoints/floorplan-step=250000-val/loss=0.0017.ckpt
uv run python -m scripts.sample --checkpoint models/checkpoints/floorplan-step=225000-val/loss=0.0015.ckpt