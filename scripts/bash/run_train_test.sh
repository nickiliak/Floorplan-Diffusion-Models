#!/bin/sh
### LSF Queue Options — TEST (low resources, fast turnaround)
#BSUB -q gpuv100
#BSUB -J train_test
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:20
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
cd "$LS_SUBCWD" || { echo "Project directory not found (submit from the repo root)"; exit 1; }

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
echo ">>> Initiate TEST training (500 steps, batch_size=16)"
uv run --no-sync python -m scripts.train --config configs/resplan_housediff_test.yaml
