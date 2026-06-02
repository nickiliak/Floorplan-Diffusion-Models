#!/bin/sh
### LSF Queue Options
#BSUB -q gpuv100
#BSUB -J eval_house_diffusion_res
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
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
cd ~/Floorplan-Diffusion-Models-BH || { echo "Project directory not found"; exit 1; }

# Module Loading & Verification
module load cuda/12.1
nvidia-smi

# Dependency Sync and Validation
echo ">>> Syncing environment with uv (incl. eval extras)..."
uv sync --extra eval

echo ">>> Validating PyTorch CUDA..."
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || {
    echo "CRITICAL: CUDA unavailable in Python environment."
    exit 1
}

## --- Execution ---
CHECKPOINT="${CHECKPOINT:-models/checkpoints/last.ckpt}"
NUM_RUNS="${NUM_RUNS:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/benchmark_${LSB_JOBID:-local}}"

echo ">>> Running benchmark"
echo "    checkpoint:  $CHECKPOINT"
echo "    num_runs:    $NUM_RUNS"
echo "    num_samples: $NUM_SAMPLES"
echo "    batch_size:  $BATCH_SIZE"
echo "    output_dir:  $OUTPUT_DIR"

uv run --no-sync python -m scripts.evaluate \
    --checkpoint "$CHECKPOINT" \
    --num_runs "$NUM_RUNS" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
