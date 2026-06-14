#!/bin/sh
### LSF Queue Options
#BSUB -q gpuv100
#BSUB -J sample_progression_res
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
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
# Sample the same conditioning plans from three training checkpoints into three
# separate folders, to visualise how generation quality progresses over training.
#
# Ground truth is IDENTICAL across the three runs by construction: sample.py
# pulls conditioning from the deterministic, seeded "eval" split by index, so
# requesting the same NUM_SAMPLES from each checkpoint reuses the exact same
# plans. Only the generated (predicted) side differs between checkpoints.
#
# sample.py appends the checkpoint file stem to --output_dir, so each step lands
# in its own subdirectory under outputs/training_progression/<step>/.

NUM_SAMPLES=16
ROOT="outputs/training_progression"

# step label -> checkpoint path (note: Lightning wrote "val/loss" as a subdir,
# so each metric checkpoint lives in its own folder).
run_step() {
    step="$1"
    ckpt="$2"
    echo ">>> Sampling step ${step} from ${ckpt}"
    uv run python -m scripts.sample \
        --checkpoint "${ckpt}" \
        --num_samples "${NUM_SAMPLES}" \
        --style architectural \
        --output_dir "${ROOT}/step_${step}"
}

run_step 100000 "models/checkpoints/floorplan-step=100000-val/loss=0.0008.ckpt"
run_step 200000 "models/checkpoints/floorplan-step=200000-val/loss=0.0009.ckpt"
run_step 250000 "models/checkpoints/floorplan-step=250000-val/loss=0.0007.ckpt"

echo ">>> Done. Samples in ${ROOT}/step_{100000,200000,250000}/"
