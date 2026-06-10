#!/bin/sh
### LSF Queue Options
#BSUB -q gpul40s
#BSUB -J train_curriculum_diffusion_res
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 48:00
#BSUB -B
#BSUB -N
#BSUB -o Simplified_%J.out
#BSUB -e Simplified_%J.err

set -e # Exit on error

echo "--------------------------------------------------------"
echo "Job ID: $LSB_JOBID | Node: $(hostname) | Date: $(date)"
echo "--------------------------------------------------------"

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

# ── Phase 1: train on simple plans (150k steps) ───────────────────────────────
echo ">>> [Phase 1] Training on ResPlan_simple (150k steps)"
uv run --no-sync python -m scripts.train \
    --config configs/resplan_housediff_simple.yaml

SIMPLE_CKPT="models/checkpoints/curriculum/simple/last.ckpt"

echo ">>> [Phase 1] Sampling from $SIMPLE_CKPT"
uv run --no-sync python -m scripts.sample \
    --checkpoint "$SIMPLE_CKPT" \
    --pickle_path data/raw/ResPlan_simple.pkl \
    --cache_dir data/processed/simple \
    --num_samples 16 \
    --batch_size 16 \
    --output_dir outputs/curriculum/simple

echo ">>> [Phase 1] Evaluating $SIMPLE_CKPT"
uv run --no-sync python -m scripts.evaluate \
    --checkpoint "$SIMPLE_CKPT" \
    --pickle_path data/raw/ResPlan_simple.pkl \
    --cache_dir data/processed/simple \
    --num_runs 3 \
    --num_samples 500 \
    --batch_size 16 \
    --output_dir "outputs/curriculum/benchmark_simple_${LSB_JOBID:-local}" \
    --device cuda

# ── Phase 2: finetune on full dataset (150k steps, lr=1e-4) ──────────────────
echo ">>> [Phase 2] Finetuning on full ResPlan dataset (150k steps, lr=1e-4)"
uv run --no-sync python -m scripts.train \
    --config configs/resplan_housediff_finetune.yaml \
    --weights_from "$SIMPLE_CKPT"

FINETUNE_CKPT="models/checkpoints/curriculum/finetune/last.ckpt"

echo ">>> [Phase 2] Sampling from $FINETUNE_CKPT"
uv run --no-sync python -m scripts.sample \
    --checkpoint "$FINETUNE_CKPT" \
    --pickle_path data/raw/ResPlan.pkl \
    --cache_dir data/processed \
    --num_samples 16 \
    --batch_size 16 \
    --output_dir outputs/curriculum/finetune

echo ">>> [Phase 2] Evaluating $FINETUNE_CKPT"
uv run --no-sync python -m scripts.evaluate \
    --checkpoint "$FINETUNE_CKPT" \
    --pickle_path data/raw/ResPlan.pkl \
    --cache_dir data/processed \
    --num_runs 3 \
    --num_samples 500 \
    --batch_size 16 \
    --output_dir "outputs/curriculum/benchmark_finetune_${LSB_JOBID:-local}" \
    --device cuda

echo "--------------------------------------------------"
echo "Curriculum training complete. Date: $(date)"
echo "--------------------------------------------------"
