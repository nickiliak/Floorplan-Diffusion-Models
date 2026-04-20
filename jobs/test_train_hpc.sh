#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
### ------------- specify job name ----------------
#BSUB -J housediff-resplan
### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=8GB]"
### ------------- specify wall-clock time (max allowed is 24:00) ----------------
#BSUB -W 12:00
#BSUB -o outputs/jobs/train_%J.out
#BSUB -e outputs/jobs/train_%J.err
set -euo pipefail
echo "========================================"
echo "Job ID:   $LSB_JOBID"
echo "Host:     $(hostname)"
echo "Started:  $(date)"
echo "========================================"

# ── Environment setup ────────────────────────────────────────
module load cuda/11.8
module load python3/3.11.13

# ── Always run from the project root ────────────────────────
cd /zhome/70/7/219373/Floorplan-Diffusion-Models

# NOTE: outputs/jobs/ must exist before submission because LSF creates the
#  #BSUB -o/-e files before this script runs. Run `mkdir -p outputs/jobs` once
# before calling `bsub < jobs/test_train_hpc.sh`.
mkdir -p outputs/jobs data/processed

source .venv/bin/activate

# ── Note on LR ────────────────────────────────────────────────────────────────
# test_train.py hardcodes lr=1e-4; the paper uses 1e-3.
# Acceptable for a test run, but training will converge slower.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo "Training HouseDiffusion + ResPlan"
echo "NOTE: First run will preprocess all 14k plans (~30-60 min)"
echo "      Processed cache will be saved to data/processed/"
echo "------------------------------------------------------------"
# Shorter test run (~2-4 hours). Change num_steps to 250000 for full paper run.
python tests/test_train.py \
    --pkl data/raw/ResPlan.pkl \
    --device cuda \
    --num_steps 10000 \
    --batch_size 64 \
    --diffusion_steps 1000 \
    --num_channels 512 \
    --target_set 8

echo ""
echo "============================================================"
echo "Training complete at $(date)"
echo "============================================================"