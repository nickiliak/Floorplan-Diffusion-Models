#!/bin/sh
### LSF Queue Options
#BSUB -q gpul40s
#BSUB -J train_eval_gat
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -B
#BSUB -N
#BSUB -o Gat_training_%J.out
#BSUB -e Gat_training_%J.err

set -e

echo "--------------------------------------------------"
echo "Job ID: $LSB_JOBID | Node: $(hostname) | Date: $(date)"
echo "--------------------------------------------------"

# Environment Setup
export PATH="$HOME/.local/bin:$PATH"
cd "$LS_SUBCWD" || { echo "Project directory not found (submit from the repo root)"; exit 1; }

# Module Loading & Verification
module load cuda/12.1
nvidia-smi

# Dependency Sync
echo ">>> Syncing environment with uv..."
uv sync

echo ">>> Validating PyTorch CUDA..."
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" || {
    echo "CRITICAL: CUDA unavailable in Python environment."
    exit 1
}

# --- Part A: Train HGT RoomGraphGAT ---
echo ">>> Training RoomGraphGAT (HGT encoder + one-shot + autoregressive decoders)..."
uv run --no-sync python scripts/train_gat.py \
    --pickle_path   data/raw/ResPlan.pkl \
    --output_dir    models/gat \
    --epochs        250 \
    --save_interval 50 \
    --lr            3e-4 \
    --weight_decay  1e-4 \
    --embed_dim     64 \
    --hidden_dim    64 \
    --num_layers    3 \
    --num_heads     4 \
    --mlp_hidden    128 \
    --dropout       0.1

echo ">>> Training complete."

# --- Part B: Evaluate ---
echo ">>> Evaluating RoomGraphGAT..."
uv run --no-sync python scripts/evaluate_gat.py \
    --config      configs/gat_resplan.yaml \
    --checkpoint  models/gat/gat_best.pt \
    --output_dir  outputs/gat \
    --n_samples   500 \
    --n_vis       12

echo ">>> Evaluation complete."
echo "--------------------------------------------------"
echo "Outputs:"
echo "  models/gat/gat_best.pt"
echo "  outputs/gat/stats/graph_stats_comparison.png"
echo "  outputs/gat/graphs/oneshot_grid.png"
echo "  outputs/gat/graphs/autoregressive_grid.png"
echo "  outputs/gat/graphs/training_grid.png"
echo "  outputs/gat/metrics.json"
echo "--------------------------------------------------"
