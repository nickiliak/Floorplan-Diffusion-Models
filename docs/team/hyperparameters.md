# Hyperparameter Reference

Training configuration for the floorplan diffusion model. All values match the
original HouseDiffusion paper settings (`external/house_diffusion/scripts/script.sh`)
unless noted otherwise.

Config file: `configs/resplan_housediff.yaml`
Code defaults: `scripts/train.py` `DEFAULT_CONFIG`

---

## Optimizer: AdamW

AdamW decouples weight decay from the gradient update, applying decay directly to
the parameters rather than through the gradient. This prevents the regularisation
effect from being scaled by the adaptive learning rate, which is important for
transformers where different parameters can have very different gradient magnitudes.

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr` | `1e-3` | Matches original HouseDiffusion. Coupled with batch_size=512 via the linear scaling rule (lr scales linearly with batch size). |
| `weight_decay` | `0.05` | Matches original. Prevents overfitting on the ~10k-plan dataset. AdamW applies decay as `param = param * (1 - lr * weight_decay)` each step. |
| `betas` | `(0.9, 0.999)` | PyTorch AdamW defaults. beta1 controls momentum (how much past gradients influence direction), beta2 controls the second moment (adaptive per-parameter learning rate). |
| `eps` | `1e-8` | PyTorch default. Prevents division by zero in the adaptive rate. |

### Linear scaling rule

When increasing batch size, the learning rate should scale proportionally to maintain
similar training dynamics. The original uses `batch_size=512, lr=1e-3`. If you need
a smaller batch (e.g. OOM), scale lr accordingly:

- batch_size=256 -> lr=5e-4
- batch_size=128 -> lr=2.5e-4
- batch_size=64  -> lr=1.25e-4

---

## Batch Size

| Parameter | Value | Why |
|-----------|-------|-----|
| `batch_size` | `512` | Matches original HouseDiffusion. Larger batches give more stable gradient estimates, which is important for diffusion models where each sample sees a random timestep. |

The model is ~26.5M parameters with sequence length 100. With fp16 mixed precision,
batch_size=512 should fit on a V100 32GB. If it OOMs, reduce to 256 and halve the
learning rate.

---

## Learning Rate Schedule

The schedule has two phases:

### 1. Linear warmup (steps 0 to `warmup_steps`)

LR ramps linearly from 0 to `lr` over the first 2000 steps. This prevents gradient
explosions when the model is randomly initialised and the loss landscape is steep.
Without warmup, a high learning rate (1e-3) can cause the model to diverge in the
first few hundred steps.

### 2. Step decay (every `lr_decay_steps`)

After warmup, the LR is multiplied by 0.1 every 100k steps:
- Steps 0-2000: warmup from 0 to 1e-3
- Steps 2000-100k: lr = 1e-3
- Steps 100k-200k: lr = 1e-4
- Steps 200k+: lr = 1e-5

| Parameter | Value | Why |
|-----------|-------|-----|
| `warmup_steps` | `2000` | ~2k steps is standard for transformer training. Short enough to not waste training time, long enough to stabilise gradients. Added because the original HouseDiffusion did not use warmup but we are using a higher learning rate. |
| `lr_decay_steps` | `100000` | Matches original. Coarse-to-fine learning: the model learns the broad structure first, then refines details at lower learning rates. |

---

## EMA (Exponential Moving Average)

| Parameter | Value | Why |
|-----------|-------|-----|
| `ema_rate` | `0.9999` | Standard for diffusion models. |

EMA maintains a slowly-updating copy of the model weights:
`ema_param = ema_rate * ema_param + (1 - ema_rate) * model_param`

At rate 0.9999, the EMA weights are effectively an average over the last ~10,000
updates. This produces smoother, more stable weights for inference/sampling.
During training, the regular (non-EMA) weights are used. At sampling time,
EMA weights are loaded via `lit_module.load_ema_weights()`.

---

## Diffusion

| Parameter | Value | Why |
|-----------|-------|-----|
| `steps` | `1000` | Standard DDPM timestep count. More steps = finer denoising trajectory but slower sampling. |
| `noise_schedule` | `cosine` | Cosine schedule (Nichol & Dhariwal 2021) provides more uniform signal-to-noise ratio across timesteps compared to linear. Better for structured outputs like floorplans where fine details matter. |
| `model_mean_type` | `EPSILON` | Model predicts the noise added to the data (epsilon-prediction). Standard for DDPM. |
| `model_var_type` | `FIXED_LARGE` | Uses the upper bound of the posterior variance. Simpler than learned variance and works well in practice. |
| `loss_type` | `MSE` | Mean squared error between predicted and actual noise. |

### Cosine schedule

Beta values are derived from: `alpha_bar(t) = cos((t/T) * pi/2)^2`

This ensures the signal-to-noise ratio decreases smoothly across timesteps.
Early timesteps have very little noise (model learns global structure), late
timesteps have almost pure noise (model learns to denoise from scratch).

---

## Training Duration

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_steps` | `250000` | Original HouseDiffusion was sampled at step 250k (`model250000.pt`). With our smaller dataset (~10k vs ~80k RPLAN), this gives sufficient passes over the data. |
| `check_val_every_n_epoch` | `40` | Validate every 40 epochs. With batch_size=512 and ~13.5k training samples, each epoch is ~26 batches, so 40 epochs ≈ 1040 steps. Epoch-based validation avoids errors when the batch count per epoch is smaller than the interval (which happens with large batch sizes on small datasets). |
| `save_interval` | `10000` | Save checkpoint every 10k steps. Keeps top-3 best by val/loss plus the last checkpoint. |

No early stopping is used. Diffusion models can have non-monotonic loss curves
where the loss plateaus or slightly increases before improving again. Early
stopping would prematurely terminate training in these cases.

---

## Data Augmentation

Applied in `dataset.py` `__getitem__` during training only:

1. **Random 90-degree rotation** (4 orientations) - floorplans are rotation-invariant
2. **Random horizontal flip** (50% probability) - mirror symmetry
3. **Random vertical flip** (50% probability) - mirror symmetry

Combined: 4 rotations x 2 h-flip x 2 v-flip = **16x effective augmentation**.
This is important because our dataset (~10k plans) is much smaller than RPLAN (~80k).

---

## Mixed Precision (fp16)

| Parameter | Value | Why |
|-----------|-------|-----|
| `fp16` | `true` | Enables PyTorch AMP (Automatic Mixed Precision). |

Mixed precision stores activations in float16 and parameters/gradients in float32.
On V100 with tensor cores this gives ~1.5-2x training throughput and reduces memory
usage (allowing larger batch sizes). PyTorch Lightning handles the GradScaler
automatically.

---

## Infrastructure (DTU HPC)

Configured in `scripts/bash/run_train.sh`:

| Setting | Value | Why |
|---------|-------|-----|
| GPU queue | `gpuv100` | V100 32GB, sufficient for this model |
| Walltime | `24:00` | Maximum allowed. At ~250k steps this should complete in ~10-15h with fp16. |
| Memory | `16GB` | Headroom for batch_size=512 with data loading |
| CPUs | `4` | 3 for DataLoader workers + 1 for main process |

### Resume support

If the job exceeds walltime, resume from the last checkpoint:

```bash
uv run --no-sync python -m scripts.train --resume models/checkpoints/last.ckpt
```

---

## Data Filtering

Two filters drop plans from the dataset during processing:

### Filter 1: Plans with >100 total vertices
Plans where the sum of all room polygon corners exceeds 100 are rejected entirely.
This is the model's maximum sequence length (MAX_NUM_POINTS). Affects ~12.2% of
the dataset (~2,086 of 17,107 plans). Recovering these requires simplifying polygons
across all rooms until the total drops below 100 — a v2 task that needs tolerance
tuning and visual validation.

### Filter 2: Rooms with >32 corners
If any single room in a plan has more than 32 polygon corners, the **entire plan**
is rejected. This affects ~268 plans (~1.78% of kept plans). The 32-corner limit
comes from the one-hot corner index encoding (32 dimensions).

Previously these rooms were silently skipped (`continue`), which corrupted the
training data — the plan's graph still referenced the dropped room, so other rooms
lost their adjacency connections in the attention masks. Clean rejection eliminates
this corrupted training signal at the cost of a negligible number of plans.

**Future improvement (v2):** Use `polygon.simplify(tolerance)` to reduce complex
rooms below 32 corners, then validate that the simplified geometry preserves the
room's area and shape faithfully before accepting it. This would recover the ~268
plans without risking geometric distortion.
