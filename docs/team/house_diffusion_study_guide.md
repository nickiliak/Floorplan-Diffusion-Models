# HouseDiffusion Codebase Study Guide

A progressive, self-guided walkthrough of `external/house_diffusion/` for the team.
Complete the stages in order — each one builds on the previous.

All file paths are relative to `external/house_diffusion/`.

---

## Architecture Overview

HouseDiffusion is a **Transformer-based conditional diffusion model** that generates vector
floorplans as sets of room corner points. Given a conditioning graph (room types + adjacency),
the model denoises a cloud of random points into a valid, structured floor plan.

```
                         TRAINING
  ┌──────────────────────────────────────────────────────────────┐
  │  JSON file                                                    │
  │    └─ reader()               room_types, boxes, edges        │
  │         └─ RPlanhgDataset.__getitem__()                      │
  │              │  corners: [N, 2]  (normalized to [-1, 1])     │
  │              │  cond:    [N, 89] (room type + indices)       │
  │              │  masks:   door_mask, self_mask, gen_mask      │
  │              ▼                                               │
  │         q_sample(x_0, t)  →  x_t  (add noise)              │
  │              ▼                                               │
  │         TransformerModel(x_t, t, cond)  →  ε_pred           │
  │              ▼                                               │
  │         MSE(ε_pred, ε_true)  →  backward  →  AdamW          │
  └──────────────────────────────────────────────────────────────┘

                         INFERENCE
  ┌──────────────────────────────────────────────────────────────┐
  │  condition graph (room types + adjacency)                    │
  │    └─ build synthetic layout (RPlanhgDataset, set_name=eval) │
  │         └─ x_T ~ N(0, I)                                    │
  │              ▼                                               │
  │         for t = 999 … 0:                                     │
  │           μ, σ = p_mean_variance(model, x_t, t, cond)       │
  │           x_{t-1} = μ + σ·ε,   ε ~ N(0,I)                  │
  │              ▼                                               │
  │         clip to [-1, 1]  →  denormalize  →  render          │
  └──────────────────────────────────────────────────────────────┘
```

**Key numbers to keep in mind:**
- Max points per floorplan: **100**
- Channels per point: **2** (continuous) or **16** (discrete/analog_bit)
- Conditioning channels per point: **89** (25 room-type + 32 corner-index + 32 room-index)
- Transformer hidden dim: **512**
- EncoderLayers: **4**
- Diffusion timesteps: **1000** (cosine schedule, default)

---

## Stage 0 — Orientation (≈30 min)

**Goal:** Get a map of the territory before reading any code deeply.

**Read:**
- [`README.md`](../../external/house_diffusion/README.md) — installation, pre-trained model link, usage examples
- [`requirements.txt`](../../external/house_diffusion/requirements.txt) — full dependency list
- [`scripts/script.sh`](../../external/house_diffusion/scripts/script.sh) — the two canonical CLI invocations
- [`house_diffusion/script_util.py:1-56`](../../external/house_diffusion/house_diffusion/script_util.py) — all default hyperparameters and dataset-specific channel counts

**Key facts to absorb:**
- The project is a fork of OpenAI's `guided-diffusion`
- Dataset is RPLAN (17k apartments) in JSON format
- The `analog_bit` flag switches between two output representations:
  - `False` (default): model outputs 2D continuous coordinates
  - `True`: model outputs 16-bit binary coords (8 bits per axis) — sharper but slower
- `target_set` filters floorplans by room count (e.g. `8` = 8-room apartments)

**Checkpoint:** Can you describe HouseDiffusion in two sentences? Can you name every file in
`house_diffusion/` and say what it does in one line?

---

## Stage 1 — The Diffusion Math (≈1–2 h)

**Goal:** Understand the probabilistic core — forward noising and reverse denoising.

### 1a. Beta schedules and the forward process

**Read:** [`house_diffusion/gaussian_diffusion.py:1-200`](../../external/house_diffusion/house_diffusion/gaussian_diffusion.py)

The **forward process** gradually destroys a clean sample `x_0` by adding Gaussian noise:

```
x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε,    ε ~ N(0, I)
```

where `ᾱ_t = ∏_{i=0}^{t} (1 − β_i)` is the cumulative noise product.

- `get_named_beta_schedule()` (line 19): produces the β_t array for T steps
  - `"linear"`: β rises linearly from ~0.0001 to ~0.02
  - `"cosine"`: β derived from a cosine α̅ curve — slower noise at start/end, better for small objects
- `betas_for_alpha_bar()` (line 48): general helper to derive β from any α̅ function
- `GaussianDiffusion.__init__()` (line 104): precomputes all schedule tensors stored as `self.*`
  - `self.alphas_cumprod` — ᾱ_t for each t (shape `[T]`)
  - `self.sqrt_alphas_cumprod` — √ᾱ_t (used in `q_sample`)
  - `self.posterior_mean_coef1/2` — used in reverse process
- `q_sample()` (line 191): implements the formula above — this is used every training step

**The three enums** (lines 68–103) control what the model predicts and how loss is computed:
- `ModelMeanType.EPSILON`: predict the noise ε (most common, default)
- `ModelMeanType.START_X`: predict x_0 directly
- `ModelMeanType.PREVIOUS_X`: predict x_{t-1}
- `ModelVarType`: whether variance is fixed or learned
- `LossType`: MSE (default) or KL divergence

### 1b. The reverse process

**Read:** [`house_diffusion/gaussian_diffusion.py:191-540`](../../external/house_diffusion/house_diffusion/gaussian_diffusion.py)

- `p_mean_variance()` (line 235): given model output at timestep t, compute the reverse-step
  Gaussian parameters μ and σ. Handles all ModelMeanType/ModelVarType combos.
- `p_sample()` (line 424): single reverse step — samples x_{t-1} from p(x_{t-1}|x_t)
- `p_sample_loop()` (line 472): the full inference loop — runs p_sample T times

### 1c. Training objective

**Read:** [`house_diffusion/gaussian_diffusion.py:787-end`](../../external/house_diffusion/house_diffusion/gaussian_diffusion.py)

- `training_losses()` (line 787): the loss function used during training
  - Calls `q_sample` to create `x_t`
  - Calls the model to get prediction
  - Returns MSE between prediction and target (ε or x_0)

### 1d. Inference acceleration

**Read:** [`house_diffusion/respace.py`](../../external/house_diffusion/house_diffusion/respace.py)

- `SpacedDiffusion`: subclass of `GaussianDiffusion` that uses a subset of timesteps
- `space_timesteps()`: maps original 1000 steps → fewer steps (e.g. 250-step DDIM-style sampling)
- The default config uses all 1000 steps; this file enables faster inference if needed

**Read:** [`house_diffusion/resample.py`](../../external/house_diffusion/house_diffusion/resample.py)

- `UniformSampler`: picks timestep t uniformly at random during training
- `LossAwareSampler`: weights harder timesteps more heavily (variance reduction)

**Checkpoint:** Given a trained model and a noisy sample x_t at step t=500, can you walk through
`p_mean_variance` → `p_sample` on paper and describe what comes out? Do you understand why
`alphas_cumprod` is precomputed rather than computed on-the-fly?

---

## Stage 2 — Data Pipeline (≈1–2 h)

**Goal:** Understand how a raw JSON floorplan becomes a training tensor.

**Read:** [`house_diffusion/rplanhg_datasets.py`](../../external/house_diffusion/house_diffusion/rplanhg_datasets.py) — full file (537 lines)

### 2a. JSON format and `reader()`

`reader()` (line 511) parses one JSON file and returns:
- `boxes`: room bounding boxes `[x0, y0, x1, y1]` (normalized to `[0,1]`)
- `edges`: door/window edge segments `[x0, y0, x1, y1]`
- `room_type`: integer per room (0–24)
- `ed_rm`: which rooms each edge connects

The 25 room type integers map to: living room, master bedroom, kitchen, bathroom, dining room,
child bedroom, study, second bathroom, guest room, balcony, entrance/corridor, and 14 reserved.

### 2b. `RPlanhgDataset.__getitem__()` — the central function

`__getitem__()` (line 318) converts one floorplan into all tensors needed for training:

**Step 1: Contour extraction**
- Room bounding boxes → pixel masks (256×256)
- OpenCV `findContours` → corner points per room
- Normalize to `[-1, 1]`

**Step 2: Corner encoding**
Each point gets a **94-channel** feature vector:

| Channels | Content | Encoding |
|----------|---------|----------|
| 0–1 | (x, y) coordinates | float, normalized |
| 2–26 | Room type | 25-dim one-hot |
| 27–58 | Corner index within room | 32-dim one-hot |
| 59–90 | Room index | 32-dim one-hot |
| 91 | Padding mask | 1 = valid, 0 = padding |
| 92–93 | Connection indices (current, next) | int indices |

All sequences are **zero-padded to 100 points**.

The model input (`x`) uses only channels 0–1 (or 0–15 in discrete mode).
The conditioning (`cond`) uses channels 2–90 = **89 channels**.

> **Important:** Channels 92–93 store which corner follows which around the room polygon
> (e.g. corner 3 → corner 4). During the model forward pass, `expand_points()` uses these
> connections to interpolate 8 additional points between each pair of adjacent corners,
> expanding the sequence from 100 → 900 tokens. See Stage 3, Step 2 for details.

**Step 3: Attention masks** (each `[100 × 100]` binary)

| Mask | Meaning | How used |
|------|---------|----------|
| `self_mask` | Two corners are in the **same room** | Room corners can attend to each other |
| `door_mask` | Two rooms share a **door/adjacency** | Cross-room attention along connections |
| `gen_mask` | A corner is **padding** | Block attention to/from padding tokens |

These masks are passed to the Transformer's three attention heads — they are the key
mechanism that encodes floor-plan structure into the attention computation.

**Step 4: Analog bit mode** (`analog_bit=True`)
Instead of returning raw coordinates, corners are quantized to 8-bit integers (0–255),
then represented as 16 binary values (8 bits × x, 8 bits × y) = 16 input channels.

### 2c. Synthetic eval set

When `set_name='eval'`, the dataset generates synthetic (empty) layouts:
- Sample the number of corners per room from the training distribution
- Initialize all coordinates to zero
- The model fills them in during sampling

### 2d. DataLoader

`load_rplanhg_data()` (line 20) wraps everything in a `DataLoader` (2 workers, shuffle for train)
and yields batches **infinitely** (the training loop is infinite).

**Checkpoint:** Draw the shape of one batch element:
```
x:          [2]   (or [16] if analog_bit)
cond:       [89]
door_mask:  [100, 100]
self_mask:  [100, 100]
```
Then explain: why does `door_mask` allow attention *between* rooms but `self_mask` only
*within* a room?

---

## Stage 3 — Transformer Architecture (≈1–2 h)

**Goal:** Understand how the model processes noisy coordinates and conditions to predict noise.

**Read:** [`house_diffusion/transformer.py`](../../external/house_diffusion/house_diffusion/transformer.py) — full file (275 lines)

**Read:** [`house_diffusion/nn.py`](../../external/house_diffusion/house_diffusion/nn.py) — focus on `timestep_embedding`, `mean_flat`, `update_ema`

### 3a. Building blocks

**`PositionalEncoding`** (line 12): standard sinusoidal position embeddings applied to the
sequence dimension (the 100 corner tokens).

**`MultiHeadAttention`** (line 53): standard scaled dot-product attention with 4 heads and
d_model=512. Accepts a binary `mask` — positions where mask=1 are filled with −1e9 (effectively
zeroed after softmax).

**`FeedForward`** (line 33): two linear layers with activation, standard Transformer FFN.

**`EncoderLayer`** (line 85): one Transformer block with **three separate attention heads**:
```python
def forward(self, x, door_mask, self_mask, gen_mask):
    x = self_attn(x, x, x, self_mask)   # within-room attention
    x = door_attn(x, x, x, door_mask)   # cross-room (door/adjacency)
    x = gen_attn(x, x, x, gen_mask)     # blocks padding tokens
    x = feed_forward(x)
    return x
```
Each head has its own parameters. This triple-attention design lets the model separately learn
room-internal geometry and inter-room spatial relationships.

### 3b. `TransformerModel.forward()` — line by line

`forward(x, timesteps, xtalpha, epsalpha, is_syn=False, **kwargs)` (line 215):

```
Input:  x           [B, C_in, 100]  — noisy corners
        timesteps   [B]             — current diffusion step
        **kwargs    door_mask, self_mask, gen_mask, condition (89 channels)
```

**Step 1: Permute** `[B, C, 100]` → `[B, 100, C]` (sequence-first for Transformer)

**Step 2: Point expansion** (default mode, `analog_bit=False`, line 163)
`expand_points()` densifies the corner sequence by inserting 8 interpolated points between
each pair of adjacent corners (9× upsampling: 100 → 900 tokens). This runs in the **default**
mode (`if not self.analog_bit:` at line 228) — it is skipped only when `analog_bit=True`.

The interpolation uses **recursive binary midpoint averaging**:
```
Given adjacent corners p1 and p5 (looked up via the connections array):
  p3 = avg(p1, p5)       — midpoint
  p2 = avg(p1, p3)       — 1/4 point
  p4 = avg(p3, p5)       — 3/4 point
  p1.5 = avg(p1, p2)     — 1/8 point
  p2.5 = avg(p2, p3)     — 3/8 point
  p3.5 = avg(p3, p4)     — 5/8 point
  p4.5 = avg(p4, p5)     — 7/8 point

Output per edge: p1, p1.5, p2, p2.5, p3, p3.5, p4, p4.5, p5
```
The expanded points are `.detach()`ed — no gradient flows through the interpolation itself.
This gives the model a denser point representation for finer geometric reasoning.

**Step 3: Time embedding** (S = 900 after expansion, or 100 if `analog_bit=True`)
```python
t_emb = timestep_embedding(timesteps, 128)  # sinusoidal, [B, 128]
t_emb = MLP(t_emb)                          # → [B, 512]
t_emb = t_emb.unsqueeze(1).expand(-1, S, -1)  # broadcast to all tokens
```

**Step 4: Input embedding**
```python
x_emb = linear(x)   # [B, S, 512]
```

**Step 5: Condition embedding**
```python
cond_emb = linear(condition)  # [B, S, 512]
```

**Step 6: Combine and encode**
```python
h = x_emb + cond_emb + t_emb              # [B, S, 512]
h = positional_encoding(h)
for layer in encoder_layers:               # 4 layers
    h = layer(h, door_mask, self_mask, gen_mask)
```

**Step 7: Continuous output head**
```python
out = MLP(h)   # [B, 100, 2]  — predicted coordinates or noise
```

**Step 8: Discrete output head** (when `analog_bit=False`, last 32 timesteps only)
A second set of EncoderLayers processes `h` to predict 8-bit binary coordinates:
```python
h2 = discrete_encoder_layers(h, ...)
bin_out = sigmoid(linear(h2))  # [B, 100, 16]  — 16 binary values
```
The model switches to discrete mode near the end of sampling (t < 32) to sharpen boundaries.

**Output:** `[B, C_out, 100]` — permuted back from `[B, 100, C_out]`

### 3c. `create_image()` — visualization

`create_image()` (line 182) renders a floorplan from corner coordinates:
- Draws each room as a polygon (corners connected in order)
- Colors rooms by type
- Returns a PIL image for logging

**Checkpoint:** Given input shape `[4, 2, 100]` and condition shape `[4, 100, 89]`, trace the
tensor shapes through every step of `TransformerModel.forward()`. Where does the mask shape
`[4, 100, 100]` appear and how does `MultiHeadAttention` handle it?

---

## Stage 4 — Training Loop (≈1 h)

**Goal:** Understand how one training step works end-to-end, including EMA and distributed
training.

**Read:** [`house_diffusion/train_util.py`](../../external/house_diffusion/house_diffusion/train_util.py) — full file (311 lines)

**Read:** [`scripts/image_train.py`](../../external/house_diffusion/scripts/image_train.py) — full file (90 lines)

**Skim:** [`house_diffusion/fp16_util.py`](../../external/house_diffusion/house_diffusion/fp16_util.py) and [`house_diffusion/dist_util.py`](../../external/house_diffusion/house_diffusion/dist_util.py)

### 4a. Training entry point (`image_train.py`)

```
parse args
  └─ setup_dist()                       # NCCL multi-GPU init
       └─ logger.configure()            # stdout + optional TensorBoard
            └─ create_model_and_diffusion()
                 └─ load_rplanhg_data()
                      └─ TrainLoop(model, diffusion, data, ...).run_loop()
```

### 4b. `TrainLoop.run_loop()` (line 155)

The training loop is **infinite** — it runs until manually stopped:
```python
while True:
    batch, cond = next(data)
    self.run_step(batch, cond)
    if step % log_interval == 0: log()
    if step % save_interval == 0: save()
```

### 4c. `run_step()` → `forward_backward()` (lines 179, 187)

```python
def run_step(self, batch, cond):
    self.forward_backward(batch, cond)   # compute loss, backprop
    self.mp_trainer.optimize(self.opt)   # AdamW parameter update
    self._update_ema()                   # EMA weight update
    self._anneal_lr()                    # optional LR decay
```

Inside `forward_backward()`:
```python
t, weights = schedule_sampler.sample(batch_size)     # sample timesteps
x_t = diffusion.q_sample(x_0, t, noise)             # add noise
model_output = model(x_t, t, **cond)                 # forward pass
losses = diffusion.training_losses(model, x_0, t, cond)
loss = (losses['loss'] * weights).mean()
loss.backward()
```

### 4d. EMA (Exponential Moving Average)

EMA maintains a shadow copy of the model weights that updates slowly:
```
ema_params = rate * ema_params + (1 - rate) * model_params
```
Default rate = **0.9999** — the EMA weights change very little each step.

**Why?** The EMA model is more stable than the raw model and produces better samples.
The checkpoint saved as `ema_*.pt` is what you use for inference.

`_update_ema()` (called via `nn.update_ema()`) runs after every optimizer step.

### 4e. Mixed precision and distributed training

- `MixedPrecisionTrainer` (`fp16_util.py`): model runs in FP16, master params in FP32,
  dynamic loss scaling prevents gradient underflow
- `dist_util.setup_dist()`: NCCL backend, broadcasts rank-0 params at init
- Single-GPU mode works without `mpirun` by checking `MPI.COMM_WORLD.Get_size() == 1`

### 4f. Checkpoints

`save()` (line 242) writes three files every N steps:
- `model{step}.pt` — raw model weights
- `ema_{rate}_{step}.pt` — EMA weights (use this for inference)
- `opt{step}.pt` — optimizer state

**Checkpoint:** Write pseudocode for one complete training step (data → checkpoint update).
Where exactly does the `schedule_sampler` interact with the loss?

---

## Stage 5 — Sampling / Inference (≈1 h)

**Goal:** Understand how a trained model generates floorplans and how quality is measured.

**Read:** [`scripts/image_sample.py`](../../external/house_diffusion/scripts/image_sample.py) — full file (378 lines)

### 5a. Setup

```bash
python image_sample.py \
  --dataset rplan --batch_size 32 --set_name eval \
  --target_set 8 --model_path ckpts/exp/ema_0.9999_250000.pt \
  --num_samples 64
```

The script:
1. Creates the same model architecture (must match training args)
2. Loads the EMA checkpoint (not the raw model)
3. Creates a synthetic eval dataset (zero-initialized corners, real condition graphs)
4. Runs the sampling loop
5. Saves output images, computes FID

### 5b. Sampling loop

The main loop is `diffusion.p_sample_loop()` from Stage 1. Each call returns one denoised
batch. The model_kwargs passed as conditions are:
```python
model_kwargs = {
    'condition': cond_tensor,    # [B, 100, 89]
    'door_mask': door_mask,      # [B, 100, 100]
    'self_mask': self_mask,      # [B, 100, 100]
    'gen_mask':  gen_mask,       # [B, 100, 100]
    'xtalpha':   ...,            # schedule param for discrete mode
    'epsalpha':  ...,            # schedule param for discrete mode
}
```

### 5c. Discrete mode: `bin_to_int_sample()`

When `analog_bit=False`, the discrete output head (active at t < 32) produces 16 binary values
per point. `bin_to_int_sample()` converts binary → 8-bit integer → normalized float coordinate.

### 5d. Output rendering

`image_sample.py` contains several `save_samples_*` functions:
- `save_samples_hd()` — renders PNG images from corner arrays using Pillow
- SVG output via `drawSvg` (vector graphics)
- GIF animation of the denoising trajectory (when requested)

### 5e. Evaluation metrics

- **FID (Fréchet Inception Distance)**: distribution-level quality score via `pytorch_fid`
  - Samples N generated floorplans, compares feature distribution to real dataset
  - Lower is better; ~10–20 is good for this domain
- **Graph accuracy**: `get_graph()` and `estimate_graph()` build NetworkX graphs from
  generated layouts by checking room polygon overlaps, then compare to the condition graph

**Checkpoint:** Starting from `--model_path ckpts/exp/ema_0.9999_250000.pt`, trace the
execution of `image_sample.py` from argument parsing to writing the first output PNG. What
does the script do between loading the model and starting the sampling loop?

---

## Stage 6 — End-to-End Mental Model (≈30 min)

**Goal:** Synthesize everything into one coherent picture.

No new files to read. Work from memory.

### Exercise A: Full data-flow diagram

Draw (on paper or in a text file) both data flows:

**Training:**
```
JSON file
  → reader()
  → RPlanhgDataset.__getitem__()   [tensors: x, cond, masks]
  → q_sample(x_0, t)               [noisy: x_t]
  → TransformerModel(x_t, t, cond) [prediction: ε_pred]
  → MSE(ε_pred, ε)                 [loss]
  → AdamW + EMA update
```

**Inference:**
```
condition graph
  → synthetic __getitem__()         [x_T = zeros, cond, masks]
  → x_T ~ N(0, I)                   [replace zeros with noise]
  → p_sample_loop (t=999…0)         [denoising]
    └→ p_mean_variance(model, x_t, t, cond)
    └→ x_{t-1} = μ + σ·ε
  → denormalize → render → PNG/SVG
```

### Exercise B: ResPlan vs RPLAN differences

This directly informs your work on `scripts/convert_resplan.py`. The conversion must produce
the exact JSON format that `reader()` (line 511) expects.

| Aspect | RPLAN (original) | ResPlan (our dataset) | Impact |
|--------|------------------|-----------------------|--------|
| Room geometry | Pixel masks → OpenCV contours | Shapely Polygons (already vector) | `reader()` uses `boxes` from JSON, not raw masks |
| Architectural elements | Doors/windows as edge segments | `LineString` geometry | Must encode `ed_rm` mapping |
| Graph structure | Implicit from pixel adjacency | Explicit NetworkX graph | Richer, can be used directly |

The conversion pipeline:
1. ResPlan Shapely polygon → bounding box `[x0, y0, x1, y1]` normalized to `[0, 1]`
2. ResPlan door `LineString` → edge segment `[x0, y0, x1, y1]`
3. NetworkX node types → integer room type (must match the 25-class encoding)
4. Write JSON matching the schema that `reader()` parses

### Exercise C: The three attention masks

Without looking at the code, explain why each mask is needed:
1. `self_mask`: ___
2. `door_mask`: ___
3. `gen_mask`: ___

Then check your answers against `RPlanhgDataset.__getitem__()` (lines ~380–430).

---

## Quick Reference: Key Tensors and Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x` (input to model) | `[B, 2, 100]` or `[B, 16, 100]` | Noisy corner coordinates (continuous or discrete) |
| `cond` (condition) | `[B, 100, 89]` | Room type + corner/room indices |
| `door_mask` | `[B, 100, 100]` | 1 where attention is blocked (adjacent rooms) |
| `self_mask` | `[B, 100, 100]` | 1 where attention is blocked (same room) |
| `gen_mask` | `[B, 100, 100]` | 1 for padding tokens |
| `x_t` after q_sample | `[B, 2, 100]` | Noisy version of x at timestep t |
| After `expand_points()` | `[B, 900, C]` | 9× densified sequence (default mode only) |
| Transformer hidden | `[B, 900, 512]`\* | Internal sequence representation |
| Model output | `[B, 2, 100]` | Predicted ε (noise) or x_0 |

\* `[B, 100, 512]` when `analog_bit=True` (no point expansion).

---

## Glossary

**Alpha bar (ᾱ_t):** Cumulative product of `(1 − β_i)` from step 0 to t. Controls how much
signal remains in x_t: at t=0, ᾱ=1 (clean signal); at t=T, ᾱ≈0 (pure noise).

**Analog bit mode:** When `analog_bit=True`, coordinates are quantized to 8-bit integers and
represented as 16 binary values. Gives sharper boundaries than continuous regression.

**Beta schedule:** The sequence of noise levels β_0, …, β_{T−1}. "Cosine" schedule (default)
adds noise more slowly at the start and end than "linear", which helps with small structures.

**Door mask / Self mask / Gen mask:** Binary `[100 × 100]` matrices that control which pairs
of corner tokens can attend to each other in the Transformer. Encodes floorplan structure
directly into the attention mechanism.

**EMA (Exponential Moving Average):** A shadow model whose weights are a slow-moving average
of the training model. Used for inference because it's more stable. Rate 0.9999 means the EMA
changes ≈0.01% per step.

**epsilon (ε) prediction:** The model predicts the noise that was added to x_0 to get x_t,
rather than predicting x_0 directly. Equivalent formulations, but ε-prediction tends to
produce sharper outputs.

**p_sample_loop:** The full denoising loop: starts from x_T ~ N(0,I) and applies p_sample
1000 times to recover x_0. This is inference.

**q_sample:** The forward process: given clean x_0 and timestep t, returns noisy x_t.
Used only during training.

**SpacedDiffusion:** A subclass that uses a subset of the 1000 timesteps (e.g. every 4th step)
for faster inference, at the cost of some quality.

**target_set:** Filters the RPLAN dataset by number of rooms (e.g. `8` = 8-room layouts).
Controls the complexity of the training distribution.

---

## Reading Order Summary

| # | File | Lines | Time | Purpose |
|---|------|-------|------|---------|
| 1 | `house_diffusion/gaussian_diffusion.py` | 979 | 1–2 h | Core diffusion math |
| 2 | `house_diffusion/rplanhg_datasets.py` | 537 | 1–2 h | Data pipeline |
| 3 | `house_diffusion/transformer.py` | 275 | 1 h | Model architecture |
| 4 | `house_diffusion/nn.py` | 172 | 20 min | Utility functions |
| 5 | `house_diffusion/train_util.py` | 311 | 1 h | Training loop |
| 6 | `scripts/image_train.py` | 90 | 15 min | Training entry point |
| 7 | `scripts/image_sample.py` | 378 | 1 h | Inference + evaluation |
| 8 | `house_diffusion/script_util.py` | 173 | 20 min | Config defaults |
| 9 | `house_diffusion/respace.py` | 128 | 20 min | Accelerated inference |
| 10 | `house_diffusion/resample.py` | 154 | 20 min | Timestep sampling |
| 11 | `house_diffusion/fp16_util.py` | 236 | 20 min | Mixed precision |
