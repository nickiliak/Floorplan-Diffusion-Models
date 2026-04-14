# Architecture Approach: ResPlan + HouseDiffusion

How we adapt HouseDiffusion to work with the ResPlan dataset. Three approaches were
considered; **Plan C (Hybrid)** is the chosen path.

---

## The Problem

HouseDiffusion is a Transformer-based conditional diffusion model that generates vector
floorplans. It was built for the **RPLAN** dataset (pixel masks, bounding boxes, per-plan
JSON files). We want to train it on **ResPlan** (Shapely vector polygons, NetworkX graphs,
single pickle file) — a richer, more modern dataset.

The two datasets differ in format, room type encoding, geometry representation, and graph
structure. We need a strategy that gets us training as fast as possible without throwing
away ResPlan's advantages.

See the [ResPlan Study Guide](resplan_study_guide.md) for a detailed comparison of the
two data formats (Stage 5).

---

## Plan A — Convert & Reuse

**Convert ResPlan → RPLAN JSON files, run HouseDiffusion unmodified.**

Write `scripts/convert_resplan.py` to produce one JSON file per floorplan matching the
schema that HouseDiffusion's `reader()` expects (`boxes`, `edges`, `room_type`, `ed_rm`).
Then train using the original codebase.

**Pros:**
- Fastest path to a first training run
- Zero changes to model or training code
- Easy to validate (if `reader()` parses it, it works)

**Cons:**
- **Lossy conversion**: ResPlan has rich arbitrary polygons (L-shapes, irregular rooms),
  but RPLAN represents rooms as bounding boxes → pixel masks → OpenCV contour corners
  (typically 4 rectangular vertices). Converting polygons to bounding boxes throws away
  the actual room geometry.
- The 100-vertex cap forces filtering out 12.2% of plans or simplifying polygons, but
  even the "simple" plans lose their polygon shape when reduced to bounding boxes.
- Room type mapping is lossy (6 types → 25, many unused slots).
- Maintaining a fragile intermediate JSON format that serves no purpose beyond compatibility.
- HouseDiffusion's training loop uses MPI for distribution, a custom infinite loop, and
  manual checkpoint management — not modern best practices.

**Verdict:** Gets a baseline fast but wastes ResPlan's main advantage (vector geometry).

---

## Plan B — Full Rewrite in PyTorch Lightning

**Rewrite everything from scratch: model, diffusion, data pipeline, training loop.**

Build a clean PyTorch Lightning codebase in `src/floorplan_diffusion/` that natively
handles ResPlan data and implements the diffusion model with modern tooling.

**Pros:**
- Full control over every component
- Native ResPlan support (Shapely polygons, NetworkX graphs, no lossy conversion)
- Modern training infrastructure (Lightning logging, checkpointing, multi-GPU, profiling)
- Opportunity to experiment with architectural improvements
- Clean, well-typed, well-tested codebase the team fully owns

**Cons:**
- Significantly more upfront work
- Risk of bugs when reimplementing diffusion math (beta schedules, q_sample, p_sample)
- Harder to validate against known HouseDiffusion results
- The team is still learning the fundamentals — building everything from scratch is riskier

**Verdict:** The ideal end state, but too much scope for the current timeline. Marked as a
**stretch goal** if time allows after Plan C is working.

---

## Plan C — Hybrid (Chosen)

**Keep the proven model architecture and diffusion math. Rewrite the data pipeline and
training loop.**

The key insight: **the Transformer model doesn't care about data format.** It takes
tensors of specific shapes and produces tensors of specific shapes. The diffusion math
is standard DDPM — also format-agnostic. What changes is how we *produce* those tensors
from raw data, and how we *orchestrate* training.

### What we keep (port from HouseDiffusion)

| Component | Source file | Lines | What it does |
|-----------|------------|-------|--------------|
| TransformerModel | `transformer.py` | 275 | Encoder-only Transformer with triple attention |
| GaussianDiffusion | `gaussian_diffusion.py` | 979 | Forward/reverse diffusion process |
| SpacedDiffusion | `respace.py` | 128 | Accelerated inference with fewer timesteps |
| nn utilities | `nn.py` | 172 | timestep_embedding, mean_flat, update_ema |

These are ported into `src/floorplan_diffusion/models/` with **no architectural changes**.
The model produces identical outputs given identical inputs. Code polish (type hints,
docstrings) is deferred to v2 — the v1 ports are direct working copies.

### What we rewrite

| Component | Why rewrite | New location |
|-----------|-------------|--------------|
| Dataset class | Read ResPlan pickle directly, extract corners from Shapely polygons, build masks from NetworkX graph | `src/floorplan_diffusion/data/dataset.py` |
| Training loop | Replace MPI-based custom loop with PyTorch Lightning | `src/floorplan_diffusion/training/` |
| Config management | Replace argparse with YAML configs | `configs/` |
| Training script | Lightning CLI entry point | `scripts/train.py` |
| Sampling script | Clean inference + visualization | `scripts/sample.py` |

### Why this works

The conversion bottleneck in Plan A is going from rich polygons → bounding boxes → pixel
masks → contour corners. This is a lossy pipeline that RPLAN needs because its source data
is pixel-based. But ResPlan already has vector polygons — we can extract corner coordinates
directly:

```
RPLAN (lossy):     bbox → pixel mask → OpenCV contours → 4 rectangular corners
ResPlan (direct):  polygon.exterior.coords → N actual corner points
```

The rest of the pipeline (one-hot encoding, attention masks, padding to 100 points) is
identical. The Transformer receives the same tensor shapes either way.

### Why not Plan B

Plan C gets us training **weeks sooner** than Plan B while still using ResPlan natively.
The diffusion math in HouseDiffusion is well-tested over thousands of training runs — reimplementing
it from scratch risks subtle bugs (wrong coefficient, off-by-one in timestep indexing) that
produce silent quality degradation rather than obvious errors.

If we later want to experiment with different architectures (different attention patterns,
different diffusion schedules, DiT-style approaches), we can refactor incrementally from
Plan C toward Plan B.

---

## Execution Plan for Plan C

The plan is split into **v1** (core pipeline — train, sample, evaluate) and **v2**
(refinements). v1 skips plans with >100 total vertices instead of simplifying polygons,
keeping ~87.8% of the dataset. This is an acceptable tradeoff to reduce complexity and
get a working pipeline faster.

### v1 — Core Pipeline

#### Step 1: ResPlan Dataset Class

**File:** `src/floorplan_diffusion/data/dataset.py`

A PyTorch `Dataset` that reads `ResPlan.pkl` and outputs tensors matching HouseDiffusion's
format. This is the critical bridge between ResPlan data and the existing model.

**Input:** `plan` dict from the pickle file.

**Output per sample** (same shapes as `RPlanhgDataset.__getitem__()`):
- `x`: `[2, 100]` — normalized corner coordinates
- `cond` dict containing:
  - `room_types`: `[100, 25]` — one-hot room type
  - `corner_indices`: `[100, 32]` — one-hot corner index within room
  - `room_indices`: `[100, 32]` — one-hot room index
  - `src_key_padding_mask`: `[100]` — 1 for padding, 0 for real
  - `connections`: `[100, 2]` — (current_index, next_index) for polygon traversal
  - `door_mask`: `[100, 100]` — 1 blocks attention between non-adjacent rooms
  - `self_mask`: `[100, 100]` — 1 blocks attention between different rooms
  - `gen_mask`: `[100, 100]` — 1 blocks attention to/from padding tokens

**Implementation steps:**
1. Load pickle, call `normalize_keys()` on each plan
2. For each plan, extract room polygons via `get_geometries()` for all 6 room types
3. Extract corner coordinates from `polygon.exterior.coords[:-1]`
4. Filter out plans where total vertices > 100 (simplification deferred to v2)
5. Normalize coordinates to `[-1, 1]` using `plan["inner"].bounds`
6. Build the 94-channel per-point encoding (2 coords + 25 room type + 32 corner idx + 32 room idx + 1 padding + 2 connections)
7. Build attention masks from `plan["graph"]` edges
8. Zero-pad to 100 points
9. Cache processed tensors as `.npz` for fast subsequent loading

**Depends on:** Nothing (can start immediately).

#### Step 2: Port Transformer Architecture

**File:** `src/floorplan_diffusion/models/transformer.py`

Copy `external/house_diffusion/house_diffusion/transformer.py` (275 lines) and `nn.py`
(172 lines) into our package. Fix import paths so internal references resolve. Keep
architecture **identical** — same layer count, same hidden dim, same attention pattern.
Code polish (type hints, docstrings, parameterizing constants) is deferred to v2.

**Depends on:** Nothing (can start in parallel with Step 1).

#### Step 3: Port Diffusion Math

**File:** `src/floorplan_diffusion/models/diffusion.py`

Port the full `gaussian_diffusion.py` (979 lines) as-is:
- `GaussianDiffusion` class with all schedule precomputation
- `q_sample()` — forward noising
- `p_mean_variance()` — reverse step mean/variance
- `p_sample()` / `p_sample_loop()` — full reverse sampling
- `training_losses()` — loss computation
- Beta schedule functions (`get_named_beta_schedule`, `betas_for_alpha_bar`)

Also port `SpacedDiffusion` from `respace.py` and `UniformSampler` from `resample.py`.
Skip `LossSecondMomentResampler` (depends on MPI `all_gather`).

**Depends on:** Nothing (can start in parallel with Steps 1–2).

#### Step 4: PyTorch Lightning Training Module

**File:** `src/floorplan_diffusion/training/lightning_module.py`

A `LightningModule` that wraps the Transformer + GaussianDiffusion:
- `training_step()`: sample timestep via `UniformSampler`, call `training_losses()`,
  return loss
- `validation_step()`: same as training_step on val split, log val loss
- `configure_optimizers()`: AdamW with step-decay LR scheduling (10× decay every 100k
  steps, matching the original HouseDiffusion schedule)
- EMA weight maintenance (via callback or manual update in `on_train_batch_end`)
- Logging: loss, learning rate, EMA decay

**File:** `src/floorplan_diffusion/training/data_module.py`

A `LightningDataModule` wrapping the ResPlan dataset with a 90/10 train/val split.

**Depends on:** Steps 1, 2, 3 (needs dataset, model, and diffusion).

#### Step 5: Training Script

**File:** `scripts/train.py`

CLI entry point using PyTorch Lightning `Trainer`:
- Load YAML config from `configs/`
- Instantiate data module, model, diffusion, Lightning module
- Configure callbacks: `ModelCheckpoint`, EMA, `EarlyStopping` (monitor val loss)
- TensorBoard or CSV logger
- `trainer.fit()`

**Depends on:** Step 4.

#### Step 6: Sampling Script

**File:** `scripts/sample.py`

Inference script:
- Load trained checkpoint (EMA weights)
- Create eval dataset (conditions from val set, zero-initialized corners)
- Run `p_sample_loop()` to generate floorplans
- Render output as PNG using matplotlib
- Compute graph accuracy metric (requires adding `graph` tensor back to dataset output)

**Depends on:** Steps 2, 3, 5 (needs model, diffusion, and a trained checkpoint).

#### Step 7: Validation

Verify the v1 pipeline produces correct results:
- [ ] Tensor shapes from ResPlan dataset match HouseDiffusion's shapes exactly
- [ ] Ported Transformer produces identical output given identical input tensors
- [ ] Ported diffusion produces identical noise schedules and loss values
- [ ] Training loss decreases over time (model is learning)
- [ ] Generated floorplans are visually reasonable after sufficient training
- [ ] Graph accuracy metric confirms generated rooms match conditioning graph

**Depends on:** All previous steps.

---

### v2 — Refinements

After v1 is validated, these improvements can be tackled independently (unless noted):

1. **Polygon simplification** — Apply `polygon.simplify(tolerance)` to recover the ~12.2%
   of plans filtered out in v1. Requires tolerance tuning and visual validation that
   simplified polygons still represent the original room shapes faithfully.
2. **WandB logging** — Replace TensorBoard with WandB for richer experiment tracking,
   hyperparameter sweeps, and team-shared dashboards.
3. **FID metric** — Compute Fréchet Inception Distance between generated and real
   floorplans. Requires generating a large sample set and building a reference distribution.
   *Depends on v1 Step 6.*
4. **Config flexibility** — Config validation, config inheritance for experiments, CLI
   overrides for hyperparameters.
5. **Code polish** — Type hints and docstrings on ported model code. Parameterize remaining
   hardcoded constants (e.g., `num_layers = 4` in Transformer).

---

### Dependency Graph

```
v1:
  Step 1 (Dataset+Cache) ──────────┐
                                    │
  Step 2 (Transformer) ────────────┤
                                    ├─→ Step 4 (Lightning+EMA+Val+LR) ─→ Step 5 (Train) ─→ Step 6 (Sample) ─→ Step 7 (Validate)
  Step 3 (Diffusion) ──────────────┘

v2 (after v1 validated):
  v2.1 Simplification ─── independent
  v2.2 WandB ──────────── independent
  v2.3 FID metric ─────── depends on v1 Step 6
  v2.4 Config flex ────── independent
  v2.5 Code polish ────── independent
```

Steps 1, 2, and 3 can be developed **in parallel** by different team members.
Step 4 integrates them. Steps 5–7 are sequential. All v2 items are independent of
each other (except FID which builds on the sampling script).

---

## Future: Plan B (Stretch Goal)

If Plan C is working and time allows, these are potential Plan B improvements:

- **Variable-length sequences**: remove the 100-point cap, use dynamic padding per batch
- **Richer conditioning**: leverage ResPlan's typed edges (direct/adjacency/via_door/via_window)
  in the attention masks instead of binary connected/not-connected
- **Alternative architectures**: DiT (Diffusion Transformer), flow matching, or consistency
  models for faster inference
- **Direct polygon generation**: predict polygon vertex sequences instead of fixed-size
  point clouds — would preserve room shape structure better
- **Multi-resolution**: generate coarse layout first, then refine room boundaries
