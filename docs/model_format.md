# ResPlan tensor format & model dimensions

This documents the **as-built** per-point tensor format used by our ResPlan port of
HouseDiffusion, and the 2026-06 capacity change that widened it. For the *original*
HouseDiffusion/RPLAN numbers (100 points / 89 cond / 94 columns) see
[`docs/team/house_diffusion_study_guide.md`](team/house_diffusion_study_guide.md) — they differ.

## Per-point feature vector (`NUM_COLUMNS = 158`)

Each floorplan is encoded as a `[MAX_NUM_POINTS, NUM_COLUMNS]` array (zero-padded), where each
row is one polygon corner. Column layout:

| Columns | Width | Field | Notes |
|--------:|------:|-------|-------|
| 0–1 | 2 | coordinates | (x, y) normalized to [-1, 1] |
| 2–26 | 25 | room_type one-hot | living/bedroom/kitchen/window/bathroom/balcony/door/front_door (+ headroom) |
| 27–90 | 64 | corner_idx one-hot | position of this vertex within its own room (`CORNER_IDX_DIMS`) |
| 91–154 | 64 | room_idx one-hot | which polygon part (room/door/window) this vertex belongs to (`ROOM_IDX_DIMS`) |
| 155 | 1 | padding flag | 1 = real point, 0 = pad |
| 156–157 | 2 | connections | (current, next) global vertex indices, wrapping within the part |

Derived: `NUM_COLUMNS = 2 + 25 + CORNER_IDX_DIMS + ROOM_IDX_DIMS + 1 + 2 = 158`.

## Key constants (single source of truth)

All defined in [`src/floorplan_diffusion/data/dataset.py`](../src/floorplan_diffusion/data/dataset.py);
everything else derives from them symbolically (slices, masks, padding, cache tag, model layer sizes).

| Constant | Value | Meaning |
|----------|------:|---------|
| `MAX_NUM_POINTS` | **192** | sequence length (max total vertices per plan; padding target) |
| `MAX_CORNERS_PER_ROOM` | 64 | reject a plan if any room exceeds this after simplification |
| `CORNER_IDX_DIMS` | 64 | corner-index one-hot width |
| `ROOM_IDX_DIMS` | **64** | room-index one-hot width → at most `ROOM_IDX_DIMS-1 = 63` parts/plan |
| `NUM_COLUMNS` | **158** | per-point feature width (computed) |
| `condition_channels` | **153** | `25 + CORNER_IDX_DIMS + ROOM_IDX_DIMS`; sizes `condition_emb` in the model |

`condition_channels` is computed (not hardcoded) in
[`scripts/train.py`](../scripts/train.py) and
[`src/floorplan_diffusion/models/sampling.py`](../src/floorplan_diffusion/models/sampling.py).

## Cache

Processed tensors are cached as `.npz` named
`resplan_{set}_{md5(pickle_path)}_c{NUM_COLUMNS}p{MAX_NUM_POINTS}.npz` → currently `…c158p192.npz`.
The schema tag means changing the format never reuses a stale cache; `_load_cache` also validates
the array shape `(192, 158)` and reprocesses on mismatch.

## 2026-06 change: 128/32 → 192/64 (why)

The previous format (`MAX_NUM_POINTS=128`, `ROOM_IDX_DIMS=32`) **silently dropped 42.6% of
ResPlan** — 7,285 of 17,107 plans — entirely via the total-vertex cap. Windows (encoded as ~7
small `MultiPolygon` boxes / ~29 vertices per plan) were the dominant driver, with the 31-part
ceiling a secondary one. Raising to 192/64:

| MAX_NUM_POINTS / parts-limit | Plans accepted |
|---|---:|
| 128 / 31 (old) | 57.4% |
| 160 / 63 | 82.2% |
| **192 / 63 (current)** | **92.8%** |
| 224 / 63 | 96.8% |

At 192/64 the parts ceiling no longer binds (0 plans rejected on parts); the only remaining skips
are the 7.2% with genuinely >192 vertices. Note: `_simplify_polygon` contributes ~nothing to
retention (only 0.2% of plans have a >64-corner room) — the gains come from the vertex cap and the
corner-rejection threshold, not simplification.

## Consequences

- **Checkpoints are not transferable.** `condition_emb` changes from `Linear(121, …)` to
  `Linear(153, …)`; any model trained on the old format must be **retrained from scratch** (load
  fails loudly with a size mismatch — no silent mis-load). Sequence length 128→192 is handled
  dynamically by the Transformer (attention is O(N²); `PositionalEncoding` is defined but unused).
- **One-time cache rebuild** on first run (new schema tag).
- **~2.25× attention cost** vs 128 (1.5× sequence, squared). On the L40s (48 GB) `batch_size=128`
  may need lowering if it OOMs — see `configs/resplan_housediff_stable_fp32.yaml`.
