# ResPlan Dataset Test Logic

This document describes **what** is being tested and **why**, without reference to code.

---

## Test Setup

All tests use synthetic floor plans built from simple rectangular or L-shaped rooms.
No real dataset file is needed. Three canonical rooms are used throughout:
- **Room A** (living): 10×10 square at the origin
- **Room B** (bedroom): 10×10 square touching Room A along one edge
- **Room C** (bathroom): 10×10 square separated from B by a gap

---

## Milestone 1 — Key Normalization

**Goal:** Ensure that known typos in room-type dictionary keys are silently corrected before any processing begins.

| What is checked | Why it matters |
|---|---|
| The misspelling `balacony` is renamed to `balcony` | Raw ResPlan data contains this typo; downstream lookups would silently miss the room |
| Correctly-spelled keys are left unchanged | Normalization must be idempotent |

---

## Milestone 2 — Vertex Extraction from Polygons

**Goal:** Confirm that a Shapely polygon is correctly converted to a plain array of (x, y) coordinates.

| What is checked | Why it matters |
|---|---|
| A rectangle yields exactly 4 vertices | Shapely stores a closing duplicate vertex; we must strip it |
| The closing duplicate is removed | Including it would produce a false extra corner in the tensor |
| An L-shaped polygon yields exactly 6 vertices | Non-rectangular rooms must be handled without truncation |
| An empty polygon yields a 0-row array | Degenerate input must not crash downstream steps |
| Output dtype is float32 | The diffusion model expects float32 tensors throughout |

---

## Milestone 3 — House Tensor Construction

**Goal:** Verify that all rooms are packed into a single fixed-size matrix where every row encodes one corner point.

### Shape and layout

The tensor has a fixed number of rows (one per possible corner slot) and exactly **94 columns**, laid out as:

| Columns | Content |
|---|---|
| 0–1 | (x, y) coordinate, normalized to [−1, 1] |
| 2–26 | Room type as a one-hot vector (25 classes) |
| 27–58 | Corner index within its room as a one-hot vector (32 slots) |
| 59–90 | Room index across the floor plan as a one-hot vector (32 slots) |
| 91 | Real-corner flag: 1 for actual data, 0 for padding |
| 92–93 | Connectivity: self index and next-corner index (cyclical) |

### What is checked

| What is checked | Why it matters |
|---|---|
| Output shape is (MAX_POINTS, 94) | Fixed-size input is required by the transformer |
| Column widths sum to 94 | Guards against silent off-by-one errors in slicing |
| All coordinates are in [−1, 1] | The diffusion model assumes normalized coordinates |
| Exactly N real corners are flagged | Ensures no corners are dropped or duplicated |
| Padding rows are all zeros | Padding must be inert; non-zero values would corrupt attention |
| Room-type column is a valid one-hot | Each corner must belong to exactly one room type |
| Corner indices are sequential within a room | The model uses these to order corners; gaps would break connectivity |
| Room indices are 1-indexed (first room = index 1) | Matches the RPLAN convention used by HouseDiffusion |
| Connectivity wraps around (last → first corner) | The polygon is closed; diffusion model relies on cyclic structure |
| `corner_bounds` lists correct [start, end) for each room | Used by attention masks to identify which rows belong to each room |
| Empty room list produces an all-zero tensor | Edge case must not crash |
| An L-shaped room produces 6 flagged corners | Non-rectangular rooms are fully preserved |

---

## Milestone 4 — Graph Triples and Attention Masks

### Graph Triples

**Goal:** Represent pairwise spatial relationships between rooms as a list of (room_i, relation, room_j) triples.

| What is checked | Why it matters |
|---|---|
| Three rooms produce C(3,2) = 3 triples | Every pair appears exactly once |
| Adjacent rooms get relation = +1 | The model must know which rooms share a wall |
| Separated rooms get relation = −1 | Non-adjacency is an explicit signal, not absence of data |
| A single room produces an empty graph | No pairs exist; must not crash |
| Zero rooms produces an empty graph | Edge case |

### Attention Masks

**Goal:** Produce three boolean matrices that control which corners can attend to which during the transformer forward pass.

Three masks are produced, each of shape (MAX_POINTS, MAX_POINTS), where **0 = attend** and **1 = block**:

| Mask | Meaning | What is checked |
|---|---|---|
| **self_mask** | Corners of the same room attend to each other freely | Diagonal blocks (same-room) are 0; cross-room blocks of non-adjacent rooms are 1 |
| **door_mask** | Adjacent rooms can attend across their shared boundary | Corners of adjacent rooms have 0; corners of non-adjacent rooms have 1 |
| **gen_mask** | Real corners attend to real corners; padding is invisible | Real-corner area is 0; any row or column involving a padding slot is 1 |

---

## Milestone 5 — Dataset Item Interface

**Goal:** Confirm that slicing the house tensor into the named fields expected by the model produces correct shapes.

This milestone simulates what `__getitem__` does, without loading a real dataset file.

| Field | Expected shape | What is checked |
|---|---|---|
| coordinates array (transposed) | (2, MAX_POINTS) | Correct transposition for model input |
| `door_mask` | (MAX_POINTS, MAX_POINTS) | — |
| `self_mask` | (MAX_POINTS, MAX_POINTS) | — |
| `gen_mask` | (MAX_POINTS, MAX_POINTS) | — |
| `room_types` | (MAX_POINTS, 25) | Correct column slice |
| `corner_indices` | (MAX_POINTS, 32) | Correct column slice |
| `room_indices` | (MAX_POINTS, 32) | Correct column slice |
| `src_key_padding_mask` | (MAX_POINTS,) | Inverted real-corner flag (1 = ignore) |
| `connections` | (MAX_POINTS, 2) | Self and next-corner indices |
| `graph` | (200, 3) | Padded to fixed length for batching |

Additionally: the `src_key_padding_mask` inversion is verified directly — real corners must have mask value 0 (attend) and padding slots must have mask value 1 (ignore), matching PyTorch's `key_padding_mask` convention.
