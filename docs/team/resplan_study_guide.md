# ResPlan Dataset Study Guide

A progressive, self-guided walkthrough of `external/ResPlan/` for the team.
Complete the stages in order — each one builds on the previous.

**Prerequisite:** Complete the [HouseDiffusion Study Guide](house_diffusion_study_guide.md) first.
You need to understand the Transformer architecture and tensor shapes that our data pipeline
must produce.

All file paths are relative to `external/ResPlan/` unless stated otherwise.

---

## Dataset Overview

ResPlan is a **large-scale vector-graph dataset of 17,107 residential floor plans** designed for
spatial AI research. Unlike RPLAN (pixel masks + bounding boxes), ResPlan stores geometry as
**Shapely polygons** and connectivity as **NetworkX graphs** — both already in vector form.

```
                     RESPLAN DATA MODEL
  ┌────────────────────────────────────────────────────────────┐
  │  ResPlan.pkl  (list of 17,107 plan dicts)                  │
  │                                                            │
  │  plan = {                                                  │
  │    # Room geometries (Shapely MultiPolygon)                │
  │    "living":     MultiPolygon(...),                        │
  │    "bedroom":    MultiPolygon(...),                        │
  │    "bathroom":   MultiPolygon(...),                        │
  │    "kitchen":    MultiPolygon(...),                        │
  │    "balcony":    MultiPolygon(...),                        │
  │    "front_door": Polygon(...),                             │
  │                                                            │
  │    # Architectural elements (Shapely MultiPolygon)         │
  │    "wall":       MultiPolygon(...),                        │
  │    "door":       MultiPolygon(...),                        │
  │    "window":     MultiPolygon(...),                        │
  │                                                            │
  │    # Pre-built room graph (NetworkX)                       │
  │    "graph":      nx.Graph(nodes=rooms, edges=connections), │
  │                                                            │
  │    # Spatial boundaries                                    │
  │    "inner":      MultiPolygon(...),  # plan outline        │
  │    "land":       MultiPolygon(...),  # land parcel         │
  │                                                            │
  │    # Metadata                                              │
  │    "id": int, "area": float, "net_area": float,           │
  │    "wall_depth": float, "unitType": str, ...               │
  │  }                                                         │
  └────────────────────────────────────────────────────────────┘
```

```
                  HOW RESPLAN FEEDS THE MODEL (Plan C — Hybrid)
  ┌──────────────────────────────────────────────────────────────┐
  │  ResPlan.pkl                                                 │
  │    └─ ResplanDataset.__getitem__()                           │
  │         │  polygon.exterior.coords → corner points [N, 2]   │
  │         │  graph nodes → room type one-hot [N, 25]           │
  │         │  graph edges → door_mask, self_mask [100, 100]     │
  │         │  zero-pad to 100 points                            │
  │         ▼                                                    │
  │    Same tensor shapes as HouseDiffusion:                     │
  │      x:    [2, 100]   cond: [100, 89]   masks: [100, 100]   │
  │         ▼                                                    │
  │    TransformerModel (ported, same architecture)               │
  │         ▼                                                    │
  │    GaussianDiffusion (ported, same math)                     │
  │         ▼                                                    │
  │    PyTorch Lightning Trainer                                 │
  └──────────────────────────────────────────────────────────────┘
```

**Key numbers to keep in mind:**
- Total plans: **17,107**
- Room types: **6** (living, bedroom, bathroom, kitchen, balcony, front_door)
- Geometry format: Shapely `MultiPolygon` (rooms), `MultiPolygon` (doors/windows/walls)
- Coordinate space: roughly **0–256** on both axes (not normalized)
- Rooms per plan: **5–42** (mean ~10.3 graph nodes)
- Median vertices per room polygon: **4** (max 241)
- Plans with <=100 total vertices: **~87.8%**
- Unit types: **92.7%** Apartment, 4.8% BuilderFloor, 2.2% Villa

---

## Stage 0 — Orientation (≈20 min)

**Goal:** Get the lay of the land before reading any code deeply.

**Read:**
- [`README.md`](../../external/ResPlan/README.md) — paper reference (arXiv:2508.14006), contents, dependencies
- [`ResPlan_demo.ipynb`](../../external/ResPlan/ResPlan_demo.ipynb) — skim all sections (loading, plotting, masks, augmentations, graph)
- [`resplan_utils.py:1-50`](../../external/ResPlan/resplan_utils.py) — imports, `CATEGORY_COLORS` dict, `DEFAULT_CANVAS_SIZE`

**Key facts to absorb:**
- The dataset ships as a single `ResPlan.pkl` inside `ResPlan.zip` (a pickled list of dicts)
- Each plan is a `Dict[str, Any]` — not a class, just a plain dictionary
- The utils file is **384 lines** of pure helper functions — no training code, no model code
- Dependencies: shapely, geopandas, matplotlib, networkx, numpy, opencv-python
- `CATEGORY_COLORS` (line 37) lists 9 geometry categories: living, bedroom, bathroom, kitchen,
  door, window, wall, front_door, balcony
- The `"balacony"` typo exists in some plans — `normalize_keys()` (line 55) fixes it
- The exploration notebook at [`notebooks/01_resplan_exploration.ipynb`](../../notebooks/01_resplan_exploration.ipynb)
  contains dataset-wide statistics referenced throughout this guide

**Checkpoint:** Can you list the directory contents of `external/ResPlan/` and say what each
file does in one line? How does this compare to the HouseDiffusion codebase in terms of
complexity and scope?

---

## Stage 1 — Plan Dictionary Anatomy (≈45 min)

**Goal:** Know every key in a plan dict, its Python type, its Shapely geometry type, and its
semantic meaning.

**Read:**
- [`resplan_utils.py:55-68`](../../external/ResPlan/resplan_utils.py) — `normalize_keys()` and `get_plan_width()`
- Notebook sections 2–3 in [`01_resplan_exploration.ipynb`](../../notebooks/01_resplan_exploration.ipynb) (key inspection, presence stats)

### 1a. The complete plan dictionary

Each plan has up to **24 keys**. They fall into four groups:

**Room geometries** (the core data — these become model input):

| Key | Type | Present | Description |
|-----|------|---------|-------------|
| `living` | MultiPolygon | 100% | Living room(s) — each polygon part = one room |
| `bedroom` | MultiPolygon | 100% | Bedroom(s) |
| `bathroom` | MultiPolygon | 100% | Bathroom(s) |
| `kitchen` | MultiPolygon | 99.5% | Kitchen(s) |
| `balcony` | MultiPolygon | 73.3% | Balcony/ies |
| `front_door` | Polygon | 100% | Main entrance (single polygon, not Multi) |

**Architectural elements** (used for graph construction and rendering):

| Key | Type | Present | Description |
|-----|------|---------|-------------|
| `wall` | MultiPolygon | 100% | Wall geometry (11 plans have plain Polygon) |
| `door` | MultiPolygon | 100% | Interior doors — each part = one door |
| `window` | MultiPolygon | 99.7% | Windows — each part = one window |

**Spatial / graph data:**

| Key | Type | Present | Description |
|-----|------|---------|-------------|
| `graph` | nx.Graph | 100% | Pre-built room adjacency graph |
| `inner` | MultiPolygon | 100% | Overall floorplan boundary |
| `land` | MultiPolygon | 100% | Land parcel polygon |
| `neighbor` | tuple | 100% | Neighboring unit info |

**Metadata:**

| Key | Type | Present | Description |
|-----|------|---------|-------------|
| `id` | int | 100% | Plan identifier |
| `area` | float | 100% | Total area (coordinate units²) |
| `net_area` | float | 100% | Net usable floor area |
| `wall_depth` | float | 100% | Wall thickness in coordinate units |
| `unitType` | str | 100% | "Apartment" (92.7%), "BuilderFloor", "Villa", etc. |

**Rare spatial keys** (appear in <10% of plans):

| Key | Present | Description |
|-----|---------|-------------|
| `storage` | 10.0% | Storage room(s) |
| `veranda` | 4.0% | Veranda |
| `stair` | 4.0% | Staircase |
| `garden` | 3.1% | Garden |
| `parking` | 1.9% | Parking |
| `pool` | <0.1% | Pool (1 plan) |

### 1b. MultiPolygon means multiple rooms

A critical detail: `plan["bedroom"]` being a `MultiPolygon` with 3 parts means the plan has
**3 separate bedrooms**. Use `get_geometries()` (line 73) to split:

```python
from resplan_utils import get_geometries
bedrooms = get_geometries(plan["bedroom"])  # → list of 3 Polygon objects
```

This is how you count rooms — not by counting keys, but by counting polygon parts within
each key.

### 1c. `normalize_keys()` and `get_plan_width()`

- `normalize_keys(plan)` (line 55): fixes `"balacony"` → `"balcony"`. **Call this first** on
  every plan before processing.
- `get_plan_width(plan)` (line 61): returns `max(width, height)` of the `inner` polygon's
  bounding box. Useful for normalization and filtering.

**Checkpoint:** For `plans[0]`, group all 24 keys into the four categories above. Then count
how many individual room polygons exist by calling `get_geometries()` on each room key. What
is the total room count?

---

## Stage 2 — Geometry Deep Dive (≈1 h)

**Goal:** Understand Shapely geometry types, coordinate space, polygon complexity, and how
room corners will be extracted for the model.

**Read:**
- [`resplan_utils.py:72-106`](../../external/ResPlan/resplan_utils.py) — `get_geometries()`, `centroid()`, `perturb_polygon()`, `noise()`
- [`resplan_utils.py:140-190`](../../external/ResPlan/resplan_utils.py) — `geometry_to_mask()`
- Notebook sections 5, 7, 8 in [`01_resplan_exploration.ipynb`](../../notebooks/01_resplan_exploration.ipynb) (coordinate space, geometry layers, vertex counts)

### 2a. Coordinate space

Plans live in a coordinate space of roughly **0–256 on both axes**:
- X range across dataset: `[0, 943]`
- Y range across dataset: `[0, 648]`
- Typical plan extent: **~278 wide × ~208 tall** (varies per plan)

The `inner` polygon defines the overall plan boundary. Its `.bounds` gives `(minx, miny, maxx, maxy)`.

Some geometries extend slightly beyond `inner` (e.g., `front_door` at x=-3.0). These are
real features that protrude from the main boundary.

> **For the model:** coordinates must be normalized to `[-1, 1]`. Use each plan's `inner.bounds`
> to compute the normalization transform.

### 2b. Polygon structure

Each room is a Shapely `Polygon` with:
- An **exterior ring**: ordered list of `(x, y)` vertex coordinates
- Optional **interior rings** (holes) — rare in room polygons
- The closing vertex duplicates the first: `len(polygon.exterior.coords)` = vertices + 1

```python
poly = get_geometries(plan["living"])[0]
coords = np.array(poly.exterior.coords)[:-1]  # drop closing vertex
# coords.shape → (N, 2) where N is the vertex count
```

### 2c. Vertex counts — the 100-point question

This is critical because HouseDiffusion caps total corner points at **100 per floorplan**.

| Metric | Value |
|--------|-------|
| Median vertices per room polygon | **4** |
| Max vertices per room polygon | **241** |
| Median total vertices per plan | **~36** |
| Plans with ≤100 total vertices | **87.8%** |
| Plans with >100 total vertices | **12.2%** |

Most rooms are roughly rectangular (4 vertices). But some rooms have complex shapes
(L-shapes, curves, irregular boundaries) with many vertices.

> **For the model:** Plans exceeding 100 total vertices need either: (a) polygon
> simplification via `polygon.simplify(tolerance)`, or (b) filtering out during
> dataset construction.

### 2d. `get_geometries()` — the safe extraction function (line 73)

```python
def get_geometries(geom_data):
    if geom_data is None: return []
    if isinstance(geom_data, (Polygon, LineString, Point)):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []
```

Handles `None`, single geometries, multi-geometries, and empty geometries. Always use this
instead of accessing `.geoms` directly — it's null-safe and filters empties.

### 2e. `geometry_to_mask()` — rasterization (line 157)

Converts any Shapely geometry to a **256×256 binary mask** using OpenCV:
- `Polygon` → `cv2.fillPoly` (filled region), handles interior holes
- `MultiPolygon` → union of filled regions
- `LineString` → `cv2.polylines` (stroked line)
- `Point` → `cv2.circle`

This demonstrates the pixel-mask approach that RPLAN uses. Our hybrid approach (Plan C)
skips rasterization entirely — we extract vertex coordinates directly from the Shapely
polygons.

### 2f. Augmentation and buffer helpers

- `augment_geom(geom, degree, flip_vertical, scale, size)` (line 111): rotate around
  canvas center, optional vertical flip, uniform scale. Useful for data augmentation
  in the training pipeline.
- `buffer_shrink_expand(geom, w)` (line 127): shrink then expand by `w` — cleans up
  jagged edges and tiny artifacts.
- `buffer_expand_shrink(geom, w)` (line 132): expand then shrink — fills small gaps
  between adjacent polygons.

These can be applied before corner extraction to improve polygon quality.

**Checkpoint:** Pick a plan with >100 total vertices. For each room polygon, print the vertex
count and bounding box. Then apply `polygon.simplify(tolerance=2.0)` to the worst offender
and check how many vertices remain. Does the simplified polygon still look reasonable?

---

## Stage 3 — Graph Structure (≈45 min)

**Goal:** Understand how rooms are connected — both via the pre-built `plan["graph"]` and
via `plan_to_graph()` — and how this maps to the attention masks the model needs.

**Read:**
- [`resplan_utils.py:247-307`](../../external/ResPlan/resplan_utils.py) — `plan_to_graph()` full function
- [`resplan_utils.py:313-383`](../../external/ResPlan/resplan_utils.py) — `plot_plan_and_graph()`
- Notebook section 9 in [`01_resplan_exploration.ipynb`](../../notebooks/01_resplan_exploration.ipynb) (graph stats and overlay visualization)

### 3a. The pre-built graph

Every plan already contains a NetworkX `Graph` at `plan["graph"]`. This was built by logic
similar to `plan_to_graph()`.

**Nodes** are named `"{room_type}_{index}"` (e.g., `"living_0"`, `"bedroom_1"`):
- Attribute `geometry`: the Shapely Polygon for that room
- Attribute `type`: room type string (`"living"`, `"bedroom"`, etc.)
- Attribute `area`: polygon area in coordinate units²

**Edges** have a single attribute `type` — one of four values:

| Edge type | Meaning | How detected |
|-----------|---------|--------------|
| `direct` | front_door physically touches living room | Spatial intersection with buffer |
| `adjacency` | Rooms share a wall (kitchen/bedroom ↔ living) | Buffered polygon intersection |
| `via_door` | Rooms connected through a door geometry | Door intersects both room buffers |
| `via_window` | Rooms connected through a window | Window intersects both room buffers |

**Typical graph stats** (from 2000-plan sample):
- Nodes: mean=10.3, median=9, range=[5, 31]
- Edges: mean=6.6, median=6, range=[2, 22]
- Edge type breakdown: direct=1999, adjacency=2989, via_door=7951, via_window=348

### 3b. How `plan_to_graph()` builds it (line 247)

Step-by-step:

1. **Extract wall_width** for buffer calculations (line 252):
   `buf = max(wall_width * buffer_factor, 0.01)`

2. **Create room nodes** (lines 258–265): for each of the 5 room types (living, kitchen,
   bedroom, bathroom, balcony), split the MultiPolygon into parts and add each as a node.

3. **Create front_door nodes** (lines 268–271): may be Polygon or LineString.

4. **Collect connection geometries** (lines 273–275): all doors and windows, tagged with
   their connection type (`"via_door"` or `"via_window"`).

5. **Connect front_door → living** (lines 278–283): `"direct"` edge if geometries intersect
   with buffer.

6. **Connect kitchen/bedroom ↔ living** (lines 286–292): `"adjacency"` edge if buffered
   polygons intersect.

7. **Connect bathroom/balcony → living/bedroom** (lines 295–306): via door or window
   geometry intersection.

### 3c. From graph to attention masks

The HouseDiffusion Transformer needs three `[100, 100]` attention masks. Here is how the
ResPlan graph provides the information for each:

| Mask | What it encodes | Source from ResPlan graph |
|------|-----------------|--------------------------|
| `self_mask` | Corners in the **same room** can attend | Corners sharing a node ID |
| `door_mask` | Corners in **connected rooms** can attend | Nodes sharing an edge (any type) |
| `gen_mask` | **Padding tokens** are blocked | Points beyond the real corner count |

> **Key insight:** HouseDiffusion's `build_graph()` returns a triples array
> `[room_i, 1_or_-1, room_j]` to encode connections. ResPlan's graph is richer
> (typed edges, geometry attributes), but for mask construction we only need the
> binary question: "are these two rooms connected?" — which is `G.has_edge(node_i, node_j)`.

**Checkpoint:** Load `plans[0]["graph"]`. Print all nodes with their type and area. Print all
edges with their type. Then for one `"via_door"` edge, verify the connection by checking that
a door geometry from `plans[0]["door"]` actually intersects both rooms using Shapely's
`.intersects()`.

---

## Stage 4 — Visualization & Masks (≈30 min)

**Goal:** Understand the plotting and rasterization functions — useful for debugging the
data pipeline and validating converted outputs.

**Read:**
- [`resplan_utils.py:196-241`](../../external/ResPlan/resplan_utils.py) — `plot_plan()`
- [`resplan_utils.py:313-383`](../../external/ResPlan/resplan_utils.py) — `plot_plan_and_graph()`
- Notebook sections 6, 9, 10 in [`01_resplan_exploration.ipynb`](../../notebooks/01_resplan_exploration.ipynb)

### 4a. `plot_plan()` (line 196)

Renders a colored floorplan using GeoPandas:
- Default categories: living, bedroom, bathroom, kitchen, door, window, wall, front_door, balcony
- Each geometry layer gets its color from `CATEGORY_COLORS`
- Uses `gpd.GeoSeries.plot()` for rendering
- Optional legend, title, custom axes

### 4b. `plot_plan_and_graph()` (line 313)

Overlays the NetworkX graph on top of the plan visualization:
- Node positions = room polygon centroids
- Node shapes encode room type (circle=living, square=bedroom, diamond=bathroom, etc.)
- Node size scales with room area
- Edge styles encode connection type (solid=direct, dashed=adjacency, etc.)

### 4c. `geometry_to_mask()` for debugging

When debugging the data pipeline, you can rasterize any geometry to a 256×256 mask
and visually inspect it:

```python
mask = geometry_to_mask(plan["living"], shape=(256, 256))
plt.imshow(mask, cmap="gray")
```

This is especially useful for verifying that polygon simplification hasn't destroyed
room shapes.

**Checkpoint:** Render three plans using `plot_plan_and_graph()`. For each, verify that
the graph edges correspond to rooms that visually share a wall or door.

---

## Stage 5 — ResPlan vs RPLAN: Full Comparison (≈30 min)

**Goal:** Systematically compare the two data formats to understand what the new dataset
class must do differently.

No new files to read — this is a synthesis stage. Reference the
[HouseDiffusion Study Guide](house_diffusion_study_guide.md) as needed.

### 5a. Structural comparison

| Aspect | RPLAN (HouseDiffusion) | ResPlan | Impact on data pipeline |
|--------|------------------------|---------|------------------------|
| **Format** | Per-plan JSON files | Single pickle (list of dicts) | Load once, index by position |
| **Room geometry** | Bounding boxes `[x0,y0,x1,y1]` → pixel masks → OpenCV contours | Shapely Polygons (already vector) | Extract `polygon.exterior.coords` directly |
| **Room types** | 25 integer codes (14 used): 1=living, 2=master_bed, 3=kitchen, 4=bath, 5=dining, 6=child_bed, 7=study, 8=second_bath, 10=balcony, 11–13=doors | 6 string keys: living, bedroom, bathroom, kitchen, balcony, front_door | Map strings → integers (see 5b) |
| **Room instances** | One entry per room in `boxes` array | MultiPolygon parts = separate rooms | `get_geometries()` to split |
| **Doors/windows** | Edge segments `[x0,y0,x1,y1]` in `edges` array | MultiPolygon geometries (each part = one door/window) | Not needed for corner extraction; used only for graph |
| **Graph** | Implicit: `ed_rm` (edge-to-room index) → `build_graph()` | Explicit: pre-built `nx.Graph` with typed edges | Use graph directly for mask construction |
| **Coordinates** | 0–256 pixels → normalized [0,1] → centered → scaled to [-1,1] | ~0–256 (varies per plan, not normalized) | Normalize using `inner.bounds` → scale to [-1,1] |
| **Vertex count** | ~4 per room (from bounding box contours) | 4–241 per room (arbitrary polygons) | Simplify or filter if >100 total |
| **Max points** | 100 (hard cap, skip plans exceeding) | No limit (12.2% exceed 100) | Apply same 100-point cap |
| **Extra spaces** | Types 15 (exterior), 17 (exterior wall) | storage, veranda, stair, garden, parking | Skip rare types or map to nearest |
| **Metadata** | None in JSON | area, net_area, wall_depth, unitType, id | Available for filtering/analysis |

### 5b. Room type mapping

| ResPlan string | RPLAN integer | Notes |
|----------------|---------------|-------|
| `living` | 1 | Direct mapping |
| `bedroom` | 2 | RPLAN distinguishes master (2) vs child (6); map all to 2 initially |
| `kitchen` | 3 | Direct mapping |
| `bathroom` | 4 | RPLAN distinguishes primary (4) vs second (8); map all to 4 initially |
| `balcony` | 10 | Direct mapping |
| `front_door` | 11 | Treat as entrance/corridor type |
| `storage` | 9 | Map to guest room or skip (only 10% of plans) |

The one-hot encoding uses 25 dimensions to match HouseDiffusion's conditioning channels.
Only 7 of the 25 slots will be active with ResPlan data.

### 5c. What changes vs. what stays the same

**Stays the same** (port from HouseDiffusion):
- Transformer architecture (encoder-only, triple attention, 4 layers, 512-dim)
- Diffusion math (GaussianDiffusion, cosine schedule, 1000 steps)
- Tensor shapes: `x=[B,2,100]`, `cond=[B,100,89]`, masks=`[B,100,100]`
- 94-channel per-point encoding (2 coords + 25 room type + 32 corner idx + 32 room idx + padding + connections)
- Point expansion (100 → 900 tokens via midpoint interpolation)
- EMA, training objective (epsilon prediction + MSE)

**Changes** (new implementation needed):
- Dataset class: reads pickle instead of JSON, extracts corners from Shapely polygons
- Graph → mask construction: uses NetworkX edges instead of `ed_rm` array
- Coordinate normalization: per-plan normalization using `inner.bounds`
- Polygon simplification: reduce vertex counts for complex rooms
- Training loop: PyTorch Lightning instead of custom `TrainLoop`

**Checkpoint:** For one ResPlan plan, write Python code that produces a `[100, 94]` tensor
matching the HouseDiffusion per-point encoding format: channels 0–1 as normalized coordinates,
2–26 as room type one-hot, 27–58 as corner index one-hot, 59–90 as room index one-hot,
91 as padding mask, 92–93 as connection indices.

---

## Stage 6 — End-to-End Mental Model (≈20 min)

**Goal:** Synthesize everything into one coherent picture.

No new files to read. Work from memory.

### Exercise A: Full data-flow diagram

Draw (on paper or in a text file) the complete data flow for the hybrid approach:

```
ResPlan.pkl
  → load pickle, normalize_keys()
  → for each plan:
      get_geometries(plan["living"]) → list of room Polygons
      ...repeat for bedroom, bathroom, kitchen, balcony, front_door
      polygon.exterior.coords[:-1] → corner point arrays
      normalize coords using inner.bounds → [-1, 1]
      simplify if total_vertices > 100
      build 94-channel encoding per point (same as HouseDiffusion)
      build masks from plan["graph"] edges
      zero-pad to 100 points
  → x: [2, 100]  cond: [100, 89]  masks: [100, 100]
  → TransformerModel.forward(x_t, t, cond, masks) → ε_pred
  → MSE(ε_pred, ε) → AdamW + EMA
```

### Exercise B: Three plans, three complexities

Pick plans with different room counts (5, 10, 20 rooms). For each, determine:
1. How many polygon parts exist across all room types?
2. What is the total vertex count? Does it exceed 100?
3. How many graph edges exist? What types?
4. What would the `door_mask` look like? (Which room pairs can attend to each other?)

### Exercise C: Risk assessment

Identify the three riskiest parts of building the ResPlan data pipeline and propose a
test for each:

1. **Polygon simplification** — does simplifying complex rooms lose important shape info?
2. **Coordinate normalization** — do all rooms fit within [-1, 1] after normalization?
3. **Graph → mask mapping** — does every room have at least one connection in `door_mask`?

---

## Quick Reference: Plan Dictionary

| Key | Type | Present | For model? |
|-----|------|---------|------------|
| `living` | MultiPolygon | 100% | Yes — room geometry |
| `bedroom` | MultiPolygon | 100% | Yes — room geometry |
| `bathroom` | MultiPolygon | 100% | Yes — room geometry |
| `kitchen` | MultiPolygon | 99.5% | Yes — room geometry |
| `balcony` | MultiPolygon | 73.3% | Yes — room geometry |
| `front_door` | Polygon | 100% | Yes — entrance geometry |
| `wall` | MultiPolygon | 100% | No — rendering only |
| `door` | MultiPolygon | 100% | Indirect — graph edges |
| `window` | MultiPolygon | 99.7% | Indirect — graph edges |
| `graph` | nx.Graph | 100% | Yes — attention masks |
| `inner` | MultiPolygon | 100% | Yes — normalization |
| `wall_depth` | float | 100% | No (graph construction) |
| `id` | int | 100% | No (metadata) |
| `area` / `net_area` | float | 100% | No (filtering) |
| `unitType` | str | 100% | No (filtering) |

---

## Quick Reference: Room Type Mapping

| ResPlan | RPLAN int | One-hot index (of 25) | Color (rendering) |
|---------|-----------|----------------------|-------------------|
| living | 1 | 1 | `#EE4D4D` (red) |
| bedroom | 2 | 2 | `#C67C7B` (mauve) |
| kitchen | 3 | 3 | `#FFD274` (yellow) |
| bathroom | 4 | 4 | `#BEBEBE` (gray) |
| balcony | 10 | 10 | `#1F849B` (teal) |
| front_door | 11 | 11 | `#727171` (dark gray) |
| storage | 9 | 9 | `#FF8C69` (salmon) |

---

## Quick Reference: Key Functions in `resplan_utils.py`

| Function | Line | Purpose | Use in pipeline |
|----------|------|---------|-----------------|
| `normalize_keys()` | 55 | Fix `balacony` typo | Call first on every plan |
| `get_plan_width()` | 61 | max(width, height) of inner | Filtering by plan size |
| `get_geometries()` | 73 | Extract individual geoms from Multi types | Core room extraction |
| `centroid()` | 83 | Centroid of Polygon/MultiPolygon | Graph node positions |
| `perturb_polygon()` | 92 | Random per-vertex noise | Data augmentation |
| `augment_geom()` | 111 | Rotate/flip/scale geometry | Data augmentation |
| `buffer_shrink_expand()` | 127 | Morphological cleanup | Optional polygon cleanup |
| `buffer_expand_shrink()` | 132 | Fill tiny gaps | Optional polygon cleanup |
| `geometry_to_mask()` | 157 | Shapely → 256×256 binary mask | Debugging / visualization |
| `plot_plan()` | 196 | Colored plan visualization | Debugging / validation |
| `plan_to_graph()` | 247 | Build NetworkX graph from geometry | Reference for graph logic |
| `plot_plan_and_graph()` | 313 | Graph overlay on plan | Debugging / validation |

---

## Glossary

**balacony / balcony:** Typo in some plan dicts. `normalize_keys()` fixes this. Always call
it before accessing plan data.

**buffer (Shapely):** Expanding or contracting a geometry by a distance. `polygon.buffer(0.5)`
grows the polygon outward by 0.5 units. Used in `plan_to_graph()` to detect room adjacency
through walls.

**exterior.coords:** Shapely property returning the ring of `(x, y)` coordinates that form a
polygon's outer boundary. The last coordinate duplicates the first (closed ring). Drop it
with `coords[:-1]`.

**front_door:** The main entrance. In ResPlan, a single `Polygon` geometry. In our model,
treated as room type 11 (entrance). Appears in 100% of plans.

**GeometryCollection:** Shapely type that can contain mixed geometry types (Polygon + LineString, etc.).
`get_geometries()` handles this transparently.

**get_geometries():** The safe way to extract individual geometries from any Shapely object.
Handles None, empty, single, and multi-geometries. Always use this instead of `.geoms`.

**inner:** The `plan["inner"]` MultiPolygon defining the overall floorplan boundary. Use its
`.bounds` for coordinate normalization.

**MultiPolygon:** A Shapely geometry containing multiple disjoint `Polygon` objects. In ResPlan,
`plan["bedroom"]` is a MultiPolygon where each part is one bedroom.

**net_area:** Usable floor area (excluding walls). Stored in plan dict for reference.

**normalize_keys():** Must be called on every plan before processing. Fixes the `balacony` typo.

**plan dict:** The fundamental data structure in ResPlan. A `Dict[str, Any]` with geometry
and metadata keys. There are 17,107 of them in the dataset.

**plan_to_graph():** Builds a room adjacency graph from plan geometry. The pre-built
`plan["graph"]` was created by similar logic. The function uses buffered spatial intersection
to detect room connections.

**simplify (Shapely):** `polygon.simplify(tolerance)` reduces vertex count by removing
points within `tolerance` distance of the simplified line. Essential for rooms with >4 vertices
when working within the 100-point cap.

**unitType:** Classification of the plan. 92.7% are `"Apartment"`, 4.8% `"BuilderFloor"`,
2.2% `"Villa"`. Useful for filtering if you want apartment-only training.

**wall_depth:** Wall thickness in coordinate units (mean ~3.5). Used by `plan_to_graph()` for
buffer calculations when detecting room adjacency.

---

## Reading Order Summary

| # | File | Lines | Time | Purpose |
|---|------|-------|------|---------|
| 1 | `external/ResPlan/README.md` | 87 | 10 min | Project context, paper reference |
| 2 | `external/ResPlan/ResPlan_demo.ipynb` | ~9 cells | 20 min | Official demo walkthrough |
| 3 | `external/ResPlan/resplan_utils.py` | 384 | 1.5 h | All utility functions |
| 4 | `notebooks/01_resplan_exploration.ipynb` | 12 sections | 30 min | Dataset statistics and visualizations |
| 5 | HouseDiffusion `rplanhg_datasets.py:318-389` | 72 | 20 min | `__getitem__()` — the tensor format target |
| 6 | HouseDiffusion `rplanhg_datasets.py:436-497` | 62 | 20 min | `build_graph()` — graph construction for comparison |
| 7 | HouseDiffusion `rplanhg_datasets.py:511-534` | 24 | 10 min | `reader()` — JSON format reference |

**Total estimated time: 4–5 hours** for a thorough read-through with exercises.
