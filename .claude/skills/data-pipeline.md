---
description: Guidance for working on data conversion between ResPlan and HouseDiffusion formats
---

# Data Pipeline

When working on data conversion between ResPlan and HouseDiffusion formats:

1. **ResPlan source format**: Rooms are Shapely Polygons. Extract bounding boxes via
   `polygon.bounds` → `(minx, miny, maxx, maxy)`. Utilities in `external/ResPlan/resplan_utils.py`.

2. **Room type mapping**: ResPlan uses string keys (living, bedroom, bathroom, kitchen, balcony).
   Map to HouseDiffusion integer encoding (0-24). Reference the RPLAN room type definitions in
   `external/house_diffusion/house_diffusion/rplanhg_datasets.py`.

3. **Edges**: ResPlan NetworkX graph edges have types (adjacency, via_door, via_window).
   HouseDiffusion expects edge line segments as `[x0, y0, x1, y1]` with `ed_rm` mappings
   indicating which rooms each edge connects.

4. **Coordinate normalization**: Convert pixel/geometry coordinates to `[-1, 1]` range
   relative to the floorplan bounding box.

5. **Conversion code**: `src/floorplan_diffusion/data/converter.py`
6. **Entry script**: `scripts/convert_resplan.py`

7. Always validate output JSON against the expected schema before writing.
8. Test with: `uv run pytest tests/test_converter.py -v`
