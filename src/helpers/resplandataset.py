"""
resplandataset.py — ResPlan dataset loader for the HouseDiffusion pipeline.

Produces the same tensor interface as rplanhg_datasets.py so HouseDiffusion can
consume ResPlan data as a drop-in replacement:
    - house tensor:  [max_num_points, 94]
    - coords output: [2, max_num_points] (analog_bit=False)  or [16, max_num_points]
    - condition dict: door_mask, self_mask, gen_mask, room_types, corner_indices,
                      room_indices, src_key_padding_mask, connections, graph

Data flow:
    ResPlan.pkl  →  load & normalize  →  extract room polygons  →  normalize coords
    →  pack [N, 94] tensor  →  build graph triples & attention masks  →  cache .npz

Dependencies:
    pip install shapely networkx numpy torch
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Maximum sequence length (total corners across all rooms in one plan).
# Rplan uses 100. Configurable here so you can raise it for complex ResPlan plans.
DEFAULT_MAX_NUM_POINTS = 175

# One-hot dimensions (must match HouseDiffusion model expectations)
NUM_ROOM_TYPE_CLASSES = 25   # columns 2–26 in the [N, 94] tensor
MAX_CORNER_INDEX = 32        # columns 27–58
MAX_ROOM_INDEX = 32          # columns 59–90

# Room types present in ResPlan and their mapping to integer IDs.
#
# Strategy: reuse rplan IDs for the 5 core overlapping types.
# Assign ResPlan-specific types to unused slots (14–19).
#
# rplan original IDs (from rplanhg_datasets.py):
#   Types 1–10  → regular rooms (living, master, kitchen, bath, dining,
#                  child, study, 2nd-bath, guest, balcony)
#   Type  15    → hallway   (remapped to 11)
#   Type  17    → outdoor   (remapped to 12)
#   Type  16    → wall-in   (remapped to 13)
#
# ResPlan mapping:
RESPLAN_ROOM_TYPE_MAP: Dict[str, int] = {
    "living": 1,       # rplan: living room
    "bedroom": 2,      # rplan: master room / bedroom
    "kitchen": 3,      # rplan: kitchen
    "bathroom": 4,     # rplan: bathroom
    "balcony": 10,     # rplan: balcony
    "storage": 11,     # rplan slot 11 (was hallway→11)
    "front_door": 14,  # new — not in rplan
    "veranda": 15,     # new maybe remove in the future
    "stair": 16,       # new maybe remove in the future
    "garden": 17,      # new maybe remove in the future
    "parking": 18,     # new maybe remove in the future
    "pool": 19,        # new maybe remove in the future
}

# Reverse map for debugging / visualization suggested by claude
ROOM_TYPE_ID_TO_NAME: Dict[int, str] = {v: k for k, v in RESPLAN_ROOM_TYPE_MAP.items()}

# Which plan keys contain room polygons that become nodes in the floorplan.
# Order matters: it determines the room index in the tensor.
ROOM_KEYS: List[str] = [
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
    "veranda",
    "stair",
    "garden",
    "parking",
    "pool",
    "front_door",
]


# ============================================================================
# Milestone 1 — Pickle loading, key normalization, room extraction
# ============================================================================

def normalize_keys(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Fix known typos in ResPlan plan dicts (in-place).

    Known issues:
        - 'balacony' → 'balcony' (present in some plan dicts)
    """
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan


def get_geometries(geom_data: Any) -> List[Polygon]:
    """Extract individual Polygon/LineString objects from single/multi/collection types.

    Filters out None and empty geometries. For MultiPolygon, unpacks into individual
    Polygons. Returns an empty list for unrecognized types.
    """
    if geom_data is None:
        return []
    if isinstance(geom_data, Polygon):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, LineString):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []


def load_resplan_pickle(pkl_path: str | Path) -> List[Dict[str, Any]]:
    """Load the ResPlan.pkl file and normalize all plan dicts.

    Args:
        pkl_path: Path to ResPlan.pkl (typically data/raw/ResPlan.pkl).

    Returns:
        List of plan dicts with normalized keys. Each dict contains Shapely
        geometries for rooms, walls, doors, windows, plus metadata (id, area,
        net_area, wall_depth, unitType, graph).

    Raises:
        FileNotFoundError: If pkl_path doesn't exist.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"ResPlan pickle not found at {pkl_path}. "
            f"Download it and place it in data/raw/ResPlan.pkl"
        )

    logger.info("Loading ResPlan pickle from %s ...", pkl_path)
    with open(pkl_path, "rb") as f:
        plans = pickle.load(f)

    for plan in plans:
        normalize_keys(plan)

    logger.info("Loaded %d plans from ResPlan.", len(plans))
    return plans


def extract_rooms_from_plan(
    plan: Dict[str, Any],
    room_keys: List[str] | None = None,
    room_type_map: Dict[str, int] | None = None,
) -> List[Tuple[Polygon, int, str]]:
    """Extract individual room polygons from a plan dict.

    Iterates over room_keys, unpacks MultiPolygon into individual Polygons,
    and assigns each room an integer type ID.

    Args:
        plan: A single ResPlan plan dict (already key-normalized).
        room_keys: Which keys to extract rooms from. Defaults to ROOM_KEYS.
        room_type_map: Mapping from key name to integer ID. Defaults to
            RESPLAN_ROOM_TYPE_MAP.

    Returns:
        List of (polygon, type_id, type_name) tuples. Each polygon is a
        Shapely Polygon (not MultiPolygon). front_door may be a Polygon
        with very few vertices — that's expected.
    """
    if room_keys is None:
        room_keys = ROOM_KEYS
    if room_type_map is None:
        room_type_map = RESPLAN_ROOM_TYPE_MAP

    rooms: List[Tuple[Polygon, int, str]] = []

    for key in room_keys:
        geom = plan.get(key)
        if geom is None:
            continue

        type_id = room_type_map.get(key)
        if type_id is None:
            logger.warning("Room key '%s' not in room_type_map — skipping.", key)
            continue

        parts = get_geometries(geom)
        seen_wkbs: set = set()
        for part in parts:
            if isinstance(part, Polygon) and not part.is_empty:
                wkb = part.wkb
                if wkb in seen_wkbs:
                    continue
                seen_wkbs.add(wkb)
                rooms.append((part, type_id, key))
            elif isinstance(part, LineString) and not part.is_empty:
                # front_door can be a LineString in some plans.
                # Buffer it into a thin polygon so it has an area and vertices.
                buffered = part.buffer(0.5)
                if isinstance(buffered, Polygon) and not buffered.is_empty:
                    wkb = buffered.wkb
                    if wkb not in seen_wkbs:
                        seen_wkbs.add(wkb)
                        rooms.append((buffered, type_id, key))

    return rooms


def filter_plans_by_room_count(
    plans: List[Dict[str, Any]],
    target_set: int,
    set_name: str = "train",
) -> List[Dict[str, Any]]:
    """Filter plans by room count, matching rplan's stratification logic.

    In rplan:
        - train set: EXCLUDES plans where room_count == target_set
        - eval set: INCLUDES ONLY plans where room_count == target_set

    This matches the original rplanhg_datasets.py behavior where target_set
    is the held-out room count for evaluation.

    Args:
        plans: Full list of plan dicts.
        target_set: The room count used for stratification.
        set_name: 'train' or 'eval'.

    Returns:
        Filtered list of plan dicts.
    """
    filtered = []
    for plan in plans:
        rooms = extract_rooms_from_plan(plan)
        # Count only "real" rooms, excluding front_door (similar to rplan
        # excluding types 15/17 from the count).
        room_count = sum(
            1 for _, type_id, _ in rooms if type_id != RESPLAN_ROOM_TYPE_MAP["front_door"]
        )
        if set_name == "train" and room_count == target_set:
            continue
        if set_name == "eval" and room_count != target_set:
            continue
        filtered.append(plan)

    logger.info(
        "Filtered %d → %d plans (set_name=%s, target_set=%d).",
        len(plans), len(filtered), set_name, target_set,
    )
    return filtered


# ============================================================================
# Milestone 2 — Polygon vertex extraction & coordinate normalization
# ============================================================================

def extract_vertices_from_polygon(polygon: Polygon) -> np.ndarray:
    """Extract exterior polygon vertices as an [N, 2] float32 array.

    The closing coordinate repeated by Shapely is removed. Consecutive duplicate (debugged by Claude)
    vertices are also removed so downstream corner counts stay stable.
    """
    if polygon.is_empty:
        return np.empty((0, 2), dtype=np.float32)

    vertices = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
    if len(vertices) < 2:
        return vertices

    keep = np.empty(len(vertices), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(vertices[1:] != vertices[:-1], axis=1)
    return vertices[keep]


# TODO: simplify_plan_if_needed()
def simplify_plan_if_needed(plan: Dict[str, Any], max_points: int = DEFAULT_MAX_NUM_POINTS) -> Dict[str, Any]:
    """Simplify room polygons in the plan if total corner count exceeds max_points.

    Uses Shapely's simplify() with a tolerance that is increased until the total
    number of vertices across all rooms is <= max_points. This is a simple way
    to handle very complex plans without losing entire rooms.

    Args:
        plan: A single ResPlan plan dict.
        max_points: Maximum allowed total corners across all rooms.

    Returns:
        The simplified plan dict.
    """
    rooms = extract_rooms_from_plan(plan)
    total_corners = sum(len(extract_vertices_from_polygon(room[0])) for room in rooms)

    if total_corners <= max_points:
        return plan  # No simplification needed

    logger.info(
        "Plan %s has %d corners, exceeding max_points=%d. Simplifying...",
        plan.get("id", "unknown"), total_corners, max_points
    )

    # Start with a small tolerance and increase until we meet the corner limit.
    tolerance = 0.5
    while total_corners > max_points:
        simplified_plan = plan.copy()
        for key in ROOM_KEYS:
            geom = simplified_plan.get(key)
            if geom is not None:
                simplified_geom = geom.simplify(tolerance, preserve_topology=True)
                simplified_plan[key] = simplified_geom
        rooms = extract_rooms_from_plan(simplified_plan)
        total_corners = sum(len(extract_vertices_from_polygon(room[0])) for room in rooms)
        tolerance *= 2  # Exponentially increase tolerance to speed up

    logger.info(
        "Simplified plan %s to %d corners with tolerance=%.2f.",
        plan.get("id", "unknown"), total_corners, tolerance
    )
    return simplified_plan

# ============================================================================
# Milestone 3 — Tensor packing [max_num_points, 94]
# ============================================================================

def _get_one_hot(x: int, z: int) -> np.ndarray:
    """One-hot encode integer ``x`` into a float32 vector of length ``z``."""
    return np.eye(z, dtype=np.float32)[x]


def build_house_tensor(
    rooms: List[Tuple[Polygon, int, str]],
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Pack room polygons into a [max_num_points, 94] tensor.

    Normalizes all room vertices globally to [-1, 1] based on the bounding box
    of the entire plan, then packs each corner into a 94-dim row:

        0–1:   x, y coordinates (normalized to [-1, 1])
        2–26:  one-hot room type (25 classes)
        27–58: one-hot corner index within room (up to 32)
        59–90: one-hot room index within plan (up to 32, 1-indexed)
        91:    padding mask (1 = real corner, 0 = padding)
        92–93: connection indices [self_global_idx, next_global_idx]

    Args:
        rooms: List of (polygon, type_id, type_name) tuples from extract_rooms_from_plan.
        max_num_points: Max corner rows in output tensor.

    Returns:
        (house_tensor, corner_bounds) where house_tensor is [max_num_points, 94]
        float32 and corner_bounds is [[start, end], ...] per room.
    """
    room_vertices = [extract_vertices_from_polygon(poly) for poly, _, _ in rooms]
    non_empty = [v for v in room_vertices if len(v) > 0]
    if not non_empty:
        return np.zeros((max_num_points, 94), dtype=np.float32), []

    # Global bounding-box normalization to [-1, 1]
    all_verts = np.concatenate(non_empty, axis=0)
    min_xy = all_verts.min(axis=0)
    max_xy = all_verts.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    extent = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
    if extent < 1e-6:
        extent = 1.0
    scale = extent / 2.0

    house_parts: List[np.ndarray] = []
    corner_bounds: List[List[int]] = []
    num_points = 0
    room_count = 0

    for (poly, type_id, type_name), verts in zip(rooms, room_vertices):
        if len(verts) == 0:
            continue

        n_corners = len(verts)
        if n_corners > MAX_CORNER_INDEX:
            logger.warning(
                "Room %s has %d corners > %d; truncating.", type_name, n_corners, MAX_CORNER_INDEX
            )
            verts = verts[:MAX_CORNER_INDEX]
            n_corners = MAX_CORNER_INDEX

        if room_count + 1 >= MAX_ROOM_INDEX:
            logger.warning("Too many rooms (%d); skipping remaining.", room_count + 1)
            break

        norm_verts = (verts - center) / scale

        rtype = np.repeat(_get_one_hot(type_id, NUM_ROOM_TYPE_CLASSES)[np.newaxis], n_corners, 0)
        corner_idx = np.array([_get_one_hot(c, MAX_CORNER_INDEX) for c in range(n_corners)])
        room_idx = np.repeat(
            _get_one_hot(room_count + 1, MAX_ROOM_INDEX)[np.newaxis], n_corners, 0
        )
        padding_mask = np.ones((n_corners, 1), dtype=np.float32)
        connections = np.array(
            [[i, (i + 1) % n_corners] for i in range(n_corners)], dtype=np.float32
        )
        connections += num_points

        corner_bounds.append([num_points, num_points + n_corners])
        num_points += n_corners
        room_count += 1

        row = np.concatenate(
            (norm_verts, rtype, corner_idx, room_idx, padding_mask, connections), axis=1
        )
        house_parts.append(row)

    if not house_parts:
        return np.zeros((max_num_points, 94), dtype=np.float32), []

    house_layouts = np.concatenate(house_parts, axis=0)

    if len(house_layouts) > max_num_points:
        house_layouts = house_layouts[:max_num_points]
        corner_bounds = [cb for cb in corner_bounds if cb[0] < max_num_points]
        if corner_bounds and corner_bounds[-1][1] > max_num_points:
            corner_bounds[-1][1] = max_num_points

    padding = np.zeros((max_num_points - len(house_layouts), 94), dtype=np.float32)
    house_layouts = np.concatenate((house_layouts, padding), axis=0)

    return house_layouts.astype(np.float32), corner_bounds

# ============================================================================
# Milestone 4 — Graph conversion & attention masks
# ============================================================================

def build_graph_triples(
    rooms: List[Tuple[Polygon, int, str]],
    adjacency_threshold: float | None = None,
) -> np.ndarray:
    """Build room-level adjacency graph as [num_pairs, 3] triples.

    For every pair (i, j) with i < j, emits [i, 1, j] if rooms are adjacent
    (polygon boundaries within adjacency_threshold) or [i, -1, j] otherwise.
    Uses geometric adjacency rather than NetworkX graph for generality.

    Args:
        rooms: List of (polygon, type_id, type_name) tuples.
        adjacency_threshold: Max distance between room boundaries to count as
            adjacent. If None, defaults to 3% of the plan's bounding-box extent.

    Returns:
        ndarray of shape [num_pairs, 3], dtype float32.
    """
    n = len(rooms)
    if n < 2:
        return np.zeros((0, 3), dtype=np.float32)

    if adjacency_threshold is None:
        all_verts = []
        for poly, _, _ in rooms:
            v = extract_vertices_from_polygon(poly)
            if len(v) > 0:
                all_verts.append(v)
        if all_verts:
            cat = np.concatenate(all_verts, axis=0)
            extent = max(cat.max(0) - cat.min(0))
            adjacency_threshold = max(extent * 0.03, 0.01)
        else:
            adjacency_threshold = 1.0

    triples = []
    for k in range(n):
        poly_k = rooms[k][0]
        for l in range(k + 1, n):
            poly_l = rooms[l][0]
            dist = poly_k.distance(poly_l)
            relation = 1 if dist < adjacency_threshold else -1
            triples.append([k, relation, l])

    return np.array(triples, dtype=np.float32) if triples else np.zeros((0, 3), dtype=np.float32)


def build_attention_masks(
    corner_bounds: List[List[int]],
    graph_triples: np.ndarray,
    max_num_points: int,
    num_real_corners: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three attention masks matching rplan format.

    Convention: 0 = attend, 1 = mask out.

    Args:
        corner_bounds: [[start, end], ...] per room.
        graph_triples: [num_pairs, 3] room adjacency triples from build_graph_triples.
        max_num_points: Sequence length (tensor rows).
        num_real_corners: Total number of real (non-padding) corners.

    Returns:
        (door_mask, self_mask, gen_mask), each [max_num_points, max_num_points] float32.
    """
    door_mask = np.ones((max_num_points, max_num_points), dtype=np.float32)
    self_mask = np.ones((max_num_points, max_num_points), dtype=np.float32)

    for i in range(len(corner_bounds)):
        for j in range(len(corner_bounds)):
            si, ei = corner_bounds[i]
            sj, ej = corner_bounds[j]
            if i == j:
                self_mask[si:ei, sj:ej] = 0
            elif len(graph_triples) > 0 and (
                np.any(np.all(np.equal([i, 1, j], graph_triples), axis=1))
                or np.any(np.all(np.equal([j, 1, i], graph_triples), axis=1))
            ):
                door_mask[si:ei, sj:ej] = 0

    gen_mask = np.ones((max_num_points, max_num_points), dtype=np.float32)
    gen_mask[:num_real_corners, :num_real_corners] = 0

    return door_mask, self_mask, gen_mask


# ============================================================================
# Milestone 5 — PyTorch Dataset & DataLoader
# ============================================================================

_get_bin = lambda x, z: [int(y) for y in format(x, "b").zfill(z)]


class ResplanDataset(Dataset):
    """PyTorch Dataset for ResPlan floorplans — drop-in replacement for RPlanhgDataset.

    Produces the same ``__getitem__`` interface::

        arr:  [2, max_num_points] float coords (analog_bit=False)
              or [16, max_num_points] binary (analog_bit=True)
        cond: dict with door_mask, self_mask, gen_mask, room_types,
              corner_indices, room_indices, src_key_padding_mask,
              connections, graph
    """

    def __init__(
        self,
        set_name: str,
        analog_bit: bool,
        target_set: int,
        pkl_path: str | Path = "data/raw/ResPlan.pkl",
        max_num_points: int = DEFAULT_MAX_NUM_POINTS,
        cache_dir: str = "processed_resplan",
    ):
        super().__init__()
        self.set_name = set_name
        self.analog_bit = analog_bit
        self.target_set = target_set
        self.max_num_points = max_num_points
        self.num_coords = 2

        cache_file = os.path.join(cache_dir, f"resplan_{set_name}_{target_set}.npz")

        if os.path.exists(cache_file):
            logger.info("Loading cached dataset from %s", cache_file)
            data = np.load(cache_file, allow_pickle=True)
            self.houses = list(data["houses"])
            self.graphs = list(data["graphs"])
            self.door_masks = list(data["door_masks"])
            self.self_masks = list(data["self_masks"])
            self.gen_masks = list(data["gen_masks"])
        else:
            self._process_from_pickle(pkl_path, cache_dir, cache_file)

    def _process_from_pickle(
        self, pkl_path: str | Path, cache_dir: str, cache_file: str
    ) -> None:
        """Load ResPlan pickle and build all tensors + masks."""
        from tqdm import tqdm

        plans = load_resplan_pickle(pkl_path)
        plans = filter_plans_by_room_count(plans, self.target_set, self.set_name)

        self.houses: List[np.ndarray] = []
        self.graphs: List[np.ndarray] = []
        self.door_masks: List[np.ndarray] = []
        self.self_masks: List[np.ndarray] = []
        self.gen_masks: List[np.ndarray] = []

        skipped = 0
        for plan in tqdm(plans, desc=f"Processing {self.set_name} plans"):
            plan = simplify_plan_if_needed(plan, self.max_num_points)
            rooms = extract_rooms_from_plan(plan)

            if len(rooms) == 0 or len(rooms) >= MAX_ROOM_INDEX:
                skipped += 1
                continue

            total_corners = sum(
                len(extract_vertices_from_polygon(r[0])) for r in rooms
            )
            if total_corners == 0 or total_corners > self.max_num_points:
                skipped += 1
                continue

            house_tensor, corner_bounds = build_house_tensor(rooms, self.max_num_points)
            graph_triples = build_graph_triples(rooms)
            door_mask, self_mask, gen_mask = build_attention_masks(
                corner_bounds, graph_triples, self.max_num_points, total_corners
            )

            self.houses.append(house_tensor)
            self.graphs.append(graph_triples)
            self.door_masks.append(door_mask)
            self.self_masks.append(self_mask)
            self.gen_masks.append(gen_mask)

        logger.info(
            "Processed %d plans, skipped %d. Dataset size: %d",
            len(plans),
            skipped,
            len(self.houses),
        )

        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_file,
            houses=self.houses,
            graphs=np.array(self.graphs, dtype=object),
            door_masks=self.door_masks,
            self_masks=self.self_masks,
            gen_masks=self.gen_masks,
        )
        logger.info("Saved cache to %s", cache_file)

    def __len__(self) -> int:
        return len(self.houses)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        house = self.houses[idx]
        arr = house[:, : self.num_coords].copy()

        graph = self.graphs[idx]
        if len(graph) < 200:
            graph = np.concatenate(
                (graph, np.zeros((200 - len(graph), 3), dtype=np.float32)), axis=0
            )
        else:
            graph = graph[:200]

        cond = {
            "door_mask": self.door_masks[idx],
            "self_mask": self.self_masks[idx],
            "gen_mask": self.gen_masks[idx],
            "room_types": house[:, self.num_coords : self.num_coords + 25],
            "corner_indices": house[:, self.num_coords + 25 : self.num_coords + 57],
            "room_indices": house[:, self.num_coords + 57 : self.num_coords + 89],
            "src_key_padding_mask": 1 - house[:, self.num_coords + 89],
            "connections": house[:, self.num_coords + 90 : self.num_coords + 92],
            "graph": graph,
        }

        # Random 90-degree rotation augmentation (train only)
        if self.set_name == "train":
            rotation = random.randint(0, 3)
            if rotation == 1:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 0] = -arr[:, 0]
            elif rotation == 2:
                arr[:, [0, 1]] = -arr[:, [1, 0]]
            elif rotation == 3:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 1] = -arr[:, 1]

        if not self.analog_bit:
            arr = np.transpose(arr, [1, 0])  # [max_num_points, 2] → [2, max_num_points]
            return arr.astype(float), cond
        else:
            ONE_HOT_RES = 256
            xs = np.clip(((arr[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int), 0, 255)
            ys = np.clip(((arr[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int), 0, 255)
            xs = np.array([_get_bin(x, 8) for x in xs])
            ys = np.array([_get_bin(x, 8) for x in ys])
            arr_onehot = np.concatenate([xs, ys], 1)
            arr_onehot = np.transpose(arr_onehot, [1, 0])
            arr_onehot[arr_onehot == 0] = -1
            return arr_onehot.astype(float), cond


def load_resplan_data(
    batch_size: int,
    analog_bit: bool,
    target_set: int = 8,
    set_name: str = "train",
    pkl_path: str | Path = "data/raw/ResPlan.pkl",
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    cache_dir: str = "processed_resplan",
):
    """Create an infinite generator over (coords, cond) batches.

    Drop-in replacement for ``load_rplanhg_data()``.

    Yields:
        (arr, cond) tuples where arr is [batch, 2, max_num_points] and cond
        is a dict of condition tensors.
    """
    logger.info("Loading %s data (target_set=%d)", set_name, target_set)
    deterministic = set_name != "train"
    dataset = ResplanDataset(
        set_name, analog_bit, target_set, pkl_path, max_num_points, cache_dir
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        drop_last=False,
    )
    while True:
        yield from loader


# ============================================================================
# Quick smoke test (run with: python -m src.helpers.resplandataset)
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Try loading the pickle if it exists
    default_path = Path("data/raw/ResPlan.pkl")
    pkl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not pkl_path.exists():
        # Run a quick synthetic test instead
        print("Pickle not found — running synthetic smoke test.")
        from shapely.geometry import box

        r_a = box(0, 0, 10, 10)
        r_b = box(10, 0, 20, 10)
        r_c = box(30, 0, 40, 10)
        rooms = [(r_a, 1, "living"), (r_b, 2, "bedroom"), (r_c, 4, "bathroom")]

        tensor, cb = build_house_tensor(rooms, max_num_points=100)
        print(f"house tensor shape: {tensor.shape}")
        print(f"corner_bounds: {cb}")
        print(f"coord range: [{tensor[:, 0].min():.3f}, {tensor[:, 0].max():.3f}]")

        graph = build_graph_triples(rooms)
        print(f"graph triples ({len(graph)}):\n{graph}")

        total = sum(len(extract_vertices_from_polygon(r)) for r, _, _ in rooms)
        dm, sm, gm = build_attention_masks(cb, graph, 100, total)
        print(f"door_mask zeros: {int((dm == 0).sum())}")
        print(f"self_mask zeros: {int((sm == 0).sum())}")
        print(f"gen_mask  zeros: {int((gm == 0).sum())}")
        print("Synthetic smoke test passed ✓")
        sys.exit(0)

    plans = load_resplan_pickle(pkl_path)

    # Show stats
    print(f"\nTotal plans: {len(plans)}")

    # Room extraction on first plan
    rooms = extract_rooms_from_plan(plans[0])
    print(f"\nPlan #0 rooms ({len(rooms)}):")
    for poly, type_id, type_name in rooms:
        n_verts = len(list(poly.exterior.coords)) - 1
        print(f"  {type_name:12s} (id={type_id:2d}): {n_verts} vertices, area={poly.area:.1f}")

    # Tensor packing on first plan
    tensor, cb = build_house_tensor(rooms)
    print(f"\nTensor shape: {tensor.shape}")
    print(f"Corner bounds: {cb}")
    real_rows = int(tensor[:, 91].sum())
    print(f"Real corners: {real_rows}")

    # Graph + masks
    graph = build_graph_triples(rooms)
    adj_count = int((graph[:, 1] == 1).sum()) if len(graph) > 0 else 0
    print(f"Graph triples: {len(graph)} ({adj_count} adjacent)")

    # Filter test
    for target in [6, 8, 10]:
        train_plans = filter_plans_by_room_count(plans, target_set=target, set_name="train")
        eval_plans = filter_plans_by_room_count(plans, target_set=target, set_name="eval")
        print(f"\ntarget_set={target}: train={len(train_plans)}, eval={len(eval_plans)}")
