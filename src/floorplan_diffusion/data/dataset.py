"""PyTorch Dataset for ResPlan floorplans in HouseDiffusion tensor format.

Loads a ResPlan pickle file, extracts room polygons and connectivity graphs,
and produces tensors that exactly match the format expected by the
HouseDiffusion Transformer model.

Each plan is encoded as a NUM_COLUMNS-wide array (MAX_NUM_POINTS rows,
zero-padded) plus three MAX_NUM_POINTS x MAX_NUM_POINTS attention masks. Plans
with more than MAX_NUM_POINTS total polygon vertices are filtered out. Processed
tensors are cached as ``.npz`` files for fast subsequent loading.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOM_TYPES: list[str] = ["living", "bedroom", "kitchen", "window", "bathroom", "balcony"]
"""Canonical ordering of room types when iterating a plan."""

DOOR_TYPES: dict[str, int] = {
    "door": 11,  # interior door opening  (matches RPLAN type 11)
    "front_door": 13,  # building entry door    (matches RPLAN type 13)
}
"""Mapping from ResPlan door-geometry keys to RPLAN integer encoding."""

ROOM_TYPE_TO_INT: dict[str, int] = {
    "living": 1,
    "bedroom": 2,
    "kitchen": 3,
    "bathroom": 4,
    "window": 5,
    "balcony": 10,
    "door": 11,
    "front_door": 13,
}
"""Mapping from ResPlan room-type strings to the RPLAN integer encoding."""

MAX_NUM_POINTS: int = 192
"""Maximum number of polygon vertices per plan (padding target).

Raised from 128 to 192 to retain ~93% of ResPlan plans (vs ~57% at 128); the
total-vertex cap was the sole filter dropping plans. See docs/model_format.md."""

MAX_CORNERS_PER_ROOM: int = 64  # configurable per-room corner limit
"""Target maximum corners per room after simplification."""

CORNER_IDX_DIMS: int = MAX_CORNERS_PER_ROOM
"""Dimensionality of the corner-index one-hot — must be >= MAX_CORNERS_PER_ROOM."""

ROOM_IDX_DIMS: int = 64
"""Dimensionality of the room-index one-hot. It is assigned 1-indexed (first
part -> 1), so a plan may contain at most ``ROOM_IDX_DIMS - 1`` polygon parts
(rooms + doors). Raised from 32 to 64 alongside MAX_NUM_POINTS so the part-count
ceiling no longer rejects plans that fit the vertex budget."""

NUM_COLUMNS: int = 2 + 25 + CORNER_IDX_DIMS + ROOM_IDX_DIMS + 1 + 2
"""Width of the per-point feature vector:
2 coords + 25 room_type + CORNER_IDX_DIMS corner_idx + ROOM_IDX_DIMS room_idx
+ 1 padding + 2 connections.
"""

# ---------------------------------------------------------------------------
# Helpers (mirrors from resplan_utils, kept local to avoid import issues)
# ---------------------------------------------------------------------------


def _normalize_keys(plan: dict[str, Any]) -> dict[str, Any]:
    """Fix known key typos (e.g. ``balacony`` -> ``balcony``) in-place."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan


def _get_polygons(geom_data: Any) -> list[Polygon]:
    """Extract individual ``Polygon`` objects from single/multi/collection geometry."""
    if geom_data is None:
        return []
    if isinstance(geom_data, Polygon):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, MultiPolygon):
        return [g for g in geom_data.geoms if isinstance(g, Polygon) and not g.is_empty]
    # GeometryCollection or other — try iterating
    if hasattr(geom_data, "geoms"):
        return [g for g in geom_data.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def _get_any_geoms(geom_data: Any) -> list[Any]:
    """Extract individual ``Polygon`` or ``LineString`` geometries from any Shapely object.

    Used for door/front_door fields which may be LineStrings instead of Polygons.
    """
    if geom_data is None:
        return []
    if isinstance(geom_data, (Polygon, LineString)):
        return [] if geom_data.is_empty else [geom_data]
    if hasattr(geom_data, "geoms"):
        return [
            g for g in geom_data.geoms if isinstance(g, (Polygon, LineString)) and not g.is_empty
        ]
    return []


def _get_one_hot(index: int, size: int) -> NDArray[np.float64]:
    """Return a one-hot vector of length *size* with a 1 at *index*."""
    return np.eye(size, dtype=np.float64)[index]


def _simplify_polygon(poly: Polygon, max_corners: int = MAX_CORNERS_PER_ROOM) -> Polygon:
    """Simplify a polygon until its exterior has at most *max_corners* vertices.

    Uses Shapely's ``simplify()`` with exponentially increasing tolerance.
    Returns the original polygon unchanged if it already meets the limit.
    """
    corners = len(poly.exterior.coords) - 1  # Shapely closes the ring with a duplicate
    if corners <= max_corners:
        return poly

    tolerance = 0.5
    simplified = poly
    for _ in range(50):
        candidate = poly.simplify(tolerance, preserve_topology=True)
        if not candidate.is_empty and isinstance(candidate, Polygon):
            simplified = candidate
        if len(simplified.exterior.coords) - 1 <= max_corners:
            break
        tolerance *= 2

    return simplified


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ResPlanDataset(Dataset):
    """PyTorch ``Dataset`` that reads a ResPlan pickle and outputs HouseDiffusion tensors.

    Args:
        pickle_path: Path to the ResPlan ``.pkl`` file containing a list of plan dicts.
        cache_dir: Optional directory for ``.npz`` tensor caches.  When provided the
            dataset will load from cache if it exists, or write a new cache after
            processing.  The cache file name is derived from the pickle path and
            *set_name* so different splits do not collide.
        set_name: One of ``"train"`` or ``"eval"``.  Controls random rotation
            augmentation in :meth:`__getitem__`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        pickle_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None = None,
        set_name: str = "train",
    ) -> None:
        super().__init__()
        self.set_name = set_name
        self.num_coords = 2
        self.max_num_points = MAX_NUM_POINTS

        cache_path = self._resolve_cache_path(pickle_path, cache_dir, set_name)

        loaded = False
        if cache_path is not None and cache_path.exists():
            logger.info("Loading cached tensors from %s", cache_path)
            loaded = self._load_cache(cache_path)

        if not loaded:
            logger.info("Processing ResPlan pickle %s …", pickle_path)
            self._process_pickle(pickle_path)
            if cache_path is not None:
                logger.info("Saving cache to %s", cache_path)
                self._save_cache(cache_path)

    # ------------------------------------------------------------------
    # Public Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.houses)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Return ``(arr, cond)`` for a single floorplan.

        ``arr`` has shape ``[2, MAX_NUM_POINTS]`` (x/y coordinates, transposed).
        ``cond`` is a dict of attention masks, room types, corner/room indices,
        padding mask, and polygon-traversal connections — all as NumPy arrays.
        """
        arr = self.houses[idx][:, : self.num_coords].copy()  # [MAX_NUM_POINTS, 2]

        # Random 90-degree rotation + flip augmentation during training.
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

            # Random flips (combined with rotations → 16x augmentation).
            if random.random() < 0.5:
                arr[:, 0] = -arr[:, 0]
            if random.random() < 0.5:
                arr[:, 1] = -arr[:, 1]

        arr = np.transpose(arr, [1, 0])  # [2, MAX_NUM_POINTS]

        # Column offsets: [coords(2) | room_type(25) | corner_idx | room_idx | pad(1) | conn(2)]
        h = self.houses[idx]
        rt_end = 2 + 25
        ci_end = rt_end + CORNER_IDX_DIMS
        ri_end = ci_end + ROOM_IDX_DIMS
        cond: dict[str, Any] = {
            "door_mask": self.door_masks[idx],
            "self_mask": self.self_masks[idx],
            "gen_mask": self.gen_masks[idx],
            "room_types": h[:, 2:rt_end],
            "corner_indices": h[:, rt_end:ci_end],
            "room_indices": h[:, ci_end:ri_end],
            "src_key_padding_mask": 1 - h[:, ri_end],
            "connections": h[:, ri_end + 1 :],
            "area": 10
        }
        return arr.astype(float), cond

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _process_pickle(self, pickle_path: str | os.PathLike[str]) -> None:
        """Load the pickle and convert every valid plan to tensors."""
        with open(pickle_path, "rb") as fh:
            plans: list[dict[str, Any]] = pickle.load(fh)

        houses: list[NDArray[np.float64]] = []
        door_masks: list[NDArray[np.float64]] = []
        self_masks: list[NDArray[np.float64]] = []
        gen_masks: list[NDArray[np.float64]] = []

        skipped = 0
        for plan in plans:
            result = self._process_plan(plan)
            if result is None:
                skipped += 1
                continue
            house_array, d_mask, s_mask, g_mask = result
            houses.append(house_array)
            door_masks.append(d_mask)
            self_masks.append(s_mask)
            gen_masks.append(g_mask)

        logger.info(
            "Processed %d plans, skipped %d (>%d vertices).",
            len(houses),
            skipped,
            MAX_NUM_POINTS,
        )

        self.houses = houses
        self.door_masks = door_masks
        self.self_masks = self_masks
        self.gen_masks = gen_masks

    def _process_plan(
        self, plan: dict[str, Any]
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
        | None
    ):
        """Convert a single ResPlan dict into the ``NUM_COLUMNS``-wide array and masks.

        Returns:
            A tuple ``(house_array, door_mask, self_mask, gen_mask)`` or ``None``
            if the plan has more than :data:`MAX_NUM_POINTS` total vertices.
        """
        plan = _normalize_keys(plan)

        # -- 1. Resolve normalization bounds early (needed for door buffer scale) --
        inner = plan.get("inner")
        if inner is None or inner.is_empty:
            return None
        minx, miny, maxx, maxy = inner.bounds  #########FLAG#########
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        half_extent = max(maxx - minx, maxy - miny) / 2.0
        if half_extent == 0:
            return None
        # Buffer for converting door LineStrings to thin polygons:
        # ~4% of plan scale gives a visually reasonable door width after normalization.
        door_buffer = half_extent * 0.04

        # -- 2. Collect rooms: (corners, room_type_int, graph_node_id) ----------
        rooms: list[tuple[NDArray[np.float64], int, str]] = []
        for room_type in ROOM_TYPES:
            geom_data = plan.get(room_type)
            if geom_data is None:
                continue
            polygons = _get_polygons(geom_data)
            for i, poly in enumerate(polygons):
                poly = _simplify_polygon(poly)
                # Shapely exterior.coords includes the closing duplicate — drop it.
                corners = np.array(poly.exterior.coords[:-1], dtype=np.float64)
                rooms.append((corners, ROOM_TYPE_TO_INT[room_type], f"{room_type}_{i}"))

        # -- 2b. Collect door polygons (LineStrings buffered to thin rectangles) --
        for door_key, door_type_int in DOOR_TYPES.items():
            geom_data = plan.get(door_key)
            if geom_data is None:
                continue
            for i, geom in enumerate(_get_any_geoms(geom_data)):
                if isinstance(geom, LineString):
                    poly = geom.buffer(door_buffer, cap_style=2)  # flat-cap rectangle
                elif isinstance(geom, Polygon):
                    poly = geom
                else:
                    continue
                if poly.is_empty:
                    continue
                poly = _simplify_polygon(poly)
                corners = np.array(poly.exterior.coords[:-1], dtype=np.float64)
                rooms.append((corners, door_type_int, f"{door_key}_{i}"))

        # -- 3. Reject plans containing rooms with >MAX_CORNERS_PER_ROOM corners --
        # Polygons are simplified to MAX_CORNERS_PER_ROOM before this check, so
        # only plans where simplification could not reduce below the limit are
        # rejected here (extremely complex geometry that can't be faithfully encoded).
        for corners, _rtype_int, node_id in rooms:
            if len(corners) > MAX_CORNERS_PER_ROOM:
                logger.debug(
                    "Room %s has %d corners (>%d); rejecting plan.",
                    node_id,
                    len(corners),
                    MAX_CORNERS_PER_ROOM,
                )
                return None

        total_vertices = sum(len(c) for c, _, _ in rooms)
        if total_vertices > MAX_NUM_POINTS or total_vertices == 0:
            return None

        # Reject plans with more parts than the room-index one-hot can encode.
        # room_idx is assigned 1-indexed into a ROOM_IDX_DIMS-wide one-hot, so at
        # most ROOM_IDX_DIMS - 1 parts (rooms + doors) fit. Skip rather than crash
        # with an IndexError on _get_one_hot below.
        if len(rooms) > ROOM_IDX_DIMS - 1:
            logger.warning(
                "Plan has %d polygon parts (> %d limit); skipping plan.",
                len(rooms),
                ROOM_IDX_DIMS - 1,
            )
            return None

        # -- 4. Build the NUM_COLUMNS-wide feature array row by row ------------
        house_parts: list[NDArray[np.float64]] = []
        corner_bounds: list[tuple[int, int]] = []
        node_id_to_room_idx: dict[str, int] = {}
        num_points = 0

        for _room_idx, (corners, rtype_int, node_id) in enumerate(rooms):
            num_room_corners = len(corners)

            # Normalize coordinates.
            coords = corners.copy()
            coords[:, 0] = (coords[:, 0] - cx) / half_extent
            coords[:, 1] = (coords[:, 1] - cy) / half_extent

            # Room type: one-hot of size 25, repeated for each corner.
            rtype_oh = np.tile(_get_one_hot(rtype_int, 25), (num_room_corners, 1))

            # Room index is 1-indexed (first room -> index 1).
            room_idx_oh = np.tile(
                _get_one_hot(len(house_parts) + 1, ROOM_IDX_DIMS), (num_room_corners, 1)
            )

            corner_idx_oh = np.array(
                [_get_one_hot(c, CORNER_IDX_DIMS) for c in range(num_room_corners)]
            )

            # Padding mask: 1 for real points.
            padding_mask = np.ones((num_room_corners, 1), dtype=np.float64)

            # Connections: (current_global_idx, next_global_idx) wrapping within room.
            connections = np.array(
                [[i, (i + 1) % num_room_corners] for i in range(num_room_corners)],
                dtype=np.float64,
            )
            connections += num_points

            # Track bounds for mask construction.
            corner_bounds.append((num_points, num_points + num_room_corners))
            node_id_to_room_idx[node_id] = len(house_parts)
            num_points += num_room_corners

            # Concatenate to NUM_COLUMNS: 2 + 25 + CORNER_IDX_DIMS + ROOM_IDX_DIMS + 1 + 2.
            room_array = np.concatenate(
                (coords, rtype_oh, corner_idx_oh, room_idx_oh, padding_mask, connections),
                axis=1,
            )
            house_parts.append(room_array)

        # Stack all rooms and pad to MAX_NUM_POINTS rows.
        house_layouts = np.concatenate(house_parts, axis=0)  # [num_points, NUM_COLUMNS]
        padding = np.zeros((MAX_NUM_POINTS - len(house_layouts), NUM_COLUMNS), dtype=np.float64)
        house_array = np.concatenate(
            (house_layouts, padding), axis=0
        )  # [MAX_NUM_POINTS, NUM_COLUMNS]

        # -- 4. Build attention masks ------------------------------------------

        # gen_mask: 0 between all real points, 1 elsewhere.
        gen_mask = np.ones((MAX_NUM_POINTS, MAX_NUM_POINTS), dtype=np.float64)
        gen_mask[:num_points, :num_points] = 0

        # self_mask: 0 between points of the *same* room, 1 elsewhere.
        self_mask = np.ones((MAX_NUM_POINTS, MAX_NUM_POINTS), dtype=np.float64)
        for start, end in corner_bounds:
            self_mask[start:end, start:end] = 0

        # door_mask: 0 between rooms that share a graph edge, 1 elsewhere.
        door_mask = np.ones((MAX_NUM_POINTS, MAX_NUM_POINTS), dtype=np.float64)
        graph: nx.Graph | None = plan.get("graph")
        if graph is not None:
            for u, v in graph.edges():
                ri = node_id_to_room_idx.get(u)
                rj = node_id_to_room_idx.get(v)
                if ri is None or rj is None:
                    # Edge involves a node we didn't extract (e.g. front_door).
                    continue
                si, ei = corner_bounds[ri]
                sj, ej = corner_bounds[rj]
                door_mask[si:ei, sj:ej] = 0
                door_mask[sj:ej, si:ei] = 0

        return house_array, door_mask, self_mask, gen_mask

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_cache_path(
        pickle_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None,
        set_name: str,
    ) -> Path | None:
        """Derive a deterministic cache file path, or ``None`` if caching is off."""
        if cache_dir is None:
            return None
        pickle_path = Path(pickle_path).resolve()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Hash the absolute pickle path to keep the cache name short but unique.
        path_hash = hashlib.md5(str(pickle_path).encode()).hexdigest()[:12]
        # Encode the tensor schema (column count + point budget) into the name so
        # that changing the feature layout cannot silently reuse a stale cache.
        schema_tag = f"c{NUM_COLUMNS}p{MAX_NUM_POINTS}"
        return cache_dir / f"resplan_{set_name}_{path_hash}_{schema_tag}.npz"

    def _save_cache(self, cache_path: Path) -> None:
        """Write processed arrays to a compressed ``.npz`` file."""
        np.savez_compressed(
            cache_path,
            houses=np.array(self.houses),
            door_masks=np.array(self.door_masks),
            self_masks=np.array(self.self_masks),
            gen_masks=np.array(self.gen_masks),
        )

    def _load_cache(self, cache_path: Path) -> bool:
        """Restore arrays from an ``.npz`` cache.

        Returns ``True`` if the cache was loaded and matches the current tensor
        schema, ``False`` if it is stale/incompatible (and should be ignored so
        the caller can reprocess from the pickle).
        """
        data = np.load(cache_path, allow_pickle=True)
        houses = list(data["houses"])
        # Defense in depth: the filename already encodes the schema, but verify
        # the actual array shape so a corrupt/mislabelled cache can never feed
        # mis-aligned conditioning into the model.
        if houses:
            shape = np.asarray(houses[0]).shape
            if shape != (MAX_NUM_POINTS, NUM_COLUMNS):
                logger.warning(
                    "Ignoring stale cache %s: house shape %s != expected %s.",
                    cache_path,
                    shape,
                    (MAX_NUM_POINTS, NUM_COLUMNS),
                )
                return False
        self.houses = houses
        self.door_masks = list(data["door_masks"])
        self.self_masks = list(data["self_masks"])
        self.gen_masks = list(data["gen_masks"])
        return True
