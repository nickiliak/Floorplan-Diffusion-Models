"""Sanity tests for the ResPlan dataset loader.

Uses synthetic Shapely polygons so no pickle file is needed.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from src.helpers.resplandataset import (
    DEFAULT_MAX_NUM_POINTS,
    MAX_CORNER_INDEX,
    MAX_ROOM_INDEX,
    NUM_ROOM_TYPE_CLASSES,
    build_attention_masks,
    build_graph_triples,
    build_house_tensor,
    extract_rooms_from_plan,
    extract_vertices_from_polygon,
    normalize_keys,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MAX_PTS = 175  

def _make_simple_rooms():
    """Three rectangular rooms: A-B adjacent, C separated."""
    r_a = box(0, 0, 10, 10)   # living
    r_b = box(10, 0, 20, 10)  # bedroom (touches A at x=10)
    r_c = box(30, 0, 40, 10)  # bathroom (gap from B)
    return [
        (r_a, 1, "living"),
        (r_b, 2, "bedroom"),
        (r_c, 4, "bathroom"),
    ]


def _make_l_shaped_room():
    """Single L-shaped room with 6 vertices."""
    poly = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
    return [(poly, 1, "living")]


# ---------------------------------------------------------------------------
# Milestone 1 — normalize_keys
# ---------------------------------------------------------------------------


class TestNormalizeKeys:
    def test_fixes_balacony_typo(self):
        plan = {"balacony": "geom", "living": "geom2"}
        result = normalize_keys(plan)
        assert "balcony" in result
        assert "balacony" not in result

    def test_noop_when_correct(self):
        plan = {"balcony": "geom"}
        result = normalize_keys(plan)
        assert result == {"balcony": "geom"}


# ---------------------------------------------------------------------------
# Milestone 2 — extract_vertices_from_polygon
# ---------------------------------------------------------------------------


class TestExtractVertices:
    def test_rectangle_has_4_vertices(self):
        rect = box(0, 0, 10, 10)
        verts = extract_vertices_from_polygon(rect)
        assert verts.shape == (4, 2)
        assert verts.dtype == np.float32

    def test_closing_vertex_removed(self):
        rect = box(0, 0, 5, 5)
        coords = list(rect.exterior.coords)
        assert len(coords) == 5  # Shapely includes closing vertex
        verts = extract_vertices_from_polygon(rect)
        assert len(verts) == 4  # we remove it

    def test_l_shape_has_6_vertices(self):
        poly = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
        verts = extract_vertices_from_polygon(poly)
        assert verts.shape == (6, 2)

    def test_empty_polygon(self):
        poly = Polygon()
        verts = extract_vertices_from_polygon(poly)
        assert verts.shape == (0, 2)


# ---------------------------------------------------------------------------
# Milestone 3 — build_house_tensor
# ---------------------------------------------------------------------------


class TestBuildHouseTensor:
    def test_output_shape(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        assert tensor.shape == (MAX_PTS, 94)
        assert tensor.dtype == np.float32

    def test_column_width_sums_to_94(self):
        # 2 + 25 + 32 + 32 + 1 + 2 = 94
        assert 2 + NUM_ROOM_TYPE_CLASSES + MAX_CORNER_INDEX + MAX_ROOM_INDEX + 1 + 2 == 94

    def test_coords_in_range(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        real_mask = tensor[:, 91] == 1
        coords = tensor[real_mask, :2]
        assert coords[:, 0].min() >= -1.0 - 1e-6
        assert coords[:, 0].max() <= 1.0 + 1e-6
        assert coords[:, 1].min() >= -1.0 - 1e-6
        assert coords[:, 1].max() <= 1.0 + 1e-6

    def test_real_corner_count(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        # 3 rectangles × 4 corners = 12
        assert int(tensor[:, 91].sum()) == 12

    def test_padding_is_zeros(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        padding_rows = tensor[12:, :]
        np.testing.assert_array_equal(padding_rows, 0)

    def test_room_type_one_hot(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        # First room (living, type_id=1): cols 2-26 should have 1 at index 1
        room_type_vec = tensor[0, 2:27]
        assert room_type_vec[1] == 1.0
        assert room_type_vec.sum() == 1.0

    def test_corner_index_sequential(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        # First room, corner 0: one-hot at idx 0
        assert tensor[0, 27] == 1.0
        # First room, corner 1: one-hot at idx 1
        assert tensor[1, 28] == 1.0

    def test_room_index_one_indexed(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        # Room 0 gets index 1 in one-hot (1-indexed like rplan)
        room_idx_vec = tensor[0, 59:91]
        assert room_idx_vec[1] == 1.0
        assert room_idx_vec.sum() == 1.0
        # Room 1 gets index 2
        room_idx_vec2 = tensor[4, 59:91]
        assert room_idx_vec2[2] == 1.0

    def test_connections_cyclical(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        # First room (4 corners, starts at idx 0): connections should be
        # [0,1], [1,2], [2,3], [3,0]
        for i in range(4):
            self_idx = tensor[i, 92]
            next_idx = tensor[i, 93]
            assert self_idx == float(i)
            assert next_idx == float((i + 1) % 4)

    def test_corner_bounds(self):
        rooms = _make_simple_rooms()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        assert cb == [[0, 4], [4, 8], [8, 12]]

    def test_empty_rooms(self):
        tensor, cb = build_house_tensor([], max_num_points=MAX_PTS)
        assert tensor.shape == (MAX_PTS, 94)
        np.testing.assert_array_equal(tensor, 0)
        assert cb == []

    def test_l_shaped_room(self):
        rooms = _make_l_shaped_room()
        tensor, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        assert int(tensor[:, 91].sum()) == 6
        assert cb == [[0, 6]]


# ---------------------------------------------------------------------------
# Milestone 4 — build_graph_triples & build_attention_masks
# ---------------------------------------------------------------------------


class TestBuildGraphTriples:
    def test_shape_and_dtype(self):
        rooms = _make_simple_rooms()
        graph = build_graph_triples(rooms)
        # 3 rooms → C(3,2) = 3 pairs
        assert graph.shape == (3, 3)
        assert graph.dtype == np.float32

    def test_adjacent_rooms_positive(self):
        rooms = _make_simple_rooms()
        graph = build_graph_triples(rooms)
        # Room 0 and 1 are adjacent (share boundary at x=10)
        pair_01 = graph[(graph[:, 0] == 0) & (graph[:, 2] == 1)]
        assert len(pair_01) == 1
        assert pair_01[0, 1] == 1.0  # relation = adjacent

    def test_separated_rooms_negative(self):
        rooms = _make_simple_rooms()
        graph = build_graph_triples(rooms)
        # Room 0 and 2 are separated (gap of 20 units)
        pair_02 = graph[(graph[:, 0] == 0) & (graph[:, 2] == 2)]
        assert len(pair_02) == 1
        assert pair_02[0, 1] == -1.0  # relation = not adjacent

    def test_single_room_empty_graph(self):
        rooms = _make_l_shaped_room()
        graph = build_graph_triples(rooms)
        assert graph.shape == (0, 3)

    def test_no_rooms_empty_graph(self):
        graph = build_graph_triples([])
        assert graph.shape == (0, 3)


class TestBuildAttentionMasks:
    def test_shapes(self):
        rooms = _make_simple_rooms()
        _, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        graph = build_graph_triples(rooms)
        dm, sm, gm = build_attention_masks(cb, graph, MAX_PTS, 12)
        assert dm.shape == (MAX_PTS, MAX_PTS)
        assert sm.shape == (MAX_PTS, MAX_PTS)
        assert gm.shape == (MAX_PTS, MAX_PTS)

    def test_self_mask_diagonal_blocks(self):
        rooms = _make_simple_rooms()
        _, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        graph = build_graph_triples(rooms)
        _, sm, _ = build_attention_masks(cb, graph, MAX_PTS, 12)
        # Same-room blocks should be 0 (attend)
        for s, e in cb:
            block = sm[s:e, s:e]
            np.testing.assert_array_equal(block, 0)
        # Cross-room blocks of non-adjacent rooms should be 1 (masked)
        # Room 0 (0:4) and Room 2 (8:12) are not same room
        assert sm[0, 8] == 1.0

    def test_door_mask_adjacent_rooms(self):
        rooms = _make_simple_rooms()
        _, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        graph = build_graph_triples(rooms)
        dm, _, _ = build_attention_masks(cb, graph, MAX_PTS, 12)
        # Room 0 (0:4) and Room 1 (4:8) are adjacent → door_mask should be 0
        assert dm[0, 4] == 0.0
        assert dm[3, 7] == 0.0

    def test_door_mask_non_adjacent_rooms(self):
        rooms = _make_simple_rooms()
        _, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        graph = build_graph_triples(rooms)
        dm, _, _ = build_attention_masks(cb, graph, MAX_PTS, 12)
        # Room 0 and Room 2 are NOT adjacent → door_mask should be 1
        assert dm[0, 8] == 1.0

    def test_gen_mask_real_vs_padding(self):
        rooms = _make_simple_rooms()
        _, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        graph = build_graph_triples(rooms)
        _, _, gm = build_attention_masks(cb, graph, MAX_PTS, 12)
        # Real corners area: 0
        assert gm[0, 0] == 0.0
        assert gm[11, 11] == 0.0
        # Padding area: 1
        assert gm[12, 12] == 1.0
        assert gm[0, 12] == 1.0


# ---------------------------------------------------------------------------
# Milestone 5 — Dataset __getitem__ interface (synthetic, no pickle needed)
# ---------------------------------------------------------------------------


class TestDatasetInterface:
    """Test that the tensor slicing in __getitem__ produces correct shapes.

    We simulate what __getitem__ does without the full Dataset class
    (which requires the pickle file).
    """

    def test_getitem_output_shapes(self):
        rooms = _make_simple_rooms()
        house, cb = build_house_tensor(rooms, max_num_points=MAX_PTS)
        graph = build_graph_triples(rooms)
        total = 12
        dm, sm, gm = build_attention_masks(cb, graph, MAX_PTS, total)

        num_coords = 2
        arr = house[:, :num_coords].copy()
        arr = np.transpose(arr, [1, 0])

        # Pad graph to [200, 3]
        if len(graph) < 200:
            graph = np.concatenate(
                (graph, np.zeros((200 - len(graph), 3), dtype=np.float32)), axis=0
            )

        cond = {
            "door_mask": dm,
            "self_mask": sm,
            "gen_mask": gm,
            "room_types": house[:, num_coords : num_coords + 25],
            "corner_indices": house[:, num_coords + 25 : num_coords + 57],
            "room_indices": house[:, num_coords + 57 : num_coords + 89],
            "src_key_padding_mask": 1 - house[:, num_coords + 89],
            "connections": house[:, num_coords + 90 : num_coords + 92],
            "graph": graph,
        }

        assert arr.shape == (2, MAX_PTS)
        assert cond["door_mask"].shape == (MAX_PTS, MAX_PTS)
        assert cond["self_mask"].shape == (MAX_PTS, MAX_PTS)
        assert cond["gen_mask"].shape == (MAX_PTS, MAX_PTS)
        assert cond["room_types"].shape == (MAX_PTS, 25)
        assert cond["corner_indices"].shape == (MAX_PTS, 32)
        assert cond["room_indices"].shape == (MAX_PTS, 32)
        assert cond["src_key_padding_mask"].shape == (MAX_PTS,)
        assert cond["connections"].shape == (MAX_PTS, 2)
        assert cond["graph"].shape == (200, 3)

    def test_src_key_padding_mask_inverted(self):
        rooms = _make_simple_rooms()
        house, _ = build_house_tensor(rooms, max_num_points=MAX_PTS)
        num_coords = 2
        padding_col = house[:, num_coords + 89]  # col 91
        src_mask = 1 - padding_col
        # Real corners: padding=1, src_mask=0 (attend)
        assert src_mask[0] == 0.0
        # Padding rows: padding=0, src_mask=1 (mask out)
        assert src_mask[MAX_PTS - 1] == 1.0
