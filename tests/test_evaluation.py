"""Tests for the evaluation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon


def _can_import(module: str) -> bool:
    """Check whether a module can be imported."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


class TestCoordinateTransform:
    """Test the [-1,1] -> [0,256] coordinate mapping used in rendering."""

    def test_corner_mapping(self) -> None:
        coords = np.array([[-1, -1], [1, 1], [0, 0]])
        expected = np.array([[0, 0], [256, 256], [128, 128]])
        result = (coords / 2 + 0.5) * 256
        np.testing.assert_allclose(result, expected)

    def test_negative_one_maps_to_zero(self) -> None:
        result = (-1 / 2 + 0.5) * 256
        assert result == 0.0

    def test_positive_one_maps_to_resolution(self) -> None:
        result = (1 / 2 + 0.5) * 256
        assert result == 256.0


class TestPointsToRoomPolygons:
    """Tests for points_to_room_polygons helper."""

    def test_basic_grouping(self) -> None:
        from floorplan_diffusion.evaluation.render import points_to_room_polygons

        n = 6
        points = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.1, 0.1],  # room 0
                [0.5, 0.5],
                [0.6, 0.5],
                [0.6, 0.6],  # room 1
            ]
        )
        room_types = np.zeros((n, 25))
        room_types[0, 1] = 1  # living
        room_types[1, 1] = 1
        room_types[2, 1] = 1
        room_types[3, 2] = 1  # bedroom
        room_types[4, 2] = 1
        room_types[5, 2] = 1

        room_indices = np.zeros((n, 32))
        room_indices[0, 0] = 1
        room_indices[1, 0] = 1
        room_indices[2, 0] = 1
        room_indices[3, 1] = 1
        room_indices[4, 1] = 1
        room_indices[5, 1] = 1

        padding_mask = np.zeros(n)

        result = points_to_room_polygons(points, room_types, room_indices, padding_mask)
        assert len(result) == 2
        assert result[0][1] == 1  # living
        assert result[1][1] == 2  # bedroom

    def test_padding_skipped(self) -> None:
        from floorplan_diffusion.evaluation.render import points_to_room_polygons

        n = 4
        points = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.0]])
        room_types = np.zeros((n, 25))
        room_types[:3, 1] = 1
        room_types[3, 2] = 1
        room_indices = np.zeros((n, 32))
        room_indices[:3, 0] = 1
        room_indices[3, 1] = 1
        padding_mask = np.array([0, 0, 0, 1])  # last point is padding

        result = points_to_room_polygons(points, room_types, room_indices, padding_mask)
        assert len(result) == 1


class TestRender:
    """Tests for render.py — SVG/PNG rendering pipeline."""

    @pytest.mark.skipif(
        not _can_import("drawsvg") or not _can_import("cairosvg"),
        reason="drawsvg/cairosvg not installed (eval extras required)",
    )
    def test_render_produces_png(self, tmp_path: Path) -> None:
        from floorplan_diffusion.evaluation.render import render_floorplan_png

        coords = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        room_polygons = [(coords, 1)]  # living room
        out = tmp_path / "test.png"
        render_floorplan_png(room_polygons, out)
        assert out.exists()
        assert out.stat().st_size > 0

    @pytest.mark.skipif(
        not _can_import("drawsvg") or not _can_import("cairosvg"),
        reason="drawsvg/cairosvg not installed (eval extras required)",
    )
    def test_render_batch(self, tmp_path: Path) -> None:
        from floorplan_diffusion.evaluation.render import render_batch_to_dir

        coords = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        samples = [[(coords, 1)], [(coords, 2)]]
        render_batch_to_dir(samples, tmp_path)
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 2


class TestCompatibility:
    """Tests for compatibility.py — polygon IoU graph reconstruction."""

    def test_overlapping_polygons_detected(self) -> None:
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])
        iou = p1.intersection(p2).area / p1.union(p2).area
        assert 0 < iou < 1

    def test_non_overlapping_polygons_zero_iou(self) -> None:
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        union_area = p1.union(p2).area
        iou = p1.intersection(p2).area / union_area if union_area > 0 else 0
        assert iou == 0.0

    def test_safe_polygon_handles_invalid(self) -> None:
        from floorplan_diffusion.evaluation.compatibility import _safe_polygon

        # Bowtie (self-intersecting) polygon.
        coords = np.array([(0, 0), (1, 1), (1, 0), (0, 1)])
        p = _safe_polygon(coords)
        assert p.is_valid

    def test_polygon_iou_zero_area(self) -> None:
        from floorplan_diffusion.evaluation.compatibility import _polygon_iou

        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        # Degenerate polygon (line).
        p2 = Polygon([(5, 5), (6, 5), (6, 5)])
        iou = _polygon_iou(p1, p2)
        assert iou >= 0.0

    def test_estimate_graph_two_rooms_one_door(self) -> None:
        """Two rooms connected by an interior door produce one room-room edge."""
        from floorplan_diffusion.evaluation.compatibility import estimate_graph

        # Room 0: large square on the left.
        room0 = (np.array([(0, 0), (5, 0), (5, 10), (0, 10)]), 1)
        # Room 1: large square on the right.
        room1 = (np.array([(5, 0), (10, 0), (10, 10), (5, 10)]), 2)
        # Door: small rectangle overlapping both rooms at the boundary.
        door = (np.array([(4.5, 4), (5.5, 4), (5.5, 6), (4.5, 6)]), 11)

        graph = estimate_graph([room0, room1, door])
        assert graph.has_edge(0, 1)

    def test_estimate_graph_front_door_connects_outside(self) -> None:
        """A front door should connect a room to the outside node (-1)."""
        from floorplan_diffusion.evaluation.compatibility import estimate_graph

        room0 = (np.array([(0, 0), (10, 0), (10, 10), (0, 10)]), 1)
        # Front door overlapping room0 on the left edge.
        front = (np.array([(-0.5, 4), (0.5, 4), (0.5, 6), (-0.5, 6)]), 13)

        graph = estimate_graph([room0, front])
        assert graph.has_edge(-1, 0)


class TestBuildGtGraph:
    """Tests for _build_gt_graph — ground-truth graph reconstruction."""

    @staticmethod
    def _make_inputs(
        n_points: int,
        room_assignments: list[tuple[int, int]],
        door_connections: list[tuple[int, int]],
        front_door_points: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build synthetic inputs for _build_gt_graph.

        Args:
            n_points: Total number of points.
            room_assignments: ``(point_idx, room_idx)`` pairs for non-door rooms.
                Room type defaults to 1 (living).
            door_connections: ``(point_a, point_b)`` pairs that should be
                connected (door_mask[a,b] = 0).
            front_door_points: Indices of front-door typed points (type 13).

        Returns:
            ``(room_types, room_indices, padding_mask, door_mask)`` arrays.
        """
        room_types = np.zeros((n_points, 25))
        room_indices = np.zeros((n_points, 32))
        padding_mask = np.ones(n_points)  # default: padding
        door_mask = np.ones((n_points, n_points))  # default: not connected

        for pt_idx, room_idx in room_assignments:
            room_types[pt_idx, 1] = 1  # living room type
            room_indices[pt_idx, room_idx] = 1
            padding_mask[pt_idx] = 0

        if front_door_points:
            for pt_idx in front_door_points:
                room_types[pt_idx] = 0  # clear
                room_types[pt_idx, 13] = 1  # front door type
                padding_mask[pt_idx] = 0

        for a, b in door_connections:
            door_mask[a, b] = 0
            door_mask[b, a] = 0

        return room_types, room_indices, padding_mask, door_mask

    def test_two_rooms_connected(self) -> None:
        """Two rooms with door_mask connection produce an edge."""
        from floorplan_diffusion.evaluation.compatibility import _build_gt_graph

        # 4 points: 2 in room 0, 2 in room 1, connected via door_mask
        rt, ri, pm, dm = self._make_inputs(
            n_points=4,
            room_assignments=[(0, 0), (1, 0), (2, 1), (3, 1)],
            door_connections=[(0, 2)],  # room 0 point <-> room 1 point
        )
        graph = _build_gt_graph(rt, ri, pm, dm)
        assert graph.has_edge(0, 1)

    def test_disconnected_rooms(self) -> None:
        """Rooms without door_mask connections have no edge."""
        from floorplan_diffusion.evaluation.compatibility import _build_gt_graph

        rt, ri, pm, dm = self._make_inputs(
            n_points=4,
            room_assignments=[(0, 0), (1, 0), (2, 1), (3, 1)],
            door_connections=[],  # no connections
        )
        graph = _build_gt_graph(rt, ri, pm, dm)
        assert not graph.has_edge(0, 1)
        # Both rooms should still be nodes
        assert 0 in graph.nodes
        assert 1 in graph.nodes

    def test_front_door_connects_outside(self) -> None:
        """A front-door point connected to a room creates an outside edge."""
        from floorplan_diffusion.evaluation.compatibility import _build_gt_graph

        # Point 0,1 = room 0 (living), point 2 = front door
        rt, ri, pm, dm = self._make_inputs(
            n_points=3,
            room_assignments=[(0, 0), (1, 0)],
            door_connections=[(2, 0)],  # front door connected to room 0 point
            front_door_points=[2],
        )
        # front door point needs a room index too
        ri[2, 5] = 1  # assign to some door "room index"
        graph = _build_gt_graph(rt, ri, pm, dm)
        assert graph.has_edge(-1, 0)

    def test_padding_points_ignored(self) -> None:
        """Padded points should not create edges or nodes."""
        from floorplan_diffusion.evaluation.compatibility import _build_gt_graph

        rt, ri, pm, dm = self._make_inputs(
            n_points=4,
            room_assignments=[(0, 0), (1, 0)],
            door_connections=[],
        )
        # Points 2 and 3 remain padding (default)
        graph = _build_gt_graph(rt, ri, pm, dm)
        assert len(graph.edges) == 0
        assert 0 in graph.nodes  # room 0 exists


class TestEstimateGraphErrors:
    """Tests for estimate_graph_errors — the main compatibility entry point."""

    def test_perfect_match_zero_errors(self) -> None:
        """When estimated graph matches GT exactly, errors should be 0."""
        from floorplan_diffusion.evaluation.compatibility import estimate_graph_errors

        # Two rooms with an interior door between them — GT and polygons match.
        n = 10
        room_types = np.zeros((n, 25))
        room_indices = np.zeros((n, 32))
        padding_mask = np.ones(n)
        door_mask = np.ones((n, n))

        # Room 0: 3 points forming a big left square
        for i in range(3):
            room_types[i, 1] = 1  # living
            room_indices[i, 0] = 1
            padding_mask[i] = 0

        # Room 1: 3 points forming a big right square
        for i in range(3, 6):
            room_types[i, 2] = 1  # bedroom
            room_indices[i, 1] = 1
            padding_mask[i] = 0

        # Interior door: 3 points overlapping boundary
        for i in range(6, 9):
            room_types[i, 11] = 1  # interior door
            room_indices[i, 2] = 1
            padding_mask[i] = 0

        # Connect rooms through door_mask (room 0 pt <-> room 1 pt)
        door_mask[0, 3] = 0
        door_mask[3, 0] = 0

        # Build polygons that match the GT graph
        room0_poly = (np.array([(0, 0), (5, 0), (5, 10), (0, 10)]), 1)
        room1_poly = (np.array([(5, 0), (10, 0), (10, 10), (5, 10)]), 2)
        door_poly = (np.array([(4.5, 4), (5.5, 4), (5.5, 6), (4.5, 6)]), 11)

        errors = estimate_graph_errors(
            [room0_poly, room1_poly, door_poly],
            room_types,
            room_indices,
            padding_mask,
            door_mask,
        )
        assert errors == 0

    def test_known_mismatch_counts_errors(self) -> None:
        """When estimated graph differs from GT, errors should be > 0."""
        from floorplan_diffusion.evaluation.compatibility import estimate_graph_errors

        n = 6
        room_types = np.zeros((n, 25))
        room_indices = np.zeros((n, 32))
        padding_mask = np.ones(n)
        door_mask = np.ones((n, n))

        # Room 0: 3 points
        for i in range(3):
            room_types[i, 1] = 1
            room_indices[i, 0] = 1
            padding_mask[i] = 0

        # Room 1: 3 points
        for i in range(3, 6):
            room_types[i, 2] = 1
            room_indices[i, 1] = 1
            padding_mask[i] = 0

        # GT says rooms are connected
        door_mask[0, 3] = 0
        door_mask[3, 0] = 0

        # But polygons have NO door, so estimated graph won't have the edge
        room0_poly = (np.array([(0, 0), (5, 0), (5, 10), (0, 10)]), 1)
        room1_poly = (np.array([(50, 50), (60, 50), (60, 60), (50, 60)]), 2)
        # No door polygon at all

        errors = estimate_graph_errors(
            [room0_poly, room1_poly],
            room_types,
            room_indices,
            padding_mask,
            door_mask,
        )
        assert errors > 0


class TestFID:
    """Tests for fid.py — FID wrapper."""

    def test_fid_import(self) -> None:
        pytest.importorskip("pytorch_fid")
