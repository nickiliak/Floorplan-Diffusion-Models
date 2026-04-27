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
        points = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.1, 0.1],  # room 0
            [0.5, 0.5], [0.6, 0.5], [0.6, 0.6],  # room 1
        ])
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


class TestFID:
    """Tests for fid.py — FID wrapper."""

    def test_fid_import(self) -> None:
        pytest.importorskip("pytorch_fid")
