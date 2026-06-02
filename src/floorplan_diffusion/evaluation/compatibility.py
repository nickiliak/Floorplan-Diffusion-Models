"""Compatibility metric: polygon IoU graph reconstruction and edge mismatch counting.

Implements the HouseDiffusion paper's graph-based Compatibility metric by:
1. Building Shapely polygons from generated room coordinates
2. Using IoU overlap to detect which rooms each door connects
3. Reconstructing the adjacency graph
4. Comparing against ground truth and counting mismatches
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

# Door type constants (matching dataset.py encoding).
_INTERIOR_DOOR = 11
_FRONT_DOOR = 13
_DOOR_TYPES = {_INTERIOR_DOOR, _FRONT_DOOR}


def _safe_polygon(coords: np.ndarray) -> Polygon:
    """Build a valid Shapely Polygon, repairing if necessary."""
    p = Polygon(coords)
    if not p.is_valid:
        p = make_valid(p)
    return p


def _polygon_iou(p1: Polygon, p2: Polygon) -> float:
    """Compute IoU between two Shapely polygons, guarding against zero area."""
    intersection = p1.intersection(p2).area
    union = p1.union(p2).area
    if union <= 0:
        return 0.0
    return intersection / union


def _build_gt_graph(
    room_types: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
    door_mask: np.ndarray,
) -> nx.Graph:
    """Reconstruct the ground-truth room-level adjacency graph from point-level door_mask.

    In the door_mask, value 0 between two points means those points' rooms are
    connected (by a door). We extract unique room-level edges.

    Args:
        room_types: ``[N, 25]`` one-hot room types.
        room_indices: ``[N, 32]`` one-hot room indices.
        padding_mask: ``[N]`` — 0 for real, 1 for padding.
        door_mask: ``[N, N]`` attention mask (0 = connected by door).

    Returns:
        NetworkX graph with room indices as nodes and door-connections as edges.
    """
    n = padding_mask.shape[0]

    # Map each point to its room index and type.
    pt_room = np.full(n, -1, dtype=int)
    pt_type = np.full(n, -1, dtype=int)
    for i in range(n):
        if padding_mask[i] > 0.5:
            continue
        pt_room[i] = int(np.argmax(room_indices[i]))
        pt_type[i] = int(np.argmax(room_types[i]))

    # Collect room-level nodes (skip doors and padding).
    room_nodes = set()
    for i in range(n):
        if pt_room[i] >= 0 and pt_type[i] not in _DOOR_TYPES:
            room_nodes.add(pt_room[i])

    graph = nx.Graph()
    graph.add_nodes_from(room_nodes)
    graph.add_node(-1)  # outside node

    # Find room-level edges from door_mask.
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        if padding_mask[i] > 0.5:
            continue
        ri = pt_room[i]
        ti = pt_type[i]
        if ti in _DOOR_TYPES:
            continue
        for j in range(i + 1, n):
            if padding_mask[j] > 0.5:
                continue
            rj = pt_room[j]
            tj = pt_type[j]
            if tj in _DOOR_TYPES:
                continue
            if ri == rj:
                continue
            if door_mask[i, j] < 0.5:  # connected
                edge = (min(ri, rj), max(ri, rj))
                edges.add(edge)

    for a, b in edges:
        graph.add_edge(a, b)

    # Front-door rooms connect to outside.
    for i in range(n):
        if padding_mask[i] > 0.5:
            continue
        ri = pt_room[i]
        ti = pt_type[i]
        if ti == _FRONT_DOOR:
            # Find the room this front door overlaps with via door_mask.
            for j in range(n):
                if padding_mask[j] > 0.5:
                    continue
                rj = pt_room[j]
                tj = pt_type[j]
                if tj in _DOOR_TYPES:
                    continue
                if door_mask[i, j] < 0.5:
                    graph.add_edge(-1, rj)
            break  # one front door is enough to establish outside edges

    return graph


def estimate_graph(
    room_polygons: list[tuple[np.ndarray, int]],
) -> nx.Graph:
    """Estimate the adjacency graph from room polygons using polygon IoU.

    For each door polygon, compute IoU with every non-door room polygon.
    Interior doors (type 11) with top-2 IoU overlapping rooms form a room-room edge.
    Front doors (type 13) connect the highest-IoU room to outside (-1).

    Args:
        room_polygons: List of ``(coords, room_type_int)`` tuples.

    Returns:
        Estimated NetworkX graph.
    """
    # Separate rooms and doors; build Shapely polygons.
    room_entries: list[tuple[int, Polygon, int]] = []  # (index, polygon, type)
    door_entries: list[tuple[int, Polygon, int]] = []

    for idx, (coords, rtype) in enumerate(room_polygons):
        if len(coords) < 3:
            continue
        poly = _safe_polygon(coords)
        if rtype in _DOOR_TYPES:
            door_entries.append((idx, poly, rtype))
        else:
            room_entries.append((idx, poly, rtype))

    graph = nx.Graph()
    for idx, _, _ in room_entries:
        graph.add_node(idx)
    graph.add_node(-1)  # outside

    # For each door, find overlapping rooms via IoU.
    doors_rooms_map: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for d_idx, d_poly, d_type in door_entries:
        for r_idx, r_poly, _ in room_entries:
            iou = _polygon_iou(d_poly, r_poly)
            if 0 < iou < 0.2:
                doors_rooms_map[d_idx].append((r_idx, iou))

    for d_idx, d_poly, d_type in door_entries:
        connections = doors_rooms_map[d_idx]
        connections = sorted(connections, key=lambda t: t[1], reverse=True)
        if d_type == _INTERIOR_DOOR:
            # Interior door: top-2 rooms form an edge.
            if len(connections) >= 2:
                graph.add_edge(connections[0][0], connections[1][0])
        elif d_type == _FRONT_DOOR:
            # Front door: top-1 room connects to outside.
            if len(connections) >= 1:
                graph.add_edge(-1, connections[0][0])

    return graph


def estimate_graph_errors(
    room_polygons: list[tuple[np.ndarray, int]],
    room_types: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
    door_mask: np.ndarray,
) -> int:
    """Estimate adjacency graph from polygons and count edge mismatches.

    Uses Shapely polygon IoU to detect door-room overlaps, reconstructs
    the adjacency graph, and compares against ground truth.

    Args:
        room_polygons: Generated room polygons (output of points_to_room_polygons).
        room_types: ``[N, 25]`` one-hot room types (ground truth).
        room_indices: ``[N, 32]`` one-hot room indices (ground truth).
        padding_mask: ``[N]`` padding mask (ground truth).
        door_mask: ``[N, N]`` attention mask (ground truth).

    Returns:
        Number of incorrect edges (false positives + missed edges).
    """
    gt_graph = _build_gt_graph(room_types, room_indices, padding_mask, door_mask)
    est_graph = estimate_graph(room_polygons)

    # Merge both edge sets.
    all_edges: set[tuple[int, int]] = set()
    for u, v in gt_graph.edges():
        all_edges.add((min(u, v), max(u, v)))
    for u, v in est_graph.edges():
        all_edges.add((min(u, v), max(u, v)))

    mistakes = 0
    for u, v in all_edges:
        gt_has = gt_graph.has_edge(u, v)
        est_has = est_graph.has_edge(u, v)
        if gt_has != est_has:
            mistakes += 1

    return mistakes
