"""Compatibility metric: geometric graph reconstruction and edge mismatch counting.

Implements the HouseDiffusion paper's graph-based Compatibility metric by:
1. Building Shapely polygons from generated room coordinates
2. Reconstructing the room graph geometrically, mirroring the edge rules of
   ``resplan_utils.plan_to_graph`` (adjacency / via_door / via_window / direct)
3. Comparing against the ground-truth graph and counting edge mismatches
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

# Room/part type constants (matching dataset.py encoding).
_LIVING = 1
_BEDROOM = 2
_KITCHEN = 3
_BATHROOM = 4
_WINDOW = 5
_BALCONY = 10
_INTERIOR_DOOR = 11
_FRONT_DOOR = 13
_DOOR_TYPES = {_INTERIOR_DOOR, _FRONT_DOOR}


def _safe_polygon(coords: np.ndarray) -> Polygon:
    """Build a valid Shapely Polygon, repairing if necessary."""
    p = Polygon(coords)
    if not p.is_valid:
        p = make_valid(p)
    return p


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
        room_indices: ``[N, ROOM_IDX_DIMS]`` one-hot room indices.
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

    return graph


def estimate_graph(
    room_polygons: list[tuple[np.ndarray, int]],
    connector_buffer: float = 0.01,
    adjacency_buffer: float = 0.05,
    min_room_area: float = 0.02,
) -> nx.Graph:
    """Estimate the room graph from polygons, mirroring ``plan_to_graph``.

    Applies the same edge rules as ``resplan_utils.plan_to_graph`` (the source
    of the ground-truth graphs) to the supplied geometry:

    - *direct*: a front door touching a living room connects it to outside (-1).
    - *adjacency*: kitchen/bedroom within *adjacency_buffer* of a living room.
    - *via_door* / *via_window*: bathroom/balcony connected to living/bedroom
      when a door or window polygon intersects both (buffered by
      *connector_buffer*).

    Rooms smaller than *min_room_area* stay isolated nodes: the stored ResPlan
    graphs were built on cleaned geometry without sliver fragments, so edges
    involving slivers never appear in the ground truth.

    Node ids are the dataset's 1-indexed room indices: the dataset assigns
    room indices contiguously from 1 and ``points_to_room_polygons`` returns
    parts sorted by room index, so list position + 1 recovers the same ids
    used by :func:`_build_gt_graph`.

    Args:
        room_polygons: List of ``(coords, room_type_int)`` tuples.
        connector_buffer: Buffer (in normalized [-1, 1] units) applied to rooms
            when testing door/window/front-door intersection.
        adjacency_buffer: Max polygon distance (normalized units) for the
            kitchen/bedroom-to-living adjacency rule.
        min_room_area: Rooms with polygon area (normalized units squared) below
            this take part in no edges.

    Returns:
        Estimated NetworkX graph.
    """
    by_type: dict[int, list[tuple[int, Polygon]]] = defaultdict(list)
    graph = nx.Graph()
    graph.add_node(-1)  # outside

    for pos, (coords, rtype) in enumerate(room_polygons):
        if len(coords) < 3:
            continue
        poly = _safe_polygon(coords)
        if poly.is_empty:
            continue
        node = pos + 1  # 1-indexed room idx (see docstring)
        if rtype not in _DOOR_TYPES and rtype != _WINDOW:
            graph.add_node(node)
            if poly.area < min_room_area:
                continue  # sliver: keep the node but form no edges
        by_type[rtype].append((node, poly))

    living = by_type[_LIVING]
    connectors = [p for _, p in by_type[_INTERIOR_DOOR]] + [p for _, p in by_type[_WINDOW]]

    # direct: front door -> living room -> outside (-1).
    for _, fd_poly in by_type[_FRONT_DOOR]:
        for l_node, l_poly in living:
            if fd_poly.intersects(l_poly.buffer(connector_buffer)):
                graph.add_edge(-1, l_node)

    # adjacency: kitchen/bedroom <-> living.
    for rtype in (_KITCHEN, _BEDROOM):
        for r_node, r_poly in by_type[rtype]:
            for l_node, l_poly in living:
                if r_poly.distance(l_poly) <= adjacency_buffer:
                    graph.add_edge(r_node, l_node)

    # via_door / via_window: bathroom/balcony <-> living/bedroom.
    targets = living + by_type[_BEDROOM]
    for rtype in (_BATHROOM, _BALCONY):
        for r_node, r_poly in by_type[rtype]:
            r_buffered = r_poly.buffer(connector_buffer)
            for conn_poly in connectors:
                if not conn_poly.intersects(r_buffered):
                    continue
                for t_node, t_poly in targets:
                    if conn_poly.intersects(t_poly.buffer(connector_buffer)):
                        graph.add_edge(r_node, t_node)

    return graph


def estimate_graph_errors(
    room_polygons: list[tuple[np.ndarray, int]],
    room_types: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
    door_mask: np.ndarray,
) -> int:
    """Estimate adjacency graph from polygons and count edge mismatches.

    Reconstructs the room graph geometrically (see :func:`estimate_graph`)
    and compares it against the ground-truth graph encoded in *door_mask*.

    Args:
        room_polygons: Generated room polygons (output of points_to_room_polygons).
        room_types: ``[N, 25]`` one-hot room types (ground truth).
        room_indices: ``[N, ROOM_IDX_DIMS]`` one-hot room indices (ground truth).
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
