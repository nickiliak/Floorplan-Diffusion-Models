"""Architectural ("real-estate plan") matplotlib renderer for floorplan samples.

A presentation-oriented alternative to ``scripts/sample.py:render_floorplan``. It
draws the *same* generated geometry but with conventional architectural symbols:

- **Door swing arcs** instead of flat door rectangles (hinge line + quarter
  circle). The swing side/hinge is a cosmetic *heuristic* (swing into the larger
  adjacent room) — it is NOT predicted by the model.
- **Window glyphs** (parallel lines in a cleared opening) instead of solid blocks.
- **Solid black walls** (the synthesized lattice from :func:`derive_walls`).
- **Room-name labels** at each room centroid.

Real-world dimensions are intentionally omitted: metric scale is discarded during
normalization and would require persisting ``half_extent`` through the data
pipeline.

Reuses the colour/order constants and wall/poly helpers from
:mod:`floorplan_diffusion.evaluation.render` so it stays consistent with the eval
renderer and the FID pipeline (which are left untouched).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from floorplan_diffusion.data.dataset import ROOM_TYPE_TO_INT
from floorplan_diffusion.evaluation.render import (
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    _is_degenerate,
    derive_walls,
    shapely_to_pathpatch,
)

INT_TO_ROOM_NAME: dict[int, str] = {v: k for k, v in ROOM_TYPE_TO_INT.items()}

DOOR_INTS: frozenset[int] = frozenset({ROOM_TYPE_TO_INT["door"], ROOM_TYPE_TO_INT["front_door"]})
WINDOW_INT: int = ROOM_TYPE_TO_INT["window"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _safe_poly(coords: np.ndarray):
    """Build a valid, non-empty Shapely polygon, or ``None`` if degenerate."""
    from shapely.geometry import Polygon as _Polygon

    if _is_degenerate(coords):
        return None
    poly = _Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _min_rect_axes(coords: np.ndarray):
    """Return ``(center, long_unit, long_len, short_unit, short_len)`` of the
    minimum rotated rectangle, or ``None`` if the geometry is unusable.

    Robust to the messy quadrilaterals the model can emit for doors/windows.
    """
    from shapely.geometry import Polygon as _Polygon

    poly = _safe_poly(coords)
    if poly is None:
        return None
    mrr = poly.minimum_rotated_rectangle
    if not isinstance(mrr, _Polygon):
        return None
    pts = np.asarray(mrr.exterior.coords)[:4]
    e0, e1 = pts[1] - pts[0], pts[2] - pts[1]
    l0, l1 = float(np.hypot(*e0)), float(np.hypot(*e1))
    if l0 <= 0 or l1 <= 0:
        return None
    if l0 >= l1:
        long_vec, long_len, short_vec, short_len = e0 / l0, l0, e1 / l1, l1
    else:
        long_vec, long_len, short_vec, short_len = e1 / l1, l1, e0 / l0, l0
    center = np.array([mrr.centroid.x, mrr.centroid.y])
    return center, long_vec, long_len, short_vec, short_len


def _swing_sign(
    center: np.ndarray,
    short_vec: np.ndarray,
    room_centroids: list[tuple[np.ndarray, float]],
) -> float:
    """Pick the perpendicular side to swing a door into (toward the larger room).

    Heuristic only — the model does not predict hinge side or swing direction.
    """
    pos_area = neg_area = 0.0
    pos_seen = neg_seen = False
    for c, area in room_centroids:
        p = float(np.dot(c - center, short_vec))
        if p > 0:
            pos_seen, pos_area = True, max(pos_area, area)
        elif p < 0:
            neg_seen, neg_area = True, max(neg_area, area)
    if pos_seen and not neg_seen:
        return 1.0
    if neg_seen and not pos_seen:
        return -1.0
    return 1.0 if pos_area >= neg_area else -1.0


# ---------------------------------------------------------------------------
# Symbol drawing
# ---------------------------------------------------------------------------


def _draw_door(ax: plt.Axes, coords: np.ndarray, room_centroids) -> None:
    """Draw a door as a cleared opening + leaf line + quarter-circle swing arc."""
    from matplotlib.patches import Arc

    res = _min_rect_axes(coords)
    if res is None:
        if len(coords) >= 3:  # fallback: just outline whatever was generated
            ax.add_patch(
                plt.Polygon(
                    coords,
                    closed=True,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=4,
                )
            )
        return
    center, long_vec, long_len, short_vec, short_len = res
    half_l = long_vec * long_len / 2
    half_s = short_vec * short_len / 2

    # White opening that breaks the black wall.
    rect = np.array(
        [
            center - half_l - half_s,
            center + half_l - half_s,
            center + half_l + half_s,
            center - half_l + half_s,
        ]
    )
    ax.add_patch(plt.Polygon(rect, closed=True, facecolor="white", edgecolor="none", zorder=3.5))

    swing = short_vec * _swing_sign(center, short_vec, room_centroids)
    hinge = center - half_l
    other = center + half_l
    tip = hinge + swing * long_len

    ax.add_line(
        Line2D([hinge[0], tip[0]], [hinge[1], tip[1]], color="black", linewidth=0.8, zorder=4)
    )

    a_closed = np.degrees(np.arctan2(other[1] - hinge[1], other[0] - hinge[0]))
    a_open = np.degrees(np.arctan2(tip[1] - hinge[1], tip[0] - hinge[0]))
    diff = (a_open - a_closed + 180.0) % 360.0 - 180.0  # signed minor sweep (~±90°)
    t1, t2 = (a_closed, a_closed + diff) if diff >= 0 else (a_closed + diff, a_closed)
    ax.add_patch(
        Arc(
            hinge,
            2 * long_len,
            2 * long_len,
            angle=0.0,
            theta1=t1,
            theta2=t2,
            color="black",
            linewidth=0.8,
            zorder=4,
        )
    )


def _draw_window(ax: plt.Axes, coords: np.ndarray) -> None:
    """Draw a window as a cleared opening with parallel mullion lines."""
    res = _min_rect_axes(coords)
    if res is None:
        return
    center, long_vec, long_len, short_vec, short_len = res
    half_l = long_vec * long_len / 2
    half_s = short_vec * short_len / 2

    rect = np.array(
        [
            center - half_l - half_s,
            center + half_l - half_s,
            center + half_l + half_s,
            center - half_l + half_s,
        ]
    )
    ax.add_patch(
        plt.Polygon(
            rect, closed=True, facecolor="white", edgecolor="black", linewidth=0.5, zorder=3
        )
    )
    for frac in (-0.5, 0.0, 0.5):
        off = short_vec * short_len * frac
        p1, p2 = center + off - half_l, center + off + half_l
        ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=0.5, zorder=3))


def _draw_labels(ax: plt.Axes, room_polygons: list[tuple[np.ndarray, int]]) -> None:
    """Place an uppercase room-name label at each room centroid."""
    for coords, rtype in room_polygons:
        if rtype in DOOR_INTS or rtype == WINDOW_INT:
            continue
        poly = _safe_poly(coords)
        if poly is None:
            continue
        name = INT_TO_ROOM_NAME.get(rtype, f"type_{rtype}").replace("_", " ").upper()
        ax.text(
            poly.centroid.x,
            poly.centroid.y,
            name,
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_floorplan_architectural(
    room_polygons: list[tuple[np.ndarray, int]],
    title: str = "",
    ax: plt.Axes | None = None,
    wall_thickness: float = 0.025,
    draw_labels: bool = True,
) -> plt.Axes:
    """Render room polygons as an architectural-style floorplan.

    Drop-in alternative to ``scripts/sample.py:render_floorplan`` with door swing
    arcs, window glyphs, solid black walls and room labels. The same geometry is
    drawn, so generated gaps/overlaps stay visible (by design).

    Args:
        room_polygons: Output of
            :func:`floorplan_diffusion.evaluation.render.points_to_room_polygons`.
        title: Optional plot title.
        ax: Optional axes to draw on.
        wall_thickness: Wall width in normalized [-1, 1] units.
        draw_labels: Whether to draw room-name labels.

    Returns:
        The matplotlib axes with the rendered floorplan.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    order = {name: i for i, name in enumerate(CATEGORY_ORDER)}
    ordered = sorted(
        room_polygons,
        key=lambda r: order.get(INT_TO_ROOM_NAME.get(r[1], ""), len(order)),
    )

    # Room centroids (rooms only) drive the door-swing heuristic.
    room_centroids: list[tuple[np.ndarray, float]] = []
    for coords, rtype in room_polygons:
        if rtype in DOOR_INTS or rtype == WINDOW_INT:
            continue
        poly = _safe_poly(coords)
        if poly is not None:
            room_centroids.append((np.array([poly.centroid.x, poly.centroid.y]), poly.area))

    # 1. Room fills (keep category colours so room type stays readable).
    for coords, rtype in ordered:
        if rtype in DOOR_INTS or rtype == WINDOW_INT or len(coords) < 3:
            continue
        name = INT_TO_ROOM_NAME.get(rtype, f"type_{rtype}")
        color = CATEGORY_COLORS.get(name, "#cccccc")
        ax.add_patch(
            plt.Polygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor="none",
                zorder=1,
                label=name.replace("_", " "),
            )
        )

    # 2. Solid black walls.
    walls = derive_walls(room_polygons, wall_thickness=wall_thickness)
    if walls is not None:
        patch = shapely_to_pathpatch(walls, facecolor="black", edgecolor="none", zorder=2)
        if patch is not None:
            ax.add_patch(patch)

    # 3. Windows then doors on top of the walls.
    for coords, rtype in ordered:
        if rtype == WINDOW_INT:
            _draw_window(ax, coords)
    for coords, rtype in ordered:
        if rtype in DOOR_INTS:
            _draw_door(ax, coords, room_centroids)

    # 4. Labels.
    if draw_labels:
        _draw_labels(ax, room_polygons)

    # Legend proxies for symbols that don't auto-register (rooms already labelled).
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor="black", edgecolor="none", label="wall"))
    ax.plot([], [], color="black", linewidth=0.8, label="door")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor="white", edgecolor="black", label="window"))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_axis_off()
    return ax
