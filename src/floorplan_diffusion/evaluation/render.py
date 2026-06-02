"""SVG/PNG rendering pipeline for floorplan evaluation.

Renders room polygons as 256x256 PNG images suitable for FID computation.
Uses drawSvg for vector drawing and cairosvg for rasterization.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

# Room-type int -> colour for rendering (matches scripts/sample.py).
ROOM_COLORS: dict[int, str] = {
    1: "#EE4D4D",  # living
    2: "#C67C7B",  # bedroom
    3: "#FFD274",  # kitchen
    4: "#BEBEBE",  # bathroom
    10: "#1F849B",  # balcony
    11: "#E78AC3",  # door (interior)
    13: "#A63603",  # front_door
}

# Room name -> colour, mirroring external/ResPlan/resplan_utils.CATEGORY_COLORS.
# Used for matplotlib previews so generated samples match the look of
# notebooks/01_resplan_exploration.ipynb (resplan_utils.plot_plan).
CATEGORY_COLORS: dict[str, str] = {
    "living": "#d9d9d9",     # light gray
    "bedroom": "#66c2a5",    # greenish
    "bathroom": "#fc8d62",   # orange
    "kitchen": "#8da0cb",    # blue
    "door": "#e78ac3",       # pink
    "window": "#a6d854",     # lime
    "wall": "#ffd92f",       # yellow
    "front_door": "#a63603", # dark reddish-brown
    "balcony": "#b3b3b3",    # dark gray
}

# Draw order (earlier = behind), matching plot_plan's category list so doors
# and balconies layer on top of rooms.
CATEGORY_ORDER: tuple[str, ...] = (
    "living", "bedroom", "bathroom", "kitchen",
    "door", "window", "wall", "front_door", "balcony",
)


def points_to_room_polygons(
    points: np.ndarray,
    room_types: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    """Group generated points into per-room polygons.

    Args:
        points: ``[N, 2]`` array of (x, y) coordinates in [-1, 1].
        room_types: ``[N, 25]`` one-hot room type per point.
        room_indices: ``[N, 32]`` one-hot room index per point.
        padding_mask: ``[N]`` — 0 for real, 1 for padding.

    Returns:
        List of ``(polygon_coords, room_type_int)`` tuples.
    """
    rooms: dict[int, list[tuple[float, float]]] = {}
    room_type_map: dict[int, int] = {}

    for i in range(points.shape[0]):
        if padding_mask[i] > 0.5:
            continue
        ridx = int(np.argmax(room_indices[i]))
        rtype = int(np.argmax(room_types[i]))
        rooms.setdefault(ridx, []).append((points[i, 0], points[i, 1]))
        room_type_map[ridx] = rtype

    result = []
    for ridx in sorted(rooms.keys()):
        coords = np.array(rooms[ridx])
        result.append((coords, room_type_map[ridx]))
    return result


DOOR_TYPES: frozenset[int] = frozenset({11, 13})


def _is_degenerate(coords: np.ndarray) -> bool:
    if coords.ndim != 2 or coords.shape[0] < 3:
        return True
    return len(np.unique(coords, axis=0)) < 3


def derive_walls(
    room_polygons: list[tuple[np.ndarray, int]],
    wall_thickness: float = 0.025,
):
    """Synthesize a wall network from room boundaries.

    Walls are *implicit* in the HouseDiffusion representation: they are the
    shared boundaries between adjacent rooms and are never generated as
    objects. This reconstructs them purely from the room-corner polygons by
    buffering each room's boundary ring into a thin strip and unioning the
    strips. Each room interior stays empty (a hole), so the result is a wall
    lattice rather than filled blocks.

    Because it depends only on the polygons, the *same* function applies to
    ground-truth and generated samples, keeping the two panels symmetric.

    Args:
        room_polygons: List of ``(coords_array, room_type_int)``. Door
            polygons are ignored so door openings are not walled over.
        wall_thickness: Total wall width in normalized [-1, 1] units (the
            boundary is buffered by half this amount on each side).

    Returns:
        A Shapely ``(Multi)Polygon`` of the wall network, or ``None`` if no
        usable room geometry was supplied.
    """
    from shapely.geometry import Polygon as _Polygon
    from shapely.ops import unary_union

    half = wall_thickness / 2.0
    bands = []
    for coords, rtype in room_polygons:
        if rtype in DOOR_TYPES or _is_degenerate(coords):
            continue
        poly = _Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        band = poly.boundary.buffer(half, cap_style=2, join_style=2)
        if not band.is_empty:
            bands.append(band)

    if not bands:
        return None
    merged = unary_union(bands)
    return merged if not merged.is_empty else None


def _wall_rings(geom) -> list[np.ndarray]:
    """Flatten a Shapely (Multi)Polygon into exterior + interior ring arrays."""
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    rings: list[np.ndarray] = []
    for poly in polys:
        if getattr(poly, "is_empty", True) or not hasattr(poly, "exterior"):
            continue
        rings.append(np.asarray(poly.exterior.coords))
        rings.extend(np.asarray(r.coords) for r in poly.interiors)
    return rings


def render_floorplan_png(
    room_polygons: list[tuple[np.ndarray, int]],
    output_path: Path,
    resolution: int = 256,
    include_doors: bool = True,
    skip_if_degenerate: bool = False,
    draw_walls: bool = False,
    wall_thickness: float = 0.025,
) -> bool:
    """Render room polygons to a PNG file.

    Args:
        room_polygons: List of ``(coords_array, room_type_int)`` from
            :func:`points_to_room_polygons`.
        output_path: Destination path for the PNG.
        resolution: Image width/height in pixels.
        include_doors: Whether to render door polygons.
        skip_if_degenerate: If True, return without writing when any visible
            room has fewer than 3 distinct points. If False (default),
            degenerate rooms are dropped individually and the rest is drawn.
        draw_walls: If True, overlay a synthesized wall lattice (see
            :func:`derive_walls`). Defaults to False so FID renders stay
            comparable to the room-fill-only HouseDiffusion baseline.
        wall_thickness: Wall width in normalized [-1, 1] units when
            *draw_walls* is enabled.

    Returns:
        True if a PNG was written, False if the floorplan was skipped.
    """
    import cairosvg
    import drawsvg

    visible = [(c, t) for c, t in room_polygons if include_doors or t not in DOOR_TYPES]
    if skip_if_degenerate and any(_is_degenerate(c) for c, _ in visible):
        return False
    visible = [(c, t) for c, t in visible if not _is_degenerate(c)]
    # Doors drawn last so they layer on top of rooms.
    visible.sort(key=lambda r: r[1] in DOOR_TYPES)

    drawing = drawsvg.Drawing(resolution, resolution, displayInline=False)
    drawing.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill="white"))
    for coords, rtype in visible:
        pixel_coords = (coords / 2 + 0.5) * resolution
        drawing.append(
            drawsvg.Lines(
                *pixel_coords.flatten().tolist(),
                close=True,
                fill=ROOM_COLORS.get(rtype, "#888888"),
                fill_opacity=1.0,
                stroke="black",
                stroke_width=1,
            )
        )

    if draw_walls:
        walls = derive_walls(visible, wall_thickness=wall_thickness)
        if walls is not None:
            path = drawsvg.Path(
                fill=CATEGORY_COLORS["wall"],
                fill_rule="evenodd",
                stroke="black",
                stroke_width=1,
            )
            for ring in _wall_rings(walls):
                pixel_ring = (ring / 2 + 0.5) * resolution
                path.M(pixel_ring[0, 0], pixel_ring[0, 1])
                for x, y in pixel_ring[1:]:
                    path.L(x, y)
                path.Z()
            drawing.append(path)

    png_bytes = cairosvg.svg2png(drawing.as_svg())
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB").resize((resolution, resolution))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return True


def render_batch_to_dir(
    samples: list[list[tuple[np.ndarray, int]]],
    output_dir: Path,
    resolution: int = 256,
    skip_if_degenerate: bool = False,
    draw_walls: bool = False,
    wall_thickness: float = 0.025,
) -> int:
    """Render a batch of floorplans to numbered PNGs in a directory.

    Args:
        samples: List of room-polygon lists (one per floorplan).
        output_dir: Directory to write ``0000.png``, ``0001.png``, etc.
        resolution: Image width/height in pixels.
        skip_if_degenerate: Forwarded to :func:`render_floorplan_png`. Skipped
            floorplans leave no gap in the output numbering.
        draw_walls: Forwarded to :func:`render_floorplan_png`. Off by default
            to keep FID renders comparable to the baseline.
        wall_thickness: Forwarded to :func:`render_floorplan_png`.

    Returns:
        Number of floorplans actually written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for room_polygons in samples:
        if render_floorplan_png(
            room_polygons,
            output_dir / f"{written:04d}.png",
            resolution,
            skip_if_degenerate=skip_if_degenerate,
            draw_walls=draw_walls,
            wall_thickness=wall_thickness,
        ):
            written += 1
    return written
