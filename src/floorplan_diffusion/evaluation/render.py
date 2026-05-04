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


def render_floorplan_png(
    room_polygons: list[tuple[np.ndarray, int]],
    output_path: Path,
    resolution: int = 256,
    include_doors: bool = True,
    skip_if_degenerate: bool = False,
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
) -> int:
    """Render a batch of floorplans to numbered PNGs in a directory.

    Args:
        samples: List of room-polygon lists (one per floorplan).
        output_dir: Directory to write ``0000.png``, ``0001.png``, etc.
        resolution: Image width/height in pixels.
        skip_if_degenerate: Forwarded to :func:`render_floorplan_png`. Skipped
            floorplans leave no gap in the output numbering.

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
        ):
            written += 1
    return written
