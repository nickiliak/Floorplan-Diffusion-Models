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
    1: "#EE4D4D",   # living
    2: "#C67C7B",   # bedroom
    3: "#FFD274",   # kitchen
    4: "#BEBEBE",   # bathroom
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


def render_floorplan_png(
    room_polygons: list[tuple[np.ndarray, int]],
    output_path: Path,
    resolution: int = 256,
    include_doors: bool = True,
) -> None:
    """Render room polygons to a PNG file.

    Args:
        room_polygons: List of ``(coords_array, room_type_int)`` from
            :func:`points_to_room_polygons`.
        output_path: Destination path for the PNG.
        resolution: Image width/height in pixels.
        include_doors: Whether to render door polygons.
    """
    import cairosvg
    import drawsvg

    drawing = drawsvg.Drawing(resolution, resolution, displayInline=False)
    drawing.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill="white"))

    door_types = {11, 13}

    # Draw rooms first, then doors on top (matching reference ordering).
    for coords, rtype in room_polygons:
        if rtype in door_types and not include_doors:
            continue
        if rtype in door_types:
            continue  # defer doors to second pass
        color = ROOM_COLORS.get(rtype, "#888888")
        pixel_coords = (coords / 2 + 0.5) * resolution
        flat = pixel_coords.flatten().tolist()
        drawing.append(
            drawsvg.Lines(
                *flat,
                close=True,
                fill=color,
                fill_opacity=1.0,
                stroke="black",
                stroke_width=1,
            )
        )

    if include_doors:
        for coords, rtype in room_polygons:
            if rtype not in door_types:
                continue
            color = ROOM_COLORS.get(rtype, "#888888")
            pixel_coords = (coords / 2 + 0.5) * resolution
            flat = pixel_coords.flatten().tolist()
            drawing.append(
                drawsvg.Lines(
                    *flat,
                    close=True,
                    fill=color,
                    fill_opacity=1.0,
                    stroke="black",
                    stroke_width=1,
                )
            )

    png_bytes = cairosvg.svg2png(drawing.as_svg())
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img = img.resize((resolution, resolution))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def render_batch_to_dir(
    samples: list[list[tuple[np.ndarray, int]]],
    output_dir: Path,
    resolution: int = 256,
) -> None:
    """Render a batch of floorplans to numbered PNGs in a directory.

    Args:
        samples: List of room-polygon lists (one per floorplan).
        output_dir: Directory to write ``0000.png``, ``0001.png``, etc.
        resolution: Image width/height in pixels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, room_polygons in enumerate(samples):
        render_floorplan_png(room_polygons, output_dir / f"{i:04d}.png", resolution)
