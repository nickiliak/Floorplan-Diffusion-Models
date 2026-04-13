"""Visual test comparing simplify_plan_if_needed ON vs OFF for plan index 9382.

Run with:
    uv run pytest tests/test_simplify.py -s

Requires data/raw/ResPlan.pkl to exist (skipped otherwise).
Saves a side-by-side comparison plot to tests/test_simplify_output.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
PKL_PATH = ROOT / "data" / "raw" / "ResPlan.pkl"
OUTPUT_PATH = Path(__file__).parent / "test_simplify_output.png"
PLAN_INDEX = 9382

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corner_count(plan):
    """Total exterior corners across all room polygons in a plan."""
    from src.helpers.resplandataset import extract_rooms_from_plan, extract_vertices_from_polygon
    rooms = extract_rooms_from_plan(plan)
    return sum(len(extract_vertices_from_polygon(r[0])) for r in rooms), rooms


def _room_summary(rooms) -> str:
    from src.helpers.resplandataset import extract_vertices_from_polygon
    lines = []
    for poly, type_id, type_name in rooms:
        n = len(extract_vertices_from_polygon(poly))
        lines.append(f"  {type_name:12s} (id={type_id}): {n} corners")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PKL_PATH.exists(), reason="ResPlan.pkl not found at data/raw/")
def test_simplify_plan_visual():
    """Load plan 9382, run preprocessing with and without simplification, plot both."""
    import copy

    # Add external/ResPlan to sys.path so resplan_utils is importable
    resplan_pkg = ROOT / "external" / "ResPlan"
    if str(resplan_pkg) not in sys.path:
        sys.path.insert(0, str(resplan_pkg))

    from resplan_utils import plot_plan  # noqa: PLC0415
    from src.helpers.resplandataset import (
        DEFAULT_MAX_NUM_POINTS,
        build_house_tensor,
        load_resplan_pickle,
        simplify_plan_if_needed,
    )

    # ------------------------------------------------------------------
    # 1. Load plan
    # ------------------------------------------------------------------
    plans = load_resplan_pickle(PKL_PATH)
    assert PLAN_INDEX < len(plans), (
        f"Plan index {PLAN_INDEX} out of range (dataset has {len(plans)} plans)."
    )
    original_plan = plans[PLAN_INDEX]
    plan_id = original_plan.get("id", PLAN_INDEX)
    print(f"\nPlan index: {PLAN_INDEX}  id={plan_id}")

    # ------------------------------------------------------------------
    # 2. Without simplification
    # ------------------------------------------------------------------
    plan_no_simp = copy.deepcopy(original_plan)
    corners_before, rooms_before = _corner_count(plan_no_simp)
    print(f"\n--- WITHOUT simplification ---")
    print(f"Total corners: {corners_before}")
    print(_room_summary(rooms_before))

    tensor_before, _ = build_house_tensor(rooms_before, DEFAULT_MAX_NUM_POINTS)
    real_corners_before = int(tensor_before[:, 91].sum())

    # ------------------------------------------------------------------
    # 3. With simplification
    # ------------------------------------------------------------------
    plan_simp = copy.deepcopy(original_plan)
    plan_simp = simplify_plan_if_needed(plan_simp, DEFAULT_MAX_NUM_POINTS)
    corners_after, rooms_after = _corner_count(plan_simp)
    print(f"\n--- WITH simplification (max_points={DEFAULT_MAX_NUM_POINTS}) ---")
    print(f"Total corners: {corners_after}")
    print(_room_summary(rooms_after))

    tensor_after, _ = build_house_tensor(rooms_after, DEFAULT_MAX_NUM_POINTS)
    real_corners_after = int(tensor_after[:, 91].sum())

    # ------------------------------------------------------------------
    # 4. Plot side by side
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    try:
        plot_plan(
            plan_no_simp,
            ax=axes[0],
            legend=True,
            title=(
                f"Plan {plan_id} — NO simplification\n"
                f"corners: {corners_before}  rooms: {len(rooms_before)}"
            ),
            tight=False,
        )
    except ValueError as exc:
        axes[0].set_title(f"Plan {plan_id} — NO simplification\n(no geometries: {exc})")
        axes[0].set_axis_off()

    try:
        plot_plan(
            plan_simp,
            ax=axes[1],
            legend=True,
            title=(
                f"Plan {plan_id} — WITH simplification\n"
                f"corners: {corners_after}  rooms: {len(rooms_after)}"
            ),
            tight=False,
        )
    except ValueError as exc:
        axes[1].set_title(f"Plan {plan_id} — WITH simplification\n(no geometries: {exc})")
        axes[1].set_axis_off()

    fig.suptitle(
        f"simplify_plan_if_needed — plan index {PLAN_INDEX} (id={plan_id})\n"
        f"max_points={DEFAULT_MAX_NUM_POINTS}  |  "
        f"corners {corners_before} → {corners_after}  |  "
        f"tensor real corners {real_corners_before} → {real_corners_after}",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {OUTPUT_PATH}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Assertions
    # ------------------------------------------------------------------
    # Simplified version should have equal or fewer corners
    assert corners_after <= corners_before, (
        f"Simplification increased corners: {corners_before} → {corners_after}"
    )
    # Simplified tensor must fit within max_num_points
    assert real_corners_after <= DEFAULT_MAX_NUM_POINTS, (
        f"Simplified plan still has {real_corners_after} corners > {DEFAULT_MAX_NUM_POINTS}"
    )
    # Room count should be preserved (simplify only touches geometry, not room count)
    assert len(rooms_after) == len(rooms_before), (
        f"Simplification changed room count: {len(rooms_before)} → {len(rooms_after)}"
    )
