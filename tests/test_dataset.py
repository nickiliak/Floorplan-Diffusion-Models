"""Tests for the ResPlanDataset deterministic train/eval split."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from shapely.geometry import LineString, Polygon

from floorplan_diffusion.data.dataset import ResPlanDataset


def _make_plan() -> dict[str, Any]:
    """Build a minimal two-room synthetic plan in the ResPlan dict format."""
    living = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    bedroom = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
    inner = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
    door = LineString([(100, 40), (100, 60)])
    front_door = LineString([(0, 40), (0, 60)])

    graph = nx.Graph()
    graph.add_node("living_0", geometry=living, type="living", area=living.area)
    graph.add_node("bedroom_0", geometry=bedroom, type="bedroom", area=bedroom.area)
    graph.add_edge("living_0", "bedroom_0", type="adjacency")

    return {
        "living": living,
        "bedroom": bedroom,
        "inner": inner,
        "door": door,
        "front_door": front_door,
        "graph": graph,
        "wall_width": 1.0,
    }


@pytest.fixture
def pickle_path(tmp_path: Path) -> Path:
    """Write a 20-plan synthetic pickle and return its path."""
    plans = [_make_plan() for _ in range(20)]
    path = tmp_path / "plans.pkl"
    with open(path, "wb") as fh:
        pickle.dump(plans, fh)
    return path


class TestDeterministicSplit:
    """The train/eval split must be disjoint, exhaustive, and reproducible."""

    def test_split_disjoint_and_exhaustive(self, pickle_path: Path) -> None:
        train = ResPlanDataset(pickle_path, cache_dir=None, set_name="train", val_fraction=0.2)
        evald = ResPlanDataset(pickle_path, cache_dir=None, set_name="eval", val_fraction=0.2)

        assert set(train.raw_indices).isdisjoint(evald.raw_indices)
        assert sorted(train.raw_indices + evald.raw_indices) == list(range(20))
        assert len(evald) == 4  # int(20 * 0.2)

    def test_split_reproducible(self, pickle_path: Path) -> None:
        eval_a = ResPlanDataset(pickle_path, cache_dir=None, set_name="eval", val_fraction=0.2)
        eval_b = ResPlanDataset(pickle_path, cache_dir=None, set_name="eval", val_fraction=0.2)
        assert eval_a.raw_indices == eval_b.raw_indices

    def test_cache_roundtrip_preserves_split(self, pickle_path: Path, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        fresh = ResPlanDataset(pickle_path, cache_dir=cache_dir, set_name="eval", val_fraction=0.2)
        caches = list(cache_dir.glob("resplan_eval_*.npz"))
        assert len(caches) == 1

        cached = ResPlanDataset(pickle_path, cache_dir=cache_dir, set_name="eval", val_fraction=0.2)
        assert cached.raw_indices == fresh.raw_indices
        assert len(cached) == len(fresh)

    def test_train_and_eval_caches_do_not_collide(self, pickle_path: Path, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        train_path = ResPlanDataset._resolve_cache_path(pickle_path, cache_dir, "train", 0.1)
        eval_path = ResPlanDataset._resolve_cache_path(pickle_path, cache_dir, "eval", 0.1)
        other_fraction = ResPlanDataset._resolve_cache_path(pickle_path, cache_dir, "eval", 0.2)
        assert train_path != eval_path
        assert eval_path != other_fraction
