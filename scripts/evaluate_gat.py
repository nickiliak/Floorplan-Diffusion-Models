#!/usr/bin/env python
"""Evaluate the trained RoomGraphGAT (HGT encoder) model.

Produces:
  <output_dir>/stats/graph_stats_comparison.png  — node degree / clustering / eigenvector centrality
  <output_dir>/graphs/oneshot_NNN.png            — visualised one-shot generated graphs
  <output_dir>/graphs/autoregressive_NNN.png     — visualised autoregressive generated graphs
  <output_dir>/metrics.json                      — numeric summary

Usage::

    uv run python scripts/evaluate_gat.py --config configs/gat_resplan.yaml
    uv run python scripts/evaluate_gat.py \\
        --config configs/gat_resplan.yaml \\
        --checkpoint models/gat/gat_best.pt \\
        --n_samples 500 --n_vis 12
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.input_generation import (
    RoomGraphGAT,
    build_seed_graph,
    default_relation_matrix,
    seed_mask,
    sort_by_canonical_order,
    ROOM_TYPES,
    ROOM_TO_INT,
    EDGE_TYPES,
    N_EDGE_TYPES,
    MAX_ROOMS,
)
from scripts.train_gat import RoomGraphDataset, _extract_sample

logger = logging.getLogger(__name__)

# ── Colour palettes ───────────────────────────────────────────────────────────

ROOM_COLOURS = {
    "living":     "#4C9BE8",
    "kitchen":    "#E8954C",
    "bedroom":    "#6ABF69",
    "bathroom":   "#B76AE8",
    "balcony":    "#E8D44C",
    "front_door": "#E86A6A",
}

EDGE_COLOURS = {
    "no_edge":    "#DDDDDD",
    "direct":     "#333333",
    "adjacency":  "#888888",
    "via_door":   "#2A7AE2",
    "via_window": "#E2A02A",
}
EDGE_WIDTHS = {
    "no_edge": 0.3, "direct": 2.5, "adjacency": 1.5,
    "via_door": 2.0, "via_window": 2.0,
}
EDGE_STYLES = {
    "no_edge": "dotted", "direct": "solid", "adjacency": "dashed",
    "via_door": "solid", "via_window": "dashed",
}


# ── Graph statistics helpers ──────────────────────────────────────────────────

def edge_matrix_to_nx(program: list[str], edge_labels: torch.Tensor) -> nx.Graph:
    """Convert a room program + edge label matrix to an undirected NetworkX graph."""
    G = nx.Graph()
    for i, rtype in enumerate(program):
        G.add_node(i, room_type=rtype)
    N = len(program)
    for i in range(N):
        for j in range(i + 1, N):
            cls = edge_labels[i, j].item()
            if cls != 0:
                G.add_edge(i, j, edge_type=EDGE_TYPES[cls])
    return G


def compute_graph_stats(graphs: list[nx.Graph]) -> dict[str, list[float]]:
    """Compute per-node statistics across a list of graphs."""
    degrees, clusterings, eigvecs = [], [], []
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        degrees.extend(dict(G.degree()).values())
        clusterings.extend(nx.clustering(G).values())
        try:
            ec = nx.eigenvector_centrality(G, max_iter=200, tol=1e-4)
        except nx.PowerIterationFailedConvergence:
            ec = {n: 0.0 for n in G.nodes()}
        eigvecs.extend(ec.values())
    return {"degree": degrees, "clustering": clusterings, "eigenvector": eigvecs}


# ── Generation ────────────────────────────────────────────────────────────────

def generate_graphs(
    model: RoomGraphGAT,
    programs: list[list[str]],
    threshold: float,
    mode: str,
) -> list[nx.Graph]:
    """Generate graphs for a list of room programs using the specified decoder."""
    graphs = []
    model.eval()
    with torch.no_grad():
        for program in programs:
            rel = default_relation_matrix(program)
            if mode == "oneshot":
                edge_labels = model.generate_oneshot(program, rel, threshold)
            else:
                edge_labels = model.generate_autoregressive(program, rel, threshold)
            graphs.append(edge_matrix_to_nx(program, edge_labels))
    return graphs


# ── Validity metric ───────────────────────────────────────────────────────────

def compute_validity(graphs: list[nx.Graph]) -> float:
    """Fraction of graphs that are connected and contain all seed room types."""
    seed_rooms = {"living", "kitchen", "bedroom", "bathroom"}
    valid = 0
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        room_types = {d["room_type"] for _, d in G.nodes(data=True)}
        if seed_rooms.issubset(room_types) and nx.is_connected(G):
            valid += 1
    return valid / max(len(graphs), 1)


def compute_edge_type_dist(graphs: list[nx.Graph]) -> dict[str, float]:
    """Normalised edge type distribution across all edges in a set of graphs."""
    counts = {e: 0 for e in EDGE_TYPES[1:]}  # exclude no_edge
    for G in graphs:
        for _, _, d in G.edges(data=True):
            etype = d.get("edge_type", "direct")
            if etype in counts:
                counts[etype] += 1
    total = max(sum(counts.values()), 1)
    return {k: v / total for k, v in counts.items()}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_stats_comparison(
    train_stats: dict,
    oneshot_stats: dict,
    autoregressive_stats: dict,
    out_path: Path,
) -> None:
    """Replicate the 3-column stats comparison grid (like the reference image)."""
    rows = [
        ("Training",       train_stats,         "#4C9BE8"),
        ("One-shot",       oneshot_stats,        "#E8954C"),
        ("Autoregressive", autoregressive_stats, "#6ABF69"),
    ]
    metrics = [
        ("degree",      "Node Degree",           range(0, 15)),
        ("clustering",  "Clustering Coefficient", np.linspace(0, 1, 20)),
        ("eigenvector", "Eigenvector Centrality", np.linspace(0, 0.7, 20)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Graph Statistics Comparison", fontsize=16, fontweight="bold")

    for row_idx, (row_label, stats, colour) in enumerate(rows):
        for col_idx, (key, col_label, bins) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            data = stats.get(key, [])
            if data:
                ax.hist(data, bins=bins, color=colour, edgecolor="white", linewidth=0.4)
            ax.set_xlim(bins[0], bins[-1])
            if row_idx == 0:
                ax.set_title(col_label, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved stats comparison → %s", out_path)


def _draw_single_graph(
    ax: plt.Axes,
    G: nx.Graph,
    title: str,
) -> None:
    """Draw one room graph on the given axes."""
    if G.number_of_nodes() == 0:
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        return

    pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colours = [
        ROOM_COLOURS.get(G.nodes[n].get("room_type", "living"), "#CCCCCC")
        for n in G.nodes()
    ]

    # Draw edges grouped by type so styles apply correctly.
    for etype in EDGE_TYPES[1:]:
        edges_of_type = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == etype
        ]
        if edges_of_type:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_of_type, ax=ax,
                edge_color=EDGE_COLOURS[etype],
                width=EDGE_WIDTHS[etype],
                style=EDGE_STYLES[etype],
                alpha=0.85,
            )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=600, linewidths=1.5, edgecolors="white")
    labels = {n: G.nodes[n].get("room_type", "?")[:3].upper() for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6, font_color="white", font_weight="bold")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _make_legend() -> list:
    room_patches = [
        mpatches.Patch(color=c, label=r.replace("_", " ").title())
        for r, c in ROOM_COLOURS.items()
    ]
    edge_lines = [
        Line2D([0], [0], color=EDGE_COLOURS[e], linewidth=2,
               linestyle=EDGE_STYLES[e], label=e.replace("_", " ").title())
        for e in EDGE_TYPES[1:]
    ]
    return room_patches + edge_lines


def plot_graph_grid(
    graphs: list[nx.Graph],
    programs: list[list[str]],
    title: str,
    out_path: Path,
    n_cols: int = 4,
) -> None:
    """Render a grid of generated graphs."""
    n = len(graphs)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes_flat = np.array(axes).flatten()

    for idx, (G, prog) in enumerate(zip(graphs, programs)):
        label = ", ".join(prog[:4]) + ("…" if len(prog) > 4 else "")
        _draw_single_graph(axes_flat[idx], G, label)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    legend_handles = _make_legend()
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 6), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved graph grid → %s", out_path)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    out_cfg   = cfg.get("output", {})
    eval_cfg  = cfg.get("evaluation", {})

    pickle_path  = args.pickle_path  or train_cfg.get("pickle_path", "data/raw/ResPlan.pkl")
    checkpoint   = args.checkpoint   or str(Path(out_cfg.get("model_dir", "models/gat")) / "gat_best.pt")
    output_dir   = Path(args.output_dir or out_cfg.get("output_dir", "outputs/gat"))
    n_samples    = args.n_samples    or eval_cfg.get("n_samples", 500)
    n_vis        = args.n_vis        or eval_cfg.get("n_vis_samples", 12)
    threshold    = args.threshold    or eval_cfg.get("threshold", 0.5)

    stats_dir  = output_dir / "stats"
    graphs_dir = output_dir / "graphs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    model = RoomGraphGAT(**model_cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    logger.info("Loaded checkpoint: %s  (%d params)",
                checkpoint, sum(p.numel() for p in model.parameters()))

    # Load dataset — sample programs from validation set.
    logger.info("Loading dataset from %s …", pickle_path)
    with open(pickle_path, "rb") as f:
        plans = pickle.load(f)

    training_graphs: list[nx.Graph] = []
    all_programs:    list[list[str]] = []
    all_train_nx:    list[nx.Graph]  = []  # parallel to all_programs for vis

    for plan in plans:
        G = plan.get("graph")
        if G is None:
            continue
        result = _extract_sample(G)
        if result is None:
            continue
        program, edge_labels = result
        nx_graph = edge_matrix_to_nx(program, edge_labels)
        training_graphs.append(nx_graph)
        all_programs.append(program)
        all_train_nx.append(nx_graph)

    logger.info("Loaded %d training graphs", len(training_graphs))

    # Sample programs for generation (reproducible).
    rng = random.Random(0)
    sampled_programs = rng.sample(all_programs, min(n_samples, len(all_programs)))

    # Generate.
    logger.info("Generating %d one-shot graphs …", len(sampled_programs))
    oneshot_graphs = generate_graphs(model, sampled_programs, threshold, "oneshot")

    logger.info("Generating %d autoregressive graphs …", len(sampled_programs))
    auto_graphs = generate_graphs(model, sampled_programs, threshold, "autoregressive")

    # ── Graph statistics comparison ──
    logger.info("Computing graph statistics …")
    train_sample = rng.sample(training_graphs, min(n_samples, len(training_graphs)))
    train_stats = compute_graph_stats(train_sample)
    os_stats    = compute_graph_stats(oneshot_graphs)
    ar_stats    = compute_graph_stats(auto_graphs)

    plot_stats_comparison(
        train_stats, os_stats, ar_stats,
        stats_dir / "graph_stats_comparison.png",
    )

    # ── Numeric metrics ──
    os_validity  = compute_validity(oneshot_graphs)
    ar_validity  = compute_validity(auto_graphs)
    os_edge_dist = compute_edge_type_dist(oneshot_graphs)
    ar_edge_dist = compute_edge_type_dist(auto_graphs)
    tr_edge_dist = compute_edge_type_dist(train_sample)

    def mean_std(values: list[float]) -> tuple[float, float]:
        a = np.array(values)
        return float(a.mean()), float(a.std())

    metrics = {
        "n_generated": len(sampled_programs),
        "oneshot": {
            "validity_rate":    os_validity,
            "mean_degree":      mean_std(os_stats["degree"]),
            "mean_clustering":  mean_std(os_stats["clustering"]),
            "edge_type_dist":   os_edge_dist,
        },
        "autoregressive": {
            "validity_rate":    ar_validity,
            "mean_degree":      mean_std(ar_stats["degree"]),
            "mean_clustering":  mean_std(ar_stats["clustering"]),
            "edge_type_dist":   ar_edge_dist,
        },
        "training_reference": {
            "mean_degree":      mean_std(train_stats["degree"]),
            "mean_clustering":  mean_std(train_stats["clustering"]),
            "edge_type_dist":   tr_edge_dist,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", metrics_path)

    logger.info("One-shot   validity: %.1f%%", os_validity * 100)
    logger.info("Autoreg.   validity: %.1f%%", ar_validity * 100)
    logger.info("Training   edge dist: %s", {k: f"{v:.3f}" for k, v in tr_edge_dist.items()})
    logger.info("One-shot   edge dist: %s", {k: f"{v:.3f}" for k, v in os_edge_dist.items()})
    logger.info("Autoreg.   edge dist: %s", {k: f"{v:.3f}" for k, v in ar_edge_dist.items()})

    # ── Graph visualisations ──
    vis_indices  = rng.sample(range(len(all_programs)), min(n_vis, len(all_programs)))
    vis_programs = [all_programs[i] for i in vis_indices]
    vis_train_nx = [all_train_nx[i] for i in vis_indices]

    logger.info("Rendering %d one-shot graph visualisations …", n_vis)
    vis_os = generate_graphs(model, vis_programs, threshold, "oneshot")
    plot_graph_grid(
        vis_os, vis_programs,
        "One-shot Generated Graphs (HGT)",
        graphs_dir / "oneshot_grid.png",
    )

    logger.info("Rendering %d autoregressive graph visualisations …", n_vis)
    vis_ar = generate_graphs(model, vis_programs, threshold, "autoregressive")
    plot_graph_grid(
        vis_ar, vis_programs,
        "Autoregressive Generated Graphs (HGT)",
        graphs_dir / "autoregressive_grid.png",
    )

    # Corresponding ground-truth training graphs for direct visual comparison.
    plot_graph_grid(
        vis_train_nx, vis_programs,
        "Training Graphs (Ground Truth)",
        graphs_dir / "training_grid.png",
    )

    logger.info("Evaluation complete. Outputs in: %s", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoomGraphGAT (HGT)")
    parser.add_argument("--config",       required=True,
                        help="Path to gat_resplan.yaml config file")
    parser.add_argument("--checkpoint",   default=None,
                        help="Override checkpoint path (default: from config)")
    parser.add_argument("--pickle_path",  default=None,
                        help="Override dataset path (default: from config)")
    parser.add_argument("--output_dir",   default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument("--n_samples",    type=int, default=None,
                        help="Number of graphs to generate for statistics")
    parser.add_argument("--n_vis",        type=int, default=None,
                        help="Number of graphs to visualise")
    parser.add_argument("--threshold",    type=float, default=None,
                        help="Edge prediction confidence threshold")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
