#!/usr/bin/env python
"""Train the RoomGraphGAT model (shared encoder, one-shot + autoregressive decoders).

Both decoders are trained jointly on the ResPlan graph dataset extracted in
notebooks/05_graph_inspection.ipynb.

Usage::

    uv run python scripts/train_gat.py
    uv run python scripts/train_gat.py --pickle_path data/raw/ResPlan_simple.pkl
    uv run python scripts/train_gat.py --epochs 100 --lr 1e-3
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.input_generation.GAT import (
    MAX_ROOMS,
    N_EDGE_TYPES,
    ROOM_TO_INT,
    RoomGraphGAT,
    build_seed_graph,
    seed_mask,
    sort_by_canonical_order,
)

logger = logging.getLogger(__name__)

# ── Dataset ───────────────────────────────────────────────────────────────────

EDGE_PRIORITY = {
    "no_edge": 0, "via_window": 1, "adjacency": 2, "via_door": 3, "direct": 4,
}
EDGE_TYPES = ["no_edge", "direct", "adjacency", "via_door", "via_window"]
EDGE_TO_INT = {e: i for i, e in enumerate(EDGE_TYPES)}


def _extract_sample(G) -> tuple[list[str], torch.Tensor] | None:
    """Convert a NetworkX graph to (room_program, edge_label_matrix).

    Returns None if the graph has fewer than 4 nodes or exceeds MAX_ROOMS.
    """
    nodes = [(n, G.nodes[n].get("type")) for n in G.nodes()]
    nodes = [(n, t) for n, t in nodes if t in ROOM_TO_INT]
    if len(nodes) < 4 or len(nodes) > MAX_ROOMS:
        return None

    program = sort_by_canonical_order([t for _, t in nodes])
    # Build mapping nx-node → program index, handling duplicate room types correctly.
    type_counters: dict[str, int] = {}
    program_positions: list[int] = []
    for pos, rtype in enumerate(program):
        type_counters[rtype] = type_counters.get(rtype, -1) + 1
    type_assigned: dict[str, int] = {}
    node_idx: dict[str, int] = {}
    for n, t in nodes:
        count = type_assigned.get(t, 0)
        # Find the count-th occurrence of t in program.
        found = 0
        for pos, rtype in enumerate(program):
            if rtype == t:
                if found == count:
                    node_idx[n] = pos
                    break
                found += 1
        type_assigned[t] = count + 1

    N = len(program)
    edge_labels = torch.zeros(N, N, dtype=torch.long)

    for u, v, d in G.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        etype = d.get("type", "unknown")
        if etype not in EDGE_TO_INT:
            continue
        i, j = node_idx[u], node_idx[v]
        new_cls = EDGE_TO_INT[etype]
        cur_cls = edge_labels[i, j].item()
        if EDGE_PRIORITY.get(etype, 0) > EDGE_PRIORITY.get(EDGE_TYPES[cur_cls], 0):
            edge_labels[i, j] = new_cls
            edge_labels[j, i] = new_cls

    return program, edge_labels


class RoomGraphDataset(Dataset):
    """Dataset of (room_program, edge_label_matrix) pairs from ResPlan graphs."""

    def __init__(self, pickle_path: str | Path) -> None:
        with open(pickle_path, "rb") as f:
            raw_plans = pickle.load(f)

        self.samples: list[tuple[list[str], torch.Tensor]] = []
        for plan in raw_plans:
            G = plan.get("graph")
            if G is None:
                continue
            result = _extract_sample(G)
            if result is not None:
                self.samples.append(result)

        logger.info("Loaded %d graph samples from %s", len(self.samples), pickle_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[str], torch.Tensor]:
        return self.samples[idx]


def collate_fn(
    batch: list[tuple[list[str], torch.Tensor]],
) -> list[tuple[list[str], torch.Tensor]]:
    """No padding needed — graphs have variable size, process one at a time."""
    return batch


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: RoomGraphDataset, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.zeros(N_EDGE_TYPES, dtype=torch.float)
    for program, edge_labels in dataset.samples:
        N = len(program)
        # Upper-triangle indices only.
        idx_i, idx_j = torch.triu_indices(N, N, offset=1)
        flat = edge_labels[idx_i, idx_j]
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    total = counts.sum()
    weights = total / (N_EDGE_TYPES * counts.clamp(min=1))
    weights = weights / weights[1:].min()
    logger.info("Class weights: %s", weights.tolist())
    return weights.to(device)


def oneshot_loss(
    logits: torch.Tensor,
    edge_labels: torch.Tensor,
    s_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Vectorised cross-entropy over upper-triangle pairs, excluding seed edges."""
    N = logits.size(0)
    idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=logits.device)
    # Exclude seed pairs.
    keep = ~s_mask[idx_i, idx_j]
    if keep.sum() == 0:
        return logits.sum() * 0.0
    logits_flat  = logits[idx_i[keep], idx_j[keep]]       # [M, N_EDGE_TYPES]
    targets_flat = edge_labels[idx_i[keep], idx_j[keep]]  # [M]
    return F.cross_entropy(logits_flat, targets_flat, weight=class_weights)


def autoregressive_loss(
    logits_list: list[torch.Tensor],
    pairs: list[tuple[int, int]],
    edge_labels: torch.Tensor,
    s_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Vectorised cross-entropy over autoregressive step predictions, excluding seed edges."""
    if not logits_list:
        return class_weights.sum() * 0.0
    keep = [k for k, (i, j) in enumerate(pairs) if not s_mask[i, j]]
    if not keep:
        return class_weights.sum() * 0.0
    logits_flat  = torch.stack([logits_list[k] for k in keep])           # [M, 5]
    targets_flat = torch.tensor(
        [edge_labels[pairs[k][0], pairs[k][1]].item() for k in keep],
        dtype=torch.long, device=class_weights.device,
    )
    return F.cross_entropy(logits_flat, targets_flat, weight=class_weights)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logger.info("Device: %s", device)

    # Data.
    dataset = RoomGraphDataset(args.pickle_path)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    class_weights = compute_class_weights(dataset, device)

    # Model.
    model = RoomGraphGAT(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss_os = train_loss_ar = 0.0
        for batch in train_loader:
            program, edge_labels = batch[0]
            edge_labels = edge_labels.to(device)
            node_types  = torch.tensor(
                [ROOM_TO_INT[r] for r in program], dtype=torch.long, device=device
            )
            s_mask = seed_mask(program).to(device)
            seed_size = s_mask.any(dim=1).sum().item()

            optimizer.zero_grad()

            # One-shot forward.
            logits_os = model.forward_oneshot(node_types)
            loss_os = oneshot_loss(logits_os, edge_labels, s_mask, class_weights)

            # Autoregressive forward.
            logits_ar, pairs = model.forward_autoregressive(node_types, seed_size)
            loss_ar = autoregressive_loss(
                logits_ar, pairs, edge_labels, s_mask, class_weights
            )

            loss = loss_os + loss_ar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_os += loss_os.item()
            train_loss_ar += loss_ar.item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss_os = val_loss_ar = 0.0
        with torch.no_grad():
            for batch in val_loader:
                program, edge_labels = batch[0]
                edge_labels = edge_labels.to(device)
                node_types  = torch.tensor(
                    [ROOM_TO_INT[r] for r in program], dtype=torch.long, device=device
                )
                s_mask = seed_mask(program).to(device)
                seed_size = s_mask.any(dim=1).sum().item()

                logits_os = model.forward_oneshot(node_types)
                val_loss_os += oneshot_loss(
                    logits_os, edge_labels, s_mask, class_weights
                ).item()

                logits_ar, pairs = model.forward_autoregressive(node_types, seed_size)
                val_loss_ar += autoregressive_loss(
                    logits_ar, pairs, edge_labels, s_mask, class_weights
                ).item()

        n_train, n_val = len(train_loader), len(val_loader)
        logger.info(
            "Epoch %3d/%d  train os=%.4f ar=%.4f  val os=%.4f ar=%.4f  lr=%.2e",
            epoch, args.epochs,
            train_loss_os / n_train, train_loss_ar / n_train,
            val_loss_os   / n_val,   val_loss_ar   / n_val,
            scheduler.get_last_lr()[0],
        )

        val_loss = (val_loss_os + val_loss_ar) / n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "gat_best.pt")
            logger.info("  ↳ saved best model (val_loss=%.4f)", best_val_loss)

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            ckpt_path = out_dir / f"gat_epoch{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  ↳ saved periodic checkpoint: %s", ckpt_path.name)

    torch.save(model.state_dict(), out_dir / "gat_last.pt")
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RoomGraphGAT")
    parser.add_argument("--pickle_path",  default="data/raw/ResPlan.pkl")
    parser.add_argument("--output_dir",   default="models/gat")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--embed_dim",    type=int,   default=64)
    parser.add_argument("--hidden_dim",   type=int,   default=64)
    parser.add_argument("--num_layers",   type=int,   default=3)
    parser.add_argument("--num_heads",    type=int,   default=4)
    parser.add_argument("--mlp_hidden",   type=int,   default=128)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--save_interval", type=int,   default=50,
                        help="Save a checkpoint every N epochs (0 = disable)")
    parser.add_argument("--cpu",            action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
