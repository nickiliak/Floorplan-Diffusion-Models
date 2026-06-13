"""Heterogeneous Graph Transformer for room-graph generation.

Architecture
------------
Replaces the homogeneous GATEncoder with an HGT encoder whose attention weights
are parameterised by both node type (room category) and edge relation type
(geometric entity connecting two rooms).

HGT reference
    Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020).
    "Heterogeneous Graph Transformer."
    The Web Conference (WWW 2020). https://arxiv.org/abs/2003.01332
    Original implementation: https://github.com/acbull/pyHGT

Node types  (6)
    living, kitchen, bedroom, bathroom, balcony, front_door

Edge relation types  (5)
    room      — two rooms share a wall/space (default for unknown pairs)
    door      — rooms connected via a door opening
    wall      — rooms separated by a solid wall segment
    window    — rooms connected via a window opening
    boundary  — rooms adjacent to the outer building boundary

Edge output classes  (5)  ← what the decoders predict
    0 = no_edge  1 = direct  2 = adjacency  3 = via_door  4 = via_window

Minimum graph
    Seed {living, kitchen, bedroom, bathroom} with hardcoded edges injected as
    a fixed prefix; neither decoder predicts those edges.

Maximum rooms
    Hard cap at MAX_ROOMS (default 15).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Node / edge / output constants ──────────────────────────────────────────

ROOM_TYPES  = ["living", "kitchen", "bedroom", "bathroom", "balcony", "front_door"]
ROOM_TO_INT = {r: i for i, r in enumerate(ROOM_TYPES)}
N_ROOM_TYPES = len(ROOM_TYPES)   # 6

# Geometric relation types that label edges in the heterogeneous graph.
# These are structural priors, not the output edge classes.
RELATION_TYPES  = ["room", "door", "wall", "window", "boundary"]
RELATION_TO_INT = {r: i for i, r in enumerate(RELATION_TYPES)}
N_RELATIONS     = len(RELATION_TYPES)   # 5

# Output edge classes (what the decoders predict).
EDGE_TYPES   = ["no_edge", "direct", "adjacency", "via_door", "via_window"]
N_EDGE_TYPES = len(EDGE_TYPES)   # 5

# Canonical addition order for the autoregressive decoder.
CANONICAL_ORDER = ["living", "kitchen", "bedroom", "bathroom", "balcony", "front_door"]

# Minimum seed graph: (type_a, type_b, edge_class)
SEED_EDGES = [
    ("living", "kitchen",    2),   # adjacency
    ("living", "bedroom",    2),   # adjacency
    ("living", "bathroom",   3),   # via_door
    ("living", "front_door", 1),   # direct
]

MAX_ROOMS = 15

# Default relation type index when no geometric context is known.
_DEFAULT_RELATION = RELATION_TO_INT["room"]


# ── HGT building blocks ──────────────────────────────────────────────────────

class HGTLayer(nn.Module):
    """Single Heterogeneous Graph Transformer layer.

    Implements the HGT attention mechanism (Hu et al., 2020):

        Attention(s→t via r):
            Q_τt(h_t) · (K_τs(h_s) @ W_r)ᵀ · λ_r / √d_k

        Message(s→t via r):
            V_τs(h_s) @ M_r

        Update(t):
            O_τt( Σ softmax(attn) · message ) + skip

    All parameters are grouped by node type (τ) and relation type (r),
    making attention weights sensitive to *what* the nodes are and *how*
    they are geometrically connected.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension (= num_heads × head_dim).
        num_node_types: Number of distinct node types.
        num_relations: Number of distinct edge relation types.
        num_heads: Number of attention heads.
        dropout: Dropout applied to attention weights.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_node_types: int = N_ROOM_TYPES,
        num_relations: int  = N_RELATIONS,
        num_heads: int      = 4,
        dropout: float      = 0.1,
    ) -> None:
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads    = num_heads
        self.head_dim     = out_dim // num_heads
        self.out_dim      = out_dim
        self.scale        = math.sqrt(self.head_dim)

        # Type-specific projections: one per node type.
        self.W_Q = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False)
                                  for _ in range(num_node_types)])
        self.W_K = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False)
                                  for _ in range(num_node_types)])
        self.W_V = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False)
                                  for _ in range(num_node_types)])
        self.W_O = nn.ModuleList([nn.Linear(out_dim, out_dim, bias=False)
                                  for _ in range(num_node_types)])

        # Relation-specific transforms: one per relation type.
        self.W_r = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
            for _ in range(num_relations)
        ])
        self.M_r = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
            for _ in range(num_relations)
        ])
        # Learnable relation priority scalar λ_r (one per relation per head).
        self.lambda_r = nn.ParameterList([
            nn.Parameter(torch.ones(num_heads))
            for _ in range(num_relations)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.W_r:
            nn.init.xavier_uniform_(p)
        for p in self.M_r:
            nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        node_type_ids: torch.Tensor,
        rel_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] node features.
            node_type_ids: [N] int64 node-type index per node.
            rel_matrix: [N, N] int64 relation-type index per directed pair.
                rel_matrix[i, j] = relation type from node j to node i.

        Returns:
            [N, out_dim] updated node features.
        """
        N   = x.size(0)
        H   = self.num_heads
        D   = self.head_dim

        # Type-specific Q, K, V for every node.
        Q = torch.zeros(N, H, D, device=x.device)
        K = torch.zeros(N, H, D, device=x.device)
        V = torch.zeros(N, H, D, device=x.device)
        for t in range(N_ROOM_TYPES):
            mask = (node_type_ids == t)
            if not mask.any():
                continue
            Q[mask] = self.W_Q[t](x[mask]).view(-1, H, D)
            K[mask] = self.W_K[t](x[mask]).view(-1, H, D)
            V[mask] = self.W_V[t](x[mask]).view(-1, H, D)

        # Relation-aware attention scores.
        # For each pair (i←j) with relation r:
        #   score[h,i,j] = Q[i,h] · (K[j,h] @ W_r[r,h]) * λ_r[h] / scale
        attn = torch.zeros(H, N, N, device=x.device)
        msg  = torch.zeros(H, N, N, D, device=x.device)

        for r in range(N_RELATIONS):
            pair_mask = (rel_matrix == r)           # [N, N] bool (i, j)
            if not pair_mask.any():
                continue
            src_idx, tgt_idx = pair_mask.nonzero(as_tuple=True)  # j, i

            # K_transformed: K[src] @ W_r[r]  per head  →  [M, H, D]
            K_src = K[src_idx]                      # [M, H, D]
            W     = self.W_r[r]                     # [H, D, D]
            K_t   = torch.einsum("mhd,hde->mhe", K_src, W)   # [M, H, D]

            # Score: Q[tgt] · K_t * λ_r  →  [M, H]
            Q_tgt = Q[tgt_idx]                      # [M, H, D]
            score = (Q_tgt * K_t).sum(-1) * self.lambda_r[r].unsqueeze(0) / self.scale

            # V_transformed: V[src] @ M_r[r]  →  [M, H, D]
            V_src = V[src_idx]                      # [M, H, D]
            M     = self.M_r[r]                     # [H, D, D]
            V_t   = torch.einsum("mhd,hde->mhe", V_src, M)   # [M, H, D]

            # Scatter into dense tensors.
            for m_idx, (j, i) in enumerate(zip(src_idx.tolist(), tgt_idx.tolist())):
                attn[:, i, j] = score[m_idx]
                msg[:, i, j]  = V_t[m_idx]

        # Softmax over source nodes for each target.
        attn = self.dropout(F.softmax(attn, dim=-1))   # [H, N, N]

        # Aggregate messages.
        out = torch.einsum("hij,hijd->ihd", attn, msg)   # [N, H, D]
        out = out.contiguous().view(N, H * D)

        # Type-specific output projection + residual + norm.
        proj = torch.zeros_like(out)
        for t in range(N_ROOM_TYPES):
            mask = (node_type_ids == t)
            if mask.any():
                proj[mask] = self.W_O[t](out[mask])

        skip = x if x.shape[-1] == self.out_dim else proj
        return self.norm(skip + proj)


class HGTEncoder(nn.Module):
    """Shared HGT encoder: room-type embeddings → contextual node representations.

    Each node starts as a learned room-type embedding.  The relation matrix is
    derived from the geometric context: if no geometric information is available
    all off-diagonal entries use the default 'room' relation.

    Args:
        embed_dim: Initial room embedding dimension.
        hidden_dim: Hidden dimension (must be divisible by num_heads).
        num_layers: Number of stacked HGTLayers.
        num_heads: Attention heads per layer.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        embed_dim:  int   = 64,
        hidden_dim: int   = 64,
        num_layers: int   = 3,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(N_ROOM_TYPES, embed_dim)

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.append(HGTLayer(in_dim, hidden_dim, N_ROOM_TYPES,
                                   N_RELATIONS, num_heads, dropout))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.out_dim = in_dim

    def forward(
        self,
        node_types: torch.Tensor,
        rel_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            node_types: [N] int64 room-type indices.
            rel_matrix: [N, N] int64 relation type per pair, or None to use the
                default 'room' relation for all pairs.

        Returns:
            [N, out_dim] contextual node representations.
        """
        N = node_types.size(0)
        if rel_matrix is None:
            rel_matrix = torch.full(
                (N, N), _DEFAULT_RELATION,
                dtype=torch.long, device=node_types.device,
            )
        x = self.embedding(node_types)
        for layer in self.layers:
            x = layer(x, node_types, rel_matrix)
        return x


# ── Edge MLP (shared between both decoders) ──────────────────────────────────

class EdgeMLP(nn.Module):
    """Predict output edge class from a concatenated pair of node representations.

    Args:
        node_dim: Node representation dimension.
        hidden_dim: MLP hidden width.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_EDGE_TYPES),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """Args: h_i, h_j [*, node_dim]. Returns [*, N_EDGE_TYPES] logits."""
        return self.net(torch.cat([h_i, h_j], dim=-1))


# ── One-shot decoder ─────────────────────────────────────────────────────────

class OneShotDecoder(nn.Module):
    """Predict all edges simultaneously from HGT node representations."""

    def __init__(self, node_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.edge_mlp = EdgeMLP(node_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Args: h [N, D]. Returns [N, N, N_EDGE_TYPES] logits."""
        N = h.size(0)
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        return self.edge_mlp(h_i, h_j)


# ── Autoregressive decoder ────────────────────────────────────────────────────

class AutoregressiveDecoder(nn.Module):
    """Add rooms in canonical order; predict edges for each new node.

    Teacher forcing at training time: existing graph uses ground-truth edges.
    The encoder re-runs at each step so the new node's representation reflects
    the current (growing) node set.

    Args:
        encoder: Shared HGTEncoder instance.
        node_dim: Encoder output dimension.
        hidden_dim: EdgeMLP hidden width.
    """

    def __init__(self, encoder: HGTEncoder, node_dim: int,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder  = encoder
        self.edge_mlp = EdgeMLP(node_dim, hidden_dim)

    def forward(
        self,
        node_types:  torch.Tensor,
        seed_size:   int,
        rel_matrix:  torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """
        Args:
            node_types: [N] int64 room-type indices in canonical order.
            seed_size: Number of seed nodes (edges not predicted).
            rel_matrix: [N, N] int64 full relation matrix, or None for default.

        Returns:
            logits_per_step: per-pair logit tensors [N_EDGE_TYPES].
            pairs: (new_node_idx, existing_node_idx) matching logits.
        """
        N = node_types.size(0)
        logits_per_step: list[torch.Tensor] = []
        pairs: list[tuple[int, int]] = []

        for new_idx in range(seed_size, N):
            cur_types = node_types[: new_idx + 1]
            cur_rel   = rel_matrix[: new_idx + 1, : new_idx + 1] \
                        if rel_matrix is not None else None
            h = self.encoder(cur_types, cur_rel)             # [new_idx+1, D]

            h_new     = h[new_idx]                           # [D]
            h_exist   = h[:new_idx]                          # [new_idx, D]
            h_new_exp = h_new.unsqueeze(0).expand(new_idx, -1)
            logits    = self.edge_mlp(h_new_exp, h_exist)    # [new_idx, 5]

            for k in range(new_idx):
                logits_per_step.append(logits[k])
                pairs.append((new_idx, k))

        return logits_per_step, pairs


# ── Full model ────────────────────────────────────────────────────────────────

class RoomGraphGAT(nn.Module):
    """Heterogeneous Graph Transformer for room-graph generation.

    Shared HGT encoder + one-shot and autoregressive decoders trained jointly.

    The name 'RoomGraphGAT' is kept for API compatibility; internally this is
    an HGT encoder (Hu et al., 2020) rather than a standard GAT.

    Reference:
        Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020).
        Heterogeneous Graph Transformer. WWW 2020.
        https://arxiv.org/abs/2003.01332
        https://github.com/acbull/pyHGT

    Args:
        embed_dim: Room type embedding dimension.
        hidden_dim: HGT hidden dimension (divisible by num_heads).
        num_layers: Number of HGT layers.
        num_heads: Attention heads per layer.
        mlp_hidden: EdgeMLP hidden width.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        embed_dim:  int   = 64,
        hidden_dim: int   = 64,
        num_layers: int   = 3,
        num_heads:  int   = 4,
        mlp_hidden: int   = 128,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = HGTEncoder(embed_dim, hidden_dim, num_layers,
                                  num_heads, dropout)
        node_dim = self.encoder.out_dim
        self.oneshot       = OneShotDecoder(node_dim, mlp_hidden)
        self.autoregressive = AutoregressiveDecoder(self.encoder, node_dim, mlp_hidden)

    def forward_oneshot(
        self,
        node_types: torch.Tensor,
        rel_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One forward pass → [N, N, N_EDGE_TYPES] logits."""
        h = self.encoder(node_types, rel_matrix)
        return self.oneshot(h)

    def forward_autoregressive(
        self,
        node_types:  torch.Tensor,
        seed_size:   int,
        rel_matrix:  torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """Sequential forward → list of per-pair logits + pair indices."""
        return self.autoregressive(node_types, seed_size, rel_matrix)

    # ── Inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_oneshot(
        self,
        room_program: list[str],
        rel_matrix:   torch.Tensor | None = None,
        threshold:    float = 0.5,
    ) -> torch.Tensor:
        """Generate an edge-label matrix using the one-shot decoder.

        Args:
            room_program: Room type strings, e.g. ["living","bedroom","kitchen"].
            rel_matrix: Optional [N, N] int64 geometric relation matrix.
            threshold: Min softmax probability to assign a non-zero edge class.

        Returns:
            [N, N] int64 edge-label matrix (0=no_edge, 1-4=edge type).
        """
        node_types  = _program_to_tensor(room_program)
        logits      = self.forward_oneshot(node_types, rel_matrix)
        probs       = F.softmax(logits, dim=-1)
        edge_labels = probs.argmax(dim=-1).clone()
        max_prob    = probs.max(dim=-1).values
        edge_labels[max_prob < threshold] = 0
        # Symmetrise: take the higher-confidence direction.
        N = edge_labels.size(0)
        conf = probs.max(dim=-1).values
        for i in range(N):
            for j in range(i + 1, N):
                if conf[i, j] >= conf[j, i]:
                    edge_labels[j, i] = edge_labels[i, j]
                else:
                    edge_labels[i, j] = edge_labels[j, i]
        return edge_labels

    @torch.no_grad()
    def generate_autoregressive(
        self,
        room_program: list[str],
        rel_matrix:   torch.Tensor | None = None,
        threshold:    float = 0.5,
    ) -> torch.Tensor:
        """Generate an edge-label matrix using the autoregressive decoder.

        Args:
            room_program: Room type strings in canonical order.
            rel_matrix: Optional [N, N] int64 geometric relation matrix.
            threshold: Min softmax probability to assign a non-zero edge class.

        Returns:
            [N, N] int64 edge-label matrix.
        """
        node_types = _program_to_tensor(room_program)
        N          = node_types.size(0)
        seed_size  = _seed_size(room_program)

        edge_labels = torch.zeros(N, N, dtype=torch.long)
        _fill_seed_edges(edge_labels, room_program)

        logits_list, pairs = self.forward_autoregressive(
            node_types, seed_size, rel_matrix
        )
        for logit, (i, j) in zip(logits_list, pairs):
            probs = F.softmax(logit, dim=-1)
            cls   = int(probs.argmax().item())
            if probs[cls] >= threshold:
                edge_labels[i, j] = cls
                edge_labels[j, i] = cls

        return edge_labels


# ── Seed and program utilities ────────────────────────────────────────────────

def build_seed_graph(room_program: list[str]) -> torch.Tensor:
    """Return the [N, N] seed edge-label matrix for a room program."""
    N = len(room_program)
    edge_labels = torch.zeros(N, N, dtype=torch.long)
    _fill_seed_edges(edge_labels, room_program)
    return edge_labels


def seed_mask(room_program: list[str]) -> torch.Tensor:
    """Return [N, N] bool mask that is True for fixed seed edges."""
    N    = len(room_program)
    mask = torch.zeros(N, N, dtype=torch.bool)
    idx  = {r: i for i, r in enumerate(room_program)}
    for ta, tb, _ in SEED_EDGES:
        if ta in idx and tb in idx:
            i, j = idx[ta], idx[tb]
            mask[i, j] = True
            mask[j, i] = True
    return mask


def sort_by_canonical_order(room_program: list[str]) -> list[str]:
    """Sort a room program into canonical order for the autoregressive decoder."""
    order = {r: i for i, r in enumerate(CANONICAL_ORDER)}
    return sorted(room_program, key=lambda r: order.get(r, len(CANONICAL_ORDER)))


def default_relation_matrix(room_program: list[str]) -> torch.Tensor:
    """Build a default [N, N] relation matrix using the 'room' relation for all pairs.

    Replace individual entries with specific relation types when geometric
    context is available, e.g.:
        rel[i, j] = RELATION_TO_INT["door"]  # when a door separates rooms i and j

    Args:
        room_program: List of room type strings.

    Returns:
        [N, N] int64 tensor filled with RELATION_TO_INT["room"].
    """
    N = len(room_program)
    return torch.full((N, N), _DEFAULT_RELATION, dtype=torch.long)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _program_to_tensor(room_program: list[str]) -> torch.Tensor:
    return torch.tensor([ROOM_TO_INT[r] for r in room_program], dtype=torch.long)


def _seed_size(room_program: list[str]) -> int:
    seed_types = {"living", "kitchen", "bedroom", "bathroom"}
    return sum(1 for r in room_program if r in seed_types)


def _fill_seed_edges(edge_labels: torch.Tensor, room_program: list[str]) -> None:
    idx = {r: i for i, r in enumerate(room_program)}
    for ta, tb, cls in SEED_EDGES:
        if ta in idx and tb in idx:
            i, j = idx[ta], idx[tb]
            edge_labels[i, j] = cls
            edge_labels[j, i] = cls
