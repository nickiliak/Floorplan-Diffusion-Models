# Room Graph Generation with Heterogeneous Graph Transformer (HGT)

**Author:** Konstandinos  
**Part of:** DTU Special Course — Recreating HouseDiffusion with the ResPlan dataset

---

## 1. The Problem: Diffusion Models Don't Understand Floor Plans

The core HouseDiffusion model generates floorplans by running a diffusion process over
room bounding boxes. It works well geometrically, but it has no understanding of what a
*valid* floor plan looks like at a structural level.

In practice, floor plans follow implicit rules that architects know by heart:

- The living room connects to everything — it is the hub of the house.
- The front door always opens into the living room.
- Bathrooms never connect directly to the front door.
- Bedrooms connect via doors, not open archways.
- Kitchens and living rooms often share an open adjacency.

When the diffusion model samples blindly, it has no way to enforce these constraints.
The result is geometrically plausible boxes that are sometimes architecturally nonsensical —
bedrooms floating with no access, front doors opening into bathrooms, etc.

**The idea:** train a separate graph generation model that learns these structural rules
from data, and use it to condition the diffusion process with a *meaningful room graph*
rather than random sampling.

---

## 2. Why a Graph?

A floor plan is naturally a graph:

- **Nodes** = rooms (living room, kitchen, bedroom, bathroom, balcony, front door)
- **Edges** = how rooms are connected (open adjacency, via a door, via a window, direct opening)

The ResPlan dataset already provides these graphs as NetworkX objects with typed edges.
We extracted 16,227 such graphs from the full dataset (3,000 from the simplified subset).

Edge class distribution in the dataset:

| Edge type   | Count  | Inv-freq weight |
|-------------|--------|-----------------|
| via_door    | 68,381 | 1.00 (reference)|
| adjacency   | 56,532 | 1.35            |
| direct      | 16,964 | 2.70            |
| via_window  |  2,877 | 122.3           |
| no_edge     |   —    | 0.08 (suppressed)|

The heavy imbalance (especially `via_window`) is handled with inverse-frequency class weights
in the cross-entropy loss.

---

## 3. Why HGT Instead of a Standard GNN?

The first approach was a standard Graph Attention Network (GAT). GAT computes attention
scores between every pair of nodes, but it treats all nodes and edges identically —
a bedroom and a kitchen are just "nodes", a door and a window are just "edges".

The key insight that motivates switching to a **Heterogeneous Graph Transformer** is that
the ResPlan graph already contains *typed* geometric entities:

- Nodes have a **room type** (6 types: living, kitchen, bedroom, bathroom, balcony, front_door)
- Edges have a **relation type** based on the geometric entity that connects them:
  `room | door | wall | window | boundary`

A bedroom next to a living room connected *via a door* is a fundamentally different
relationship than a kitchen connected to a living room *via an open archway*. A homogeneous
model cannot distinguish these; HGT can.

**Reference:**  
Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020).  
*Heterogeneous Graph Transformer.* The Web Conference (WWW 2020).  
[https://arxiv.org/abs/2003.01332](https://arxiv.org/abs/2003.01332)  
Original implementation: [https://github.com/acbull/pyHGT](https://github.com/acbull/pyHGT)

---

## 4. How HGT Works

In a standard transformer, attention between two nodes i and j is:

```
score(i, j) = Q(hᵢ) · K(hⱼ) / √d
```

Every pair uses the same Q and K projection matrices. HGT parameterises these by
**node type** (τ) and **edge relation type** (r):

```
score(i←j via r) = Q_τᵢ(hᵢ) · (K_τⱼ(hⱼ) @ W_r) · λ_r / √dₖ
message(j via r) = V_τⱼ(hⱼ) @ M_r
```

In plain English:

- **Q_τ, K_τ, V_τ, O_τ** — separate Query/Key/Value/Output projection matrices *per room type*.
  A bedroom's query vector is computed differently from a kitchen's query vector.

- **W_r, M_r** — separate key-transform and value-transform matrices *per relation type*.
  An attention score computed over a `door` edge uses a different projection than one
  computed over a `wall` edge.

- **λ_r** — a learnable scalar (one per relation per attention head) that lets the model
  up- or down-weight entire relation types during training.

Each node aggregates messages from all its neighbours, weighted by these type-aware attention
scores, then passes through a type-specific output projection O_τ and a residual + LayerNorm.

The model stacks 3 such HGT layers. The node embeddings start as learned room-type embeddings
(a 64-dimensional lookup table) and evolve into contextual representations that reflect
*what the node is* and *how it connects to its neighbours*.

---

## 5. Architecture Overview

```
Room program  →  [Type embedding]  →  HGT Layer × 3  →  node representations h
                                                               ↓            ↓
                                                        One-shot       Autoregressive
                                                        Decoder        Decoder
                                                               ↓            ↓
                                                        [N×N edge     [edge logits
                                                         logits]       per step]
```

**Total parameters:** ~394,000 (with default hidden_dim=64, num_layers=3, num_heads=4)

### 5.1 The Shared Encoder

`HGTEncoder` in `src/input_generation/GAT.py`

Takes a room program (a list of room type strings) and an optional `rel_matrix`
(an N×N integer tensor where entry [i,j] is the relation type from room j to room i).

If no `rel_matrix` is provided, all pairs default to the `room` relation — a safe prior
when no geometric context is available. When the full ResPlan geometry is available you
can populate it with specific relation types (e.g. `RELATION_TO_INT["door"]`).

### 5.2 One-shot Decoder

`OneShotDecoder` in `src/input_generation/GAT.py`

Runs the encoder once on the full room set. For every pair (i, j) concatenates their
representations `[hᵢ, hⱼ]` and passes it through a 3-layer MLP (`EdgeMLP`) that outputs
5-class logits. Predicts all N×N edges simultaneously in a single forward pass.

- **Advantage:** fast, O(N²) MLP evaluations after one encoder call.
- **Disadvantage:** no sequential dependency — the prediction for edge (i,j) does not
  "know" what was predicted for edge (i,k).

### 5.3 Autoregressive Decoder

`AutoregressiveDecoder` in `src/input_generation/GAT.py`

Adds rooms in canonical order: `living → kitchen → bedroom → bathroom → balcony → front_door`.
At each step, the encoder re-runs on the *current partial graph* (seed nodes + all nodes
added so far). The new node's representation is then used to predict edges to all previously
placed nodes.

- **Advantage:** each edge prediction is informed by the growing graph context.
- **Disadvantage:** slower — the encoder runs once per new node added.

At training time, **teacher forcing** is used: the ground-truth edges from the dataset
are used as the graph state at each step, rather than the model's own predictions. This
stabilises training.

---

## 6. The Seed Graph

Rather than generating from an empty graph, we inject a **fixed minimum seed**:

```
{living, kitchen, bedroom, bathroom}
```

with hardcoded edges based on common architectural priors:

| Pair                       | Edge type  |
|----------------------------|------------|
| living ↔ kitchen           | adjacency  |
| living ↔ bedroom           | adjacency  |
| living ↔ bathroom          | via_door   |
| living ↔ front_door        | direct     |

Neither decoder predicts these seed edges — they are injected as a fixed prefix and masked
out of the loss. The model only learns to predict edges for nodes *beyond* the seed (balcony,
front_door when not already in seed, additional bedrooms, etc.) and between non-seed pairs.

A hard cap of **MAX_ROOMS = 15** prevents unbounded graph growth during generation.

---

## 7. Joint Training

Both decoders are trained **simultaneously** on the same batch, sharing the encoder weights.

The total loss per sample is:

```
L = L_oneshot + L_autoregressive
```

where both terms are cross-entropy over the 5 edge classes, with inverse-frequency class
weights applied to handle the `via_window` imbalance.

**Why train jointly?**

The encoder is shared. If we trained only one decoder, the encoder would be optimised
purely for that decoder's objective. By training both decoders simultaneously:

1. The encoder must learn representations that are useful for *both* prediction strategies —
   one that needs to predict all edges at once and one that predicts them sequentially.
   This acts as a regulariser and tends to produce richer node representations.

2. After training we can compare the two decoders fairly — they used the same encoder
   throughout, so any performance difference is attributable to the decoder design, not
   to differences in encoder quality.

3. We get two models for the price of one training run.

The two decoders have *separate* EdgeMLP weights — only the encoder is shared. Gradients
from both losses flow back through the encoder at each step.

---

## 8. Training Configuration

All hyperparameters live in `configs/gat_resplan.yaml`.

| Parameter      | Value  | Notes                              |
|----------------|--------|------------------------------------|
| embed_dim      | 64     | Room type embedding dimension      |
| hidden_dim     | 64     | HGT hidden dim (= num_heads × 16)  |
| num_layers     | 3      | Stacked HGT layers                 |
| num_heads      | 4      | Attention heads per layer          |
| mlp_hidden     | 128    | EdgeMLP hidden width               |
| dropout        | 0.1    | Attention dropout                  |
| epochs         | 250    | Full passes over the dataset       |
| save_interval  | 50     | Periodic checkpoint every N epochs |
| lr             | 3e-4   | AdamW learning rate                |
| weight_decay   | 1e-4   | AdamW weight decay                 |
| val_fraction   | 0.1    | 10% held-out validation split      |

Scheduler: CosineAnnealingLR over epochs.  
Gradient clipping: max norm 1.0.

---

## 9. Running It

**Training + evaluation (batch job on DTU HPC):**
```bash
# Submit from the repo root
bsub < scripts/bash/run_gat.sh
```

**Training only (local/interactive):**
```bash
uv run python scripts/train_gat.py --config ... # all args, or:
uv run python scripts/train_gat.py \
    --pickle_path data/raw/ResPlan.pkl \
    --output_dir  models/gat \
    --epochs 50
```

**Evaluation only (after training):**
```bash
uv run python scripts/evaluate_gat.py \
    --config     configs/gat_resplan.yaml \
    --checkpoint models/gat/gat_best.pt \
    --output_dir outputs/gat
```

---

## 10. Outputs

After a full run you will find:

```
models/gat/
  gat_best.pt          ← best validation loss checkpoint
  gat_last.pt          ← final epoch checkpoint

outputs/gat/
  stats/
    graph_stats_comparison.png    ← 3×3 grid (Training / One-shot / Autoregressive)
                                     columns: node degree, clustering coefficient,
                                              eigenvector centrality
  graphs/
    training_grid.png             ← 12 ground-truth graphs from the dataset
    oneshot_grid.png              ← 12 one-shot generated graphs
    autoregressive_grid.png       ← 12 autoregressive generated graphs
  metrics.json                    ← numeric summary (validity rate, mean degree,
                                     clustering coefficient, edge type distribution)
```

**Key metric — validity rate:** fraction of generated graphs that are (a) connected and
(b) contain all four seed room types. A fully random graph scores ~0%; a well-trained model
should reach >80%.

---

## 11. Relevant Files

| File | Description |
|------|-------------|
| `src/input_generation/GAT.py` | HGT encoder, both decoders, all utilities |
| `src/input_generation/__init__.py` | Public exports |
| `scripts/train_gat.py` | Training loop, dataset class, loss functions |
| `scripts/evaluate_gat.py` | Evaluation: statistics, visualisations, metrics.json |
| `configs/gat_resplan.yaml` | All hyperparameters |
| `scripts/bash/run_gat.sh` | LSF batch job (train → evaluate) |
| `notebooks/05_graph_inspection.ipynb` | Dataset exploration and edge type analysis |
