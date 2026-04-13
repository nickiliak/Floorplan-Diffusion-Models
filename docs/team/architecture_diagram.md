# HouseDiffusion — Transformer Architecture Diagram

This is an **encoder-only** Transformer. There is no decoder — the diffusion loop
(1000 iterations of the same encoder) handles generation.

---

## High-Level Overview

```
  Noisy corners x_t         Timestep t          Condition (room types + indices)
     [B, 2, 100]               [B]                    [B, 100, 89]
         │                      │                          │
         ▼                      │                          │
    permute to                  │                          │
    [B, 100, 2]                 │                          │
         │                      │                          │
         ▼                      │                          │
  ┌─────────────┐               │                          │
  │expand_points│ (9× interp)   │                          │
  │ 100→900 pts │               │                          │
  └──────┬──────┘               │                          │
         │ [B, 900, 18]         │                          │
         ▼                      ▼                          ▼
  ┌────────────┐    ┌──────────────────┐         ┌──────────────┐
  │  Linear    │    │ Sinusoidal Embed │         │    Linear    │
  │  18 → 512  │    │ t → 128 → 512   │         │  89 → 512   │
  └─────┬──────┘    └────────┬─────────┘         └──────┬───────┘
        │                    │                          │
        │ [B,900,512]        │ [B,1,512]                │ [B,900,512]
        │                    │ (broadcast)              │
        └────────────────────┼──────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Element-wise  │
                    │   Addition (+)  │
                    │ input+cond+time │
                    └────────┬────────┘
                             │ [B, 900, 512]
                             ▼
                    ╔═════════════════╗
                    ║  EncoderLayer 1 ║──┐
                    ╚════════╤════════╝  │
                             ▼           │  × 4 layers
                    ╔═════════════════╗  │  (each identical structure,
                    ║  EncoderLayer 2 ║  │   separate parameters)
                    ╚════════╤════════╝  │
                             ▼           │
                    ╔═════════════════╗  │
                    ║  EncoderLayer 3 ║  │
                    ╚════════╤════════╝  │
                             ▼           │
                    ╔═════════════════╗  │
                    ║  EncoderLayer 4 ║──┘
                    ╚════════╤════════╝
                             │ [B, 900, 512]
                             ▼
                ┌────────────┴────────────┐
                ▼                         ▼
    ┌───────────────────┐     ┌───────────────────────┐
    │  Continuous Head   │     │   Discrete Head       │
    │  (always active)   │     │  (when analog_bit=F)  │
    │                    │     │                        │
    │  Linear 512→512    │     │  Binary encoding       │
    │  ReLU              │     │  + 2 EncoderLayers    │
    │  Linear 512→256    │     │    (1-head attention)  │
    │  Linear 256→2      │     │  + Linear → 16        │
    └────────┬───────────┘     └────────────┬───────────┘
             │ [B, 2, 100]                  │ [B, 16, 100]
             ▼                              ▼
          ε_pred (noise)            binary coordinates
         or x_0 prediction         (last 32 timesteps)
```

---

## EncoderLayer Detail (× 4)

Each layer has **three separate attention mechanisms** with different masks,
plus a feed-forward network. All use **pre-normalization** (norm before, not after).

```
    Input h  [B, 900, 512]
        │
        ▼
  ┌──────────────┐
  │ InstanceNorm │
  └──────┬───────┘
         │ h_normed
         ├──────────────────────┬──────────────────────┐
         ▼                      ▼                      ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Door Attn   │     │  Self Attn   │     │  Gen Attn    │
  │  (4 heads)   │     │  (4 heads)   │     │  (4 heads)   │
  │              │     │              │     │              │
  │ mask: rooms  │     │ mask: same   │     │ mask: pad    │
  │ sharing a    │     │ room corners │     │ tokens       │
  │ door/edge    │     │ only         │     │ blocked      │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │ Dropout             │ Dropout             │ Dropout
         └──────────────────────┼──────────────────────┘
                                │
                          Sum of three
                                │
                                ▼
                    ┌───────────────────┐
                    │  Residual Add (+) │  h = h + door + self + gen
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │ InstanceNorm │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ FeedForward  │
                    │ 512→1024→512 │
                    │ (ReLU, drop) │
                    └──────┬───────┘
                           │ Dropout
                           ▼
                    ┌───────────────────┐
                    │  Residual Add (+) │  h = h + ff(norm(h))
                    └─────────┬─────────┘
                              │
                              ▼
                    Output h  [B, 900, 512]
```

---

## MultiHeadAttention Detail (4 heads, d_k = 128)

```
    Q, K, V  [B, 900, 512]           Mask [B, 100, 100]
        │                                    │
        ▼                                    │
  ┌──────────────────────┐                   │
  │ Linear Q: 512 → 512  │                   │
  │ Linear K: 512 → 512  │                   │
  │ Linear V: 512 → 512  │                   │
  └──────────┬───────────┘                   │
             │                               │
             ▼                               │
  Reshape to [B, 900, 4, 128]               │
  Transpose  [B, 4, 900, 128]               │
             │                               │
             ▼                               ▼
  ┌────────────────────────────────────────────┐
  │  scores = (Q · K^T) / √128                │
  │  scores.masked_fill(mask == 1, −1e9)       │
  │  weights = softmax(scores)                 │
  │  output = weights · V                      │
  └──────────────────┬─────────────────────────┘
                     │ [B, 4, 900, 128]
                     ▼
           Transpose + Reshape → [B, 900, 512]
                     │
                     ▼
              Linear 512 → 512 (output projection)
                     │
                     ▼
               [B, 900, 512]
```

---

## Point Expansion Detail (expand_points)

Densifies the 100 corner tokens into 900 by interpolating 8 midpoints
between each pair of adjacent corners. Gradients are **detached**.

```
    Corner p1 ●───────────────────────● Corner p5 (next corner in polygon)

    After expansion (recursive binary midpoint averaging):

    p1    p1.5   p2    p2.5   p3    p3.5   p4    p4.5   p5
    ●──────●──────●──────●──────●──────●──────●──────●──────●
    0     1/8    1/4   3/8    1/2   5/8    3/4   7/8    1

    Interpolation tree:
                        p3 = avg(p1, p5)
                       /                \
              p2 = avg(p1, p3)    p4 = avg(p3, p5)
             /        \          /        \
    p1.5=avg(p1,p2) p2.5=avg(p2,p3) p3.5=avg(p3,p4) p4.5=avg(p4,p5)

    Input:  [B, 100, 2]  (100 corners × 2 coords)
    Output: [B, 100, 18] (100 corners × 9 points × 2 coords)
            reshaped as [B, 900, 2] for transformer input
```

---

## The Three Attention Masks

All masks are `[B, 100, 100]` binary matrices. **1 = block attention, 0 = allow**.

```
    Example: 3-room floorplan (rooms A, B, C; door between A↔B)

    self_mask (within-room):          door_mask (cross-room):
    ┌─────┬─────┬─────┐              ┌─────┬─────┬─────┐
    │  0  │  1  │  1  │  A           │  1  │  0  │  1  │  A
    ├─────┼─────┼─────┤              ├─────┼─────┼─────┤
    │  1  │  0  │  1  │  B           │  0  │  1  │  1  │  B
    ├─────┼─────┼─────┤              ├─────┼─────┼─────┤
    │  1  │  1  │  0  │  C           │  1  │  1  │  1  │  C
    └─────┴─────┴─────┘              └─────┴─────┴─────┘
    0 = corners attend freely        0 = rooms A,B share door
    within their own room            so their corners can attend

    gen_mask (padding):
    ┌─────┬─────┬─────┬─────┐
    │  0  │  0  │  0  │  1  │  ← pad token blocked
    ├─────┼─────┼─────┼─────┤
    │  0  │  0  │  0  │  1  │
    ├─────┼─────┼─────┼─────┤
    │  0  │  0  │  0  │  1  │
    ├─────┼─────┼─────┼─────┤
    │  1  │  1  │  1  │  1  │  ← pad row: blocks all attention
    └─────┴─────┴─────┴─────┘
```

---

## Discrete Output Head Detail

Active only when `analog_bit=False`. Refines the continuous prediction into
sharp 8-bit coordinates during the last 32 diffusion timesteps.

```
    Continuous prediction (ε_pred)     Expanded input x
              │                              │
              ▼                              ▼
    ┌──────────────────────────────────────────────┐
    │  x̂₀ ≈ x_t · α_t − ε_pred · α_ε             │
    │  Quantize x̂₀ → 8-bit per axis              │
    │  Convert to 16 binary values (8×x + 8×y)    │
    │  Map {0,1} → {−1,1}                         │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
            cat(x̂₀, binary_repr, cond_emb)
                [B, 900, 18+144+512]
                       │
                       ▼
              Linear → 512, ReLU
                       │
                       ▼
              EncoderLayer (1 head)   ← uses same 3 masks
                       │
                       ▼
              EncoderLayer (1 head)   ← uses same 3 masks
                       │
                       ▼
              Linear 512 → 16
                       │
                       ▼
                [B, 16, 100]
            (16-bit binary output)
```

---

## Summary: Why Encoder-Only?

| Architecture | Use case | Output structure |
|---|---|---|
| Encoder-Decoder | Input → different-structure output (translation) | Autoregressive, variable-length |
| Decoder-only | Autoregressive generation (GPT) | Token-by-token, causal mask |
| **Encoder-only** | **Input → same-structure output (denoising)** | **Parallel, fixed-length** |

HouseDiffusion maps `[B, 2, 100]` → `[B, 2, 100]` — same shape in, same shape out.
The 1000-step diffusion loop replaces the decoder's role as a generator.
