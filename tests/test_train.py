"""
test_train.py — Smoke test: load ResPlan data through resplandataset.py and run
a few training steps through the HouseDiffusion model to verify end-to-end
compatibility.

Adapted from external/house_diffusion/scripts/image_train.py.
Stripped of MPI/distributed dependencies so it runs on a single CPU/GPU.

Usage:
    python tests/test_train.py                          # synthetic data (no pickle needed)
    python tests/test_train.py --pkl data/raw/ResPlan.pkl  # real ResPlan data
"""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch as th
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Path setup: make house_diffusion importable from external/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "house_diffusion"))
sys.path.insert(0, PROJECT_ROOT)

# HouseDiffusion imports (from external/house_diffusion/house_diffusion/)
from external.house_diffusion.house_diffusion.transformer import TransformerModel
from external.house_diffusion.house_diffusion.respace import SpacedDiffusion, space_timesteps
from external.house_diffusion.house_diffusion import gaussian_diffusion as gd
from external.house_diffusion.house_diffusion.resample import UniformSampler

# Our ResPlan dataset
from src.helpers.resplandataset import (
    ResplanDataset,
    load_resplan_data,
    build_house_tensor,
    build_graph_triples,
    build_attention_masks,
    extract_rooms_from_plan,
    extract_vertices_from_polygon,
    DEFAULT_MAX_NUM_POINTS,
    NUM_ROOM_TYPE_CLASSES,
    MAX_CORNER_INDEX,
    MAX_ROOM_INDEX,
    ROW_WIDTH,
)


def enable_cpu_compatibility_patch() -> None:
    """Patch HouseDiffusion's internal `.cuda()` assumption for CPU-only smoke tests."""
    if th.cuda.is_available() or getattr(th.Tensor, "_floorplan_cpu_safe_cuda", False):
        return

    def _cpu_safe_cuda(self, device=None, non_blocking=False, memory_format=th.preserve_format):
        return self.to("cpu")

    th.Tensor.cuda = _cpu_safe_cuda
    th.Tensor._floorplan_cpu_safe_cuda = True


# ---------------------------------------------------------------------------
# Model + Diffusion factory (simplified from script_util.py, no argparse)
# ---------------------------------------------------------------------------

def create_model(
    analog_bit: bool = False,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    num_channels: int = 512,
) -> TransformerModel:
    """Create a TransformerModel configured for ResPlan/rplan-style data."""
    num_coords = 16 if analog_bit else 2
    input_channels = num_coords + (2 * 8 if not analog_bit else 0)
    condition_channels = 121 # room_types(25) + corner_indices(32) + room_indices(32) + src_key_padding_mask(1) + connections(2) + door_mask(1) + self_mask(1) + gen_mask(1)
    out_channels = num_coords

    model = TransformerModel(
        in_channels=input_channels,
        condition_channels=condition_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        dataset="rplan",  # reuse rplan arch (same tensor layout)
        use_checkpoint=False,
        use_unet=False,
        analog_bit=analog_bit,
    )
    return model


def create_diffusion(
    diffusion_steps: int = 100,
    noise_schedule: str = "cosine",
) -> SpacedDiffusion:
    """Create a SpacedDiffusion for testing (fewer steps than production)."""
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, [diffusion_steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


# ---------------------------------------------------------------------------
# Synthetic data generator (when pickle is unavailable)
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    batch_size: int,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    analog_bit: bool = False,
):
    """Create a single synthetic batch mimicking ResplanDataset.__getitem__ output.

    Builds 2-3 random rectangular rooms per sample so the tensor structure
    is realistic (valid one-hot encodings, proper masks, etc).
    """
    from shapely.geometry import box as shapely_box
    import random

    all_arrs = []
    all_conds = []

    for _ in range(batch_size):
        # Generate 2-4 random rooms
        n_rooms = random.randint(2, 4)
        rooms = []
        x_offset = 0.0
        for i in range(n_rooms):
            w = random.uniform(5, 15)
            h = random.uniform(5, 15)
            poly = shapely_box(x_offset, 0, x_offset + w, h)
            type_id = [1, 2, 3, 4][i % 4]  # living, bedroom, kitchen, bathroom
            type_name = ["living", "bedroom", "kitchen", "bathroom"][i % 4]
            rooms.append((poly, type_id, type_name))
            x_offset += w + random.uniform(0, 2)  # small gap or touching

        house_tensor, corner_bounds = build_house_tensor(rooms, max_num_points)
        graph_triples = build_graph_triples(rooms)
        total_corners = sum(len(extract_vertices_from_polygon(r[0])) for r in rooms)
        door_mask, self_mask, gen_mask = build_attention_masks(
            corner_bounds, graph_triples, max_num_points, total_corners
        )

        # Mimic __getitem__
        arr = house_tensor[:, :2].copy()
        arr = np.transpose(arr, [1, 0])  # [2, max_num_points]

        # Pad graph to 200
        if len(graph_triples) < 200:
            graph = np.concatenate(
                (graph_triples, np.zeros((200 - len(graph_triples), 3), dtype=np.float32))
            )
        else:
            graph = graph_triples[:200]

        _ci = 2 + NUM_ROOM_TYPE_CLASSES
        _ri = _ci + MAX_CORNER_INDEX
        _pm = _ri + MAX_ROOM_INDEX
        _cn = _pm + 1

        cond = {
            "door_mask": door_mask,
            "self_mask": self_mask,
            "gen_mask": gen_mask,
            "room_types": house_tensor[:, 2 : _ci],
            "corner_indices": house_tensor[:, _ci : _ri],
            "room_indices": house_tensor[:, _ri : _pm],
            "src_key_padding_mask": 1 - house_tensor[:, _pm],
            "connections": house_tensor[:, _cn : _cn + 2],
            "graph": graph,
        }

        all_arrs.append(arr.astype(np.float64))
        all_conds.append(cond)

    # Stack into batched tensors
    batch_arr = th.tensor(np.stack(all_arrs), dtype=th.float32)
    batch_cond = {}
    for key in all_conds[0]:
        stacked = np.stack([c[key] for c in all_conds])
        batch_cond[key] = th.tensor(stacked, dtype=th.float32)

    return batch_arr, batch_cond


def synthetic_data_generator(batch_size, max_num_points, analog_bit):
    """Infinite generator of synthetic batches."""
    while True:
        yield make_synthetic_batch(batch_size, max_num_points, analog_bit)


# ---------------------------------------------------------------------------
# Training loop (simplified — no DDP, no EMA, no checkpointing)
# ---------------------------------------------------------------------------

def run_training_test(
    data_iter,
    model: TransformerModel,
    diffusion: SpacedDiffusion,
    num_steps: int = 5,
    lr: float = 1e-4,
    analog_bit: bool = False,
    device: str = "cpu",
):
    """Run a few training steps and verify the loss is finite."""
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    schedule_sampler = UniformSampler(diffusion)

    print(f"\n{'='*60}")
    print(f"Running {num_steps} training steps on {device}")
    print(f"{'='*60}")

    losses = []
    for step in range(num_steps):
        t0 = time.time()

        batch, cond = next(data_iter)
        batch = batch.to(device)
        cond = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in cond.items()}

        # Sample timesteps
        t, weights = schedule_sampler.sample(batch.shape[0], device)

        # Compute loss
        optimizer.zero_grad()
        loss_dict = diffusion.training_losses(
            model, batch, t, model_kwargs=cond, analog_bit=analog_bit
        )
        loss = (loss_dict["loss"] * weights).mean()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        elapsed = time.time() - t0

        print(f"  Step {step+1}/{num_steps}  loss={loss_val:.6f}  time={elapsed:.2f}s")

        # Sanity checks
        assert np.isfinite(loss_val), f"Loss is not finite at step {step}: {loss_val}"
        assert loss_val >= 0, f"Loss is negative at step {step}: {loss_val}"

    print(f"\nAll {num_steps} steps completed successfully!")
    print(f"  Loss range: [{min(losses):.6f}, {max(losses):.6f}]")
    print(f"  Mean loss:  {np.mean(losses):.6f}")

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_pkl_path(pkl_arg: str | None) -> str | None:
    """Resolve a pickle path from either the current working directory or project root."""
    if not pkl_arg:
        return None

    pkl_path = Path(pkl_arg)
    if pkl_path.exists():
        return str(pkl_path.resolve())

    project_relative = Path(PROJECT_ROOT) / pkl_arg
    if project_relative.exists():
        return str(project_relative.resolve())

    return str(pkl_path)



def resolve_device(device_arg: str | None) -> str:
    """Choose a safe torch device, falling back to CPU if CUDA is unavailable."""
    if device_arg is None:
        return "cuda" if th.cuda.is_available() else "cpu"

    requested = device_arg.lower()
    if requested.startswith("cuda") and not th.cuda.is_available():
        print("  WARNING: CUDA was requested, but this PyTorch build has no CUDA support.")
        print("  Falling back to CPU instead.")
        return "cpu"

    return requested



def main():
    parser = argparse.ArgumentParser(description="Smoke test: ResPlan → HouseDiffusion training")
    parser.add_argument("--pkl", type=str, default=None,
                        help="Path to ResPlan.pkl. If omitted, uses synthetic data.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Number of training steps to run")
    parser.add_argument("--target_set", type=int, default=8)
    parser.add_argument("--max_num_points", type=int, default=DEFAULT_MAX_NUM_POINTS)
    parser.add_argument("--diffusion_steps", type=int, default=100,
                        help="Fewer steps for faster testing (production uses 1000)")
    parser.add_argument("--num_channels", type=int, default=128,
                        help="Model hidden dim. 128 for fast testing, 512 for production")
    parser.add_argument("--analog_bit", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None,
                        help="'cpu' or 'cuda'. Auto-detects if omitted.")
    args = parser.parse_args()

    resolved_pkl = resolve_pkl_path(args.pkl)
    device = resolve_device(args.device)
    enable_cpu_compatibility_patch()

    # --- Create model ---
    print("\n--- Creating model ---")
    model = create_model(
        analog_bit=args.analog_bit,
        max_num_points=args.max_num_points,
        num_channels=args.num_channels,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,}")

    # --- Create diffusion ---
    print("\n--- Creating diffusion ---")
    diffusion = create_diffusion(diffusion_steps=args.diffusion_steps)
    print(f"  Steps: {args.diffusion_steps}, Schedule: cosine")

    # --- Create data loader ---
    print("\n--- Creating data loader ---")
    if resolved_pkl and os.path.exists(resolved_pkl):
        print(f"  Using real ResPlan data from {resolved_pkl}")
        data_iter = load_resplan_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            target_set=args.target_set,
            set_name="train",
            pkl_path=resolved_pkl,
            max_num_points=args.max_num_points,
        )
    else:
        if args.pkl:
            print(
                f"  WARNING: could not find '{args.pkl}' from the current directory or project root; "
                "falling back to synthetic data"
            )
        else:
            print("  Using synthetic data (no --pkl provided)")
        data_iter = synthetic_data_generator(
            args.batch_size, args.max_num_points, args.analog_bit
        )

    # --- Verify one batch shape ---
    print("\n--- Verifying batch shapes ---")
    sample_batch, sample_cond = next(data_iter)
    print(f"  batch shape:        {sample_batch.shape}")
    print(f"  room_types shape:   {sample_cond['room_types'].shape}")
    print(f"  door_mask shape:    {sample_cond['door_mask'].shape}")
    print(f"  self_mask shape:    {sample_cond['self_mask'].shape}")
    print(f"  gen_mask shape:     {sample_cond['gen_mask'].shape}")
    print(f"  corner_indices:     {sample_cond['corner_indices'].shape}")
    print(f"  room_indices:       {sample_cond['room_indices'].shape}")
    print(f"  padding_mask:       {sample_cond['src_key_padding_mask'].shape}")
    print(f"  connections:        {sample_cond['connections'].shape}")
    print(f"  graph:              {sample_cond['graph'].shape}")

    expected_seq = args.max_num_points
    B = args.batch_size
    if not args.analog_bit:
        assert sample_batch.shape == (B, 2, expected_seq), \
            f"Expected ({B}, 2, {expected_seq}), got {sample_batch.shape}"
    else:
        assert sample_batch.shape == (B, 16, expected_seq), \
            f"Expected ({B}, 16, {expected_seq}), got {sample_batch.shape}"
    assert sample_cond["door_mask"].shape == (B, expected_seq, expected_seq)
    assert sample_cond["room_types"].shape == (B, expected_seq, 25)
    print("  Shape checks PASSED ✓")

    # --- Need to re-create the iterator since we consumed one batch ---
    # For synthetic data, the generator is still alive.
    # For real data, it's an infinite generator too, so this is fine.
    # But we already consumed one batch. For synthetic, just keep going.
    # For real data via load_resplan_data, it's infinite so also fine.

    # --- Run training ---
    losses = run_training_test(
        data_iter=data_iter,
        model=model,
        diffusion=diffusion,
        num_steps=args.num_steps,
        analog_bit=args.analog_bit,
        device=device,
    )

    # --- Save model checkpoint ---
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "resplan_housediff.pt")
    th.save({
        "model_state_dict": model.state_dict(),
        "num_channels": args.num_channels,
        "max_num_points": args.max_num_points,
        "analog_bit": args.analog_bit,
        "diffusion_steps": args.diffusion_steps,
        "condition_channels": 121,
        "losses": losses,
    }, checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — data loads and trains successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
