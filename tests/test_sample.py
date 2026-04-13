"""
test_sample.py — Load a trained model checkpoint and generate floorplan samples.

Loads the model from models/resplan_housediff.pt, runs the reverse diffusion
process conditioned on a real ResPlan batch, and saves the output as PNG.

Usage:
    python tests/test_sample.py                                    # defaults
    python tests/test_sample.py --pkl data/raw/ResPlan.pkl         # real data conditioning
    python tests/test_sample.py --model models/resplan_housediff.pt --num_samples 4
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "house_diffusion"))
sys.path.insert(0, PROJECT_ROOT)

from external.house_diffusion.house_diffusion.transformer import TransformerModel
from external.house_diffusion.house_diffusion.respace import SpacedDiffusion, space_timesteps
from external.house_diffusion.house_diffusion import gaussian_diffusion as gd

from src.helpers.resplandataset import (
    load_resplan_data,
    DEFAULT_MAX_NUM_POINTS,
    NUM_ROOM_TYPE_CLASSES,
    MAX_CORNER_INDEX,
    MAX_ROOM_INDEX,
    ROOM_TYPE_ID_TO_NAME,
)

# Color map for room types (matches resplan_utils / image_sample.py style)
ROOM_TYPE_COLORS = {
    1: "#EE4D4D",   # living
    2: "#C67C7B",   # bedroom
    3: "#FFD274",   # kitchen
    4: "#BEBEBE",   # bathroom
    10: "#1F849B",  # balcony
    11: "#727171",  # storage
    14: "#a63603",  # front_door
    15: "#D3A2C7",  # veranda
    16: "#785A67",  # stair
    17: "#7BA779",  # garden
    18: "#BFE3E8",  # parking
    19: "#E87A90",  # pool
}


def enable_cpu_compatibility_patch() -> None:
    """Patch .cuda() calls for CPU-only environments."""
    if th.cuda.is_available() or getattr(th.Tensor, "_floorplan_cpu_safe_cuda", False):
        return

    def _cpu_safe_cuda(self, device=None, non_blocking=False, memory_format=th.preserve_format):
        return self.to("cpu")

    th.Tensor.cuda = _cpu_safe_cuda
    th.Tensor._floorplan_cpu_safe_cuda = True


def create_model(
    analog_bit: bool = False,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    num_channels: int = 512,
    condition_channels: int = 121,
) -> TransformerModel:
    """Create a TransformerModel matching the training configuration."""
    num_coords = 16 if analog_bit else 2
    input_channels = num_coords + (2 * 8 if not analog_bit else 0)
    out_channels = num_coords

    return TransformerModel(
        in_channels=input_channels,
        condition_channels=condition_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        dataset="rplan",
        use_checkpoint=False,
        use_unet=False,
        analog_bit=analog_bit,
    )


def create_diffusion(diffusion_steps: int = 100) -> SpacedDiffusion:
    """Create a SpacedDiffusion matching the training configuration."""
    betas = gd.get_named_beta_schedule("cosine", diffusion_steps)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, [diffusion_steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


def sample_from_model(
    model: TransformerModel,
    diffusion: SpacedDiffusion,
    model_kwargs: dict,
    shape: tuple,
    device: str,
    analog_bit: bool = False,
) -> th.Tensor:
    """Run reverse diffusion to generate samples.

    Uses p_sample_loop_progressive directly (p_sample_loop hardcodes i>970
    which only works for 1000-step diffusion).
    """
    final = None
    for sample in tqdm(
        diffusion.p_sample_loop_progressive(
            model,
            shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            device=device,
            analog_bit=analog_bit,
        ),
        desc="Sampling",
        total=diffusion.num_timesteps,
    ):
        final = sample
    return final["sample"]  # [B, 2, max_num_points]


def extract_rooms_from_sample(
    sample: np.ndarray,
    room_types: np.ndarray,
    room_indices: np.ndarray,
    padding_mask: np.ndarray,
) -> list:
    """Convert model output back to list of (polygon_coords, room_type_id) tuples.

    Args:
        sample: [max_num_points, 2] coordinate array (in [-1, 1]).
        room_types: [max_num_points, 25] one-hot room types.
        room_indices: [max_num_points, MAX_ROOM_INDEX] one-hot room indices.
        padding_mask: [max_num_points] where 0 = real corner, 1 = padding.

    Returns:
        List of (coords_array, room_type_id) for each room.
    """
    rooms = []
    current_room_idx = -1
    current_coords = []
    current_type = 0

    for i in range(len(sample)):
        if padding_mask[i] == 1:  # padding (src_key_padding_mask: 1=masked)
            continue

        room_idx = int(np.argmax(room_indices[i]))
        type_id = int(np.argmax(room_types[i]))

        if room_idx != current_room_idx:
            # Save previous room
            if current_coords and len(current_coords) >= 3:
                rooms.append((np.array(current_coords), current_type))
            current_coords = []
            current_room_idx = room_idx
            current_type = type_id

        current_coords.append(sample[i])

    # Don't forget last room
    if current_coords and len(current_coords) >= 3:
        rooms.append((np.array(current_coords), current_type))

    return rooms


def render_floorplan(
    rooms: list,
    resolution: int = 256,
    title: str = "Generated Floorplan",
) -> plt.Figure:
    """Render extracted rooms as colored polygons using matplotlib.

    Args:
        rooms: List of (coords_array, room_type_id) from extract_rooms_from_sample.
        resolution: Pixel resolution for scaling coords.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)

    legend_entries = {}

    for coords, type_id in rooms:
        # Convert from [-1, 1] to [0, resolution]
        pixel_coords = (coords / 2 + 0.5) * resolution
        color = ROOM_TYPE_COLORS.get(type_id, "#CCCCCC")
        type_name = ROOM_TYPE_ID_TO_NAME.get(type_id, f"type_{type_id}")

        try:
            poly = Polygon(pixel_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.7, fc=color, ec="black", linewidth=1.0)

            # Draw corners
            ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                       c=color, edgecolors="gray", s=15, zorder=5, linewidths=0.5)

            if type_name not in legend_entries:
                legend_entries[type_name] = mpatches.Patch(
                    facecolor=color, edgecolor="black", label=type_name
                )
        except Exception:
            # Skip malformed polygons
            continue

    if legend_entries:
        ax.legend(
            handles=list(legend_entries.values()),
            loc="upper left", bbox_to_anchor=(1, 1), frameon=False,
        )

    fig.tight_layout()
    return fig


def resolve_path(path_arg: str, fallback_name: str) -> str:
    """Resolve path relative to CWD or project root."""
    from pathlib import Path
    p = Path(path_arg)
    if p.exists():
        return str(p.resolve())
    project_rel = Path(PROJECT_ROOT) / path_arg
    if project_rel.exists():
        return str(project_rel.resolve())
    # Return project-root-relative for new files
    return str(Path(PROJECT_ROOT) / fallback_name)


def main():
    parser = argparse.ArgumentParser(description="Generate floorplan samples from trained model")
    parser.add_argument("--model", type=str, default="models/resplan_housediff.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--pkl", type=str, default="data/raw/ResPlan.pkl",
                        help="Path to ResPlan.pkl for conditioning data")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of floorplans to generate")
    parser.add_argument("--output_dir", type=str, default="outputs/samples",
                        help="Directory to save generated PNGs")
    parser.add_argument("--device", type=str, default=None,
                        help="'cpu' or 'cuda'. Auto-detects if omitted.")
    parser.add_argument("--target_set", type=int, default=8)
    args = parser.parse_args()

    enable_cpu_compatibility_patch()

    # Resolve device
    if args.device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    elif args.device.startswith("cuda") and not th.cuda.is_available():
        print("WARNING: CUDA requested but unavailable, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # --- Load checkpoint ---
    model_path = resolve_path(args.model, "models/resplan_housediff.pt")
    print(f"Loading checkpoint from {model_path}")
    checkpoint = th.load(model_path, map_location="cpu", weights_only=False)

    num_channels = checkpoint.get("num_channels", 128)
    max_num_points = checkpoint.get("max_num_points", DEFAULT_MAX_NUM_POINTS)
    analog_bit = checkpoint.get("analog_bit", False)
    diffusion_steps = checkpoint.get("diffusion_steps", 100)
    condition_channels = checkpoint.get("condition_channels", 121)

    print(f"  num_channels={num_channels}, max_num_points={max_num_points}")
    print(f"  diffusion_steps={diffusion_steps}, analog_bit={analog_bit}")
    print(f"  condition_channels={condition_channels}")

    # --- Create model ---
    model = create_model(
        analog_bit=analog_bit,
        max_num_points=max_num_points,
        num_channels=num_channels,
        condition_channels=condition_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # --- Create diffusion ---
    diffusion = create_diffusion(diffusion_steps=diffusion_steps)

    # --- Load conditioning data ---
    pkl_path = resolve_path(args.pkl, "data/raw/ResPlan.pkl")
    print(f"Loading conditioning data from {pkl_path}")
    data_iter = load_resplan_data(
        batch_size=args.num_samples,
        analog_bit=analog_bit,
        target_set=args.target_set,
        set_name="eval",
        pkl_path=pkl_path,
        max_num_points=max_num_points,
    )

    data_batch, model_kwargs = next(data_iter)
    print(f"Conditioning batch shape: {data_batch.shape}")

    # Move conditioning to device and add syn_ prefixed copies
    # (the model uses is_syn=True during sampling, which looks up 'syn_' + key)
    for key in list(model_kwargs.keys()):
        if isinstance(model_kwargs[key], th.Tensor):
            model_kwargs[key] = model_kwargs[key].to(device)
            model_kwargs[f"syn_{key}"] = model_kwargs[key]

    # --- Sample ---
    print(f"\nGenerating {args.num_samples} floorplans...")
    with th.no_grad():
        sample = sample_from_model(
            model=model,
            diffusion=diffusion,
            model_kwargs=model_kwargs,
            shape=data_batch.shape,
            device=device,
            analog_bit=analog_bit,
        )

    # sample: [B, 2, max_num_points] → [B, max_num_points, 2]
    sample_np = sample.cpu().numpy().transpose(0, 2, 1)

    # --- Also get ground truth for comparison ---
    gt_np = data_batch.cpu().numpy().transpose(0, 2, 1)

    # --- Slice offsets for condition tensors ---
    _ci = 2 + NUM_ROOM_TYPE_CLASSES
    _ri = _ci + MAX_CORNER_INDEX
    _pm = _ri + MAX_ROOM_INDEX

    # --- Render and save ---
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    room_types_np = model_kwargs["room_types"].cpu().numpy()
    room_indices_np = model_kwargs["room_indices"].cpu().numpy()
    padding_mask_np = model_kwargs["src_key_padding_mask"].cpu().numpy()

    for i in range(args.num_samples):
        # Extract rooms from generated sample
        gen_rooms = extract_rooms_from_sample(
            sample_np[i], room_types_np[i], room_indices_np[i], padding_mask_np[i],
        )

        # Extract rooms from ground truth
        gt_rooms = extract_rooms_from_sample(
            gt_np[i], room_types_np[i], room_indices_np[i], padding_mask_np[i],
        )

        # Render side-by-side: GT vs Generated
        fig, (ax_gt, ax_gen) = plt.subplots(1, 2, figsize=(16, 8))

        # Ground truth
        ax_gt.set_xlim(0, 256)
        ax_gt.set_ylim(0, 256)
        ax_gt.set_aspect("equal")
        ax_gt.set_axis_off()
        ax_gt.set_title("Ground Truth")

        legend_entries = {}
        for coords, type_id in gt_rooms:
            pixel_coords = (coords / 2 + 0.5) * 256
            color = ROOM_TYPE_COLORS.get(type_id, "#CCCCCC")
            type_name = ROOM_TYPE_ID_TO_NAME.get(type_id, f"type_{type_id}")
            try:
                poly = Polygon(pixel_coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    xs, ys = poly.exterior.xy
                    ax_gt.fill(xs, ys, alpha=0.7, fc=color, ec="black", linewidth=1.0)
                    ax_gt.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                                  c=color, edgecolors="gray", s=15, zorder=5, linewidths=0.5)
                    if type_name not in legend_entries:
                        legend_entries[type_name] = mpatches.Patch(
                            facecolor=color, edgecolor="black", label=type_name
                        )
            except Exception:
                continue

        # Generated
        ax_gen.set_xlim(0, 256)
        ax_gen.set_ylim(0, 256)
        ax_gen.set_aspect("equal")
        ax_gen.set_axis_off()
        ax_gen.set_title("Generated")

        for coords, type_id in gen_rooms:
            pixel_coords = (coords / 2 + 0.5) * 256
            color = ROOM_TYPE_COLORS.get(type_id, "#CCCCCC")
            type_name = ROOM_TYPE_ID_TO_NAME.get(type_id, f"type_{type_id}")
            try:
                poly = Polygon(pixel_coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    xs, ys = poly.exterior.xy
                    ax_gen.fill(xs, ys, alpha=0.7, fc=color, ec="black", linewidth=1.0)
                    ax_gen.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                                  c=color, edgecolors="gray", s=15, zorder=5, linewidths=0.5)
                    if type_name not in legend_entries:
                        legend_entries[type_name] = mpatches.Patch(
                            facecolor=color, edgecolor="black", label=type_name
                        )
            except Exception:
                continue

        # Shared legend
        if legend_entries:
            fig.legend(
                handles=list(legend_entries.values()),
                loc="lower center", ncol=min(len(legend_entries), 6),
                frameon=False, fontsize=10,
            )

        fig.suptitle(f"Sample {i}", fontsize=14)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        save_path = os.path.join(output_dir, f"sample_{i:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path}")

    print(f"\nDone! {args.num_samples} samples saved to {output_dir}")


if __name__ == "__main__":
    main()
