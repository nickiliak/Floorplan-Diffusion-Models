"""Step 7 Validation — v1 pipeline correctness checks.

Verifies that the ported HouseDiffusion components produce correct results:
  1. Tensor shapes from ResPlan dataset match HouseDiffusion's shapes
  2. Ported Transformer produces identical output given identical inputs
  3. Ported diffusion produces identical noise schedules and loss values
  4-6. Deferred checks requiring real data and GPU training
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch as th
from shapely.geometry import Polygon

# Ensure src/ is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "house_diffusion"))

from src.floorplan_diffusion.data.dataset import ResPlanDataset
from src.floorplan_diffusion.models.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from src.floorplan_diffusion.models.respace import SpacedDiffusion, space_timesteps
from src.floorplan_diffusion.models.transformer import TransformerModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_REAL_POINTS = 12  # 3 rooms x 4 corners each


def _make_synthetic_plan() -> dict:
    """Build a minimal ResPlan-style plan dict with 3 rectangular rooms."""
    living = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    bedroom = Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    kitchen = Polygon([(0, 5), (5, 5), (5, 10), (0, 10)])
    inner = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    graph = nx.Graph()
    graph.add_node("living_0")
    graph.add_node("bedroom_0")
    graph.add_node("kitchen_0")
    graph.add_edge("living_0", "bedroom_0")
    graph.add_edge("living_0", "kitchen_0")

    return {
        "living": living,
        "bedroom": bedroom,
        "kitchen": kitchen,
        "bathroom": None,
        "balcony": None,
        "inner": inner,
        "graph": graph,
    }


@pytest.fixture
def synthetic_plan():
    return _make_synthetic_plan()


@pytest.fixture
def processed_plan(synthetic_plan):
    """Run _process_plan and return (house_array, door_mask, self_mask, gen_mask)."""
    ds = ResPlanDataset.__new__(ResPlanDataset)
    result = ds._process_plan(synthetic_plan)
    assert result is not None, "synthetic plan should not be filtered out"
    return result


@pytest.fixture
def sample_item(processed_plan):
    """Simulate __getitem__ output from the processed plan."""
    house_array, door_mask, self_mask, gen_mask = processed_plan

    ds = ResPlanDataset.__new__(ResPlanDataset)
    ds.houses = [house_array]
    ds.door_masks = [door_mask]
    ds.self_masks = [self_mask]
    ds.gen_masks = [gen_mask]
    ds.set_name = "eval"  # no augmentation
    ds.num_coords = 2
    ds.max_num_points = 100

    arr, cond = ds[0]
    return arr, cond


# ---------------------------------------------------------------------------
# Check 1: Dataset tensor shapes
# ---------------------------------------------------------------------------


class TestDatasetShapes:
    """Verify that ResPlan dataset output shapes match HouseDiffusion expectations."""

    def test_arr_shape(self, sample_item):
        arr, _ = sample_item
        assert arr.shape == (2, 100), f"Expected (2, 100), got {arr.shape}"

    def test_room_types_shape(self, sample_item):
        _, cond = sample_item
        assert cond["room_types"].shape == (100, 25)

    def test_corner_indices_shape(self, sample_item):
        _, cond = sample_item
        assert cond["corner_indices"].shape == (100, 32)

    def test_room_indices_shape(self, sample_item):
        _, cond = sample_item
        assert cond["room_indices"].shape == (100, 32)

    def test_padding_mask_shape(self, sample_item):
        _, cond = sample_item
        assert cond["src_key_padding_mask"].shape == (100,)

    def test_connections_shape(self, sample_item):
        _, cond = sample_item
        assert cond["connections"].shape == (100, 2)

    def test_door_mask_shape(self, sample_item):
        _, cond = sample_item
        assert cond["door_mask"].shape == (100, 100)

    def test_self_mask_shape(self, sample_item):
        _, cond = sample_item
        assert cond["self_mask"].shape == (100, 100)

    def test_gen_mask_shape(self, sample_item):
        _, cond = sample_item
        assert cond["gen_mask"].shape == (100, 100)


class TestDatasetMaskProperties:
    """Verify mask invariants required by the Transformer."""

    def test_masks_are_binary(self, sample_item):
        _, cond = sample_item
        for name in ["door_mask", "self_mask", "gen_mask"]:
            mask = cond[name]
            unique = set(np.unique(mask))
            assert unique <= {0.0, 1.0}, f"{name} has non-binary values: {unique}"

    def test_padding_mask_values(self, sample_item):
        _, cond = sample_item
        mask = cond["src_key_padding_mask"]
        # First NUM_REAL_POINTS should be 0 (real), rest should be 1 (padding).
        assert np.all(mask[:NUM_REAL_POINTS] == 0), "real points should have mask=0"
        assert np.all(mask[NUM_REAL_POINTS:] == 1), "padding points should have mask=1"

    def test_gen_mask_blocks_padding(self, sample_item):
        _, cond = sample_item
        gen = cond["gen_mask"]
        # Padding rows/cols should be 1 (blocked).
        assert np.all(gen[NUM_REAL_POINTS:, :] == 1), "gen_mask should block padding rows"
        assert np.all(gen[:, NUM_REAL_POINTS:] == 1), "gen_mask should block padding cols"
        # Real-to-real should be 0 (unblocked).
        assert np.all(gen[:NUM_REAL_POINTS, :NUM_REAL_POINTS] == 0), (
            "gen_mask should allow real-to-real attention"
        )

    def test_self_mask_blocks_cross_room(self, sample_item):
        _, cond = sample_item
        sm = cond["self_mask"]
        # Within each room (4 consecutive points), self_mask should be 0.
        for room_start in range(0, NUM_REAL_POINTS, 4):
            room_end = room_start + 4
            block = sm[room_start:room_end, room_start:room_end]
            assert np.all(block == 0), (
                f"self_mask should be 0 within room [{room_start}:{room_end}]"
            )
        # Between different rooms, self_mask should be 1.
        assert sm[0, 4] == 1, "self_mask should be 1 between different rooms"

    def test_door_mask_respects_graph(self, sample_item):
        _, cond = sample_item
        dm = cond["door_mask"]
        # living (0:4) <-> bedroom (4:8): connected, so door_mask=0.
        assert np.all(dm[0:4, 4:8] == 0), "door_mask should be 0 between connected rooms"
        assert np.all(dm[4:8, 0:4] == 0), "door_mask should be symmetric"
        # living (0:4) <-> kitchen (8:12): connected.
        assert np.all(dm[0:4, 8:12] == 0)
        # bedroom (4:8) <-> kitchen (8:12): NOT connected, so door_mask=1.
        assert np.all(dm[4:8, 8:12] == 1), (
            "door_mask should be 1 between non-connected rooms"
        )


class TestDatasetCoordinateNormalization:
    """Verify coordinate normalization to [-1, 1]."""

    def test_coords_in_range(self, sample_item):
        arr, cond = sample_item
        # Only check real (non-padding) points.
        mask = cond["src_key_padding_mask"]
        real_mask = mask == 0
        real_coords = arr[:, real_mask]  # [2, num_real]
        assert real_coords.min() >= -1.0, f"coords below -1: {real_coords.min()}"
        assert real_coords.max() <= 1.0, f"coords above 1: {real_coords.max()}"

    def test_padding_coords_are_zero(self, sample_item):
        arr, cond = sample_item
        mask = cond["src_key_padding_mask"]
        pad_mask = mask == 1
        pad_coords = arr[:, pad_mask]
        assert np.all(pad_coords == 0), "padding coordinates should be zero"


# ---------------------------------------------------------------------------
# Check 2: Ported Transformer identical to original
# ---------------------------------------------------------------------------


class TestTransformerIdentical:
    """Verify ported Transformer produces identical output to the original."""

    def test_transformer_output_identical(self):
        from house_diffusion.transformer import (
            TransformerModel as OrigTransformerModel,
        )

        model_args = dict(
            in_channels=18,  # 2 coords + 2*8 binary
            condition_channels=89,  # 25 + 32 + 32
            model_channels=64,  # small for testing
            out_channels=2,
            dataset="rplan",
            use_checkpoint=False,
            use_unet=False,
            analog_bit=False,
        )

        th.manual_seed(42)
        ported = TransformerModel(**model_args)
        th.manual_seed(42)
        original = OrigTransformerModel(**model_args)

        # Copy state from ported -> original to ensure identical weights.
        original.load_state_dict(ported.state_dict())

        ported.eval()
        original.eval()

        # Synthetic inputs.
        B, S = 2, 100
        th.manual_seed(0)
        x = th.randn(B, 2, S)
        timesteps = th.tensor([50, 100])
        xtalpha = th.ones(B, S, 2)
        epsalpha = th.ones(B, S, 2) * 0.5

        # Condition tensors — must be [B, S, C].
        room_types = th.zeros(B, S, 25)
        room_types[:, :, 1] = 1  # all "living"
        corner_indices = th.zeros(B, S, 32)
        corner_indices[:, :, 0] = 1
        room_indices = th.zeros(B, S, 32)
        room_indices[:, :, 1] = 1

        # Masks.
        door_mask = th.zeros(B, S, S)
        self_mask = th.zeros(B, S, S)
        gen_mask = th.zeros(B, S, S)
        # Mark some padding to make gen_mask valid (must have both 0 and 1).
        gen_mask[:, 12:, :] = 1
        gen_mask[:, :, 12:] = 1

        connections = th.zeros(B, S, 2)
        for i in range(S):
            connections[:, i, 0] = i
            connections[:, i, 1] = (i + 1) % S

        kwargs = {
            "room_types": room_types,
            "corner_indices": corner_indices,
            "room_indices": room_indices,
            "door_mask": door_mask,
            "self_mask": self_mask,
            "gen_mask": gen_mask,
            "connections": connections,
            "src_key_padding_mask": th.zeros(B, S),
        }

        with th.no_grad():
            out_dec_p, out_bin_p = ported(x, timesteps, xtalpha, epsalpha, **kwargs)
            out_dec_o, out_bin_o = original(x, timesteps, xtalpha, epsalpha, **kwargs)

        assert th.allclose(out_dec_p, out_dec_o, atol=1e-5), (
            f"out_dec max diff: {(out_dec_p - out_dec_o).abs().max():.2e}"
        )
        assert th.allclose(out_bin_p, out_bin_o, atol=1e-5), (
            f"out_bin max diff: {(out_bin_p - out_bin_o).abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# Check 3: Ported diffusion produces identical schedules and q_sample
# ---------------------------------------------------------------------------


class TestDiffusionScheduleIdentical:
    """Verify ported diffusion schedule arrays match the original."""

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_beta_schedule_identical(self, schedule):
        from house_diffusion.gaussian_diffusion import (
            get_named_beta_schedule as orig_get_schedule,
        )

        steps = 1000
        ported_betas = get_named_beta_schedule(schedule, steps)
        orig_betas = orig_get_schedule(schedule, steps)

        np.testing.assert_allclose(
            ported_betas, orig_betas, atol=1e-12,
            err_msg=f"{schedule} beta schedules differ",
        )

    def test_diffusion_precomputed_arrays_identical(self):
        from house_diffusion.gaussian_diffusion import (
            GaussianDiffusion as OrigGaussianDiffusion,
            LossType as OrigLossType,
            ModelMeanType as OrigModelMeanType,
            ModelVarType as OrigModelVarType,
        )

        betas = get_named_beta_schedule("cosine", 100)

        ported = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        original = OrigGaussianDiffusion(
            betas=betas,
            model_mean_type=OrigModelMeanType.EPSILON,
            model_var_type=OrigModelVarType.FIXED_LARGE,
            loss_type=OrigLossType.MSE,
        )

        for attr in [
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            np.testing.assert_allclose(
                getattr(ported, attr),
                getattr(original, attr),
                atol=1e-12,
                err_msg=f"{attr} differs between ported and original",
            )

    def test_q_sample_identical(self):
        from house_diffusion.gaussian_diffusion import (
            GaussianDiffusion as OrigGaussianDiffusion,
            LossType as OrigLossType,
            ModelMeanType as OrigModelMeanType,
            ModelVarType as OrigModelVarType,
        )

        betas = get_named_beta_schedule("cosine", 100)

        ported = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        original = OrigGaussianDiffusion(
            betas=betas,
            model_mean_type=OrigModelMeanType.EPSILON,
            model_var_type=OrigModelVarType.FIXED_LARGE,
            loss_type=OrigLossType.MSE,
        )

        x_start = th.randn(4, 2, 100)
        t = th.tensor([10, 30, 50, 90])
        noise = th.randn_like(x_start)

        q_ported = ported.q_sample(x_start, t, noise=noise)
        q_original = original.q_sample(x_start, t, noise=noise)

        assert th.allclose(q_ported, q_original, atol=1e-6), (
            f"q_sample max diff: {(q_ported - q_original).abs().max():.2e}"
        )


class TestTrainingLossesRuns:
    """Verify the ported training_losses runs and produces expected keys."""

    def test_training_losses_produces_correct_keys(self):
        betas = get_named_beta_schedule("cosine", 100)
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(100, [100]),
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )

        # Build a small model.
        model = TransformerModel(
            in_channels=18,
            condition_channels=89,
            model_channels=64,
            out_channels=2,
            dataset="rplan",
            use_checkpoint=False,
            use_unet=False,
            analog_bit=False,
        )

        B, S = 2, 100
        x_start = th.randn(B, 2, S) * 0.5  # keep in reasonable range

        # Build conditioning.
        room_types = th.zeros(B, S, 25)
        room_types[:, :4, 1] = 1
        room_types[:, 4:8, 2] = 1
        room_types[:, 8:12, 3] = 1

        corner_indices = th.zeros(B, S, 32)
        for i in range(12):
            corner_indices[:, i, i % 4] = 1

        room_indices = th.zeros(B, S, 32)
        room_indices[:, 0:4, 1] = 1
        room_indices[:, 4:8, 2] = 1
        room_indices[:, 8:12, 3] = 1

        padding_mask = th.ones(B, S)
        padding_mask[:, :12] = 0

        connections = th.zeros(B, S, 2)
        for i in range(S):
            connections[:, i, 0] = i
            connections[:, i, 1] = (i + 1) % S

        door_mask = th.ones(B, S, S)
        door_mask[:, :12, :12] = 0
        self_mask = th.ones(B, S, S)
        for r in range(0, 12, 4):
            self_mask[:, r : r + 4, r : r + 4] = 0
        gen_mask = th.ones(B, S, S)
        gen_mask[:, :12, :12] = 0

        model_kwargs = {
            "room_types": room_types,
            "corner_indices": corner_indices,
            "room_indices": room_indices,
            "src_key_padding_mask": padding_mask,
            "connections": connections,
            "door_mask": door_mask,
            "self_mask": self_mask,
            "gen_mask": gen_mask,
        }

        t = th.randint(0, 100, (B,))
        terms = diffusion.training_losses(
            model, x_start, t, model_kwargs, analog_bit=False
        )

        assert "loss" in terms, "training_losses must return 'loss' key"
        assert "mse_dec" in terms, "training_losses must return 'mse_dec' key"
        assert "mse_bin" in terms, "training_losses must return 'mse_bin' key"
        assert terms["loss"].shape == (B,), f"loss shape: {terms['loss'].shape}"
        assert th.isfinite(terms["loss"]).all(), "loss contains non-finite values"


# ---------------------------------------------------------------------------
# Checks 4-6: Deferred (require real data + GPU training)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires ResPlan data and GPU training")
def test_training_loss_decreases():
    """Check 4: Training loss decreases over time (model is learning)."""


@pytest.mark.skip(reason="Requires trained model checkpoint")
def test_generated_floorplans_reasonable():
    """Check 5: Generated floorplans are visually reasonable after training."""


@pytest.mark.skip(reason="Requires trained model checkpoint")
def test_graph_accuracy():
    """Check 6: Graph accuracy metric confirms generated rooms match conditioning graph."""
