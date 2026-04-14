"""Schedule samplers for diffusion training.

Ported from ``external/house_diffusion/house_diffusion/resample.py``.
Only ``UniformSampler`` and ``ScheduleSampler`` are included;
``LossSecondMomentResampler`` (which depends on ``torch.distributed``) is omitted.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as th


def create_named_schedule_sampler(name: str, diffusion) -> "ScheduleSampler":
    """Create a ScheduleSampler from a library of pre-defined samplers.

    Args:
        name: The name of the sampler.
        diffusion: The diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self) -> np.ndarray:
        """Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size: int, device: th.device) -> tuple[th.Tensor, th.Tensor]:
        """Importance-sample timesteps for a batch.

        Args:
            batch_size: The number of timesteps.
            device: The torch device to save to.

        Returns:
            A tuple (timesteps, weights):
                - timesteps: A tensor of timestep indices.
                - weights: A tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """A sampler that draws timesteps uniformly."""

    def __init__(self, diffusion) -> None:
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self) -> np.ndarray:
        return self._weights
