"""Various utilities for neural networks.

Ported from ``external/house_diffusion/house_diffusion/nn.py`` with identical
behaviour.  Only import paths and style were adjusted.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    """Sigmoid Linear Unit activation (compatibility shim for older PyTorch)."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that casts to float32 internally for numerical stability."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args: Any, **kwargs: Any) -> nn.Module:
    """Create a 1D, 2D, or 3D convolution module.

    Args:
        dims: Number of spatial dimensions (1, 2, or 3).
        *args: Positional arguments forwarded to the convolution constructor.
        **kwargs: Keyword arguments forwarded to the convolution constructor.

    Returns:
        The corresponding ``nn.Conv{dims}d`` module.

    Raises:
        ValueError: If *dims* is not in {1, 2, 3}.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args: Any, **kwargs: Any) -> nn.Linear:
    """Create a linear module.

    Args:
        *args: Positional arguments forwarded to ``nn.Linear``.
        **kwargs: Keyword arguments forwarded to ``nn.Linear``.

    Returns:
        An ``nn.Linear`` instance.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args: Any, **kwargs: Any) -> nn.Module:
    """Create a 1D, 2D, or 3D average pooling module.

    Args:
        dims: Number of spatial dimensions (1, 2, or 3).
        *args: Positional arguments forwarded to the pooling constructor.
        **kwargs: Keyword arguments forwarded to the pooling constructor.

    Returns:
        The corresponding ``nn.AvgPool{dims}d`` module.

    Raises:
        ValueError: If *dims* is not in {1, 2, 3}.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(
    target_params: Iterable[th.Tensor],
    source_params: Iterable[th.Tensor],
    rate: float = 0.99,
) -> None:
    """Update target parameters to be closer to source via exponential moving average.

    Args:
        target_params: The target parameter sequence.
        source_params: The source parameter sequence.
        rate: The EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it.

    Args:
        module: The module whose parameters will be zeroed.

    Returns:
        The same module, with all parameters set to zero.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale the parameters of a module and return it.

    Args:
        module: The module whose parameters will be scaled.
        scale: The multiplicative scale factor.

    Returns:
        The same module, with all parameters multiplied by *scale*.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: th.Tensor, padding_mask: th.Tensor) -> th.Tensor:
    """Take the mean over all non-batch dimensions, respecting a padding mask.

    Args:
        tensor: Input tensor of shape ``[B, C, ...]``.
        padding_mask: Binary mask of shape ``[B, ...]`` indicating valid positions.

    Returns:
        A 1-D tensor of per-sample means.
    """
    tensor = tensor * padding_mask.unsqueeze(1)
    tensor = tensor.mean(dim=list(range(1, len(tensor.shape)))) / th.sum(padding_mask, dim=1)
    return tensor


def normalization(channels: int) -> nn.Module:
    """Make a standard normalization layer.

    Args:
        channels: Number of input channels.

    Returns:
        An ``nn.Module`` for normalization (GroupNorm with 32 groups).
    """
    return GroupNorm32(32, channels)


def timestep_embedding(
    timesteps: th.Tensor,
    dim: int,
    max_period: int = 10000,
) -> th.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: A 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        dim: The dimension of the output.
        max_period: Controls the minimum frequency of the embeddings.

    Returns:
        An ``[N x dim]`` Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(
    func: Any,
    inputs: Sequence[th.Tensor],
    params: Sequence[th.Tensor],
    flag: bool,
) -> th.Tensor:
    """Evaluate a function without caching intermediate activations.

    This allows reduced memory at the expense of extra compute in the backward
    pass.

    Args:
        func: The function to evaluate.
        inputs: The argument sequence to pass to *func*.
        params: A sequence of parameters *func* depends on but does not
            explicitly take as arguments.
        flag: If ``False``, disable gradient checkpointing.

    Returns:
        The result of calling *func* on *inputs*.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    """Custom autograd function for gradient checkpointing."""

    @staticmethod
    def forward(ctx: Any, run_function: Any, length: int, *args: th.Tensor) -> th.Tensor:
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx: Any, *output_grads: th.Tensor) -> tuple[None, None, ...]:
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
