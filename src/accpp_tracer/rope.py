"""Rotary Position Embedding (RoPE) matrix computation."""

import math

import einops
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from ._typecheck import typechecked


def get_rotary_matrix(
    idx_rotation: int,
    rotary_dim: int,
    d_head: int,
    angles: Tensor,
    device: str,
) -> Float[Tensor, "d_head d_head"]:
    """Compute the rotary matrix for a specific token position.

    Args:
        idx_rotation: Token position index.
        rotary_dim: Dimension of the rotary embedding.
        d_head: Head dimension.
        angles: Pre-computed rotation angles.
        device: Torch device.

    Returns:
        The rotation matrix of shape (d_head, d_head).
    """
    assert rotary_dim <= d_head
    return _build_rotation_stack(
        angles[idx_rotation : idx_rotation + 1], rotary_dim, d_head, device
    )[0]


def _rotation_angles(model: HookedTransformer) -> Float[Tensor, "n_ctx rotary_dim"]:
    """Per-(position, rotary-dim) rotation angles for the model's RoPE config.

    Based on the calculate_sin_cos_rotary function from TransformerLens.
    Handles standard RoPE, NTK-by-parts scaling, and adjacent pairs format.
    """
    rotary_dim = model.cfg.rotary_dim
    n_ctx = model.cfg.n_ctx
    pos = torch.arange(n_ctx)
    dim = torch.arange(rotary_dim // 2)
    base = model.cfg.rotary_base

    if model.cfg.use_NTK_by_parts_rope:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
        )
        factor = model.cfg.NTK_by_parts_factor
        low_freq_factor = model.cfg.NTK_by_parts_low_freq_factor
        high_freq_factor = model.cfg.NTK_by_parts_high_freq_factor
        old_context_len = model.cfg.NTK_original_ctx_len

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        freq = 1 / inv_freq_llama
    else:
        freq = base ** (dim / (rotary_dim / 2))

    if model.cfg.rotary_adjacent_pairs:
        freq = einops.repeat(freq, "d -> (d 2)")
    else:
        freq = einops.repeat(freq, "d -> (2 d)")
    return pos[:, None] / freq[None, :]


def _build_rotation_stack(
    angles: Float[Tensor, "n_positions rotary_dim"],
    rotary_dim: int,
    d_head: int,
    device: str,
) -> Float[Tensor, "n_positions d_head d_head"]:
    """Assemble rotation matrices for a batch of positions (vectorized).

    Reproduces exactly the per-position layout of the original loop-based
    construction: cos on the rotary diagonal, identity on the pass-through
    diagonal, and the +/- sin blocks pairing dimension ``i`` with
    ``i + rotary_dim/2``.
    """
    n_positions = angles.shape[0]
    sin_angles = torch.sin(angles).to(device)
    cos_angles = torch.cos(angles).to(device)
    half = rotary_dim // 2

    R = torch.zeros((n_positions, d_head, d_head), device=device)
    i_rot = torch.arange(rotary_dim, device=device)
    R[:, i_rot, i_rot] = cos_angles[:, i_rot]
    i_pass = torch.arange(rotary_dim, d_head, device=device)
    R[:, i_pass, i_pass] = 1.0
    i1 = torch.arange(half, device=device)
    R[:, i1, i1 + half] = -sin_angles[:, i1]
    i2 = torch.arange(half, rotary_dim, device=device)
    R[:, i2, i2 - half] = sin_angles[:, i2]
    return R


@typechecked
def get_rotation_matrices(
    model: HookedTransformer,
    n_positions: int,
    device: str,
) -> Float[Tensor, "n_positions d_head d_head"]:
    """Compute the stacked RoPE rotation matrices for positions 0..n_positions-1.

    Vectorized over positions — the angle table is built once and the stack
    is assembled with indexed tensor ops (no Python loop over d_head).
    ``Tracer`` caches the result, since it depends only on the model config
    and the maximum position.

    Args:
        model: A HookedTransformer model instance.
        n_positions: Number of token positions (must be <= model.cfg.n_ctx).
        device: Torch device.

    Returns:
        Rotation matrices of shape (n_positions, d_head, d_head).
    """
    assert n_positions <= model.cfg.n_ctx
    angles = _rotation_angles(model)[:n_positions]
    return _build_rotation_stack(
        angles, model.cfg.rotary_dim, model.cfg.d_head, device
    )


@typechecked
def get_rotation_matrix(
    model: HookedTransformer,
    token: int,
    device: str,
) -> Float[Tensor, "d_head d_head"]:
    """Compute the RoPE rotation matrix for a specific token position.

    Based on the calculate_sin_cos_rotary function from TransformerLens.
    Handles standard RoPE, NTK-by-parts scaling, and adjacent pairs format.

    Args:
        model: A HookedTransformer model instance.
        token: Token position index.
        device: Torch device.

    Returns:
        The rotation matrix of shape (d_head, d_head).
    """
    angles = _rotation_angles(model)[token : token + 1]
    return _build_rotation_stack(
        angles, model.cfg.rotary_dim, model.cfg.d_head, device
    )[0]
