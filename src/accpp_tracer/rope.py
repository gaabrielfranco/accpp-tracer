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

    sin_angles, cos_angles = torch.sin(angles), torch.cos(angles)

    R_m = torch.zeros((d_head, d_head), device=device)
    for i in range(d_head):
        if i < rotary_dim:
            R_m[i, i] = cos_angles[idx_rotation][i]
        else:
            R_m[i, i] = 1.0

    idx = 0
    for i, j in zip(range(0, rotary_dim // 2), range(rotary_dim // 2, rotary_dim)):
        R_m[i, j] = -sin_angles[idx_rotation][idx]
        idx += 1

    idx = rotary_dim // 2
    for i, j in zip(range(rotary_dim // 2, rotary_dim), range(0, rotary_dim // 2)):
        R_m[i, j] = sin_angles[idx_rotation][idx]
        idx += 1

    return R_m


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
    angles = pos[:, None] / freq[None, :]

    return get_rotary_matrix(token, rotary_dim, model.cfg.d_head, angles, device)
