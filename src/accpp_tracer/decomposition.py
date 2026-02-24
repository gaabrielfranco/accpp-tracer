"""Omega (QK^T) SVD decomposition for attention heads."""

from typing import Tuple

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from .models import ModelConfig
from ._typecheck import typechecked


@typechecked
def get_omega_decomposition(
    model: HookedTransformer,
    config: ModelConfig,
    device: str = "cpu",
) -> Tuple[
    Float[Tensor, "n_layers n_heads d_model d_head"],
    Float[Tensor, "n_layers n_heads d_head"],
    Float[Tensor, "n_layers n_heads d_head d_model"],
]:
    """Compute SVD decomposition of Q@K^T (Omega) for all attention heads.

    Factorizes the attention weight matrix Omega = W_Q @ W_K^T into
    U @ diag(S) @ VT for each attention head, enabling decomposition of
    attention patterns into rank-1 components (singular vectors).

    Args:
        model: A HookedTransformer model instance.
        config: Model configuration (from get_model_config).
        device: Torch device for output tensors.

    Returns:
        Tuple of (U, S, VT) where:
            U: Left singular vectors, shape (n_layers, n_heads, d_model, d_head).
            S: Singular values, shape (n_layers, n_heads, d_head).
            VT: Right singular vectors, shape (n_layers, n_heads, d_head, d_model).
    """
    rank = model.cfg.d_head

    omega = einsum(
        model.W_Q if not config.use_numpy_svd else model.W_Q.cpu(),
        model.W_K if not config.use_numpy_svd else model.W_K.cpu(),
        "n_layers n_heads d_model d_head, n_layers n_heads d_model_out d_head "
        "-> n_layers n_heads d_model d_model_out",
    )

    if config.use_numpy_svd:
        U_np, S_np, VT_np = np.linalg.svd(omega)
        U = torch.from_numpy(U_np[:, :, :, :rank]).to(device)
        S = torch.from_numpy(S_np[:, :, :rank]).to(device)
        VT = torch.from_numpy(VT_np[:, :, :rank, :]).to(device)
    else:
        U, S, VT = torch.linalg.svd(omega)
        U = U[:, :, :, :rank].to(device)
        S = S[:, :, :rank].to(device)
        VT = VT[:, :, :rank, :].to(device)

    return U, S, VT


@typechecked
def compute_weight_pseudoinverses(
    model: HookedTransformer,
    config: ModelConfig,
    device: str = "cpu",
) -> Tuple[
    Float[Tensor, "n_layers n_heads d_head d_model"],
    Float[Tensor, "n_layers n_heads d_head d_model"],
]:
    """Compute pseudoinverses of W_Q and W_K weight matrices.

    Used for computing bias offsets in the trace_firing algorithm.
    Uses numpy for models that require it for numerical stability.

    Args:
        model: A HookedTransformer model instance.
        config: Model configuration (from get_model_config).
        device: Torch device for output tensors.

    Returns:
        Tuple of (W_Q_pinv, W_K_pinv).
    """

    if config.use_numpy_svd:
        W_Q_pinv = torch.from_numpy(np.linalg.pinv(model.W_Q.cpu())).to(device)
        W_K_pinv = torch.from_numpy(np.linalg.pinv(model.W_K.cpu())).to(device)
    else:
        W_Q_pinv = torch.linalg.pinv(model.W_Q).to(device)
        W_K_pinv = torch.linalg.pinv(model.W_K).to(device)

    return W_Q_pinv, W_K_pinv
