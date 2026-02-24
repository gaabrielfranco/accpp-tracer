"""Signal extraction utilities for ACC++ circuit edges.

Provides functions to extract upstream component outputs and convert
component labels, used by Tracer.extract_edge_signal() for computing
per-edge signal vectors.
"""

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from typing import Union

from .models import ModelConfig


def component_label_to_id(component: Union[str, int], n_heads: int) -> int:
    """Convert component label to integer ID.

    Inverse of get_ah_idx_label() in circuit.py.

    Args:
        component: Component label — int for regular attention heads, or
            one of "MLP", "AH bias", "Embedding", "AH offset".
        n_heads: Number of attention heads in the model.

    Returns:
        Integer component ID.

    Raises:
        ValueError: If the component label is not recognized.
    """
    if isinstance(component, int):
        return component
    if component == "MLP":
        return n_heads
    elif component == "AH bias":
        return n_heads + 1
    elif component == "Embedding":
        return n_heads + 2
    elif component == "AH offset":
        return n_heads + 3
    else:
        raise ValueError(f"Unknown component label: {component}")


def get_component_output(
    model: HookedTransformer,
    cache: ActivationCache,
    config: ModelConfig,
    prompt_idx: int,
    downstream_layer: int,
    downstream_ah_idx: int,
    upstream_dest_token: int,
    upstream_src_token: int,
    upstream_layer: int,
    upstream_component_id: int,
    c_term: Float[Tensor, "n_layers n_heads d_model"] | None,
) -> Float[Tensor, "d_model"]:
    """Extract a single upstream component's output, normalized by downstream LN.

    Given an edge in a traced ACC++ circuit graph, extracts the output vector
    of the upstream component at the specified positions, normalized by the
    downstream attention head's layer norm scale.

    Component types (indexed by upstream_component_id):
        0..n_heads-1: Attention head (A * V @ W_O, with GQA and post-attn LN)
        n_heads:      MLP output
        n_heads+1:    Attention bias (b_O)
        n_heads+2:    Embedding (residual stream at layer 0)
        n_heads+3:    AH offset (bias projection c_d or c_s)

    Args:
        model: HookedTransformer model.
        cache: Activation cache from model.run_with_cache().
        config: Model configuration.
        prompt_idx: Index of the prompt in the cache batch.
        downstream_layer: Layer of the downstream attention head.
        downstream_ah_idx: Head index of the downstream attention head.
        upstream_dest_token: Destination (query) token position of the
            upstream component.
        upstream_src_token: Source (key) token position of the upstream
            component.
        upstream_layer: Layer of the upstream component.
        upstream_component_id: Integer ID of the upstream component type.
        c_term: Bias offset tensor (c_d for dest edges, c_s for src edges).
            Shape: (n_layers, n_heads, d_model). Required only for AH offset
            component (n_heads+3), can be None otherwise.

    Returns:
        Component output vector of shape (d_model,), normalized by the
        downstream layer norm scale at upstream_dest_token.
    """
    n_heads = model.cfg.n_heads

    # Downstream layer norm scale at the upstream dest token position
    ln1_scale = cache[f"blocks.{downstream_layer}.ln1.hook_scale"][
        prompt_idx, upstream_dest_token
    ]  # shape: (1,)

    # --- AH component ---
    if upstream_component_id < n_heads:
        # Attention weight (scalar) at (dest, src) for this specific head
        A_val = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][
            prompt_idx, upstream_component_id,
            upstream_dest_token, upstream_src_token
        ]

        # Value vector with GQA head mapping
        if config.has_gqa:
            kv_idx = upstream_component_id // config.gqa_repeats
        else:
            kv_idx = upstream_component_id

        v = cache[f"blocks.{upstream_layer}.attn.hook_v"][
            prompt_idx, upstream_src_token, kv_idx, :
        ]  # (d_head,)

        # AH output: A * v @ W_O
        x = A_val * (v @ model.W_O[upstream_layer, upstream_component_id])  # (d_model,)

        # Post-attention layer norm (Gemma-2 has unfold post-attn LN)
        if config.has_post_attn_ln:
            x = x * model.blocks[upstream_layer].ln1_post.w.detach()
            ln_post_scale = cache[
                f"blocks.{upstream_layer}.ln1_post.hook_scale"
            ][prompt_idx, upstream_dest_token]
            x = x / ln_post_scale

        # Normalize by downstream LN
        x = x / ln1_scale

    # --- MLP ---
    elif upstream_component_id == n_heads:
        x = (
            cache[f"blocks.{upstream_layer}.hook_mlp_out"][
                prompt_idx, upstream_dest_token, :
            ]
            / ln1_scale
        )

    # --- AH bias (b_O) ---
    elif upstream_component_id == n_heads + 1:
        x = model.b_O[upstream_layer].clone().detach() / ln1_scale

    # --- Embedding (residual stream at layer 0) ---
    elif upstream_component_id == n_heads + 2:
        x = (
            cache["blocks.0.hook_resid_pre"][prompt_idx, upstream_dest_token, :]
            / ln1_scale
        )

    # --- AH offset (bias projection term) ---
    elif upstream_component_id == n_heads + 3:
        if c_term is None:
            raise ValueError(
                "c_term (bias offset) is required for AH offset component "
                f"(component_id={upstream_component_id}). Pass c_d for dest "
                "edges or c_s for src edges."
            )
        x = c_term[downstream_layer, downstream_ah_idx]  # No LN normalization

    else:
        raise ValueError(
            f"upstream_component_id={upstream_component_id} exceeds maximum "
            f"valid ID ({n_heads + 3})"
        )

    return x
