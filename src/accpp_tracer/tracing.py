"""Core ACC++ circuit tracing algorithm.

Decomposes attention head firing patterns by tracing upstream contributions
through the model's residual stream, using singular vector (SV) decomposition.
"""

from collections import defaultdict

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache

from .attribution import ig_softmax_attributions
from .models import ModelConfig
from .rope import get_rotation_matrix
from ._typecheck import typechecked

def _greedy_algorithm(
    cache: ActivationCache,
    prompt_id: int,
    layer: int,
    ah_idx: int,
    dest_token: int,
    src_token: int,
    upstream_attention_scores_breakdown: Float[
        Tensor, "n_layers n_components d_head n_tokens n_tokens"
    ],
    attn_weight_thresh: float,
    recalculate_A_d: bool = False,
    list_order: list | None = None,
) -> tuple[dict, dict]:
    """Greedy component selection algorithm.

    Iteratively selects the upstream component that contributes most to the
    attention weight, zeroes it out, and repeats until the attention weight
    drops below the threshold.

    Args:
        cache: Model activation cache.
        prompt_id: Prompt index in the batch.
        layer: Target attention layer.
        ah_idx: Target attention head index.
        dest_token: Destination token position.
        src_token: Source token position.
        upstream_attention_scores_breakdown: Per-component attention score contributions.
        attn_weight_thresh: Minimum attention weight to continue selecting.
        recalculate_A_d: Whether to recompute attention distribution each step.
        list_order: Pre-computed ordering of components to select.

    Returns:
        Tuple of (svs_per_component dict, edge_weights dict).
    """
    MAX_ITER = 1000
    n_iter = 0

    curr_attn_weight = (
        cache[f"blocks.{layer}.attn.hook_pattern"][
            prompt_id, ah_idx, dest_token, src_token
        ]
        .clone()
        .item()
    )
    prev_attn_weight = curr_attn_weight
    attention_scores = cache[
        f"blocks.{layer}.attn.hook_attn_scores"
    ][prompt_id, ah_idx, dest_token, : dest_token + 1].clone()
    A_d = cache[f"blocks.{layer}.attn.hook_pattern"][
        prompt_id, ah_idx, dest_token, : dest_token + 1
    ].clone()

    important_components: list[tuple] = []
    edge_weights: dict[tuple, float] = defaultdict(float)

    while curr_attn_weight >= attn_weight_thresh and n_iter <= MAX_ITER:
        if list_order is None:
            # Adaptive ordering: pick the component that most increases src attention
            delta_s_col = upstream_attention_scores_breakdown[
                :, :, :, :, src_token
            ].unsqueeze(-1)
            diff_matrix = delta_s_col - upstream_attention_scores_breakdown
            exp_matrix = torch.exp(diff_matrix)

            if recalculate_A_d:
                attention_scores = upstream_attention_scores_breakdown.sum(
                    dim=[0, 1, 2, 3]
                )
                if n_iter == 0:
                    assert torch.allclose(
                        torch.softmax(attention_scores, dim=0), A_d, atol=1e-3
                    )
                    assert torch.allclose(
                        attention_scores,
                        cache[
                            f"blocks.{layer}.attn.hook_attn_scores"
                        ][prompt_id, ah_idx, dest_token, : dest_token + 1],
                        atol=1e-2,
                    )
                A_d = torch.softmax(attention_scores, dim=0)

            scores = exp_matrix @ A_d

            # Mask out already-selected components
            for component in important_components:
                scores[component] = -torch.inf

            top_component = np.unravel_index(
                torch.argmax(scores).item(),
                upstream_attention_scores_breakdown[:, :, :, :, 0].shape,
            )
        else:
            # Fixed ordering from pre-computed attribution
            top_component = np.unravel_index(
                list_order[n_iter],
                upstream_attention_scores_breakdown[:, :, :, :, 0].shape,
            )

        top_component = tuple(int(x) for x in top_component)
        # Zero out the selected component and recompute attention
        upstream_attention_scores_breakdown[top_component] = 0.0
        attention_scores = upstream_attention_scores_breakdown.sum(dim=[0, 1, 2, 3])
        curr_attn_weight = torch.softmax(attention_scores, dim=0)[src_token].item()
        important_components.append(top_component)
        n_iter += 1

        # Track cumulative contribution of this component
        edge_weights[top_component[0], top_component[1], top_component[3]] += (
            prev_attn_weight - curr_attn_weight
        )
        prev_attn_weight = curr_attn_weight

    # Reorder to (layer, component, token_pos, sv_idx) format
    important_components = [(a, b, d, c) for a, b, c, d in important_components]

    svs_used_upstream_component: dict[tuple, list] = defaultdict(list)
    for upstream_component_svs in important_components:
        upstream_component = upstream_component_svs[:3]
        svs_used_upstream_component[upstream_component].append(
            upstream_component_svs[3]
        )

    if n_iter >= MAX_ITER:
        print(
            f"Max iterations reached. Debug: "
            f"({prompt_id}, {layer}, {ah_idx}, {dest_token}, {src_token})"
        )
        svs_used_upstream_component = {}

    return svs_used_upstream_component, edge_weights


def _trace_firing_inner(
    model: HookedTransformer,
    cache: ActivationCache,
    prompt_id: int,
    layer: int,
    ah_idx: int,
    dest_token: int,
    src_token: int,
    U: Tensor,
    S: Tensor,
    VT: Tensor,
    device: str,
    c_d: Float[Tensor, "n_layers n_heads d_model"],
    c_s: Float[Tensor, "n_layers n_heads d_model"],
    M_d: Float[Tensor, "d_model d_model"],
    M_s_all: Float[Tensor, "n_tokens d_model d_model"],
    attn_weight_thresh: float,
    config: ModelConfig,
) -> tuple[dict, dict, dict, dict]:
    """Internal trace firing implementation.

    Decomposes the attention score at (dest_token, src_token) into contributions
    from all upstream components (MLPs, attention heads, embeddings, biases)
    across all preceding layers.

    Args:
        model: HookedTransformer model.
        cache: Activation cache from a forward pass.
        prompt_id: Index of the prompt in the batch.
        layer: Target attention layer to decompose.
        ah_idx: Target attention head index.
        dest_token: Query (destination) token position.
        src_token: Key (source) token position.
        U: Left singular vectors from Omega decomposition.
        S: Singular values from Omega decomposition.
        VT: Right singular vectors from Omega decomposition.
        device: Torch device.
        c_d: Query bias offset, shape (n_layers, n_heads, d_model).
        c_s: Key bias offset, shape (n_layers, n_heads, d_model).
        M_d: Destination rotation/projection matrix.
        M_s_all: Source rotation/projection matrices per token.
        attn_weight_thresh: Minimum attention weight for greedy selection.
        config: Model configuration.

    Returns:
        Tuple of (dest_components, dest_edge_weights, src_components, src_edge_weights).
    """
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    if model.cfg.attn_scores_soft_cap > 0:
        logit_softcapping = lambda x: model.cfg.attn_scores_soft_cap * torch.tanh(
            x / model.cfg.attn_scores_soft_cap
        )
    else:
        logit_softcapping = lambda x: x

    ln1_scale = cache[f"blocks.{layer}.ln1.hook_scale"][
        prompt_id, : dest_token + 1, :
    ]
    X = cache[f"blocks.{layer}.ln1.hook_normalized"][
        prompt_id, : dest_token + 1, :
    ].clone()

    # Tensors with per-component attention score contributions
    upstream_attention_scores_breakdown_src = torch.zeros(
        layer, n_heads + 4, d_head, dest_token + 1, dest_token + 1, device=device
    )
    upstream_attention_scores_breakdown_dest = torch.zeros(
        layer, n_heads + 4, d_head, dest_token + 1, dest_token + 1, device=device
    )

    # Computing Omega as d_head rank-1 matrices
    Omega = einsum(
        U[layer, ah_idx] * S[layer, ah_idx],
        VT[layer, ah_idx],
        "d_model d_head, d_head d_model_out -> d_head d_model d_model_out",
    )

    # Pre-compute fixed terms for dest and source contributions
    X_s_all_rot = einsum(
        M_s_all,
        X + c_s[layer, ah_idx],
        "n_tokens d_model d_model_out, n_tokens d_model_out -> n_tokens d_model",
    )
    X_d_rot = (X[dest_token] + c_d[layer, ah_idx]) @ M_d

    for upstream_layer in range(layer):
        # MLP
        mlp_idx = n_heads
        X_mlp = (
            cache[f"blocks.{upstream_layer}.hook_mlp_out"][
                prompt_id, : dest_token + 1, :
            ]
            / ln1_scale
        )

        # AH bias
        bias_idx = n_heads + 1
        X_ah_bias = (
            model.b_O[upstream_layer].clone().detach().repeat(dest_token + 1, 1)
            / ln1_scale
        )

        # AH contributions via OV linearity
        A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][
            prompt_id, :, : dest_token + 1, : dest_token + 1
        ]
        V = cache[f"blocks.{upstream_layer}.attn.hook_v"][
            prompt_id, : dest_token + 1, :, :
        ]

        if config.has_gqa:
            V = torch.repeat_interleave(V, dim=1, repeats=config.gqa_repeats)

        W_O = model.W_O[upstream_layer, :, :, :]
        V_permuted = V.permute(1, 0, 2)
        AV = einsum(
            A,
            V_permuted,
            "n_heads n_tokens_q n_tokens_k, n_heads n_tokens_k d_head "
            "-> n_heads n_tokens_q n_tokens_k d_head",
        )
        upstream_output_all = torch.einsum("hijk,hkl->hijl", AV, W_O)

        if config.has_post_attn_ln:
            upstream_output_all *= model.blocks[upstream_layer].ln1_post.w.detach()
            ln_post_term = cache[
                f"blocks.{upstream_layer}.ln1_post.hook_scale"
            ][prompt_id, : dest_token + 1]
            upstream_output_all /= ln_post_term.view(
                1, ln_post_term.shape[0], 1, 1
            )

        X_ah = upstream_output_all / ln1_scale.unsqueeze(0).unsqueeze(2)

        # Embedding when upstream layer is 0
        if upstream_layer == 0:
            embed_idx = n_heads + 2
            X_embed = (
                cache["blocks.0.hook_resid_pre"][prompt_id, : dest_token + 1, :]
                / ln1_scale
            )

        # MLP update
        upstream_attention_scores_breakdown_dest[
            upstream_layer, mlp_idx, :, dest_token, range(dest_token + 1)
        ] = einsum(
            X_mlp[dest_token] @ M_d,
            Omega,
            X_s_all_rot,
            "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
        )
        upstream_attention_scores_breakdown_src[
            upstream_layer, mlp_idx, :, src_token, range(dest_token + 1)
        ] = einsum(
            X_d_rot,
            Omega,
            einsum(
                M_s_all,
                X_mlp,
                "n_tokens d_model d_model_out, n_tokens d_model_out -> n_tokens d_model",
            ),
            "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
        )

        # AH bias update
        upstream_attention_scores_breakdown_dest[
            upstream_layer, bias_idx, :, dest_token, range(dest_token + 1)
        ] = einsum(
            X_ah_bias[dest_token] @ M_d,
            Omega,
            X_s_all_rot,
            "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
        )
        upstream_attention_scores_breakdown_src[
            upstream_layer, bias_idx, :, src_token, range(dest_token + 1)
        ] = einsum(
            X_d_rot,
            Omega,
            einsum(
                M_s_all,
                X_ah_bias,
                "n_tokens d_model d_model_out, n_tokens d_model_out -> n_tokens d_model",
            ),
            "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
        )

        # AH update
        upstream_attention_scores_breakdown_dest[
            upstream_layer, range(n_heads), :, :, :
        ] = einsum(
            X_ah[:, dest_token, :, :] @ M_d,
            Omega,
            X_s_all_rot,
            "n_heads n_tokens_breakdown d_model, d_head d_model d_model_out, "
            "n_tokens d_model_out -> n_heads d_head n_tokens_breakdown n_tokens",
        )
        upstream_attention_scores_breakdown_src[
            upstream_layer, range(n_heads), :, :, :
        ] = einsum(
            X_d_rot,
            Omega,
            einsum(
                M_s_all,
                X_ah,
                "n_tokens d_model d_model_out, n_heads n_tokens n_tokens_breakdown "
                "d_model_out -> n_heads n_tokens n_tokens_breakdown d_model",
            ),
            "d_model, d_head d_model d_model_out, n_heads n_tokens n_tokens_breakdown "
            "d_model_out -> n_heads d_head n_tokens_breakdown n_tokens",
        )

        # Embedding update
        if upstream_layer == 0:
            upstream_attention_scores_breakdown_dest[
                upstream_layer, embed_idx, :, dest_token, range(dest_token + 1)
            ] = einsum(
                X_embed[dest_token] @ M_d,
                Omega,
                X_s_all_rot,
                "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
            )
            upstream_attention_scores_breakdown_src[
                upstream_layer, embed_idx, :, src_token, range(dest_token + 1)
            ] = einsum(
                X_d_rot,
                Omega,
                einsum(
                    M_s_all,
                    X_embed,
                    "n_tokens d_model d_model_out, n_tokens d_model_out -> n_tokens d_model",
                ),
                "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
            )

    # Constant (bias offset) term
    constant_idx = n_heads + 3
    upstream_attention_scores_breakdown_dest[
        0, constant_idx, :, dest_token, range(dest_token + 1)
    ] = einsum(
        c_d[layer, ah_idx] @ M_d,
        Omega,
        X_s_all_rot,
        "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
    )
    upstream_attention_scores_breakdown_src[
        0, constant_idx, :, src_token, range(dest_token + 1)
    ] = einsum(
        X_d_rot,
        Omega,
        einsum(
            M_s_all,
            c_s[layer, ah_idx].repeat(dest_token + 1, 1),
            "n_tokens d_model d_model_out, n_tokens d_model_out -> n_tokens d_model",
        ),
        "d_model, d_head d_model d_model_out, n_tokens d_model_out -> d_head n_tokens",
    )

    # Normalize by attention scale (usually sqrt(d_head))
    upstream_attention_scores_breakdown_dest /= model.cfg.attn_scale
    upstream_attention_scores_breakdown_src /= model.cfg.attn_scale

    # Verify decomposition correctness.
    # CHANGED: atol raised from 1e-3 to 1e-2. On CUDA, .sum() uses a parallel
    # tree reduction whose summation order differs from CPU's sequential order.
    # Since fp32 addition is not associative, the accumulated rounding error
    # across ~80k terms (d_head × layers × components × tokens for Gemma) can
    # exceed 1e-3 even with TF32 disabled. 1e-2 matches the existing rtol and
    # is still tight enough to catch real formula / indexing errors.
    assert torch.allclose(
        logit_softcapping(
            upstream_attention_scores_breakdown_dest.sum(dim=[0, 1, 2, 3])
        ),
        cache[f"blocks.{layer}.attn.hook_attn_scores"][
            prompt_id, ah_idx, dest_token, : dest_token + 1
        ],
        rtol=1e-2,
        atol=1e-2,  # CHANGED: was 1e-3
    )
    assert torch.allclose(
        logit_softcapping(
            upstream_attention_scores_breakdown_src.sum(dim=[0, 1, 2, 3])
        ),
        cache[f"blocks.{layer}.attn.hook_attn_scores"][
            prompt_id, ah_idx, dest_token, : dest_token + 1
        ],
        rtol=1e-2,
        atol=1e-2,  # CHANGED: was 1e-3
    )

    # Only IG (Integrated Gradients) attribution is used by ACC++.
    contribs_dest, _ = ig_softmax_attributions(
        upstream_attention_scores_breakdown_dest.reshape(
            -1, upstream_attention_scores_breakdown_dest.shape[-1]
        ),
        src_token, T=64, quadrature="trapezoid",
    )
    list_order_dest = np.argsort(contribs_dest)[::-1]

    contribs_src, _ = ig_softmax_attributions(
        upstream_attention_scores_breakdown_src.reshape(
            -1, upstream_attention_scores_breakdown_src.shape[-1]
        ),
        src_token, T=64, quadrature="trapezoid",
    )
    list_order_src = np.argsort(contribs_src)[::-1]

    # Run greedy selection for both dest and src decompositions
    svs_dest, edge_weights_dest = _greedy_algorithm(
        cache, prompt_id, layer, ah_idx, dest_token, src_token,
        upstream_attention_scores_breakdown_dest, attn_weight_thresh,
        list_order=list_order_dest,
    )
    svs_src, edge_weights_src = _greedy_algorithm(
        cache, prompt_id, layer, ah_idx, dest_token, src_token,
        upstream_attention_scores_breakdown_src, attn_weight_thresh,
        list_order=list_order_src,
    )

    return svs_dest, edge_weights_dest, svs_src, edge_weights_src


@typechecked
def trace_firing(
    model: HookedTransformer,
    cache: ActivationCache,
    prompt_id: int,
    layer: int,
    ah_idx: int,
    dest_token: int,
    src_token: int,
    U: Float[Tensor, "n_layers n_heads d_model d_head"],
    S: Float[Tensor, "n_layers n_heads d_head"],
    VT: Float[Tensor, "n_layers n_heads d_head d_model"],
    W_Q_pinv: Float[Tensor, "n_layers n_heads d_head d_model"],
    W_K_pinv: Float[Tensor, "n_layers n_heads d_head d_model"],
    config: ModelConfig,
    device: str,
    attn_weight_thresh: float,
) -> tuple[dict, dict, dict, dict]:
    """Trace an attention head's firing pattern back through upstream layers.

    Decomposes the attention weight at position (dest_token, src_token) in the
    specified attention head into contributions from upstream components (MLPs,
    attention heads, embeddings, biases) using the Omega (QK^T) SVD decomposition.

    Args:
        model: HookedTransformer model.
        cache: Activation cache from model.run_with_cache().
        prompt_id: Index of the prompt in the batch.
        layer: Target attention layer to trace.
        ah_idx: Target attention head index.
        dest_token: Query (destination) token position.
        src_token: Key (source) token position.
        U: Left singular vectors from get_omega_decomposition().
        S: Singular values from get_omega_decomposition().
        VT: Right singular vectors from get_omega_decomposition().
        W_Q_pinv: Pseudoinverse of W_Q from compute_weight_pseudoinverses().
        W_K_pinv: Pseudoinverse of W_K from compute_weight_pseudoinverses().
        config: Model configuration (from get_model_config).
        device: Torch device.
        attn_weight_thresh: Minimum attention weight for greedy selection.

    Returns:
        Tuple of (dest_components, dest_edge_weights, src_components, src_edge_weights)
        where components map (layer, component_id, token_pos) -> [sv_indices]
        and edge_weights map (layer, component_id, token_pos) -> weight.
    """
    c_d = einsum(
        model.b_Q,
        W_Q_pinv,
        "n_layers n_heads d_head, n_layers n_heads d_head d_model "
        "-> n_layers n_heads d_model",
    )
    c_s = einsum(
        model.b_K,
        W_K_pinv,
        "n_layers n_heads d_head, n_layers n_heads d_head d_model "
        "-> n_layers n_heads d_model",
    )

    if config.has_rope:
        R = torch.stack(
            [get_rotation_matrix(model, i, device) for i in range(dest_token + 1)]
        )
        M_d = model.W_Q[layer, ah_idx] @ R[dest_token].T @ W_Q_pinv[layer, ah_idx]
        M_s_all = einsum(
            W_K_pinv[layer, ah_idx].T,
            R,
            model.W_K[layer, ah_idx].T,
            "d_model d_head_a, n_tokens d_head_a d_head_b, d_head_b d_model_out "
            "-> n_tokens d_model d_model_out",
        )
    else:
        # Identity rotation for models without RoPE
        M_s_all = torch.stack(
            [torch.eye(model.cfg.d_model, device=device) for _ in range(dest_token + 1)]
        )
        M_d = M_s_all[0]

    return _trace_firing_inner(
        model, cache, prompt_id, layer, ah_idx, dest_token, src_token,
        U, S, VT, device, c_d, c_s, M_d, M_s_all,
        attn_weight_thresh, config,
    )
