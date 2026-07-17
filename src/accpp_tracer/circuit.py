"""Full ACC++ circuit builder.

Provides the Tracer class that orchestrates the complete circuit tracing pipeline:
precomputing model-level quantities, identifying seed components, and recursively
building circuit graphs.
"""

import warnings
from collections import defaultdict
from typing import Callable, Literal, Union

import networkx as nx
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache

from einops import einsum

from functools import partial

import torch.nn.functional as F

from .decomposition import (
    compute_weight_pseudoinverses,
    get_omega_decomposition,
    load_decomposition_cache,
    save_decomposition_cache,
)
from .intervention import (
    EdgeSpec,
    InterventionResult,
    _should_center,
)
from .models import get_model_config, ModelConfig
from .rope import get_rotation_matrix
from .signals import get_component_output
from .tracing import trace_firing
from ._typecheck import typechecked


def get_ah_idx_label(ah_idx: int, n_heads: int) -> Union[int, str]:
    """Convert attention head index to a human-readable label.

    Args:
        ah_idx: Attention head index (can exceed n_heads for special components).
        n_heads: Number of attention heads in the model.

    Returns:
        The head index (int) for regular heads, or a string label for
        special components: "MLP", "AH bias", "Embedding", "AH offset".
    """
    if ah_idx == n_heads:
        return "MLP"
    elif ah_idx == n_heads + 1:
        return "AH bias"
    elif ah_idx == n_heads + 2:
        return "Embedding"
    elif ah_idx == n_heads + 3:
        return "AH offset"
    return ah_idx


def get_upstream_contributors_seed(
    contrib: np.ndarray, frac_contrib_thresh: float = 1.0
) -> list[tuple]:
    """Identify seed components by greedily selecting top contributors.

    Finds the minimal set of (layer, ah_idx, token) tuples whose cumulative
    contribution reaches frac_contrib_thresh of the total contribution.

    Args:
        contrib: Contribution array, shape (n_layers, n_components, n_tokens).
        frac_contrib_thresh: Fraction of total contribution to capture.

    Returns:
        List of (layer, ah_idx, token) tuples.
    """
    sorted_contribs = np.sort(np.ravel(contrib))[::-1]
    thresh = frac_contrib_thresh * np.sum(np.ravel(contrib))
    cutoff = sorted_contribs[np.where(np.cumsum(sorted_contribs) > thresh)[0][0]]
    # Reducing the cutoff if it's greater than 50% of the logit
    if cutoff > contrib.sum() / 2:
        cutoff = contrib.sum() / 2
    upstream_contributors = np.where(contrib >= cutoff)
    upstream_contributors = [
        (int(layer), int(ah_idx), int(token))
        for layer, ah_idx, token in zip(
            upstream_contributors[0],
            upstream_contributors[1],
            upstream_contributors[2],
        )
    ]
    return upstream_contributors


@typechecked
def _compute_residual_shares(
    model: HookedTransformer,
    config: ModelConfig,
    cache: ActivationCache,
    prompt_idx: int,
    end_token_pos: int,
    device: str,
) -> Float[Tensor, "n_layers n_components n_tokens d_model"]:
    """Compute the post-LN residual share of every upstream component.

    Decomposes the residual stream at the output token position into upstream
    component contributions and divides by the final LayerNorm scale (frozen
    from the clean forward pass). The result ``r_c[layer, ah_idx, src_token]``
    is the share that component ``(layer, ah_idx, src_token)`` contributes to
    ``ln_final.hook_normalized`` at ``end_token_pos``; the shares sum to it
    (up to LN centering, which is folded into the weights for LNPre models).

    Component index convention (axis 1, size ``n_heads + 3``):
    ``0..n_heads-1`` = attention heads (per source token), ``n_heads`` = MLP,
    ``n_heads + 1`` = AH bias (b_O), ``n_heads + 2`` = embedding.
    MLP / AH bias / embedding live at ``src_token == end_token_pos``.

    Args:
        model: HookedTransformer model.
        config: Model configuration.
        cache: Activation cache from forward pass.
        prompt_idx: Index of the prompt in the cache batch.
        end_token_pos: Position of the output token.
        device: Torch device.

    Returns:
        Tensor of shape (n_layers, n_heads + 3, n_tokens, d_model) with the
        post-LN share of each component at the output position.
    """
    n_tokens = end_token_pos + 1

    # Breaking down the OV for all AHs in the model
    upstream_output_breakdown = torch.zeros(
        (
            model.cfg.n_layers,
            model.cfg.n_heads + 3,
            n_tokens,
            n_tokens,
            model.cfg.d_model,
        ),
        device=device,
    )
    # Embedding
    upstream_output_breakdown[0, -1, end_token_pos, end_token_pos] = (
        cache["blocks.0.hook_resid_pre"][prompt_idx, end_token_pos].clone().detach()
    )
    for upstream_layer in range(model.cfg.n_layers):
        for upstream_ah_idx in range(model.cfg.n_heads + 3):
            if upstream_ah_idx < model.cfg.n_heads:  # AHs
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][
                    prompt_idx, upstream_ah_idx, :n_tokens, :n_tokens
                ]
                if config.has_gqa:
                    V = cache[f"blocks.{upstream_layer}.attn.hook_v"][
                        prompt_idx,
                        :n_tokens,
                        upstream_ah_idx // config.gqa_repeats,
                        :,
                    ]
                else:
                    V = cache[f"blocks.{upstream_layer}.attn.hook_v"][
                        prompt_idx, :n_tokens, upstream_ah_idx, :
                    ]
                upstream_output_breakdown[upstream_layer, upstream_ah_idx] = (
                    torch.einsum("ti,ij->tij", A, V)
                    @ model.W_O[upstream_layer, upstream_ah_idx, :, :]
                )

                if config.has_post_attn_ln:
                    upstream_output_breakdown[upstream_layer, upstream_ah_idx] *= (
                        model.blocks[upstream_layer].ln1_post.w.detach()
                    )
                    ln_post_term = cache[
                        f"blocks.{upstream_layer}.ln1_post.hook_scale"
                    ][prompt_idx, :n_tokens]
                    upstream_output_breakdown[upstream_layer, upstream_ah_idx] /= (
                        ln_post_term.view(ln_post_term.shape[0], 1, 1)
                    )

            # For all these cases, both dest and src tokens are the same
            elif upstream_ah_idx == model.cfg.n_heads:  # MLP
                upstream_output_breakdown[
                    upstream_layer, upstream_ah_idx, end_token_pos, end_token_pos
                ] = (
                    cache[f"blocks.{upstream_layer}.hook_mlp_out"][
                        prompt_idx, end_token_pos
                    ].clone().detach()
                )
            elif upstream_ah_idx == model.cfg.n_heads + 1:  # AH bias
                upstream_output_breakdown[
                    upstream_layer, upstream_ah_idx, end_token_pos, end_token_pos
                ] = model.b_O[upstream_layer].clone().detach()

    return (
        upstream_output_breakdown[:, :, end_token_pos, :, :]
        / cache["ln_final.hook_scale"][prompt_idx, end_token_pos]
    )


@typechecked
def get_seeds(
    model: HookedTransformer,
    config: ModelConfig,
    cache: ActivationCache,
    prompt_idx: int,
    logit_direction: Float[Tensor, "d_model"],
    end_token_pos: int,
    device: str,
) -> tuple[list[tuple], dict[tuple, float]]:
    """Identify seed components for circuit tracing (linear seeding).

    Decomposes the residual stream at the output token position into upstream
    component contributions, projects onto the logit direction, and selects
    the top contributors as seeds for recursive tracing.

    Args:
        model: HookedTransformer model.
        config: Model configuration.
        cache: Activation cache from forward pass.
        prompt_idx: Index of the prompt in the cache batch.
        logit_direction: Direction vector in residual stream space
            (e.g., W_U[:, IO] - W_U[:, S]).
        end_token_pos: Position of the output token.
        device: Torch device.

    Returns:
        Tuple of (trace_seeds, seeds_contrib) where trace_seeds is a list of
        (layer, ah_idx, token) tuples and seeds_contrib maps each seed to its
        contribution value.
    """

    r_c = _compute_residual_shares(
        model, config, cache, prompt_idx, end_token_pos, device
    )
    contrib_end_f_W_U_tensor = r_c @ logit_direction

    if contrib_end_f_W_U_tensor.sum() > 0:
        trace_seeds = get_upstream_contributors_seed(
            contrib_end_f_W_U_tensor.detach().cpu().numpy(), 1.0
        )
        seeds_contrib = {
            seed: contrib_end_f_W_U_tensor[seed].item() for seed in trace_seeds
        }
    else:
        # The logit difference is coming from b_U. Don't trace these cases.
        trace_seeds = []
        seeds_contrib = {}

    return trace_seeds, seeds_contrib


# ---------------------------------------------------------------------------
# Probability-aware seeding (v0.3.0)
#
# Implements the destruction-counterfactual seeding objective: find a minimal
# set of upstream components whose removal from the final residual stream
# drops the model's log-probability of a target-token support T by at least a
# fraction tau of the total achievable drop (the "completeness", measured
# against the bias-only baseline z' = b_U). Probabilities are FULL-VOCABULARY
# softmax probabilities — the support is never renormalized.
#
# Internal convention: functions work with the bias-free logit contribution
# z_clean = ln_final.hook_normalized @ W_U, and add b_U at evaluation time
# (z' = z_clean + b_U). This matches the decomposition identity
# z_clean = sum_c r_c @ W_U exactly, and removal of a component subtracts its
# logit lens vector from z_clean while b_U stays fixed.
# ---------------------------------------------------------------------------

def _J_support(
    z_logits: Float[Tensor, "d_vocab"],
    support: Tensor,
    q_star: Tensor,
) -> float:
    """Weighted full-vocab log-likelihood J(z') = sum_t q*_t log softmax(z')_t.

    Args:
        z_logits: Full logits (bias included), shape (d_vocab,).
        support: Target token ids, shape (n_support,), dtype long.
        q_star: Frozen weights over the support, shape (n_support,),
            non-negative and summing to 1.

    Returns:
        The scalar objective value (float).
    """
    log_p_full = torch.log_softmax(z_logits, dim=-1)
    return (q_star * log_p_full[support]).sum().item()


def _ig_attribution_seeds(
    r_c: Float[Tensor, "n_layers n_components n_tokens d_model"],
    z_clean: Float[Tensor, "d_vocab"],
    b_U: Float[Tensor, "d_vocab"],
    support: Tensor,
    q_star: Tensor,
    W_U: Float[Tensor, "d_model d_vocab"],
    ig_steps: int,
) -> Float[Tensor, "n_layers n_components n_tokens"]:
    """Integrated-gradients attribution of J over the residual decomposition.

    Integrates the gradient of J(z') = sum_t q*_t log softmax(z')_t along the
    straight path z'(alpha) = b_U + alpha * z_clean (from the bias-only
    baseline to the clean logits) and contracts it with each component's
    residual share. By IG completeness, the attributions sum to
    J(z_clean + b_U) - J(b_U) up to quadrature error.

    The gradient is mapped to d_model space first —
    grad_d(alpha) = W_U[:, support] @ q* - W_U @ p(alpha) — so the integrand
    is a (candidates x d_model) @ (d_model x steps) contraction and no
    (candidates x d_vocab) tensor is ever materialized.

    Args:
        r_c: Post-LN residual shares from :func:`_compute_residual_shares`.
        z_clean: Bias-free logit contribution, ln_final.hook_normalized @ W_U.
        b_U: Unembedding bias, shape (d_vocab,).
        support: Target token ids, shape (n_support,), dtype long.
        q_star: Frozen weights over the support (sum to 1).
        W_U: Unembedding matrix, shape (d_model, d_vocab).
        ig_steps: Number of trapezoidal quadrature intervals.

    Returns:
        Attribution tensor A_c of shape (n_layers, n_components, n_tokens).
    """
    device = r_c.device
    alphas = torch.linspace(0.0, 1.0, ig_steps + 1, device=device)
    z_path = b_U.unsqueeze(0) + alphas.unsqueeze(1) * z_clean.unsqueeze(0)  # (S+1, V)
    p_path = torch.softmax(z_path, dim=-1)                                  # (S+1, V)

    # Constant in alpha: W_U[:, support] @ q*. For |support| == 1 with
    # q* = [1.0] this is exactly W_U[:, t].
    W_U_T_q = W_U[:, support] @ q_star                                      # (d,)
    grad_d = W_U_T_q.unsqueeze(0) - p_path @ W_U.T                          # (S+1, d)

    integrand = r_c @ grad_d.T                                              # (L, H+3, NT, S+1)
    w = torch.ones(ig_steps + 1, device=device) / ig_steps
    w[0] = 0.5 / ig_steps
    w[-1] = 0.5 / ig_steps
    return (integrand * w).sum(dim=-1)                                      # (L, H+3, NT)


def _select_seeds_prob(
    A_c: Float[Tensor, "n_layers n_components n_tokens"],
    r_c: Float[Tensor, "n_layers n_components n_tokens d_model"],
    z_clean: Float[Tensor, "d_vocab"],
    b_U: Float[Tensor, "d_vocab"],
    W_U: Float[Tensor, "d_model d_vocab"],
    support: Tensor,
    q_star: Tensor,
    tau: float,
) -> tuple[list[tuple], dict[tuple, float]]:
    """Greedily select seeds until the removal drop reaches tau * completeness.

    Candidates are ranked once by IG attribution (descending). Each selected
    candidate's logit-lens vector r_c @ W_U is subtracted from the running
    logits and the exact objective J is recomputed; selection stops when
    J(z) - J(z^{-M}) >= tau * C, or at the first non-positive attribution
    (components that help the prediction are never removed "for coverage").

    Args:
        A_c: IG attributions from :func:`_ig_attribution_seeds`.
        r_c: Post-LN residual shares.
        z_clean: Bias-free logit contribution at the output position.
        b_U: Unembedding bias.
        W_U: Unembedding matrix.
        support: Target token ids (long tensor).
        q_star: Frozen weights over the support.
        tau: Fraction of the completeness C = J(z) - J(b_U) to destroy.

    Returns:
        Tuple of (trace_seeds, seeds_contrib): list of (layer, ah_idx, token)
        tuples for ALL selected components (AHs, MLP, AH bias, embedding) and
        a dict mapping each seed to its IG attribution A_c (the root-edge
        weight).
    """
    n_layers, n_components, n_tokens = A_c.shape
    flat_A = A_c.reshape(-1)
    order = torch.argsort(flat_A, descending=True)

    J_clean = _J_support(z_clean + b_U, support, q_star)
    completeness = J_clean - _J_support(b_U, support, q_star)
    target = tau * completeness

    z_running = z_clean.clone()
    selected: list[tuple] = []
    drop = 0.0
    reached = False
    for k in range(order.shape[0]):
        idx = int(order[k].item())
        if flat_A[idx].item() <= 0:
            break
        layer = idx // (n_components * n_tokens)
        ah_idx = (idx // n_tokens) % n_components
        src_token = idx % n_tokens
        z_running = z_running - r_c[layer, ah_idx, src_token] @ W_U
        selected.append((layer, ah_idx, src_token))
        drop = J_clean - _J_support(z_running + b_U, support, q_star)
        if drop >= target:
            reached = True
            break

    if not reached:
        warnings.warn(
            f"Probability-aware seeding exhausted all positive-attribution "
            f"candidates before reaching tau * completeness "
            f"({tau} * {completeness:.4f} = {target:.4f} nats); achieved "
            f"drop = {drop:.4f} nats with {len(selected)} seeds. Returning "
            f"the selected set.",
            UserWarning,
        )

    seeds_contrib = {seed: A_c[seed].item() for seed in selected}
    return selected, seeds_contrib


@typechecked
def get_seeds_prob(
    model: HookedTransformer,
    config: ModelConfig,
    cache: ActivationCache,
    prompt_idx: int,
    target_tokens: list[int],
    end_token_pos: int,
    device: str,
    *,
    q_star: Float[Tensor, "n_support"] | None = None,
    tau: float = 0.8,
    ig_steps: int = 64,
) -> tuple[list[tuple], dict[tuple, float]]:
    """Identify seed components via the probability-aware objective.

    Decomposes the residual stream at the output position into upstream
    component shares (same decomposition as :func:`get_seeds`), scores each
    component by integrated gradients of the weighted full-vocabulary
    log-likelihood J(z') = sum_{t in T} q*_t log softmax(z')_t along the path
    from the bias-only baseline b_U to the clean logits, and greedily selects
    components until removing them drops J by at least ``tau`` of the
    completeness C = J(z) - J(b_U).

    NOTE (final logit soft-cap): the objective is built from uncapped logits
    ``ln_final.hook_normalized @ W_U + b_U``. For models with a final logit
    soft-cap (e.g. Gemma-2, ``output_logits_soft_cap = 30``) these differ
    from the sampling logits. The cap is pointwise monotone increasing, so
    the token *probability ranking* is unchanged, but probability values —
    and therefore IG attribution values, and possibly the candidate ranking —
    are computed on the uncapped distribution. A ``UserWarning`` is emitted
    for such models.

    Args:
        model: HookedTransformer model.
        config: Model configuration.
        cache: Activation cache from forward pass.
        prompt_idx: Index of the prompt in the cache batch.
        target_tokens: Token ids forming the support T (no duplicates). The
            order only fixes the pairing with ``q_star``.
        end_token_pos: Position of the output token.
        device: Torch device.
        q_star: Optional frozen weights over the support (non-negative,
            summing to 1). Default: the clean full-vocabulary probabilities
            of the support tokens, renormalized within the support.
        tau: Fraction of the completeness to destroy. Default 0.8.
        ig_steps: Trapezoidal quadrature intervals for IG. Default 64.

    Returns:
        Tuple of (trace_seeds, seeds_contrib) with the same contract as
        :func:`get_seeds`: a list of (layer, ah_idx, token) tuples and a dict
        mapping each seed to its weight (here: the IG attribution A_c).
    """
    if len(target_tokens) == 0:
        raise ValueError("target_tokens must contain at least one token id.")
    if len(set(target_tokens)) != len(target_tokens):
        raise ValueError(f"target_tokens contains duplicates: {target_tokens}")

    soft_cap = getattr(model.cfg, "output_logits_soft_cap", 0.0) or 0.0
    if soft_cap > 0:
        warnings.warn(
            f"Model has a final logit soft-cap ({soft_cap}); probability-"
            f"aware seeding uses UNCAPPED logits. Token probability ranking "
            f"is unaffected (the cap is monotone), but probabilities and IG "
            f"attribution values are computed on the uncapped distribution.",
            UserWarning,
        )

    W_U = model.W_U.detach()
    b_U = model.b_U.detach()
    support = torch.as_tensor(target_tokens, dtype=torch.long, device=device)

    r_c = _compute_residual_shares(
        model, config, cache, prompt_idx, end_token_pos, device
    )
    z_clean = (
        cache["ln_final.hook_normalized"][prompt_idx, end_token_pos] @ W_U
    )

    if q_star is None:
        p_clean_T = torch.softmax(z_clean + b_U, dim=-1)[support]
        q_star = p_clean_T / p_clean_T.sum()
    else:
        q_star = q_star.to(device)
        if q_star.shape[0] != support.shape[0]:
            raise ValueError(
                f"q_star has length {q_star.shape[0]} but target_tokens has "
                f"length {support.shape[0]}."
            )
        if (q_star < 0).any():
            raise ValueError("q_star must be non-negative.")
        if abs(q_star.sum().item() - 1.0) > 1e-4:
            raise ValueError(
                f"q_star must sum to 1 (got {q_star.sum().item():.6f})."
            )

    # Guard: completeness must be positive — otherwise the support is no more
    # likely under the model than under the bias-only baseline, and there is
    # nothing to destroy (analogue of get_seeds' "logit diff coming from b_U"
    # guard).
    completeness = (
        _J_support(z_clean + b_U, support, q_star)
        - _J_support(b_U, support, q_star)
    )
    if completeness <= 0:
        warnings.warn(
            f"Completeness J(z) - J(b_U) = {completeness:.4f} <= 0 for "
            f"target_tokens={target_tokens}; no seeds returned.",
            UserWarning,
        )
        return [], {}

    A_c = _ig_attribution_seeds(
        r_c, z_clean, b_U, support, q_star, W_U, ig_steps
    )
    return _select_seeds_prob(
        A_c, r_c, z_clean, b_U, W_U, support, q_star, tau
    )


class Tracer:
    """ACC++ circuit tracer.

    Precomputes expensive model-level quantities (Omega SVD, weight pseudoinverses)
    once at initialization, then reuses them across all trace calls.

    Args:
        model: A HookedTransformer model instance.
        device: Torch device. If None, uses model's device.
        use_numpy_svd: Use numpy for SVD (more stable for some models
            like Pythia). Default: False.
        dynamic_threshold_scale: Numerator for the "dynamic" attention weight
            threshold formula: min(1.0, scale / (dest_token + 1)). Default: 2.5.
        cache_dir: Optional path to a directory used as a disk cache for the
            Omega SVD (``U, S, VT``) and weight pseudoinverses (``W_Q_pinv,
            W_K_pinv``). When set, the constructor first tries to load
            ``{cache_dir}/{model_name}_{torch|numpy}.h5``; on miss or failure
            it computes the tensors from scratch and writes them to disk in
            gzip-compressed h5 format for subsequent runs. When ``None``
            (default) the constructor always recomputes. Bias offsets
            ``c_d`` / ``c_s`` are not cached (recomputed cheaply from
            ``model.b_Q`` / ``model.b_K`` and the pseudoinverses).

    Example:
        >>> from accpp_tracer import Tracer
        >>> from transformer_lens import HookedTransformer
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> tracer = Tracer(model)
        >>> # Probability-aware seeding (default): support T = {" Mary"}
        >>> graph = tracer.trace(
        ...     "When Mary and John went to the store, John gave a drink to",
        ...     answer_token=" Mary",
        ... )
        >>> # Multi-token support T = {" Mary", " John"} — still one root node
        >>> graph = tracer.trace(
        ...     "When Mary and John went to the store, John gave a drink to",
        ...     answer_token=[" Mary", " John"],
        ... )
        >>> # Pre-0.3.0 contrastive logit-diff tracing (paper reproduction)
        >>> graph = tracer.trace(
        ...     "When Mary and John went to the store, John gave a drink to",
        ...     answer_token=" Mary",
        ...     wrong_token=" John",
        ...     seeding="linear",
        ... )
    """

    def __init__(
        self,
        model: HookedTransformer,
        device: str | None = None,
        use_numpy_svd: bool = False,
        dynamic_threshold_scale: float = 2.5,
        cache_dir: str | None = None,
    ) -> None:
        self.model = model
        self.device = device or str(model.cfg.device)
        self.config = get_model_config(model, use_numpy_svd=use_numpy_svd)
        self.dynamic_threshold_scale = dynamic_threshold_scale

        # Disable TF32 on CUDA. Ampere+ GPUs use TF32 by default for
        # matmul (10 mantissa bits vs 23 for fp32), causing accumulated rounding
        # errors across the many einsum calls in _trace_firing_inner that push
        # the decomposition-sum beyond the atol=1e-3 correctness assertion.
        # Full fp32 is required here for numerical equivalence with CPU/MPS.
        if "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # Optionally load precomputed Omega SVD + pseudoinverses from
        # disk cache; recompute on cache-miss or load failure. When recomputed
        # and cache_dir is set, write the result back to disk for future runs.
        cached = None
        if cache_dir is not None:
            cached = load_decomposition_cache(
                cache_dir, model, use_numpy_svd, self.device
            )

        if cached is not None:
            self.U = cached["U"]
            self.S = cached["S"]
            self.VT = cached["VT"]
            self.W_Q_pinv = cached["W_Q_pinv"]
            self.W_K_pinv = cached["W_K_pinv"]
        else:
            self.U, self.S, self.VT = get_omega_decomposition(
                model, self.config, self.device
            )
            self.W_Q_pinv, self.W_K_pinv = compute_weight_pseudoinverses(
                model, self.config, self.device
            )
            if cache_dir is not None:
                save_decomposition_cache(
                    cache_dir,
                    self.U, self.S, self.VT,
                    self.W_Q_pinv, self.W_K_pinv,
                    model_name=model.cfg.model_name,
                    use_numpy_svd=use_numpy_svd,
                )

        # Precompute bias offsets c_d and c_s (used by trace_firing and
        # extract_edge_signal for the AH offset component). Previously
        # recomputed on every trace_firing call.
        # Shape: (n_layers, n_heads, d_model). For models without bias
        # (e.g. Gemma), b_Q and b_K are zeros → c_d and c_s are zeros.
        self.c_d = einsum(
            model.b_Q, self.W_Q_pinv,
            "n_layers n_heads d_head, n_layers n_heads d_head d_model "
            "-> n_layers n_heads d_model",
        )
        self.c_s = einsum(
            model.b_K, self.W_K_pinv,
            "n_layers n_heads d_head, n_layers n_heads d_head d_model "
            "-> n_layers n_heads d_model",
        )

    def trace(
        self,
        prompt: str,
        answer_token: str | int | list[str | int] | None = None,
        wrong_token: str | int | None = None,
        top_p: float | None = None,
        attn_weight_thresh: str | float | Callable[[int], float] = "dynamic",
        signals: Literal[None, "rotated_normalized", "normalized", "raw"] = None,
        prepend_bos: bool | None = None,
        seeding: Literal["prob", "linear"] = "prob",
        top_k: int | None = None,
        tau: float | None = None,
        ig_steps: int | None = None,
    ) -> nx.MultiDiGraph:
        """Trace a single prompt (Level 3 — simplest API).

        Handles tokenization, forward pass, token mapping, seed
        identification, and recursive circuit tracing.

        Two seeding modes:

        - ``seeding="prob"`` (default, v0.3.0): probability-aware seeding.
          A support T of target tokens is chosen via exactly one of
          ``answer_token`` / ``top_p`` / ``top_k``, and seeds are the minimal
          component set whose removal destroys a ``tau`` fraction of the
          model's log-likelihood of T (full-vocabulary probabilities,
          bias-only baseline). The graph always has ONE root node.
          ``wrong_token`` is not supported here (a contrastive log-probability
          objective is mathematically identical to the linear logit-diff
          direction — use ``seeding="linear"`` for that).
        - ``seeding="linear"``: the pre-0.3.0 behavior, kept for paper
          reproduction. Each answer token (or each top-p token) becomes its
          own logit direction and root node; seeds are selected by linear
          logit attribution.

        Args:
            prompt: Input text string.
            answer_token: Correct next token (str, int, or list of str/int).
                In prob mode a list defines the support T of one objective
                (one root node); in linear mode each element becomes its own
                logit direction and root node.
            wrong_token: Contrastive token — linear mode only. Each direction
                becomes W_U[:, answer_i] - W_U[:, wrong]. Raises in prob mode.
            top_p: Nucleus selection: the minimum set of top tokens whose
                cumulative probability >= top_p (computed from the model's
                clean output distribution, frozen). In prob mode the selected
                tokens form the support T of ONE objective (one root node);
                in linear mode each becomes its own direction and root.
            top_k: Top-k selection of the support T (prob mode only): the k
                most likely tokens of the clean output distribution, frozen.
            tau: Prob mode only — fraction of the completeness to destroy.
                Default 0.8.
            ig_steps: Prob mode only — trapezoidal quadrature intervals for
                the integrated-gradients attribution. Default 64.
            attn_weight_thresh: "dynamic" (= scale/context_size, where scale is
                dynamic_threshold_scale from __init__), a float in [0, 1], or a
                callable that takes dest_token position (int) and returns a float.
            signals: If non-None, compute and store the **primary** signal tensor
                (detached, CPU) on each non-seed edge during tracing, in the
                requested flavor. The signal of a dest edge is ``signal_u``; the
                signal of a src edge is ``signal_v``. Each edge also gains a
                ``"signal_flavor"`` attribute matching this value. Default:
                ``None`` (no signals stored). One of:

                  - ``"rotated_normalized"`` — analysis / autointerp flavor (QK frame)
                  - ``"normalized"`` — intervention flavor (LN-normalized, unrotated)
                  - ``"raw"`` — residual-stream-space flavor (MLP upstream tracing)

                See :meth:`extract_edge_signal` for the full flavor semantics.
            prepend_bos: Whether to prepend BOS token. None (default) uses
                TransformerLens's model default. True/False overrides explicitly.

        Returns:
            nx.MultiDiGraph — the traced circuit graph.
        """
        model = self.model

        # Tokenize
        if prepend_bos is not None:
            tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)  # shape: (1, seq_len)
        else:
            tokens = model.to_tokens(prompt)  # shape: (1, seq_len)
        end_token_pos = tokens.shape[1] - 1

        # Forward pass — logits needed for top_p
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        # Build idx_to_token from actual tokens (with duplicate handling)
        idx_to_token: dict[int, str] = {}
        count_dict: dict[str, int] = defaultdict(int)
        for i in range(end_token_pos + 1):
            tok_str = model.tokenizer.decode(tokens[0, i])
            count = count_dict[tok_str]
            if count > 0:
                idx_to_token[i] = f"{tok_str} ({count})"
            else:
                idx_to_token[i] = tok_str
            count_dict[tok_str] += 1

        if seeding == "prob":
            if wrong_token is not None:
                raise ValueError(
                    "wrong_token is not supported with seeding='prob': a "
                    "contrastive log-probability objective is identical to "
                    "the linear logit-diff direction. Use seeding='linear' "
                    "for contrastive tracing."
                )
            n_provided = sum(
                x is not None for x in (answer_token, top_p, top_k)
            )
            if n_provided != 1:
                raise ValueError(
                    "With seeding='prob', provide exactly one of "
                    "answer_token, top_p, or top_k "
                    f"(got {n_provided})."
                )

            if answer_token is not None:
                if isinstance(answer_token, list):
                    token_ids = [
                        model.to_single_token(t) if isinstance(t, str) else t
                        for t in answer_token
                    ]
                else:
                    token_ids = [
                        model.to_single_token(answer_token)
                        if isinstance(answer_token, str)
                        else answer_token
                    ]
            else:
                # Support from the model's clean output distribution (frozen).
                probs = torch.softmax(logits[0, end_token_pos], dim=-1)
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(
                        probs, descending=True
                    )
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    n_selected = int(
                        (torch.where(cumsum >= top_p)[0][0] + 1).item()
                    )
                    token_ids = sorted_indices[:n_selected].tolist()
                else:
                    token_ids = torch.topk(probs, top_k).indices.tolist()

            # One root node always (one objective over the support T).
            # Labelled "Prob" (not "Logit"): the prob-seeding objective is
            # the (log-)probability of the support, not a logit direction.
            tok_strs = [model.tokenizer.decode(t) for t in token_ids]
            if len(token_ids) == 1:
                root_label = f"Prob '{tok_strs[0]}'"
            else:
                root_label = "Prob {" + ", ".join(
                    f"'{t}'" for t in tok_strs
                ) + "}"
            root_node = (root_label, idx_to_token[end_token_pos])

            return self.trace_from_cache(
                cache=cache,
                logit_direction=None,
                end_token_pos=end_token_pos,
                idx_to_token=idx_to_token,
                root_node=root_node,
                prompt_idx=0,
                attn_weight_thresh=attn_weight_thresh,
                signals=signals,
                target_tokens=token_ids,
                tau=tau,
                ig_steps=ig_steps,
            )

        if seeding != "linear":
            raise ValueError(
                f"Unknown seeding mode: {seeding!r}. "
                "Must be 'prob' or 'linear'."
            )
        
        if top_k is not None or tau is not None or ig_steps is not None:
            raise ValueError(
                "top_k, tau, and ig_steps are only valid with "
                "seeding='prob'."
            )

        # Resolve wrong_token once; applied to every direction
        if wrong_token is not None and isinstance(wrong_token, str):
            wrong_token = model.to_single_token(wrong_token)
        wrong_dir = model.W_U[:, wrong_token].clone() if wrong_token is not None else None

        # Determine which token IDs to trace
        if top_p is None and answer_token is None:
            raise ValueError("Either answer_token or top_p must be provided.")

        if top_p is not None:
            # Select minimum set of top tokens covering top_p probability mass
            probs = torch.softmax(logits[0, end_token_pos], dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            n_selected = int((torch.where(cumsum >= top_p)[0][0] + 1).item())
            token_ids: list[int] = sorted_indices[:n_selected].tolist()
            multi = True
        elif isinstance(answer_token, list):
            # Explicit list of tokens — resolve strings to ints
            token_ids = [
                model.to_single_token(t) if isinstance(t, str) else t
                for t in answer_token
            ]
            multi = True
        else:
            # Single token — backward-compatible path
            if isinstance(answer_token, str):
                answer_token = model.to_single_token(answer_token)
            token_ids = [answer_token]
            multi = False

        # Build one logit direction and root node per selected token
        logit_directions: list[Tensor] = []
        root_nodes: list[tuple] = []
        for tok_id in token_ids:
            direction = model.W_U[:, tok_id].clone()
            if wrong_dir is not None:
                direction = direction - wrong_dir
            logit_directions.append(direction)
            if multi:
                tok_str = model.tokenizer.decode(tok_id)
                root_nodes.append((f"Logit '{tok_str}'", idx_to_token[end_token_pos]))
            else:
                # Preserve backward-compatible root node label for single-token callers
                root_nodes.append(("Logit direction", idx_to_token[end_token_pos]))

        return self.trace_from_cache(
            cache=cache,
            logit_direction=logit_directions[0] if not multi else logit_directions,
            end_token_pos=end_token_pos,
            idx_to_token=idx_to_token,
            root_node=root_nodes[0] if not multi else root_nodes,
            prompt_idx=0,
            attn_weight_thresh=attn_weight_thresh,
            signals=signals,
        )

    def trace_from_cache(
        self,
        cache: ActivationCache,
        logit_direction: Tensor | list[Tensor] | None,
        end_token_pos: int,
        idx_to_token: dict[int, str],
        root_node: tuple | list[tuple],
        prompt_idx: int = 0,
        attn_weight_thresh: str | float | Callable[[int], float] = "dynamic",
        signals: Literal[None, "rotated_normalized", "normalized", "raw"] = None,
        target_tokens: list[int] | None = None,
        q_star: Tensor | None = None,
        tau: float | None = None,
        ig_steps: int | None = None,
    ) -> nx.MultiDiGraph:
        """Trace from a pre-computed cache (Level 2 — advanced API).

        The user provides the cache, the tracing objective, token mapping, and
        root node(s). This is what paper reproduction scripts call in a loop.

        The seeding mode is selected by which objective argument is given —
        exactly one of ``logit_direction`` (linear seeding) or
        ``target_tokens`` (probability-aware seeding) must be non-None:

        - **Linear**: pass ``logit_direction`` (single Tensor or list) and a
          matching ``root_node`` (tuple or list). Multiple directions are
          traced as separate root nodes in the same graph; the is_traced state
          is shared, so attention head subgraphs common to multiple directions
          are traced only once. Identical to the pre-0.3.0 behavior.
        - **Probability-aware**: pass ``target_tokens`` (the support T, token
          ids), a single ``root_node`` tuple, and ``logit_direction=None``.
          The clean logits are recomputed from the cache
          (``ln_final.hook_normalized @ W_U + b_U``) — no logits argument is
          needed. The graph has one root node.

        Args:
            cache: ActivationCache from model.run_with_cache().
            logit_direction: Single direction tensor or list of direction tensors
                in residual stream space (e.g., W_U[:, IO] - W_U[:, S]). Must be
                None when target_tokens is given.
            end_token_pos: Position of the output token.
            idx_to_token: Dict mapping token position (int) to label (str).
            root_node: Single tuple or list of tuples (one per direction) for
                the root/output node label(s) in the graph. With target_tokens,
                a single tuple.
            prompt_idx: Index of this prompt in the cache batch.
            attn_weight_thresh: "dynamic" (= scale/context_size), a float in
                [0, 1], or a callable taking dest_token position → float.
            target_tokens: Support T for probability-aware seeding (token ids,
                no duplicates). Mutually exclusive with logit_direction.
            q_star: Optional frozen weights over target_tokens (non-negative,
                summing to 1). Default: clean probabilities renormalized
                within the support. Prob mode only.
            tau: Prob mode only — fraction of the completeness to destroy.
                Default 0.8.
            ig_steps: Prob mode only — IG quadrature intervals. Default 64.
            signals: If non-None, compute and store the **primary** signal tensor
                (detached, CPU) on each non-seed edge during tracing, in the
                requested flavor. The signal of a dest edge is ``signal_u``; the
                signal of a src edge is ``signal_v``. Each edge also gains a
                ``"signal_flavor"`` attribute matching this value. One of
                ``"rotated_normalized" / "normalized" / "raw"``. Default:
                ``None`` (no signals stored). See :meth:`extract_edge_signal` for
                the full flavor semantics.

        Returns:
            nx.MultiDiGraph — the traced circuit graph. Empty if no direction
            produces seeds with at least one attention head.
        """
        model = self.model

        if (logit_direction is None) == (target_tokens is None):
            raise ValueError(
                "Provide exactly one of logit_direction (linear seeding) or "
                "target_tokens (probability-aware seeding)."
            )
        if target_tokens is None and (
            q_star is not None or tau is not None or ig_steps is not None
        ):
            raise ValueError(
                "q_star, tau, and ig_steps are only valid with "
                "target_tokens (probability-aware seeding)."
            )
        if target_tokens is not None and not isinstance(root_node, tuple):
            raise ValueError(
                "With target_tokens, root_node must be a single tuple "
                "(probability-aware seeding always has one root node)."
            )

        dirs = [logit_direction] if isinstance(logit_direction, Tensor) else logit_direction
        roots = [root_node] if isinstance(root_node, tuple) else root_node

        return self._trace_from_cache_inner(
            model, cache, dirs, roots, prompt_idx, end_token_pos,
            idx_to_token, attn_weight_thresh, signals,
            target_tokens=target_tokens,
            q_star=q_star,
            tau=0.8 if tau is None else tau,
            ig_steps=64 if ig_steps is None else ig_steps,
        )

    @torch.no_grad()
    def trace_from_probe(
        self,
        cache: ActivationCache,
        probe_direction: Tensor,
        layer: int,
        end_token_pos: int,
        idx_to_token: dict[int, str],
        root_node: tuple,
        prompt_idx: int = 0,
        attn_weight_thresh: str | float | Callable[[int], float] = "dynamic",
        signals: Literal[None, "rotated_normalized", "normalized", "raw"] = None,
    ) -> nx.MultiDiGraph:
        """Trace the circuit that writes a linear probe direction read at an
        intermediate layer (no logit / vocabulary objective).

        Mirrors :meth:`trace_from_cache`, but the seed step decomposes the
        residual stream at ``blocks.{layer}.hook_resid_post`` (the ``end_token_pos``
        position) into per-component shares from layers ``0..layer`` only,
        projects them onto ``probe_direction``, and selects seeds. From there the
        upstream recursion is identical to the logit case (``_trace_recursive``).

        ``probe_direction`` must already be oriented toward the side to trace
        (e.g. +w toward "true", -w toward "false"); orientation and any
        mean-centering are the caller's responsibility.
        """
        model = self.model

        # --- Step 1: per-component contributions to the probe direction ---
        # We just call the _compute_residual_shares and slice the result to 
        # layers 0..layer. Also, we undo the ln_final divide (logit
        # space) so contributions are measured in the probe's raw residual
        # space; it is a per-position scalar and never affects seed selection.

        # Decompose the residual stream at the end_token_pos into per-component shares
        r_c = _compute_residual_shares(
            model, self.config, cache, prompt_idx, end_token_pos, self.device
        )
        r_c = r_c[: layer + 1] # slice to layers 0..layer
        # undo ln_final divide
        r_c = r_c * cache["ln_final.hook_scale"][prompt_idx, end_token_pos]
        contrib = r_c @ probe_direction          # (layer+1, n_heads+3, n_tokens)

        # --- Step 2: select seeds (mirrors get_seeds' linear selection) ---
        if contrib.sum() > 0:
            trace_seeds = get_upstream_contributors_seed(
                contrib.detach().cpu().numpy(), 1.0
            )
            seeds_contrib = {seed: contrib[seed].item() for seed in trace_seeds}
        else:
            # No contribution toward the probe direction (analogue of
            # get_seeds' "coming from b_U" guard); nothing to trace.
            trace_seeds, seeds_contrib = [], {}

        # --- Step 3: build the graph (root edges + recurse into AH seeds) ---
        # Mirrors _trace_from_cache_inner: only build if at least one AH seed 
        # exists (leaf-only seeds carry no circuit to recurse).
        G = nx.MultiDiGraph()
        is_traced: dict[tuple, int] = {}
        if len(trace_seeds) > 0 and any(
            ah_idx < model.cfg.n_heads for _, ah_idx, _ in trace_seeds
        ):
            self._add_seeds_to_graph(
                G, is_traced, trace_seeds, seeds_contrib, root_node,
                cache, idx_to_token, prompt_idx, end_token_pos,
                attn_weight_thresh, signals,
            )
        return G

    @torch.no_grad()
    def _trace_from_cache_inner(
        self,
        model,
        cache,
        dirs,
        roots,
        prompt_idx,
        end_token_pos,
        idx_to_token,
        attn_weight_thresh,
        signals,
        target_tokens=None,
        q_star=None,
        tau=0.8,
        ig_steps=64,
    ):
        # Build circuit graph — shared across all directions
        G = nx.MultiDiGraph()
        is_traced: dict[tuple, int] = {}

        # CHANGED (v0.3.0): probability-aware seeding path — one objective,
        # one root node.
        if target_tokens is not None:
            trace_seeds, seeds_contrib = get_seeds_prob(
                model,
                self.config,
                cache,
                prompt_idx,
                target_tokens,
                end_token_pos,
                self.device,
                q_star=q_star,
                tau=tau,
                ig_steps=ig_steps,
            )
            # Same emptiness / AH-presence policy as the linear path.
            if len(trace_seeds) > 0 and any(
                ah_idx < model.cfg.n_heads for _, ah_idx, _ in trace_seeds
            ):
                self._add_seeds_to_graph(
                    G, is_traced, trace_seeds, seeds_contrib, roots[0],
                    cache, idx_to_token, prompt_idx, end_token_pos,
                    attn_weight_thresh, signals,
                )
            return G

        for direction, root in zip(dirs, roots):
            # Get seeds for this direction
            trace_seeds, seeds_contrib = get_seeds(
                model,
                self.config,
                cache,
                prompt_idx,
                direction,
                end_token_pos,
                self.device,
            )

            if len(trace_seeds) == 0:
                continue

            # Check if any AH seeds exist for this direction
            has_ah_seed = any(
                ah_idx < model.cfg.n_heads for _, ah_idx, _ in trace_seeds
            )
            if not has_ah_seed:
                continue

            self._add_seeds_to_graph(
                G, is_traced, trace_seeds, seeds_contrib, root,
                cache, idx_to_token, prompt_idx, end_token_pos,
                attn_weight_thresh, signals,
            )

        return G

    def _add_seeds_to_graph(
        self,
        G: nx.MultiDiGraph,
        is_traced: dict[tuple, int],
        trace_seeds: list[tuple],
        seeds_contrib: dict[tuple, float],
        root: tuple,
        cache: ActivationCache,
        idx_to_token: dict[int, str],
        prompt_idx: int,
        end_token_pos: int,
        attn_weight_thresh: str | float | Callable[[int], float],
        signals: Literal[None, "rotated_normalized", "normalized", "raw"],
    ) -> None:
        """Add root edges for the given seeds and recurse into AH seeds.

        Mutates ``G`` and ``is_traced`` in place. Every seed gets an edge to
        ``root`` weighted by its ``seeds_contrib`` value; attention-head seeds
        are additionally traced recursively upstream.
        """
        model = self.model
        for layer, ah_idx, src_token in trace_seeds:
            ah_idx_label = get_ah_idx_label(ah_idx, model.cfg.n_heads)

            # Add edge from seed to this direction's root node
            if end_token_pos in idx_to_token and src_token in idx_to_token:
                G.add_edge(
                    (
                        layer,
                        ah_idx_label,
                        idx_to_token[end_token_pos],
                        idx_to_token[src_token],
                    ),
                    root,
                    weight=seeds_contrib[(layer, ah_idx, src_token)],
                    type="d",
                    color="#E41A1C",
                )

            # Recursively trace upstream for AH seeds
            if ah_idx < model.cfg.n_heads:
                if (layer, ah_idx, end_token_pos, src_token) not in is_traced:
                    self._trace_recursive(
                        cache,
                        idx_to_token,
                        G,
                        is_traced,
                        prompt_idx,
                        layer,
                        ah_idx,
                        end_token_pos,
                        src_token,
                        attn_weight_thresh,
                        signals,
                    )

    def _trace_recursive(
        self,
        cache: ActivationCache,
        idx_to_token: dict[int, str],
        G: nx.MultiDiGraph,
        is_traced: dict[tuple, int],
        prompt_idx: int,
        layer: int,
        ah_idx: int,
        dest_token: int,
        src_token: int,
        attn_weight_thresh: str | float | Callable[[int], float],
        signals: Literal[None, "rotated_normalized", "normalized", "raw"] = None,
    ) -> None:
        """Recursively trace upstream contributions and build circuit graph.

        Args:
            cache: Activation cache.
            idx_to_token: Token position to label mapping.
            G: Graph being built (mutated in place).
            is_traced: Already-traced (layer, ah_idx, dest, src) tuples.
            prompt_idx: Index of this prompt in the cache batch.
            layer: Attention layer to trace.
            ah_idx: Attention head index.
            dest_token: Destination token position.
            src_token: Source token position.
            attn_weight_thresh: "dynamic", float, or callable(dest_token) -> float.
            signals: When non-None, attach the primary signal (in the requested
                flavor) to each non-seed edge under the key ``"signal"``, plus a
                ``"signal_flavor"`` attribute recording which flavor was stored.
        """
        is_traced[(layer, ah_idx, dest_token, src_token)] = 1

        if layer == 0 or dest_token == 0:
            return

        if src_token > dest_token:
            return

        if attn_weight_thresh == "dynamic":
            attn_weight_thresh_eval = min(
                1.0, self.dynamic_threshold_scale / (dest_token + 1)
            )
        elif callable(attn_weight_thresh):
            attn_weight_thresh_eval = attn_weight_thresh(dest_token)
        else:
            attn_weight_thresh_eval = float(attn_weight_thresh)

        assert 0.0 <= attn_weight_thresh_eval <= 1.0

        attn_weight = cache[f"blocks.{layer}.attn.hook_pattern"][
            prompt_idx, ah_idx, dest_token, src_token
        ].item()

        if attn_weight < attn_weight_thresh_eval:
            return

        svs_dest, edge_weights_dest, svs_src, edge_weights_src = trace_firing(
            self.model,
            cache,
            prompt_idx,
            layer,
            ah_idx,
            dest_token,
            src_token,
            self.U,
            self.S,
            self.VT,
            self.W_Q_pinv,
            self.W_K_pinv,
            self.config,
            self.device,
            attn_weight_thresh_eval,
        )

        ah_idx_label = get_ah_idx_label(ah_idx, self.model.cfg.n_heads)

        if dest_token not in idx_to_token or src_token not in idx_to_token:
            return
        node_downstream = (
            layer,
            ah_idx_label,
            idx_to_token[dest_token],
            idx_to_token[src_token],
        )

        # Store the attention weight on the AH firing node
        if node_downstream not in G:
            G.add_node(node_downstream)
        G.nodes[node_downstream]["attn_weight"] = attn_weight

        # Tracing dest
        for (
            upstream_layer,
            upstream_ah_idx,
            upstream_src_token,
        ) in svs_dest.keys():
            upstream_ah_idx_label = get_ah_idx_label(
                upstream_ah_idx, self.model.cfg.n_heads
            )

            if upstream_src_token > dest_token:
                continue

            if (
                dest_token in idx_to_token
                and upstream_src_token in idx_to_token
            ):
                node_upstream = (
                    upstream_layer,
                    upstream_ah_idx_label,
                    idx_to_token[dest_token],
                    idx_to_token[upstream_src_token],
                )
                svs_used = svs_dest[
                    (upstream_layer, upstream_ah_idx, upstream_src_token)
                ]

                G.add_edge(
                    node_upstream,
                    node_downstream,
                    weight=edge_weights_dest[
                        upstream_layer, upstream_ah_idx, upstream_src_token
                    ],
                    svs_used=str(svs_used),
                    type="d",
                    color="#E41A1C",
                )

                # Set attn_weight on upstream AH nodes (may be leaf nodes
                # that _trace_recursive won't visit, e.g. layer 0)
                if (
                    upstream_ah_idx < self.model.cfg.n_heads
                    and "attn_weight" not in G.nodes[node_upstream]
                ):
                    G.nodes[node_upstream]["attn_weight"] = cache[
                        f"blocks.{upstream_layer}.attn.hook_pattern"
                    ][
                        prompt_idx, upstream_ah_idx,
                        dest_token, upstream_src_token,
                    ].item()

                if signals is not None:
                    signal = self.extract_edge_signal(
                        cache, prompt_idx,
                        layer, ah_idx, dest_token, src_token,
                        upstream_layer, upstream_ah_idx,
                        dest_token, upstream_src_token,
                        edge_type="d", svs_used=svs_used,
                        flavor=signals,
                    )
                    key = G.number_of_edges(node_upstream, node_downstream) - 1
                    G.edges[node_upstream, node_downstream, key]["signal"] = signal.detach().cpu()
                    G.edges[node_upstream, node_downstream, key]["signal_flavor"] = signals

                if upstream_ah_idx < self.model.cfg.n_heads:
                    if (
                        upstream_layer,
                        upstream_ah_idx,
                        dest_token,
                        upstream_src_token,
                    ) not in is_traced:
                        self._trace_recursive(
                            cache,
                            idx_to_token,
                            G,
                            is_traced,
                            prompt_idx,
                            upstream_layer,
                            upstream_ah_idx,
                            dest_token,
                            upstream_src_token,
                            attn_weight_thresh,
                            signals,
                        )

        # Tracing src
        for (
            upstream_layer,
            upstream_ah_idx,
            upstream_src_token,
        ) in svs_src.keys():
            upstream_ah_idx_label = get_ah_idx_label(
                upstream_ah_idx, self.model.cfg.n_heads
            )

            if upstream_src_token > src_token:
                continue

            if (
                src_token in idx_to_token
                and upstream_src_token in idx_to_token
            ):
                node_upstream = (
                    upstream_layer,
                    upstream_ah_idx_label,
                    idx_to_token[src_token],
                    idx_to_token[upstream_src_token],
                )
                svs_used = svs_src[
                    (upstream_layer, upstream_ah_idx, upstream_src_token)
                ]

                G.add_edge(
                    node_upstream,
                    node_downstream,
                    weight=edge_weights_src[
                        upstream_layer, upstream_ah_idx, upstream_src_token
                    ],
                    svs_used=str(svs_used),
                    type="s",
                    color="#377EB8",
                )

                # Set attn_weight on upstream AH nodes (may be leaf nodes
                # that _trace_recursive won't visit, e.g. layer 0)
                if (
                    upstream_ah_idx < self.model.cfg.n_heads
                    and "attn_weight" not in G.nodes[node_upstream]
                ):
                    G.nodes[node_upstream]["attn_weight"] = cache[
                        f"blocks.{upstream_layer}.attn.hook_pattern"
                    ][
                        prompt_idx, upstream_ah_idx,
                        src_token, upstream_src_token,
                    ].item()

                if signals is not None:
                    signal = self.extract_edge_signal(
                        cache, prompt_idx,
                        layer, ah_idx, dest_token, src_token,
                        upstream_layer, upstream_ah_idx,
                        src_token, upstream_src_token,
                        edge_type="s", svs_used=svs_used,
                        flavor=signals,
                    )
                    key = G.number_of_edges(node_upstream, node_downstream) - 1
                    G.edges[node_upstream, node_downstream, key]["signal"] = signal.detach().cpu()
                    G.edges[node_upstream, node_downstream, key]["signal_flavor"] = signals

                if upstream_ah_idx < self.model.cfg.n_heads:
                    if (
                        upstream_layer,
                        upstream_ah_idx,
                        src_token,
                        upstream_src_token,
                    ) not in is_traced:
                        self._trace_recursive(
                            cache,
                            idx_to_token,
                            G,
                            is_traced,
                            prompt_idx,
                            upstream_layer,
                            upstream_ah_idx,
                            src_token,
                            upstream_src_token,
                            attn_weight_thresh,
                            signals,
                        )

    def extract_edge_signal(
        self,
        cache: ActivationCache,
        prompt_idx: int,
        downstream_layer: int,
        downstream_ah_idx: int,
        downstream_dest_token: int,
        downstream_src_token: int,
        upstream_layer: int,
        upstream_component_id: int,
        upstream_dest_token: int,
        upstream_src_token: int,
        edge_type: str,
        svs_used: list[int],
        *,
        flavor: Literal["rotated_normalized", "normalized", "raw"],
    ) -> Tensor:
        """Extract the signal of a single circuit edge in the requested flavor.

        Returns the **primary** signal of the edge:

        - ``edge_type="d"`` → ``signal_u`` (in the U / destination / query side)
        - ``edge_type="s"`` → ``signal_v`` (in the V / source / key side)

        For the autointerp use case where both U-side and V-side characterize the
        SVD communication channel, use :meth:`extract_edge_signal_pair_autointerp`.

        The ``flavor`` argument controls which space the signal lives in:

        ============================  ==========  ==============  =================================
        flavor                        rotate?     LN-normalize?   intended use
        ============================  ==========  ==============  =================================
        ``"rotated_normalized"``      yes         yes             analysis / autointerp (QK frame)
        ``"normalized"``              no (*)      yes             causal intervention
        ``"raw"``                     no          no              MLP upstream tracing
        ============================  ==========  ==============  =================================

        (*) **AH offset is the structural exception.** ``c_d`` / ``c_s`` are
        precomputed bias projections that live in pseudo-d_model space; they
        never see LN division, and they are rotated even in the ``"normalized"``
        flavor because intervention requires it. They are NOT rotated in the
        ``"raw"`` flavor.

        The returned signal is UNNORMALIZED in magnitude. Apply
        ``signal / signal.norm()`` at the call site if unit-norm vectors are
        needed (e.g., for autointerp).

        Math summary (see paper Appendix B, C). With
        :math:`P_U = U[:,\\mathrm{svs}] U[:,\\mathrm{svs}]^T` and
        :math:`P_V = V_T[\\mathrm{svs},:]^T V_T[\\mathrm{svs},:]`:

        For destination edges (``edge_type="d"``):

        - rotated:   :math:`\\tilde x = x \\, W_Q[l,h] \\, R^T \\, W_Q^{+}[l,h]`
        - unrotated: :math:`\\tilde x = x`
        - signal:    :math:`s_u = P_U \\, \\tilde x`

        For source edges (``edge_type="s"``):

        - rotated:   :math:`\\tilde x = W_K^{+}[l,h]^T \\, R \\, W_K[l,h]^T \\, x`
        - unrotated: :math:`\\tilde x = x`
        - signal:    :math:`s_v = P_V \\, \\tilde x`

        Args:
            cache: Activation cache from model.run_with_cache().
            prompt_idx: Index of the prompt in the cache batch.
            downstream_layer: Layer of the downstream attention head.
            downstream_ah_idx: Head index of the downstream attention head.
            downstream_dest_token: Dest (query) position of the downstream AH.
            downstream_src_token: Src (key) position of the downstream AH.
            upstream_layer: Layer of the upstream component.
            upstream_component_id: Integer ID of the upstream component type.
            upstream_dest_token: Dest position of the upstream component.
            upstream_src_token: Src position of the upstream component.
            edge_type: "d" for destination (query) edge, "s" for source (key).
            svs_used: List of singular vector indices used by this edge.
            flavor: REQUIRED. One of ``"rotated_normalized"``, ``"normalized"``,
                or ``"raw"``. There is no default — the three flavors are
                semantically distinct and callers must choose explicitly.

        Returns:
            The primary signal in d_model space:
                - ``signal_u`` for dest edges (``edge_type="d"``)
                - ``signal_v`` for src edges (``edge_type="s"``)
        """
        if flavor == "rotated_normalized":
            ln_normalize = True
            rotate_non_offset = True
            rotate_offset = True
        elif flavor == "normalized":
            ln_normalize = True
            rotate_non_offset = False
            rotate_offset = True   # structural exception: intervention requires rotation
        elif flavor == "raw":
            ln_normalize = False
            rotate_non_offset = False
            rotate_offset = False
        else:
            raise ValueError(
                f"Unknown flavor: {flavor!r}. Must be one of "
                '"rotated_normalized", "normalized", "raw".'
            )

        l = downstream_layer
        h = downstream_ah_idx
        n_heads = self.model.cfg.n_heads
        is_ah_offset = (upstream_component_id == n_heads + 3)
        do_rotate = rotate_offset if is_ah_offset else rotate_non_offset

        # --- Step 1: Get upstream component output ---
        c_term = self.c_d if edge_type == "d" else self.c_s
        x = get_component_output(
            self.model, cache, self.config, prompt_idx,
            downstream_layer, downstream_ah_idx,
            upstream_dest_token, upstream_src_token,
            upstream_layer, upstream_component_id,
            c_term,
            ln_normalize=ln_normalize,
        )

        # --- Step 2: Apply RoPE rotation (if applicable AND requested) ---
        # Per-edge approach: compute rotation for a single position, avoiding
        # precomputation of full M_d_all / M_s_all tensors (which would be
        # ~88 GB for Gemma). Three mat-vec products through d_head space.
        if self.config.has_rope and do_rotate:
            # Rotation position: upstream_dest_token for both edge types.
            # For dest edges: upstream_dest_token == downstream_dest_token.
            # For src edges: upstream_dest_token == downstream_src_token.
            R = get_rotation_matrix(
                self.model, upstream_dest_token, self.device
            )  # shape: (d_head, d_head)

            if edge_type == "d":
                # x @ M_d where M_d = W_Q @ R.T @ W_Q_pinv
                # Decomposed into three mat-vec products:
                #   t = x @ W_Q[l,h]           → (d_head,)
                #   t = t @ R.T                 → (d_head,)  [equiv. to R @ t for 1D]
                #   transformed = t @ W_Q_pinv  → (d_model,)
                t = x @ self.model.W_Q[l, h]
                t = t @ R.T
                transformed = t @ self.W_Q_pinv[l, h]
            else:
                # M_s @ x where M_s = W_K_pinv.T @ R @ W_K.T
                # Decomposed into three mat-vec products:
                #   t = W_K.T @ x               → (d_head,)  [equiv. to x @ W_K for 1D]
                #   t = R @ t                   → (d_head,)
                #   transformed = W_K_pinv.T @ t → (d_model,) [equiv. to t @ W_K_pinv for 1D]
                t = x @ self.model.W_K[l, h]
                t = R @ t
                transformed = t @ self.W_K_pinv[l, h]
                # NOTE on src rotation asymmetry: The matrix formula has R (not R.T)
                # applied from the left in M_s = W_K_pinv.T @ R @ W_K.T. When decomposing
                # M_s @ x into steps, the R is applied via R @ t (matrix-on-left).
                # For dest, the formula has R.T in M_d = W_Q @ R.T @ W_Q_pinv, and
                # the decomposition gives t @ R.T (vector-on-left). For 1D tensors,
                # t @ R.T == R @ t, so both effectively apply R to the d_head vector.
                # The actual asymmetry is in the surrounding weight matrices (W_Q vs W_K).
        else:
            transformed = x

        # --- Step 3: Project onto singular vector subspace and return PRIMARY ---
        if edge_type == "d":
            # Project onto U subspace (destination/query side)
            # P_U @ transformed = U[:,svs] @ (U[:,svs].T @ transformed)
            U_svs = self.U[l, h, :, svs_used]  # (d_model, n_svs)
            signal_u = U_svs @ (U_svs.T @ transformed)  # (d_model,)
            return signal_u
        else:
            # Project onto V subspace (source/key side)
            # P_V @ transformed = VT[svs,:].T @ (VT[svs,:] @ transformed)
            VT_svs = self.VT[l, h, svs_used, :]  # (n_svs, d_model)
            signal_v = VT_svs.T @ (VT_svs @ transformed)  # (d_model,)
            return signal_v

    def extract_edge_signal_pair_autointerp(
        self,
        cache: ActivationCache,
        prompt_idx: int,
        downstream_layer: int,
        downstream_ah_idx: int,
        downstream_dest_token: int,
        downstream_src_token: int,
        upstream_layer: int,
        upstream_component_id: int,
        upstream_dest_token: int,
        upstream_src_token: int,
        edge_type: str,
        svs_used: list[int],
        *,
        flavor: Literal["rotated_normalized", "normalized", "raw"] = "rotated_normalized",
    ) -> tuple[Tensor, Tensor]:
        """Extract the (signal_u, signal_v) pair for a single circuit edge.

        Autointerp use case: the SVD of QK yields paired query-side (U) and
        key-side (V) directions, and the *pair* characterizes the
        communication channel that an upstream component writes into. Per the
        paper (Sec. 2): "each pair defines a candidate low-dimensional
        communication channel through which upstream components can influence
        that head's attention to a destination–source token pair."

        For non-autointerp use cases (intervention, MLP tracing), call
        :meth:`extract_edge_signal` instead — only the primary signal
        (matching ``edge_type``) is meaningful there, and the cross-projected
        complement is not.

        Implementation: computes the primary signal via
        :meth:`extract_edge_signal` and cross-projects through Omega to get
        the complementary signal:

        - For dest edges: ``signal_u`` is primary, ``signal_v = Omega.T @ signal_u``
        - For src edges: ``signal_v`` is primary, ``signal_u = Omega @ signal_v``

        Both directions use the SVD factors ``U, S, VT`` directly (no full
        Omega matrix is formed).

        Args:
            (same as ``extract_edge_signal``)
            flavor: Defaults to ``"rotated_normalized"`` — the only flavor that
                is conceptually meaningful for autointerp pair analysis. The
                argument is exposed for debugging / unusual investigations only.

        Returns:
            (signal_u, signal_v) tuple where:
                signal_u: Signal in the U (query/dest) space, shape (d_model,).
                signal_v: Signal in the V (key/source) space, shape (d_model,).
        """
        # Compute the primary side via extract_edge_signal
        primary = self.extract_edge_signal(
            cache, prompt_idx,
            downstream_layer, downstream_ah_idx,
            downstream_dest_token, downstream_src_token,
            upstream_layer, upstream_component_id,
            upstream_dest_token, upstream_src_token,
            edge_type, svs_used,
            flavor=flavor,
        )

        l = downstream_layer
        h = downstream_ah_idx

        if edge_type == "d":
            signal_u = primary
            # Cross-project: signal_v = Omega.T @ signal_u = VT.T @ (S * (U.T @ signal_u))
            t = signal_u @ self.U[l, h]    # (d_head,) — equiv. to U.T @ signal_u
            t = self.S[l, h] * t           # (d_head,)
            signal_v = t @ self.VT[l, h]   # (d_model,) — t @ VT == VT.T @ t for 1D
        else:
            signal_v = primary
            # Cross-project: signal_u = Omega @ signal_v = U @ (S * (VT @ signal_v))
            t = self.VT[l, h] @ signal_v   # (d_head,)
            t = self.S[l, h] * t           # (d_head,)
            signal_u = self.U[l, h] @ t    # (d_model,)

        return signal_u, signal_v

    # ------------------------------------------------------------------
    # Causal intervention API
    # ------------------------------------------------------------------

    def run_intervention(
        self,
        tokens: Tensor,
        cache: ActivationCache,
        logits: Tensor,
        edge: EdgeSpec,
        prompt_idx: int = 0,
        intervention_type: Literal["local", "global"] = "local",
        boost: bool = False,
        center: bool | None = None,
    ) -> InterventionResult:
        """Run a single-edge causal intervention.

        Mirrors the structure of the ground-truth ``run_intervention`` function
        in ``new-code/experiments/interventions.py``, with all upstream-output
        and signal math inlined as explicit Step 1–7 blocks. The seven steps
        match the comments in the ground truth.

        For a destination edge (``edge_type="d"``) the intervention is placed
        at ``downstream_dest_token``. For a source edge (``edge_type="s"``) it
        is placed at ``downstream_src_token``. **local** mode modifies the
        Q (for ``"d"``) or K (for ``"s"``) of the downstream head only;
        **global** mode modifies the upstream component's output in the
        residual stream. AH offset + global is unsupported and raises.

        Centering policy: LN-pre / LN models center the signal (residuals are
        zero-mean), RMS-pre / RMS models do not. Auto-detected from
        ``model.cfg.normalization_type`` unless ``center`` is set explicitly.

        Args:
            tokens: Input tokens, shape ``(batch, seq)``. Must match the batch
                dimension of ``cache``.
            cache: Clean ``ActivationCache`` from a prior
                ``model.run_with_cache(tokens)`` call.
            logits: Clean logits from the same forward pass, shape
                ``(batch, seq, d_vocab)``. Passed through to the result.
            edge: ``EdgeSpec`` for the single edge to intervene on.
            prompt_idx: Index of the prompt within the batch to intervene on.
                Only this prompt's slice of the delta tensor is non-zero.
            intervention_type: ``"local"`` (modify Q/K input to downstream
                head) or ``"global"`` (modify upstream component output).
            boost: If ``True``, *add* the signal (boost the edge). If ``False``
                (default), *subtract* it (ablate the edge).
            center: Whether to mean-subtract the signal. ``None`` auto-detects
                from ``model.cfg.normalization_type``.

        Returns:
            ``InterventionResult`` carrying logits (clean and interv), the
            intervention cache, the applied delta tensor, and per-prompt
            metrics ``norm_ratio``, ``cos_sim``, ``attn_scores_clean``,
            ``attn_scores_interv``.
        """
        model = self.model
        config = self.config
        device = self.device
        n_heads = model.cfg.n_heads
        d_model = model.cfg.d_model

        # Step 0: validation
        if intervention_type not in ("local", "global"):
            raise ValueError(
                f"intervention_type must be 'local' or 'global', "
                f"got {intervention_type!r}"
            )
        if (intervention_type == "global"
                and edge.upstream_component_id == n_heads + 3):
            raise ValueError(
                "AH offset (component_id == n_heads + 3) cannot be "
                "intervened on in 'global' mode — no natural hook point "
                "exists for a bias-projection term. Use 'local' instead."
            )
        if center is None:
            center = _should_center(model.cfg.normalization_type)

        # Resolve downstream layer / head / tokens
        layer_downstream = edge.downstream_layer
        ah_idx_downstream = edge.downstream_ah_idx
        dest_token = edge.downstream_dest_token
        src_token = edge.downstream_src_token

        layer_upstream = edge.upstream_layer
        ah_idx_upstream = edge.upstream_component_id
        dest_token_upstream_idx = edge.upstream_dest_token
        src_token_upstream_idx = edge.upstream_src_token

        # Intervention position: downstream_dest for "d", downstream_src for "s"
        pos_interv = dest_token if edge.edge_type == "d" else src_token

        # Step 1: getting the SVs (carried by the EdgeSpec)
        svs_used = list(edge.svs_used)
        if len(svs_used) == 0:
            raise ValueError(
                "EdgeSpec has empty svs_used; intervention would be a no-op."
            )

        # Step 2: computing the projection P
        if edge.edge_type == "d":
            basis = self.U[layer_downstream, ah_idx_downstream]   # (d_model, rank)
            B_s = basis[:, svs_used]
            P = B_s @ B_s.T   # (d_model, d_model)
        else:
            basis = self.VT[layer_downstream, ah_idx_downstream].T   # (d_model, rank)
            B_s = basis[:, svs_used]
            P = B_s @ B_s.T

        # Step 3: computing upstream_out (case dispatch per component type)
        if ah_idx_upstream < n_heads:
            # AH: upstream_out = A * V @ W_O at the (dest_upstream, src_upstream) cell.
            # Mathematically equivalent to einsum('ti,ij->tij', A, V) @ W_O
            # indexed at [dest_upstream, src_upstream], but cheaper.
            A_val = cache[
                f"blocks.{layer_upstream}.attn.hook_pattern"
            ][prompt_idx, ah_idx_upstream,
              dest_token_upstream_idx, src_token_upstream_idx]
            if config.has_gqa:
                kv_idx = ah_idx_upstream // config.gqa_repeats
            else:
                kv_idx = ah_idx_upstream
            V_vec = cache[
                f"blocks.{layer_upstream}.attn.hook_v"
            ][prompt_idx, src_token_upstream_idx, kv_idx, :]   # (d_head,)
            upstream_out = A_val * (
                V_vec @ model.W_O[layer_upstream, ah_idx_upstream, :, :]
            )   # (d_model,)

            if config.has_post_attn_ln:
                # Gemma-2: post-attention LN is not folded into the weights.
                upstream_out = upstream_out * (
                    model.blocks[layer_upstream].ln1_post.w.detach()
                )
                ln_post_term = cache[
                    f"blocks.{layer_upstream}.ln1_post.hook_scale"
                ][prompt_idx, dest_token_upstream_idx]
                upstream_out = upstream_out / ln_post_term

        elif ah_idx_upstream == n_heads:   # MLP
            upstream_out = cache[
                f"blocks.{layer_upstream}.hook_mlp_out"
            ][prompt_idx, dest_token_upstream_idx].clone().detach()

        elif ah_idx_upstream == n_heads + 1:   # AH bias (b_O)
            upstream_out = model.b_O[layer_upstream].clone().detach()

        elif ah_idx_upstream == n_heads + 2:   # Embedding (residual at layer 0)
            upstream_out = cache[
                "blocks.0.hook_resid_pre"
            ][prompt_idx, dest_token_upstream_idx].clone().detach()

        elif ah_idx_upstream == n_heads + 3:   # AH offset (bias projection)
            # The signal is a projection of the downstream head's bias offset
            # through the QK rotation (RoPE) at the intervention position.
            if config.has_rope:
                # R applied at the downstream (dest_token for "d", src_token
                # for "s") — these are exactly pos_interv for each edge type.
                R = get_rotation_matrix(model, pos_interv, device)
                if edge.edge_type == "d":
                    # M_d = W_Q @ R.T @ W_Q_pinv; upstream_out = c_d @ M_d
                    M_d = (
                        model.W_Q[layer_downstream, ah_idx_downstream]
                        @ R.T
                        @ self.W_Q_pinv[layer_downstream, ah_idx_downstream]
                    )
                    upstream_out = (
                        self.c_d[layer_downstream, ah_idx_downstream] @ M_d
                    )
                else:
                    # M_s = W_K_pinv.T @ R @ W_K.T; upstream_out = M_s @ c_s
                    M_s = (
                        self.W_K_pinv[layer_downstream, ah_idx_downstream].T
                        @ R
                        @ model.W_K[layer_downstream, ah_idx_downstream].T
                    )
                    upstream_out = (
                        M_s @ self.c_s[layer_downstream, ah_idx_downstream]
                    )
            else:
                # No RoPE: rotation is the identity.
                if edge.edge_type == "d":
                    upstream_out = self.c_d[
                        layer_downstream, ah_idx_downstream
                    ].clone()
                else:
                    upstream_out = self.c_s[
                        layer_downstream, ah_idx_downstream
                    ].clone()

        else:
            raise ValueError(
                f"Unknown upstream_component_id={ah_idx_upstream} "
                f"(max valid = {n_heads + 3})"
            )

        # Step 3.1: computing the intervention value
        signal = upstream_out @ P   # (d_model,)

        # Step 3.2: centering the intervention value (LN-pre / LN only)
        if center:
            signal = signal - signal.mean()

        # Step 3.3: scaling by the downstream LN scale (local only)
        # The signal is added in LN-normalized space; for local interventions
        # we divide by the downstream head's ln1 scale at pos_interv to make
        # the addition consistent with what would have been observed had the
        # upstream output been different upstream of the LN.
        if intervention_type == "local":
            scaling_pos_interv = cache[
                f"blocks.{layer_downstream}.ln1.hook_scale"
            ][prompt_idx, pos_interv]
            signal = signal / scaling_pos_interv

        # Step 3.4: place the intervention value in the delta tensor at pos_interv
        if intervention_type == "local":
            ref_hook = f"blocks.{layer_downstream}.ln1.hook_normalized"
        else:
            ref_hook = _global_hook_name(
                layer_upstream, ah_idx_upstream, n_heads
            )
        delta_interv_tensor = torch.zeros(
            cache[ref_hook].shape, device=device
        )
        delta_interv_tensor[prompt_idx, pos_interv, :] = signal

        # Step 4: intervening in the model
        # For local interventions, `_run_local_intervention` returns the
        # explicitly modified `attn_input_interv` because the Q/K hook does
        # NOT change `ln1.hook_normalized` in the interv cache — the hook is
        # applied AFTER ln1, on hook_q/hook_k. We use this returned tensor for
        # the Step-6 metrics. (The ground truth `run_intervention` does the
        # same thing — see its `after_interv` variable from
        # `run_local_intervention`.)
        if intervention_type == "local":
            interv_logits, interv_cache, attn_input_interv = (
                self._run_local_intervention(
                    tokens, cache,
                    layer_downstream, ah_idx_downstream,
                    delta_interv_tensor, boost, edge.edge_type,
                )
            )
        else:
            interv_logits, interv_cache = self._run_global_intervention(
                tokens, layer_upstream, ah_idx_upstream,
                delta_interv_tensor, boost,
            )

        # Step 6: computing how much we are changing in the intervention
        # `before_interv` / `after_interv` mirror the ground truth: for local
        # the downstream LN-normalized input changes; for global the upstream
        # component output changes.
        if intervention_type == "local":
            before_interv = cache[
                f"blocks.{layer_downstream}.ln1.hook_normalized"
            ]
            after_interv = attn_input_interv
        else:
            if ah_idx_upstream < n_heads or ah_idx_upstream == n_heads + 1:
                # AH and AH bias both ride with hook_attn_out
                before_interv = cache[
                    f"blocks.{layer_upstream}.hook_attn_out"
                ]
                after_interv = interv_cache[
                    f"blocks.{layer_upstream}.hook_attn_out"
                ]
            elif ah_idx_upstream == n_heads:   # MLP
                before_interv = cache[f"blocks.{layer_upstream}.hook_mlp_out"]
                after_interv = interv_cache[f"blocks.{layer_upstream}.hook_mlp_out"]
            else:   # Embedding
                before_interv = cache["blocks.0.hook_resid_pre"]
                after_interv = interv_cache["blocks.0.hook_resid_pre"]

        # Restrict to the intervened position (per-prompt). For prompts other
        # than `prompt_idx` the position is the same but the delta is zero, so
        # the metrics fall back to their identity values (norm_ratio=1,
        # cos_sim=1) — clean signal.
        n_prompts = tokens.shape[0]
        interv_pos_arange = torch.full(
            (n_prompts,), pos_interv, device=device, dtype=torch.long
        )
        after_at_pos = after_interv[
            torch.arange(n_prompts, device=device), interv_pos_arange, :
        ]
        before_at_pos = before_interv[
            torch.arange(n_prompts, device=device), interv_pos_arange, :
        ]
        norm_ratio = (
            torch.norm(after_at_pos, dim=1)
            / torch.norm(before_at_pos, dim=1)
        )
        cos_sim = F.cosine_similarity(after_at_pos, before_at_pos, dim=1)

        # Step 7: attention scores at the downstream head's (dest, src) cell
        attn_scores_clean = cache[
            f"blocks.{layer_downstream}.attn.hook_pattern"
        ][:, ah_idx_downstream, dest_token, src_token]
        attn_scores_interv = interv_cache[
            f"blocks.{layer_downstream}.attn.hook_pattern"
        ][:, ah_idx_downstream, dest_token, src_token]

        return InterventionResult(
            logits_clean=logits,
            logits_interv=interv_logits,
            interv_cache=interv_cache,
            delta=delta_interv_tensor,
            norm_ratio=norm_ratio,
            cos_sim=cos_sim,
            attn_scores_clean=attn_scores_clean,
            attn_scores_interv=attn_scores_interv,
        )

    def _run_local_intervention(
        self,
        tokens: Tensor,
        cache: ActivationCache,
        layer_downstream: int,
        ah_idx_downstream: int,
        delta: Tensor,
        boost: bool,
        edge_type: Literal["d", "s"],
    ) -> tuple[Tensor, ActivationCache, Tensor]:
        """Install Q or K hook for a local intervention and run the model.

        Mirrors ``run_local_intervention`` from the ground-truth experiment.
        The LN-normalized attention input is shifted by ``+delta`` (boost) or
        ``-delta`` (ablate), then Q (for ``"d"``) or K (for ``"s"``) is
        recomputed for the targeted head only and patched into the forward
        pass via a hook.
        """
        model = self.model
        gqa_repeats = self.config.gqa_repeats if self.config.has_gqa else 1

        # Getting the attention input (same for all heads in the layer)
        attn_input = cache[
            f"blocks.{layer_downstream}.ln1.hook_normalized"
        ]
        if boost:
            attn_input_interv = attn_input + delta
        else:
            attn_input_interv = attn_input - delta

        # Recompute Q (for "d") or K (for "s") for the targeted head
        if edge_type == "d":
            q_interv = F.linear(
                attn_input_interv,
                model.W_Q[layer_downstream, ah_idx_downstream, :, :].T,
                model.b_Q[layer_downstream, ah_idx_downstream, :],
            )
            hook = (
                f"blocks.{layer_downstream}.attn.hook_q",
                partial(_local_q_hook, ah_idx=ah_idx_downstream, q_interv=q_interv),
            )
        else:
            k_interv = F.linear(
                attn_input_interv,
                model.W_K[layer_downstream, ah_idx_downstream, :, :].T,
                model.b_K[layer_downstream, ah_idx_downstream, :],
            )
            kv_idx = ah_idx_downstream // gqa_repeats
            hook = (
                f"blocks.{layer_downstream}.attn.hook_k",
                partial(_local_k_hook, kv_idx=kv_idx, k_interv=k_interv),
            )

        # Run the model with the hook + caching
        with model.hooks(fwd_hooks=[hook]):
            interv_logits, interv_cache = model.run_with_cache(tokens)

        return interv_logits, interv_cache, attn_input_interv

    def _run_global_intervention(
        self,
        tokens: Tensor,
        layer_upstream: int,
        ah_idx_upstream: int,
        delta: Tensor,
        boost: bool,
    ) -> tuple[Tensor, ActivationCache]:
        """Install a residual-stream hook for a global intervention and run.

        Mirrors ``run_global_intervention`` from the ground-truth experiment.
        The upstream component's output is shifted by ``+delta`` (boost) or
        ``-delta`` (ablate) via a hook on ``hook_attn_out`` (AH or AH bias),
        ``hook_mlp_out`` (MLP), or ``blocks.0.hook_resid_pre`` (embedding).
        """
        model = self.model
        n_heads = model.cfg.n_heads

        sign = +1.0 if boost else -1.0

        if ah_idx_upstream < n_heads or ah_idx_upstream == n_heads + 1:
            hook_name = f"blocks.{layer_upstream}.hook_attn_out"
        elif ah_idx_upstream == n_heads:
            hook_name = f"blocks.{layer_upstream}.hook_mlp_out"
        elif ah_idx_upstream == n_heads + 2:
            hook_name = "blocks.0.hook_resid_pre"
        else:
            raise ValueError(
                f"Invalid ah_idx_upstream={ah_idx_upstream} for global "
                "intervention"
            )

        hook = (
            hook_name,
            partial(_global_residual_hook, sign=sign, delta=delta),
        )

        with model.hooks(fwd_hooks=[hook]):
            interv_logits, interv_cache = model.run_with_cache(tokens)

        return interv_logits, interv_cache


def _global_hook_name(
    layer: int, component_id: int, n_heads: int
) -> str:
    """Return the TL hook name for a global intervention's upstream component.

    Raises ``ValueError`` for AH offset (which has no global hook point).
    """
    if component_id < n_heads:
        return f"blocks.{layer}.hook_attn_out"      # AH → summed attn output
    elif component_id == n_heads:
        return f"blocks.{layer}.hook_mlp_out"       # MLP
    elif component_id == n_heads + 1:
        return f"blocks.{layer}.hook_attn_out"      # AH bias rides with attn_out
    elif component_id == n_heads + 2:
        return "blocks.0.hook_resid_pre"            # Embedding
    elif component_id == n_heads + 3:
        raise ValueError(
            "AH offset has no global hook point — call this only after "
            "validating that the component is not AH offset."
        )
    else:
        raise ValueError(
            f"Unknown component_id={component_id} (max valid = "
            f"{n_heads + 3})"
        )


def _local_q_hook(x, hook, ah_idx, q_interv):
    """TL hook fn — replace one head's Q with the intervention value."""
    # x: (batch, seq, n_heads, d_head); q_interv: (batch, seq, d_head)
    x[:, :, ah_idx, :] = q_interv
    return x


def _local_k_hook(x, hook, kv_idx, k_interv):
    """TL hook fn — replace one KV head's K with the intervention value."""
    # x: (batch, seq, n_kv_heads, d_head); k_interv: (batch, seq, d_head)
    x[:, :, kv_idx, :] = k_interv
    return x


def _global_residual_hook(x, hook, sign, delta):
    """TL hook fn — add/subtract a residual-stream-space delta to the output."""
    # x and delta have the same shape (batch, seq, d_model)
    return x + sign * delta
