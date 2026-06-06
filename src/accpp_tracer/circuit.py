"""Full ACC++ circuit builder.

Provides the Tracer class that orchestrates the complete circuit tracing pipeline:
precomputing model-level quantities, identifying seed components, and recursively
building circuit graphs.
"""

from collections import defaultdict
from typing import Callable, Literal, Union

import networkx as nx
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache

from einops import einsum

from functools import partial   # CHANGED: needed for intervention hooks

import torch.nn.functional as F

from .decomposition import compute_weight_pseudoinverses, get_omega_decomposition
from .intervention import (   # CHANGED: intervention API
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
def get_seeds(
    model: HookedTransformer,
    config: ModelConfig,
    cache: ActivationCache,
    prompt_idx: int,
    logit_direction: Float[Tensor, "d_model"],
    end_token_pos: int,
    device: str,
) -> tuple[list[tuple], dict[tuple, float]]:
    """Identify seed components for circuit tracing.

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

    # Layer norming the upstream outputs and projecting onto logit_direction
    contrib_end_f_W_U_tensor = (
        upstream_output_breakdown[:, :, end_token_pos, :, :]
        / cache["ln_final.hook_scale"][prompt_idx, end_token_pos]
    ) @ logit_direction

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

    Example:
        >>> from accpp_tracer import Tracer
        >>> from transformer_lens import HookedTransformer
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> tracer = Tracer(model)
        >>> graph = tracer.trace(
        ...     "When Mary and John went to the store, John gave a drink to",
        ...     answer_token=" Mary",
        ...     wrong_token=" John",
        ... )
    """

    def __init__(
        self,
        model: HookedTransformer,
        device: str | None = None,
        use_numpy_svd: bool = False,
        dynamic_threshold_scale: float = 2.5,
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

        # Precompute expensive model-level quantities
        self.U, self.S, self.VT = get_omega_decomposition(
            model, self.config, self.device
        )
        self.W_Q_pinv, self.W_K_pinv = compute_weight_pseudoinverses(
            model, self.config, self.device
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
    ) -> nx.MultiDiGraph:
        """Trace a single prompt (Level 3 — simplest API).

        Handles tokenization, forward pass, token mapping, logit direction
        computation, seed identification, and recursive circuit tracing.

        Supports tracing multiple logit directions simultaneously. Each direction
        becomes a separate root node in the returned merged graph. Attention head
        subgraphs shared between directions are traced only once.

        Args:
            prompt: Input text string.
            answer_token: Correct next token (str, int, or list of str/int). When
                a list is supplied each element becomes its own logit direction and
                root node in the merged circuit graph. Ignored when top_p is set.
                Must be provided when top_p is None.
            wrong_token: Optional contrastive token. When supplied, every direction
                becomes W_U[:, answer_i] - W_U[:, wrong]. Applied to all directions.
            top_p: If set, ignores answer_token and automatically selects the minimum
                set of top tokens whose cumulative probability >= top_p (standard
                nucleus / top-p definition). Each selected token becomes its own
                direction. wrong_token is still applied if provided.
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
        logit_direction: Tensor | list[Tensor],
        end_token_pos: int,
        idx_to_token: dict[int, str],
        root_node: tuple | list[tuple],
        prompt_idx: int = 0,
        attn_weight_thresh: str | float | Callable[[int], float] = "dynamic",
        signals: Literal[None, "rotated_normalized", "normalized", "raw"] = None,
    ) -> nx.MultiDiGraph:
        """Trace from a pre-computed cache (Level 2 — advanced API).

        The user provides the cache, logit direction(s), token mapping, and root
        node(s). This is what paper reproduction scripts call in a loop.

        Supports tracing multiple logit directions simultaneously. Pass parallel
        lists for logit_direction and root_node to trace each direction as a
        separate root node in the same returned graph. The is_traced state is
        shared across directions, so attention head subgraphs common to multiple
        directions are traced only once. Single-direction callers are unaffected:
        passing a single Tensor and tuple behaves identically to before.

        Args:
            cache: ActivationCache from model.run_with_cache().
            logit_direction: Single direction tensor or list of direction tensors
                in residual stream space (e.g., W_U[:, IO] - W_U[:, S]).
            end_token_pos: Position of the output token.
            idx_to_token: Dict mapping token position (int) to label (str).
            root_node: Single tuple or list of tuples (one per direction) for
                the root/output node label(s) in the graph.
            prompt_idx: Index of this prompt in the cache batch.
            attn_weight_thresh: "dynamic" (= scale/context_size), a float in
                [0, 1], or a callable taking dest_token position → float.
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

        dirs = [logit_direction] if isinstance(logit_direction, Tensor) else logit_direction
        roots = [root_node] if isinstance(root_node, tuple) else root_node

        return self._trace_from_cache_inner(
            model, cache, dirs, roots, prompt_idx, end_token_pos,
            idx_to_token, attn_weight_thresh, signals,
        )

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
    ):
        # Build circuit graph — shared across all directions
        G = nx.MultiDiGraph()
        is_traced: dict[tuple, int] = {}

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

        return G

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
            after_interv = attn_input_interv   # CHANGED: not the cache (see Step 4 note)
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
