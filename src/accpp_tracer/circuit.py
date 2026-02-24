"""Full ACC++ circuit builder.

Provides the Tracer class that orchestrates the complete circuit tracing pipeline:
precomputing model-level quantities, identifying seed components, and recursively
building circuit graphs.
"""

from collections import defaultdict
from copy import deepcopy
from typing import Union

import networkx as nx
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache

from einops import einsum

from .decomposition import compute_weight_pseudoinverses, get_omega_decomposition
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
    upstream_output_breakdown[0, -1, end_token_pos, end_token_pos] = deepcopy(
        cache["blocks.0.hook_resid_pre"][prompt_idx, end_token_pos]
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
                ] = deepcopy(
                    cache[f"blocks.{upstream_layer}.hook_mlp_out"][
                        prompt_idx, end_token_pos
                    ]
                )
            elif upstream_ah_idx == model.cfg.n_heads + 1:  # AH bias
                upstream_output_breakdown[
                    upstream_layer, upstream_ah_idx, end_token_pos, end_token_pos
                ] = deepcopy(model.b_O[upstream_layer])

    # Layer norming the upstream outputs and projecting onto logit_direction
    contrib_end_f_W_U_tensor = (
        upstream_output_breakdown[:, :, end_token_pos, :, :]
        / cache["ln_final.hook_scale"][prompt_idx, end_token_pos]
    ) @ logit_direction

    if contrib_end_f_W_U_tensor.sum() > 0:
        trace_seeds = get_upstream_contributors_seed(
            contrib_end_f_W_U_tensor.cpu().numpy(), 1.0
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
    ) -> None:
        self.model = model
        self.device = device or str(model.cfg.device)
        self.config = get_model_config(model, use_numpy_svd=use_numpy_svd)

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
        answer_token: str | int,
        wrong_token: str | int | None = None,
        attn_weight_thresh: str | float = "dynamic",
    ) -> nx.MultiDiGraph:
        """Trace a single prompt (Level 3 — simplest API).

        Handles tokenization, forward pass, token mapping, logit direction
        computation, seed identification, and recursive circuit tracing.

        Args:
            prompt: Input text string.
            answer_token: Correct next token (str or token id).
            wrong_token: Optional contrastive token for logit diff direction.
                If provided, logit_direction = W_U[:, answer] - W_U[:, wrong].
                If None, logit_direction = W_U[:, answer].
            attn_weight_thresh: "dynamic" (= 2.5/context_size) or a float
                in [0, 1].

        Returns:
            nx.MultiDiGraph — the traced circuit graph.
        """
        model = self.model

        # Tokenize
        tokens = model.to_tokens(prompt)  # shape: (1, seq_len)
        end_token_pos = tokens.shape[1] - 1

        # Forward pass
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Compute logit direction
        if isinstance(answer_token, str):
            answer_token = model.to_single_token(answer_token)
        logit_direction = model.W_U[:, answer_token].clone()
        if wrong_token is not None:
            if isinstance(wrong_token, str):
                wrong_token = model.to_single_token(wrong_token)
            logit_direction = logit_direction - model.W_U[:, wrong_token]

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

        # Root node
        root_node = ("Logit direction", idx_to_token[end_token_pos])

        return self.trace_from_cache(
            cache=cache,
            logit_direction=logit_direction,
            end_token_pos=end_token_pos,
            idx_to_token=idx_to_token,
            root_node=root_node,
            prompt_idx=0,
            attn_weight_thresh=attn_weight_thresh,
        )

    @typechecked
    def trace_from_cache(
        self,
        cache: ActivationCache,
        logit_direction: Float[Tensor, "d_model"],
        end_token_pos: int,
        idx_to_token: dict[int, str],
        root_node: tuple,
        prompt_idx: int = 0,
        attn_weight_thresh: str | float = "dynamic",
    ) -> nx.MultiDiGraph:
        """Trace from a pre-computed cache (Level 2 — advanced API).

        The user provides the cache, logit direction, token mapping, and root
        node. This is what paper reproduction scripts call in a loop.

        Args:
            cache: ActivationCache from model.run_with_cache().
            logit_direction: Direction in residual stream to trace
                (e.g., W_U[:, IO] - W_U[:, S]).
            end_token_pos: Position of the output token.
            idx_to_token: Dict mapping token position (int) to label (str).
            root_node: Tuple label for the root/output node in the graph.
            prompt_idx: Index of this prompt in the cache batch.
            attn_weight_thresh: "dynamic" or float in [0, 1].

        Returns:
            nx.MultiDiGraph — the traced circuit graph.
        """
        model = self.model

        # Get seeds
        trace_seeds, seeds_contrib = get_seeds(
            model,
            self.config,
            cache,
            prompt_idx,
            logit_direction,
            end_token_pos,
            self.device,
        )

        if len(trace_seeds) == 0:
            return nx.MultiDiGraph()

        # Check if any AH seeds exist
        has_ah_seed = any(
            ah_idx < model.cfg.n_heads for _, ah_idx, _ in trace_seeds
        )
        if not has_ah_seed:
            return nx.MultiDiGraph()

        # Build circuit graph
        G = nx.MultiDiGraph()
        is_traced: dict[tuple, int] = {}

        for layer, ah_idx, src_token in trace_seeds:
            ah_idx_label = get_ah_idx_label(ah_idx, model.cfg.n_heads)

            # Add edge from seed to root node
            if end_token_pos in idx_to_token and src_token in idx_to_token:
                G.add_edge(
                    (
                        layer,
                        ah_idx_label,
                        idx_to_token[end_token_pos],
                        idx_to_token[src_token],
                    ),
                    root_node,
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
        attn_weight_thresh: str | float,
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
            attn_weight_thresh: "dynamic" or float.
        """
        is_traced[(layer, ah_idx, dest_token, src_token)] = 1

        if layer == 0 or dest_token == 0:
            return

        if src_token > dest_token:
            return

        if attn_weight_thresh == "dynamic":
            attn_weight_thresh_eval = min(1.0, 2.5 / (dest_token + 1))
        else:
            attn_weight_thresh_eval = float(attn_weight_thresh)

        assert 0.0 <= attn_weight_thresh_eval <= 1.0

        if (
            cache[f"blocks.{layer}.attn.hook_pattern"][
                prompt_idx, ah_idx, dest_token, src_token
            ].item()
            < attn_weight_thresh_eval
        ):
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
    ) -> tuple[Tensor, Tensor]:
        """Extract signal pair (signal_u, signal_v) for a single circuit edge.

        Given an edge in a traced ACC++ circuit graph, computes the destination
        and source signal vectors by:

        1. Extracting the upstream component's output (normalized by downstream LN)
        2. Applying RoPE rotation if the model uses rotary embeddings
        3. Projecting onto the relevant singular vector subspace
        4. Cross-projecting through Omega to get the complementary signal

        The returned signals are UNNORMALIZED. Normalize at load time if needed
        (e.g., signal / signal.norm() for unit-norm signals).

        Math summary (see paper Appendix B, C):

        For destination edges (edge_type="d"):
            rotation:  transformed = x @ W_Q[l,h] @ R[dest].T @ W_Q_pinv[l,h]
            projection: signal_u = P_U @ transformed   (P_U = U[:,svs] @ U[:,svs].T)
            cross-proj: signal_v = Omega.T @ signal_u   (via U, S, VT — no full matrix)

        For source edges (edge_type="s"):
            rotation:  transformed = W_K_pinv[l,h].T @ R[src] @ W_K[l,h].T @ x
            projection: signal_v = P_V @ transformed   (P_V = VT[svs,:].T @ VT[svs,:])
            cross-proj: signal_u = Omega @ signal_v     (via U, S, VT — no full matrix)

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

        Returns:
            (signal_u, signal_v) tuple where:
                signal_u: Signal in the U (query/dest) space, shape (d_model,).
                signal_v: Signal in the V (key/source) space, shape (d_model,).
        """
        l = downstream_layer
        h = downstream_ah_idx

        # --- Step 1: Get upstream component output ---
        c_term = self.c_d if edge_type == "d" else self.c_s
        x = get_component_output(
            self.model, cache, self.config, prompt_idx,
            downstream_layer, downstream_ah_idx,
            upstream_dest_token, upstream_src_token,
            upstream_layer, upstream_component_id,
            c_term,
        )

        # --- Step 2: Apply RoPE rotation (if applicable) ---
        # Per-edge approach: compute rotation for a single position, avoiding
        # precomputation of full M_d_all / M_s_all tensors (which would be
        # ~88 GB for Gemma). Three mat-vec products through d_head space.
        if self.config.has_rope:
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

        # --- Step 3: Project onto singular vector subspace ---
        if edge_type == "d":
            # Project onto U subspace (destination/query side)
            # P_U @ transformed = U[:,svs] @ (U[:,svs].T @ transformed)
            U_svs = self.U[l, h, :, svs_used]  # (d_model, n_svs)
            signal_u = U_svs @ (U_svs.T @ transformed)  # (d_model,)

            # --- Step 4: Cross-project through Omega.T ---
            # signal_v = Omega.T @ signal_u = VT.T @ (S * (U.T @ signal_u))
            t = signal_u @ self.U[l, h]  # (d_head,) — equiv. to U.T @ signal_u
            t = self.S[l, h] * t  # (d_head,)
            signal_v = t @ self.VT[l, h]  # (d_model,) — equiv. to VT.T.T @ t? No: t @ VT = (VT.T @ t) for 1D... see note

            # NOTE: t @ VT[l,h] where t is (d_head,) and VT is (d_head, d_model)
            # gives result[k] = sum_j t[j] * VT[j,k], which is the same as
            # (VT.T @ t)[k] = sum_j VT.T[k,j] * t[j] = sum_j VT[j,k] * t[j].
            # But we need VT.T @ t (since Omega.T = VT.T @ diag(S) @ U.T).
            # Since t @ VT == VT.T @ t for 1D vectors, this is correct.

        else:
            # Project onto V subspace (source/key side)
            # P_V @ transformed = VT[svs,:].T @ (VT[svs,:] @ transformed)
            VT_svs = self.VT[l, h, svs_used, :]  # (n_svs, d_model)
            signal_v = VT_svs.T @ (VT_svs @ transformed)  # (d_model,)

            # --- Step 4: Cross-project through Omega ---
            # signal_u = Omega @ signal_v = U @ (S * (VT @ signal_v))
            t = self.VT[l, h] @ signal_v  # (d_head,)
            t = self.S[l, h] * t  # (d_head,)
            signal_u = self.U[l, h] @ t  # (d_model,)

        return signal_u, signal_v
