"""Causal interventions on traced ACC++ circuits.

Provides three building blocks:

- ``EdgeSpec`` — a resolved circuit-edge specification with integer token
  positions, used as input to ``Tracer.run_intervention()``.
- ``edges_from_graph()`` — parses a traced circuit graph into a list of
  ``EdgeSpec`` objects.
- ``InterventionResult`` — the dataclass returned by
  ``Tracer.run_intervention()`` carrying logits, cache, delta tensor, and
  per-prompt metrics.

The ``Tracer.run_intervention()`` method (in ``circuit.py``) is the entry
point.

Two intervention modes are supported:

- **local**: modify the LN-normalized Q/K input of the downstream attention
  head, then recompute Q (for ``edge_type="d"``) or K (for ``edge_type="s"``)
  for that head only. Surgical — affects only the targeted edge's downstream
  head.
- **global**: modify the upstream component's output directly in the residual
  stream (``hook_attn_out`` for attention heads / AH bias, ``hook_mlp_out`` for
  MLPs, ``blocks.0.hook_resid_pre`` for embeddings). Broad — affects every
  downstream consumer of that component.

The AH **offset** component has no natural residual-stream-space hook point
and therefore is not supported in ``global`` mode.

Centering policy: residuals are zero-mean for ``LNPre`` / ``LN`` normalization
(centering is folded into the weights). For these models the intervention
signal is mean-subtracted before being applied; for ``RMSPre`` / ``RMS``
models (e.g. Gemma-2-2b) it is not. Auto-detected from
``model.cfg.normalization_type``.
"""

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import torch
from torch import Tensor
from transformer_lens import ActivationCache

from .signals import component_label_to_id


@dataclass(frozen=True)
class EdgeSpec:
    """Fully resolved specification of a circuit edge for intervention.

    All token positions are integer indices into the prompt's token sequence
    (not string labels). Component IDs follow the convention used elsewhere in
    the library:

        0..n_heads-1  : regular attention head
        n_heads       : MLP
        n_heads + 1   : AH bias (b_O)
        n_heads + 2   : Embedding (layer 0 residual)
        n_heads + 3   : AH offset (downstream bias projection)
    """

    downstream_layer: int
    downstream_ah_idx: int
    downstream_dest_token: int
    downstream_src_token: int
    upstream_layer: int
    upstream_component_id: int
    upstream_dest_token: int
    upstream_src_token: int
    edge_type: Literal["d", "s"]
    svs_used: tuple[int, ...]   # frozen tuple so EdgeSpec is hashable

    def __post_init__(self):
        if self.edge_type not in ("d", "s"):
            raise ValueError(
                f"edge_type must be 'd' or 's', got {self.edge_type!r}"
            )


def edges_from_graph(
    graph: nx.MultiDiGraph,
    token_to_idx: dict[str, int],
    n_heads: int,
    edges: list[tuple] | None = None,
) -> list[EdgeSpec]:
    """Parse graph edges into ``EdgeSpec`` objects.

    Reads circuit edges produced by ``Tracer.trace()`` / ``trace_from_cache()``
    and converts each into a fully-resolved ``EdgeSpec`` with integer token
    positions and component IDs. Edges where either endpoint is not a valid
    4-tuple node (e.g. root / output edges with string labels like
    ``("Logit direction", " Mary")``) are silently skipped.

    Args:
        graph: A traced circuit graph (a ``nx.MultiDiGraph`` whose internal
            nodes are 4-tuples ``(layer, ah_idx_label, dest_label, src_label)``).
        token_to_idx: Mapping from token label (str) → integer position in the
            prompt. Typically built from the inverse of the ``idx_to_token``
            dict used during tracing.
        n_heads: Number of attention heads in the model (needed to resolve
            string component labels such as ``"MLP"`` to integer IDs).
        edges: Optional list of ``(u, v, key)`` tuples to parse. If ``None``,
            every edge in the graph is parsed.

    Returns:
        A list of ``EdgeSpec`` objects. May be empty if all edges were skipped
        (e.g. graph contains only root edges).
    """
    if edges is None:
        edges_iter = graph.edges(keys=True, data=True)
    else:
        edges_iter = [
            (u, v, k, graph.get_edge_data(u, v, key=k)) for u, v, k in edges
        ]

    out: list[EdgeSpec] = []
    for u, v, _key, data in edges_iter:
        # Skip root / output edges where one endpoint is a string-tuple
        if not (isinstance(u, tuple) and len(u) == 4):
            continue
        if not (isinstance(v, tuple) and len(v) == 4):
            continue

        upstream_layer, upstream_ah_label, upstream_dest_lbl, upstream_src_lbl = u
        downstream_layer, downstream_ah_idx, downstream_dest_lbl, downstream_src_lbl = v

        if not isinstance(downstream_ah_idx, int):
            # Downstream MUST be an attention head; skip otherwise (defensive)
            continue

        upstream_component_id = component_label_to_id(upstream_ah_label, n_heads)

        # Resolve token labels to integer positions
        try:
            downstream_dest_token = token_to_idx[downstream_dest_lbl]
            downstream_src_token = token_to_idx[downstream_src_lbl]
            upstream_dest_token = token_to_idx[upstream_dest_lbl]
            upstream_src_token = token_to_idx[upstream_src_lbl]
        except KeyError:
            continue   # token label not in mapping (defensive)

        # Parse svs_used: stored as ``str([0, 1])`` by ``_trace_recursive``
        svs_used_raw = data.get("svs_used")
        if svs_used_raw is None:
            continue
        if isinstance(svs_used_raw, str):
            import ast
            svs_list = ast.literal_eval(svs_used_raw)
        else:
            svs_list = list(svs_used_raw)

        edge_type = data.get("type")
        if edge_type not in ("d", "s"):
            continue

        out.append(EdgeSpec(
            downstream_layer=int(downstream_layer),
            downstream_ah_idx=int(downstream_ah_idx),
            downstream_dest_token=int(downstream_dest_token),
            downstream_src_token=int(downstream_src_token),
            upstream_layer=int(upstream_layer),
            upstream_component_id=int(upstream_component_id),
            upstream_dest_token=int(upstream_dest_token),
            upstream_src_token=int(upstream_src_token),
            edge_type=edge_type,
            svs_used=tuple(int(i) for i in svs_list),
        ))

    return out


def _should_center(normalization_type: str | None) -> bool:
    """Auto-detect centering policy from ``model.cfg.normalization_type``.

    LN-pre / LN-post (GPT-2, Pythia) normalize residuals to zero mean → the
    intervention signal must be zero-mean to remain in-distribution. RMS-pre /
    RMS-post (Gemma-2) does not center → the intervention signal is not
    re-centered.
    """
    if normalization_type is None:
        return False   # no normalization at all
    if normalization_type.startswith("LN"):
        return True
    if normalization_type.startswith("RMS"):
        return False
    raise ValueError(
        f"Unknown normalization_type {normalization_type!r}; cannot auto-detect "
        "centering policy. Pass center=True/False explicitly."
    )


@dataclass
class InterventionResult:
    """Result of a causal intervention from ``Tracer.run_intervention()``.

    All per-prompt metric tensors have shape ``(n_prompts,)`` where
    ``n_prompts`` is the batch dimension of the input ``tokens``. For
    single-prompt interventions (the default API), only the entry at the
    targeted ``prompt_idx`` is affected by the intervention; the other entries
    reflect the clean (no-op) values because the delta tensor is zero at those
    positions.

    Attributes:
        logits_clean: Clean logits from the un-intervened forward pass, shape
            ``(n_prompts, n_tokens, d_vocab)``. Passed through from the input.
        logits_interv: Intervention logits, same shape as ``logits_clean``.
        interv_cache: ``ActivationCache`` from the intervention forward pass.
            Useful for measuring downstream effects beyond the metrics below.
        delta: The delta tensor that was applied. For ``local`` interventions
            it has shape ``(n_prompts, n_tokens, d_model)`` (the value added
            to / subtracted from ``ln1.hook_normalized``). For ``global``
            interventions it has the same shape but lives in residual-stream
            space.
        norm_ratio: ``norm(after) / norm(before)`` at the intervention
            position, per prompt. Shape ``(n_prompts,)``. For ``local``
            interventions this compares ``ln1.hook_normalized`` before vs.
            after intervention; for ``global`` interventions it compares the
            upstream component output.
        cos_sim: Cosine similarity between before and after vectors at the
            intervention position, per prompt. Shape ``(n_prompts,)``.
        attn_scores_clean: Attention weight at the downstream head's
            ``(dest, src)`` cell, computed on the clean cache. Shape
            ``(n_prompts,)``. Useful for measuring the targeted edge's
            attention contribution before intervention.
        attn_scores_interv: Same as ``attn_scores_clean`` but from the
            intervention cache.
    """

    logits_clean: Tensor
    logits_interv: Tensor
    interv_cache: ActivationCache
    delta: Tensor
    norm_ratio: Tensor
    cos_sim: Tensor
    attn_scores_clean: Tensor
    attn_scores_interv: Tensor
