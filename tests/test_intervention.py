"""Tests for edge-signal extraction and causal interventions."""

import pytest
import torch

from accpp_tracer import edges_from_graph

from conftest import build_idx_to_token


@pytest.fixture(scope="module")
def gpt2_edges(gpt2_model, gpt2_prob_graph, gpt2_ioi):
    tokens, _, _ = gpt2_ioi
    idx_to_token = build_idx_to_token(gpt2_model, tokens)
    token_to_idx = {v: k for k, v in idx_to_token.items()}
    edges = edges_from_graph(
        gpt2_prob_graph, token_to_idx, gpt2_model.cfg.n_heads
    )
    assert len(edges) > 0, "traced graph produced no parseable edges"
    return edges


def _edge_of_type(edges, edge_type):
    for e in edges:
        if e.edge_type == edge_type:
            return e
    pytest.skip(f"no edge of type {edge_type!r} in traced graph")


@pytest.mark.parametrize("flavor", ["rotated_normalized", "normalized", "raw"])
def test_extract_edge_signal_flavors(gpt2_tracer, gpt2_model, gpt2_ioi, gpt2_edges, flavor):
    _, _, cache = gpt2_ioi
    edge = gpt2_edges[0]
    signal = gpt2_tracer.extract_edge_signal(
        cache, 0,
        edge.downstream_layer, edge.downstream_ah_idx,
        edge.downstream_dest_token, edge.downstream_src_token,
        edge.upstream_layer, edge.upstream_component_id,
        edge.upstream_dest_token, edge.upstream_src_token,
        edge.edge_type, list(edge.svs_used),
        flavor=flavor,
    )
    assert signal.shape == (gpt2_model.cfg.d_model,)
    assert torch.isfinite(signal).all()
    assert signal.norm().item() > 0


def test_extract_edge_signal_pair(gpt2_tracer, gpt2_model, gpt2_ioi, gpt2_edges):
    _, _, cache = gpt2_ioi
    edge = gpt2_edges[0]
    sig_u, sig_v = gpt2_tracer.extract_edge_signal_pair_autointerp(
        cache, 0,
        edge.downstream_layer, edge.downstream_ah_idx,
        edge.downstream_dest_token, edge.downstream_src_token,
        edge.upstream_layer, edge.upstream_component_id,
        edge.upstream_dest_token, edge.upstream_src_token,
        edge.edge_type, list(edge.svs_used),
    )
    d_model = gpt2_model.cfg.d_model
    assert sig_u.shape == (d_model,)
    assert sig_v.shape == (d_model,)
    assert torch.isfinite(sig_u).all() and torch.isfinite(sig_v).all()


def test_extract_edge_signal_bad_flavor(gpt2_tracer, gpt2_ioi, gpt2_edges):
    _, _, cache = gpt2_ioi
    edge = gpt2_edges[0]
    with pytest.raises(ValueError):
        gpt2_tracer.extract_edge_signal(
            cache, 0,
            edge.downstream_layer, edge.downstream_ah_idx,
            edge.downstream_dest_token, edge.downstream_src_token,
            edge.upstream_layer, edge.upstream_component_id,
            edge.upstream_dest_token, edge.upstream_src_token,
            edge.edge_type, list(edge.svs_used),
            flavor="bogus",
        )


@pytest.mark.parametrize("edge_type", ["d", "s"])
def test_run_intervention_local(gpt2_tracer, gpt2_ioi, gpt2_edges, edge_type):
    tokens, logits, cache = gpt2_ioi
    edge = _edge_of_type(gpt2_edges, edge_type)

    result = gpt2_tracer.run_intervention(
        tokens, cache, logits, edge, intervention_type="local"
    )
    assert result.logits_interv.shape == logits.shape
    assert result.norm_ratio.shape == (1,)
    assert result.cos_sim.shape == (1,)
    assert torch.isfinite(result.norm_ratio).all()
    # Ablating a traced (positive-contribution) edge should not leave the
    # downstream attention weight exactly unchanged
    assert not torch.allclose(
        result.attn_scores_clean, result.attn_scores_interv
    )
    assert result.delta.abs().sum() > 0


def test_run_intervention_global(gpt2_tracer, gpt2_ioi, gpt2_edges):
    tokens, logits, cache = gpt2_ioi
    n_heads = gpt2_tracer.model.cfg.n_heads
    # Global mode is unsupported for AH offset — pick any other component
    edge = next(
        e for e in gpt2_edges if e.upstream_component_id != n_heads + 3
    )
    result = gpt2_tracer.run_intervention(
        tokens, cache, logits, edge, intervention_type="global"
    )
    assert result.logits_interv.shape == logits.shape
    assert torch.isfinite(result.logits_interv).all()


def test_run_intervention_ah_offset_global_raises(gpt2_tracer, gpt2_ioi, gpt2_edges):
    tokens, logits, cache = gpt2_ioi
    n_heads = gpt2_tracer.model.cfg.n_heads
    offset_edges = [
        e for e in gpt2_edges if e.upstream_component_id == n_heads + 3
    ]
    if not offset_edges:
        pytest.skip("no AH-offset edge in traced graph")
    with pytest.raises(ValueError):
        gpt2_tracer.run_intervention(
            tokens, cache, logits, offset_edges[0], intervention_type="global"
        )
