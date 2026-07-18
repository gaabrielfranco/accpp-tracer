"""Tests for the Tracer Level-2/3 APIs: trace, trace_from_cache,
trace_from_probe, seeding, signals, and golden graph regression."""

import networkx as nx
import pytest
import torch

from accpp_tracer import Tracer, get_seeds, get_seeds_prob
from accpp_tracer.models import get_model_config

from conftest import GOLDEN_DIR, IOI_PROMPT, build_idx_to_token
from golden_utils import assert_graph_matches


def _internal_nodes(G):
    return [n for n in G.nodes if isinstance(n, tuple) and len(n) == 4
            and isinstance(n[0], int)]


def _check_graph_structure(G):
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_edges() > 0
    for u, v, data in G.edges(data=True):
        assert "weight" in data
        assert data.get("type") in ("d", "s")
        assert data.get("color") in ("#E41A1C", "#377EB8")


def test_trace_prob_seeding(gpt2_prob_graph):
    G = gpt2_prob_graph
    _check_graph_structure(G)
    roots = [n for n in G.nodes if isinstance(n[0], str) and n[0].startswith("Prob")]
    # The prompt has two " to" tokens; the end token gets a " (1)" suffix
    assert roots == [("Prob ' Mary'", " to (1)")]
    assert len(_internal_nodes(G)) > 0
    # Traced AH nodes carry the attention weight (seed nodes that fall below
    # the threshold or sit at layer 0 are never visited and legitimately
    # lack it); every stored value must be a valid attention weight.
    weights = [
        G.nodes[n]["attn_weight"]
        for n in _internal_nodes(G)
        if isinstance(n[1], int) and "attn_weight" in G.nodes[n]
    ]
    assert len(weights) > 0
    assert all(0.0 <= w <= 1.0 for w in weights)


def test_trace_linear_seeding(gpt2_linear_graph):
    G = gpt2_linear_graph
    _check_graph_structure(G)
    assert any(
        isinstance(n[0], str) and n[0] == "Logit direction" for n in G.nodes
    )


def test_trace_pythia_topk(pythia_topk_graph):
    _check_graph_structure(pythia_topk_graph)


def test_trace_argument_validation(gpt2_tracer):
    with pytest.raises(ValueError):
        gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John")
    with pytest.raises(ValueError):
        gpt2_tracer.trace(IOI_PROMPT)  # no objective
    with pytest.raises(ValueError):
        gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", top_k=2)
    with pytest.raises(ValueError):
        gpt2_tracer.trace(
            IOI_PROMPT, answer_token=" Mary", seeding="linear", tau=0.5
        )


def test_trace_from_cache_linear(gpt2_model, gpt2_tracer, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx_to_token = build_idx_to_token(gpt2_model, tokens)
    io_id = gpt2_model.to_single_token(" Mary")
    s_id = gpt2_model.to_single_token(" John")
    direction = gpt2_model.W_U[:, io_id] - gpt2_model.W_U[:, s_id]

    G = gpt2_tracer.trace_from_cache(
        cache=cache,
        logit_direction=direction,
        end_token_pos=end,
        idx_to_token=idx_to_token,
        root_node=("Logit direction", idx_to_token[end]),
    )
    _check_graph_structure(G)


def test_trace_from_cache_validation(gpt2_tracer, gpt2_ioi, gpt2_model):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx_to_token = build_idx_to_token(gpt2_model, tokens)
    with pytest.raises(ValueError):
        gpt2_tracer.trace_from_cache(
            cache=cache, logit_direction=None, end_token_pos=end,
            idx_to_token=idx_to_token, root_node=("r", "x"),
        )  # neither objective


def test_trace_from_probe(gpt2_model, gpt2_tracer, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx_to_token = build_idx_to_token(gpt2_model, tokens)
    probe_direction = gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]

    G = gpt2_tracer.trace_from_probe(
        cache=cache,
        probe_direction=probe_direction,
        layer=8,
        end_token_pos=end,
        idx_to_token=idx_to_token,
        root_node=("Probe", idx_to_token[end]),
    )
    assert isinstance(G, nx.MultiDiGraph)
    if G.number_of_edges() > 0:
        # All internal nodes must be at layers <= probe layer
        for n in _internal_nodes(G):
            assert n[0] <= 8


def test_get_seeds_linear(gpt2_model, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    config = get_model_config(gpt2_model)
    io_id = gpt2_model.to_single_token(" Mary")
    s_id = gpt2_model.to_single_token(" John")
    direction = (gpt2_model.W_U[:, io_id] - gpt2_model.W_U[:, s_id]).detach()

    seeds, contrib = get_seeds(
        gpt2_model, config, cache, 0, direction, end, "cpu"
    )
    assert len(seeds) > 0
    assert set(contrib) == set(seeds)
    n_heads = gpt2_model.cfg.n_heads
    for layer, ah_idx, token in seeds:
        assert 0 <= layer < gpt2_model.cfg.n_layers
        assert 0 <= ah_idx < n_heads + 3
        assert 0 <= token <= end


def test_get_seeds_prob(gpt2_model, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    config = get_model_config(gpt2_model)
    io_id = gpt2_model.to_single_token(" Mary")

    seeds, contrib = get_seeds_prob(
        gpt2_model, config, cache, 0, [io_id], end, "cpu"
    )
    assert len(seeds) > 0
    assert all(contrib[s] > 0 for s in seeds)

    with pytest.raises(ValueError):
        get_seeds_prob(gpt2_model, config, cache, 0, [], end, "cpu")
    with pytest.raises(ValueError):
        get_seeds_prob(
            gpt2_model, config, cache, 0, [io_id, io_id], end, "cpu"
        )


def test_signals_stored_on_edges(gpt2_tracer, gpt2_model):
    G = gpt2_tracer.trace(
        IOI_PROMPT, answer_token=" Mary", signals="rotated_normalized",
        attn_weight_thresh=0.3,
    )
    non_root_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if isinstance(v, tuple) and isinstance(v[0], int)
    ]
    assert len(non_root_edges) > 0
    for u, v, d in non_root_edges:
        assert d.get("signal_flavor") == "rotated_normalized"
        sig = d.get("signal")
        assert sig is not None
        assert sig.shape == (gpt2_model.cfg.d_model,)
        assert torch.isfinite(sig).all()


def test_tracer_disk_cache(tmp_path, gpt2_model, gpt2_tracer):
    t1 = Tracer(gpt2_model, device="cpu", cache_dir=str(tmp_path))
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".h5"
    # Cached tensors must match the freshly-computed session tracer's
    assert torch.allclose(t1.S, gpt2_tracer.S, atol=1e-5)

    t2 = Tracer(gpt2_model, device="cpu", cache_dir=str(tmp_path))
    assert torch.equal(t1.U, t2.U)
    assert torch.equal(t1.W_K_pinv, t2.W_K_pinv)


def test_golden_graph_gpt2_prob(gpt2_prob_graph, update_golden):
    assert_graph_matches(
        gpt2_prob_graph, GOLDEN_DIR / "graph_gpt2_prob.json", update_golden
    )


def test_golden_graph_gpt2_linear(gpt2_linear_graph, update_golden):
    assert_graph_matches(
        gpt2_linear_graph, GOLDEN_DIR / "graph_gpt2_linear.json", update_golden
    )


def test_golden_graph_pythia_topk(pythia_topk_graph, update_golden):
    assert_graph_matches(
        pythia_topk_graph, GOLDEN_DIR / "graph_pythia_topk.json", update_golden
    )


def test_trace_qwen_prob(qwen_prob_graph):
    _check_graph_structure(qwen_prob_graph)


def test_golden_graph_qwen_prob(qwen_prob_graph, update_golden):
    assert_graph_matches(
        qwen_prob_graph, GOLDEN_DIR / "graph_qwen_prob.json", update_golden
    )
