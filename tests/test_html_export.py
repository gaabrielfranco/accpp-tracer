"""Tests for the interactive HTML circuit export."""

import json
import re

from accpp_tracer.graphs import circuit_to_html

from conftest import IOI_PROMPT, build_idx_to_token


def test_circuit_to_html(tmp_path, gpt2_model, gpt2_prob_graph, gpt2_ioi):
    tokens_t, _, _ = gpt2_ioi
    idx_to_token = build_idx_to_token(gpt2_model, tokens_t)
    tokens = [idx_to_token[i] for i in range(tokens_t.shape[1])]

    out = circuit_to_html(
        gpt2_prob_graph, tokens, n_layers=gpt2_model.cfg.n_layers,
        out_path=tmp_path / "circuit.html",
        title="gpt2 test circuit", subtitle="test",
        stats={"nodes": str(gpt2_prob_graph.number_of_nodes())},
    )
    html = out.read_text()

    # No unfilled template placeholders
    assert not re.search(r"__[A-Z_]+__", html)
    assert "gpt2 test circuit" in html

    # Embedded data parses back and matches the graph
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert m, "embedded DATA block not found"
    data = json.loads(m.group(1))
    assert len(data["nodes"]) == gpt2_prob_graph.number_of_nodes()
    assert len(data["edges"]) == gpt2_prob_graph.number_of_edges()
    assert data["tokens"] == tokens

    # Every node's dest token must be positionable on the token axis
    for n in data["nodes"]:
        if n["kind"] != "root":
            assert n["dest"] in tokens
