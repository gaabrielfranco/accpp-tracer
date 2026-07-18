"""Tests for the Level-1 trace_firing API.

trace_firing carries built-in decomposition-correctness assertions (the sum
of the per-component attention-score breakdown must reproduce the model's
actual attention scores), so simply completing without AssertionError is a
strong correctness check.
"""

import torch

from accpp_tracer import trace_firing

from conftest import GOLDEN_DIR
from golden_utils import assert_trace_firing_matches, trace_firing_to_jsonable

ATTN_THRESH = 0.2


def pick_strong_firing(cache, n_layers, n_heads, end_pos):
    """Deterministically pick the strongest attention firing.

    Chooses (layer, head, src) maximizing the attention weight from end_pos,
    over layers >= 1 (layer 0 has no upstream components to trace) and
    src >= 1 (skip the BOS sink so the trace is non-trivial).
    """
    best = None
    best_w = -1.0
    for layer in range(1, n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][
            0, :, end_pos, 1:end_pos
        ]
        w, flat_idx = pattern.reshape(-1).max(dim=0)
        if w.item() > best_w:
            best_w = w.item()
            head = int(flat_idx.item()) // (end_pos - 1)
            src = int(flat_idx.item()) % (end_pos - 1) + 1
            best = (layer, head, src)
    return best, best_w


def run_firing(model, tracer, cache, layer, head, dest, src):
    with torch.no_grad():
        return trace_firing(
            model, cache, 0, layer, head, dest, src,
            tracer.U, tracer.S, tracer.VT,
            tracer.W_Q_pinv, tracer.W_K_pinv,
            tracer.config, "cpu", ATTN_THRESH,
        )


def _check_result(model, result, layer, dest):
    svs_dest, ew_dest, svs_src, ew_src = result
    n_heads, d_head = model.cfg.n_heads, model.cfg.d_head
    for svs, ew in ((svs_dest, ew_dest), (svs_src, ew_src)):
        assert len(svs) > 0
        for (up_layer, up_comp, up_tok), sv_list in svs.items():
            assert 0 <= up_layer < layer or (up_layer == 0 and up_comp == n_heads + 3)
            assert 0 <= up_comp <= n_heads + 3
            assert 0 <= up_tok <= dest
            assert all(0 <= s < d_head for s in sv_list)
        for k, w in ew.items():
            assert isinstance(w, float) or hasattr(w, "item")


def test_trace_firing_gpt2(gpt2_model, gpt2_tracer, gpt2_ioi, update_golden):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    (layer, head, src), w = pick_strong_firing(
        cache, gpt2_model.cfg.n_layers, gpt2_model.cfg.n_heads, end
    )
    assert w > ATTN_THRESH, "no strong firing found — bad test prompt?"

    result = run_firing(gpt2_model, gpt2_tracer, cache, layer, head, end, src)
    _check_result(gpt2_model, result, layer, end)

    assert_trace_firing_matches(
        trace_firing_to_jsonable(*result),
        GOLDEN_DIR / "trace_firing_gpt2.json",
        update_golden,
    )


def test_trace_firing_pythia(pythia_model, pythia_tracer, pythia_ioi, update_golden):
    """RoPE path: rotation matrices M_d / M_s_all are exercised here."""
    tokens, _, cache = pythia_ioi
    end = tokens.shape[1] - 1
    (layer, head, src), w = pick_strong_firing(
        cache, pythia_model.cfg.n_layers, pythia_model.cfg.n_heads, end
    )
    assert w > ATTN_THRESH, "no strong firing found — bad test prompt?"

    result = run_firing(pythia_model, pythia_tracer, cache, layer, head, end, src)
    _check_result(pythia_model, result, layer, end)

    assert_trace_firing_matches(
        trace_firing_to_jsonable(*result),
        GOLDEN_DIR / "trace_firing_pythia.json",
        update_golden,
    )


def test_residual_shares_sum_to_normalized_stream(gpt2_model, gpt2_ioi):
    """The component decomposition must sum to ln_final.hook_normalized.

    This is the invariant Blocker-2's refactor of _compute_residual_shares
    must preserve (shares sum to the post-LN residual at end_token_pos).
    """
    from accpp_tracer.circuit import _compute_residual_shares
    from accpp_tracer.models import get_model_config

    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    config = get_model_config(gpt2_model)
    r_c = _compute_residual_shares(gpt2_model, config, cache, 0, end, "cpu")

    n_heads = gpt2_model.cfg.n_heads
    assert r_c.shape == (
        gpt2_model.cfg.n_layers, n_heads + 3, end + 1, gpt2_model.cfg.d_model
    )
    total = r_c.sum(dim=(0, 1, 2))
    expected = cache["ln_final.hook_normalized"][0, end]
    max_err = (total - expected).abs().max().item()
    assert max_err < 1e-2, f"residual share decomposition max err {max_err}"
