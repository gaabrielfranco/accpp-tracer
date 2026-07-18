"""Llama-3-style architecture smoke test on a tiny random-weight model.

Neither gpt2 nor pythia-160m exercises GQA, RMSNorm, or NTK-by-parts RoPE —
the three Llama-3 specifics. This synthetic model does, at trivial size.
The decomposition assertions inside trace_firing (breakdown must sum to the
model's actual attention scores) verify that our rotation matrices and GQA
value-repeat handling agree with TransformerLens's own attention forward
pass, which is exactly what a real Llama-3.x trace relies on.
"""

import pytest
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig

from accpp_tracer import Tracer, get_seeds, trace_firing
from accpp_tracer.models import get_model_config


@pytest.fixture(scope="module")
def llama_like_model():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=64,
        n_ctx=32,
        d_head=16,
        n_heads=8,
        n_key_value_heads=4,
        d_mlp=128,
        d_vocab=101,
        act_fn="silu",
        normalization_type="RMSPre",
        positional_embedding_type="rotary",
        rotary_dim=16,
        rotary_base=10000,
        use_NTK_by_parts_rope=True,
        NTK_by_parts_factor=8.0,
        NTK_by_parts_low_freq_factor=1.0,
        NTK_by_parts_high_freq_factor=4.0,
        NTK_original_ctx_len=8192,
        attn_only=False,
        seed=0,
        device="cpu",
    )
    return HookedTransformer(cfg)


@pytest.fixture(scope="module")
def llama_like_setup(llama_like_model):
    model = llama_like_model
    tracer = Tracer(model, device="cpu")
    torch.manual_seed(0)
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 12))
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    return model, tracer, tokens, logits, cache


def test_config_flags(llama_like_model):
    config = get_model_config(llama_like_model)
    assert config.has_rope
    assert config.has_gqa
    assert config.gqa_repeats == 2


def test_trace_firing_gqa_ntk_rope(llama_like_setup):
    """trace_firing's internal decomposition assertions are the real check."""
    model, tracer, tokens, _, cache = llama_like_setup
    end = tokens.shape[1] - 1

    n_traced = 0
    with torch.no_grad():
        for layer in range(1, model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][
                    0, head, end, : end + 1
                ]
                src = int(pattern.argmax().item())
                svs_dest, ew_dest, svs_src, ew_src = trace_firing(
                    model, cache, 0, layer, head, end, src,
                    tracer.U, tracer.S, tracer.VT,
                    tracer.W_Q_pinv, tracer.W_K_pinv,
                    tracer.config, "cpu", 0.05,
                    R_stack=tracer._rotation_stack(end + 1),
                )
                n_traced += 1
                assert len(svs_dest) > 0 and len(svs_src) > 0
    assert n_traced == (model.cfg.n_layers - 1) * model.cfg.n_heads


def test_residual_shares_gqa(llama_like_setup):
    """Component shares must sum to the normalized stream (GQA V-mapping)."""
    from accpp_tracer.circuit import _compute_residual_shares

    model, tracer, tokens, _, cache = llama_like_setup
    end = tokens.shape[1] - 1
    r_c = _compute_residual_shares(
        model, tracer.config, cache, 0, end, "cpu"
    )
    total = r_c.sum(dim=(0, 1, 2))
    expected = cache["ln_final.hook_normalized"][0, end]
    assert (total - expected).abs().max().item() < 1e-2


def test_trace_from_cache_smoke(llama_like_setup):
    model, tracer, tokens, logits, cache = llama_like_setup
    end = tokens.shape[1] - 1
    idx_to_token = {i: f"t{i}" for i in range(end + 1)}
    direction = model.W_U[:, int(logits[0, end].argmax())].detach()
    G = tracer.trace_from_cache(
        cache=cache,
        logit_direction=direction,
        end_token_pos=end,
        idx_to_token=idx_to_token,
        root_node=("Logit direction", idx_to_token[end]),
        attn_weight_thresh=0.1,
    )
    # Random weights: the graph may legitimately be empty; the check is that
    # seeding + recursion run the full pipeline without violating any
    # internal decomposition assertion.
    assert G is not None
