"""Shared fixtures for the accpp_tracer test suite.

All models run on CPU in fp32 for determinism. Model weights come from the
local HuggingFace cache (gpt2 and EleutherAI/pythia-160m).

Golden regression files live in tests/golden/. Regenerate them with:

    pytest --update-golden

Run this ONCE on a known-good revision; subsequent runs compare against the
stored goldens (SVD-sign-invariant, small float tolerance).
"""

from pathlib import Path

import pytest
import torch

GOLDEN_DIR = Path(__file__).parent / "golden"

IOI_PROMPT = "When Mary and John went to the store, John gave a drink to"


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden reference files instead of comparing.",
    )


@pytest.fixture(scope="session")
def update_golden(request):
    GOLDEN_DIR.mkdir(exist_ok=True)
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="session")
def gpt2_model():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def pythia_model():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        "EleutherAI/pythia-160m", device="cpu"
    )


@pytest.fixture(scope="session")
def qwen_model():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B", device="cpu")


@pytest.fixture(scope="session")
def gpt2_tracer(gpt2_model):
    from accpp_tracer import Tracer

    return Tracer(gpt2_model, device="cpu")


@pytest.fixture(scope="session")
def qwen_tracer(qwen_model):
    from accpp_tracer import Tracer

    return Tracer(qwen_model, device="cpu")


@pytest.fixture(scope="session")
def pythia_tracer(pythia_model):
    from accpp_tracer import Tracer

    return Tracer(pythia_model, device="cpu")


@pytest.fixture(scope="session")
def gpt2_ioi(gpt2_model):
    """(tokens, logits, cache) for the IOI prompt on gpt2."""
    tokens = gpt2_model.to_tokens(IOI_PROMPT)
    with torch.no_grad():
        logits, cache = gpt2_model.run_with_cache(tokens)
    return tokens, logits, cache


@pytest.fixture(scope="session")
def pythia_ioi(pythia_model):
    tokens = pythia_model.to_tokens(IOI_PROMPT)
    with torch.no_grad():
        logits, cache = pythia_model.run_with_cache(tokens)
    return tokens, logits, cache


@pytest.fixture(scope="session")
def qwen_ioi(qwen_model):
    tokens = qwen_model.to_tokens(IOI_PROMPT)
    with torch.no_grad():
        logits, cache = qwen_model.run_with_cache(tokens)
    return tokens, logits, cache


# ---------------------------------------------------------------------------
# Traced graphs (expensive — traced once per session, asserted in many tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpt2_prob_graph(gpt2_tracer):
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary")


@pytest.fixture(scope="session")
def gpt2_linear_graph(gpt2_tracer):
    return gpt2_tracer.trace(
        IOI_PROMPT, answer_token=" Mary", wrong_token=" John", seeding="linear"
    )


@pytest.fixture(scope="session")
def pythia_topk_graph(pythia_tracer):
    return pythia_tracer.trace(IOI_PROMPT, top_k=1)


@pytest.fixture(scope="session")
def qwen_prob_graph(qwen_tracer):
    """RoPE + GQA + large QK biases: the only fixture where the AH bias /
    AH offset components are strongly active."""
    return qwen_tracer.trace(IOI_PROMPT, answer_token=" Mary")


def build_idx_to_token(model, tokens):
    """Reproduce the idx_to_token construction from Tracer.trace()."""
    from collections import defaultdict

    idx_to_token = {}
    count_dict = defaultdict(int)
    for i in range(tokens.shape[1]):
        tok_str = model.tokenizer.decode(tokens[0, i])
        count = count_dict[tok_str]
        idx_to_token[i] = f"{tok_str} ({count})" if count > 0 else tok_str
        count_dict[tok_str] += 1
    return idx_to_token
