# accpp-tracer

A pip-installable library for **ACC++**, a circuit tracing algorithm for mechanistic
interpretability of transformer attention heads.

ACC++ decomposes attention head firings into upstream contributions using SVD of the
bilinear form $\Omega = W_Q W_K^T$, producing per-prompt circuit graphs that reveal how
information flows through the model.

From the paper: ["Finding Interpretable Prompt-Specific Circuits in Language Models"](https://arxiv.org/abs/2602.13483).

## Installation

Requires Python 3.10.

```bash
# Install the library
pip install -e .

# Or with runtime shape checking (beartype + jaxtyping)
pip install -e ".[typecheck]"
```

For exact numerical reproducibility of the paper results, use the pinned environment:

```bash
pip install -r requirements.txt
```

## Quick Start

### Trace a single prompt (Level 3 API)

Since v0.3.0 the default seeding is **probability-aware** (`seeding="prob"`): seeds
are the minimal set of upstream components whose removal from the final residual
stream destroys a `tau` fraction (default 0.8) of the model's log-likelihood of a
target-token support `T`, measured in full-vocabulary probabilities against the
bias-only baseline. The traced graph always has exactly **one root node**.

```python
import torch
from transformer_lens import HookedTransformer
from accpp_tracer import Tracer

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

# Single-token support: T = {" Mary"}
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=" Mary",
)

print(f"Circuit: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

The support `T` is chosen via **exactly one** of `answer_token`, `top_p`, or `top_k`:

```python
# Explicit multi-token support T = {" Mary", " John"} — one objective, one root node
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=[" Mary", " John"],
)
# Root node: ("Prob {' Mary', ' John'}", " to (1)")

# Nucleus support: smallest token set covering 90% of the clean probability mass
graph = tracer.trace(prompt="...", top_p=0.9)

# Top-k support: the k most likely next tokens
graph = tracer.trace(prompt="...", top_k=2)
```

Knobs: `tau` (fraction of the completeness to destroy, default 0.8) and `ig_steps`
(integrated-gradients quadrature intervals, default 64).

> **Note (Gemma-2)**: the seeding objective is built from uncapped logits and ignores
> the final logit soft-cap. Token ranking is unaffected (the cap is monotone), but
> probability and attribution values are computed on the uncapped distribution; a
> `UserWarning` is emitted.

### Linear (contrastive) seeding — pre-0.3.0 behavior

`seeding="linear"` reproduces the pre-0.3.0 behavior exactly: seeds are selected by
linear attribution of a logit direction, and each answer token (or each top-p token)
becomes its own direction and root node. This is the mode used for paper reproduction,
and the only mode that supports `wrong_token` (a contrastive *probability* objective
would be mathematically identical to the linear logit-diff direction, so it is not a
separate method).

```python
# IO - S contrastive direction, root node ("Logit direction", ...)
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=" Mary",
    wrong_token=" John",
    seeding="linear",
)

# One direction and root node per token
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=[" Mary", " John"],   # two root nodes
    seeding="linear",
)
```

### Trace from a pre-computed cache (Level 2 API)

For batch processing (paper reproduction), use `trace_from_cache()`. The seeding mode
is selected by which objective argument is given — exactly one of `logit_direction`
(linear) or `target_tokens` (probability-aware):

```python
# Probability-aware at Level 2: pass token ids and a single root node.
# Clean logits are recomputed from the cache; q_star defaults to the clean
# probabilities renormalized within the support (pass q_star to override).
graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=None,
    end_token_pos=end_pos,
    idx_to_token=idx_to_token,
    root_node=("Prob ' Mary'", idx_to_token[end_pos]),
    prompt_idx=prompt_id,
    target_tokens=[io_token_id],          # or [io_token_id, s_token_id], ...
)
```

Passing a single direction and root node works exactly as before:

```python
from accpp_tracer import Tracer
from accpp_tracer.datasets import IOIDataset

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

dataset = IOIDataset(
    model_family="gpt2", prompt_type="mixed", N=8,
    tokenizer=model.tokenizer, prepend_bos=False, seed=0, device="cpu",
)

logits, cache = model.run_with_cache(dataset.toks)

prompt_id = 0
logit_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]]] \
          - model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]]

graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=logit_dir,
    end_token_pos=dataset.word_idx["end"][prompt_id].item(),
    idx_to_token={i: model.tokenizer.decode(dataset.toks[prompt_id, i])
                  for i in range(dataset.word_idx["end"][prompt_id].item() + 1)},
    root_node=("IO-S direction", "to"),
    prompt_idx=prompt_id,
)
```

Pass lists to `trace_from_cache()` for multi-direction tracing at Level 2:

```python
io_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]]]
s1_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]]

graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=[io_dir, s1_dir],           # two directions
    end_token_pos=dataset.word_idx["end"][prompt_id].item(),
    idx_to_token={i: model.tokenizer.decode(dataset.toks[prompt_id, i])
                  for i in range(dataset.word_idx["end"][prompt_id].item() + 1)},
    root_node=[("IO direction", "to"), ("S1 direction", "to")],  # one label per direction
    prompt_idx=prompt_id,
)
```

### Caching the Omega SVD on disk

The Tracer recomputes the Omega SVD (``U, S, VT``) and weight pseudoinverses
(``W_Q_pinv, W_K_pinv``) on every instantiation — a few seconds for GPT-2 /
Pythia, noticeably longer for Gemma-2-2b. Pass ``cache_dir`` to save these
tensors to disk on the first run and reuse them on subsequent runs:

```python
tracer = Tracer(model, cache_dir="~/.cache/accpp_tracer")
# First call: computes the SVD, writes
#   ~/.cache/accpp_tracer/{model_name}_torch.h5
# Subsequent calls: loads from disk, skips SVD.
```

Cache files are gzip-compressed h5 (fp32) and reusable across processes. One
file per ``(model_name, use_numpy_svd)`` pair. Sizes: ~105 MB for GPT-2 /
Pythia, ~1.8 GB for Gemma-2-2b. Default (``cache_dir=None``) recomputes
every time.

### Causal interventions

Once a circuit is traced, `Tracer.run_intervention()` ablates (or boosts) one edge of
the circuit and returns the perturbed logits, cache, and per-prompt metrics. Two modes:

- `"local"` — modify the LN-normalized Q/K input of the downstream head only.
  Surgical: affects just the targeted edge's downstream head.
- `"global"` — modify the upstream component's output directly in the residual
  stream. Broad: affects every downstream consumer. The AH **offset** component
  has no residual-stream hook point and is not supported here.

Centering is auto-detected from `model.cfg.normalization_type` (LN → center,
RMS → don't); pass `center=True/False` to override.

```python
from accpp_tracer import edges_from_graph, InterventionResult

# Re-use model, tracer, dataset, prompt_id, idx_to_token, graph, logits, cache
# from the `trace_from_cache` example above.
token_to_idx = {label: idx for idx, label in idx_to_token.items()}
edges = edges_from_graph(graph, token_to_idx, n_heads=model.cfg.n_heads)

result: InterventionResult = tracer.run_intervention(
    tokens=dataset.toks,            # (batch, seq)
    cache=cache,                    # clean ActivationCache
    logits=logits,                  # (batch, seq, d_vocab)
    edge=edges[0],                  # one EdgeSpec
    prompt_idx=prompt_id,
    intervention_type="local",      # or "global"
    boost=False,                    # False = ablate, True = boost
)

# result.logits_interv        (batch, seq, d_vocab)
# result.interv_cache         intervention ActivationCache
# result.delta                (batch, seq, d_model) — applied delta
# result.norm_ratio           (batch,) at intervention position
# result.cos_sim              (batch,) at intervention position
# result.attn_scores_clean    (batch,) at downstream (dest, src)
# result.attn_scores_interv   (batch,) at downstream (dest, src)
```

The API is intentionally one-edge-per-call: multi-edge experiments loop over `edges`
and compose at the call site.

## Supported Models

| Model | Positional encoding | Notes |
|-------|---------------------|-------|
| `gpt2-small` | Standard | Attention bias |
| `EleutherAI/pythia-160m` | RoPE | Attention bias, parallel attn+MLP |
| `gemma-2-2b` | RoPE | GQA, attention softcapping, no bias |
| `meta-llama/Llama-3.1-8B` | RoPE (NTK-by-parts) | GQA, no QK bias; fp32 weights ~30 GB — needs a large-memory GPU or high-RAM CPU |
| `Qwen/Qwen2.5-0.5B` … `7B` | RoPE | GQA (up to 7× KV sharing), large QK biases (AH bias / AH offset components are strongly active) |

### Prompt length limit

`trace_firing` keeps a per-singular-vector attention-score breakdown of shape
`(layer, n_heads + 4, d_head, T, T)` for both the query and key sides. This
T² scaling is inherent to the per-SV granularity of the method (the greedy
selection needs the full tensor), and bounds the usable prompt length: at
Llama-3.1-8B scale it is ~1.1 MB x T² at the deepest layer (~1 GB at T=30,
~19 GB at T=128). On an 80 GB device, prompts up to roughly 100-150 tokens
are practical; IOI-style prompts are far below this. If longer prompts are
ever needed, chunk over upstream layers or store per-SV scores sparsely
after thresholding.

## API levels

The library exposes three levels of abstraction:

| Level | Function / class | Description |
|-------|-----------------|-------------|
| 1 | `trace_firing()` | Decomposes one attention firing into upstream contributions (mathematical core) |
| 2 | `Tracer.trace_from_cache()` | Full circuit from a pre-computed activation cache |
| 3 | `Tracer.trace()` | End-to-end: string prompt → circuit graph |

## Paper reproduction

The companion repository contains all experiment scripts and shell pipelines that
reproduce the paper's figures and tables:

**https://github.com/gaabrielfranco/finding-highly-interpretable-circuits**

It has a pinned copy of `accpp_tracer` under `lib/accpp_tracer/` and pins
TransformerLens to the version used for the paper. A single `pip install -e .` from
the repo root installs everything. The pipelines cover:

- **Tracing & figures** — paper §2, appendices B–D
- **Causal interventions** — appendix E
- **Clustering & signals** — §3, appendix F
- **Autointerpretation** — quantitative and qualitative tracks

See the companion repo's README for run instructions.

## Interactive circuit visualization

`accpp_tracer.graphs.circuit_to_html` renders a traced graph as a single
self-contained HTML file — no external dependencies or network access, so the
file can be opened locally, attached to a paper, or shared as-is:

```python
from accpp_tracer.graphs import circuit_to_html

graph = tracer.trace(prompt, answer_token=" Mary")

# Token labels must match the graph's node labels — the same construction
# Tracer.trace uses internally (duplicate tokens get " (1)", " (2)", ...).
circuit_to_html(
    graph,
    tokens=tokens,                       # position-ordered token labels
    n_layers=model.cfg.n_layers,
    out_path="circuit.html",
    title="Llama-3.2-3B — traced IOI circuit",
    subtitle="objective: P(' Mary') at the final position",  # HTML allowed
    stats={"P( Mary)": "0.387", "nodes": "63"},              # header chips
)
```

The page lays the circuit out with token positions on the x-axis and layers
on the y-axis (each head sits at its *destination* token; the root objective
sits above the final position). Component kinds get distinct shapes and
colors (attention heads, MLPs, embeddings, attention biases, AH offsets —
only kinds present appear in the legend); query-side (`d`) and key-side
(`s`) edges are distinguished by color *and* dash pattern; line width scales
with edge weight. Hovering any node or edge shows details (attention
weights, per-edge singular-vector channels); clicking a node pins its
incident edges; a collapsible table lists every edge sorted by weight. Light
and dark color schemes are both supported (OS preference, or set
`data-theme="dark"`/`"light"` on the document root).

## Test suite

`tests/` contains a fast regression suite (~3 minutes, CPU-only) that
exercises every public API on two small models from the local HF cache —
**gpt2** (LayerNorm, absolute positions) and **EleutherAI/pythia-160m**
(RoPE) — plus a tiny random-weight Llama-3-style model (GQA + RMSNorm +
NTK-by-parts RoPE) that covers the architecture features the real test
models lack.

```bash
pip install -e ".[dev,typecheck]"
pytest                    # run everything, compare against tests/golden/
pytest --update-golden    # re-baseline golden files (see below)
```

Two kinds of tests:

1. **Invariant tests** — mathematical properties that hold for any model and
   survive any refactor: the Omega SVD reconstructs `W_Q @ W_K^T`, component
   residual shares sum to the LN-normalized stream, RoPE matrices are
   orthogonal and match TransformerLens's own sin/cos tables, and
   `trace_firing`'s internal decomposition assertions pass.
2. **Golden regression tests** — outputs captured in `tests/golden/` from a
   known-good revision: SVD fingerprints (sign-invariant random-projection
   sketches), rotation-matrix stacks, `trace_firing` results, and fully
   serialized traced graphs. Comparisons require exact topology and
   singular-vector selection, with a small float tolerance (5e-3) on weights
   to absorb fp32 contraction-reorder noise.

Workflow: for a behavior-preserving change (memory/speed refactor), all
tests must pass against the existing goldens — do not regenerate them. For
an intended behavioral change, inspect the golden diffs, confirm they match
the intent, then re-baseline with `pytest --update-golden` and commit the
new goldens together with the code. `tests/README.md` has the full details,
including how to add a new model to the suite.

## Citation

```bibtex
@article{franco2026finding,
  title={Finding Interpretable Prompt-Specific Circuits in Language Models},
  author={Franco, Gabriel and Tassis, Lucas M and Rohr, Azalea and Crovella, Mark},
  journal={arXiv preprint arXiv:2602.13483},
  year={2026}
}
```
