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

```python
import torch
from transformer_lens import HookedTransformer
from accpp_tracer import Tracer

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=" Mary",
    wrong_token=" John",
)

print(f"Circuit: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### Trace multiple directions simultaneously (Level 3 API)

Pass a list of tokens to trace each as a separate logit direction. All directions share
the same merged graph; overlapping attention head subgraphs are traced only once.
Each direction gets its own root node labelled `("Logit '<token>'", <last_token>)`.

```python
# Explicit list: one root node per token, merged into a single graph
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=[" Mary", " John"],   # two root nodes
)
print(graph.nodes())  # includes ("Logit ' Mary'", "to") and ("Logit ' John'", "to")
```

### Top-p tracing (Level 3 API)

Automatically select the minimum set of top tokens covering `top_p` probability mass
(standard nucleus / top-p definition), then trace each as its own direction.

```python
# Trace all tokens that together cover 90% of the next-token probability mass
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    top_p=0.9,
)
# Each selected token becomes a separate root node in the merged graph
for node in graph.nodes():
    if isinstance(node, tuple) and node[0].startswith("Logit"):
        print(node)  # e.g. ("Logit ' Mary'", "to"), ("Logit ' John'", "to"), ...
```

### Trace from a pre-computed cache (Level 2 API)

For batch processing (paper reproduction), use `trace_from_cache()`. Passing a single
direction and root node works exactly as before:

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

## Citation

```bibtex
@article{franco2026finding,
  title={Finding Interpretable Prompt-Specific Circuits in Language Models},
  author={Franco, Gabriel and Tassis, Lucas M and Rohr, Azalea and Crovella, Mark},
  journal={arXiv preprint arXiv:2602.13483},
  year={2026}
}
```
