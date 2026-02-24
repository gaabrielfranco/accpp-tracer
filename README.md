# accpp-tracer

A pip-installable library for **ACC++**, a circuit tracing algorithm for mechanistic
interpretability of transformer attention heads.

ACC++ decomposes attention head firings into upstream contributions using SVD of the
bilinear form $\Omega = W_Q W_K^T$, producing per-prompt circuit graphs that reveal how
information flows through the model.

From the paper: ["Finding Highly Interpretable Prompt-Specific Circuits in Language Models"](https://arxiv.org/abs/2602.13483).

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

### Trace from a pre-computed cache (Level 2 API)

For batch processing (paper reproduction), use `trace_from_cache()`:

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

All experiment scripts are in the companion paper repository. See that repository's
README for full instructions on reproducing the paper results.

## Citation

```bibtex
@article{franco2026finding,
  title={Finding Highly Interpretable Prompt-Specific Circuits in Language Models},
  author={Franco, Gabriel and Tassis, Lucas M and Rohr, Azalea and Crovella, Mark},
  journal={arXiv preprint arXiv:2602.13483},
  year={2026}
}
```
