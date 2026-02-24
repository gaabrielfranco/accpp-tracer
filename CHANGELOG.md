# Changelog

All notable changes to `accpp-tracer` are documented here.

## [0.1.0] — 2026-02-24

Initial release. Library extracted and refactored from the paper code
(originally split across `sparse-attn-decomposition-research/` and
`interpreting-signals/`).

### Features

- **`Tracer` class** with two public APIs:
  - `trace(prompt, answer_token, wrong_token)` — Level 3: trace a single prompt from a string
  - `trace_from_cache(cache, logit_direction, ...)` — Level 2: trace from a pre-computed activation cache
- **`trace_firing()`** — Level 1: per-firing decomposition (mathematical core, Appendix C)
- **Models supported**: GPT-2 small, Pythia-160m, Gemma-2-2b
  - Handles standard positional embeddings, RoPE, GQA, attention softcapping
  - Handles attention bias (GPT-2, Pythia) and no-bias (Gemma) uniformly
- **Datasets**: IOI, Greater-Than, Gendered Pronoun, Facts (as examples + paper reproduction)
- **Graph utilities**: unification (`graphs/unification.py`), pruning (`graphs/pruning.py`),
  Cytoscape export (`graphs/visualization.py`)
- **Signal extraction**: `Tracer.extract_edge_signal()` for per-edge signal vectors
  (used in autointerpretation pipeline)
- **Runtime shape checking**: opt-in via `pip install accpp-tracer[typecheck]`
  (`beartype` + `jaxtyping`); disable with `ACCPP_TYPECHECK=0`

### Validation

Numerically validated against the original `sparse-attn-decomposition-research` code
for all model/task combinations: GPT-2 (IOI/GT/GP), Pythia-160m (IOI/GT/GP),
Gemma-2-2b (IOI/GP). All differences traced to TransformerLens version changes
(TL 2.16 → 2.17), not algorithmic errors.
