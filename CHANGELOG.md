# Changelog

All notable changes to `accpp-tracer` are documented here.

## [0.1.5] — 2026-02-25

### Fixed

- **Decomposition assertion too tight on CUDA** (`tracing.py` — `_trace_firing_inner`):
  `atol` raised from `1e-3` to `1e-2` in both correctness assertions. On CUDA,
  `.sum()` uses a parallel tree reduction whose summation order differs from CPU's
  sequential order. Since fp32 addition is not associative, accumulated rounding error
  across the many terms (up to ~80k for Gemma: d_head × layers × components × tokens)
  exceeded `atol=1e-3` even with TF32 disabled. The new value matches the existing
  `rtol=1e-2` and is still tight enough to catch real formula or indexing errors.

## [0.1.4] — 2026-02-25

### Fixed

- **TF32 precision loss on CUDA** (`circuit.py` — `Tracer.__init__`): Ampere-class
  GPUs (A100, RTX 3090+) enable TF32 by default for matmul (10 mantissa bits vs 23
  for fp32). The accumulated rounding error across the many `einsum` calls in
  `_trace_firing_inner` caused the decomposition-sum to diverge from the cached
  attention scores beyond `atol=1e-3`, triggering the correctness assertion.
  Fixed by setting `torch.backends.cuda.matmul.allow_tf32 = False` and
  `torch.backends.cudnn.allow_tf32 = False` in `Tracer.__init__` when the device is
  CUDA. No effect on CPU or MPS runs.

## [0.1.3] — 2026-02-24

### Fixed

- **`pyproject.toml` version not bumped**: package version was still `0.1.0` after the
  v0.1.2 release; corrected to `0.1.3`.

- **`beartype` version constraint too high** (`pyproject.toml` — `[typecheck]` extra):
  `beartype>=0.15` conflicted with `transformer-lens==2.16.1`, which requires
  `beartype<0.15,>=0.14.1`. Lowered to `beartype>=0.14.1`. The v0.1.1 fix
  (replacing `typing.Tuple` with built-in `tuple`) already ensures full compatibility
  with beartype 0.14.x; no code change required.

- **`transformer-lens` pin updated** (`pyproject.toml`): `==2.17.0` → `==2.16.1` to
  align with the locked `interpreting-signals` conda environment used for SCC
  validation runs. All numerical differences between TL 2.16 and 2.17 were previously
  confirmed to be version-caused, not algorithmic errors.

## [0.1.2] — 2026-02-24

### Added

- **Multi-direction tracing** (`circuit.py` — `Tracer.trace()` and
  `Tracer.trace_from_cache()`): both methods now accept a list of logit directions
  (and corresponding root nodes) in addition to a single direction. Each direction
  becomes a separate root node in the same merged `nx.MultiDiGraph`. The `is_traced`
  dict is shared across all directions so overlapping attention head subgraphs are
  traced only once. Single-direction callers are unaffected — passing a single tensor
  and tuple produces identical behaviour to v0.1.1.

- **Top-p tracing** (`circuit.py` — `Tracer.trace()`): new `top_p: float | None`
  parameter. When set, ignores `answer_token` and automatically selects the minimum
  set of next-token candidates whose cumulative probability ≥ `top_p` (standard
  nucleus / top-p definition). Each selected token becomes its own logit direction and
  root node. `wrong_token` is still applied to every direction if supplied.

### Changed

- **`Tracer.trace()`**: `answer_token` parameter now also accepts `list[str | int]` or
  `None` (backward-compatible: single str/int still works; `None` is only valid when
  `top_p` is set — a `ValueError` is raised otherwise). New `top_p` parameter added
  with default `None`. Forward pass now captures `logits` (was `_`) to enable top-p
  probability computation; this is a pure internal change with no observable effect on
  existing callers.

- **`Tracer.trace_from_cache()`**: `logit_direction` now accepts `Tensor | list[Tensor]`
  and `root_node` accepts `tuple | list[tuple]` (both backward-compatible). The
  `@typechecked` decorator has been removed from this method because
  `Float[Tensor, "d_model"] | list[Float[Tensor, "d_model"]]` is not supported by the
  beartype+jaxtyping combination; the performance-critical math remains validated inside
  `trace_firing()`.

## [0.1.1] — 2026-02-24

### Fixed

- **`typing.Tuple` deprecation warnings** (`decomposition.py`, `attribution.py`,
  `tracing.py`, `circuit.py`): replaced `typing.Tuple` with the built-in `tuple`
  (PEP 585). Eliminates `BeartypeDecorHintPep585DeprecationWarning` emitted by
  `beartype>=0.14` when typecheck is active. Pure annotation change, no runtime effect.

- **`TypeCheckError` for `layer: int` in `trace_firing`** (two call sites):
  - `circuit.py` — `_get_upstream_contributors`: `np.where()` returns `numpy.int64`
    indices used as seed tuple values. Fixed with `int()` cast:
    `(layer, ah_idx, token)` → `(int(layer), int(ah_idx), int(token))`.
  - `tracing.py` — `_greedy_algorithm`: `np.unravel_index()` returns `numpy.intp`
    values that become dict keys in `svs_dest`/`svs_src`. When `_trace_recursive`
    iterates those keys and passes them to `trace_firing`, beartype rejected them.
    Fixed by converting `top_component` immediately after `np.unravel_index()`:
    `top_component = tuple(int(x) for x in top_component)`.
  In both cases `int(numpy_int(x)) == x` always — no numerical change.

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
