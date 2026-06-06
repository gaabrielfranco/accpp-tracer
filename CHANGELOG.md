# Changelog

All notable changes to `accpp-tracer` are documented here.

## [0.2.2] — 2026-06-05

Intervention API redesign — same math, ground-truth structure. Replaces the
0.2.1 `compute_edge_intervention()` + `intervene()` pair with a single
`Tracer.run_intervention()` method whose body mirrors the ground-truth
`run_intervention()` function in `new-code/experiments/interventions.py`
step-by-step. The signal computation (project → center → divide-by-LN) is now
**inlined explicitly** rather than dispatched through
`extract_edge_signal(flavor="normalized")` + `get_component_output(ln_normalize=True)`,
making the math directly auditable against the verified ground truth.

### Breaking changes

- **`Tracer.compute_edge_intervention()`** removed.
- **`Tracer.intervene()`** removed.
- **`Tracer._global_hook_name()`** removed (replaced by module-level
  `_global_hook_name()` helper).
- **`accpp_tracer.intervention` module**: the module-level hook helper names
  (`_local_q_hook`, `_local_k_hook`, `_global_residual_hook`) moved from
  `circuit.py` (they were already private; mention only for completeness).

### Added

- **`Tracer.run_intervention()`** (`circuit.py`): single-edge,
  single-prompt-targeted intervention. Takes one `EdgeSpec`, the clean
  `tokens` / `cache` / `logits`, a `prompt_idx`, and an `intervention_type`
  (`"local"` or `"global"`); returns an `InterventionResult` dataclass.
  Math flow follows the ground truth's Steps 1–7 verbatim:
  1. Get singular-vector indices (from `EdgeSpec.svs_used`).
  2. Build the projection `P = B_s @ B_s.T`.
  3. Compute the upstream component output (case dispatch by
     `upstream_component_id`: AH → `A * V @ W_O` with optional Gemma post-attn
     LN; MLP → `hook_mlp_out`; AH bias → `b_O`; embedding → `hook_resid_pre`;
     AH offset → `c_d @ M_d` / `M_s @ c_s` with RoPE at the intervention
     position).
  3.1. `signal = upstream_out @ P`.
  3.2. Center (LN-pre / LN only; auto-detected from `normalization_type`).
  3.3. Divide by `ln1.hook_scale[pos_interv]` (local only).
  3.4. Write into the delta tensor at `(prompt_idx, pos_interv, :)`.
  4. Run the model via `_run_local_intervention()` (Q/K hook + recompute)
     or `_run_global_intervention()` (residual hook).
  6. Compute `norm_ratio` / `cos_sim` at the intervention position.
  7. Compute `attn_scores_clean` / `attn_scores_interv` at the downstream
     head's `(dest, src)` cell.

- **`Tracer._run_local_intervention()`** (private helper): mirrors
  `run_local_intervention()` from the ground truth. Builds `q_interv` (for
  `"d"`) or `k_interv` (for `"s"`) via `F.linear`, installs the
  `hook_q` / `hook_k` patch hook, and runs `model.run_with_cache(tokens)` in
  a `model.hooks(...)` context for clean teardown.

- **`Tracer._run_global_intervention()`** (private helper): mirrors
  `run_global_intervention()` from the ground truth. Installs an
  add/subtract hook on `hook_attn_out` (AH / AH bias), `hook_mlp_out` (MLP),
  or `blocks.0.hook_resid_pre` (embedding); runs with cache.

- **`InterventionResult`** dataclass (`intervention.py`): carries
  `logits_clean`, `logits_interv`, `interv_cache`, `delta`, `norm_ratio`,
  `cos_sim`, `attn_scores_clean`, `attn_scores_interv`. All per-prompt
  metrics are tensors of shape `(n_prompts,)`; only the entry at the
  targeted `prompt_idx` is affected by the intervention.

### Notes

- **No math change vs. 0.2.1**: the underlying computation
  (project → center → divide-by-LN, with the AH offset rotation exception)
  is the same. The change is structural: inlined and laid out to match the
  verified ground truth, with explicit Step 1–7 comments. The signal
  computation no longer goes through `extract_edge_signal` /
  `get_component_output`, so the intervention path can be audited
  independently of the autointerp signal API.
- **`extract_edge_signal()` and `extract_edge_signal_pair_autointerp()`**
  are unchanged. They remain the entry points for the autointerp signal
  flavors (`rotated_normalized`, `normalized`, `raw`).
- **API trade-off**: 0.2.1's batched `intervene(edges: list[EdgeSpec])`
  is replaced by single-edge `run_intervention(edge: EdgeSpec)`. Multi-edge
  experiments now loop in caller code (one forward pass per edge),
  matching the ground truth's iteration pattern.

## [0.2.1] — 2026-06-04

Causal intervention API (item 1 of the `interventions_api` branch). Extracts the
task-agnostic core of `new-code/experiments/interventions.py` (~850 lines)
into a library feature so users can ablate or boost individual circuit edges
without re-implementing the math each time. Builds directly on the signal
flavors introduced in 0.2.0 (`"normalized"` for local, `"raw"` for global).

### Added

- **`accpp_tracer.intervention` module** (new):
  - `EdgeSpec` — frozen dataclass holding a fully-resolved circuit edge
    specification (integer token positions, component ID, edge type, list of
    singular-vector indices).
  - `edges_from_graph(graph, token_to_idx, n_heads, edges=None)` — parses
    graph edges produced by `Tracer.trace()` into `EdgeSpec` objects. Skips
    root / output edges automatically.

- **`Tracer.compute_edge_intervention()`** (`circuit.py`): returns the
  per-target delta tensors that an intervention would apply, without
  running the model. Useful for inspection / debugging.

- **`Tracer.intervene()`** (`circuit.py`): runs the model with intervention
  hooks. Two modes:
  - `intervention_type="local"`: modifies the LN-normalized attention input,
    recomputes Q (for `"d"` edges) or K (for `"s"` edges), and hooks
    `hook_q` / `hook_k` for the targeted head only. Surgical.
  - `intervention_type="global"`: modifies the upstream component's output
    in the residual stream via `hook_attn_out` / `hook_mlp_out` /
    `hook_resid_pre`. Broader effect.

  AH offset + global is unsupported (no hook point exists for a
  bias-projection term) and raises `ValueError`. Edges are applied
  cumulatively in a single forward pass; per-target deltas accumulate.

### Notes

- **Signal flavor mapping**: local intervention consumes
  `flavor="normalized"` (LN-normalized, unrotated except AH offset); global
  intervention consumes `flavor="raw"` (residual-stream-space, unrotated
  except AH offset).
- **Centering policy**: auto-detected from `model.cfg.normalization_type`.
  `LN`/`LNPre` (GPT-2, Pythia) → center; `RMS`/`RMSPre` (Gemma-2) → don't.
  Override via `center=True/False`.
- **GQA support**: K-hook indexing uses `ah_idx // config.gqa_repeats` for
  Gemma-2-2b (n_kv_heads < n_heads); transparent on non-GQA models.
- **Backward-compatible numerics**: the delta tensors match (within fp32
  rounding) what `new-code/experiments/interventions.py:run_intervention()`
  produces on the same inputs.

## [0.2.0] — 2026-06-04

Breaking signal-API redesign. Three signal "flavors" are now first-class citizens
of the library, addressing the fact that analysis (autointerp), causal intervention,
and MLP upstream tracing each need the per-edge signal in a *different* space.

### Breaking changes

- **`Tracer.extract_edge_signal()`** (`circuit.py`):
  - Now requires a keyword argument ``flavor: Literal["rotated_normalized",
    "normalized", "raw"]`` (no default). The three values map to:
    - ``"rotated_normalized"`` — LN-normalized + RoPE-rotated (analysis / autointerp
      flavor; matches pre-0.2.0 behavior).
    - ``"normalized"`` — LN-normalized, unrotated (intervention flavor).
    - ``"raw"`` — raw residual-stream-space output (MLP upstream tracing flavor).
  - Return type changed from ``tuple[Tensor, Tensor]`` (``(signal_u, signal_v)``)
    to a single ``Tensor`` — the **primary** signal matching the edge type
    (``signal_u`` for ``edge_type="d"``, ``signal_v`` for ``edge_type="s"``).
    The complementary (cross-projected) signal is no longer computed here.
  - **AH offset structural exception**: ``c_d`` / ``c_s`` never see LN division
    (regardless of flavor), and are rotated in ``"rotated_normalized"`` and
    ``"normalized"`` but not in ``"raw"``. This matches the old code path for
    AH offset under intervention (rotation required) and keeps autointerp
    behavior identical.

- **`Tracer.trace()` / `Tracer.trace_from_cache()`** (`circuit.py`):
  - ``compute_signals: bool = False`` was REMOVED and replaced with
    ``signals: Literal[None, "rotated_normalized", "normalized", "raw"] = None``.
    When non-None, the primary signal in that flavor is attached to each
    non-seed edge under the key ``"signal"``, along with a new
    ``"signal_flavor"`` attribute recording which flavor was stored.
  - Migration: ``compute_signals=True`` → ``signals="rotated_normalized"`` for
    byte-identical edge values to the pre-0.2.0 path.
  - The graph stores **one signal per edge** (the primary, matching edge type).
    Pre-0.2.0 also stored one per edge, so no storage-layout change.

### Added

- **`Tracer.extract_edge_signal_pair_autointerp()`** (`circuit.py`): new method
  returning the ``(signal_u, signal_v)`` pair for autointerp use cases — the
  paper interprets both U-side and V-side of an SVD-paired channel.
  ``flavor`` defaults to ``"rotated_normalized"`` (the autointerp flavor). The
  method computes the primary via ``extract_edge_signal()`` and cross-projects
  through Omega for the complement.

- **`get_component_output()`** (`signals.py`): new keyword ``ln_normalize:
  bool = True`` controlling whether the output is divided by the downstream LN
  scale. AH offset is unaffected (never had LN division). Existing callers
  without the flag continue to get the prior behavior.

### Notes on AH offset

AH offset (``upstream_component_id == n_heads + 3``) is the lone structural
carve-out from the otherwise orthogonal flag scheme. The intervention path
requires AH offset to be rotated even when other components are not, because
``c_d`` / ``c_s`` live in pseudo-d_model space. This is encoded inside
``extract_edge_signal()`` (see ``rotate_offset`` vs ``rotate_non_offset``).

### Migration guide for existing callers

| Pre-0.2.0 call                                              | 0.2.0 call                                                            |
|-------------------------------------------------------------|-----------------------------------------------------------------------|
| ``tracer.trace(..., compute_signals=True)``                 | ``tracer.trace(..., signals="rotated_normalized")``                   |
| ``tracer.trace(..., compute_signals=False)``                | ``tracer.trace(...)`` (default ``signals=None``)                      |
| ``u, v = tracer.extract_edge_signal(...)``                  | ``u, v = tracer.extract_edge_signal_pair_autointerp(...)``            |
| (analysis flavor only, primary signal)                       | ``signal = tracer.extract_edge_signal(..., flavor="rotated_normalized")`` |

External experiment scripts (`new-code/experiments/extract_signals.py`) need
to migrate to the new pair method; that change ships in the paper repo, not
this library.

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
