# accpp_tracer test suite

Quick regression suite exercising every main API of the library on two small
models from the local HF cache: **gpt2** (LayerNorm, absolute positions) and
**EleutherAI/pythia-160m** (RoPE). Everything runs on CPU in fp32 for
determinism. Full run: ~4 minutes.

```bash
pytest                    # run the suite (compares against tests/golden/)
pytest --update-golden    # re-baseline the golden files (see below)
```

## Structure

| File | Covers |
|---|---|
| `test_decomposition.py` | Omega SVD (shapes, reconstruction, torch-vs-numpy backend), pseudoinverses, disk-cache round-trip and failure modes |
| `test_rope.py` | RoPE rotation matrices (orthogonality, identity at pos 0, agreement with TransformerLens sin/cos tables) |
| `test_tracing.py` | `trace_firing` (Level-1 API) on both models; the residual-share decomposition invariant |
| `test_circuit.py` | `Tracer.trace` (prob / linear / top-k seeding), `trace_from_cache`, `trace_from_probe`, `get_seeds`, `get_seeds_prob`, signal storage, Tracer disk cache, argument validation |
| `test_intervention.py` | `extract_edge_signal` (all flavors), `extract_edge_signal_pair_autointerp`, `run_intervention` (local + global), `edges_from_graph` |

Two kinds of tests:

1. **Invariant tests** — model-agnostic mathematical properties (SVD
   reconstructs Omega, component shares sum to the normalized residual
   stream, rotation matrices are orthogonal, decomposition assertions inside
   `trace_firing` pass, …). These need no golden files and remain valid
   under any refactor.
2. **Golden regression tests** — outputs captured in `tests/golden/` from a
   known-good revision: decomposition fingerprints (SVD-sign-invariant
   random-projection sketches), RoPE matrix stacks, `trace_firing` results,
   and fully serialized traced graphs. Comparisons use small float
   tolerances (1e-5 for graphs/firings) so behavior-preserving refactors
   pass while real changes fail loudly.

## Workflow for future upgrades / refactors

1. On the current known-good revision run `pytest` — everything must pass.
2. Make your changes.
3. Run `pytest` again:
   - **Behavior-preserving change** (memory/speed refactor): all tests must
     pass against the *existing* goldens. Do not regenerate them.
   - **Intended behavioral change** (algorithm change, new TL pin): inspect
     the golden-test diffs, confirm they match the intended change, then
     re-baseline with `pytest --update-golden` and commit the new goldens
     together with the code change.

## Adding a model (e.g. a Llama-family model)

Add a `<model>_model` fixture and `<model>_tracer` fixture in `conftest.py`,
plus a traced-graph fixture; then extend the parametrized tests and add
golden files with `--update-golden`. The invariant tests will work unchanged
for any TL-supported model that passes `get_model_config` validation.
