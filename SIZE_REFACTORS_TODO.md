# Size Refactors TODO — Scaling ACC++ to Llama-3.1-8B

Status update (2026-07-17): **all three blockers and secondary items 1, 4, 5
are DONE**, validated by the regression suite in `tests/` (invariant tests +
golden comparisons against pre-refactor outputs on gpt2 and pythia-160m; see
`tests/README.md`). Remaining: run a first end-to-end Llama-3.1-8B (or
Llama-3.2-3B) trace on suitable hardware, and the attention-sink research
question (secondary item 3). One correction to the original plan is noted
inline at Blocker 1.

Status of the codebase (2026-07-10): **architecturally ready** for
`meta-llama/Llama-3.1-8B`, blocked only by memory/compute scaling issues.

Verified against the pinned TransformerLens 2.16.1:

- `meta-llama/Llama-3.1-8B` is an officially supported TL model, and its config
  passes every guard in `get_model_config()` (`src/accpp_tracer/models.py:52-107`).
- GQA (32 query heads, 8 KV heads, `gqa_repeats=4`) is handled throughout:
  `hook_v` repeat-interleave in `tracing.py:266-267`, per-head KV mapping in
  `signals.py:110-113` and `circuit.py:137-143`. TL exposes `W_K`/`b_K` already
  repeated to `n_heads`, so the Omega SVD is shape-correct without changes.
- Llama-3.1's NTK-by-parts RoPE scaling is already implemented
  (`rope.py:82-104`); TL reports `use_NTK_by_parts_rope=True`, factor 8,
  low/high freq factors 1/4, original ctx 8192.
- RMSNorm: with `fold_ln=True`, `ln1.hook_normalized == x/scale` (no mean
  subtraction), so the additive decomposition holds. Same hook names.
- No Q/K biases: `b_Q`/`b_K` are zeros → `c_d`/`c_s` are zero vectors (same as
  Gemma, already noted at `circuit.py:254`).
- `attn_scores_soft_cap = -1.0` in TL → the `> 0` check at `tracing.py:204`
  correctly falls through to identity.

Relevant Llama-3.1-8B dimensions: `n_layers=32`, `n_heads=32`,
`n_key_value_heads=8`, `d_model=4096`, `d_head=128`. Scale factor vs GPT-2
small is roughly 12× on the `n_layers · n_heads · d_head` axis and 28× on the
`d_model²` axis.

---

## Blocker 1 [DONE]: `get_omega_decomposition` allocates ~200 GB

> **Correction (2026-07-17):** `full_matrices=False` does NOT help here —
> Omega is a *square* d_model x d_model matrix, so the reduced SVD returns
> U/V of the same full size. The savings come entirely from the per-layer
> loop + immediate rank-d_head slicing (implemented); transient is now
> bounded by one layer's (n_heads, d_model, d_model) tensors (~6 GiB at 8B
> scale). Disk caching was already implemented previously.

**Where:** `src/accpp_tracer/decomposition.py:43-59`

**Problem:** Omega is built for all layers/heads at once:
`(n_layers, n_heads, d_model, d_model)` = `(32, 32, 4096, 4096)` fp32 =
**64 GiB**. Worse, `torch.linalg.svd` defaults to `full_matrices=True`, so U
and V come back the same size — roughly **200 GiB transient**. (GPT-2 small:
~340 MB total, which is why this is invisible today.)

**Fix (small):**
- Loop per layer (or per head) instead of batching all layers.
- Pass `full_matrices=False` to `torch.linalg.svd`.
- Slice to the top `rank = d_head = 128` singular vectors immediately inside
  the loop, before accumulating.
- Final storage for U + VT drops to ~4.3 GiB.

**Also:** cache (U, S, VT, pinvs) to disk keyed on model name. 1,024 SVDs of
4096×4096 matrices is ~minutes on GPU but hours on CPU, and depends only on
the weights.

## Blocker 2 [DONE]: `get_seeds` allocates a mostly-unused T×T buffer

**Where:** `src/accpp_tracer/circuit.py:117-125` (allocation),
`circuit.py:179` (the only read)

**Problem:** `upstream_output_breakdown` is
`(n_layers, n_heads+3, n_tokens, n_tokens, d_model)` — at Llama size that is
**~18 MB × T²**: 16 GB for a 30-token prompt, 75 GB for 64 tokens. But line
179 only ever reads the `dest = end_token_pos` slice; the rest of the dest
dimension is dead weight.

**Fix (trivial):** drop the dest-token dimension and compute only the
`end_token_pos` row → shape `(n_layers, n_heads+3, n_tokens, d_model)` ≈
18 MB × T. Pure win for currently supported models too.

## Blocker 3 [DONE]: `_trace_firing_inner` materializes Omega full-rank

**Where:** `src/accpp_tracer/tracing.py:227-231` (Omega),
`tracing.py:535-541` (`M_s_all`)

**Problem:**
- Omega is materialized as `(d_head, d_model, d_model)` = `(128, 4096, 4096)`
  fp32 = **8 GiB per trace_firing call**, rebuilt for every traced firing
  (GPT-2: 150 MB).
- `M_s_all` is `(n_tokens, d_model, d_model)` = **64 MB × T**.
- All downstream einsums then contract against these dense tensors; compute
  cost depends on einsum path selection finding the low-rank structure, which
  it cannot once Omega is materialized.

**Fix (moderate refactor — the main real work):** never materialize Omega.
Since `Omega = U · diag(S) · VT`, project each side into `d_head` space:

- dest side: `(X @ M_d) @ (U[layer, head] * S[layer, head])` → `(…, d_head)`
- src side:  `VT[layer, head] @ (M_s @ X)` → `(d_head, …)`
- combine in `d_head` space to get the per-SV score breakdown.

Same trick for `M_s_all = W_K_pinv.T @ R @ W_K.T`: keep the
`(n_tokens, d_head, d_head)` rotation stack `R` plus the two fixed
`d_model × d_head` factors, and apply them in sequence.

**Reference implementation already in the codebase:**
`Tracer.extract_edge_signal` (`circuit.py:807-872`) does exactly this factored
per-edge computation — the comment there notes the full matrices "would be
~88 GB for Gemma". Apply the same pattern to the tracing hot path.

## Constraint [DOCUMENTED in README]: breakdown tensors bound prompt length

**Where:** `src/accpp_tracer/tracing.py:219-224`

`upstream_attention_scores_breakdown_{src,dest}` are each
`(layer, n_heads+4, d_head, T, T)` — combined ≈ **1.1 MB × T²** at the deepest
layer (~1 GB at T=30, ~19 GB at T=128). This is inherent to keeping per-SV
granularity, not a bug. On an 80 GB GPU, prompts are limited to roughly
100–150 tokens. Fine for IOI-style prompts; document the limit. If longer
prompts are ever needed, options are chunking over upstream layers or storing
per-SV scores sparsely after thresholding.

---

## Secondary items

1. **[DONE] RoPE rotation matrices are recomputed per call**
   (`rope.py:76-114`): each `get_rotation_matrix` call rebuilds the full
   `(n_ctx, rotary_dim)` angle table and constructs `R` with Python loops over
   `d_head`, and `trace_firing` calls it `dest_token+1` times per firing
   (`tracing.py:531-533`). Cache the stacked `R` tensor on the `Tracer`
   (depends only on model + max position). TL sets `n_ctx=2048` for
   Llama-3.1, so the table itself is small; this is a speed fix, not memory.

2. **fp32 is required → hardware floor.** Weights alone are ~30 GB fp32, and
   `Tracer.__init__` deliberately disables TF32 (`circuit.py:238-239`) because
   the decomposition assertions (`tracing.py:418-437`, atol/rtol 1e-2) need
   full precision; bf16/fp16 would likely trip them. Realistic target: one
   A100/H100-80GB, or high-RAM CPU with patience. Compute per firing is
   ~10-100× GPT-2; expect minutes per traced prompt on GPU.

3. **Attention sinks (research question, not a bug).** Llama-3 heads park
   large attention mass on BOS. The dynamic threshold `2.5/(dest_token+1)`
   (`circuit.py:546`) will pull many BOS-source firings into the greedy loop,
   potentially adding noise edges. Consider whether BOS-src firings should be
   filtered or reported separately.

4. **[DONE] `datasets/ioi.py` needs a `"llama3"` model family.** The existing
   `"llama2"` branches assume SentencePiece tokenization (no leading-space
   tokens). Llama-3's tokenizer behaves like GPT-2 here (`" Mary"` is a single
   token with leading space), so `"llama3"` should follow the gpt2/pythia
   branches. A few lines.

5. **[DONE] Dependency pin skew:** `pyproject.toml` pins
   `transformer-lens==2.16.1`, `requirements.txt` pinned `2.17.0`. Resolved
   to `2.16.1` (the version the Llama-3.1 verification above was done
   against) in both files.

## Suggested order of work

1. Blocker 2 (trivial, benefits all models now)
2. Blocker 1 + disk caching (small, unblocks Tracer init)
3. Blocker 3 (moderate refactor; validate against GPT-2/Gemma-2 traces —
   results should be bit-identical up to SVD sign conventions)
4. Secondary items 1 and 4, then a first end-to-end Llama-3.1-8B trace on an
   80 GB GPU with a short IOI prompt, relying on the built-in decomposition
   assertions as the correctness check.
