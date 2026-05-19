# Session Kickoff Prompt — Ideas 31–40

> Copy-paste this into a fresh session to continue the established pattern.

## Context

We are working through the `perf-research/perf-ideas.md` hopper in the **metaltile** repo. Ideas 1–30 are fully assessed with individual files in `perf-research/ideas/NNN-<name>.md`. STATUS.md and RESEARCH-LOG.md are kept current.

### Your task
Assess ideas **31 through 40** from `perf-ideas.md`. For each idea:
1. Read the target code file(s)
2. Determine if it's feasible, blocked, a no-op, or needs re-scoping
3. Create an individual file: `perf-research/ideas/NNN-<short-name>.md`
4. Update `perf-research/STATUS.md` and `perf-research/RESEARCH-LOG.md`
5. Commit to `dev`

### Worktree convention
Use a worktree off `dev` for any actual code experiments:
```bash
git worktree add -b perf/idea-NNN-<name> ../metaltile-perf-idea-NNN dev
```
Skip the worktree for purely analytical assessments.

### Individual idea file format
Use the standard template from `ideas/TEMPLATE.md`. Every idea gets its own file with standardized sections: Metadata, Hypothesis, Target, Current Code Reality Check, Baseline, Risk Register, Final Verdict, Related Ideas.

### Commit pattern
Group analytical assessments into commits like `perf-research: feasibility files for ideas 31–35`. Actual experiments get `perf(idea-NNN): <description>` — FOR REVIEW LATER.

---

## Ideas 31–35 (Op-level structural changes)

### 31. Unary chains: emit single `metal::precise::*` calls
*Target*: `mlx/unary.rs` + codegen fusion.
*Hypothesis*: `sigmoid(x)` should not be `1 / (1 + exp(-x))` — Metal has `metal::precise::sigmoid` directly.
*Measure*: `tile bench -f sigmoid -vv`; `tile inspect`.
*Risk*: precision difference (fast vs precise) — likely fine for inference.

### 32. SwiGLU/GELU: fuse with downstream matmul write
*Target*: `mlx/unary.rs` + new fused emitter.
*Hypothesis*: activation right after a GEMM is everywhere in transformers. Fuse the epilogue so output never round-trips to HBM.
*Measure*: would need a new bench `gemm_silu`.
*Risk*: scope — touches GEMM epilogue codegen.

### 33. arg_reduce variants: argmin in same kernel
*Target*: `ffai/arg_reduce.rs`.
*Hypothesis*: argmax is great; argmin shares 90% of structure. Confirm both are equally tuned.
*Measure*: `tile bench -f argmin` (add if missing).
*Risk*: cheap to check.

### 34. softmax + attention epilogue fusion
*Target*: `mlx/softmax.rs` + new fused kernel in `ffai/`.
*Hypothesis*: standalone softmax bench is one number, but in real attention softmax + matmul-with-V is one operator. Fuse and bench against the two-kernel version.
*Measure*: add `softmax_v` bench.
*Risk*: scope.

### 35. random (xorshift32): use 64-bit state, vec4 generation
*Target*: `mlx/random.rs`.
*Hypothesis*: 32-bit xorshift wastes a register that could hold the high 32 bits; vec4 generation amortizes constant load.
*Measure*: `tile bench -f random`.
*Risk*: correctness of statistical properties — confirm with chi-square or just match MLX byte-for-byte.

---

## Ideas 36–40 (Codegen passes)

### 36. `vectorize.rs`: 8-wide on f16/bf16
*Target*: `crates/metaltile-codegen/src/passes/vectorize.rs`.
*Hypothesis*: currently caps at vec4 (one MSL `*4` type). Metal supports `half8`/`bfloat8`. Emitting 8-wide doubles LSU throughput on bandwidth-bound kernels.
*Measure*: re-run *all* kernels with a single switch — that's the test.
*Risk*: a single bug regresses everything; gate behind a feature flag, A/B with `MTLT_VEC_WIDTH=8`.

### 37. `vectorize.rs`: detect strided-but-aligned stores
*Target*: same.
*Hypothesis*: stride==1 isn't the only vectorizable case — interleaved stores into different output buffers are also fair game.
*Measure*: `tile bench` aggregate.
*Risk*: aliasing; require explicit no-alias annotation on tensor inputs (already the convention).

### 38. `fusion.rs`: epilogue fusion onto reductions
*Target*: `crates/metaltile-codegen/src/passes/fusion.rs`.
*Hypothesis*: `softmax(x) * w` and `rms_norm(x) * w` are common; fuse the multiply into the reduction kernel.
*Measure*: bench fused vs unfused — would need a harness to express the chain.
*Risk*: type-check needs to allow the new fused op.

### 39. `fusion.rs`: multi-reduction in one pass
*Target*: same.
*Hypothesis*: variance computation reads x twice (mean, then mean^2). Read once, accumulate both — saves half the memory traffic.
*Measure*: `tile bench -f layer_norm` (layer-norm is the canonical case).
*Risk*: numerical stability with Welford's algorithm — well-known fix.

### 40. `unroll.rs`: register-pressure-aware unroll count
*Target*: `crates/metaltile-codegen/src/passes/unroll.rs` + `register_estimate.rs`.
*Hypothesis*: today's unroll factor is likely fixed-per-loop. Pick `unroll_count = max_regs / regs_per_iter`.
*Measure*: aggregate bench; `regs` column should fill closer to 80% of cap.
*Risk*: regress small kernels that don't have register headroom.

---

## Key files to read

| Idea | Files |
|------|-------|
| 31 | `crates/metaltile-std/src/mlx/unary.rs`, codegen `emit_block.rs` |
| 32 | `crates/metaltile-std/src/mlx/unary.rs`, `steel/gemm/` |
| 33 | `crates/metaltile-std/src/ffai/arg_reduce.rs` |
| 34 | `crates/metaltile-std/src/mlx/softmax.rs`, `ffai/sdpa*.rs` |
| 35 | `crates/metaltile-std/src/mlx/random.rs` |
| 36 | `crates/metaltile-codegen/src/passes/vectorize.rs` |
| 37 | `crates/metaltile-codegen/src/passes/vectorize.rs` |
| 38 | `crates/metaltile-codegen/src/passes/fusion.rs` |
| 39 | `crates/metaltile-codegen/src/passes/fusion.rs` |
| 40 | `crates/metaltile-codegen/src/passes/unroll.rs` |

## Already-known blockers
- **No vector types in DSL** (`load()` is scalar) — affects 35, 36
- **Codegen pass changes are repo-wide** — one bug in `vectorize.rs` or `fusion.rs` affects every kernel. These are Multi-day, not Quick-win.
- **`fusion.rs` changes need type-check support** — the DSL type system may not allow fused ops today
- **New bench entries need `BenchDispatch` variants or `#[bench_kernel]` macro support** — affects 32, 34

## What to skip vs assess
- **Codegen pass ideas (36–40)** — these are structural changes to the compiler. Assess by reading the pass code, not by benching. Determine if the pass already does what's claimed, or if the idea is a genuine missing feature.
- **New-kernel ideas (32, 34)** — these need entirely new kernels + bench harnesses. Mark 🔴 blocked with effort estimate.
- **"Verify codegen emits X" ideas (31)** — run `tile inspect mt_sigmoid`, check MSL for `metal::precise::sigmoid` vs manual `1/(1+exp(-x))`. Mark ⚪ no-op if already optimal, or ⚠️ feasible if a codegen change is needed.
- **Cheap additions (33, 35)** — actually try them if they're simple (add argmin bench, random variant).

## Output expectation
At the end of the session, `ls perf-research/ideas/` should show files for **31, 32, 33, 34, 35, 36, 37, 38, 39, 40**. STATUS.md and RESEARCH-LOG.md updated. One commit grouping the analytical codegen assessments, plus any separate commits for experiments.
