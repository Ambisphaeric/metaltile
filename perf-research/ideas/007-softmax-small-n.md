# 007 — Softmax: simdgroup reduce for small N (≤ 32)

## Metadata
- **Number**: 007
- **Name**: softmax-small-n
- **Source**: `perf-ideas.md` § Quick-wins — item 7
- **Status**: 🟢 **done** — genuine win, committed for review
- **Worktree**: `../metaltile-perf-idea-7` (branch `perf/idea-7-softmax-small-n`)
- **Assignee**: pi

## Hypothesis
> For N ≤ 32 the two-pass threadgroup-memory reduction is overkill; use `simd_max` + `simd_sum` directly.

## Target
- **Primary file**: `crates/metaltile-std/src/mlx/softmax.rs`
- **Bench filter**: `tile bench -vv -f softmax`
- **Shapes / dtypes**: `B=1024 N=32` for f32, f16, bf16

## Current Code Reality Check
The softmax kernel already uses `simd_max`/`simd_sum` at the simdgroup level. The "two-pass" overhead comes from the `reduce_max` and `reduce_sum` calls, which codegen lowers to a two-level threadgroup reduction (simdgroup level + cross-simdgroup level via threadgroup memory). When `tpg > 32`, the second level is necessary. When `tpg = 32` (exactly one simdgroup), the second level is pure overhead: lane 0 of simdgroup 0 still writes to `tg_max_sg[0]`, barrier, then reads it back.

## Baseline
The existing bench only tests `N=4096, tpg=256`. We added `N=32, tpg=32` and a temporary `N=32, tpg=256` baseline for direct comparison.

## Experiment Log

### Cycle 1 — 2026-05-18
- **Change**: Added `mt_softmax_small_n` bench variant with `b=1024, n=32, tpg=32`.
- **Bench result**:
  | variant | tpg | f32 GB/s | f16 GB/s | bf16 GB/s |
  |---------|-----|----------|----------|-----------|
  | baseline (N=4096) | 256 | 275.8 | 460.7 | 597.4 |
  | small_n | 32 | 47.7 | 23.8 | 23.8 |
  | small_n_baseline | 256 | 28.5 | 14.3 | 14.4 |
- **Correctness**: `ok = ✓` (9/9 correct across all variants)
- **Trust**: cv% mostly < 3%. small_n f32 had 16.2% cv on first run, but 2.7% on second run — DVFS stabilization issue.
- **Decision**: remove temporary `small_n_baseline` variant, keep `small_n` as a new bench entry.

## Analysis

### Why tpg=32 wins for N=32
With `N=32` and `tpg=256`:
- The tail loop `range(rs + tid, rs+32, 256)` only executes for `tid = 0..31`.
- Threads 32–255 are idle but still participate in `reduce_max`/`reduce_sum`.
- The two `threadgroup_barrier` calls in the reduction synchronize all 256 threads.
- Total active threads: 1024 rows × 32 = 32,768.

With `N=32` and `tpg=32`:
- All 32 threads are active.
- `reduce_max`/`reduce_sum` still go through the codegen's two-level path, but `n_simd = 1`, so the second level is trivial (lane 0 writes, lane 0 reads).
- Total active threads: same 32,768, but no idle-thread overhead.
- Speedup: **~1.65×** across all dtypes.

### Is this a real-world win?
Small-N softmax (N≤32) is not a dominant workload in LLM inference. It might appear in:
- Classification heads with small vocab
- Attention scoring for very short sequences
- Nested reductions in custom ops

The absolute throughput is low (24–48 GB/s) because there's barely any arithmetic per byte. The win is relative to a poorly-configured dispatch, not an absolute performance boost.

### Register pressure
Both variants report **63r** — no change. The kernel is thread-limited, not register-limited.

## Risk Register
- **Bench bloat**: Adding a new `#[bench_kernel]` variant increases the total bench suite size and startup overhead. The `small_n` variant is niche.
- **No MLX reference**: MLX's `softmax.metal` doesn't have a direct `N=32` instantiation, so this is MT-only comparison.
- **Small absolute throughput**: 24–48 GB/s is tiny compared to the large-N baseline (275–600 GB/s). The win is relative, not headline-grabbing.
- **Production dispatch**: The real fix isn't in the bench — it's in the runtime dispatcher choosing `tpg=32` when `N≤32`. The bench variant validates that this is correct and faster.

## Final Verdict
**Small but genuine win (~1.65×) for small-N softmax.** The `softmax_small_n` bench variant is kept as a regression test / microbenchmark. The real production value is informing the dispatch heuristic: for `N≤32`, prefer `tpg=32` over `tpg=256`.

## Merged Commit
Mark: FOR REVIEW LATER — adds `softmax_small_n` bench variant. Could be merged as-is for bench coverage, or stripped back to just the research doc if bench bloat is a concern.

## Related Ideas
- **13**: row-reduce pack multiple rows per TG when N is small — same dispatch-shape logic, different kernel family.
- **9**: LayerNorm mirror RMS tweaks — now abandoned after idea-6 showed 8-wide is register-limited.
