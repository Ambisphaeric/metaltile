# Perf Idea 029 — Reductions over short rows: cooperative groups

## Metadata
- **Number**: 029
- **Name**: short-row-cooperative-groups
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: ⚠️ feasible (dispatch-level)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> For N ≤ 32, one simdgroup does the whole row. No threadgroup memory, no barrier.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/reduce.rs`, `crates/metaltile-std/src/run_spec.rs`
- **Bench filter**: `tile bench -f all_reduce` with small N; `tile bench -f row_reduce`
- **Shapes / dtypes to watch**: N=32, N=64, N=128

## Assessment

### Current kernel architecture
`mt_all_reduce` and `mt_row_reduce` use `strided_reduce` (a DSL builtin) followed by `reduce_sum` / `reduce_max` / `reduce_min` / `reduce_product` (also DSL builtins).

The `strided_reduce` builtin is effectively a loop over elements with a stride of `lsize`, accumulating per-thread partial results. Then the second-stage `reduce_*` builtin performs a threadgroup-level tree reduction using `simd_sum` / `simd_max` / `simd_min` + `threadgroup_barrier` + a second `simd_*` — exactly the two-stage pattern confirmed in idea #012.

### Why small N is suboptimal
For `N ≤ 32`:
- `strided_reduce` with `lsize=256` (default tpg) means each thread processes at most 1 element (since `stride = 256 > N`).
- The second-stage reduction runs across 256 threads, but only ~32 threads have meaningful data. The rest are zeros / identity elements.
- A `threadgroup_barrier` for 224 idle threads is pure overhead.

### What the optimization would be
A **dispatch-level heuristic**: when `N ≤ simd_size` (32), dispatch with `tpg=32` instead of `tpg=256`.

This is structurally identical to idea #007 (softmax small N), which proved `tpg=32` wins for N=32 by eliminating idle threads and redundant barriers.

For reductions, the change is even simpler:
- `strided_reduce` with `lsize=32` and `N ≤ 32` means each thread processes 1 element (or 0 if `tid >= N`).
- `reduce_sum` with 32 active lanes uses one `simd_sum` + one `threadgroup_barrier` (for the 32 lanes). Since all lanes are in one simdgroup, the barrier is effectively a no-op.
- Result: one `simd_sum` instead of two, and no idle threads.

### Effort
Small — it's a dispatch heuristic, not a kernel rewrite. The existing `#[bench_kernel]` macro supports per-shape `tpg` values. A new bench variant `all_reduce_small_n` with `tpg=32` is a one-file change.

## Verdict

- **Outcome**: feasible — dispatch-level heuristic, same pattern as #007
- **Why**: The kernel already uses `simd_sum` + `threadgroup_barrier`. The inefficiency is dispatching 256 threads for 32 elements of work. Changing `tpg` to 32 for small N eliminates the waste.
- **Measure**: `tile bench -f all_reduce` with N=32, tpg=32 vs tpg=256.

## Risk Register
- The `strided_reduce` builtin may not be optimal for `N < lsize` — need to verify it handles the tail correctly (threads with `tid >= N` should not participate).
- `reduce_sum` on a partially-active simdgroup (e.g., N=17 with tpg=32) must use the mask correctly so inactive lanes don't pollute the sum.

## Notes for Next Person
- Same playbook as idea #007: add a `*_small_n` bench variant with `tpg=32`.
- Check both `all_reduce` and `row_reduce` — the optimization applies to both.
- Verify that `reduce_sum` / `reduce_max` builtins correctly ignore inactive lanes when `N < lsize`.
