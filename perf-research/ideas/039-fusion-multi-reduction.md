# Perf Idea 039 — `fusion.rs`: multi-reduction in one pass

## Metadata
- **Number**: 039
- **Name**: fusion-multi-reduction
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚠️ feasible (needs re-scoping)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Variance computation reads x twice (mean, then mean^2). Read once, accumulate both — saves half the memory traffic.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/fusion.rs`, new pass or kernel
- **Bench filter**: `tile bench -f layer_norm` (layer-norm is the canonical case)
- **Shapes / dtypes to watch**: N=4096, f32/f16/bf16

## Assessment

### Current state
MetalTile has no `layer_norm` kernel in `mlx/`. The `perf-ideas.md` references `mlx/layer_norm.rs` but no such file exists in the current codebase. LayerNorm requires both `sum(x)` and `sum(x^2)` to compute mean and variance.

The `fusion.rs` pass fuses **elementwise chains**, not reduction loops. It has no concept of:
- Detecting two `strided_reduce` calls with the same input tensor and same loop bounds.
- Merging them into a single loop with two accumulators.

### What multi-reduction fusion means
A LayerNorm kernel in pseudocode:
```rust
let row = program_id::<0>();
let rs = row * n;
let re = rs + n;
let mut sum = 0.0f32;
let mut sum_sq = 0.0f32;
for i in range(rs + tid, re, lsize) {
    let x = load(inp[i]).cast::<f32>();
    sum = sum + x;
    sum_sq = sum_sq + x * x;
}
let mean = reduce_sum(sum) / n;
let var = reduce_sum(sum_sq) / n - mean * mean;
let inv_std = rsqrt(var + eps);
for i in range(rs + tid, re, lsize) {
    let x = load(inp[i]).cast::<f32>();
    let normed = (x - mean) * inv_std;
    store(out[i], (normed * w + b).cast::<T>());
}
```

This is a **hand-written fused kernel** — it already does the multi-reduction in one pass. The question is whether the **codegen** can automatically produce this from separate `mean` and `variance` operations in a higher-level IR.

### What the fusion pass would need
The current `fusion.rs` operates on a single block, fusing adjacent elementwise ops. Multi-reduction fusion would require:
1. A higher-level IR that expresses `mean` and `variance` as separate ops.
2. A pass that detects they share the same input and loop structure.
3. Merging their loop bodies into one.

This is **loop fusion**, not **operator fusion**. The current `fusion.rs` is operator fusion (expression tree merging). Loop fusion is a different compiler phase.

### MLX reference
MLX does not automatically fuse multiple reductions either. Its `layer_norm` kernel (if it exists) is hand-written to compute both accumulators in one loop. The MetalTile equivalent would also be hand-written in the DSL.

## Verdict

- **Outcome**: feasible but needs re-scoping — genuine optimization, but not in `fusion.rs`
- **Why**: The current `fusion.rs` fuses expression trees within a block. Multi-reduction fusion is **loop fusion** — detecting and merging separate reduction loops. This requires a new pass or extending the kernel authoring pattern, not modifying `fusion.rs`.
- **Re-scope**: A `multi_reduce.rs` pass that detects `strided_reduce` pairs with identical inputs and merges them into a single loop body with multiple accumulators. Or, more practically, write `layer_norm` as a hand-written kernel in the DSL (since there's no `layer_norm.rs` today).

## Risk Register
- Welford's algorithm for numerical stability: computing variance as `E[x^2] - E[x]^2` is less numerically stable than Welford's online algorithm. A fused multi-reduction pass should use Welford's method for accuracy.
- The DSL `strided_reduce` builtin currently takes a single combiner (`sum`, `max`, etc.). Extending it to multiple combiners would need DSL syntax changes.

## Notes for Next Person
- The fastest path to a fast LayerNorm is writing it by hand in the DSL, not building a loop-fusion pass. See `rms_norm.rs` (if it exists) as a template.
- If loop fusion is pursued, it should be a new pass (`loop_fusion.rs`) that runs before `vectorize.rs` and `unroll.rs`.
