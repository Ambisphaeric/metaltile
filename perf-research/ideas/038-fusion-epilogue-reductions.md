# Perf Idea 038 — `fusion.rs`: epilogue fusion onto reductions

## Metadata
- **Number**: 038
- **Name**: fusion-epilogue-reductions
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚠️ feasible (needs re-scoping)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> `softmax(x) * w` and `rms_norm(x) * w` are common; fuse the multiply into the reduction kernel.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/fusion.rs`
- **Bench filter**: would need fused kernel harness (does not exist)
- **Shapes / dtypes to watch**: layer_norm, rms_norm shapes where post-reduction multiply is a second dispatch

## Assessment

### Current fusion pass
`fusion.rs` fuses **elementwise chains** into `Op::FusedElementwise`:
- `BinOp`, `UnaryOp`, `Activation`, `Cast`, `Select`, `Zeros`, `Splat`, `Broadcast` are fusible.
- `Reduce`, `StrideReduce`, `Scan`, `Load`, `Store`, `Barrier`, `Loop`, etc. are NOT fusible.
- The pass traces backward from stores, collecting single-use producers until it hits a non-fusible op.

### Why reduction + elementwise is not currently fused
A typical reduction kernel IR looks like:
```
v0 = strided_reduce(inp, ...)
v1 = reduce_sum(v0)       // ← Reduce — not fusible
v2 = Cast(v1, f32)        // ← could fuse with v3
v3 = BinOp(Mul, v2, w)    // ← could fuse with v2
store(out, v3)
```

The fusion pass would fuse `v2 → v3` into a `FusedElementwise` chain, but it stops at `v1` because `reduce_sum` is not fusible. The chain becomes:
```
v1 = reduce_sum(v0)
v23 = FusedElementwise([Cast, Mul])  // v2 and v3 fused
store(out, v23)
```

This is already partially fused — the post-reduction elementwise ops are fused. But the perf-ideas.md hypothesis wants to go further: fuse the elementwise ops **into** the reduction kernel itself, eliminating the separate `FusedElementwise` dispatch.

### What's actually missing
MetalTile kernels are single-dispatch. A kernel like `mt_rms_norm` (if it exists) would already compute the reduction and the scaling in one kernel body. The idea assumes that `softmax(x) * w` is currently two dispatches: one for `softmax`, one for `mul`. But in MetalTile's `#[kernel]` DSL, you write the entire kernel body — there is no automatic splitting into multiple dispatches within a single kernel.

Wait — let me reconsider. The fusion pass runs on the IR **before** MSL generation. If a kernel's DSL source contains:
```rust
let mean = reduce_sum(x) / n;
let normed = x / mean;
store(out, normed * w);
```

The IR would have `Reduce` + `BinOp` + `BinOp` + `Store`. The fusion pass would fuse the two `BinOp`s into `FusedElementwise([BinOp(Div), BinOp(Mul)])`. But it cannot fuse the `Reduce` into that chain because `Reduce` is not fusible.

However, the kernel is still a **single dispatch**. There is no separate "reduction kernel" and "elementwise kernel" — it's all one kernel. The question is whether the MSL emitter can inline the `Reduce` result into the `FusedElementwise` expression. The answer is: the MSL emitter already does this, because the IR is lowered to MSL source. The `FusedElementwise` just means the emitter writes `((x / mean) * w)` as a single expression instead of separate `auto` variables.

### The real optimization
The actual win from "epilogue fusion" in cuBLAS/MLX is when a **library GEMM** dispatches a kernel, and a subsequent elementwise op is fused into the GEMM's epilogue. Since MetalTile has no separate GEMM library (kernels are written in DSL), there is no dispatch boundary to fuse across.

For standalone kernels, the fusion pass already handles the elementwise chain after a reduction. The only gap is if the reduction result is used in multiple places (multi-use breaks the chain) or if the reduction is followed by a non-elementwise op.

## Verdict

- **Outcome**: feasible but marginal — the pass already fuses post-reduction elementwise chains; the hypothesized "fusion into the reduction kernel" is already how MetalTile works (single dispatch)
- **Why**: In MetalTile, a reduction and its downstream elementwise ops are already in the same kernel dispatch. The `fusion.rs` pass already merges the elementwise part into `FusedElementwise`. Making `Reduce` fusible would be a small extension but the benefit is limited because the MSL emitter already inlines values.
- **Note**: The real value of epilogue fusion appears when MetalTile has a GEMM library or auto-generated kernels where reduction and elementwise are separate IR stages. Today, they are not.

## Risk Register
- Making `Reduce` fusible is non-trivial: reductions have barrier semantics and threadgroup-memory staging. A fused `Reduce + BinOp` would need the MSL emitter to understand that the reduction result is live after the barrier and can participate in a downstream expression.

## Notes for Next Person
- The current `fusion.rs` already provides most of the benefit for MetalTile's single-kernel model.
- If MetalTile ever adds multi-kernel graphs (e.g., a runtime that chains kernels), then cross-kernel fusion becomes the high-value optimization this idea describes.
