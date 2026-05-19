# Perf Idea 037 — `vectorize.rs`: detect strided-but-aligned stores

## Metadata
- **Number**: 037
- **Name**: vectorize-strided-aligned-stores
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚠️ feasible (needs re-scoping)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> stride==1 isn't the only vectorizable case — interleaved stores into different output buffers are also fair game.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/vectorize.rs`
- **Bench filter**: `tile bench` aggregate
- **Shapes / dtypes to watch**: kernels with stores to multiple output buffers at the same index

## Assessment

### Current vectorizer capabilities
The `vectorize.rs` pass handles **contiguous, aligned accesses with power-of-2 element strides**:
- Loads: `src[base+0]`, `src[base+1]`, `src[base+2]`, `src[base+3]` → `VectorLoad { len: 4 }`
- Stores: `dst[base+0]`, `dst[base+1]`, `dst[base+2]`, `dst[base+3]` → `VectorStore { len: 4 }`

It uses `decompose_index` to find `BinOp(Add, invariant_base, Const(k))` with consecutive `k`.

### What "strided-but-aligned" means
Two distinct patterns:

1. **Interleaved stores to different buffers** (the hypothesis's framing):
   ```
   store(c[idx], x + y);
   store(d[idx], x * y);
   ```
   Both use the same index `idx` but different dst buffers. Each store is scalar. Could they be vectorized by processing 4 elements at once and storing 4-wide vectors to each buffer?

2. **Stride-N contiguous blocks** (the classical SIMD pattern):
   ```
   store(out[i*4+0], v0);
   store(out[i*4+1], v1);
   store(out[i*4+2], v2);
   store(out[i*4+3], v3);
   ```
   These ARE contiguous in the output buffer and ARE already handled by the current pass (since `i*4+k` with consecutive `k` is contiguous).

### Why pattern 1 is not currently vectorizable
- The current pass groups by `(src, invariant_base, offset)` — it requires the same buffer.
- Different dst buffers (`c` vs `d`) break the grouping.
- Even if grouped, a `VectorStore` writes multiple contiguous elements to a single buffer. It cannot write to two different buffers simultaneously.
- To vectorize pattern 1, you would need to:
  1. Compute 4 values for `c` and 4 values for `d` in parallel (requires vectorizing the upstream elementwise ops).
  2. Emit two `VectorStore` ops (one for `c`, one for `d`), each with `len: 4`.
  
  This is essentially a **loop vectorization** problem, not just a load/store coalescing problem. The current pass is a load-store coalescer, not a loop vectorizer.

### Pattern 2 is already handled
As noted above, `out[i*4+k]` with consecutive `k` is detected as contiguous by `decompose_index`. This already vectorizes.

## Verdict

- **Outcome**: feasible (needs re-scoping) — genuine missing feature, but the hypothesis framing is imprecise
- **Why**: The current pass only coalesces contiguous accesses within a single buffer. Strided/interleaved vectorization across buffers requires a different approach (loop-level vectorization or SLP vectorization). The specific "interleaved stores to different buffers" pattern would need significant pass redesign.
- **Re-scope**: A more actionable version of this idea would be: "Extend `vectorize.rs` to handle strided loads with constant stride (e.g., `out[i*2]`, `out[i*2+1]` where the stride is a compile-time constant)." This is a narrower, more achievable extension.

## Risk Register
- Aliasing: the current pass requires explicit no-alias annotation (mentioned in perf-ideas.md). This is already the convention.
- The DSL `VectorStore` op only supports a single dst buffer; multi-buffer vectorization would need IR changes.

## Notes for Next Person
- Before pursuing this, verify which kernels actually generate non-contiguous but strided loads/stores. Most MetalTile kernels use contiguous access patterns (row-major tensors).
- The `binary_two` kernel (`store(c[idx], x+y); store(d[idx], x*y)`) is the clearest example of pattern 1. Profiling it to confirm if scalar stores are the bottleneck would help prioritize.
