# Feasibility Study — Ideas 12 through 20

> Op-level structural changes and micro-optimizations from `perf-ideas.md`, assessed against current code, codegen, and bench infrastructure.

---

## 12. all_reduce: two-stage simd→threadgroup

**Target:** `crates/metaltile-std/src/mlx/reduce.rs`

### Claim
Native simdgroup reduce intrinsics eliminate one barrier.

### Current reality
The kernel uses `strided_reduce(inp, zero, n, sum)` then `reduce_sum(acc)`. From `tile inspect mt_all_reduce`, the codegen already emits:
1. `simd_sum(float(v_acc))` — simdgroup-level intra-warp reduction
2. `threadgroup_barrier(mem_flags::mem_threadgroup)` — sync across warps
3. Second `simd_sum(_wv)` — cross-warp reduction via `tg_result_sg[0]`
4. Another `threadgroup_barrier`

This is **already** the two-stage pattern the idea describes. The "native simdgroup reduce intrinsic" (`simd_sum`) is already used. There is no third barrier to eliminate.

### Why it's a no-op
The bench is `N=64M` — the reduction overhead (2 barriers) is completely negligible compared to the 64M elementwise loads. Even if you shaved one barrier, it wouldn't show in GB/s.

### Verdict
⚪ **No-op.** Codegen already optimal. Idea was speculative — `tile inspect` confirms the intrinsics are emitted.

---

## 13. row-reduce: rows-per-threadgroup when N is small

**Target:** `crates/metaltile-std/src/mlx/reduce.rs`

### Claim
N < 256 means one row fits in one simdgroup. Pack multiple rows per threadgroup.

### Current reality
Current bench: `ROW_REDUCE_SHAPES = [(1024, 4096)]`, `tpg=256`.  
Each threadgroup processes **one row** (4096 elements / 256 threads = 16 elements per thread, 4 iterations of 4-wide unroll).

The dispatch is `BenchDispatch::Generic` with `grid=RowsB`. The `run_generic` function in `run_spec.rs` dispatches `b` threadgroups, one per row.

### What would need to change
To pack multiple rows per threadgroup, you'd change the **dispatch grid**, not the kernel body. For example, with `N=128` and `tpg=256`, you could fit 2 rows per threadgroup (128×2 = 256). The grid would be `b/2` threadgroups instead of `b`.

This requires:
1. Adding a small-N `ShapeSpec` to the bench macro
2. Changing the dispatch grid calculation in `run_spec.rs` or the macro-generated `DispatchGrid`

The kernel body (`strided_reduce`) is already correct for any row count — it just needs the right `program_id::<0>()` mapping.

### Risk
- Divergent strides: rows packed into one tg must have contiguous memory layout for the stride math to work. The `strided_reduce` primitive takes `offset` and `end` parameters, so this is fine.
- Only wins when N is small enough that multiple rows fit in one tg. For the current `N=4096`, no win.

### Verdict
⚠️ **Feasible but touches dispatch logic.** Not a single-file kernel tweak — you'd modify `run_spec.rs` or the macro's `DispatchGrid` logic. Effort: medium (one day). Value: small (only for small-N workloads).

---

## 14. scan: prefer `simd_prefix_inclusive_sum`

**Target:** `crates/metaltile-std/src/mlx/scan.rs`

### Claim
Metal 3.1 intrinsic should beat manual Kogge-Stone. Verify codegen already uses it.

### Current reality
From `crates/metaltile-codegen/src/msl/emit_block.rs` (lines 675-694):
```rust
Op::Scan { value, axis: _, op: rk, exclusive } => {
    let fn_name = match (rk, *exclusive) {
        (ReduceKind::Sum | ReduceKind::Mean, true) =>
            Some("simd_prefix_exclusive_sum"),
        (ReduceKind::Sum | ReduceKind::Mean, false) =>
            Some("simd_prefix_inclusive_sum"),
        _ => None,
    };
```

And from `tile inspect mt_scan`:
```metal
float v_thread_excl = simd_prefix_exclusive_sum(v_s3);
```

The kernel already uses `simd_scan_exclusive` in the DSL, which maps directly to Metal's `simd_prefix_exclusive_sum`.

### Verdict
⚪ **No-op.** Already implemented and emitting the optimal intrinsic. The idea's risk note was "zero (drop-in)" — it turns out it's already dropped in.

---

## 15. argmax: refuse to slow down 847%

**Target:** `crates/metaltile-std/src/ffai/arg_reduce.rs`

### Claim
argmax already crushes MLX (847%). Can we hold this while shrinking register pressure?

### Current reality
The kernel body:
```rust
let mut best_val = neg_infinity();
let mut best_idx = lid - lid;
threadgroup_alloc("tg_vals", 256);
threadgroup_alloc("tg_idxs", 256);
let n_iters = (n + lsize - 1u32) / lsize;
for _r in range(0u32, n_iters, 1u32) {
    let pos = _r * lsize + lid;
    if pos < n {
        let v = load(inp[pos]).cast::<f32>();
        let bet = v > best_val;
        best_val = select(bet, v, best_val);
        best_idx = select(bet, pos, best_idx);
    }
}
threadgroup_store("tg_vals", lid, best_val);
threadgroup_store("tg_idxs", lid, best_idx);
threadgroup_barrier();
// 7-step binary tree reduction
argmax_step!(lid, 128);
argmax_step!(lid, 64);
argmax_step!(lid, 32);
argmax_step!(lid, 16);
argmax_step!(lid, 8);
argmax_step!(lid, 4);
argmax_step!(lid, 2);
if lid == 0 {
    let final_v = threadgroup_load("tg_vals", 0);
    let final_i = threadgroup_load("tg_idxs", 0);
    let last_v = threadgroup_load("tg_vals", 1);
    let last_i = threadgroup_load("tg_idxs", 1);
    let bet = (last_v > final_v) | ((last_v == final_v) & (last_i < final_i));
    store(out[0], select(bet, last_i, final_i));
}
```

Register usage is minimal: `best_val`, `best_idx`, loop index, `v`, `pos`. Threadgroup memory holds the tree state (512 floats), not registers.

### Why it's hard to improve
- The kernel is already register-light. There's almost nothing to strip out.
- The 847% figure comes from crushing MLX's scalar tree reduction with a more efficient pattern — the win is structural, not micro-optimization.
- "Lowering regs frees occupancy for other kernels in fused graphs" — this is a **graph-level** optimization, not a single-kernel one. You'd need to profile a fused graph to measure it.

### What could be tried
- Replace the 7 manual tree steps with `simd_max` + `simd_shuffle_xor` butterfly. But Metal doesn't have a built-in `simd_argmax`, so you'd still need the index tracking.
- Reduce `threadgroup_alloc` from 256 slots to `lsize` slots (currently hardcoded to 256, but the kernel dispatches with `tpg=256` anyway, so it's a no-op).

### Verdict
⚪ **Marginal / hard to measure.** The kernel is already optimal for its scope. The stated win (reducing register pressure for fused graphs) requires graph-level profiling, not a single-kernel bench. Not a Quick-win.

---

## 16. RoPE: precompute sin/cos to threadgroup memory

**Target:** `ffai/rope_llama.rs`, `mlx/rope.rs`

### Claim
sin/cos table for D=128 is 1 KB. Compute once per threadgroup, reuse across heads.

### Current reality
`rope_llama.rs` dispatches one thread per `(head, i)` pair:
```rust
let head = program_id::<0>();
let i = program_id::<1>();
```

`inv_freq` depends only on `i`, not `head`. `theta` depends on `position * inv_freq`, so also only on `i` (position is constexpr). `cos_t` and `sin_t` are **identical for all heads with the same `i`**.

With `h=32` heads and `d=128` (half_dim=64), there are 32×64 = 2048 threads. Each threadgroup size is not explicitly set — `run_rope` in `run_spec.rs` dispatches with `tpg=256` (one dimension) or similar.

### Why it's blocked
- **Dispatch mismatch:** Threads with the same `i` may not be in the same threadgroup. The current 2D grid doesn't guarantee this. To share tg-mem, you'd need to remap the dispatch so all heads for a given `i` share a tg.
- **No bench for `rope_llama.rs`:** The FFAI kernel is `inventory::submit!` only (no `#[bench_kernel]`). The MLX bench is `mlx/rope.rs` which has a different structure (batch/seq grid, not head-per-thread).
- **1 KB in tg mem is fine** (budget is ~32 KB), but only if the threads sharing it are actually colocated.

### What would need to change
1. Remap dispatch: one threadgroup per `i` value, with `h` threads in the tg (one per head). This is a **dispatcher rewrite**.
2. Add `#[bench_kernel]` to `rope_llama.rs` or bench via `mlx/rope.rs`.

### Verdict
🔴 **Blocked.** Requires dispatch restructuring. The sin/cos redundancy is real, but exploiting it needs a different grid shape. One-day to Multi-day effort.

---

## 17. RoPE-into-QKV fusion

**Target:** New fused kernel in `ffai/`

### Claim
Write Q/K from projection straight into rotated form, skipping global memory round-trip.

### Current reality
There is no existing kernel that does `qkv_proj + rope` fused. The project has separate `qkv_proj` (steel GEMM), `rope` (standalone), and attention.

### What would need to change
1. Write a new `#[kernel]` that takes the input hidden state, projects to Q/K, and applies RoPE inline.
2. Add a bench harness entry for it (new `BenchDispatch` variant or `Generic` shape).
3. Compare against unfused chain.

### Verdict
🔴 **Blocked.** New kernel + new bench harness. Not a Quick-win. Effort: One-day minimum.

---

## 18. KV-cache append: vectorized aligned copy

**Target:** `ffai/kv_cache.rs`

### Claim
Bump scalar copy to vec4/vec8 with alignment guard.

### Current reality
```rust
#[kernel]
pub fn kv_cache_update<T>(
    src: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] position: u32,
) {
    let idx = program_id::<0>();
    let h = idx / head_dim;
    let d = idx - h * head_dim;
    let dst_idx = h * max_seq * head_dim + position * head_dim + d;
    store(out[dst_idx], load(src[idx]));
}
```

One thread = one element. With `head_dim=128`, that's 128 threads per head per layer.

### Why it's blocked
Same blocker as ideas 5 and 8: **the DSL has no vector store primitive.** `load()` and `store()` are scalar. To do a vec4 copy, you'd need `load_vec4<T>()` or raw pointer casting, neither of which exists.

With `head_dim=128` (divisible by 4 and 8), the geometry is perfect for vectorization. But the DSL can't express it.

### Verdict
🔴 **Blocked.** DSL lacks vector load/store primitives.

---

## 19. Gather: prefetch-to-threadgroup for hot indices

**Target:** `ffai/gather.rs`, `mlx/strided.rs`

### Claim
When indices show locality, stage a window of the table into tg mem.

### Current reality
```rust
#[kernel]
pub fn gather<T>(table: Tensor<T>, indices: Tensor<u32>, out: Tensor<T>, #[constexpr] dim: u32) {
    let idx = program_id::<0>();
    let token = idx / dim;
    let d = idx - token * dim;
    let token_id = load(indices[token]);
    let src = token_id * dim + d;
    store(out[idx], load(table[src]));
}
```

One thread per output element. Each thread has its own `token` and `token_id`. Threads in the same threadgroup likely have **different** `token_id`s unless the index tensor has repeated values.

### Why it's blocked
- **No shared work:** Each thread loads a different row of the embedding table. There's no common data to prefetch into tg memory.
- **To make prefetching work**, you'd need to restructure dispatch so threads in the same tg share the same `token_id`. For example, dispatch `dim` threads per token, all working on the same row. But the current dispatch is a flat 1D grid of `n_tokens * dim` threads.
- **Restructuring required:** Change from "one thread per output element" to "one threadgroup per token, dim threads per tg". This is a dispatcher change, not a kernel constant tweak.

### Verdict
🔴 **Blocked.** Requires dispatch restructuring. Only wins when indices have locality (repeated tokens), which is a workload-dependent heuristic.

---

## 20. Strided copy: emit vec types for stride-1 axes

**Target:** `mlx/copy.rs` + codegen `vectorize.rs`

### Claim
Contiguous inner axis should always vectorize. If codegen misses it, that's a codegen bug.

### Current reality
`mt_copy` is the simplest possible kernel:
```rust
#[kernel]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}
```

The codegen `vectorize.rs` pass (`crates/metaltile-codegen/src/passes/vectorize.rs`) may or may not vectorize this. The bench runs `N=64M` elementwise, so if the vectorizer is working, you'd see `float4` loads in the generated MSL. If not, you'd see scalar loads.

### How to verify
```bash
tile inspect mt_copy --stats   # check if vectorize pass fired
tile inspect mt_copy           # inspect MSL for float4 loads
```

From the generated MSL earlier (`mt_all_reduce`, `mt_gemv`), the codegen already emits 4-wide unrolled loops for reduction primitives. But for a pure elementwise kernel with no loop (just `load(a[idx])`), the vectorizer may not trigger because there's no loop to unroll.

### Verdict
⚠️ **Feasible but requires codegen investigation, not a kernel tweak.** If the vectorizer is missing this, it's a bug in `vectorize.rs`. The fix would be in the codegen pass, not in `copy.rs`. Effort: unknown (could be a one-line pass heuristic, could be a missing pattern).

---

## Summary table

| # | Idea | Verdict | Kernel edit? | Dispatch/ codegen? | Blocker |
|---|------|---------|--------------|-------------------|---------|
| 12 | all_reduce two-stage | ⚪ no-op | No | No | Already optimal |
| 13 | row-reduce pack rows | ⚠️ feasible | No | **Yes** | Small-N workloads only |
| 14 | scan simd_prefix | ⚪ no-op | No | No | Already implemented |
| 15 | argmax hold 847% | ⚪ marginal | Maybe | No | Hard to measure; kernel already optimal |
| 16 | RoPE sin/cos tg-mem | 🔴 blocked | No | **Yes** | Dispatch restructuring needed |
| 17 | RoPE-QKV fusion | 🔴 blocked | **New kernel** | **New bench** | Entirely new work |
| 18 | KV-cache vec copy | 🔴 blocked | No | No | DSL lacks vector primitives |
| 19 | Gather tg prefetch | 🔴 blocked | No | **Yes** | Dispatch restructuring needed |
| 20 | Copy vectorize | ⚠️ feasible | No | **Yes** | Investigate `vectorize.rs` pass |

## Recommended next steps

1. **Verify 20 quickly:** Run `tile inspect mt_copy --stats` to see if `vectorize.rs` fired. If not, you've found a codegen bug to chase.
2. **If 20 is real:** Fix in `vectorize.rs` — this could benefit *every* elementwise kernel, not just copy. High leverage.
3. **Skip the rest:** Ideas 12, 14 are no-ops. Ideas 13, 16, 17, 19 need dispatcher work. Ideas 15, 18 are marginal/blocked.

Want me to run `tile inspect mt_copy --stats` to check idea 20?