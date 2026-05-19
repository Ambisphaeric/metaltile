# Perf Idea M5 — Block-sparse SDPA exploiting real mask patterns

## Metadata
- **Number**: M5
- **Name**: block-sparse-sdpa
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: ⚠️ feasible (high effort, high impact)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Sliding-window attention, sink-token, BigBird — all have known sparsity structure. A codegen path that takes mask metadata as a constexpr and emits a kernel skipping zero blocks could 4–8x decode throughput at long context.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/sdpa_decode.rs`
- **Bench filter**: `tile bench -f sdpa_decode` at long context (N=8192, 16384, 32768)
- **Shapes / dtypes to watch**: head_dim=128, n_kv=16384, f16

## Assessment

### Current `sdpa_decode` kernel
The kernel walks **all KV positions** in a loop:
```rust
for _t in range(sg, n_kv, ns) {
    let base = kv_head_base + _t * head_dim;
    // load K, compute dot(Q,K), load V, accumulate
}
```

There is **no mask parameter**. Every position from `0` to `n_kv-1` is processed. For `n_kv=32768`, this is 32K iterations per head.

### What block-sparse attention means
Real attention masks have structure:
- **Sliding-window**: Only attend to the last W positions (e.g., W=4096). Positions beyond the window are masked.
- **Sink tokens**: A few fixed positions (e.g., first 4 tokens) are always attended to, regardless of window.
- **BigBird / Longformer**: Random global + local window + block-diagonal patterns.

For a sliding window of W=4096 and n_kv=32768:
- Dense attention: 32K positions × head_dim loads = 32K loads.
- Sliding-window: 4K positions + sink tokens = ~4K loads.
- **8× reduction in K/V memory traffic.**

### What would need to change
1. **Mask metadata as kernel input**: A small buffer describing the sparsity pattern:
   - Sliding window: `(window_size, num_sink_tokens)` as two `u32` constants.
   - Or a block-sparse bitmask: a `u32` array where each bit indicates whether a block of 64/128 positions is active.

2. **Skip logic in the K/V loop**:
   ```rust
   for _t in range(sg, n_kv, ns) {
       if !is_position_active(mask, _t) { continue; }
       // ... load K/V, compute, accumulate
   }
   ```
   
   The `continue` skips the load and computation for masked-out positions.

3. **SIMD divergence**: In a simdgroup, if some lanes hit `continue` and others don't, the hardware serializes both paths. For a block-sparse pattern where **entire blocks** are masked, the skip should be at the **simdgroup level** (all 32 lanes skip together), not per-lane:
   ```rust
   if simd_all(!is_block_active(mask, block_id)) { continue; }
   ```
   
   This requires a block-level mask, not a per-position mask.

4. **Dispatch shape**: The current dispatch is `[n_q_heads, 1, 1]`. For block-sparse, if the active positions are concentrated in a few blocks, occupancy may drop (fewer total iterations). But the per-iteration work is the same, and memory bandwidth is the bottleneck.

### Effort estimate
- Add mask metadata param + skip logic to `sdpa_decode`: **one-day**.
- Bench at long context to verify speedup: **one-day**.
- Optimize simdgroup-level skipping (avoid per-lane divergence): **one-day**.
- Add sink-token support: **small**.
- **Total**: **one-day to multi-day** for sliding window. BigBird/block-sparse is more complex.

### MLX comparison
MLX does not currently have block-sparse SDPA in its public kernel set. The `sdpa_vector` kernel in MLX is dense. Block-sparse attention is typically implemented at the framework level (e.g., in `transformers` or `vllm`) by splitting the query into blocks and dispatching separate kernels. MetalTile could be the first to have a native block-sparse SDPA kernel.

## Verdict

- **Outcome**: feasible — high impact for long-context inference
- **Why**: The current `sdpa_decode` kernel is dense. Adding a mask metadata buffer + skip logic is a localized kernel change. The 4–8× speedup claim is realistic for sliding-window attention at long context (n_kv ≫ window_size).
- **Note**: The key is simdgroup-level skipping (all lanes in a simdgroup skip together) to avoid divergence. A per-lane mask would cause serialization and negate the win.

## Risk Register
- SIMD divergence: per-lane `if` in the K/V loop serializes execution. Must use block-level masks + simdgroup-wide skip.
- Memory layout: the KV cache is stored densely. Skipping positions still requires strided loads. The win is from fewer loads, not coalescing improvement.
- Occupancy: for very sparse patterns (e.g., only 1% active), the threadgroups may not have enough work to fill the GPU. But sliding window (e.g., 12.5% active at W=4096, N=32768) is dense enough.

## Notes for Next Person
- Start with sliding-window attention (simplest pattern). Add two `#[constexpr]` params: `window_size` and `sink_tokens`.
- The skip condition is: `if _t < n_kv - window_size && _t >= sink_tokens { continue; }`.
- For BigBird, use a bitmask array where each bit represents a block of 128 positions. Precompute the active block list on the host and pass it as a `device uint*` buffer.
- Verify with `tile bench -f sdpa_decode` at N=8192, 16384, 32768.
