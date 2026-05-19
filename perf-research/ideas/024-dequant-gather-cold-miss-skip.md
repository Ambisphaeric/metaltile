# 024 — dequant_gather: skip dequant for cold misses

## Metadata
- **Number**: 024
- **Name**: dequant-gather-cold-miss-skip
- **Source**: `perf-ideas.md` § Op-level structural changes — item 24
- **Status**: ⚪ no-op / premature
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> rare — most lookups hit the L1; the kernel currently dequants unconditionally. Profile to confirm there's a measurable cold-miss frequency before chasing.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/dequant_gather.rs`
- **Bench filter**: `tile bench -vv -f dequant_gather`
- **Shapes / dtypes**: int3/4/5/6/8, embedding-table gather shapes

## Current Code Reality Check

The target file `dequant_gather.rs` implements **five** quantized embedding-table gather kernels (int3/4/5/6/8). Each thread computes one output element `(token, d)`:

1. Loads the token index from `indices[token]`.
2. Computes the bit offset for element `d` in the token's packed weight row.
3. Loads 1–2 `u32` words from `weight` (the bit stream for that token's row).
4. Extracts the quantized value via shift+mask+merge (`lo | hi`).
5. Loads `scale` and `bias` for the group from `scales`/`biases`.
6. Dequantizes: `q.cast::<f32>() * scale + bias`.
7. Stores the result.

Dispatch is `KernelMode::Grid3D`, one thread per output element. The kernel is **memory-bound by design**: each thread does ~4 device-memory loads (indices, weight×2, scales, biases) and 1 store, with a small amount of bit-extraction arithmetic. The dequantization itself (`q * scale + bias`) is 2 FMAs — negligible compared to memory latency.

### Measurement tool does not exist

The proposed measurement is `tile profile mt_dequant_gather`. The `tile` CLI **does not have a `profile` command**. Available commands are `bench`, `build`, `inspect`, `device`, `snap`, `diff`. There is no facility for L1 cache-miss profiling, counter sampling, or hardware-event capture.

Even if a profiler existed, Metal does **not** expose L1 cache miss counters to shader code. There is no `cache_hit` predicate, no `prefetch` with feedback, and no way for a thread to know whether a `load()` hit L1 or went to DRAM.

### The optimization is not kernel-level

"Skip dequant for cold misses" implies:
- Detect that the weight data is not in cache.
- Skip the dequantization step.
- Write something else (or nothing) to the output.

This is impossible at the kernel level because:
1. There is no cache-state visibility in MSL.
2. Skipping dequantization would produce incorrect output values (zeros or garbage) unless the consumer knows which elements were skipped — a contract change requiring graph-level support.
3. The kernel is element-parallel; there is no shared cache of dequantized rows across threads of the same token.

The real optimization would be a **graph-level embedding cache**: keep recently-used token embeddings in dequantized form in device memory, and only call `dequant_gather` for tokens not in the cache. That is a dispatcher/graph-scheduler feature, not a kernel tweak.

## Baseline
Not benched — analytical assessment only. The kernel is fundamentally memory-bound; the arithmetic (dequantization) is already minimal. `tile bench -f dequant_gather` would show GB/s close to the device's effective DRAM bandwidth, confirming the bottleneck is memory, not ALU.

## Risk Register
- **`tile profile` does not exist** — the proposed measurement tool is not implemented. (new finding)
- **No cache-state visibility in MSL/Metal** — even with a profiler, "skipping dequant for cold misses" cannot be implemented as a kernel-level branch. (new finding)
- **Graph-level optimization, not kernel-level** — the real fix is a dispatcher-level embedding cache, requiring `#[bench_kernel]` or `run_spec.rs` changes plus a new graph-scheduler primitive. (new finding)
- **Premature by the idea's own admission** — the risk section says "premature; verify the assumption first." We have verified: the assumption cannot be measured with current tools, and the optimization is structurally impossible at the kernel level. (from perf-ideas.md)

## Final Verdict
**No-op / premature.**

The idea asks us to profile cache misses before optimizing. The profiling tool (`tile profile`) does not exist, and Metal does not expose cache-state information to kernel code. The kernel is already memory-bound with minimal ALU work; the dequantization is not the bottleneck.

If cold-cache latency is a problem in practice, the correct optimization is a **graph-level dequantized-embedding cache** (keep hot token embeddings in unpacked device memory). That is a dispatcher/graph feature, not a single-file kernel tweak. It should be filed as a separate multi-day idea if needed.

## Related Ideas
- **019** — Gather tg prefetch hot indices (dispatch restructuring, related to caching).
- **021–023** — Other quantized kernels (same family, different optimization targets).
