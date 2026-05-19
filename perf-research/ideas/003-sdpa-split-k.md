# 003 — SDPA: split-K for low-occupancy H=8 shapes

## Metadata
- **Number**: 003
- **Name**: sdpa-split-k
- **Source**: `perf-ideas.md` § Quick-wins — item 3
- **Status**: 🔴 **blocked / needs re-scope** (dispatcher + new kernel variant)
- **Worktree**: —
- **Assignee**: —

## Hypothesis
> H=8 fills only 25% of M1 Max's cores. Splitting K-sequence into 4 chunks × 8 threadgroups = 32 threadgroups = full occupancy. Cost is a tiny second-stage merge.

## Target
- **Primary file(s)**:
  - `crates/metaltile-std/src/run_spec.rs` (the `run_attention` arm)
  - New split-K kernel variant in `mlx/scaled_dot_product_attention.rs`
- **Bench filter**: `tile bench -f sdpa -vv`
- **Shapes / dtypes to watch**: `H=8 N=2048 D=128 f32` (idea claims ~150 GB/s baseline, target ~250 GB/s)

## Current Code Reality Check

### Dispatcher today
`run_attention` in `run_spec.rs` dispatches:
```rust
// MT dispatch
grid: [h, 1, 1]    // h = n_q_heads = 8
tpg: 1024           // 32 simdgroups × 32 lanes
```

With H=8, only **8 threadgroups** are launched. M1 Max has ~32 GPU cores, so each core gets at most 1 threadgroup. Occupancy is low.

### The kernel body
The `mt_sdpa` kernel (in `scaled_dot_product_attention.rs`) walks the full `n_kv` sequence in the inner loop:
```rust
for _t in range(sg, n_kv, ns) {
    // each simdgroup walks every ns-th KV position
}
```

Split-K would require:
1. **First stage**: Split `n_kv` into `k` chunks. Each of the H=8 threadgroups processes one chunk. Now we have `8 × k` threadgroups.
2. **Second stage**: Merge the partial online-softmax results from each chunk. This is a separate kernel (or a second pass in the same kernel).

### What "split-K" actually requires
| Component | Current | Needed for split-K |
|-----------|---------|-------------------|
| Dispatch grid | `[h, 1, 1]` | `[h, k, 1]` or `[h*k, 1, 1]` |
| Kernel loop | `range(sg, n_kv, ns)` | `range(sg, chunk_end, ns)` |
| Online softmax state | one per threadgroup | one per (head, chunk) |
| Merge kernel | none | new kernel or second pass |
| Threadgroup memory | `tg_max[32]`, `tg_sum[32]` | `tg_max[k][32]`, `tg_sum[k][32]` |

### Effort estimate
- Modify `run_attention` dispatch to launch split-K grid: **medium**
- Modify kernel to accept `chunk_start`/`chunk_end` instead of full `n_kv`: **low** (just change loop bounds)
- Implement merge kernel or second pass: **medium** (needs new `BenchSpec`, new kernel)
- Correctness verification: split-K changes the reduction tree; online-softmax merge must be numerically exact.
- **Overall**: **One-day to Multi-day**, not Quick-win.

## Baseline
```bash
tile bench -vv -f sdpa
tile snap -o results/003-baseline.json
```
Blocked on running this until we decide to implement split-K.

## Risk Register
- **Epilogue merge cost**: The idea says "tiny second-stage merge" but it's a separate kernel dispatch. For small H=8, the merge might be a significant fraction of total time.
- **Online-softmax numerics**: Split-K requires merging partial softmax results (max + sum_exp) across chunks. The formula is `new_max = max(max_a, max_b)` and `new_sum = sum_a * exp(max_a - new_max) + sum_b * exp(max_b - new_max)`. This is exact in f32 but adds operations.
- **Threadgroup memory**: With k=4 chunks, the output reduction arrays grow by 4×. Still fits in 32 KB, but changes the memory layout.
- **Occupancy math**: M1 Max has 32 GPU cores, but each core can run multiple threadgroups concurrently. The "25% occupancy" claim may be overstated — 8 threadgroups on 32 cores might still achieve decent parallelism if each threadgroup has enough work (N=2048 is significant).

## Decision Needed
| Option | Effort | Notes |
|--------|--------|-------|
| A. Close as blocked, pick a genuine Quick-win | 0 | Ideas 10, 7 are actual single-file/param changes. |
| B. Re-scope to Multi-day "implement split-K SDPA" | Medium-High | Promote to ideas 36–55 range. Requires new kernel + dispatch. |
| C. Verify occupancy claim first | Low | Profile H=8 N=2048 with existing kernel to see if occupancy is actually the bottleneck. Could be memory-bound, not occupancy-bound. |

## Final Verdict (preliminary)
**Blocked / needs re-scoping.** The idea requires a new split-K kernel variant + dispatcher changes + merge kernel. This is not a Quick-win constant tweak. Before pursuing, verify whether the H=8 case is actually occupancy-limited (profile it) or already memory-bound.

## Related Ideas
- **001–002** — BLOCK_M/BLOCK_N tweaks (same kernel family, different constants)
- **M1** — ML-driven autotuner (could learn that split-K is optimal for H<16)
