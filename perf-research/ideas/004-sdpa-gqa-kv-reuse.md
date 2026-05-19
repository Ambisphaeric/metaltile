# 004 — SDPA-vector decode: GQA-aware K/V reuse

## Metadata
- **Number**: 004
- **Name**: sdpa-gqa-kv-reuse
- **Source**: `perf-ideas.md` § Quick-wins — item 4
- **Status**: 🔴 **blocked / needs re-scope** (kernel architecture change, not a single-file tweak)
- **Worktree**: `../metaltile-perf-idea-4` (branch `perf/idea-4-gqa-kv-reuse`)
- **Assignee**: —

## Original Hypothesis
> With `gqa=4`, every 4 Q-heads share the same K/V row. Currently each Q-head loads K/V independently. Load once per kv-group, `simd_shuffle` to each Q-head.

## Target
- **Primary file(s)**:
  - `crates/metaltile-std/src/mlx/sdpa_vector.rs` (bench-compared MT kernel)
  - `crates/metaltile-std/src/ffai/sdpa_decode.rs` (production FFAI decode kernel)
  - `crates/metaltile-std/src/run_spec.rs` (bench dispatch for `SdpaVector`)
- **Bench filter**: `tile bench -vv -f sdpa_vector`
- **Shapes / dtypes to watch**: `H=32 N=4096 D=128 gqa=4 bf16` (idea claims MT% bf16 = 61% today, aim 80%+)

## Current Code Reality Check

### Dispatch model today
Both `mt_sdpa_vector` (MLX bench) and `sdpa_decode` (FFAI production) use **one threadgroup per Q-head**:

```
Grid:   [n_q_heads, 1, 1]
TPG:    1024  (= 32 simdgroups × 32 lanes)
```

With `gqa_factor=4` and `n_q_heads=32`:
- Threadgroups 0–3 all map to `kv_head = 0`
- Threadgroups 4–7 all map to `kv_head = 1`
- ...etc.

Each threadgroup independently walks **all** `n_kv` positions. Every KV cache element is loaded **4 times** from device memory (once per Q-head in the GQA group). The kernel body does not use threadgroup memory for K/V caching — loads go straight from device buffers to thread registers.

### MLX reference comparison
MLX's `sdpa_vector<T,D,V>` (from `.cache/mlx/mlx/backend/metal/kernels/sdpa_vector.h`) has **identical dispatch**: `tid.x = q_batch_head_idx`, `kv_head_idx = q_batch_head_idx / gqa_factor`. It does **not** coalesce K/V loads across GQA heads either. So MetalTile matches MLX here; any win would be a speedup *over* MLX, not parity.

### Why `simd_shuffle` doesn't directly apply
`simd_shuffle` operates **within a simdgroup** (32 lanes). The 4 Q-heads that share a KV head are in **4 separate threadgroups** today. They cannot `simd_shuffle` with each other. To share K/V loads, they must be in the **same threadgroup**.

### What "GQA-aware K/V reuse" actually requires

The idea as stated is a **kernel architecture change**, not a constant tweak:

| Aspect | Current | Required for K/V reuse |
|--------|---------|------------------------|
| Dispatch grid | `[n_q_heads, 1, 1]` | `[n_kv_heads, 1, 1]` |
| Threadgroup purpose | 1 Q-head | `gqa_factor` Q-heads |
| Simdgroup mapping | all 32 simdgroups serve 1 Q-head | partition into `gqa_factor` sub-groups |
| K/V source | device memory, loaded by every simdgroup | threadgroup memory, loaded cooperatively once per stride |
| Cross-simdgroup reduction | 32-entry `tg_max`/`tg_sum` | `gqa_factor` × `n_simd_per_qhead` entries |
| Output write | 1 Q-head per threadgroup | `gqa_factor` Q-heads per threadgroup |

#### Threadgroup memory math (feasibility check)
With a stride of `ns = 32` (the current loop stride) and `D = 128`:
- K chunk per stride: `32 × 128 × 2 bytes` (f16) = **8,192 B**
- V chunk per stride: `32 × 128 × 2 bytes` = **8,192 B**
- Total: **16,384 B** = 16 KB
- Threadgroup memory budget: ~32 KB on Apple GPUs

This **fits**. A cooperative load of one stride-chunk into threadgroup memory, followed by all Q-heads reading from it, is mechanically possible.

#### Reduction arrays with gqa=4
If 32 simdgroups are partitioned into 4 groups of 8 simdgroups (1 group per Q-head):
- `tg_max`: 4 × 8 = 32 entries (same size, re-partitioned)
- `tg_sum`: 4 × 8 = 32 entries (same size)
- `tg_out0–3`: 4 × 8 × 32 lanes = 1,024 entries (same size, re-indexed)

This also fits.

### Effort estimate
- New kernel variant or heavy rewrite of existing kernel: **medium**
- Bench dispatch update (`run_spec.rs`) to change grid shape and buffer bindings: **low**
- Correctness verification: GQA changes the mapping between lanes and Q-heads; any indexing bug affects 4 heads at once and may be subtle.
- **Overall**: **One-day to Multi-day**, not Quick-win.

## Baseline
```bash
tile bench -vv -f sdpa_vector
tile snap -o perf-research/results/004-baseline.json
```

Blocked on running this until we decide whether to pursue the architecture change.

## Risk Register
- **Alignment / indexing**: Repartitioning simdgroups among Q-heads is easy to get wrong. The `q_head = tgid_x * gqa_factor + sg / simdgroups_per_qhead` mapping must exactly match the output buffer layout.
- **Threadgroup memory bank conflicts**: Cooperative K/V loading needs a strided store pattern so 32 lanes writing 128 elements don't collide on the same bank.
- **Occupancy**: Fewer threadgroups (`n_kv_heads` instead of `n_q_heads`) could *reduce* occupancy for small head counts. With H=32, gqa=4 → n_kv_heads=8. That's only 8 threadgroups, which may not fill all GPU cores. The original H=8 case would have only 2 threadgroups — worse than today.
- **Performance may not improve**: The bottleneck might be Q-vector ALU (dot products) or the online-softmax updates, not K/V bandwidth. Need profiling to confirm.
- **MLX comparison**: Since MLX also doesn't do this optimization, the "MT%" metric in the bench is comparing against an equal-footing reference. A win here would be a genuine MetalTile advantage, not just catching up.

## Decision Needed
| Option | Effort | Notes |
|--------|--------|-------|
| A. Close as `blocked`, pick a different Quick-win | 0 | Ideas 5 (8-wide loads) or 6 (RMS unroll) are actual single-file tweaks. |
| B. Re-scope to One-day / Multi-day | Medium | Requires new kernel + dispatch rewrite. Promote to `ideas/004-sdpa-gqa-kv-reuse.md` as a project card. |
| C. Partial implementation: load K/V into registers once per simdgroup, no threadgroup sharing | Low? | Each simdgroup already loads K once per iteration. The redundancy is across threadgroups, not within. This option doesn't help. |

## Final Verdict (preliminary)
**Blocked / needs re-scoping.** The stated mechanism (`simd_shuffle`) cannot work across threadgroups. The real optimization is a **dispatch-shape change + cooperative threadgroup-memory K/V caching**, which is a kernel architecture rewrite, not a Quick-win constant tweak.

## Related Ideas
- **003** — Split-K for low-occupancy H=8 (also touches dispatcher, complementary: split-K increases occupancy; GQA reuse decreases threadgroups).
- **005** — 8-wide vectorized loads on f16/bf16 (genuine single-file tweak in `sdpa_vector.rs`).
- **M1** — ML-driven autotuner (a learned model could discover that GQA+split-K is the right combo).
