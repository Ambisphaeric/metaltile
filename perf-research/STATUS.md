# Perf Research Status Board

> Living tracker for the `perf-ideas.md` hopper.  
> One row per idea. Update as worktrees spin up and experiments complete.

| # | Name | Category | Status | Worktree | Baseline Snap | Final Snap | Verdict | Notes |
|---|------|----------|--------|----------|---------------|------------|---------|-------|
| 001 | SDPA tile: bump BLOCK_M on f16/bf16 | Quick-win | 🔴 blocked | `../metaltile-perf-idea-1` | — | — | — | Target kernel is scalar vector; BLOCK_M constant does not exist. Needs re-scope or prerequisite tiled kernel. [Details](ideas/001-sdpa-tile-block-m.md) |
| 002 | SDPA: BLOCK_N 64 → 128 on D=128 | Quick-win | ⚪ not-started | — | — | — | — | |
| 003 | SDPA: split-K for low-occupancy H=8 | Quick-win | ⚪ not-started | — | — | — | — | |
| 004 | SDPA-vector decode: GQA-aware K/V reuse | Quick-win | 🔴 blocked | `../metaltile-perf-idea-4` | — | — | — | `simd_shuffle` can't cross threadgroups; real fix is dispatch-shape change + cooperative tg-mem K/V caching. [Details](ideas/004-sdpa-gqa-kv-reuse.md) |
| 005 | SDPA-vector: 8-wide vec loads f16/bf16 | Quick-win | 🔴 blocked | — | — | — | — | DSL has no vector-load primitive. [Details](ideas/005-sdpa-vec8-loads.md) |
| 006 | RMS-norm: unroll 4 → 8 | Quick-win | ⚫ abandoned | `../metaltile-perf-idea-6` | — | — | Regression | 8-wide unroll pushes register pressure to 162r (was 9r), occupancy drops to 73%, kernel becomes register-limited. Reverted. [Details](ideas/006-rms-norm-unroll-8.md) |
| 007 | Softmax: simdgroup reduce for small N | Quick-win | 🟢 done | `../metaltile-perf-idea-7` | — | — | Small win | tpg=32 beats tpg=256 by ~1.65× on N=32. Adds `softmax_small_n` bench variant. Real value is dispatch heuristic. [Details](ideas/007-softmax-small-n.md) |
| 008 | Softmax: float4 loads on f16/bf16 | Quick-win | 🔴 blocked | — | — | — | — | DSL has no vector-load primitive. Same blocker as #5. [Details](ideas/008-softmax-float4-loads.md) |
| 009 | LayerNorm: mirror RMS-norm tweaks | Quick-win | ⚫ abandoned | — | — | — | — | Same register pressure issue as #6. Not benched — predicted worse due to more live state. [Details](ideas/009-layernorm-mirror-rms.md) |
| 010 | GEMV: tune `simd_per_tg` per K | Quick-win | 🟢 done | `../metaltile-perf-idea-10` | `010-run2.json` | `010-run2.json` | Small win for f16 | tpg=512 beats baseline by +1.8% on f16. tpg=1024 is a −20% regression on f16. f32/bf16 flat. [Details](ideas/010-gemv-tpg-sweep.md) |
| 011 | GEMV-masked: dense fallback | Quick-win | 🔴 blocked | — | — | — | — | Dispatcher-level heuristic. `#[bench_kernel]` doesn't support runtime kernel selection. [Details](ideas/011-gemv-masked-dense-fallback.md) |
| 012 | all_reduce: two-stage simd→tg | Quick-win | ⚪ no-op | — | — | — | — | Already optimal — `tile inspect` confirms. [Details](ideas/012-020-feasibility-study.md#12-all_reduce-two-stage-simdthreadgroup) |
| 013 | row-reduce: rows-per-tg small N | Quick-win | ⚠️ feasible | — | — | — | — | Dispatch-level change. Small-N only. [Details](ideas/012-020-feasibility-study.md#13-row-reduce-rows-per-threadgroup-when-n-is-small) |
| 014 | scan: simd_prefix_inclusive_sum | Quick-win | ⚪ no-op | — | — | — | — | Already implemented. [Details](ideas/012-020-feasibility-study.md#14-scan-prefer-simd_prefix_inclusive_sum) |
| 015 | argmax: hold 847% | Quick-win | ⚪ marginal | — | — | — | — | Already optimal; graph-level profiling needed. [Details](ideas/012-020-feasibility-study.md#15-argmax-refuse-to-slow-down-847) |
| 016 | RoPE: precompute sin/cos tg-mem | One-day | 🔴 blocked | — | — | — | — | Dispatch restructuring needed. [Details](ideas/012-020-feasibility-study.md#16-rope-precompute-sincos-to-threadgroup-memory) |
| 017 | RoPE-into-QKV fusion | One-day | 🔴 blocked | — | — | — | — | New kernel + bench harness. [Details](ideas/012-020-feasibility-study.md#17-rope-into-qkv-fusion) |
| 018 | KV-cache: vectorized copy | One-day | 🔴 blocked | — | — | — | — | DSL lacks vector primitives. [Details](ideas/012-020-feasibility-study.md#18-kv-cache-append-vectorized-aligned-copy) |
| 019 | Gather: tg prefetch hot indices | One-day | 🔴 blocked | — | — | — | — | Dispatch restructuring needed. [Details](ideas/012-020-feasibility-study.md#19-gather-prefetch-to-threadgroup-for-hot-indices) |
| 020 | Copy: vectorize stride-1 | One-day | ⚠️ feasible | — | — | — | — | Investigate `vectorize.rs` pass. [Details](ideas/012-020-feasibility-study.md#20-strided-copy-emit-vec-types-for-stride-1-axes) |
| 021 | FP4 dequant: packed bit ops | One-day | 🔴 blocked | — | — | — | — | Target file is scalar quantize-dequant roundtrip, not packed FP4 dequant. Ill-formed against current code. [Details](ideas/021-fp4-dequant-packed-bit-ops.md) |
| 022 | Quantized int4 GEMV: simdgroup_matrix multiply | One-day | 🔴 blocked | — | — | — | — | `simdgroup_matrix_multiply` is a GEMM primitive, not GEMV. GEMV has N=1; using it requires padding and discarding 7/8 of work. MLX's own qmv kernels use scalar dequant+simd_sum. [Details](ideas/022-dequant-gemv-simdgroup-matmul.md) |
| 023–035 | *(one-day items)* | One-day | ⚪ not-started | — | — | — | — | |
| 036–055 | *(multi-day items)* | Multi-day | ⚪ not-started | — | — | — | — | |
| M1–M10 | *(moonshots)* | Moonshot | ⚪ not-started | — | — | — | — | |

## Legend
- 🔴 **blocked** — prerequisite missing or idea ill-formed against current code
- 🟡 **in-progress** — worktree checked out, bench cycles running
- 🟢 **done** — final snap saved, verdict recorded
- ⚪ **not-started** — no worktree yet
- ⚫ **abandoned** — idea discarded with reason

## Quick Commands

Spin up a new worktree for idea NNN:
```bash
git fetch upstream dev
git worktree add -b perf/idea-NNN-<name> ../metaltile-perf-idea-NNN dev
```

Save a snapshot:
```bash
tile snap -o perf-research/results/NNN-<label>.json
```

Diff two snapshots:
```bash
tile diff perf-research/results/NNN-baseline.json perf-research/results/NNN-final.json
```

## Methodology Reminders
1. Run bench *twice* before claiming a regression (DVFS stabilization after recompile).
2. Always check the `ok` column. Speedup with a correctness regression is not a win.
3. Watch `cv%` — anything > 5% means the win is bench noise.
4. `min_us` drives GB/s; `p95`/`p99`/`cv%` from `-vv` are the trust signals.
