# Perf Research Status Board

> Living tracker for the `perf-ideas.md` hopper.  
> One row per idea. Update as worktrees spin up and experiments complete.

| # | Name | Category | Status | Worktree | Baseline Snap | Final Snap | Verdict | Notes |
|---|------|----------|--------|----------|---------------|------------|---------|-------|
| 001 | SDPA tile: bump BLOCK_M on f16/bf16 | Quick-win | 🔴 blocked | `../metaltile-perf-idea-1` | — | — | — | Target kernel is scalar vector; BLOCK_M constant does not exist. Needs re-scope or prerequisite tiled kernel. [Details](ideas/001-sdpa-tile-block-m.md) |
| 002 | SDPA: BLOCK_N 64 → 128 on D=128 | Quick-win | 🔴 blocked | — | — | — | — | Same blocker as #1: scalar vector kernel, no BLOCK_N constant. [Details](ideas/002-sdpa-block-n.md) |
| 003 | SDPA: split-K for low-occupancy H=8 | Quick-win | 🔴 blocked | — | — | — | — | Requires split-K kernel variant + dispatcher changes + merge kernel. Multi-day effort. [Details](ideas/003-sdpa-split-k.md) |
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
| 023 | Quantized GEMV: int4 pack-of-2 lookup | One-day | 🔴 blocked | — | — | — | — | Requires `half2` vector type + constant-array primitive in DSL. Per-group scale/bias makes 256-entry LUT impractical (32KB+ tg mem or 64 barriers). MLX doesn't use this. [Details](ideas/023-quantized-int4-pack-of-2-lookup.md) |
| 024 | dequant_gather: skip dequant for cold misses | One-day | ⚪ no-op | — | — | — | — | `tile profile` does not exist. No cache-state visibility in MSL. Real fix is graph-level embedding cache, not kernel-level skip. [Details](ideas/024-dequant-gather-cold-miss-skip.md) |
| 025 | Sort: 4-way bitonic merge | One-day | ⚠️ feasible / high risk | — | — | — | — | Current kernel at 117r (thread-limited). 4-way merge would double registers, likely spill. Real gap is MLX uses merge sort, not bitonic. [Details](ideas/025-sort-4-way-bitonic-merge.md) |
| 026 | Sampling: radix-select top-k | One-day | 🔴 blocked | — | — | — | — | Target file contains categorical sampling kernel, not top-k. No top-k kernel in MetalTile or MLX. New kernel + bench harness required. [Details](ideas/026-sampling-radix-select-topk.md) |
| 027 | SSM: scan with state vectorization | One-day | ⚪ no-op | — | — | — | — | `mt_ssm_step` already vectorizes state_dim across 32 threads with `simd_sum`. No scan exists to fuse — state dims are independent. [Details](ideas/027-ssm-scan-state-vectorization.md) |
| 028 | logsumexp: fuse max + sum-exp | One-day | ⚪ no-op | — | — | — | — | Kernel already uses single-pass online logsumexp (matches MLX `logsumexp_looped`). MT%=154–238%. [Details](ideas/028-logsumexp-fuse-max-sum-exp.md) |
| 029 | Short-row cooperative groups | One-day | ⚠️ feasible | — | — | — | — | Dispatch heuristic: tpg=32 for N≤32, same pattern as #007. [Details](ideas/029-short-row-cooperative-groups.md) |
| 030 | binary_two: FMA autovec diagnostic | One-day | ⚪ no-op | — | — | — | — | Kernel computes x+y and x*y independently — no FMA pattern exists. [Details](ideas/030-binary-two-fma-autovec.md) |
| 031 | Unary: emit `metal::precise::sigmoid` | One-day | ⚠️ feasible | — | — | — | — | `mt_sigmoid` manually expands formula; DSL has `sigmoid()` builtin. One-line fix. [Details](ideas/031-unary-precise-intrinsics.md) |
| 032 | SwiGLU/GELU fuse with matmul epilogue | One-day | 🔴 blocked | — | — | — | — | No GEMM kernel in DSL; needs multi-day project. [Details](ideas/032-swiglu-gelu-fuse-matmul-epilogue.md) |
| 033 | argmin variant in arg_reduce | One-day | ⚠️ feasible | — | — | — | — | Copy-paste argmax, flip init + comparison. ~30 lines. [Details](ideas/033-argmin-variant.md) |
| 034 | softmax + attention epilogue fusion | One-day | 🔴 blocked | — | — | — | — | No tiled attention kernel; moonshot-level scope. [Details](ideas/034-softmax-attention-fusion.md) |
| 035 | random: 64-bit state / vec4 generation | One-day | 🔴 blocked | — | — | — | — | `mt_random_hash` is a toy hash, not a PRNG. Hypothesis ill-formed. [Details](ideas/035-random-xorshift-vec4.md) |
| 041 | schedule.rs: software pipelining | Codegen | 🔴 blocked | — | — | — | — | Target pass is a tile annotator, not a loop scheduler. Needs new pass. [Details](ideas/041-schedule-software-pipelining.md) |
| 042 | licm.rs: hoist gather indices | Codegen | ⚪ no-op | — | — | — | — | Already hoists loop-invariant `Load` from read-only params. [Details](ideas/042-licm-hoist-gather-indices.md) |
| 043 | cse.rs: extend across simdgroup boundaries | Codegen | ⚠️ feasible | — | — | — | — | Block-local CSE misses cross-branch common subexpressions. Needs re-scoping. [Details](ideas/043-cse-across-simdgroup-boundaries.md) |
| 044 | if_conversion.rs: predicate tiny ifs | Codegen | ⚪ no-op | — | — | — | — | Pass already handles Diamond shapes; `gemv_masked` has no `Op::If`. [Details](ideas/044-if-conversion-predicate-tiny-ifs.md) |
| 045 | value_sink.rs: sink threadgroup stores | Codegen | 🔴 blocked | — | — | — | — | Pass excludes side-effecting ops; threadgroup-store motion is unsafe here. [Details](ideas/045-value-sink-threadgroup-stores.md) |
| 036 | vectorize.rs: 8-wide on f16/bf16 | Codegen | ⚪ no-op | — | — | — | — | `MAX_VEC_LEN=8` already in pass; BF16 already vectorizable. [Details](ideas/036-vectorize-8-wide-f16-bf16.md) |
| 037 | vectorize.rs: strided-but-aligned stores | Codegen | ⚠️ feasible | — | — | — | — | Pass only coalesces single-buffer contiguous. Strided/interleaved needs re-scope. [Details](ideas/037-vectorize-strided-aligned-stores.md) |
| 038 | fusion.rs: epilogue fusion onto reductions | Codegen | ⚠️ feasible | — | — | — | — | Post-reduction elementwise already fused; reducing into FusedElementwise is marginal. [Details](ideas/038-fusion-epilogue-reductions.md) |
| 039 | fusion.rs: multi-reduction in one pass | Codegen | ⚠️ feasible | — | — | — | — | Is loop fusion, not operator fusion. Needs new pass or hand-written kernel. [Details](ideas/039-fusion-multi-reduction.md) |
| 040 | unroll.rs: register-pressure-aware unroll | Codegen | ⚠️ feasible | — | — | — | — | `register_estimate.rs` exists but is not consulted by `UnrollPass`. Prevents #006-style catastrophes. [Details](ideas/040-unroll-register-aware.md) |
| 046–055 | *(runtime / build items)* | Runtime/Build | ⚪ not-started | — | — | — | — | |
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
