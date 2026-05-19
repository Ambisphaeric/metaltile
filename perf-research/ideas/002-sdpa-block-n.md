# 002 — SDPA: BLOCK_N 64 → 128 on D=128

## Metadata
- **Number**: 002
- **Name**: sdpa-block-n
- **Source**: `perf-ideas.md` § Quick-wins — item 2
- **Status**: 🔴 **blocked** — same root cause as idea #1
- **Worktree**: — (not created; same file as #1)
- **Assignee**: —

## Hypothesis
> FlashAttention-2 paper shows BLOCK_N=128 wins on D=128 once K/V fits in threadgroup memory. M-series threadgroup mem (32 KB) easily fits 128×128 f16.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/scaled_dot_product_attention.rs`
- **Bench filter**: `tile bench -vv -f sdpa`
- **Shapes / dtypes to watch**: `H=32 N=4096 D=128 f16/bf16`

## Current Code Reality Check
Same target as idea #1. The file `scaled_dot_product_attention.rs` implements **`mt_sdpa`**, a scalar vector decode kernel. There is **no BLOCK_N constant** in this file.

The MLX reference `sdpa_vector.h` does define `BN = 32` (block size for KV sequence stride) and `BD = 32` (block size for head_dim). The MetalTile kernel uses `n_simd` (runtime) instead of `BN` (compile-time).

Key difference from idea #1: this idea specifically wants to bump `BLOCK_N` from 64→128. But the current kernel doesn't even have a `BLOCK_N` concept — it uses `range(sg, n_kv, ns)` where `ns = n_simd` and `sg = simd_id`. The KV walk stride is `ns` (number of simdgroups), not a fixed `BLOCK_N`.

### So where could "BLOCK_N" come from?
Same as idea #1:
1. **The MLX reference** — `sdpa_vector<T,D,V>` has `BN=32`, but MetalTile has no `#[kernel]` port of it.
2. **The dispatch shape** — could be interpreted as how many KV positions each simdgroup covers, but that's `n_kv / ns`, not a constant.

## Baseline
Blocked until a tiled SDPA kernel exists.

## Risk Register
- **Same as #1**: the constant this idea wants to tweak does not exist in the target file.
- **D=64 shapes**: original idea notes this may regress. Without a tiled kernel, we can't test this.

## Final Verdict
**Blocked / needs re-scoping.** Same root cause as idea #1: the target kernel is scalar vector, not tiled FlashAttention. BLOCK_N does not exist.

## Related Ideas
- **001** — SDPA tile: bump BLOCK_M (same file, same blocker)
- **003** — Split-K for low-occupancy H=8 (touches dispatcher, not kernel constants)
