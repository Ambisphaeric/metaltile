# 001 — SDPA tile: bump BLOCK_M on f16/bf16

## Metadata
- **Number**: 001
- **Name**: sdpa-tile-block-m
- **Source**: `perf-ideas.md` § Quick-wins — item 1
- **Status**: `blocked` (target kernel is scalar vector, not tiled; BLOCK_M does not exist yet)
- **Worktree**: `../metaltile-perf-idea-1` (branch `perf/idea-1-sdpa-block-m`)
- **Assignee**: —

## Original Hypothesis
> f16/bf16 halve register/threadgroup pressure vs f32, so the Q-tile rows can grow (16→32) without spilling. K/V load amortization scales with BLOCK_M.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/scaled_dot_product_attention.rs`
- **Bench filter**: `tile bench -vv -f sdpa`
- **Shapes / dtypes to watch**: `H=32 N=4096 D=128 f16/bf16`

## Current Code Reality Check

The target file `scaled_dot_product_attention.rs` implements **mt_sdpa**, a scalar vector decode kernel (single Q-row per work item). There is **no BLOCK_M constant** in this file. The kernel dispatches `[H, 1, 1]` threadgroups with `tpg=1024` (32 simdgroups × 32 lanes). Each threadgroup processes exactly one query position.

Constants in the kernel today:
- `tpg = 1024` (fixed by the `#[bench_kernel]` macro)
- Threadgroup arrays: `tg_max` (32), `tg_sum` (32), `tg_out0–3` (1024)
- Head dim is hardcoded to 128 via `lane * 4u32` loads (4 × 32 = 128)

The perf idea as written assumes a **tiled FlashAttention kernel** (like MLX's `steel_attention` or the MLX `sdpa_vector` with `BN=32` interpreted as BLOCK_N). In MetalTile, the tiled SDPA kernels live in:
- `crates/metaltile-std/src/mlx/steel/attn/steel_attention.rs` — **NOT YET IMPLEMENTED** in `#[kernel]` DSL
- `crates/metaltile-std/src/mlx/steel/attn/steel_attention_nax.rs` — **NOT YET IMPLEMENTED**

These stubs explicitly say: *"No simdgroup matrix or multi-level attention tiling is implemented."*

### So where could "BLOCK_M" come from?
1. **The MLX reference** — `scaled_dot_product_attention.metal` ships `sdpa_vector<T,D,V>` (single Q) and `steel_attention` (tiled). The `steel_attention` template has `BM`, `BN`, `BK` block constants, but MetalTile has no `#[kernel]` port of it.
2. **Dispatch shape** — one could reinterpret `tgid_y` as a BLOCK_M dimension and dispatch `[H, BLOCK_M, 1]`, but the kernel body has no `program_id::<1>()` usage and would need a rewrite.

## Baseline
Blocked until a tiled SDPA kernel exists. If we want to pursue this idea, the prerequisite work is:
- Port `steel_attention.metal` (or `steel_attention_nax.metal`) to the `#[kernel]` DSL, **or**
- Rewrite `scaled_dot_product_attention.rs` into a multi-Q-row tile kernel.

For reference, the current vector kernel baseline can be captured with:
```bash
tile bench -vv -f sdpa
tile snap -o results/001-baseline-vector.json
```

## Risk Register
- **Spill at D=128** if registers exceed 128/thread (from original idea).
- **Scope creep**: "bump one constant" becomes "implement a whole tiled attention kernel" because the constant doesn't exist yet.
- **Correctness**: Any multi-Q-row rewrite must preserve the online-softmax numerics exactly.

## Decision Needed
| Option | Effort | Notes |
|--------|--------|-------|
| A. Close as `blocked`, pick a later Quick-win | 0 | Idea 2 (BLOCK_N 64→128) is in the same file and has actual constants to tweak. |
| B. Re-scope to "implement tiled SDPA" | Multi-day | Moves from Quick-wins to Multi-day category. |
| C. Re-interpret BLOCK_M as `tpg` or dispatch dim | Low? | Might not yield the hypothesized K/V amortization because the kernel body doesn't tile. |

## Final Verdict
**Blocked / needs re-scoping.** The constant this idea wants to tweak does not exist in the target file. The next actionable step is either (a) pick Idea 2 which is adjacent and well-formed, or (b) formally promote this to a Multi-day "Implement tiled SDPA" project.

## Related Ideas
- **002** — `BLOCK_N` 64→128 on D=128 (same file family, actual constants exist in MLX reference).
- **003** — Split-K for low-occupancy H=8 (touches dispatcher, not kernel constants).
- **M3** — Persistent-kernel graph capture (moonshot, would change the whole dispatch model).
