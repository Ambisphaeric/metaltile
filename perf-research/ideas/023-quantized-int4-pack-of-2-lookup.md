# 023 — Quantized GEMV: int4 pack-of-2 lookup

## Metadata
- **Number**: 023
- **Name**: quantized-int4-pack-of-2-lookup
- **Source**: `perf-ideas.md` § Op-level structural changes — item 23
- **Status**: 🔴 blocked
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> pack `(int4_lo, int4_hi)` as uint8 index into a 256-entry half2 LUT — single load yields two dequanted values.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/quantized.rs`
- **Bench filter**: `tile bench -vv -f quantized`
- **Shapes / dtypes**: int4 dequantize / qmv, `group_size=64`, `(4096, 4096)`

## Current Code Reality Check

The target file `quantized.rs` contains four categories of kernels:

1. **`mt_qmv_f32`** — quantized matvec (GEMV). Each threadgroup handles one output row. Threads stride over groups (`tid` → `_g` loop), and within each group they iterate over 8 `u32` packs (64 int4 values total). For each pack, 8 int4 values are extracted via shift+mask (`(packed >> shift) & 15`), dequantized (`s * int4_val + bias`), and FMA'd with the corresponding input element. Reduction is via `reduce_sum(acc)`.

2. **`mt_affine_dequantize_int4`** — elementwise dequantization. One thread per `u32` pack (8 nibbles). Extracts all 8 nibbles via explicit shift+mask, applies `scale * q + bias`, stores 8 outputs.

3. **`mt_affine_quantize_int4`** — elementwise quantization. One threadgroup per group of 64 elements. Finds min/max via `simd_min`/`simd_max`, computes scale/bias, then packs 8 quantized values per `u32`.

4. **int3/5/6 dequantize variants** — non-power-of-2 bit widths with byte-stream extraction.

### DSL capabilities and limitations

**What exists:**
- `threadgroup_alloc(name, size)` — creates a named threadgroup buffer (used in `sort.rs`, `arg_reduce.rs`, `sdpa_vector.rs`).
- `threadgroup_store(name, idx, val)` and `threadgroup_load(name, idx)` — dynamic indexing into threadgroup memory.
- `threadgroup_barrier()` — sync within threadgroup.

**What is missing (blocking this idea):**
- **No `half2` / vector type in user-facing DSL.** The hypothesis explicitly requires a "256-entry half2 LUT" — a vector-of-two-halves that can be loaded with one memory instruction. The DSL `load()` is scalar; `VectorLoad`/`VectorStore` exist at the IR level but are only created by the `VectorizePass` codegen pass from consecutive scalar accesses. Kernel authors cannot write `half2` or `float2` types.
- **No constant-array / LUT primitive.** While `threadgroup_alloc` creates a buffer, there is no way to declare a precomputed constant array (e.g., `constant half2 lut[256] = {...}`). A threadgroup LUT would need to be populated at runtime by the threads — one store per entry, then a barrier. For a 256-entry LUT with 64 threads, that's ~4 stores + barrier per threadgroup before any actual work begins.

### Per-group scale/bias dependency

The dequantization formula is `scale * q + bias`, where `scale` and `bias` are looked up **per group** (`g_idx = oindex / group_size`). This means:
- A global LUT storing pre-dequantized values would need to encode scale and bias for every group. For `n_groups=4096`, a 256-entry `half2` LUT per group is `4096 × 256 × 4 = 4 MB` — far exceeding the 64 KB Metal constant segment.
- A per-group LUT in threadgroup memory is possible in principle, but each threadgroup processes `gs_per_row = k / group_size` groups. For `k=4096, group_size=64`, that's **64 groups per row**. A 256-entry `f16` LUT per group would be `64 × 256 × 2 = 32 KB`, exactly filling the threadgroup memory limit — leaving zero room for any other tg state. A `f32` LUT would be 64 KB, exceeding the limit.
- Rebuilding the LUT for each group iteration (inside the `_g` loop) would require a `threadgroup_barrier()` + 256 stores per group. With 64 groups, that's 64 barriers and ~16K stores per threadgroup — catastrophic overhead.

### MLX reference check

MLX's actual quantized kernels (`qmv_fast_impl`, `qmv_impl` in `quantized.h`) use scalar dequantization + `simd_sum`, not LUT-based lookup. There is no `half2`, no LUT, and no pack-of-2 optimization in the MLX quantized path. This idea does not match the reference implementation.

## Baseline
Not benched — analytical assessment only. The hypothesized optimization requires DSL primitives that do not exist and produces impractical memory usage given the per-group scale/bias dependency.

## Risk Register
- **No vector type in DSL** — `half2` cannot be expressed in the `#[kernel]` DSL body. Same blocker as ideas 5, 8, 18. (from established patterns)
- **No constant-array primitive** — `threadgroup_alloc` exists but cannot be initialized with compile-time data. Runtime init costs 4 stores + barrier per threadgroup before work begins. (new finding)
- **Per-group scale/bias** — a LUT must either be per-group (too large for threadgroup memory when processing multiple groups per row) or rebuilt per group (too many barriers). The idea as written ignores this dependency. (new finding)
- **MLX doesn't do this** — the reference implementation confirms this is not a standard or proven optimization for quantized GEMV. (new finding)
- **Threadgroup memory limit** — 32 KB on M-series. Even a modest per-group LUT rapidly exhausts this budget when a row contains many groups. (from perf-ideas.md risk)

## Adjacent but different idea

**Raw int4→float 16-entry LUT in threadgroup memory.** If we ignore the `half2` vector requirement and the per-group scale/bias issue, a 16-entry `f32` LUT (just the integer values 0..15 as floats) is 64 bytes — trivial. Using it would replace the shift+mask extraction with a single `threadgroup_load`. However, this still requires:
1. Populating the LUT at runtime (16 stores + barrier).
2. The `scale * q + bias` arithmetic still happens after the lookup.
3. The shift+mask to get the int4 value is replaced by a shift+mask to get the LUT index — not clearly a win.

Given the overhead and the fact that MLX doesn't use this approach, the simpler path is to trust the current scalar extraction.

## Final Verdict
**Blocked / requires DSL extensions and kernel restructuring.**

The hypothesis requires three things the current DSL cannot provide:
1. `half2` vector types for paired-value LUT loads.
2. Constant-array / LUT declaration primitives.
3. A memory layout that accommodates per-group scale/bias without exhausting threadgroup memory or adding barrier overhead.

Even if vector types existed, the per-group dependency makes the 256-entry LUT design impractical for the GEMV kernel (64 groups per row × 256 entries = 32 KB tg memory, or 64 barriers for per-group rebuild). MLX's own implementation confirms the simpler scalar approach is the right one.

If the intent was to optimize int4 extraction, the more plausible path is a **codegen-level vectorization** of the 8 sequential `(packed >> shift) & 15` extractions into a `VectorLoad` + `VectorExtract` pattern — but that is a compiler-pass change, not a kernel-body tweak.

## Related Ideas
- **021** — FP4 dequant packed bit ops (blocked for different reasons).
- **022** — dequant GEMV simdgroup matmul (blocked: GEMM primitive on GEMV).
- **005** — SDPA vec8 loads (same DSL vector-type blocker).
- **020** — Copy vectorize (investigate `vectorize.rs` pass — the codegen path for automatic vectorization).
