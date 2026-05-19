# 022 — Quantized int4 GEMV: simdgroup_matrix multiply

## Metadata
- **Number**: 022
- **Name**: dequant-gemv-simdgroup-matmul
- **Source**: `perf-ideas.md` § Op-level structural changes — item 22
- **Status**: 🔴 blocked
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> currently dequant→scalar mul. Dequant a tile into threadgroup memory then use `simdgroup_matrix_multiply` (16×16×16 tile).

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/dequant_gemv.rs`
- **Bench filter**: `tile bench -vv -f dequant_gemv`
- **Shapes / dtypes**: int4, shapes like `(m=4096, k=4096)` with `group_size=128`

## Current Code Reality Check

The target file `dequant_gemv.rs` implements **five** quantized GEMV kernels (int3/4/5/6/8) using two macro-generated bodies:

- **Pack-strided** (`int4`, `int8`): Each thread loads one `u32` pack containing `32/bits` quantized values (8 for int4), extracts each via shift+mask, dequantizes (`q * scale + bias`), and performs a scalar FMA with the corresponding input element. One `u32` load amortises across 8 FMAs.
- **Element-strided** (`int3`, `int5`, `int6`): Each thread strides over individual elements using a two-word bit-stream formula (1–2 `u32` loads + 5 bit-extraction ops + 1 FMA).

All kernels are `KernelMode::Reduction`, dispatching `[m, 1, 1]` threadgroups with `tpg` threads. Each threadgroup computes **one output row**. The final result is obtained via `reduce_sum(acc)` and stored by thread 0.

### DSL simdgroup primitives: available but irrelevant

The DSL *does* expose `simdgroup_alloc`, `simdgroup_elem_store`, `simdgroup_elem_load`, and `simdgroup_matmul` — they are used in `steel_gemm_fused.rs`. The MSL codegen emits `simdgroup_matrix<T,M,N>` and `simdgroup_multiply_accumulate`. So the **primitives exist**.

However, **`simdgroup_matrix_multiply` is a GEMM primitive, not a GEMV primitive**:
- It computes `C += A × B` where all three operands are `simdgroup_matrix` tiles (e.g., 8×8, 16×8, 8×16).
- A GEMV is `y = W × x` where `W` is `[M, K]` and `x` is `[K, 1]`. The "N" dimension is 1.
- To use `simdgroup_matrix_multiply` for GEMV, one would need to pad the vector to a matrix shape (e.g., replicate `x` across 8 columns to form a K×8 tile), compute an M×8 result, and discard 7/8 of the work. This is architecturally wasteful.

### MLX reference reality check

MLX's actual quantized GEMV kernels (`qmv_fast_impl`, `qmv_impl` in `quantized.h`) do **not** use `simdgroup_matrix_multiply`. They use the same scalar dequant+dot approach:
- `load_vector<T, U, values_per_thread, bits>()` to unpack/dequant weights into a `thread` array.
- `qdot<U, values_per_thread, bits>()` to compute a local dot product.
- `simd_sum()` to reduce across the simdgroup.

MLX *does* use `simdgroup_index_in_threadgroup` and `thread_index_in_simdgroup`, but only for **thread indexing** (to map threads to output rows and pack offsets), not for matrix multiplication. The only MLX kernels using `simdgroup_matrix_multiply` are **steel GEMM** and **convolution** (`mma.h`), both of which are full matrix-matrix operations.

### What the hypothesis actually describes

The hypothesis conflates **GEMM tiling** (where `simdgroup_matrix_multiply` is the right tool) with **GEMV** (where it is not). Dequantizing a tile into threadgroup memory and then doing a simdgroup matmul is the steel GEMM pattern, not the quantized GEMV pattern.

## Baseline
Not benched — analytical assessment only. The hypothesized optimization cannot be applied to the target kernel's operation (GEMV).

## Risk Register
- **Architectural mismatch**: `simdgroup_matrix_multiply` requires a matrix-matrix shape; GEMV is matrix-vector with N=1. Padding to a tile wastes 7/8 of the simdgroup matmul work. (new finding)
- **MLX doesn't do this**: The reference implementation (`quantized.h`) uses scalar dequant + `simd_sum`, not simdgroup matmul. The idea was likely written by analogy to steel GEMM without checking the actual quantized GEMV path. (new finding)
- **Kernel mode / dispatch restructuring**: Even if we ignored the shape mismatch, switching from `Reduction` to `SimdGroup2D` mode and re-mapping thread roles would require rewriting the kernel body and the bench harness (`run_quantized_mat_vec`). (from perf-ideas.md risk: "bigger kernel")
- **Register pressure**: A simdgroup matmul kernel with tg-memory staging would have more live state (tile buffers + matrix accumulators) than the current scalar kernel. Given idea #6's register explosion (9r→162r), this needs careful scrutiny. (from established patterns)

## Related (but different) feasible idea

**Process multiple rows per threadgroup** (dispatch-level change). MLX's `qmv_fast_impl` computes `num_simdgroups × results_per_simdgroup = 8` rows per threadgroup (`tid.y` maps to row groups, `simd_gid` maps to 4 rows each). MetalTile's current dispatch is one threadgroup per row. Packing rows would improve occupancy, especially for small `M`. This is a dispatch restructuring idea (⚠️ feasible, multi-day effort), but it is **not** what idea #22 hypothesized.

## Final Verdict
**Blocked / ill-formed against the operation.**

The hypothesis proposes applying a GEMM primitive (`simdgroup_matrix_multiply`) to a GEMV kernel. The shape mismatch (N=1) makes the optimization architecturally unsuitable — it would require padding the vector to a matrix tile and discarding most of the computed results. MLX's own quantized GEMV implementation confirms this: it uses scalar dequantization + `simd_sum`, not simdgroup matrix ops.

If the intent was to improve quantized GEMV throughput, the correct adjacent idea is **multiple rows per threadgroup** (dispatch restructuring), which matches MLX's actual design. That should be filed as a separate idea if desired.

## Related Ideas
- **021** — `fp_quantized.rs` (blocked for different reasons).
- **023** — `mlx/quantized.rs` int4 pack-of-2 lookup (LUT-based dequantization, closer to what MLX actually does).
- **010** — GEMV tpg sweep (demonstrates that dispatch parameters matter for GEMV performance).
