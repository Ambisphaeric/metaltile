# 021 — FP4 dequant: packed bit ops

## Metadata
- **Number**: 021
- **Name**: fp4-dequant-packed-bit-ops
- **Source**: `perf-ideas.md` § Op-level structural changes — item 21
- **Status**: 🔴 blocked
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> unpack 8 fp4 → 8 half in one 32-bit shuffle with a precomputed LUT in const memory.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/fp_quantized.rs`
- **Bench filter**: `tile bench -vv -f fp_quantized`
- **Shapes / dtypes**: `N=1M f32 gs=32`

## Current Code Reality Check

The target file `fp_quantized.rs` contains a single kernel, `mt_fp4_quant_dequant`, which implements a **scalar quantize-dequantize roundtrip** on `Tensor<f32>`:

1. Each thread loads one `f32` element (`program_id::<0>()` → `load(inp[gid])`).
2. It computes a per-simdgroup max (`simd_max(ax)`) to derive an inverse scale.
3. It quantizes `norm` to one of 8 FP4 levels via a cascade of **7 nested `select` statements**.
4. It immediately dequantizes back to `f32` (`sign * q * (group_max / 6.0)`).
5. It stores one `f32` result.

There is **no packed FP4 representation** in this kernel:
- Input is `Tensor<f32>`, not a packed `uint32`/`uint8` buffer of 4-bit values.
- Output is `Tensor<f32>`, not `half` or a dequantized vector.
- There is no bit unpacking, no 32-bit shuffle, no LUT load, and no `half` type usage.

The benchmark harness (`run_spec.rs::run_fp_quantized`) confirms this: it generates `Vec<f32>` input, runs the kernel, and compares against a CPU reference that also operates on `f32` scalars grouped by 32 (group-size=32 scale sharing).

### What the hypothesis actually describes
The hypothesis assumes a **weight-dequantization** kernel where 8 FP4 weights are packed into a single 32-bit word, and the kernel unpacks them in bulk using bitwise operations and a 256-entry (or similar) LUT in constant memory. This is a common pattern in quantized GEMM/GEMV kernels (e.g., loading a `uint32` of weights and expanding to 8 `half` values for matmul).

That operation **does not exist** in `fp_quantized.rs`. The only FP4-related code is the roundtrip quantization benchmark.

## Baseline
Not benched — analytical assessment only. The kernel body is scalar and the hypothesized packed-bit optimization is inapplicable to the current code.

## Risk Register
- **Target mismatch**: The idea describes packed-bit dequantization, but the target file is a scalar quantize-dequantize roundtrip. (from perf-ideas.md)
- **LUT in `constant` address space**: Even if the right kernel existed, a 256-entry `half2` LUT is ~1 KB, well under the 64 KB Metal constant-segment limit. But there is no constant-array primitive in the current DSL. (from perf-ideas.md)
- **DSL lacks packed-bit / vector primitives**: No `uint32` packed load, no `simd_shuffle`, no `half8` type. These are recurring blockers across ideas 5, 8, 18, etc.
- **Premature optimization on wrong kernel**: Optimizing the existing scalar roundtrip with packed ops would require changing the benchmark contract (input/output dtypes and shapes), which is out of scope for a Quick-win.

## Final Verdict
**Blocked / ill-formed against current code.**

The target file does not contain the operation this idea wants to optimize. `mt_fp4_quant_dequant` is a scalar `f32` quantize-dequantize benchmark, not a packed FP4 weight-dequantization kernel. Implementing the hypothesized optimization would require:

1. A new kernel that accepts packed `uint32`/`uint8` FP4 weights and outputs `half`/`f16` vectors, **or**
2. A major rewrite of the existing benchmark to operate on packed data, changing the I/O contract.

Either path is a **new-kernel / new-bench-harness** effort, not a single-file kernel tweak.

## Related Ideas
- **022** — `ffai/dequant_gemv.rs` (actual quantized GEMV; may contain packed dequant logic).
- **023** — `mlx/quantized.rs` (int4 pack-of-2 lookup; closer to the LUT pattern described here).
- **005** — SDPA-vector vec8 loads (same DSL vector-primitive blocker).
- **018** — KV-cache vectorized copy (same DSL vector-primitive blocker).
