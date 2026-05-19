# Perf Idea 036 — `vectorize.rs`: 8-wide on f16/bf16

## Metadata
- **Number**: 036
- **Name**: vectorize-8-wide-f16-bf16
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚪ no-op
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Currently caps at vec4 (one MSL `*4` type). Metal supports `half8`/`bfloat8`. Emitting 8-wide doubles LSU throughput on bandwidth-bound kernels.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/vectorize.rs`
- **Bench filter**: re-run all kernels with `MTLT_VEC_WIDTH=8` (if such env var exists)
- **Shapes / dtypes to watch**: f16/bf16 bandwidth-bound kernels (copy, unary, rms_norm)

## Assessment

The `vectorize.rs` pass **already supports up to 8-wide vectorization**.

Key evidence:
- `MAX_VEC_LEN = 8` is hardcoded in the pass.
- The header comment explicitly states: "**Width 8**: `MAX_VEC_LEN` is 8; the emitter decomposes `float8`/`half8` into `float2x4` when the native 8-wide vector isn't available."
- BF16 support is also present: `"BF16 support: DType::BF16 params are now vectorizable (bfloat4 on Metal 3.1+)."`

The pass scans for consecutive `Op::Load` / `Op::Store` with contiguous indices (`base+0, base+1, …`) and replaces them with `Op::VectorLoad` / `Op::VectorStore` of width `run_indices.len()` (up to 8). The MSL generator then emits the appropriate vector type.

### Why the hypothesis is wrong
The perf-ideas.md entry was written before the vectorize pass reached its current state. The pass was upgraded from 4-wide to 8-wide (with `float2x4` decomposition fallback) as part of "CODEGEN_OVERHAUL §4.4" (per the comment).

### Verification
`tile inspect` on kernels like `mt_copy` or `mt_unary` should show `VectorLoad` / `VectorStore` ops in the IR when contiguous access patterns are present. The `vectorize_block` test `vectorizes_f16_loads` confirms f16 is in the `is_vectorizable` set.

## Verdict

- **Outcome**: no-op — pass already implements the hypothesized feature
- **Why**: `MAX_VEC_LEN = 8` is already in the code. BF16 is already vectorizable. The MSL emitter already handles decomposition.
- **Measure**: `tile inspect` any kernel with contiguous f16 loads to verify `VectorLoad { len: 8 }` appears.

## Risk Register
- (none — already implemented)

## Notes for Next Person
- If 8-wide vectorization is not showing up in generated MSL for a specific kernel, the issue is likely in the **structural contiguity detection** (`decompose_index`), not the vector width cap. The pass requires `BinOp(Add, base, Const(k))` with consecutive `k` values. If LICM or CSE changes the index structure, the vectorizer may miss the pattern.
- The `MTLT_VEC_WIDTH=8` env var mentioned in perf-ideas.md does not appear to exist. The pass always uses `MAX_VEC_LEN = 8`.
