# Perf Idea 044 — `if_conversion.rs`: predicate tiny ifs in inner loops

## Metadata
- **Number**: 044
- **Name**: if-conversion-predicate-tiny-ifs
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚪ no-op
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Divergent simdgroup execution from `if (mask) { ... }` costs more than always-executing both sides for short bodies.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/if_conversion.rs`, `crates/metaltile-std/src/mlx/gemv_masked.rs`
- **Bench filter**: `tile bench -f gemv_masked`
- **Shapes / dtypes to watch**: `b=4096, n=4096, tpg=256`

## Assessment

### The pass already predicates tiny ifs (Diamond shapes)

`if_conversion.rs` already implements the hypothesized transformation:
- It converts `Op::If` with short bodies into `Op::Select` chains.
- Diamond shapes (both arms) are converted if total ops ≤ 8.
- Triangle shapes (one arm empty) are rejected in Phase 1 (explicit `continue`).
- Arms containing unpredictable ops (`Barrier`, `Atomic`, `Loop`, etc.) are rejected.

Test `converts_simple_diamond` confirms: a two-arm `If` with one `Const` op per arm is replaced by a single `Select`.

### `gemv_masked` has no `Op::If` to predicate

Source inspection of `gemv_masked.rs`:
```rust
for _i in range(rs + tid, re, lsize) {
    let col = _i - rs;
    let m_val = load(mask[col]).cast::<f32>();
    acc = acc + load(mat[_i]).cast::<f32>() * load(vec[col]).cast::<f32>() * m_val;
}
```

The mask is applied via scalar multiplication (`* m_val`), not via a branch. The generated IR for `mt_gemv_masked` contains:
- `Op::Loop`
- `Op::Load` (from `mask`, `mat`, `vec`)
- `Op::BinOp` (mul, add)
- `Op::Reduce`, `Op::Store`

There is **no `Op::If`** in this kernel's IR. Therefore `if_conversion.rs` has nothing to transform, and the bench filter `gemv_masked` will show no delta.

## Verdict

- **Outcome**: no-op — pass already exists for when `Op::If` is present; target kernel has no `Op::If`
- **Why**: `gemv_masked` does not generate branches. The mask is applied unconditionally as a scalar multiply.
- **Note on Triangle shapes**: The pass currently skips `CfgShape::Triangle` (one-arm `if`). If future kernels generate single-arm branches, enabling Triangle conversion (≤5 ops) would be a small extension.

## Risk Register
- (none for this assessment)
- Original risk from perf-ideas.md: "predicating ops with side effects (loads with OOB are bad on Metal)" — the pass already rejects unpredictable ops, including `Load` from mutable params, so this is already guarded.

## Notes for Next Person
- If you want to test if-conversion, you need a kernel that actually contains a branch in the DSL. `gemv_masked` is not that kernel.
- The `if_conversion` pass is already active in the default codegen pipeline. Any kernel with small `if` expressions is already being predicated automatically.
