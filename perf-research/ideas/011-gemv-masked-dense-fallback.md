# 011 — GEMV-masked: dense fallback above 50% density

## Metadata
- **Number**: 011
- **Name**: gemv-masked-dense-fallback
- **Source**: `perf-ideas.md` § Quick-wins — item 11
- **Status**: 🔴 **blocked / dispatcher-level** — not a kernel tweak
- **Worktree**: —
- **Assignee**: —

## Hypothesis
> Mask evaluation cost dominates when most rows are unmasked. Detect density at launch and route to dense `gemv`.

## Target
- **Primary file(s)**:
  - `crates/metaltile-std/src/mlx/gemv_masked.rs` (current masked kernel)
  - `crates/metaltile-std/src/mlx/gemv.rs` (dense kernel to route to)
  - Dispatcher logic in runtime or bench harness
- **Bench filter**: `tile bench -f gemv_masked`
- **Shapes / dtypes to watch**: `B=4096 N=4096` with synthetic dense-mask inputs

## Current Code Reality Check

### Masked kernel today
```rust
#[kernel]
pub fn mt_gemv_masked<T>(
    mat: Tensor<T>, vec: Tensor<T>, mask: Tensor<T>, out: Tensor<T>, #[constexpr] k: u32,
) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let mut acc = 0.0f32;
    for _i in range(rs + tid, re, lsize) {
        let col = _i - rs;
        let m_val = load(mask[col]).cast::<f32>();
        acc = acc + load(mat[_i]).cast::<f32>() * load(vec[col]).cast::<f32>() * m_val;
    }
    let result = reduce_sum(acc);
    store(out[row], result.cast::<T>());
}
```

The kernel is **already dense-aware** in the sense that `m_val` multiplies every element. When the mask is all-ones, every thread does a full dot product. The "cost" is one extra `load(mask[col])` and one extra `* m_val` per iteration.

### Dense kernel for comparison
```rust
// gemv.rs
let acc = strided_reduce_dot(mat, vec, rs, rs, re);
let result = reduce_sum(acc);
```

The dense kernel uses `strided_reduce_dot` which codegen lowers to a **4-wide unrolled loop** with tail handling. The masked kernel uses a **1-wide scalar loop** (no unroll).

### Why the masked kernel is slower even at 100% density
Even if mask is all-ones, the masked kernel:
1. Loads `mask[col]` every iteration (extra memory traffic)
2. Uses 1-wide scalar loop instead of 4-wide unroll (lower ILP)
3. Has no `strided_reduce_dot` primitive (manual loop)

### What "dense fallback" actually requires
This is **not** a kernel tweak — it's a **dispatcher heuristic**:
1. Inspect the mask tensor before dispatch
2. Compute density (sum of mask / count)
3. If density > threshold (e.g., 50%), dispatch the dense `gemv` kernel instead
4. If density < threshold, dispatch the masked kernel

The bench harness doesn't support runtime kernel selection based on tensor content. The `#[bench_kernel]` macro generates static `BenchSpec` entries registered at compile time.

### Effort estimate
- Add density detection to bench harness or runtime: **medium**
- Wire up dual-kernel dispatch: **medium**
- Bench with synthetic dense/sparse masks: **low**
- **Overall**: **One-day**, not Quick-win.

## Baseline
```bash
tile bench -vv -f gemv_masked
tile snap -o results/011-baseline.json
```
Current bench only tests with `BufInit::AltZeroOne` (alternating 0/1 mask). No dense-mask shape exists.

## Risk Register
- **Classification overhead**: Must be cheaper than savings. For B=4096, a single `all_reduce` on the mask to compute density might take ~10-20 μs. If the dense kernel saves 20 μs, it's a wash.
- **Pessimization for mixed density**: A mask with 60% density might still have clustered zeros. The dense kernel would waste compute on zero-masked elements.
- **Kernel signature mismatch**: `gemv` takes `(mat, vec, out, k)`; `gemv_masked` takes `(mat, vec, mask, out, k)`. A dispatcher would need to handle different buffer bindings.
- **Bench harness limitation**: `#[bench_kernel]` generates static specs. Runtime kernel selection would need a new dispatch mechanism.

## Decision Needed
| Option | Effort | Notes |
|--------|--------|-------|
| A. Close as blocked, pick genuine Quick-win | 0 | Ideas 10, 7 are actual param sweeps. |
| B. Re-scope to One-day "runtime kernel selection" | Medium | Would benefit multiple kernels (softmax small-N, this, etc.). |
| C. Optimize masked kernel instead of routing to dense | Low-Medium | Make masked kernel use 4-wide unroll + `strided_reduce_dot` when possible. |

## Final Verdict (preliminary)
**Blocked / dispatcher-level.** The idea is valid but requires runtime tensor inspection + kernel selection, neither of which the current bench harness supports. A lower-friction path is optimizing the masked kernel body to match the dense kernel's loop structure (4-wide unroll).

## Related Ideas
- **010** — GEMV tune tpg (actual param sweep, already done)
- **018** — KV-cache vectorized copy (same masked/unmasked pattern)
- **M3** — Persistent-kernel graph capture (would make runtime kernel selection obsolete)
