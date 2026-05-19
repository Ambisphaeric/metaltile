# Perf Idea 031 — Unary chains: emit `metal::precise::sigmoid` directly

## Metadata
- **Number**: 031
- **Name**: unary-precise-intrinsics
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: ⚠️ feasible (small — one kernel change)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> `sigmoid(x)` should not be `1 / (1 + exp(-x))` — Metal has `metal::precise::sigmoid` directly.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/unary.rs`
- **Bench filter**: `tile bench -f sigmoid -vv`
- **Shapes / dtypes to watch**: f32, f16, bf16

## Assessment

### Current `mt_sigmoid` implementation
```rust
let x = load(a[idx]).cast::<f32>();
let result = 1.0f32 / (1.0f32 + exp(-x));
store(out[idx], result.cast::<T>());
```

This is a **manual expansion** of the sigmoid formula. The comment notes `tol=1e-3` because f16 compounds ULP error across `exp`, `+`, and `/`.

### Metal has a native `sigmoid` intrinsic
Metal 3.1's `<metal_math>` header provides:
- `metal::sigmoid(T x)` — fast-math variant
- `metal::precise::sigmoid(T x)` — precise variant

The DSL has a `sigmoid()` builtin (used by `mt_silu` which calls `silu(x)` — `silu` is `x * sigmoid(x)`). So the DSL **already knows** about `sigmoid` as a primitive.

### Why `mt_sigmoid` doesn't use the builtin
The manual expansion was likely written to match MLX's precision exactly. MLX's `v_Sigmoid` template may also expand manually, or may use the intrinsic. The `tol=1e-3` suggests the authors already know there's a precision gap.

### What changing it would do
Replace the manual formula with `sigmoid(x)` builtin:
```rust
store(out[idx], sigmoid(load(a[idx])));
```

Expected effects:
- **Code size**: smaller MSL (one call vs 4 ops).
- **Precision**: `metal::precise::sigmoid` is a single intrinsic, no intermediate rounding between `exp` and `reciprocal`. Likely **more accurate** than the manual expansion.
- **Performance**: Elementwise unary is bandwidth-bound. The ALU savings from 4 ops → 1 op are negligible unless the kernel is ALU-bound on small tensors (unlikely).

### Other unary ops
Most other unary kernels (`mt_exp`, `mt_log`, `mt_sqrt`, `mt_silu`, `mt_gelu`, etc.) already use DSL builtins (`exp()`, `log()`, `sqrt()`, `silu()`, `gelu()`). The codegen presumably maps these to Metal intrinsics directly. `mt_sigmoid` is the **only outlier** that manually expands.

## Verdict

- **Outcome**: feasible — small, low-risk cleanup
- **Why**: `mt_sigmoid` manually expands a formula that the DSL already has as a builtin. Using `sigmoid()` directly is cleaner, likely more accurate, and matches the pattern of every other unary kernel.
- **Measure**: `tile bench -f sigmoid` before/after. Expect no throughput change (bandwidth-bound), but lower `tol` requirement and smaller MSL.

## Risk Register
- Precision difference: `metal::precise::sigmoid` may differ from the manual `1/(1+exp(-x))` by a few ULP. The bench `tol=1e-3` already allows this, but a tighter tolerance might be achievable.
- If MLX also manually expands sigmoid, switching to the intrinsic may improve MT% (MetalTile becomes more accurate / faster than MLX).

## Notes for Next Person
- This is a one-line change in `unary.rs`.
- Also check `mt_recip` — it does `1.0f32.cast::<T>() / load(a[idx])` which is also a manual expansion. Metal has `recip()` / `rsqrt()` builtins. But `mt_rsqrt` already uses the `rsqrt()` builtin.
