# Perf Idea 050 — Fast-math + disable shader-validation in release

## Metadata
- **Number**: 050
- **Name**: fast-math-shader-validation
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (small)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Shader validation is on by default in debug; ensure release path disables it. Combine with `MTLLanguageVersion::Metal3_1` + `mathMode = fast`.

## Target
- **Primary file(s)**: `crates/metaltile-runtime/src/context.rs` (`dispatch_metal`)
- **Bench filter**: `tile bench` aggregate
- **Shapes / dtypes to watch**: unary ops (exp, sin, tanh) where fast-math matters

## Assessment

### Current compile options
In `context.rs` `dispatch_metal`:
```rust
let src = NSString::from_str(msl_source);
let lib = dev
    .newLibraryWithSource_options_error(&src, None)  // <-- None = default options
    .map_err(...)?;
```

In `runner.rs` `compile()`:
```rust
let opts = objc2_metal::MTLCompileOptions::new();
let lib = self.device.newLibraryWithSource_options_error(&src, Some(&opts))?;
```

Both use **default** compile options. Neither explicitly sets:
- `mathMode` (default is `MTLMathModeDefault`, not `Fast`)
- `languageVersion` (default is whatever the OS supports)
- `preserveInvariance` / shader validation flags

### Metal compile options
`MTLCompileOptions` has:
- `mathMode`: `Default`, `Fast`, `Precise`
- `languageVersion`: `Metal2_4`, `Metal3_0`, `Metal3_1`, `Metal3_2`
- `preserveInvariance`: for debug / validation

Shader validation is **not controlled by `MTLCompileOptions`**. It is an Xcode / environment setting (`MTL_DEBUG_LAYER`, `MTL_SHADER_VALIDATION`). In release builds outside Xcode, shader validation is typically off by default.

### What the change would do
1. **Set `mathMode = .Fast`**: Allows the Metal compiler to use less-accurate but faster approximations for transcendental functions (`exp`, `sin`, `log`, etc.). This is safe for inference workloads where ULP-level accuracy doesn't matter.
2. **Set `languageVersion = .Metal3_1`**: Ensures the compiler targets Metal 3.1, unlocking newer intrinsics and optimizations.

### Expected impact
- **Code gen**: `exp` may use a polynomial approximation instead of the precise Taylor series. `sin` may use a lookup table.
- **Performance**: Modest for ALU-heavy kernels (unary, transcendental). Negligible for bandwidth-bound kernels (copy, reduce).
- **Accuracy**: Minor numerical drift. The `tol` values in bench specs already allow ~1e-3 for f16, which should absorb fast-math differences.

## Verdict

- **Outcome**: feasible — small, low-risk change
- **Why**: The compile options are created with defaults. Setting `mathMode = Fast` and `languageVersion = Metal3_1` is a two-line change. Shader validation is already off in release builds outside Xcode.
- **Measure**: `tile bench -f unary` before/after. Watch for correctness regressions (the `ok` column). Expect minor speedups on transcendental ops.

## Risk Register
- Numerical drift: already gated by `precise::` annotations in the DSL where exact math matters. But some kernels (e.g., `softmax` online algorithm) are sensitive to `exp` accuracy.
- `languageVersion = Metal3_1` may fail on older macOS versions. Need fallback to Metal3_0 if the device doesn't support 3.1.

## Notes for Next Person
- This is a one-day change. Add `opts.setMathMode(MTLMathModeFast)` and `opts.setLanguageVersion(MTLLanguageVersion::Metal3_1)` in both `context.rs` and `runner.rs`.
- Gate it behind a feature flag or check `chip_family` (Metal3_1 requires Apple7+).
- Run the full bench suite to catch any correctness regressions.
