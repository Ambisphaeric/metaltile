# Perf Idea 055 — Build: precompile `.metallib` per Apple GPU family

## Metadata
- **Number**: 055
- **Name**: precompile-metallib-gpu-family
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Today the runtime compiles MSL on first dispatch. Pre-compiled per-family `.metallib` (Apple7, Apple8, Apple9) eliminates first-dispatch latency.

## Target
- **Primary file(s)**: `crates/metaltile-std/build.rs`
- **Bench filter**: `time tile bench` cold-cache
- **Shapes / dtypes to watch**: any — first-dispatch latency is kernel-agnostic

## Assessment

### Current build.rs
`build.rs` does **not compile Metal code**. It:
1. Fetches MLX kernels via git sparse-checkout.
2. Copies `.metal` files from `.cache/mlx/mlx/backend/metal/kernels/` to `OUT_DIR/metal/`.
3. Resolves `#include "..."` directives recursively.

The actual compilation happens at **runtime** in `context.rs` via `newLibraryWithSource`.

### Why pre-compilation is incompatible with MetalTile's model
MetalTile generates **MSL at runtime** from the DSL `#[kernel]` macros. The MSL source is not known at build time because:
1. **Shape-dependent dispatch**: Kernels use `#[constexpr]` params that change the generated MSL (e.g., loop bounds, unroll factors).
2. **Dtype specialization**: The `#[kernel]` macro generates different MSL for `f32` vs `f16` vs `bf16`.
3. **Function constants**: Some kernels use `[[function_constant(N)]]` for runtime specialization.

The combinatorial space is: `kernels × dtypes × shapes × fn_consts × chip_families`. Pre-compiling all variants at build time is **infeasible**.

### What `.metallib` actually is
`.metallib` is a **compiled Metal library binary** produced by:
1. `xcrun -sdk macosx metal -c foo.metal -o foo.air` (compile to Apple IR).
2. `xcrun -sdk macosx metallib foo.air -o foo.metallib` (link to library).

This requires the **MSL source to be known at build time** and the **target GPU family to be known** (via `-arch` flags).

### Partial alternatives
1. **Pre-compile MLX reference kernels only**: The MLX kernels (in `.cache/mlx/`) are static `.metal` files. These *could* be pre-compiled to `.metallib` per family. But MetalTile's runtime already handles MLX kernels by compiling them on first use (same path as DSL kernels).
2. **Cache `.metallib` at runtime**: After the first `newLibraryWithSource`, save the resulting binary data to disk. On restart, load via `newLibraryWithData`. However, `MTLLibrary` does not expose its binary data on macOS.
3. **Use `MTLDynamicLibrary`**: Metal 3.1+ supports dynamic libraries that can be serialized. This is a different API path from `newLibraryWithSource`.

## Verdict

- **Outcome**: blocked — MetalTile's JIT MSL generation model is incompatible with build-time `.metallib` compilation
- **Why**: The MSL source is generated at runtime by `MslGenerator` from the DSL IR. It depends on dtype, shape, and function constants. Pre-compiling would require enumerating all possible variants at build time, which is infeasible.
- **Re-scope**: A runtime `.metallib` cache (caching compiled libraries after first dispatch) is theoretically possible but requires `MTLDynamicLibrary` or serializing `MTLLibrary` data, neither of which is straightforward on macOS.

## Risk Register
- Build-time cost: even if it were possible, compiling per-family `.metallib` for all kernels would take minutes per build.
- Device mismatch: the build machine's GPU family may differ from the target machine, requiring multiple `.metallib` variants.

## Notes for Next Person
- If cold-start latency is a real problem, investigate `MTLDynamicLibrary` (Metal 3.1+) as a runtime caching mechanism, not build-time pre-compilation.
- The in-memory PSO cache (`PSO_CACHE` in `context.rs`) already handles warm-start within a process. The gap is only across process restarts.
- For the MLX reference kernels (static `.metal` files), build-time compilation is theoretically possible but adds significant build complexity for marginal gain.
