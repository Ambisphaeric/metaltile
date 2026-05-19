# Perf Idea 047 — PSO disk cache

## Metadata
- **Number**: 047
- **Name**: pso-disk-cache
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (needs re-scoping)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Cold-start compile time dominates first-run latency; serialize compiled PSOs to `~/.cache/metaltile/pso/`.

## Target
- **Primary file(s)**: `crates/metaltile-runtime/src/context.rs`
- **Bench filter**: `time tile bench` cold vs warm
- **Shapes / dtypes to watch**: any — compile time is kernel-agnostic

## Assessment

### Current state
`context.rs` already has an **in-memory PSO cache**:
```rust
static PSO_CACHE: OnceLock<Mutex<FxHashMap<u64, Retained<Pso>>>> = OnceLock::new();
```

The cache key is an FNV-1a hash of `(kernel.name + ":" + msl_source + fn_consts)`. The cache is looked up before compiling, and new PSOs are inserted after compilation.

**But the cache is in-memory only.** Process restart means recompile.

### Can PSOs be serialized to disk?
`MTLComputePipelineState` is an opaque Metal driver object. It **cannot be directly serialized**. However, there are two possible approaches:

1. **Cache the `.metallib` binary**: When `dev.newLibraryWithSource_options_error()` compiles MSL, it produces an `MTLLibrary`. Metal libraries can be serialized to `.metallib` files (via `MTLLibrary` → `NSData` → file). On restart, load the `.metallib` via `newLibraryWithData` instead of recompiling from source.

2. **Cache the MSL source + hash**: This is trivial but doesn't save compile time — it just skips MSL generation. The expensive part is `newLibraryWithSource` (driver compilation), not `MslGenerator::generate()` (our codegen, ~tens of µs).

### Approach 1 analysis
- After `dev.newLibraryWithSource_options_error(&src, None)`, we have an `MTLLibrary`.
- `MTLLibrary` on macOS has no public `data` accessor in the Metal API. You can only create libraries from source or from `.metallib` data (via `newLibraryWithData`).
- `.metallib` files are produced by the Metal offline compiler (`metal` / `metallib` tools), not by runtime compilation.
- The offline compiler takes `.metal` source files and produces `.air` (Apple IR) → `.metallib`. This is a build-time step.
- **Runtime `newLibraryWithSource` does not produce a `.metallib`.** It compiles directly to a driver-internal representation.

### Conclusion
Direct PSO serialization is **not supported by Metal**. The practical options are:
- **Option A**: Cache the MSL source string (trivial, already fast to regenerate).
- **Option B**: Add a build-time step that runs `metal -c foo.metal` → `metallib` for common kernels. This is what idea #055 describes.
- **Option C**: Use `MTLDynamicLibrary` (Metal 3.1+) which supports serialization, but it's a different API path.

## Verdict

- **Outcome**: feasible (needs re-scoping) — the goal is valid but the mechanism needs adjustment
- **Why**: Metal's `MTLComputePipelineState` cannot be serialized. The real savings come from caching `.metallib` binaries (build-time compilation, idea #055) or from `MTLDynamicLibrary` serialization (Metal 3.1+). The in-memory PSO cache already handles warm-start within a process.
- **Measure**: Profile `time tile bench` on a cold process vs warm process to quantify the compile overhead.

## Risk Register
- `MTLDynamicLibrary` requires Metal 3.1+ and may not be available on all target devices.
- Build-time `.metallib` generation (idea #055) is the more robust path.

## Notes for Next Person
- Before building disk caching, measure the actual cold-start compile time. MSL generation is ~tens of µs; driver compilation (`newLibraryWithSource`) is the expensive part and may be 1–10 ms.
- If driver compilation is the bottleneck, investigate `MTLDynamicLibrary` or switch to build-time `.metallib` (idea #055).
