# Perf Idea M9 — CPU SIMD fallback codegen (NEON)

## Metadata
- **Number**: M9
- **Name**: cpu-neon-fallback
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: ⚠️ feasible (project-scale)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Same `#[kernel]` macro, second backend: NEON via Rust's `std::simd`. Unlocks unit-testing on CI (no Mac required), and gives CPU-only Macs (none ship now, but Linux ARM does) a coherent story.

## Target
- **Primary file(s)**: new codegen backend: `crates/metaltile-codegen/src/neon/` (does not exist)
- **Bench filter**: `cargo test` on CI (Linux ARM, x86)
- **Shapes / dtypes to watch**: all kernels — correctness validation on CPU

## Assessment

### Current codegen architecture
The `metaltile-codegen` crate has a single backend:
- `msl/` — Metal Shading Language generator (`MslGenerator`).
- `passes/` — IR passes (vectorize, fusion, unroll, CSE, LICM) that run before MSL generation.
- `msl/mod.rs` — `MslGenerator::generate(kernel: &Kernel) -> Result<String>`.

The `#[kernel]` macro generates a `Kernel` IR object. The `BenchSpec` stores `kernel_ir: fn(DType) -> Kernel`. The runtime calls `MslGenerator::generate()` to produce MSL source.

### What a NEON backend would need
1. **New generator**: `NeonGenerator` (or `CpuGenerator`) that produces Rust code using `std::simd`:
   ```rust
   pub struct NeonGenerator;
   impl NeonGenerator {
       pub fn generate(&self, kernel: &Kernel) -> Result<String> {
           // emit Rust code with std::simd types
       }
   }
   ```

2. **SIMD type mapping**:
   - `f32x4`, `f32x8` for vectorized loads.
   - `std::simd::f32x4` (stable in Rust 1.78+).
   - `f16` types may need `half` crate or `std::simd` support (still maturing).

3. **Threading model**: CPU execution needs a thread pool (e.g., `rayon`). The `GridSpec` dispatch dimensions map to `rayon` parallel iterators.

4. **Memory model**: CPU buffers are `Vec<u8>` / slices. No `MTLBuffer` wrapper needed.

5. **Feature parity**: Every MSL intrinsic used by the generator needs a NEON/`std::simd` equivalent:
   - `simd_sum` → `std::simd::Simd::reduce_sum`.
   - `simd_max` → `std::simd::Simd::reduce_max`.
   - `threadgroup_barrier` → `std::sync::Barrier` (per thread group).
   - `simd_shuffle` → lane extraction / `permute`.

### Effort estimate
- `NeonGenerator` scaffolding: **multi-day**.
- Emission for elementwise ops (load, store, binop, unary): **one-day**.
- Emission for reductions (simd_sum, simd_max, threadgroup tree reduce): **multi-day**.
- Emission for threadgroup memory (`threadgroup_alloc`, `threadgroup_load`, `threadgroup_store`): **multi-day** (maps to CPU shared memory / L1 cache simulation).
- Thread pool integration (`rayon`): **one-day**.
- **Total**: **project-scale** (weeks).

### Why it's valuable
- **CI testing**: Currently all GPU tests are `#[cfg(target_os = "macos")]` gated. A CPU backend would allow running correctness tests on CI (GitHub Actions Linux runners).
- **Linux ARM support**: Apple Silicon Macs can run Linux. A CPU backend would give MetalTile a story there.
- **Debugging**: CPU execution is easier to debug (GDB, print statements) than GPU kernels.

### Why it's a moonshot
- `std::simd` is still maturing. `f16` SIMD types are not yet stable.
- Threadgroup memory has no direct CPU equivalent. Simulating it (e.g., per-thread-group L1 cache) is complex.
- The MSL generator has ~2000+ lines of emission code across `matmul.rs`, `reduce.rs`, `emit_block.rs`, `fused.rs`, etc. Duplicating all of this for NEON is a massive undertaking.

### Pragmatic path
Instead of a full NEON backend, consider:
- A **reference CPU interpreter** for the `Kernel` IR: single-threaded, scalar-only, for correctness validation. This avoids SIMD complexity entirely.
- Port the most critical kernels (elementwise, reduce) to `ndarray` + `rayon` for CI smoke tests.

## Verdict

- **Outcome**: feasible — project-scale, but valuable for CI
- **Why**: The `Kernel` IR is backend-agnostic in principle. A second generator is architecturally possible. But `std::simd` maturity and the sheer volume of MSL emission code make this a weeks-long project, not a quick add.
- **Re-scope**: A reference scalar CPU interpreter for correctness validation is a more achievable first step.

## Risk Register
- `std::simd` f16 support is not yet stable.
- Threadgroup memory simulation on CPU is complex and may not match GPU semantics exactly.
- Performance on CPU will be poor (orders of magnitude slower than GPU). This is fine for CI but not for production use.

## Notes for Next Person
- If you want CI testing, start with a scalar reference interpreter for `Kernel` IR, not a NEON generator.
- If you want a production CPU path, use `ndarray` + `rayon` for high-level ops, not a kernel-by-kernel NEON port.
- `std::simd` is stabilizing rapidly; recheck in 6–12 months.
