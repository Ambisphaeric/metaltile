# Perf Idea M8 — Codegen → Metal 3.2 tensor descriptors

## Metadata
- **Number**: M8
- **Name**: metal-3-2-tensor-descriptors
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: ⚠️ feasible (blocked on Metal 3.2 availability)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Metal 3.2 (M4-era) exposes hardware tensor descriptors closer to NVIDIA's TMA. Once GA, the codegen layer can target it for D=128 GEMM/SDPA tiles, getting H/W async copy + autoswizzle for free.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/msl/` (new generator path)
- **Bench filter**: `tile bench` aggregate on Metal 3.2+ hardware
- **Shapes / dtypes to watch**: GEMM/SDPA with D=128 tiles

## Assessment

### Current Metal version support
The codebase targets **Metal 3.1**:
- `bfloat` type support (Metal 3.1+).
- `async_copy` prefetch (Metal 3 / M2+).
- `MTLLanguageVersion::Metal3_1` is the effective target (see idea #050).

Metal 3.2 is **M4-era** (announced ~2024, shipping with macOS 15 / iOS 18 on M4 devices). It adds:
- **Tensor descriptors**: Hardware-managed memory descriptors for matrix tiles, with automatic swizzling and async copy.
- **Ray tracing improvements**: Not relevant to compute.
- **Mesh shaders**: Not relevant.

### What tensor descriptors are
Similar to NVIDIA's **Tensor Memory Accelerator (TMA)**:
- A hardware unit that fetches matrix tiles from global memory into registers or shared memory.
- Handles **swizzling** (rearranging elements for optimal SIMD layout) automatically.
- Supports **async copy** — the hardware prefetches the next tile while the current tile is being computed.
- Reduces instruction count and improves memory bandwidth utilization for GEMM/SDPA.

### What the codegen would need
1. **Feature detection**: At runtime, detect if the device supports Metal 3.2 tensor descriptors.
2. **New MSL emission path**: When targeting Metal 3.2 + tensor descriptors:
   - Declare tensor descriptors for Q, K, V, output tiles.
   - Use `load_tensor_descriptor` / `store_tensor_descriptor` instead of manual `threadgroup_load` / `threadgroup_store`.
   - Emit `async_copy` with tensor descriptor source/destination.
3. **Tile layout changes**: Tensor descriptors may require specific tile dimensions (e.g., 16×16×8). The `ScheduleConfig` tile dims would need to align.

### Current state of async copy
`MslGenerator` already has `async_copy` support:
```rust
// crates/metaltile-codegen/src/msl/config.rs
pub async_copy: bool,
```

But this is **software async copy** (prefetch into threadgroup memory), not hardware tensor descriptor async copy.

### Blocker: Metal 3.2 is not universally available
- Metal 3.2 requires macOS 15+ and M4 hardware.
- M1/M2/M3 devices do not support tensor descriptors.
- The codegen would need dual paths: legacy (current) and tensor-descriptor (new).

## Verdict

- **Outcome**: feasible — blocked on Metal 3.2 hardware / OS availability
- **Why**: The codegen infrastructure (`MslGenerator`, `ScheduleConfig`, `async_copy`) is ready to be extended. Tensor descriptors would be a new emission path, not a rewrite. But it requires Metal 3.2 hardware for validation, and a dual-path fallback for older devices.
- **Note**: This is a forward-looking optimization. The benefit is real (TMA-like async copy + swizzle), but the timeline depends on Metal 3.2 adoption.

## Risk Register
- Metal 3.2 may change before GA. Apple has been known to modify APIs between beta and release.
- Dual-path maintenance: the legacy path and tensor-descriptor path must both be maintained and tested.
- Tile dimension constraints: tensor descriptors may enforce specific tile sizes that don't match current heuristics.

## Notes for Next Person
- Monitor `MTLGPUFamily::Apple9` and `Apple10` for tensor descriptor support announcements.
- When Metal 3.2 is available, start with a minimal test: emit a tensor descriptor for a 16×16×8 GEMM tile and measure vs manual loads.
- The `MslGenerator` architecture (separate files for matmul, reduce, emit_block) should make adding a new emission path manageable.
