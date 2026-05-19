# Perf Idea M2 — AMX / ANE offload for small-batch f16 GEMM

## Metadata
- **Number**: M2
- **Name**: amx-ane-offload
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> The Apple matrix coprocessor and Neural Engine sit idle during Metal kernel runs. Small-batch f16 GEMMs (≤ batch 4, ≤ 1024 dim) can be faster via AMX (CPU-side, through Accelerate's hidden APIs) or ANE (via CoreML).

## Target
- **Primary file(s)**: new runtime module + dispatch path
- **Bench filter**: micro-bench small-batch f16 GEMM vs Metal GEMM
- **Shapes / dtypes to watch**: batch ≤ 4, M/N/K ≤ 1024, f16

## Assessment

### What AMX and ANE are
- **AMX** (Apple Matrix Extensions): A CPU-side matrix coprocessor on Apple Silicon (M1+). It accelerates f16/f32 GEMM/GEMV operations via dedicated matrix multiply units. There is **no public API** for AMX — access is through private `libsystem_m.dylib` symbols or via Accelerate framework (`vDSP`, `BNNS`) which may use AMX under the hood.
- **ANE** (Apple Neural Engine): A dedicated NPU on Apple Silicon (A14+/M1+). Accessible via CoreML, `mlcompute`, or private `ane` framework. The ANE is optimized for inference workloads with fixed graph structures.

### Why this is blocked

1. **No public AMX API**: Apple has never documented the AMX instruction set or C API. Reverse-engineered bindings exist (e.g., ` simdjson`'s AMX usage, `tensorflow-macos`'s AMX path) but they are:
   - Fragile across OS versions.
   - Not guaranteed to be available on all Apple Silicon chips.
   - May violate App Store guidelines if used in shipped apps.

2. **ANE via CoreML has massive overhead**: CoreML models require:
   - `MLModel` compilation (seconds on first load).
   - Input tensor wrapping (`MLMultiArray`).
   - Output tensor unwrapping.
   For a small GEMM (batch=4, dim=1024), the CoreML overhead likely exceeds any compute savings.

3. **MetalTile is a GPU-first framework**: The entire codebase assumes Metal dispatch. Adding an AMX/ANE path would require:
   - A new runtime dispatch branch (`if small_batch_f16_gemm { dispatch_amx() } else { dispatch_metal() }`).
   - New buffer management (AMX uses CPU memory, Metal uses shared/private GPU memory).
   - New correctness validation (AMX results may differ slightly from Metal).

4. **Accelerate framework alternatives**: `vDSP_mmul` (vector DSP matrix multiply) and `BNNS` (Basic Neural Network Subroutines) are public APIs. They may use AMX under the hood on M1+. However:
   - `vDSP_mmul` is limited to f32, not f16.
   - `BNNS` is deprecated in favor of CoreML / Metal Performance Shaders.
   - Neither is clearly faster than Metal for small batches.

### What would need to happen to unblock
- Apple would need to publish a public AMX API (unlikely).
- Or someone would need to build and maintain a reverse-engineered AMX binding with robust feature detection.
- Or CoreML would need a near-zero-overhead inference API (currently not the case).

## Verdict

- **Outcome**: blocked — no public API for AMX, ANE overhead too high
- **Why**: AMX has no documented API. CoreML's overhead dominates for small batches. The existing Metal path is already competitive for the shapes in question.
- **Re-scope**: If Apple ever publishes an AMX API, this becomes a high-value optimization. Until then, it is a research curiosity, not an actionable project.

## Risk Register
- Private API fragility: reverse-engineered AMX bindings break on OS updates.
- CoreML overhead: model compilation + tensor wrapping makes small-batch GEMM slower, not faster.
- Memory model mismatch: AMX operates on CPU memory; MetalTile buffers are GPU-resident. Copying data back and forth would negate any compute win.

## Notes for Next Person
- If you want to pursue this, start with a standalone micro-bench: measure `vDSP_mmul` (f32) vs Metal GEMM for small batches. If vDSP wins for f32, investigate whether an f16 path exists via private APIs.
- The `Accelerate` framework's `vDSP_mmul` is public and safe. It won't use AMX for f16 (vDSP is f32-only), but it's a legitimate baseline for "can CPU beat GPU for small GEMM."
- Consider MLX's path: MLX does not use AMX/ANE for GEMM — it uses Metal for everything. If MLX (Apple's own framework) doesn't bother with AMX, that's strong evidence the win is small or negative.
