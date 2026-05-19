# Perf Idea 032 — SwiGLU/GELU: fuse with downstream matmul write

## Metadata
- **Number**: 032
- **Name**: swiglu-gelu-fuse-matmul-epilogue
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Activation right after a GEMM is everywhere in transformers. Fuse the epilogue so output never round-trips to HBM.

## Target
- **Primary file(s)**: `mlx/unary.rs` + new fused emitter / kernel
- **Bench filter**: would need `tile bench -f gemm_silu` (does not exist)
- **Shapes / dtypes to watch**: GEMM output shapes (e.g., M=4096, N=11008 for SwiGLU in Llama)

## Assessment

### What the idea describes
A fused kernel that computes:
```
C = silu(A @ W_gate) * (A @ W_up)   # SwiGLU
# or
C = gelu(A @ W)                     # GELU epilogue
```
in a single dispatch, without writing the GEMM output to memory and reloading it for the activation.

### Current state
MetalTile has:
- `unary.rs` — standalone elementwise activations (`mt_silu`, `mt_gelu`, etc.)
- No GEMM kernel in the DSL. The `#[kernel]` macro generates elementwise and reduction kernels, not matrix multiplication.
- `Dot` ops in the IR are tile-annotated by `schedule.rs` but there is no GEMM kernel body in `metaltile-std/src/mlx/`.

### What would be needed
1. **GEMM kernel in the DSL**: A `#[kernel]` that implements tiled matrix multiplication. This does not exist.
2. **Epilogue fusion mechanism**: A way to attach an elementwise op to the GEMM output tile before it is written to global memory. This is a codegen-level feature (fusing `Dot` + `Activation` in the IR → single kernel).
3. **Bench harness**: A `gemm_silu` bench spec that allocates inputs, dispatches the fused kernel, and compares against MLX.

MLX has `steel_matmul` (its GEMM implementation) with epilogue fusion in `steel/conv/steel_matmul.metal`, but MetalTile does not have a port.

### Effort estimate
- New GEMM kernel: **multi-day**
- Epilogue fusion in codegen: **multi-day**
- Bench harness: **one-day**
- **Total**: **multi-day to project-scale**

## Verdict

- **Outcome**: blocked — prerequisite missing
- **Why**: MetalTile has no GEMM kernel in the DSL. The idea describes a fusion that requires both a GEMM implementation and an epilogue-fusion codegen pass. Neither exists.
- **Re-scope**: This is a genuine optimization for transformer inference, but it belongs in the moonshot range (M4–M6) or as a post-GEMM-implementation follow-up.

## Risk Register
- (not applicable — blocked by missing infra)

## Notes for Next Person
- If MetalTile ever adds a GEMM kernel, epilogue fusion (silu, gelu, relu, add-bias) should be the first follow-up. The pattern is well-established in cuBLAS (Epilogue APIs) and MLX (steel matmul).
- Until then, this idea is not actionable.
