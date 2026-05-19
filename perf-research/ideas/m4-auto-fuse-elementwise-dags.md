# Perf Idea M4 — Auto-fuse arbitrary elementwise DAGs at runtime

## Metadata
- **Number**: M4
- **Name**: auto-fuse-elementwise-dags
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Build a runtime IR that captures `softmax(qk).matmul(v).rms_norm(g)` and JIT-compiles the whole chain. Same compiler infrastructure already exists for the `#[kernel]` macro — generalize it to runtime-constructed graphs.

## Target
- **Primary file(s)**: new runtime graph IR + codegen generalization (does not exist)
- **Bench filter**: would need a fused-chain bench harness
- **Shapes / dtypes to watch**: transformer decode loop chain (SDPA → RMSNorm → MLP → RMSNorm)

## Assessment

### Current fusion capabilities
`fusion.rs` already fuses **elementwise chains within a single block** into `Op::FusedElementwise`:
- `BinOp` → `UnaryOp` → `Activation` → `Cast` chains are merged.
- Multi-use values break the chain (correctly).
- The MSL emitter writes a single MSL expression for the fused chain.

But `fusion.rs` is **block-local and kernel-local**. It cannot fuse across:
- Separate `#[kernel]` functions.
- Reduction ops (`Reduce`, `StrideReduce`, `Scan`).
- Memory barriers or threadgroup operations.
- Kernels with different dispatch grids.

### Why runtime graph fusion is blocked

1. **No runtime graph IR**: MetalTile has no IR that represents a graph of ops. Each `#[kernel]` generates a standalone `Kernel` IR object. There is no `Graph` or `DAG` type that connects kernels.

2. **No cross-kernel codegen**: The `MslGenerator` generates MSL for a single `Kernel`. To fuse `softmax(qk).matmul(v).rms_norm(g)` into one kernel, the codegen would need to:
   - Inline the body of `mt_softmax`, `mt_matmul`, and `mt_rms_norm` into a single MSL function.
   - Rename all variables to avoid collisions.
   - Merge buffer binding tables (each kernel has its own `[[buffer(N)]]` layout).
   - Reconcile dispatch grids (SDPA uses 1D per-head, matmul uses 2D tile, RMSNorm uses 1D per-row).
   - Insert barriers between reduction stages and elementwise stages.

   This is **whole-program MSL fusion** — a major compiler project.

3. **Dispatch grid mismatch**: A fused kernel must have a single dispatch grid. But:
   - `sdpa_decode` dispatches `[n_q_heads, 1, 1]` with tpg=1024.
   - `rms_norm` dispatches `[n_rows, 1, 1]` with tpg=256.
   - `gemm` (if it existed) would dispatch a 2D tile grid.
   
   Reconciling these into one grid requires tiling the entire computation — essentially reimplementing XLA or TVM.

4. **Reduction + elementwise ordering**: Fusing `softmax` (which has a `reduce_max` + `reduce_sum` + barrier) with downstream `matmul` requires the barrier to complete before the matmul starts. A single kernel can do this (FlashAttention does), but it must be hand-written, not auto-generated from separate kernels.

### What the hypothesis gets right
The "same compiler infrastructure" comment is partially correct:
- The `MslGenerator`, `Pass` pipeline, and `Kernel` IR all exist.
- What does not exist is a **runtime graph constructor** that builds a `Kernel` from a DAG of ops, and a **cross-op scheduler** that maps the DAG to tiles and barriers.

### `dispatch_chain` is the pragmatic limit
`context.rs` `dispatch_chain` already chains kernels through a single command buffer with private intermediate buffers. It eliminates:
- Per-kernel command buffer overhead.
- Host↔device memory copies between ops.

What remains (PSO switching, encoder setup, barriers) is small compared to kernel execution time for most transformer ops. The hypothesis overstates the remaining overhead.

## Verdict

- **Outcome**: blocked — no runtime graph IR, no cross-kernel codegen
- **Why**: MetalTile has no mechanism to construct a DAG of ops at runtime and JIT-compile it into a single kernel. The `fusion.rs` pass is block-local only. Cross-kernel fusion would require a new graph IR, a scheduler, and a whole-program MSL generator — essentially reimplementing XLA/TVM.
- **Re-scope**: A more achievable path is hand-writing fused kernels for specific chains (e.g., `softmax + matmul(v)` as FlashAttention, or `rms_norm + gate_proj + up_proj` as fused MLP). These are new kernels, not an auto-fusion system.

## Risk Register
- Dispatch grid reconciliation is the hardest problem — different ops have different natural grids.
- Barrier placement in auto-generated fused kernels is error-prone.
- Register pressure in a fused kernel is the sum of all constituent ops, easily exceeding the register file.

## Notes for Next Person
- `dispatch_chain` is the pragmatic ceiling for runtime op chaining. If you need lower overhead, write hand-fused kernels.
- If you want auto-fusion, study XLA's fusion heuristics or TVM's `te.compute` + `te.schedule` model. MetalTile's IR is not designed for this.
