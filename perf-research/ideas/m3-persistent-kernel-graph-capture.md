# Perf Idea M3 — Persistent-kernel graph capture

## Metadata
- **Number**: M3
- **Name**: persistent-kernel-graph-capture
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Replace the dispatch-per-op model with a "graph capture" mode: a stream of ops becomes one persistent Metal kernel that pulls work items from a producer-consumer queue. Eliminates dispatch overhead entirely for inference-loop hot paths.

## Target
- **Primary file(s)**: new runtime module + codegen path (does not exist)
- **Bench filter**: would need a microbench of an inference-loop chain
- **Shapes / dtypes to watch**: transformer decode loop (SDPA → RMSNorm → MLP → RMSNorm → SDPA)

## Assessment

### Current state: `dispatch_chain`
`context.rs` already has `dispatch_chain` (idea #049 / #052):
```rust
pub fn dispatch_chain(&self, specs: &[DispatchSpec<'_>])
    -> Result<Vec<DispatchResult>, MetalTileError>
```

This dispatches **multiple kernels through a single `MTLCommandBuffer`**:
- One `commandBuffer()`, one `commit()`, one `waitUntilCompleted()`.
- Each kernel gets its own `MTLComputeCommandEncoder` + `setComputePipelineState`.
- Barriers (`memoryBarrierWithScope(Buffers)`) between consecutive passes.
- Intermediate buffers that are outputs of pass *i* and inputs of pass *j* are allocated once in `MTLStorageModePrivate`.

`dispatch_chain` already eliminates most of the per-kernel driver overhead. What remains:
1. **PSO switching**: Each kernel has its own `MTLComputePipelineState`. `setComputePipelineState` per encoder has a small cost.
2. **Encoder setup/teardown**: Each kernel needs its own encoder ( Metal requires one encoder per PSO).
3. **Barrier overhead**: `memoryBarrierWithScope` between kernels.

### What persistent-kernel graph capture would mean
The hypothesis describes a **single monolithic kernel** that replaces the entire op chain:
```
[[kernel]] void inference_loop(
    device Queue* q,
    device void** buffers,
    ...
) {
    while (true) {
        WorkItem w = dequeue(q);
        switch (w.op_id) {
            case OP_SDPA:  run_sdpa(buffers, w.args); break;
            case OP_RMS:   run_rms(buffers, w.args); break;
            case OP_MLP:   run_mlp(buffers, w.args); break;
            case OP_EXIT:  return;
        }
    }
}
```

This is essentially a **GPU task scheduler** written in MSL. Each "op" is a function inside the mega-kernel, and the host pushes work items into a device-memory queue.

### Why this is blocked

1. **No graph IR**: MetalTile has no runtime IR that represents a graph of ops. Kernels are standalone `#[kernel]` functions. There is no mechanism to compose them into a chain at the IR level.

2. **No producer-consumer queue mechanism**: Metal has no native device-memory queue or work-distribution API for persistent kernels. Implementing one requires:
   - A device-memory ring buffer for work items.
   - Atomic enqueue/dequeue operations.
   - Idle-spin polling when the queue is empty (wastes power and occupancy).

3. **Cross-kernel fusion at MSL level**: The hypothesis says "a stream of ops becomes one persistent Metal kernel." This requires inlining the MSL bodies of `mt_sdpa`, `mt_rms_norm`, etc. into a single mega-kernel. This is **whole-program MSL fusion** — a major codegen project. It would need:
   - Renaming all variables to avoid collisions.
   - Merging buffer bindings (each kernel currently has its own `[[buffer(N)]]` layout).
   - Handling divergent control flow (each op has its own grid shape and thread mapping).

4. **Metal has no graph capture API**: Unlike CUDA's `cudaGraphCreate` / `cudaGraphLaunch`, Metal has no API to record a sequence of dispatches and replay it. The closest equivalent is simply encoding multiple dispatches in one command buffer, which `dispatch_chain` already does.

### What `dispatch_chain` already achieves
- **Single command buffer**: Eliminates per-kernel `commandBuffer()` + `commit()` overhead.
- **Private intermediate buffers**: Eliminates host↔device copies between passes.
- **Barriers**: Ensures correct memory ordering.

The remaining gap (PSO switching, encoder setup) is small relative to the kernel execution time for most transformer ops. For tiny ops (elementwise), the overhead may be significant, but those are the easiest to fuse (see M4).

### Comparison to other frameworks
- **MLX**: Does not have persistent-kernel graph capture. Uses per-op dispatch with Metal.
- **PyTorch (CUDA)**: Has `torch.cuda.CUDAGraph` for CUDA only. No Metal equivalent.
- **TensorFlow (XLA)**: Fuses ops into a single kernel at compile time, but this is op fusion (M4), not persistent execution.

## Verdict

- **Outcome**: blocked — Metal lacks the API; MetalTile lacks the graph IR
- **Why**: Metal has no graph capture API and no persistent-kernel mechanism. The closest practical equivalent (`dispatch_chain`) already eliminates most per-dispatch driver overhead. A persistent mega-kernel would require cross-kernel MSL fusion (M4) plus a device-memory work queue — both are moonshot-scale.
- **Re-scope**: If the goal is to reduce dispatch overhead for inference loops, `dispatch_chain` is the right mechanism. If the goal is to fuse the entire decode loop into one kernel, that's M4 (auto-fuse arbitrary DAGs), not M3.

## Risk Register
- Device-memory queue polling wastes GPU cycles and power.
- Divergent grid shapes (SDPA uses 1D grid, MLP uses 2D tile grid) make unified thread mapping impossible in a single kernel.
- Metal's SIMD execution model assumes all threads in a group run the same kernel. A persistent kernel with per-thread op switching would have catastrophic divergence.

## Notes for Next Person
- `dispatch_chain` in `context.rs` is the pragmatic path. It already handles SDPA → RMSNorm → MLP chains in the FFAI 2-pass decode test.
- If you need lower overhead, push for M4 (op fusion) rather than M3. Fusing SDPA+RMSNorm+MLP into one kernel eliminates dispatch boundaries by eliminating the separate kernels, not by making them persistent.
