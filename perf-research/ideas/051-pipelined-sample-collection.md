# Perf Idea 051 — Bench: pipelined sample collection

## Metadata
- **Number**: 051
- **Name**: pipelined-sample-collection
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (medium effort)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Today the SLC flush + warmup + 10 samples is serial per kernel. Encode all warmups + samples in one command buffer, read timestamps from `MTLCounterSampleBuffer`.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/runner.rs` (`measure()`)
- **Bench filter**: total bench wall time
- **Shapes / dtypes to watch**: all kernels — wall time of the full suite

## Assessment

### Current `measure()` flow
```rust
for pass in 0..(warmup + iters) {
    let cb = self.queue.commandBuffer();
    let enc = cb.computeCommandEncoder();
    // ... setup, dispatch, endEncoding, commit, waitUntilCompleted
    if pass >= warmup {
        results.push(gpu_us);  // GPUEndTime - GPUStartTime per buffer
    }
}
```

Serial per-pass overhead:
1. `commandBuffer()` — driver allocation
2. `computeCommandEncoder()` — encoder setup
3. `setComputePipelineState()` — state binding
4. `setBuffer_offset_atIndex()` — buffer binding
5. `dispatchThreadgroups()` — encode work
6. `endEncoding()` — finish encoder
7. `commit()` — submit to GPU
8. `waitUntilCompleted()` — CPU stalls until GPU finishes

For 15 warmups + 10 samples = 25 iterations, this overhead repeats 25×.

### What pipelining would look like
Encode all 25 dispatches into **one** command buffer:
```rust
let cb = self.queue.commandBuffer();
for pass in 0..(warmup + iters) {
    let enc = cb.computeCommandEncoder();
    // ... setup + dispatch (same as today)
    enc.endEncoding();
    // Insert barrier between dispatches to prevent overlap
    if pass + 1 < warmup + iters {
        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }
}
cb.commit();
cb.waitUntilCompleted();
```

Then extract per-dispatch timing via `MTLCounterSampleBuffer` or `MTLCommandBuffer` event timestamps.

### MTLCounterSampleBuffer
Available in Metal 3.1+, allows placing timestamp samples at specific encoder boundaries. Each sample records the GPU clock at that point. The delta between consecutive samples gives per-dispatch GPU time.

### Benefits
- **Driver overhead reduction**: One `commandBuffer` + `commit` for the whole sequence.
- **GPU pipelining**: The GPU can start dispatch N+1 while the CPU is still encoding dispatch N+2 (if the command buffer is large enough).
- **CPU efficiency**: `waitUntilCompleted` blocks once for the whole sequence, not 25×.

### Comparison to idea #049
Idea #049 is about reusing the command buffer within a single kernel's bench. Idea #051 extends this to **all kernels in the suite** — but that would require a single monolithic command buffer for the entire bench run, which is impractical because different kernels have different PSOs and buffer bindings. More realistically, idea #051 means:
- For each kernel: encode all warmups + samples into one command buffer.
- Not across different kernels (which have different PSOs and would need separate encoders anyway).

## Verdict

- **Outcome**: feasible — medium effort, changes timing infrastructure
- **Why**: The serial per-iteration model has measurable overhead for the 25 iterations per kernel. Encoding them into one command buffer reduces driver overhead. Per-dispatch timing requires `MTLCounterSampleBuffer` or event timestamps.
- **Measure**: `time tile bench` wall time before/after.

## Risk Register
- Counter sample resolution: verify that timestamp deltas still match the current serial model.
- Memory barriers between dispatches: if the same buffers are reused, a barrier is needed to prevent the GPU from overlapping dispatches (which would corrupt the timing).
- Cross-kernel pipelining is not practical — different kernels need different PSOs and encoder state. The win is per-kernel, not suite-wide.

## Notes for Next Person
- Start with a single kernel. Encode 15 warmups + 10 samples into one command buffer, with barriers between each dispatch. Read back timestamps via `MTLCounterSampleBuffer`.
- Compare the resulting `min_us` against the current serial model to verify accuracy.
- Only after per-kernel pipelining is validated should you consider any cross-kernel batching.
