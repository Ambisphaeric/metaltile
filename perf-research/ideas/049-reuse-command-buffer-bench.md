# Perf Idea 049 — Reuse command buffer across bench iterations

## Metadata
- **Number**: 049
- **Name**: reuse-command-buffer-bench
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (changes semantics)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Each iteration currently makes a fresh `MTLCommandBuffer`. Encoding N dispatches into one buffer cuts driver overhead.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/runner.rs` (`measure()`)
- **Bench filter**: tiny kernels (`copy`, `arange`) — should see GB/s rise
- **Shapes / dtypes to watch**: small element counts where dispatch overhead dominates

## Assessment

### Current `measure()` implementation
```rust
for pass in 0..(warmup + iters) {
    let cb = self.queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
    enc.setComputePipelineState(&pso.pso);
    // ... bind buffers, dispatch, endEncoding, commit, waitUntilCompleted
    if pass >= warmup {
        let gpu_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
        results.push(gpu_us);
    }
}
```

Each iteration creates a new `MTLCommandBuffer`, encodes one dispatch, commits it, and waits for completion. The `GPUEndTime - GPUStartTime` measures the GPU execution time of that single dispatch.

### What reusing a command buffer would look like
Encode all `warmup + iters` dispatches into **one** `MTLCommandBuffer`, each in its own `MTLComputeCommandEncoder`:
```rust
let cb = self.queue.commandBuffer();
for pass in 0..(warmup + iters) {
    let enc = cb.computeCommandEncoder();
    // ... setup + dispatch
    enc.endEncoding();
}
cb.commit();
cb.waitUntilCompleted();
```

Then use `MTLCounterSampleBuffer` or `MTLCommandBuffer` event timestamps to get per-dispatch timing.

### Benefits
- **Reduced driver overhead**: One `commandBuffer()` + one `commit()` instead of N.
- **Better pipelining**: The GPU can start the next dispatch while the CPU is still encoding (if the command buffer is large enough).

### Risks / semantic changes
1. **Timing model changes**: `GPUEndTime - GPUStartTime` for the whole buffer measures total time, not per-dispatch. Per-dispatch timing requires `MTLCounterSampleBuffer` or `MTLCommandBuffer` events.
2. **DVFS / clock stability**: The current serial model with `flush_slc` before each bench ensures the GPU is at peak clock. A single long command buffer might have different thermal behavior.
3. **Barrier semantics**: Without barriers between dispatches, the GPU may overlap consecutive kernels if they use different resources. This changes what is being measured.
4. **Error isolation**: If one dispatch crashes, the whole command buffer fails. Serial dispatch isolates errors per iteration.

### Effort estimate
- Switch `measure()` to encode multiple dispatches in one buffer: **low**.
- Add per-dispatch timing (counter samples or events): **medium**.
- Handle the semantic change in `bench_gbps` / stats: **medium**.
- **Total**: **one-day to multi-day**.

## Verdict

- **Outcome**: feasible — real win for tiny kernels, but changes bench semantics
- **Why**: The driver overhead of creating/committing N command buffers is measurable for tiny kernels. However, the timing model must change from "per-dispatch GPU time" to "per-dispatch timestamp within a shared buffer".
- **Measure**: Bench `copy` (tiny, bandwidth-bound) before/after. Should see GB/s rise if dispatch overhead was significant.

## Risk Register
- Per-sample timer resolution: verify that `gpu_time` deltas from counter samples still match the current serial model.
- Thermal throttling: a long command buffer with many back-to-back dispatches may hit thermal limits that serial dispatch with `waitUntilCompleted` avoids.
- Correctness: if the same input/output buffers are reused across iterations, the next iteration may see stale data unless barriers are inserted.

## Notes for Next Person
- For a first experiment, try encoding just the 10 timed iterations (not warmups) into one buffer, with a `memoryBarrierWithScope(Buffers)` between each dispatch. This preserves happens-on-GPU ordering while reducing command buffer overhead.
- `MTLCounterSampleBuffer` is the cleanest per-dispatch timing API on Metal 3.1+.
