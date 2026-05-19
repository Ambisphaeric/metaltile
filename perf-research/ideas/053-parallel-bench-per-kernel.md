# Perf Idea 053 — CLI: parallelize per-kernel benches across non-overlapping shapes

## Metadata
- **Number**: 053
- **Name**: parallel-bench-per-kernel
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (risky)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Dev-loop friction. Doesn't change kernel perf, but speeds the loop — same benches in less wall time means more tweak cycles per hour.

## Target
- **Primary file(s)**: `crates/metaltile-cli/src/cmd/bench.rs`
- **Bench filter**: `time tile bench`
- **Shapes / dtypes to watch**: full suite

## Assessment

### Current bench flow
`bench.rs` runs serially:
1. For each `BenchSpec` in `inventory::iter::<BenchSpec>`:
2. For each `dtype` in `spec.dtypes`:
3. Call `run_spec(spec, &runner, dt)`.
4. `run_spec` compiles the kernel, dispatches warmups + samples, validates.
5. `flush_slc()` is called before each spec.

This is strictly single-threaded on the CPU side.

### What parallelization would look like
Dispatch multiple kernels concurrently using **multiple command queues** or **concurrent command buffers** on the same queue:
- Metal allows multiple `MTLCommandQueue` objects per device.
- Each queue can encode and commit independently.
- The GPU scheduler interleaves work from multiple queues.

### Risks (correctly identified in perf-ideas.md)
1. **DVFS pollution**: Running kernels back-to-back on the same queue with `flush_slc` keeps the GPU at peak clock. Running them in parallel may cause thermal throttling if the combined load exceeds the thermal budget.
2. **SLC state**: `flush_slc` evicts the System Level Cache. If two kernels run concurrently, one kernel's data may pollute the other's cache.
3. **Occupancy interference**: Two kernels competing for the same GPU cores may reduce each other's effective occupancy.
4. **Buffer aliasing**: Each concurrent kernel needs its own input/output buffers to avoid data races.

### Practical constraints
- `inventory::iter::<BenchSpec>` gives us all specs, but each spec's `run_spec` uses the same `GpuRunner` (single device, single queue).
- To parallelize, we'd need either:
  - Multiple `GpuRunner` instances (multiple queues).
  - Or a single runner that dispatches concurrent command buffers.
- Both require significant refactoring of `bench.rs` and `runner.rs`.

### Effort vs. benefit
The perf-ideas.md entry is honest: "dev-loop friction... speeds the loop." The wall-time savings would be sub-linear (e.g., 2× parallelism → maybe 1.5× speedup due to contention). For a full suite of ~50 kernels × 3 dtypes = 150 dispatches, the current wall time is likely dominated by GPU execution, not CPU overhead.

## Verdict

- **Outcome**: feasible but risky — marginal benefit for significant complexity
- **Why**: Parallel dispatch is possible with multiple queues, but DVFS and cache pollution make the results less reliable. The wall-time savings are likely small (GPU execution dominates).
- **Alternative**: Idea #051 (pipelined sample collection) gives a larger per-kernel win with no DVFS risk.

## Risk Register
- DVFS pollution: concurrent kernels may throttle each other.
- SLC cache interference: one kernel's data pollutes the cache for another.
- Correctness: buffer reuse across concurrent dispatches would cause data races.

## Notes for Next Person
- Before parallelizing, measure the current `tile bench` wall time. If it's < 30 seconds, parallelization is not worth the complexity.
- A safer alternative: run independent kernel families in parallel (e.g., "unary" on queue 1, "reduce" on queue 2) with separate `GpuRunner` instances.
