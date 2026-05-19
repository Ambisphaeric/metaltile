# Perf Idea 052 — Multi-launch occupancy headroom: persistent threadgroups

## Metadata
- **Number**: 052
- **Name**: persistent-threadgroups
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> For ops dispatched in tight sequence, persistent threadgroups that pull work from a queue beat re-dispatching every step.

## Target
- **Primary file(s)**: `crates/metaltile-runtime/src/context.rs`
- **Bench filter**: would need a microbench of a chain of small ops
- **Shapes / dtypes to watch**: chain of small ops (e.g., elementwise → reduce → elementwise)

## Assessment

### What persistent threadgroups mean
In CUDA, "persistent kernels" are threads that stay resident on the GPU and poll a work queue for new tasks, rather than terminating and being re-dispatched. This eliminates:
1. Dispatch launch overhead.
2. Threadgroup setup/teardown cost.
3. Context switching between kernels.

### Metal's capabilities
Metal does **not** have a native persistent threadgroup or work-queue API. The closest equivalents are:

1. **Multiple dispatches in one command buffer** (`dispatch_chain` in `context.rs`): Already implemented. This eliminates per-dispatch command buffer overhead but still creates/destroys threadgroups for each kernel.

2. **Multiple kernels in one threadgroup**: Not supported. A threadgroup runs one kernel function. To run a different kernel, you must dispatch a new threadgroup.

3. **Polling with atomic counters**: You could write a single mega-kernel that contains multiple sub-kernels and uses an atomic counter in device memory to decide which sub-kernel to run. This is technically possible but:
   - Requires fusing all ops into one monolithic kernel (op fusion at the graph level).
   - Threadgroups that finish their work early would idle-poll, wasting occupancy.
   - Divergent execution (different lanes doing different ops) kills SIMD efficiency.

### What already exists
`context.rs` already has `dispatch_chain`, which dispatches multiple kernels through a single command buffer with barriers between them. This achieves most of the practical benefit (reduced driver overhead) without requiring persistent threadgroups.

### Why this is blocked
Metal has no API for persistent threadgroups. The concept requires either:
- A monolithic fused kernel (moonshot M3/M4).
- Or an Apple-private API that doesn't exist in public Metal.

- **Outcome**: blocked — Metal has no persistent threadgroup API
- **Why**: Metal threadgroups are created and destroyed per dispatch. There is no work-queue or persistent-kernel API. The closest practical equivalent (`dispatch_chain`) is already implemented.
- **Re-scope**: If op fusion reaches the point of generating monolithic fused kernels, persistent execution becomes moot — the fused kernel *is* the persistent execution.

## Risk Register
- (not applicable — blocked by missing Metal API)

## Notes for Next Person
- `dispatch_chain` in `context.rs` already achieves most of the practical benefit for multi-kernel sequences. Focus optimization efforts there, not on persistent threadgroups.
- If Apple ever adds a persistent-threadgroup API, this idea should be revisited.
