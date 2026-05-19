# Perf Idea 048 — Heap-backed buffer pool

## Metadata
- **Number**: 048
- **Name**: heap-backed-buffer-pool
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚪ no-op
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> `newBufferWithBytes` allocations have non-zero cost; pre-allocate a heap, slice from it.

## Target
- **Primary file(s)**: `crates/metaltile-runtime/src/buffer.rs`, `crates/metaltile-runtime/src/context.rs`
- **Bench filter**: micro-bench a no-op kernel
- **Shapes / dtypes to watch**: tiny kernels where dispatch overhead dominates

## Assessment

### Current state
`context.rs` already implements a **thread-local buffer pool**:
```rust
std::thread_local! {
    static BUF_POOL: RefCell<FxHashMap<PoolKey, Vec<BufRc>>>
        = RefCell::new(FxHashMap::default());
}

fn pool_acquire(dev, len, opts) -> BufRc {
    let bucket = len.max(4).next_power_of_two();
    let key = (bucket, opts.0 as u64);
    // Return a buffer whose strong_count == 1 (only pool owns it).
    // Otherwise allocate new.
}
```

The pool:
- Buckets by `(next_power_of_two(len), storage_mode)`.
- Reuses buffers whose `Rc::strong_count == 1` (not in use elsewhere).
- Is already used by `upload_resident` and `dispatch_chain`.

### What the hypothesis describes
A `MTLHeap`-based allocator:
- Pre-allocate a large `MTLHeap` (e.g., 1 GB).
- Suballocate buffers from the heap via `MTLHeap::newBufferWithLength`.
- Return sub-allocations to the heap when done.

### Why the current pool is sufficient
The existing `BUF_POOL` already eliminates the `newBufferWithLength` allocation cost for repeated buffer sizes. It works at the `MTLBuffer` granularity, not the heap sub-allocation granularity. For MetalTile's use case (bench harness with repeated buffer sizes), the pool is functionally equivalent to a heap allocator:
- Both avoid `newBufferWithLength` on the hot path.
- Both reuse memory.
- The pool is simpler (no heap size management, no fragmentation tracking).

### When a heap would matter
A heap would be better if:
- Buffer sizes are highly variable (fragmentation in the pool).
- You need `MTLStorageModePrivate` buffers that alias across dispatches (the pool already handles this).
- You need memory-budgeting / GPU memory limits.

None of these are current pain points.

## Verdict

- **Outcome**: no-op — buffer pool already exists
- **Why**: `BUF_POOL` in `context.rs` already caches and reuses `MTLBuffer` objects by size bucket. The hypothesized optimization is already implemented, just using `MTLBuffer` pooling instead of `MTLHeap` sub-allocation.
- **Measure**: A micro-bench of a no-op kernel would show that buffer allocation is not on the critical path (the pool handles it).

## Risk Register
- (none — already implemented)

## Notes for Next Person
- If you ever see `newBufferWithLength` in a profile, the first thing to check is whether `pool_acquire` is being used. In `context.rs`, both `dispatch_metal` and `dispatch_chain` use `pool_acquire` for all buffer allocations.
- The `buffer.rs` file (`GpuBuffer` / `HostData`) is just metadata — it doesn't do Metal allocation. The real allocation happens in `context.rs`.
