# Perf Idea M6 — KV-cache via Metal heaps + virtual remap

## Metadata
- **Number**: M6
- **Name**: kv-cache-metal-heaps
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: ⚠️ feasible (marginal benefit)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Append to KV cache currently means copy. With Metal heaps and `MTLBufferAccessUsage::TIER2`, you can carve a fresh slice off a pre-allocated heap each step and treat it as the new tail — zero copy, zero allocation.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/kv_cache.rs`, `crates/metaltile-runtime/src/context.rs`
- **Bench filter**: micro-bench KV cache append at long context
- **Shapes / dtypes to watch**: n_kv_heads=8, max_seq=32768, head_dim=128, f16

## Assessment

### Current `kv_cache_update` kernel
```rust
#[kernel]
pub fn kv_cache_update<T>(
    src: Tensor<T>,      // [n_kv_heads, head_dim] — one token
    out: Tensor<T>,      // [n_kv_heads, max_seq, head_dim]
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] position: u32,
) {
    let idx = program_id::<0>();
    let h = idx / head_dim;
    let d = idx - h * head_dim;
    let dst_idx = h * max_seq * head_dim + position * head_dim + d;
    store(out[dst_idx], load(src[idx]));
}
```

This kernel **writes directly into the pre-allocated cache** at `position`. There is **no copy** in the kernel — `src` is a one-token slice, and `out` is the full cache. Each thread copies one element.

### What the hypothesis describes
The idea seems to conflate two things:

1. **KV cache append at the framework level**: Some frameworks (e.g., naive PyTorch) allocate a new tensor `[n_kv_heads, new_seq, head_dim]` and `cat()` the old cache with the new token. This IS a copy. But MetalTile's `kv_cache_update` already writes in-place to a pre-allocated buffer.

2. **Metal heap sub-allocation**: `MTLHeap` allows creating a large pool and suballocating `MTLBuffer` slices from it. The hypothesis says "carve a fresh slice off a pre-allocated heap each step and treat it as the new tail."

   But the KV cache in MetalTile is already a single `MTLBuffer` of size `[n_kv_heads * max_seq * head_dim * elem_size]`. The "new tail" is just `position * head_dim` elements past the previous tail. There is no allocation on the append path — just a kernel dispatch.

### Metal heaps would help if...
- The KV cache buffer is currently allocated per-step via `newBufferWithLength`. But `context.rs` already uses `BUF_POOL` (see idea #048), so allocations are pooled.
- The framework does `newBufferWithLength` for each layer's KV cache. A heap could pre-allocate one large buffer for all layers and carve slices. But the pool already handles this efficiently.

### `MTLBufferAccessUsage::TIER2`
Metal 3.1+ adds `MTLBufferAccessUsage::TIER2` which allows:
- More efficient buffer aliasing.
- Better memory compression.
- Support for `MTLHeap` with sparse backing.

But `TIER2` is a **capability flag**, not a magic performance boost. It enables features that require explicit use (sparse textures, heap aliasing). Simply setting it on a buffer does not improve performance.

### What the real optimization might be
If the hypothesis is about **avoiding framework-level copies** (not kernel-level copies), the fix is at the dispatch level:
- Ensure the KV cache buffer is allocated once at model load time.
- Ensure `kv_cache_update` writes in-place (already true).
- Use `ResidentBuffer` / `upload_resident` to keep the cache GPU-resident across steps (already supported by `context.rs`).

All of these are already implemented. The hypothesis may be describing a problem that doesn't exist in MetalTile's current architecture.

## Verdict

- **Outcome**: feasible but marginal — the "copy" the hypothesis describes does not exist in the current kernel
- **Why**: `kv_cache_update` already writes directly into a pre-allocated cache buffer. There is no copy on the append path. Metal heaps would not improve the current architecture because `BUF_POOL` already handles buffer reuse.
- **Re-scope**: If the framework ever does a `cat()`-style reallocation per step, then Metal heaps + in-place growth would matter. But the current `kv_cache_update` is already optimal for the kernel-level append.

## Risk Register
- The hypothesis may be based on observing framework-level (not kernel-level) KV cache behavior. Verify whether the FFAI integration tests show any `newBufferWithLength` calls on the hot path.
- `MTLHeap` fragmentation: if multiple layers have different cache sizes, a shared heap may fragment. Per-layer pools (current design) avoid this.

## Notes for Next Person
- Before pursuing this, profile the KV cache append path end-to-end. If the profile shows `newBufferWithLength` or `memcpy` on the hot path, then the hypothesis is valid. If not, this is a no-op.
- The `BUF_POOL` in `context.rs` already eliminates allocation overhead. The kernel already writes in-place. The only remaining win is framework-level buffer lifetime management, which is outside MetalTile's scope.
