# Perf Idea 046 — Wire the autotuner `lookup()` (currently a placeholder)

## Metadata
- **Number**: 046
- **Name**: autotuner-lookup
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (medium effort)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> A real lookup pipeline that selects pre-tuned schedules per (kernel, shape, dtype) tuple unlocks every other tweak in this list.

## Target
- **Primary file(s)**: `crates/metaltile-runtime/src/autotune.rs:87` (`TuneCache::lookup`)
- **Bench filter**: end-to-end speedup once even one kernel is plumbed
- **Shapes / dtypes to watch**: any kernel with multiple schedule candidates

## Assessment

### Current state
`TuneCache::lookup()` is a **placeholder**:
```rust
pub fn lookup(&self, _constexprs: &ConstExprValues) -> Option<&TuneEntry> {
    // In production: bucket the values, then hash the bucket key,
    // then look up in entries. For now, return None (always re-tune).
    None
}
```

The `Autotuner::get_or_tune()` method falls through to `None` when enabled and the cache is empty, meaning **every dispatch triggers a tuning search** (or returns no config at all, depending on the caller).

### What already exists
- `TuneCache` has `entries: BTreeMap<String, TuneEntry>` with save/load to `~/.cache/metaltile/<chip>/<kernel_hash>.json`.
- `TuneEntry` contains `bucket: Vec<ShapeBucket>` with `dim_name`, `lo`, `hi`.
- `TuneConfig` has `tile_dims`, `threads`, `unroll_factor`, `use_simd_matrix`, `use_async_copy`.
- The `autotuner_get_or_tune_enabled_with_empty_cache_returns_none` test confirms the placeholder behavior.

### What's missing
1. **Bucketing**: `ConstExprValues` must be mapped to `ShapeBucket` keys (e.g., `N=4096` → bucket `"N"` with `lo=2048, hi=8192`).
2. **Hashing**: A key format like `"kernel_name@N=2048..8192@D=64..128"` to look up in `entries`.
3. **Plumbing**: `Context::dispatch` must call `tuner.get_or_tune(kernel.name, constexprs)` and apply the returned `TuneConfig` to the `ScheduleConfig` before running codegen.
4. **Search strategy**: The comment mentions "grid search with exponential backoff" but no search code is visible in the current file.

### Effort estimate
- Bucketing + hashing: **low**.
- Plumbing into `Context::dispatch`: **medium** (needs to thread `TuneConfig` through `MslGenerator` or `SchedulePass`).
- Implementing the actual grid-search tuner: **multi-day**.
- **Overall**: **medium** to get basic lookup working; **multi-day** for full autotune pipeline.

## Verdict

- **Outcome**: feasible — genuine missing feature, high ROI
- **Why**: The `lookup()` stub means the autotuner infrastructure (cache, config, entries) is built but not wired. Implementing bucketing + lookup + plumbing would unlock schedule selection for every kernel.
- **Note**: The perf-ideas.md description says "highest-ROI moonshot-adjacent item — should be #1 if the loop allows broader refactors." This is accurate.

## Risk Register
- Tuning search overhead: the first dispatch of a new shape would still need a search. The cache amortizes this for repeated shapes.
- Key collision: different kernels with the same name hash would collide. Include kernel name + chip family in the key.

## Notes for Next Person
- Start with `lookup()`: given `ConstExprValues`, iterate the buckets and find the entry where all dimensions fall within the bucket ranges.
- The cache already has the JSON persistence layer. Don't rebuild that.
- A minimal viable implementation: `lookup` returns `Some(entry)` if all constexprs match bucket ranges, else `None`. Then `get_or_tune` returns the cached config instead of `None`.
