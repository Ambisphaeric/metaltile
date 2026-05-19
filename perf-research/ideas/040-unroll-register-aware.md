# Perf Idea 040 — `unroll.rs`: register-pressure-aware unroll count

## Metadata
- **Number**: 040
- **Name**: unroll-register-aware
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚠️ feasible — genuine missing feature, high value
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Today's unroll factor is likely fixed-per-loop. Pick `unroll_count = max_regs / regs_per_iter`.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/unroll.rs`, `crates/metaltile-codegen/src/passes/register_estimate.rs`
- **Bench filter**: aggregate bench; watch `regs` column
- **Shapes / dtypes to watch**: kernels with small constant trip-count loops (rms_norm, softmax, copy)

## Assessment

### Current `UnrollPass`
- Fixed `factor` (default 4, max 8 via `MAX_UNROLL_TRIP`).
- Unrolls a loop if `trip_count ≤ factor`.
- **No register pressure awareness.**
- **No partial unrolling** — loops with `trip_count > factor` are left rolled.

### `register_estimate.rs`
Already exists and computes:
- `max_live`: maximum simultaneously-live ValueIds across blocks.
- `regs_per_thread = max_live * 1.5` (heuristic).

It is used by the autotuner to compare tile size candidates, but **not by the unroller**.

### The problem this idea addresses
Idea #006 (RMS-norm unroll 4→8) was a catastrophic failure: register pressure exploded from 9r to 162r, occupancy dropped to 73%, and throughput regressed −50% to −80%. The root cause was unrolling without checking if the extra live values would fit in the register file.

A register-aware unroller would have prevented this: before unrolling, estimate the post-unroll register pressure. If it exceeds a threshold (e.g., 100r for f32, 80r for f16), reduce the unroll factor or skip unrolling entirely.

### What the integration would look like
1. Before unrolling a loop with trip count `tc` and body register pressure `body_regs`:
   - Estimate unrolled_regs = `current_regs + (tc - 1) * body_regs`.
   - Or more precisely: run `register_estimate.rs` on a cloned, unrolled version of the kernel body.
2. Compare against a threshold (e.g., 80% of architecture limit: ~120r for Apple GPUs).
3. If the estimate exceeds the threshold, reduce `factor` or skip this loop.
4. For partial unrolling: even if `tc > factor`, unroll by `factor` if registers allow. This is currently not supported at all (`MAX_UNROLL_TRIP` is the hard ceiling).

### Effort estimate
- Wire `register_estimate.rs` into `unroll_block`: **low** (both are in the same crate).
- Add a threshold / clamp logic: **low**.
- Add partial unrolling (unroll by `factor` even when `tc > factor`, with a cleanup loop): **medium**.
- **Total**: **one-day** for the basic register-aware clamp; **multi-day** for partial unrolling.

## Verdict

- **Outcome**: feasible — high-value, prevents the #006 catastrophe
- **Why**: The `UnrollPass` uses a fixed factor with no register check. `register_estimate.rs` already exists but is not consulted. Connecting them is a straightforward integration that would have caught the 8-wide unroll disaster.
- **Measure**: Run `tile bench` aggregate on kernels with unrolled loops. `regs` column should stay below ~100r for most kernels.

## Risk Register
- Register estimation is conservative (`max_live * 1.5` is an overestimate). The threshold must be chosen with this in mind — using 80% of the physical limit with a 1.5× heuristic means the effective cutoff is ~53 `max_live` values.
- Partial unrolling adds complexity: the cleanup loop for remaining iterations must handle the IV correctly and not duplicate too much code.
- The `register_estimate.rs` analysis is block-local and conservative (values are never killed). For unrolled loops, this overestimate is even more pronounced. A post-unroll re-estimate (after cloning the unrolled body) would be more accurate but more expensive.

## Notes for Next Person
- Start simple: before unrolling each loop, estimate the post-unroll register pressure. If it exceeds a threshold, cap the unroll factor to what fits.
- The threshold should be configurable (e.g., `MTLT_MAX_REGS=120`) so different GPU families can use different limits.
- Partial unrolling is the bigger win — many kernels have loops with trip counts like 32 or 64 that don't fit in `MAX_UNROLL_TRIP=8` today. Unrolling by 4 or 8 and keeping a cleanup loop would still expose vectorization opportunities for the unrolled portion.
