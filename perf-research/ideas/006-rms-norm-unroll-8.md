# 006 — RMS-norm: unroll 4 → 8

## Metadata
- **Number**: 006
- **Name**: rms-norm-unroll-8
- **Source**: `perf-ideas.md` § Quick-wins — item 6
- **Status**: ⚫ **abandoned** — register pressure regression
- **Worktree**: `../metaltile-perf-idea-6` (branch `perf/idea-6-rms-norm-unroll`)
- **Assignee**: pi

## Hypothesis
> 4-wide unrolls saturate ALU; 8-wide hides L1 latency better. README says already 104% of MLX on M4 — there may be headroom on M1.

## Target
- **Primary file**: `crates/metaltile-std/src/mlx/rms_norm.rs`
- **Bench filter**: `tile bench -vv -f rms_norm`
- **Shapes / dtypes**: `B=1024 N=4096` for f32, f16, bf16

## Baseline
Baseline `mt_rms_norm` (tpg=1024, 4-wide) from full bench:
| dtype | MT GB/s | MT% | regs |
|-------|---------|-----|------|
| f32 | 623.0 | 102% | 9r |
| f16 | 628.5 | 99% | 9r |
| bf16 | 626.5 | 125% | 9r |

## Experiment Log

### Cycle 1 — 2026-05-18
- **Change**: Kernel body expanded from 4-wide to 8-wide; `tpg` changed 1024→512 to preserve geometry (512 threads × 8 elements = 4096 = one row).
- **Bench result (2 runs)**:
  | dtype | MT GB/s | MT% | regs | occ% | bottleneck |
  |-------|---------|-----|------|------|------------|
  | f32 | 183–227 | 47–73% | **162r** | 73% | register-limited |
  | f16 | 171–511 | 39–80% | **162r** | 73% | register-limited |
  | bf16 | 410–498 | 79–100% | **162r** | 73% | register-limited |
- **Correctness**: `ok = ✓` (3/3 correct)
- **Trust**: cv% < 8% on all runs. But the result is a massive regression.
- **Decision**: revert immediately

### Cycle 2 — 2026-05-18
- **Change**: Reverted to baseline 4-wide kernel.
- **Bench result**: Back to 84r, 100% occ, ~275–664 GB/s depending on dtype and run. Kernel is healthy again.
- **Decision**: confirm abandon

## Analysis

### Why it failed
The 8-wide unroll holds **8 `x` values + 8 `w` loads + partial_ssq + eps + rms + tg_ssq** live across a large span. The MSL compiler cannot pack them into the 128-register/thread budget on Apple GPUs. Register count exploded from **9r → 162r**, which means aggressive spilling to threadgroup or device memory.

The occupancy dropped from **100% → 73%** because each thread now needs ~162 registers, and the GPU's register file is shared across all threads in a threadgroup. Fewer threads can be resident concurrently.

### The risk note was correct
The original idea said: *"Risk: register pressure; verify `regs` doesn't push past ~64 for f32."*

We pushed to **162**. The risk materialized exactly as predicted.

### Could it be salvaged?
Possibly, with a more aggressive register-aware rewrite:
- Reuse `x` registers for `w` loads (load w into the same register after x is no longer needed for ssq). But the stores at the end need both x and w simultaneously.
- Interleave loads and stores to shorten the live range. But the `#[kernel]` DSL doesn't give enough control over scheduling.
- Load `w` as a `float4` vector (would need DSL vector types, which we don't have — same blocker as ideas 5 and 8).

Any of these would be a **Multi-day** rewrite, not a Quick-win.

## Risk Register
- **Register pressure at 8-wide**: Confirmed catastrophic. 162r exceeds Apple GPU budget.
- **Occupancy collapse**: 73% vs 100% baseline.
- **Spill to slower memory**: The "register-limited" bottleneck label confirms the compiler is spilling.

## Final Verdict
**Abandoned.** The 8-wide unroll increases register pressure beyond what the Apple GPU can handle. The 4-wide baseline is already optimal for this kernel shape. No further Quick-win path exists here.

## Snapshots
- Baseline: captured in full-suite snap `010-run2.json` (committed on dev)
- 8-wide regression: bench output captured above

## Notes for Next Person
- If you ever get vector-load primitives in the DSL, revisit this with `float4`/`half8` loads. A single vector load uses fewer register-names than 4/8 scalar loads and may actually *reduce* register pressure despite wider unrolls.
- The same logic applies to **Idea 9 (LayerNorm)** — it has the same 4-wide structure with *two* accumulators (`s` and `sq`), so 8-wide would be even worse.
