# Perf Idea 041 — `schedule.rs`: software pipelining

## Metadata
- **Number**: 041
- **Name**: schedule-software-pipelining
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> emit `load(i+1)` before `compute(i)` in the inner loop; classic 2-stage pipeline hides 50%+ of L1 latency.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/schedule.rs`
- **Bench filter**: `tile bench -vv -f rms_norm` / `softmax` / `copy`
- **Shapes / dtypes to watch**: any memory-bound kernel

## Assessment

The `schedule.rs` pass **does not perform loop scheduling** of any kind.

Source inspection shows it is a minimal tile-annotation pass:
- It walks the IR and annotates `Op::Dot { .. }` ops with tile dimensions `(M, N, K)` from a `ScheduleConfig`.
- These annotations are consumed later by the MSL generator so the autotuner can vary tile sizes without regenerating MSL source.
- The pass never looks at `Op::Loop` bodies, never reorders instructions, and has no concept of pipeline stages.

Software pipelining (interleaving iteration `i+1` loads with iteration `i` compute) is **not implemented anywhere** in the current codegen pipeline. It would require:
1. A new pass that identifies load-compute patterns inside `Op::Loop` bodies.
2. Unrolling by at least 2× to create distinct `load(i+1)` and `compute(i)` stages.
3. Careful handling of the loop epilogue / prologue to avoid out-of-bounds loads.
4. Register-pressure analysis to ensure the extra live values don't spill.

## Verdict

- **Outcome**: blocked — target file does not contain the hypothesized mechanism
- **Why**: `schedule.rs` is a tile-dimension annotator, not a loop scheduler. The hypothesis describes a completely different compiler pass.
- **Re-scope**: A software-pipelining pass would be a genuine multi-day codegen feature, but it belongs in a new file (e.g., `software_pipeline.rs`), not in `schedule.rs`.

## Risk Register
- Register pressure from extra live values across stages (as noted in perf-ideas.md).
- Needs to compose with the unroll heuristic from #40 — both require register estimation.

## Notes for Next Person
- Don't be misled by the file name "schedule". In this codebase, "schedule" means "tile size schedule for the autotuner", not "instruction schedule within a loop".
- If someone implements a software-pipelining pass, it should run after unrolling and before register estimation.
