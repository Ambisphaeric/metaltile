# Session Kickoff Prompt — Ideas 41–55

> Copy-paste this into a fresh session to continue the established pattern.

## Context

We are working through the `perf-research/perf-ideas.md` hopper in the **metaltile** repo. Ideas 1–40 are fully assessed with individual files in `perf-research/ideas/NNN-<name>.md`. STATUS.md and RESEARCH-LOG.md are kept current.

### Your task
Assess ideas **41 through 55** from `perf-ideas.md`. For each idea:
1. Read the target code file(s)
2. Determine if it's feasible, blocked, a no-op, or needs re-scoping
3. Create an individual file: `perf-research/ideas/NNN-<short-name>.md`
4. Update `perf-research/STATUS.md` and `perf-research/RESEARCH-LOG.md`
5. Commit to `dev`

### Worktree convention
Use a worktree off `dev` for any actual code experiments:
```bash
git worktree add -b perf/idea-NNN-<name> ../metaltile-perf-idea-NNN dev
```
Skip the worktree for purely analytical assessments.

### Individual idea file format
Use the standard template from `ideas/TEMPLATE.md`. Every idea gets its own file.

### Commit pattern
Group analytical assessments: `perf-research: feasibility files for ideas 41–45`. Actual experiments: `perf(idea-NNN): <description>` — FOR REVIEW LATER.

---

## Ideas 41–45 (Codegen passes)

### 41. `schedule.rs`: software pipelining
*Target*: `crates/metaltile-codegen/src/passes/schedule.rs`.
*Hypothesis*: emit load(i+1) before compute(i) in the inner loop; classic 2-stage pipeline hides 50%+ of L1 latency.
*Measure*: any memory-bound kernel — `rms_norm`, `softmax`, `copy`.
*Risk*: register pressure; needs to compose with unroll heuristic from #40.

### 42. `licm.rs`: hoist gather indices when loop-invariant
*Target*: `crates/metaltile-codegen/src/passes/licm.rs`.
*Hypothesis*: `tensor[constant_idx]` inside an inner loop should be hoisted; verify currently is.
*Measure*: inspect MSL on a kernel known to have invariant indices.
*Risk*: easy diagnostic.

### 43. `cse.rs`: extend across simdgroup boundaries
*Target*: `crates/metaltile-codegen/src/passes/cse.rs`.
*Hypothesis*: shared subexpressions across simdgroup branches are likely missed today; broaden the scope.
*Measure*: code-size delta on MSL; runtime delta on bench.
*Risk*: aliasing across threads — be conservative with side-effecting ops.

### 44. `if_conversion.rs`: predicate tiny ifs in inner loops
*Target*: `crates/metaltile-codegen/src/passes/if_conversion.rs`.
*Hypothesis*: divergent simdgroup execution from `if (mask) { ... }` costs more than always-executing both sides for short bodies.
*Measure*: `tile bench -f gemv_masked`.
*Risk*: predicating ops with side effects (loads with OOB are bad on Metal).

### 45. `value_sink.rs`: sink threadgroup-memory stores
*Target*: `crates/metaltile-codegen/src/passes/value_sink.rs`.
*Hypothesis*: sinking a `threadgroup` store to right before the next barrier shortens the live range and frees a register.
*Measure*: aggregate bench + `regs` column.
*Risk*: barrier semantics — must preserve happens-before across threads.

---

## Ideas 46–55 (Runtime / dispatch / build)

### 46. Wire the autotuner `lookup()` (currently a placeholder)
*Target*: `crates/metaltile-runtime/src/autotune.rs:228` — the comment says "placeholder, see comment".
*Hypothesis*: a real lookup pipeline that selects pre-tuned schedules per (kernel, shape, dtype) tuple unlocks every other tweak in this list.
*Measure*: end-to-end speedup once even one kernel is plumbed.
*Risk*: highest-ROI moonshot-adjacent item — should be #1 if the loop allows broader refactors.

### 47. PSO disk cache
*Target*: `crates/metaltile-runtime/src/context.rs`.
*Hypothesis*: cold-start compile time dominates first-run latency; serialize compiled PSOs to `~/.cache/metaltile/pso/`.
*Measure*: `time tile bench` cold vs warm.
*Risk*: invalidation on toolchain bump — hash the MSL.

### 48. Heap-backed buffer pool
*Target*: `crates/metaltile-runtime/src/buffer.rs`.
*Hypothesis*: `newBufferWithBytes` allocations have non-zero cost; pre-allocate a heap, slice from it.
*Measure*: bench dispatch overhead — micro-bench a no-op kernel.
*Risk*: lifetime tracking; integrate carefully with Rust ownership.

### 49. Reuse command buffer across bench iterations
*Target*: `crates/metaltile-std/src/runner.rs` (`bench_gbps`).
*Hypothesis*: each iteration currently makes a fresh `MTLCommandBuffer`. Encoding N dispatches into one buffer cuts driver overhead.
*Measure*: tiny kernels (`copy`, `arange`) — should see GB/s rise.
*Risk*: changes the semantics of per-iter timing — GPU timestamps still work, but the model becomes "average per dispatch within one buffer". Document carefully.

### 50. Fast-math + disable shader-validation in release
*Target*: `crates/metaltile-runtime/src/context.rs` `MTLCompileOptions`.
*Hypothesis*: shader validation is on by default in debug; ensure release path disables it. Combine with `MTLLanguageVersion::Metal3_1` + `mathMode = fast`.
*Measure*: `tile bench` aggregate.
*Risk*: minor numerical drift in unary ops — already gated by `precise::` annotations where it matters.

### 51. Bench: pipelined sample collection
*Target*: `crates/metaltile-std/src/runner.rs`.
*Hypothesis*: today the SLC flush + warmup + 10 samples is serial per kernel. Encode all warmups + samples in one command buffer, read timestamps from `MTLCounterSampleBuffer`.
*Measure*: total bench wall time.
*Risk*: per-sample timer resolution — verify `gpu_time` deltas still match.

### 52. Multi-launch occupancy headroom: persistent threadgroups
*Target*: `crates/metaltile-runtime/src/context.rs`, optional.
*Hypothesis*: for ops dispatched in tight sequence, persistent threadgroups that pull work from a queue beat re-dispatching every step.
*Measure*: a chain of small ops — would need a microbench.
*Risk*: big refactor, mostly relevant once op fusion is in place.

### 53. CLI: parallelize per-kernel benches across non-overlapping shapes
*Target*: `crates/metaltile-cli/src/cmd/bench.rs`.
*Hypothesis*: dev-loop friction. *Doesn't change kernel perf*, but speeds the loop — same benches in less wall time means more tweak cycles per hour.
*Measure*: `time tile bench`.
*Risk*: cross-kernel DVFS pollution; need to keep `flush_slc` between independent kernels (already does).

### 54. CLI: `tile bench --compare-against <baseline.json>` inline
*Target*: `crates/metaltile-cli/src/cmd/bench.rs`.
*Hypothesis*: half of the autoresearch loop's value is "did my last tweak improve or regress?" Save the previous run automatically, diff inline.
*Measure*: loop velocity (qualitative).
*Risk*: minor UX.

### 55. Build: precompile `.metallib` per Apple GPU family
*Target*: `crates/metaltile-std/build.rs`.
*Hypothesis*: today the runtime compiles MSL on first dispatch. Pre-compiled per-family `.metallib` (Apple7, Apple8, Apple9) eliminates first-dispatch latency.
*Measure*: `time tile bench` cold-cache.
*Risk*: build-time cost (acceptable); fall back to JIT compile when family unknown.

---

## Key files to read

| Idea | Files |
|------|-------|
| 41 | `crates/metaltile-codegen/src/passes/schedule.rs` |
| 42 | `crates/metaltile-codegen/src/passes/licm.rs` |
| 43 | `crates/metaltile-codegen/src/passes/cse.rs` |
| 44 | `crates/metaltile-codegen/src/passes/if_conversion.rs`, `gemv_masked.rs` |
| 45 | `crates/metaltile-codegen/src/passes/value_sink.rs` |
| 46 | `crates/metaltile-runtime/src/autotune.rs` |
| 47 | `crates/metaltile-runtime/src/context.rs` |
| 48 | `crates/metaltile-runtime/src/buffer.rs` |
| 49 | `crates/metaltile-std/src/runner.rs` |
| 50 | `crates/metaltile-runtime/src/context.rs` |
| 51 | `crates/metaltile-std/src/runner.rs` |
| 52 | `crates/metaltile-runtime/src/context.rs` |
| 53 | `crates/metaltile-cli/src/cmd/bench.rs` |
| 54 | `crates/metaltile-cli/src/cmd/bench.rs` |
| 55 | `crates/metaltile-std/build.rs` |

## Already-known blockers
- **Codegen pass changes are repo-wide** — ideas 41–45 are compiler infrastructure, not single-file tweaks
- **Runtime changes affect cold-start and dispatch overhead** — ideas 46–52 change how kernels are launched, not the kernels themselves
- **Build.rs changes affect the entire crate graph** — idea 55 touches the build system
- **Some ideas are UX / loop-velocity, not kernel perf** — ideas 53–54 speed up the research loop without changing throughput

## What to skip vs assess
- **Codegen pass ideas (41–45)** — read the pass source. Determine if the pass already does what's claimed, or if the idea describes a genuine missing feature. Mark ⚪ no-op, ⚠️ feasible, or 🔴 blocked depending on scope.
- **Runtime ideas (46–52)** — these touch `context.rs`, `runner.rs`, `buffer.rs`. Read the relevant code, assess whether the mechanism already exists or is missing. Note that 46 is the highest-ROI item but also the biggest refactor.
- **Build / CLI ideas (50, 53–55)** — check if the feature already exists (e.g., does release already disable shader validation? does `tile diff` already exist?). Mark ⚪ no-op if already done, ⚠️ feasible if a small patch, 🔴 blocked if blocked by missing infra.

## Output expectation
At the end of the session, `ls perf-research/ideas/` should show files for **41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55**. STATUS.md and RESEARCH-LOG.md updated. Group commits by category (codegen, runtime, build/cli).
