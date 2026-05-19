# Session Kickoff Prompt — Ideas 21–30

> Copy-paste this into a fresh session to continue the established pattern.

## Context

We are working through the `perf-research/perf-ideas.md` hopper in the **metaltile** repo. The repo is a Metal GPU kernel compiler/benchmark project using a custom `#[kernel]` DSL. `#[bench_kernel]` macros generate benchmark specs. The CLI is `tile bench -vv -f <op>` and `tile inspect <kernel>`.

### What we've done so far
- Ideas 1–20 are **fully assessed** with individual files in `perf-research/ideas/NNN-<name>.md`
- STATUS.md and RESEARCH-LOG.md are updated after each idea
- Two actual changes committed on `dev` (marked FOR REVIEW LATER):
  - `perf(gemv): tune tpg=256→512`
  - `perf(softmax): add small-N bench variant`

### Your task
Assess ideas **21 through 30** from `perf-ideas.md`. For each idea:
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

Only create a worktree if you're actually going to run `tile bench`. For purely analytical assessments (reading code + writing the file), skip the worktree.

### Individual idea file format
Every idea must get its own file. Use this structure:

```markdown
# NNN — <Title>

## Metadata
- **Number**: NNN
- **Name**: <kebab-name>
- **Source**: `perf-ideas.md` § <Section> — item NNN
- **Status**: <⚪ not-started / 🟡 in-progress / 🔴 blocked / ⚫ abandoned / 🟢 done / ⚪ no-op / ⚠️ feasible / ⚪ marginal>
- **Worktree**: `<../metaltile-perf-idea-NNN or —>`
- **Assignee**: pi

## Hypothesis
> Paste original hypothesis.

## Target
- **Primary file(s)**: `crates/...`
- **Bench filter**: `tile bench -vv -f <op>`
- **Shapes / dtypes**: ...

## Current Code Reality Check
<Describe what the code actually does. What constants exist? What dispatch shape? What codegen emits?>

## Baseline
<If benched, paste numbers. If not, note "not benched — analytical assessment only.">

## Risk Register
<List risks, including ones from perf-ideas.md plus any new ones found.>

## Final Verdict
<win / no-change / regression / inconclusive / blocked / abandoned>

<If benched: include table of results.>
<If blocked: explain the blocker and list decision options.>

## Related Ideas
- <links to other idea files>
```

### Commit pattern
Group related changes into commits:
- Analytical assessments (no bench): `perf-research: feasibility files for ideas 21–25`
- Actual experiments: `perf(idea-NNN): <change description>` — FOR REVIEW LATER
- Always update STATUS.md and RESEARCH-LOG.md in the same commit as the new files.

---

## Ideas 21–30 (raw text from perf-ideas.md)

### 21. FP4 dequant: packed bit ops
*Target*: `mlx/fp_quantized.rs`.
*Hypothesis*: unpack 8 fp4 → 8 half in one 32-bit shuffle with a precomputed LUT in const memory.
*Measure*: `tile bench -f fp_quantized`.
*Risk*: LUT in `constant` address space — make sure it fits in the 64 KB const segment.

### 22. Quantized int4 GEMV: simdgroup_matrix multiply
*Target*: `ffai/dequant_gemv.rs`.
*Hypothesis*: currently dequant→scalar mul. Dequant a tile into threadgroup memory then use `simdgroup_matrix_multiply` (16×16×16 tile).
*Measure*: `tile bench -f dequant_gemv`.
*Risk*: bigger kernel — only wins above some K threshold.

### 23. Quantized GEMV: int4 pack-of-2 lookup
*Target*: `mlx/quantized.rs`.
*Hypothesis*: pack `(int4_lo, int4_hi)` as uint8 index into a 256-entry half2 LUT — single load yields two dequanted values.
*Measure*: `tile bench -f quantized`.
*Risk*: LUT init cost (one-time, threadgroup-shared).

### 24. dequant_gather: skip dequant for cold misses
*Target*: `ffai/dequant_gather.rs`.
*Hypothesis*: rare — most lookups hit the L1; the kernel currently dequants unconditionally. Profile to confirm there's a measurable cold-miss frequency before chasing.
*Measure*: `tile profile mt_dequant_gather`.
*Risk*: premature; verify the assumption first.

### 25. Sort: 4-way bitonic merge
*Target*: `mlx/sort.rs`.
*Hypothesis*: stride-2 bitonic does N/2 compares per stage; stride-4 does N/4 with the same simdgroup width.
*Measure*: `tile bench -f sort` (warning: rarely benched — may need a small wrapper).
*Risk*: register usage doubles — `regs` column will tell you.

### 26. Sampling: radix-select top-k
*Target*: `ffai/sampling.rs`.
*Hypothesis*: current top-k probably sorts then slices; radix-select is O(N) for fixed k.
*Measure*: add a `tile bench -f sampling` case if missing.
*Risk*: only wins for k ≪ N (typical k ≤ 64).

### 27. SSM: scan with state vectorization
*Target*: `ffai/ssm.rs`.
*Hypothesis*: state vector update per token is the hot path; fuse the scan with the state-update mul.
*Measure*: `tile bench -f ssm`.
*Risk*: SSM math is delicate — keep correctness check tight.

### 28. logsumexp: fuse max + sum-exp
*Target*: `mlx/logsumexp.rs`.
*Hypothesis*: two-pass max-then-sum can collapse into one numerically-stable pass with a running update (same trick as online softmax).
*Measure*: `tile bench -f logsumexp`.
*Risk*: numerical accuracy — verify against CPU reference at fp32.

### 29. Reductions over short rows: cooperative groups
*Target*: `mlx/reduce.rs`.
*Hypothesis*: for N ≤ 32, one simdgroup does the whole row. No threadgroup memory, no barrier.
*Measure*: `tile bench -f all_reduce` with N=32.
*Risk*: dispatch shape must match — codegen specialization.

### 30. binary_two (fused add+mul): autovec test
*Target*: `mlx/binary_two.rs`.
*Hypothesis*: `fma` should auto-emit. If MT% lags MLX, codegen is missing it; inspect MSL.
*Measure*: `tile bench -f binary_two`.
*Risk*: zero (diagnostic).

---

## First commands to run

```bash
git fetch upstream dev && git merge --ff-only upstream/dev
cd /Users/zstarer_entera/LLMs/metaltile
cat perf-research/perf-ideas.md | grep -n "^### 2[1-9]\.\|^### 30\."
```

## Key files to read

| Idea | Files |
|------|-------|
| 21 | `crates/metaltile-std/src/mlx/fp_quantized.rs` |
| 22 | `crates/metaltile-std/src/ffai/dequant_gemv.rs` |
| 23 | `crates/metaltile-std/src/mlx/quantized.rs` |
| 24 | `crates/metaltile-std/src/ffai/dequant_gather.rs` |
| 25 | `crates/metaltile-std/src/mlx/sort.rs` |
| 26 | `crates/metaltile-std/src/ffai/sampling.rs` |
| 27 | `crates/metaltile-std/src/ffai/ssm.rs` |
| 28 | `crates/metaltile-std/src/mlx/logsumexp.rs` |
| 29 | `crates/metaltile-std/src/mlx/reduce.rs` |
| 30 | `crates/metaltile-std/src/mlx/binary_two.rs` |

## Already-known blockers to watch for
- **No vector-load primitive in DSL** (`load()` is scalar) — affects any idea wanting vec4/vec8/half8
- **No `simd_shuffle` across threadgroups** — affects any cross-tg sharing idea
- **Dispatch grid is fixed by `#[bench_kernel]` macro** — changing it requires macro or `run_spec.rs` work
- **Register pressure is unpredictable** — the 8-wide RMS-norm unroll went 9r→162r. Always check `regs` if editing kernel body.

## What to skip vs assess
- If an idea needs **vector types in DSL** → mark 🔴 blocked, one paragraph
- If an idea needs **new kernel + new bench harness** → mark 🔴 blocked, note effort
- If an idea needs **dispatch restructuring** → mark 🔴 blocked or ⚠️ feasible depending on complexity
- If an idea is **"verify codegen already does X"** → run `tile inspect`, confirm, mark ⚪ no-op
- If an idea is a **single param tweak** (like tpg, unroll width) → actually try it, bench it

## Output expectation
At the end of the session, `ls perf-research/ideas/` should show files for **21, 22, 23, 24, 25, 26, 27, 28, 29, 30**. STATUS.md and RESEARCH-LOG.md should have entries for all of them. Aim for a single commit grouping the analytical assessments, plus any separate commits for experiments that were actually run.
