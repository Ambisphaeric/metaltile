# Perf Research Status Board

> Living tracker for the `perf-ideas.md` hopper.  
> One row per idea. Update as worktrees spin up and experiments complete.

| # | Name | Category | Status | Worktree | Baseline Snap | Final Snap | Verdict | Notes |
|---|------|----------|--------|----------|---------------|------------|---------|-------|
| 001 | SDPA tile: bump BLOCK_M on f16/bf16 | Quick-win | 🔴 blocked | `../metaltile-perf-idea-1` | — | — | — | Target kernel is scalar vector; BLOCK_M constant does not exist. Needs re-scope or prerequisite tiled kernel. [Details](ideas/001-sdpa-tile-block-m.md) |
| 002 | SDPA: BLOCK_N 64 → 128 on D=128 | Quick-win | ⚪ not-started | — | — | — | — | |
| 003 | SDPA: split-K for low-occupancy H=8 | Quick-win | ⚪ not-started | — | — | — | — | |
| 004 | SDPA-vector decode: GQA-aware K/V reuse | Quick-win | 🔴 blocked | `../metaltile-perf-idea-4` | — | — | — | `simd_shuffle` can't cross threadgroups; real fix is dispatch-shape change + cooperative tg-mem K/V caching. [Details](ideas/004-sdpa-gqa-kv-reuse.md) |
| 005 | SDPA-vector: 8-wide vec loads f16/bf16 | Quick-win | 🔴 blocked | — | — | — | — | DSL has no vector-load primitive. [Details](ideas/005-010-feasibility-study.md#5-sdpa-vector-8-wide-vectorized-loads-on-f16bf16) |
| 006 | RMS-norm: unroll 4 → 8 | Quick-win | ⚫ abandoned | `../metaltile-perf-idea-6` | — | — | Regression | 8-wide unroll pushes register pressure to 162r (was 9r), occupancy drops to 73%, kernel becomes register-limited. Reverted. [Details](ideas/006-rms-norm-unroll-8.md) |
| 007 | Softmax: simdgroup reduce for small N | Quick-win | ⚪ not-started | — | — | — | — | Kernel already optimal; bench doesn't exercise small N. Need `n=32` shape. [Details](ideas/005-010-feasibility-study.md#7-softmax-simdgroup-reduce-for-small-n-32) |
| 008 | Softmax: float4 loads on f16/bf16 | Quick-win | 🔴 blocked | — | — | — | — | DSL has no vector-load primitive. Same blocker as #5. [Details](ideas/005-010-feasibility-study.md#8-softmax-float4-loads-on-f16bf16-inner-loop) |
| 009 | LayerNorm: mirror RMS-norm tweaks | Quick-win | ⚪ not-started | — | — | — | — | Same as #6: trivial kernel edit, need param adjustment. [Details](ideas/005-010-feasibility-study.md#9-layernorm-mirror-rms-norm-tweaks) |
| 010 | GEMV: tune `simd_per_tg` per K | Quick-win | 🟢 done | `../metaltile-perf-idea-10` | `010-run2.json` | `010-run2.json` | Small win for f16 | tpg=512 beats baseline by +1.8% on f16. tpg=1024 is a −20% regression on f16. f32/bf16 flat. [Details](ideas/010-gemv-tpg-sweep.md) |
| 011–015 | *(reserved for future quick-wins)* | Quick-win | ⚪ not-started | — | — | — | — | |
| 016–035 | *(one-day items)* | One-day | ⚪ not-started | — | — | — | — | |
| 036–055 | *(multi-day items)* | Multi-day | ⚪ not-started | — | — | — | — | |
| M1–M10 | *(moonshots)* | Moonshot | ⚪ not-started | — | — | — | — | |

## Legend
- 🔴 **blocked** — prerequisite missing or idea ill-formed against current code
- 🟡 **in-progress** — worktree checked out, bench cycles running
- 🟢 **done** — final snap saved, verdict recorded
- ⚪ **not-started** — no worktree yet
- ⚫ **abandoned** — idea discarded with reason

## Quick Commands

Spin up a new worktree for idea NNN:
```bash
git fetch upstream dev
git worktree add -b perf/idea-NNN-<name> ../metaltile-perf-idea-NNN dev
```

Save a snapshot:
```bash
tile snap -o perf-research/results/NNN-<label>.json
```

Diff two snapshots:
```bash
tile diff perf-research/results/NNN-baseline.json perf-research/results/NNN-final.json
```

## Methodology Reminders
1. Run bench *twice* before claiming a regression (DVFS stabilization after recompile).
2. Always check the `ok` column. Speedup with a correctness regression is not a win.
3. Watch `cv%` — anything > 5% means the win is bench noise.
4. `min_us` drives GB/s; `p95`/`p99`/`cv%` from `-vv` are the trust signals.
