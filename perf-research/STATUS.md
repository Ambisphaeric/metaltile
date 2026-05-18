# Perf Research Status Board

> Living tracker for the `perf-ideas.md` hopper.  
> One row per idea. Update as worktrees spin up and experiments complete.

| # | Name | Category | Status | Worktree | Baseline Snap | Final Snap | Verdict | Notes |
|---|------|----------|--------|----------|---------------|------------|---------|-------|
| 001 | SDPA tile: bump BLOCK_M on f16/bf16 | Quick-win | 🔴 blocked | `../metaltile-perf-idea-1` | — | — | — | Target kernel is scalar vector; BLOCK_M constant does not exist. Needs re-scope or prerequisite tiled kernel. [Details](ideas/001-sdpa-tile-block-m.md) |
| 002 | SDPA: BLOCK_N 64 → 128 on D=128 | Quick-win | ⚪ not-started | — | — | — | — | |
| 003 | SDPA: split-K for low-occupancy H=8 | Quick-win | ⚪ not-started | — | — | — | — | |
| 004 | SDPA-vector decode: GQA-aware K/V reuse | Quick-win | ⚪ not-started | — | — | — | — | |
| 005–015 | *(reserved for future quick-wins)* | Quick-win | ⚪ not-started | — | — | — | — | |
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
