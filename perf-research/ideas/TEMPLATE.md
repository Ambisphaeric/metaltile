# Perf Idea Template

> Copy this file to `ideas/NNN-<short-name>.md` when starting a new idea.

## Metadata
- **Number**: NNN
- **Name**: <short descriptive name>
- **Source**: `perf-ideas.md` section (Quick-wins / One-day / Multi-day / Moonshot)
- **Status**: `not-started` | `in-progress` | `blocked` | `done` | `abandoned`
- **Worktree**: `../metaltile-perf-idea-NNN` (branch `perf/idea-NNN-<name>`)
- **Assignee**: (self or other)

## Hypothesis (from perf-ideas.md)
> Paste the original hypothesis here.

## Target
- **Primary file(s)**:
- **Bench filter**: `tile bench -vv -f <op>`
- **Shapes / dtypes to watch**:

## Baseline
Run the bench *twice* before touching code (DVFS stabilization). Capture:

```
$ tile bench -vv -f <op>
# paste headline numbers here
```

Save snapshot:
```bash
tile snap -o results/NNN-baseline.json
```

## Experiment Log

### Cycle 1 — <date>
- **Change**: (one-sentence)
- **Diff**: (link or paste)
- **Bench result**:
- **Correctness**: `ok = ✓` / fail
- **Trust**: `cv%` value
- **Decision**: keep / revert / iterate

### Cycle 2 — <date>
- **Change**:
- ...

## Risk Register
- (from perf-ideas.md risk section, plus any new ones found)

## Final Verdict
- **Outcome**: win / no-change / regression / inconclusive
- **MT% / GB/s delta**:
- **Merged commit**:
- **Snapshot diff**: `tile diff results/NNN-baseline.json results/NNN-final.json`

## Notes for Next Person
- What we learned, what to avoid, where the bodies are buried.
