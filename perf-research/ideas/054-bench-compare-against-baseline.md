# Perf Idea 054 — CLI: `tile bench --compare-against <baseline.json>` inline

## Metadata
- **Number**: 054
- **Name**: bench-compare-against-baseline
- **Source**: `perf-ideas.md` — Runtime / dispatch / build (Multi-day)
- **Status**: ⚠️ feasible (small UX feature)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Half of the autoresearch loop's value is "did my last tweak improve or regress?" Save the previous run automatically, diff inline.

## Target
- **Primary file(s)**: `crates/metaltile-cli/src/cmd/bench.rs`
- **Bench filter**: `tile bench` (any run)
- **Shapes / dtypes to watch**: all

## Assessment

### Current state
`bench.rs` already supports saving results to JSON:
```rust
if let Some(path) = json_out {
    save_json(&runner.device_name, &all, path);
}
```

But there is **no** `--compare-against` flag, no automatic previous-run saving, and no inline diff.

### What the feature would do
1. Add `--compare-against <path>` to `BenchArgs`.
2. Load the baseline JSON and index it by `(op, subop, shape, dtype)`.
3. During printing, for each result row:
   - If baseline exists, compute `delta% = (current - baseline) / baseline * 100`.
   - Print an inline indicator: `↑ +5.2%` (green), `↓ -3.1%` (red), `= 0.0%` (gray).
4. Optionally auto-save the current run to a default path (e.g., `~/.cache/metaltile/bench-last.json`) so the next run can diff against it without explicit `--compare-against`.

### Effort
- Add CLI flag to `BenchArgs`: **low**.
- Parse baseline JSON and build lookup map: **low**.
- Modify `SuitePrinter` to print delta column: **low**.
- Auto-save default path: **low**.
- **Total**: **one-day**.

### Why it's valuable
The autoresearch loop is: tweak → bench → observe → tweak. The "observe" step currently requires:
1. Remembering the previous run's numbers.
2. Or running `tile diff` between two saved JSONs.

Inline comparison eliminates this friction and makes regressions immediately visible.

## Verdict

- **Outcome**: feasible — small, high-value UX improvement
- **Why**: The infra already exists (JSON save/load). Adding diff is a thin layer. The perf-ideas.md "risk: minor UX" understates the value — this significantly speeds the research loop.

## Risk Register
- JSON schema changes: the baseline format must be stable. The current schema is already additive (see `format_result_row` tests).
- Missing baseline entries: if a kernel was added since the baseline, there's nothing to compare against. Handle gracefully (no delta column).

## Notes for Next Person
- Start with `--compare-against` flag. The auto-save default path is a nice follow-up.
- Use `serde_json` to parse the baseline. The current `save_json` output is already valid JSON.
- Consider adding a `tile diff` command if it doesn't already exist (the `STATUS.md` mentions `tile diff` but the CLI may not implement it yet).
