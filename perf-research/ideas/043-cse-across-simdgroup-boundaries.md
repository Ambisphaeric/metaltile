# Perf Idea 043 — `cse.rs`: extend across simdgroup boundaries

## Metadata
- **Number**: 043
- **Name**: cse-across-simdgroup-boundaries
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚠️ feasible (needs re-scoping)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Shared subexpressions across simdgroup branches are likely missed today; broaden the scope.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/cse.rs`
- **Bench filter**: `tile bench` aggregate + `tile inspect` for code-size delta
- **Shapes / dtypes to watch**: kernels with `Op::If` branches that perform redundant ALU

## Assessment

The CSE pass is **strictly block-local** today.

Algorithm:
1. Run `cse_block` on `kernel.body`, building a `table: FxHashMap<OpKey, ValueId>`.
2. Propagate body-level eliminations into child blocks by remapping `ValueId`s.
3. Run `cse_block` independently on every nested block (`kernel.blocks`).

Each block gets its own fresh `table`. There is **no cross-block value numbering**. If the `then_block` and `else_block` of an `Op::If` both compute the same subexpression, each arm will independently eliminate duplicates *within* that arm, but the common computation is not hoisted to the parent block.

The framing "simdgroup boundaries" is imprecise — the IR has no explicit simdgroup boundary blocks. The correct scope is **cross-branch CSE** for `Op::If` (and potentially `Op::Loop` preheaders). This is a genuine missing optimization.

Example of missed optimization:
```
if cond {
    v1 = a + b
    ...
} else {
    v2 = a + b
    ...
}
```
Today both arms compute `a + b`. Cross-branch CSE would hoist `v = a + b` to before the `If`, then use `v` in both arms.

## Verdict

- **Outcome**: feasible — genuine missing feature, but needs re-scoping
- **Why**: CSE does not share subexpressions across sibling blocks. The hypothesis maps poorly to the IR (there are no "simdgroup boundary" blocks), but the underlying optimization (cross-branch CSE) is real and unimplemented.
- **Effort estimate**: one-day to multi-day. Requires:
  - Collecting common `OpKey`s across `then_block` and `else_block`.
  - Hoisting the common ops to the parent block before the `Op::If`.
  - Remapping `ValueId` references in both child blocks.
  - Safety: only pure ops; side-effecting ops must stay in their respective arms.

## Risk Register
- Aliasing across threads is a runtime concern, not an IR-level CSE concern. At the IR level, the risk is hoisting a load that has side effects (e.g., from a mutable param) — but the existing `read_only` check already guards this.
- Must be conservative with `Op::Load` from non-constant buffers; hoisting a load that is only executed in one arm today to unconditionally before the branch changes memory access patterns.

## Notes for Next Person
- Start with `Op::If` Diamond shapes where both arms are pure ALU. This is the safest cross-branch CSE target.
- `Op::Loop` bodies can also share subexpressions with their parent block (preheader CSE), but that's a different optimization (related to LICM, not CSE).
- Measure code-size delta on MSL output; if the MSL generator already emits `select` for small branches, cross-branch CSE may have limited impact on final code size.
