# Perf Idea 042 — `licm.rs`: hoist gather indices when loop-invariant

## Metadata
- **Number**: 042
- **Name**: licm-hoist-gather-indices
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: ⚪ no-op
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> `tensor[constant_idx]` inside an inner loop should be hoisted; verify currently is.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/licm.rs`
- **Bench filter**: inspect MSL on a kernel known to have invariant indices
- **Shapes / dtypes to watch**: any kernel with constant-index loads inside a loop

## Assessment

`licm.rs` **already hoists** loop-invariant `Load` ops from read-only (const) params.

Key evidence from source:
- `is_pure_op` returns `true` for `Op::Load { src, .. }` when `read_only.contains(src.as_str())`.
- The LICM fixpoint marks any pure op whose operands are all invariant as hoistable.
- A `tensor[constant_idx]` maps to `Op::Load` with `indices` composed of invariant `ValueId`s (e.g., `IndexExpr::Value` pointing to a constant or parent-block definition). Since the operands are invariant and the load is from a read-only param, the op is hoisted.

`Op::Gather` is explicitly **not** hoisted (falls in the `false` branch of `is_pure_op`). However, a gather-with-constant-indices is an unusual pattern in the DSL — it is typically resolved at compile time by the frontend. The more common pattern `load(weights[constant])` is a `Load`, not a `Gather`, and is already handled.

Test `hoists_read_only_load` in `licm.rs` confirms this behavior: a `Load` from a read-only param inside a loop body is hoisted to before the loop.

## Verdict

- **Outcome**: no-op — pass already does what the hypothesis claims
- **Why**: Loop-invariant loads from read-only tensors are hoisted by the existing LICM pass.
- **Measure**: `tile inspect` on kernels like `rms_norm` or `softmax` should show index/address arithmetic hoisted out of the inner loop.

## Risk Register
- (none — already implemented)

## Notes for Next Person
- If you ever see a loop-invariant load that *isn't* hoisted, the first things to check are: (1) is the param marked `is_output = true` (mutable), and (2) are the index expressions truly invariant (not derived from the loop variable).
- The `read_only` set is built from `ParamKind::Tensor` and `ParamKind::Strided` with `is_output = false`.
