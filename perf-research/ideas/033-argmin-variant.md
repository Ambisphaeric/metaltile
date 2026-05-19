# Perf Idea 033 — arg_reduce variants: argmin in same kernel

## Metadata
- **Number**: 033
- **Name**: argmin-variant
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: ⚠️ feasible (small — copy-paste + flip comparison)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> argmax is great; argmin shares 90% of structure. Confirm both are equally tuned.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/arg_reduce.rs`, `crates/metaltile-std/src/ffai/arg_reduce.rs`
- **Bench filter**: `tile bench -f argmin` (does not exist yet — would need adding)
- **Shapes / dtypes to watch**: n=1048576, tpg=256

## Assessment

### Current state
**MLX variant** (`mlx/arg_reduce.rs`):
- `mt_argmax_f32` — f32-only, outputs f32 (the index cast to float). Uses `>` comparison, `neg_infinity()` init.

**FFAI variant** (`ffai/arg_reduce.rs`):
- `argmax<T>` — generic over input dtype, outputs `u32`. Uses `>` comparison, `neg_infinity()` init.

Neither file has an argmin variant.

### What argmin would look like
The argmax kernel is a threadgroup tree reduction. To get argmin:
1. Change `neg_infinity()` → `infinity()` (initial best value).
2. Change `>` → `<` (comparison direction).
3. On ties, keep the smaller index (same rule as argmax — already `oi < ti`).

That's it. The tree reduction, barrier pattern, threadgroup memory layout, and dispatch are all identical.

### Effort
- Add `mt_argmin_f32` in `mlx/arg_reduce.rs`: copy `mt_argmax_f32`, flip init and comparison. ~10 lines.
- Add `argmin<T>` in `ffai/arg_reduce.rs`: copy `argmax<T>`, flip init and comparison. ~10 lines.
- Add bench specs for both. ~10 lines each.

**Total: ~30 lines, one file per variant.**

### Why it's worthwhile
- Completeness: MLX has both `argmax` and `argmin` in its `arg_reduce.metal`. MetalTile only has argmax.
- The FFAI `argmax` is used for greedy token sampling. Argmin is less common in LLMs but appears in other ML workloads (e.g., finding nearest neighbors, min-distance queries).

## Verdict

- **Outcome**: feasible — trivial copy-paste variant
- **Why**: Argmin is structurally identical to argmax. Only the initial value and comparison direction differ.
- **Measure**: Add `argmin` bench variant, run `tile bench -f argmin`. Should match argmax throughput (~206% of MLX for f32).

## Risk Register
- `infinity()` may not exist as a DSL builtin. Check if `infinity::<f32>()` or `pos_infinity()` is available. If not, use `f32::MAX` or add the builtin.
- Tie-breaking semantics: argmin should pick the **smallest** index on ties, same as argmax. The existing `oi < ti` logic already does this.

## Notes for Next Person
- Start with the FFAI variant (`ffai/arg_reduce.rs`) since it's generic and more useful.
- The MLX bench variant (`mlx/arg_reduce.rs`) is f32-only; less urgent but nice for parity.
- Copy the `argmax` macro structure exactly — the `argmax_step!` macro works for argmin with no changes (it just uses `>` from the surrounding code).
