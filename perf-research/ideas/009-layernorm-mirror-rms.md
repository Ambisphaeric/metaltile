# 009 — LayerNorm: mirror RMS-norm tweaks

## Metadata
- **Number**: 009
- **Name**: layernorm-mirror-rms
- **Source**: `perf-ideas.md` § Quick-wins — item 9
- **Status**: ⚫ **abandoned by extension** — same register pressure issue as idea #6
- **Worktree**: —
- **Assignee**: —

## Hypothesis
> Same structural improvements (unroll 8, simdgroup reduce) apply.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/layer_norm.rs`
- **Bench filter**: `tile bench -vv -f layer_norm`
- **Shapes / dtypes to watch**: `B=1024 N=4096 f32/f16/bf16`

## Current Code Reality Check
The kernel is structurally identical to RMS-norm but with **two accumulators** (`s` and `sq`) instead of one (`ssq`):
```rust
let v0 = load(x[base]).cast::<f32>();
... // 4-wide
s = s + v0 + v1 + v2 + v3;
sq = sq + v0*v0 + v1*v1 + v2*v2 + v3*v3;
```

Bench: `b=1024, n=4096, tpg=1024`.

### Feasibility assessment
Same arithmetic as idea #6:
- Current: 1024 threads × 4 elements = 4096 → exactly covers the row, no loop.
- 8-wide with same params: 1024 threads × 8 elements = 8196 → overflows 4096.
- Must adjust `tpg=512` or `n=8192`.

### Register pressure check
Current: `v0..v3`, `s`, `sq`, `mean`, `var`, `is`, `w[col]..w[col+3]`, `b[col]..b[col+3]` → ~18 floats.  
Adding `v4..v7` → ~22 floats.

**But** — idea #6 proved that the actual compiled register count is **not** predictable from source analysis. The 8-wide RMS-norm exploded from 9r → 162r. LayerNorm has **more** live state than RMS-norm (two accumulators + weight + bias), so it would be **worse**.

### Why it's abandoned without benching
After idea #6's catastrophic result (9r→162r, −50% throughput), there is no reason to believe LayerNorm would fare better. It has strictly more live state:
- RMS-norm: `x0..x3`, `partial_ssq`, `tg_ssq`, `rms`, `w[col]..w[col+3]`
- LayerNorm: `x0..x3`, `s`, `sq`, `mean`, `var`, `is`, `w[col]..w[col+3]`, `b[col]..b[col+3]`

The kernel body edit is identical (copy-paste 4 more loads/accumulates/stores). The bench param adjustment is identical (`tpg=512` or `n=8192`). But the expected outcome is the same or worse than idea #6.

## Risk Register
- **Register pressure** — same as #6. Confirmed catastrophic.
- **Correctness** — any indexing bug affects all elements in the row. Low risk for mechanical copy-paste.

## Final Verdict
**Abandoned without benching.** Same pattern as idea #6, but with *more* register pressure. Not worth the compile-and-revert cycle when we already know the outcome from #6.

## Related Ideas
- **006** — RMS-norm unroll 4→8 (the experiment that proved this pattern fails)
- **M1** — ML autotuner (could learn that 4-wide is the sweet spot for these kernels)
