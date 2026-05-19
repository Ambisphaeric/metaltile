# 028 — logsumexp: fuse max + sum-exp

## Metadata
- **Number**: 028
- **Name**: logsumexp-fuse-max-sum-exp
- **Source**: `perf-ideas.md` § Op-level structural changes — item 28
- **Status**: ⚪ no-op
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> two-pass max-then-sum can collapse into one numerically-stable pass with a running update (same trick as online softmax).

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/logsumexp.rs`
- **Bench filter**: `tile bench -vv -f logsumexp`
- **Shapes / dtypes**: `B=1024 N=4096`, f32/f16/bf16

## Current Code Reality Check

The target kernel `mt_logsumexp` is already a **single-pass** online logsumexp implementation:

```rust
let mut lm = neg_infinity();
let mut nz = 0.0f32;
for _r in range(0u32, nf, 1u32) {
    let base = rs + (_r * lsize + tid) * 4u32;
    let v0 = load(inp[base]).cast::<f32>();
    // ... v1, v2, v3
    let cm = max(max(v0, v1), max(v2, v3));
    let pm = lm;
    let nm = max(pm, cm);
    nz = nz * exp(pm - nm) + exp(v0 - nm) + exp(v1 - nm) + exp(v2 - nm) + exp(v3 - nm);
    lm = nm;
}
```

This is exactly the "running update" trick the hypothesis describes:
- `pm` = previous max
- `nm` = new max (max of previous max and chunk max)
- `nz` = previous normalizer, rescaled by `exp(pm - nm)` to the new max's exponent, plus the new chunk's contributions.

The tail loop and final `reduce_max` / `reduce_sum` combine thread-local results into the per-row output. No two-pass over the input data occurs.

### MLX reference

MLX ships two logsumexp kernels:
1. **`logsumexp`** — **two-pass**: first `simd_max` to find global max, then `simd_sum` of `exp(x - maxval)`.
2. **`logsumexp_looped`** — **one-pass**: uses the same online running-update trick as MetalTile.

MetalTile's kernel matches MLX's `logsumexp_looped` (the `looped_logsumexp_{tn}` pattern is the bench reference).

### Baseline numbers

```
$ tile bench -vv -f logsumexp
B=1024 N=4096 f32  Ref=311.7 GB/s  MT=480.6 GB/s  MT%=154%  ok=✓  regs=54r
B=1024 N=4096 f16  Ref=156.3 GB/s  MT=371.5 GB/s  MT%=238%  ok=✓  regs=54r
B=1024 N=4096 bf16 Ref=152.6 GB/s  MT=361.5 GB/s  MT%=237%  ok=✓  regs=54r
```

MetalTile is **1.5×–2.4× faster** than MLX. The kernel uses only 54 registers and is thread-limited. The one-pass algorithm is already optimal.

## Risk Register
- **Already one-pass** — the kernel implements the exact optimization hypothesized. (new finding)
- **Already faster than MLX** — 154–238% of reference, with low register pressure. (new finding)
- **Numerical accuracy** — the online update formula is numerically stable; `tol=1e-4` passes. (from perf-ideas.md)

## Final Verdict
**No-op.**

The kernel already uses the single-pass online logsumexp algorithm. It matches MLX's `logsumexp_looped` and outperforms it significantly. There is no two-pass max-then-sum to collapse.

## Related Ideas
- **014** — scan: `simd_prefix_inclusive_sum` (already implemented; same "feature already landed" pattern).
- **027** — SSM state vectorization (already implemented; same no-op pattern).
