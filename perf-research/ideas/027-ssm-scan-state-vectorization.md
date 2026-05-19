# 027 — SSM: scan with state vectorization

## Metadata
- **Number**: 027
- **Name**: ssm-scan-state-vectorization
- **Source**: `perf-ideas.md` § Op-level structural changes — item 27
- **Status**: ⚪ no-op / already implemented
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> state vector update per token is the hot path; fuse the scan with the state-update mul.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/ssm.rs`
- **Bench filter**: `tile bench -vv -f ssm`
- **Shapes / dtypes**: `bf16`, Mamba 2 shapes (dh=64, ds=64, state_dim varies)

## Current Code Reality Check

The target file contains **three** SSM kernels:

### 1. `conv1d_causal_step` — depthwise causal convolution
One thread per channel. Serial loop over `kernel_size-1` state taps. Not the target of this idea.

### 2. `ssm_step` — selective scan, serial per `(head, d)`
```rust
for n in range(0u32, state_dim, 1u32) {
    let h_old = load(h[h_idx]);
    let b_n = load(b[n]).cast::<f32>();
    let new_h = decay * h_old + dt_val * b_n * x_d;   // state update
    store(h[h_idx], new_h);
    let c_n = load(c[n]).cast::<f32>();
    y_d = y_d + c_n * new_h;                           // dot-product accum
}
```

One thread per `(head, d)` pair. Iterates serially over `state_dim`. Loads old state, computes new state, stores new state, and accumulates `c[n] * new_h` into `y_d`.

**Important:** This is **not a scan**. Each state dimension `n` is independent — `h[h_idx]` is the previous token's state for that specific `(head, n, d)` slot. There is no cross-`n` dependency (no `h[n]` depends on `h[n-1]`). The loop is an **elementwise map + dot product**, both already fused in the same loop body.

### 3. `mt_ssm_step` — selective scan, vectorized/cooperative
```rust
let mut acc = 0.0f32;
for i in range(0u32, n_per_t, 1u32) {
    let s_idx = n_per_t * ds_idx + i;
    let db_by_x = x_val * dt_val * load(b_mat[...]);
    let new_state = da * load(state_in[idx]).cast::<f32>() + db_by_x;
    store(state_out[idx], new_state.cast::<T>());
    acc = acc + new_state * load(c_mat[...]).cast::<f32>();
}
let total = simd_sum(acc);
```

One threadgroup per `(d_idx, n)` output element, 32 threads (`ds_idx = tid`), each handling `n_per_t = ds / 32` state elements. The dot-product accumulation uses `simd_sum` across the simdgroup.

This is exactly the **state vectorization** the hypothesis describes. The serial loop over `state_dim` is parallelized across 32 threads, with cooperative reduction (`simd_sum`) for the output dot product.

### Bench status

`tile bench -f ssm` fails because all three kernels have `mlx_src: None` — there is no MLX reference kernel to compare against (MLX's `ssm.metal` is not in the pinned commit). The kernels are registered and compile, but the bench harness cannot find a reference.

## Baseline
Not benched — `tile bench -f ssm` fails due to missing MLX reference. Analytical assessment only.

## Risk Register
- **Misidentifies the operation** — the hypothesis calls the loop a "scan," but it is elementwise map + dot product with no cross-iteration dependency. There is no scan to fuse. (new finding)
- **Already vectorized** — `mt_ssm_step` already parallelizes the state-dim loop across 32 threads with `simd_sum`. The optimization hypothesized here already exists in the codebase. (new finding)
- **Math delicacy** — the kernels run accumulators in `f32` for numerical stability; changing the loop structure could introduce reordering-dependent drift. (from perf-ideas.md)
- **Missing MLX reference** — no side-by-side comparison possible until MLX ships `ssm.metal`. (new finding)

## Final Verdict
**No-op / already implemented.**

The hypothesis describes vectorizing the state-dimension loop and fusing the state update with the output dot product. Both are already present:
- In `ssm_step`, the state update and dot-product accumulation are fused in the same serial loop.
- In `mt_ssm_step`, the state-dimension loop is parallelized across a 32-thread simdgroup with `simd_sum` reduction — this is exactly the "state vectorization" the idea proposes.

There is no additional "scan" to fuse because the state dimensions are independent (no cross-`n` recurrence). The idea was likely written before `mt_ssm_step` was added to the codebase.

## Related Ideas
- **014** — scan: `simd_prefix_inclusive_sum` (already implemented; same pattern of "idea written before feature landed").
- **012** — all_reduce two-stage (already optimal; same no-op pattern).
