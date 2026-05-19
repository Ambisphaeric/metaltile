# Perf Idea 034 — softmax + attention epilogue fusion

## Metadata
- **Number**: 034
- **Name**: softmax-attention-fusion
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Standalone softmax bench is one number, but in real attention softmax + matmul-with-V is one operator. Fuse and bench against the two-kernel version.

## Target
- **Primary file(s)**: `mlx/softmax.rs` + new fused kernel in `ffai/`
- **Bench filter**: would need `tile bench -f softmax_v` (does not exist)
- **Shapes / dtypes to watch**: attention shapes (B=1, H=32, N=4096, D=128)

## Assessment

### What the idea describes
A fused kernel combining:
1. Online softmax over Q·K^T rows (the attention scores).
2. Multiply the softmax probabilities by V (the value matrix).

This is the second half of FlashAttention: after computing `S = Q @ K^T`, you softmax each row of S, then multiply by V. In standard attention, this is two dispatches (softmax kernel + GEMM kernel). In FlashAttention, it's fused inside the tile loop.

### Current state
MetalTile has:
- `mt_softmax` in `softmax.rs` — standalone row-wise softmax.
- No GEMM kernel (same blocker as #032).
- No attention kernel that computes `softmax(QK^T) @ V`.

The `mt_sdpa` kernel in `scaled_dot_product_attention.rs` is a **vector decode kernel** (one Q token at a time), not a tiled FlashAttention kernel. It does not have the `Q @ K^T` → softmax → `@ V` structure; instead it walks KV cache entries one at a time.

### What would be needed
1. **Tiled attention kernel**: A `#[kernel]` that computes `S = Q @ K^T` in tiles, applies online softmax per tile, and accumulates `P @ V`. This is the core of FlashAttention-2.
2. **Bench harness**: A `softmax_v` or `attention_fused` bench spec.

### Effort estimate
- Tiled FlashAttention kernel in DSL: **project-scale** (the original FlashAttention paper took months; a DSL port is non-trivial).
- Even a simplified fused `softmax + matmul(V)` for pre-computed S is a new kernel + dispatch.

## Verdict

- **Outcome**: blocked — prerequisite missing
- **Why**: MetalTile has no tiled attention kernel or GEMM kernel. The idea describes the core optimization of FlashAttention, which is a major kernel architecture, not a one-file tweak.
- **Re-scope**: This is a moonshot-level item (M5 or M7). It should be tracked there, not in the one-day hopper.

## Risk Register
- (not applicable — blocked by missing infra)

## Notes for Next Person
- If someone implements a tiled SDPA kernel (ideas #1–#3 are blocked waiting for this), then softmax+V fusion is the natural next step.
- MLX's `sdpa_full` kernel already does this fusion. MetalTile's `mt_sdpa` is the decode path only.
