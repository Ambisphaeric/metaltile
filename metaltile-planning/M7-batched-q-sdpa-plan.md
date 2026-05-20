# M7 — Batched-Q SDPA Implementation Plan

**Plan ID**: M7
**Project item**: [M7] Speculative-decode batched-Q SDPA (status: Todo, category: Moonshot)
**Upstream link**: #47 (FA2 prefill kernel) — 70% match, type: Moonshot engine
**Source spec**: `perf-research/ideas/m7-speculative-decode-batched-q-sdpa.md`
**Locked decisions** (from research session 2026-05-19 + 2026-05-19 follow-up):
- K targets in v1: **K ∈ {2, 4, 8, 16}** (single PR)
- v1 scope: **kernel + bench harness + M4/M5 baselines**
- PR sequencing: **land M7 v1 before #46** (BenchDispatch refactor)
- Base branch: rebase on `upstream/dev` first (picks up #50 sliding-window + #48 GEMV)

Credit: kernel patterns at K=16 follow the production verify_qmm.py and
gated_delta_tree_tape_kernel kernels from the dflash-mlx repo (BM=16, BN=16,
BK=32, NSG=8, fp16 accumulators with threadgroup tree reduction).

---

## 1. Why this design (the structural fork)

The user's spec describes a single hero kernel at K=16 with FA-2 MMA tiling
(BM=BN=16, BK=32, NSG=8, fp16 accum, threadgroup tree reduction). The
existing `sdpa_decode.rs` is **not** that pattern — it's a single-Q
simdgroup-per-head, lane-per-quartile design that scales poorly to large K
because per-lane softmax/output state grows linearly in K.

The metaltile codebase **already ships an FA-2 MMA prefill tile**
(`mt_sdpa_prefill_mma` from #47 / #52). That tile is structurally identical
to dflash-mlx's verify_qmm: BQ rows × full head_dim, online softmax in
registers, KV reuse across BQ. **Driving the prefill tile with `q_len=K,
k_len=n_kv, BQ=16, BK=32` is the "K=16 batched-Q SDPA" the spec asks for.**

We split M7 across the fork:

| K | Architecture | Reason |
|---|---|---|
| 2, 4 | Decode-form (extend `sdpa_decode.rs` with `#[constexpr] batch_q`) | Low register pressure; reuses the proven lane-quartile pattern; constexpr specialization mirrors BM=2/BM=4 QMM (#56/#57). |
| 8, 16 | Prefill-tile reuse (`mt_sdpa_prefill_mma` with `q_len=K`) | Tile already exists; KV reuse mechanism is already proven. No new MSL unless this underperforms. |

This avoids inventing a second MMA kernel from scratch when an equivalent
one already ships, while still covering the spec's K=16 hero claim.

---

## 2. File-level surface

### New files
- `crates/metaltile-std/src/ffai/sdpa_decode_batched.rs` — K=2/4 kernel.
- `crates/metaltile-std/tests/sdpa_decode_batched_gpu.rs` — correctness vs K independent dispatches.

### Modified files
- `crates/metaltile-std/src/spec.rs:194` — new `BenchDispatch::SdpaBatchedDecode { ... }` variant.
- `crates/metaltile-std/src/spec.rs:338` — `default_mode` arm (Reduction for K=2/4, SimdGroup2D for K=8/16).
- `crates/metaltile-std/src/run_spec.rs:78` — dispatch arm → `run_sdpa_batched_decode`.
- `crates/metaltile-std/src/run_spec.rs` (~line 1868) — add `run_sdpa_batched_decode` after `run_sdpa_vector`.
- `crates/metaltile-std/src/ffai/mod.rs` — add the new module.
- `baselines/m4_max.json`, `baselines/m5_max.json` — refresh with new K rows (mirrors #54).

### Touch-only-if-necessary
- `crates/metaltile-std/src/ffai/sdpa_decode.rs` — keep as-is. The batched variant lives in a separate file so the single-Q kernel stays trivially diffable against upstream (and #50 sliding-window stays clean).

---

## 3. Phase 0 — Rebase + scaffolding

```bash
git fetch upstream
git checkout dev
git rebase upstream/dev    # picks up #48, #50
# verify sliding-window logic in sdpa_decode.rs after rebase
```

Add `SdpaBatchedDecode` to `BenchDispatch`:

```rust
// spec.rs:194 (in the enum)
SdpaBatchedDecode {
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    batch_q: usize,           // K — 2, 4, 8, or 16
    variant: BatchedDecodeVariant,
    tpg: usize,
},

pub enum BatchedDecodeVariant {
    /// K=2/4: extends sdpa_decode lane-quartile pattern with K independent softmax streams.
    Decode,
    /// K=8/16: reuses mt_sdpa_prefill_mma with q_len=K.
    PrefillTile { bq: usize, bk: usize, wm: usize, wn: usize },
}
```

`default_mode` returns `KernelMode::Reduction` for `Decode`, `SimdGroup2D`
for `PrefillTile`. Wire the runner dispatch in `run_spec.rs:78` to call
`run_sdpa_batched_decode`. Skeleton uses `todo!()` for both variants.

**Done when**: `cargo build -p metaltile-std` succeeds, `tile bench --list`
shows the new variant, `cargo test -p metaltile-std --test sdpa_decode_gpu_correctness`
still passes (no regression to the single-Q path).

---

## 4. Phase 1 — `sdpa_decode_batched<T>` for K ∈ {2, 4}

### Kernel signature

```rust
#[kernel]
pub fn sdpa_decode_batched<T>(
    q: Tensor<T>,         // [n_q_heads, batch_q, head_dim]
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,       // [n_q_heads, batch_q, head_dim]
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] kv_stride: u32,
    #[constexpr] heads_per_group: u32,
    #[constexpr] batch_q: u32,    // 2 or 4 — constexpr-specialized
    #[constexpr] scale: f32,
)
```

### Per-lane state (vs single-Q)

```
single-Q:    q0..q3 (T*4), run_max (f32), run_sum (f32), o0..o3 (f32*4)  = 10 reg-equiv
batched K=2: q[0..2][0..3], run_max[2], run_sum[2], o[0..2][0..3]        = 20
batched K=4:                run_max[4], run_sum[4], o[0..4][0..3]        = 40
```

At K=4 we're at ~40 fp32-equivalent registers per lane, still inside Apple
GPU's 64-reg/lane sweet spot (no spill). K=8 in decode-form would push past
this — that's why K=8/16 goes through the prefill tile instead.

### Threadgroup layout

`tg_max`, `tg_sum` widened from `[32]` to `[32 × batch_q]` (still ≤ 128
fp32 each — trivial). `tg_out0..3` widened from `[1056]` to
`[1056 × batch_q]` (≤ 16 KiB total at K=4 — within threadgroup memory
limits on all Apple GPUs).

### KV-walk loop (the win)

KV is loaded once per `_t` step, dot-producted against all K Q vectors
inside the lane via fp32 multiply-add. The simd_sum reduces each of the K
scores. Online softmax updates each of the K (run_max, run_sum, o[0..4])
tuples in lockstep. **Memory traffic for KV stays at N loads regardless of
K.** That's the bandwidth amortization the perf-research doc and the
user's spec both target.

### Module doc-comment (top of file)

```rust
//! Batched-Q SDPA decode — K independent online-softmax streams share
//! one KV walk. Extends the single-Q sdpa_decode.rs lane-quartile pattern
//! with #[constexpr] batch_q for K∈{2,4}. Larger K (8,16) ships via
//! prefill-tile reuse — see BenchDispatch::SdpaBatchedDecode docs.
//!
//! Kernel pattern (KV reuse via single load + multi-Q dot product, online
//! softmax with per-stream rescale) follows the production
//! `verify_qmm` / `gated_delta_tree_tape_kernel` kernels from the
//! dflash-mlx repo.
```

### Bench inventory rows

One `inventory::submit!` per K∈{2,4}, with dtypes [F16, BF16, F32] and
shapes (head_dim=128, n_q_heads=32, gqa_factor=4, n_kv∈{1024, 4096, 16384}).

---

## 5. Phase 2 — K ∈ {8, 16} via `mt_sdpa_prefill_mma` reuse

No new MSL. The runner for `BatchedDecodeVariant::PrefillTile` allocates Q
shaped `[n_q_heads, K, head_dim]` and dispatches the existing
`mt_sdpa_prefill_mma` kernel with `q_len=K, k_len=n_kv, qL_off=n_kv-K`
(causal-mask-trimmed so each Q position only attends to its prefix). BQ=16
covers K=8 (one tile) and K=16 (one tile exactly).

### Why this works

`mt_sdpa_prefill_mma` already implements the dflash-mlx verify_qmm pattern:
BQ rows × full head_dim, KV loaded once into shared memory per BK block,
multiplied against all BQ Q rows simultaneously via simdgroup MMA, online
softmax in registers, fp16 accumulators where the tuning chose them. The
"16× KV amortization" the spec claims **already exists in this kernel** —
M7 just exposes it through the decode-shape dispatch surface.

### Fallback

If the prefill tile underperforms the K-decode baseline at long n_kv
(unlikely given #47's M4/M5 numbers, but possible because prefill tuning
assumes long q_len), Phase 2 adds a `sdpa_decode_batched_mma<T>` kernel —
a verify_qmm-shaped hand-roll with BM=16, BN=16, BK=32, NSG=8, fp16 accum,
threadgroup tree-reduction. This is the user's spec verbatim. We only
write it if the data demands it.

### Bench inventory rows

One submit per K∈{8,16} at the same shape grid as Phase 1.

---

## 6. Correctness

`tests/sdpa_decode_batched_gpu.rs`. Golden = K independent `sdpa_decode`
dispatches with the same KV cache. Tolerance 1e-3 (matches the existing
sdpa tests).

| K | head_dim | n_kv | gqa_factor | dtypes |
|---|---|---|---|---|
| 2, 4, 8, 16 | 128 | 512, 4096, 16384 | 1, 4, 8 | F16, BF16, F32 |

Plus the `n_kv=0` guard (matches `sdpa_decode.rs:158`'s `rescale =
select(g_sum > 0, ..., 0.0)`).

---

## 7. Performance targets

| K | Baseline | Target | Rationale |
|---|---|---|---|
| 2 | 2× single-Q dispatches | ≥ 1.7× | 85% of theoretical 2× KV amortization |
| 4 | 4× single-Q dispatches | ≥ 3.4× | 85% of theoretical 4× |
| 8 | 8× single-Q dispatches | ≥ 6.8× | 85% via prefill tile |
| 16 | 16× single-Q dispatches | ≥ 13.6× | The user's "≥14× / 16× max" claim |

Long-N regime (n_kv ≥ 16K) is the regime the win matters most — that's
where KV bandwidth dominates. At n_kv=1024 the kernel may be compute-bound
and the K-amortization ratio drops; that's acceptable and expected.

Refreshed baselines/m4_max.json and baselines/m5_max.json mirror the #54
PR pattern.

---

## 8. Risks & open items

| Risk | Mitigation |
|---|---|
| Register pressure at K=4 spills | Constexpr specialization keeps the unroll explicit; if K=4 spills on M-series, drop to K=2/3 only and document. |
| Prefill tile suboptimal at q_len=K (small Q, long KV) | Phase 2 fallback: hand-rolled verify_qmm-shaped MMA kernel with fp16 accum. |
| #46 (BenchDispatch refactor) lands first | Mitigated by user-confirmed sequencing: land M7 before #46. If #46 races us, rewire the dispatch in a follow-up. |
| PR #50 (sliding-window) merge conflict | The batched-Q kernel lives in a separate file; sliding-window in single-Q `sdpa_decode.rs` stays untouched. |
| GQA factor interaction | The 2-pass kernel's `gqa_factor` co-load is orthogonal to batch_q — they stack cleanly. We do NOT touch the 2-pass kernel in v1. |

### Explicitly out of scope (deferred or belongs in dflash-mlx)

- Tree-structured attention mask (ancestor-only visibility) — needs a DDTree consumer; no consumer in metaltile.
- Per-candidate RoPE inside the kernel — RoPE lives in `rope_llama.rs`; fusion is a separate idea.
- GatedDeltaNet `history[16][n_per_t]` tape — recurrent state op, not SDPA.
- DFlash auto-enable policy (MoE + dense ≥40 layers) — belongs in the consumer that picks K.
- 2-pass batched decode — separate PR after v1 ratifies the win.

---

## 9. PR layout (single PR, per locked decision)

**Title**: `perf(std): M7 — batched-Q SDPA decode (K∈{2,4,8,16})`

**Body sections**:
1. Summary — what + the bench number.
2. Architecture — the K=2/4 vs K=8/16 fork and why.
3. Credit — verify_qmm.py and gated_delta_tree_tape_kernel patterns from the dflash-mlx repo.
4. Test plan — correctness matrix + bench rows.
5. Baseline refresh — links to before/after numbers.

**Commit hygiene** (per `memory/feedback_commit_hygiene.md`): no
Co-Authored-By trailers, no AI platform names anywhere in commits or PR
body. CI will reject otherwise.

---

## 10. Task index

See harness task list (TaskList) for live status. Sequence:

1. Rebase dev on upstream/dev (#48 + #50) — blocks all.
2. Add `SdpaBatchedDecode` dispatch + runner skeleton — blocks 3, 4.
3. Implement K=2/4 decode-form kernel — blocks 5.
4. Wire K=8/16 via prefill tile — blocks 5.
5. Correctness tests — blocks 6.
6. Bench rows + baseline refresh — blocks 7.
7. Open PR before #46 lands.
