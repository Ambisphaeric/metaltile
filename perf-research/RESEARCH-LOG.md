# Research Log — What We've Investigated & Results

> Living document. One entry per investigated idea. Updated after each cycle.  
> For methodology, see [TEMPLATE.md](ideas/TEMPLATE.md). For current status, see [STATUS.md](STATUS.md).

---

## Quick-wins (ideas 1–15)

### 001 — SDPA tile: bump BLOCK_M on f16/bf16
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-1`

**Result:** The target kernel (`scaled_dot_product_attention.rs`) implements `mt_sdpa`, a **scalar vector decode kernel** (one Q-row per work item). There is no `BLOCK_M` constant to tweak. The tiled FlashAttention kernels (`steel_attention.rs`, `steel_attention_nax.rs`) are explicitly marked **NOT YET IMPLEMENTED** in the `#[kernel]` DSL.

**Verdict:** BLOCK_M does not exist in the target file. Idea is ill-formed against current code. Would need prerequisite tiled kernel implementation.

**File:** [ideas/001-sdpa-tile-block-m.md](ideas/001-sdpa-tile-block-m.md)

---

### 004 — SDPA-vector decode: GQA-aware K/V reuse
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-4`

**Result:** The stated mechanism (`simd_shuffle` to share K/V across Q-heads) is **physically impossible** with current dispatch. The 4 Q-heads sharing a KV head are in **4 separate threadgroups**. `simd_shuffle` only works within a simdgroup (32 lanes), not across threadgroups.

The real optimization would require: (a) dispatch `[n_kv_heads, 1, 1]` instead of `[n_q_heads, 1, 1]`, (b) partition simdgroups among Q-heads, (c) load K/V into threadgroup memory cooperatively. This is a **kernel architecture rewrite**, not a constant tweak.

**Verdict:** `simd_shuffle` can't cross threadgroups. Real fix is dispatch-shape change + cooperative tg-mem caching — Multi-day effort.

**File:** [ideas/004-sdpa-gqa-kv-reuse.md](ideas/004-sdpa-gqa-kv-reuse.md)

---

### 005–010 — Feasibility study (ideas 5 through 10)
**Status:** documented, 010 executed, 006 executed, 007 executed  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-4` (study), `../metaltile-perf-idea-10` (010), `../metaltile-perf-idea-6` (006), `../metaltile-perf-idea-7` (007)

**Results:**

| # | Idea | Verdict | Notes |
|---|------|---------|-------|
| 5 | SDPA vec8 loads | 🔴 blocked | DSL has no vector-load primitive |
| 6 | RMS-norm 4→8 unroll | ⚫ abandoned | Register pressure 9r→162r, −50% throughput. Reverted. See below. |
| 7 | Softmax simd reduce small N | 🟢 done | tpg=32 beats tpg=256 by ~1.65× on N=32. See below. |
| 8 | Softmax float4 loads | 🔴 blocked | Same as #5: no vector-load DSL |
| 9 | LayerNorm mirror #6 | ⚫ abandoned by extension | Same register pressure issue as #6 |
| 10 | GEMV tune tpg | 🟢 done | tpg=512 wins +1.8% on f16. See below. |

**File:** [ideas/005-010-feasibility-study.md](ideas/005-010-feasibility-study.md)

---

### 006 — RMS-norm: unroll 4 → 8
**Status:** ⚫ abandoned  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-6`  
**Commit:** `perf-research: idea-6 RMS-norm 8-wide unroll — abandoned`

**Result:** Expanded kernel from 4-wide to 8-wide with `tpg=512` (512×8=4096). Register pressure exploded from **9r → 162r**, occupancy dropped to **73%**, kernel became **register-limited**. Throughput regressed by **−50% to −80%** across all dtypes. Reverted to baseline immediately.

**Verdict:** 8-wide unroll holds too many live values for Apple GPU register file. The risk note was correct: "verify regs doesn't push past ~64." It pushed to 162.

**File:** [ideas/006-rms-norm-unroll-8.md](ideas/006-rms-norm-unroll-8.md)

---

### 007 — Softmax: simdgroup reduce for small N (≤32)
**Status:** 🟢 done — committed for review  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-7`  
**Commit:** `perf(softmax): add small-N bench variant (tpg=32)` — FOR REVIEW LATER

**Result:** Added `softmax_small_n` bench variant with `b=1024, n=32, tpg=32`. Compared against temporary `tpg=256` baseline on same shape. tpg=32 is **~1.65× faster** across all dtypes because tpg=256 wastes 224 idle threads and redundant second-level reduction barriers.

| dtype | tpg=32 | tpg=256 | speedup |
|-------|--------|---------|---------|
| f32 | 47.7 | 28.5 | 1.67× |
| f16 | 23.8 | 14.3 | 1.66× |
| bf16 | 23.8 | 14.4 | 1.65× |

**Verdict:** Small but genuine win for small-N softmax. Real production value is informing the dispatch heuristic: for N≤32, prefer tpg=32. The bench variant is kept as regression test.

**File:** [ideas/007-softmax-small-n.md](ideas/007-softmax-small-n.md)

---

### 010 — GEMV: tune `simd_per_tg` per K dimension
**Status:** 🟢 done — committed for review  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-10`  
**Commit:** `perf(gemv): tune tpg=256→512 for f16 GEMV (+1.8%)` — FOR REVIEW LATER

**Result:** Cloned same kernel body across `tpg={64,128,256,512,1024}`. Zero kernel-body changes. Ran bench twice for DVFS stabilization.

| dtype | best tpg | delta vs baseline | key finding |
|-------|----------|-------------------|-------------|
| f32 | 1024 | +2.5% (within noise) | Flat across all tpgs |
| **f16** | **512** | **+1.8%** | tpg=1024 regresses −20% (zero latency hiding) |
| bf16 | 128 | +1.8% (within noise) | Basically flat |

**Verdict:** tpg=512 gives a small but real f16 win. tpg=1024 is a disaster for f16. Recommended change: default tpg=256→512 in `gemv.rs`. Safe, no regressions.

**File:** [ideas/010-gemv-tpg-sweep.md](ideas/010-gemv-tpg-sweep.md)

---

## One-day / structural changes (ideas 12–20)

### 012–020 — Feasibility study
**Status:** documented  
**Investigated:** 2026-05-18  
**Commit:** `perf-research: feasibility study for ideas 12–20`

**Results:**

| # | Idea | Verdict | Why |
|---|------|---------|-----|
| 12 | all_reduce two-stage | ⚪ no-op | Already optimal — `simd_sum` + barrier + `simd_sum` confirmed by `tile inspect` |
| 13 | row-reduce pack rows | ⚠️ feasible | Dispatch-level change for small N. Not a kernel tweak |
| 14 | scan simd_prefix | ⚪ no-op | Already implemented — `simd_scan_exclusive` → `simd_prefix_exclusive_sum` |
| 15 | argmax hold 847% | ⚪ marginal | Kernel already optimal. Graph-level profiling needed |
| 16 | RoPE sin/cos tg-mem | 🔴 blocked | Needs dispatch restructuring to colocate heads by `i` |
| 17 | RoPE-QKV fusion | 🔴 blocked | Entirely new kernel + bench harness |
| 18 | KV-cache vec copy | 🔴 blocked | DSL lacks vector primitives |
| 19 | Gather tg prefetch | 🔴 blocked | Needs dispatch restructuring |
| 20 | Copy vectorize | ⚠️ feasible | Investigate `vectorize.rs` codegen pass on `mt_copy` |

**File:** [ideas/012-020-feasibility-study.md](ideas/012-020-feasibility-study.md)

---

## Commits on `dev` ready for review

| Commit | Message | Status |
|--------|---------|--------|
| `d5d1ed2` | perf-research: add tracking structure for perf-ideas.md hopper | merged |
| `e592505` | perf(gemv): tune tpg=256→512 for f16 GEMV (+1.8%) | **FOR REVIEW LATER** |
| `5aca204` | perf-research: idea-6 RMS-norm 8-wide unroll — abandoned | merged |
| `05d4037` | perf(softmax): add small-N bench variant (tpg=32) | **FOR REVIEW LATER** |
| `4265734` | perf-research: feasibility study for ideas 12–20 | merged |

---

## Patterns learned

1. **Constant-tweak Quick-wins are rare.** Most ideas in the hopper assume a constant exists or a mechanism is available. In practice, many are blocked by missing DSL primitives (vector loads) or ill-formed assumptions (BLOCK_M in a scalar kernel).

2. **Register pressure is the hidden killer.** The 8-wide RMS-norm unroll (idea 6) looked like a trivial copy-paste but destroyed performance. Always check `regs` before claiming a win.

3. **Dispatch shape matters as much as kernel body.** Ideas 4, 7, 13, 16, 19 all require or benefit from dispatch changes. The `#[bench_kernel]` macro makes this easy to test (idea 10, 7), but production dispatch is separate.

4. **Codegen is already quite good.** Ideas 12 and 14 were no-ops because `simd_sum` and `simd_prefix_exclusive_sum` were already emitted. The ideas were written before the codegen reached this state.

5. **tpg is the easiest knob to turn.** Changing `tpg` requires zero kernel edits and the bench runner handles everything. Idea 10 is the cleanest win in the whole set.
