# Research Log — What We've Investigated & Results

> Living document. One entry per investigated idea. Updated after each cycle.  
> For methodology, see [TEMPLATE.md](ideas/TEMPLATE.md). For current status, see [STATUS.md](STATUS.md).

---

## Quick-wins (ideas 1–15)

### 001 — SDPA tile: bump BLOCK_M on f16/bf16
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-1`

**Result:** The target kernel (`scaled_dot_product_attention.rs`) implements `mt_sdpa`, a **scalar vector decode kernel**. There is no `BLOCK_M` constant. Tiled FlashAttention kernels are **NOT YET IMPLEMENTED** in the DSL.

**File:** [ideas/001-sdpa-tile-block-m.md](ideas/001-sdpa-tile-block-m.md)

---

### 002 — SDPA: BLOCK_N 64 → 128 on D=128
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** Same target as #1. The `mt_sdpa` kernel is scalar vector, not tiled. No `BLOCK_N` constant exists. MLX `sdpa_vector.h` has `BN=32`, but MetalTile doesn't have a `#[kernel]` port.

**File:** [ideas/002-sdpa-block-n.md](ideas/002-sdpa-block-n.md)

---

### 003 — SDPA: split-K for low-occupancy H=8 shapes
**Status:** 🔴 blocked / needs re-scope  
**Investigated:** 2026-05-18

**Result:** Requires new split-K kernel variant + merge kernel + dispatcher changes. The `run_attention` arm dispatches `[h, 1, 1]` threadgroups. Split-K needs `[h, k, 1]` or similar. Effort: One-day to Multi-day.

**File:** [ideas/003-sdpa-split-k.md](ideas/003-sdpa-split-k.md)

---

### 004 — SDPA-vector decode: GQA-aware K/V reuse
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-4`

**Result:** `simd_shuffle` can't cross threadgroups. Real fix is dispatch restructuring + cooperative tg-mem caching — Multi-day effort.

**File:** [ideas/004-sdpa-gqa-kv-reuse.md](ideas/004-sdpa-gqa-kv-reuse.md)

---

### 005 — SDPA-vector: 8-wide vectorized loads on f16/bf16
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** DSL `load()` is scalar. No `half8` or `load_vec8<T>()` exists. Metal driver may auto-vectorize, but we can't verify or force it. Needs DSL extension.

**File:** [ideas/005-sdpa-vec8-loads.md](ideas/005-sdpa-vec8-loads.md)

---

### 006 — RMS-norm: unroll 4 → 8
**Status:** ⚫ abandoned  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-6`  
**Commit:** `perf-research: idea-6 RMS-norm 8-wide unroll — abandoned`

**Result:** Register pressure exploded **9r → 162r**, occupancy dropped to **73%**, kernel became register-limited. Throughput regressed **−50% to −80%**. Reverted.

**File:** [ideas/006-rms-norm-unroll-8.md](ideas/006-rms-norm-unroll-8.md)

---

### 007 — Softmax: simdgroup reduce for small N (≤32)
**Status:** 🟢 done — committed for review  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-7`  
**Commit:** `perf(softmax): add small-N bench variant (tpg=32)` — FOR REVIEW LATER

**Result:** tpg=32 is **~1.65× faster** than tpg=256 for N=32. Eliminates 224 idle threads + redundant second-level reduction barriers.

| dtype | tpg=32 | tpg=256 | speedup |
|-------|--------|---------|---------|
| f32 | 47.7 | 28.5 | 1.67× |
| f16 | 23.8 | 14.3 | 1.66× |
| bf16 | 23.8 | 14.4 | 1.65× |

**File:** [ideas/007-softmax-small-n.md](ideas/007-softmax-small-n.md)

---

### 008 — Softmax: float4 loads on f16/bf16 inner loop
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** Same blocker as #5: DSL has no vector-load primitive. `load()` is scalar; `float4`/`half4` loads require DSL extension or raw MSL.

**File:** [ideas/008-softmax-float4-loads.md](ideas/008-softmax-float4-loads.md)

---

### 009 — LayerNorm: mirror RMS-norm tweaks
**Status:** ⚫ abandoned by extension  
**Investigated:** 2026-05-18

**Result:** LayerNorm has **more** live state than RMS-norm (two accumulators `s` + `sq`, plus weight + bias). After idea #6 proved 8-wide is catastrophic (9r→162r), this would be worse. Not worth benching.

**File:** [ideas/009-layernorm-mirror-rms.md](ideas/009-layernorm-mirror-rms.md)

---

### 010 — GEMV: tune `simd_per_tg` per K dimension
**Status:** 🟢 done — committed for review  
**Investigated:** 2026-05-18  
**Worktree:** `../metaltile-perf-idea-10`  
**Commit:** `perf(gemv): tune tpg=256→512 for f16 GEMV (+1.8%)` — FOR REVIEW LATER

**Result:** tpg=512 wins for f16 (+1.8%). tpg=1024 regresses −20% on f16 (zero latency hiding). f32/bf16 flat.

**File:** [ideas/010-gemv-tpg-sweep.md](ideas/010-gemv-tpg-sweep.md)

---

### 011 — GEMV-masked: dense fallback above 50% density
**Status:** 🔴 blocked / dispatcher-level  
**Investigated:** 2026-05-18

**Result:** Requires runtime mask density inspection + kernel routing. The `#[bench_kernel]` macro generates static specs; no runtime selection exists. Also, masked kernel uses 1-wide scalar loop vs dense kernel's 4-wide unroll — the gap is structural, not just the mask load.

**File:** [ideas/011-gemv-masked-dense-fallback.md](ideas/011-gemv-masked-dense-fallback.md)

---

## One-day / structural changes (ideas 12–20)

### 012 — all_reduce: two-stage simd→threadgroup
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** Codegen already emits `simd_sum` + `threadgroup_barrier` + `simd_sum`. `tile inspect mt_all_reduce` confirms. The idea was written before codegen reached this state.

---

### 013 — row-reduce: rows-per-threadgroup when N is small
**Status:** ⚠️ feasible  
**Investigated:** 2026-05-18

**Result:** Dispatch-level change. Current bench only tests `N=4096`. For `N<256`, packing multiple rows per tg would improve occupancy. Requires modifying `run_spec.rs` or macro `DispatchGrid` logic.

---

### 014 — scan: prefer `simd_prefix_inclusive_sum`
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** Already implemented. `simd_scan_exclusive` in DSL maps to `simd_prefix_exclusive_sum` in MSL. `tile inspect mt_scan` confirms.

---

### 015 — argmax: refuse to slow down 847%
**Status:** ⚪ marginal  
**Investigated:** 2026-05-18

**Result:** Kernel is already register-light. The 847% figure is structural (vs MLX's scalar tree reduction). "Lowering regs for fused graphs" requires graph-level profiling, not single-kernel bench.

---

### 016–020 — Feasibility study
**Status:** documented  
**Investigated:** 2026-05-18  
**Commit:** `perf-research: feasibility study for ideas 12–20`

| # | Idea | Verdict | Why |
|---|------|---------|-----|
| 16 | RoPE sin/cos tg-mem | 🔴 blocked | Dispatch restructuring needed |
| 17 | RoPE-QKV fusion | 🔴 blocked | New kernel + bench harness |
| 18 | KV-cache vec copy | 🔴 blocked | DSL lacks vector primitives |
| 19 | Gather tg prefetch | 🔴 blocked | Dispatch restructuring needed |
| 20 | Copy vectorize | ⚠️ feasible | Investigate `vectorize.rs` pass |

**File:** [ideas/012-020-feasibility-study.md](ideas/012-020-feasibility-study.md)

---

### 021 — FP4 dequant: packed bit ops
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** The target kernel (`mt_fp4_quant_dequant`) is a **scalar quantize-dequantize roundtrip** on `Tensor<f32>`, not a packed FP4 dequantization kernel. It uses 7 nested `select` statements to map a float to one of 8 FP4 levels, then immediately rescales back to `f32`. There is no packed `uint32`/`uint8` load, no bit shuffle, no LUT, and no `half` type usage.

The hypothesis describes a **weight-dequantization** optimization (8 FP4 values packed in a 32-bit word → 8 `half` values via LUT), which is a common quantized-GEMV/GEMM pattern. That operation does not exist in `mlx/fp_quantized.rs`. Implementing it would require a new kernel + new bench harness (packed I/O contract), making this a multi-day effort rather than a single-file tweak.

**File:** [ideas/021-fp4-dequant-packed-bit-ops.md](ideas/021-fp4-dequant-packed-bit-ops.md)

---

### 022 — Quantized int4 GEMV: simdgroup_matrix multiply
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** The hypothesis proposes applying `simdgroup_matrix_multiply` (a GEMM primitive) to a **GEMV** kernel. The current `dequant_gemv_int4` kernel is `KernelMode::Reduction`, dispatching `[m, 1, 1]` with one threadgroup per output row. It does scalar dequant→FMA, accumulating with `reduce_sum(acc)`.

**Why this doesn't work:** GEMV is matrix-vector (`W[M,K] × x[K,1]`). The "N" dimension is 1. `simdgroup_matrix_multiply` computes `C += A × B` where all operands are matrix tiles (e.g., 8×8, 16×8). To use it for GEMV, one would need to pad the vector to a K×8 tile, compute an M×8 result, and discard 7/8 of the output — architecturally wasteful.

**MLX reference check:** MLX's actual quantized GEMV kernels (`qmv_fast_impl`, `qmv_impl` in `quantized.h`) do **not** use `simdgroup_matrix_multiply`. They use scalar dequantization + `simd_sum`, exactly like MetalTile's current approach. The only MLX kernels using simdgroup matmul are steel GEMM and convolution.

**Adjacent but different idea:** MLX processes **8 rows per threadgroup** (`num_simdgroups=2 × results_per_simdgroup=4`), which improves occupancy. That is a dispatch-level change (⚠️ feasible, multi-day), not the simdgroup-matmul optimization hypothesized here.

**File:** [ideas/022-dequant-gemv-simdgroup-matmul.md](ideas/022-dequant-gemv-simdgroup-matmul.md)

---

### 023 — Quantized GEMV: int4 pack-of-2 lookup
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** The hypothesis proposes a 256-entry `half2` LUT for dequantizing pairs of int4 values with a single load. The target file (`mlx/quantized.rs`) contains scalar GEMV and dequantize kernels that extract int4 values via shift+mask, then apply per-group `scale * q + bias`.

**Blockers:**
1. **No `half2` vector type in user-facing DSL.** `load()` is scalar. `VectorLoad` exists at the IR level but is codegen-generated, not author-writable.
2. **No constant-array / LUT primitive.** `threadgroup_alloc` creates buffers but cannot be initialized with compile-time data. Runtime init costs stores + barrier.
3. **Per-group scale/bias dependency.** A LUT storing pre-dequantized values would need to encode scale/bias per group. For `k=4096, group_size=64`, that's 64 groups per row. A 256-entry `f16` LUT per group is `64 × 256 × 2 = 32 KB` — exactly the threadgroup memory limit. A `f32` LUT exceeds it. Rebuilding per group adds 64 barriers per threadgroup.
4. **MLX doesn't use this.** `qmv_fast_impl` uses scalar dequant + `simd_sum`, confirming this is not a standard optimization path.

**File:** [ideas/023-quantized-int4-pack-of-2-lookup.md](ideas/023-quantized-int4-pack-of-2-lookup.md)

---

### 024 — dequant_gather: skip dequant for cold misses
**Status:** ⚪ no-op / premature  
**Investigated:** 2026-05-18

**Result:** The hypothesis proposes profiling L1 cache misses for `dequant_gather` and skipping dequantization on cold misses. The target kernel is a quantized embedding-table gather: each thread loads a token index, fetches 1–2 `u32` words from packed weights, extracts a quantized value, loads scale/bias, dequantizes, and stores.

**Findings:**
1. **`tile profile` does not exist.** The CLI has `bench`, `build`, `inspect`, `device`, `snap`, `diff` — no profiling or counter-sampling command.
2. **No cache-state visibility in MSL.** Metal does not expose L1 hit/miss counters or cache-state predicates to shader code. A kernel cannot branch on "is this data in cache?"
3. **Kernel is already memory-bound.** The dequantization is 2 FMAs; the bottleneck is 4–5 device-memory loads per thread. Skipping dequant would not materially reduce memory traffic.
4. **Real fix is graph-level.** A dispatcher-level dequantized-embedding cache (keep hot tokens unpacked in device memory) is the correct architecture, but requires graph-scheduler support — not a kernel tweak.

**File:** [ideas/024-dequant-gather-cold-miss-skip.md](ideas/024-dequant-gather-cold-miss-skip.md)

---

### 025 — Sort: 4-way bitonic merge
**Status:** ⚠️ feasible / high risk  
**Investigated:** 2026-05-18

**Result:** The target kernel `mt_sort` implements bitonic sort for 1024 elements with `tpg=256`, each thread handling 4 elements. Baseline: **117r, thread-limited, 76–77% of MLX**.

**Analysis:**
- A 4-way bitonic merge would fuse 4 compare-swaps into one 8-element merge, halving tg memory ops for stages where the partner block is contiguous (distance ≥ 4).
- **Register risk:** The kernel is already at 117r. Holding 8 live scalars (4 local + 4 partner) plus merge temporaries would likely push usage past the spill threshold (~128r). Idea #6 showed register explosions can destroy performance.
- **Algorithm mismatch:** MLX uses **merge sort** (`BlockMergeSort`), not bitonic sort. The 23% performance gap is structural, not stride-related. A merge-sort port would be a separate multi-day idea.
- **DSL limitation:** No register-array type; 8 scalars require 8 separate variables, each consuming a physical register slot.

**Decision:** Technically feasible but high risk and low expected value. The real optimization is algorithmic (bitonic → merge sort), not stride width.

**File:** [ideas/025-sort-4-way-bitonic-merge.md](ideas/025-sort-4-way-bitonic-merge.md)

---

### 026 — Sampling: radix-select top-k
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** The target file `sampling.rs` contains `softmax_categorical_sample` — a softmax + inverse-CDF random-sampling kernel, not a top-k kernel. The idea assumes "current top-k probably sorts then slices," but no top-k implementation exists in MetalTile.

**Findings:**
1. **Target mismatch** — `sampling.rs` has categorical sampling, not top-k.
2. **Bench exists but wrong op** — `tile bench -f sampling` runs `softmax_categorical_sample`, which is already registered.
3. **No MLX top-k kernel** — MLX does not ship a dedicated top-k Metal kernel. Top-k is likely implemented at the framework level.
4. **New kernel + bench harness required** — radix-select top-k would need a new `#[kernel]`, new `run_spec.rs` arm, and CPU reference for correctness.

**File:** [ideas/026-sampling-radix-select-topk.md](ideas/026-sampling-radix-select-topk.md)

---

### 027 — SSM: scan with state vectorization
**Status:** ⚪ no-op / already implemented  
**Investigated:** 2026-05-18

**Result:** The target file `ssm.rs` contains three kernels. The `ssm_step` kernel does a serial loop over `state_dim` per `(head, d)` pair: `new_h = decay * h_old + dt * b[n] * x`, then `y_d += c[n] * new_h`. The `mt_ssm_step` kernel already parallelizes this across 32 threads with `simd_sum`.

**Key finding:** This is **not a scan** — each state dimension `n` is independent. `h_old` is the previous token's state for slot `(head, n, d)`, not the previous `n` in the loop. The state update and dot-product accumulation are already fused in the same loop body.

**Conclusion:** `mt_ssm_step` is exactly the vectorized variant the hypothesis describes. The idea was written before `mt_ssm_step` existed.

**File:** [ideas/027-ssm-scan-state-vectorization.md](ideas/027-ssm-scan-state-vectorization.md)

---

## Codegen pass assessments (ideas 41–45)

### 041 — `schedule.rs`: software pipelining
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** `schedule.rs` is a tile-dimension annotation pass for `Op::Dot`, not a loop scheduler. Software pipelining would require a completely new codegen pass. [Details](ideas/041-schedule-software-pipelining.md)

---

### 042 — `licm.rs`: hoist gather indices when loop-invariant
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** LICM already hoists loop-invariant `Load` ops from read-only params. `tensor[constant_idx]` maps to `Op::Load`, which is covered. [Details](ideas/042-licm-hoist-gather-indices.md)

---

### 043 — `cse.rs`: extend across simdgroup boundaries
**Status:** ⚠️ feasible (needs re-scoping)  
**Investigated:** 2026-05-18

**Result:** CSE is strictly block-local. Cross-branch CSE (e.g., common subexpressions in both arms of `Op::If`) is unimplemented and would be a real win. The "simdgroup boundary" framing is imprecise against the IR. [Details](ideas/043-cse-across-simdgroup-boundaries.md)

---

### 044 — `if_conversion.rs`: predicate tiny ifs in inner loops
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** The pass already predicates Diamond-shaped `If` blocks. `gemv_masked.rs` has no `If` in its DSL source — the mask is applied unconditionally via scalar multiply — so there is nothing to convert. [Details](ideas/044-if-conversion-predicate-tiny-ifs.md)

---

### 045 — `value_sink.rs`: sink threadgroup-memory stores
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** `value_sink.rs` explicitly excludes side-effecting ops, including `Op::ThreadgroupStore`. Moving threadgroup stores is unsafe without alias analysis and barrier reasoning. The hypothesized register benefit is also moot — threadgroup stores don't hold registers. [Details](ideas/045-value-sink-threadgroup-stores.md)

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

1. **Constant-tweak Quick-wins are rare.** Most ideas assume a constant exists or a mechanism is available. Many are blocked by missing DSL primitives (vector loads) or ill-formed assumptions (BLOCK_M in a scalar kernel).

2. **Register pressure is the hidden killer.** The 8-wide RMS-norm unroll (idea 6) looked like trivial copy-paste but destroyed performance. Always check `regs` before claiming a win.

3. **Dispatch shape matters as much as kernel body.** Ideas 4, 7, 13, 16, 19 all require or benefit from dispatch changes. The `#[bench_kernel]` macro makes this easy to test.

4. **Codegen is already quite good.** Ideas 12 and 14 were no-ops because `simd_sum` and `simd_prefix_exclusive_sum` were already emitted.

5. **tpg is the easiest knob to turn.** Changing `tpg` requires zero kernel edits. Ideas 7 and 10 are the cleanest wins.
