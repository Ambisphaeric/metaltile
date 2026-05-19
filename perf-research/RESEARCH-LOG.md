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

### 028 — logsumexp: fuse max + sum-exp
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** The target kernel `mt_logsumexp` is already a single-pass online logsumexp implementation. It uses the running-update trick: `nz = nz * exp(pm - nm) + sum(exp(vi - nm))` where `nm = max(pm, cm)`.

**MLX reference:** MLX has `logsumexp` (two-pass) and `logsumexp_looped` (one-pass). MetalTile matches `logsumexp_looped` and is already faster: **MT%=154% (f32), 238% (f16), 237% (bf16)** at 54r.

**Conclusion:** The optimization is already implemented.

**File:** [ideas/028-logsumexp-fuse-max-sum-exp.md](ideas/028-logsumexp-fuse-max-sum-exp.md)

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

## One-day structural changes (ideas 29–35)

### 029 — Short-row cooperative groups
**Status:** ⚠️ feasible  
**Investigated:** 2026-05-18

**Result:** `strided_reduce` + `reduce_sum` already uses `simd_sum` + `threadgroup_barrier`. For N≤32, dispatching tpg=256 wastes 224 threads. Same pattern as #007: smaller tpg eliminates idle lanes and redundant barriers. [Details](ideas/029-short-row-cooperative-groups.md)

---

### 030 — `binary_two`: FMA autovec diagnostic
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** `mt_binary_two` computes `x+y` and `x*y` independently to two separate outputs. There is no `a*b+c` pattern, so FMA cannot emit. Kernel is bandwidth-bound regardless. [Details](ideas/030-binary-two-fma-autovec.md)

---

### 031 — Unary: emit `metal::precise::sigmoid` directly
**Status:** ⚠️ feasible  
**Investigated:** 2026-05-18

**Result:** `mt_sigmoid` manually expands `1/(1+exp(-x))`. The DSL already has a `sigmoid()` builtin (used by `silu`). Switching to the builtin is a one-line cleanup — likely more accurate and smaller MSL. Other unary kernels already use builtins. [Details](ideas/031-unary-precise-intrinsics.md)

---

### 032 — SwiGLU/GELU: fuse with downstream matmul write
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** No GEMM kernel exists in the DSL. Epilogue fusion requires both a GEMM implementation and a fused-emitter codegen pass. Multi-day to project-scale effort. [Details](ideas/032-swiglu-gelu-fuse-matmul-epilogue.md)

---

### 033 — argmin variant in arg_reduce
**Status:** ⚠️ feasible  
**Investigated:** 2026-05-18

**Result:** Both `mlx/arg_reduce.rs` and `ffai/arg_reduce.rs` have argmax only. Argmin is a copy-paste with `neg_infinity()` → `infinity()` and `>` → `<`. ~30 lines total. [Details](ideas/033-argmin-variant.md)

---

### 034 — softmax + attention epilogue fusion
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** MetalTile has no tiled attention kernel (only scalar vector decode). Fusing softmax with matmul(V) is the core of FlashAttention — a moonshot-level item, not a one-day tweak. [Details](ideas/034-softmax-attention-fusion.md)

---

### 035 — random: 64-bit state / vec4 generation
**Status:** 🔴 blocked / ill-formed  
**Investigated:** 2026-05-18

**Result:** `mt_random_hash` is a toy hash (`gid + 1` → 3 xorshifts), not a PRNG. No state to widen, no constants to amortize. Hypothesis describes a different kernel entirely. [Details](ideas/035-random-xorshift-vec4.md)

---

## Codegen pass assessments (ideas 36–40)

### 036 — `vectorize.rs`: 8-wide on f16/bf16
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** `MAX_VEC_LEN = 8` is already hardcoded. BF16 is already in the `is_vectorizable` set. The MSL emitter decomposes `float8`/`half8` into `float2x4` when native 8-wide is unavailable. The hypothesis predates the CODEGEN_OVERHAUL upgrade. [Details](ideas/036-vectorize-8-wide-f16-bf16.md)

---

### 037 — `vectorize.rs`: detect strided-but-aligned stores
**Status:** ⚠️ feasible (needs re-scoping)  
**Investigated:** 2026-05-18

**Result:** The pass coalesces contiguous single-buffer accesses (`src[base+k]`). Strided/interleaved patterns (e.g., `store(c[i], v0); store(d[i], v1)`) are not handled. This is a genuine gap but requires SLP/loop-level vectorization, not just load-store coalescing. [Details](ideas/037-vectorize-strided-aligned-stores.md)

---

### 038 — `fusion.rs`: epilogue fusion onto reductions
**Status:** ⚠️ feasible (marginal)  
**Investigated:** 2026-05-18

**Result:** The pass already fuses post-reduction elementwise chains into `FusedElementwise`. Since MetalTile kernels are single-dispatch, there is no separate "reduction kernel" to fuse into — the elementwise ops are already in the same dispatch. Extending `is_fusible` to include `Reduce` would be a small change with limited benefit. [Details](ideas/038-fusion-epilogue-reductions.md)

---

### 039 — `fusion.rs`: multi-reduction in one pass
**Status:** ⚠️ feasible (needs re-scoping)  
**Investigated:** 2026-05-18

**Result:** Multi-reduction fusion (e.g., computing mean and variance in one loop) is **loop fusion**, not operator fusion. The current `fusion.rs` merges expression trees within a block. A new `loop_fusion.rs` pass would be needed, or LayerNorm can be written as a hand-written kernel. [Details](ideas/039-fusion-multi-reduction.md)

---

### 040 — `unroll.rs`: register-pressure-aware unroll count
**Status:** ⚠️ feasible (high value)  
**Investigated:** 2026-05-18

**Result:** `UnrollPass` uses a fixed `factor` (default 4, max 8) with no register check. `register_estimate.rs` already exists but is not consulted by the unroller. Connecting them would have prevented idea #006's register explosion (9r → 162r). Partial unrolling (unroll by factor even when trip_count > factor) is also missing. [Details](ideas/040-unroll-register-aware.md)

---

## Runtime / build / CLI assessments (ideas 46–55)

### 046 — Wire the autotuner `lookup()`
**Status:** ⚠️ feasible (medium effort)  
**Investigated:** 2026-05-18

**Result:** `TuneCache::lookup()` is a placeholder returning `None`. The cache infrastructure (save/load, `TuneEntry`, `ShapeBucket`) exists but lookup doesn't bucket `ConstExprValues`. Plumbing it into `Context::dispatch` would unlock schedule selection. [Details](ideas/046-autotuner-lookup.md)

---

### 047 — PSO disk cache
**Status:** ⚠️ feasible (needs re-scoping)  
**Investigated:** 2026-05-18

**Result:** `PSO_CACHE` in `context.rs` is in-memory only (per-process). Metal `MTLComputePipelineState` cannot be serialized directly. Runtime `.metallib` caching is possible via `MTLDynamicLibrary` (Metal 3.1+). [Details](ideas/047-pso-disk-cache.md)

---

### 048 — Heap-backed buffer pool
**Status:** ⚪ no-op  
**Investigated:** 2026-05-18

**Result:** `BUF_POOL` in `context.rs` already caches `MTLBuffer` objects by `(next_power_of_two(len), storage_mode)`. Functionally equivalent to a heap allocator for MetalTile's use case. [Details](ideas/048-heap-backed-buffer-pool.md)

---

### 049 — Reuse command buffer across bench iterations
**Status:** ⚠️ feasible (changes semantics)  
**Investigated:** 2026-05-18

**Result:** `runner.rs` `measure()` creates a fresh `MTLCommandBuffer` per pass. Reusing one buffer for all warmups+samples reduces driver overhead but requires `MTLCounterSampleBuffer` for per-dispatch timing. [Details](ideas/049-reuse-command-buffer-bench.md)

---

### 050 — Fast-math + disable shader-validation in release
**Status:** ⚠️ feasible (small)  
**Investigated:** 2026-05-18

**Result:** `MTLCompileOptions` uses defaults in both `context.rs` and `runner.rs`. Setting `mathMode = Fast` and `languageVersion = Metal3_1` is a two-line change. Shader validation is already off outside Xcode. [Details](ideas/050-fast-math-shader-validation.md)

---

### 051 — Bench: pipelined sample collection
**Status:** ⚠️ feasible (medium effort)  
**Investigated:** 2026-05-18

**Result:** `measure()` does serial warmup + samples with `waitUntilCompleted` per pass. Encoding all into one command buffer requires `MTLCounterSampleBuffer` for per-dispatch timestamps. [Details](ideas/051-pipelined-sample-collection.md)

---

### 052 — Persistent threadgroups
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** Metal has no persistent threadgroup or work-queue API. `dispatch_chain` in `context.rs` already dispatches multiple kernels through a single command buffer, achieving most of the practical benefit. [Details](ideas/052-persistent-threadgroups.md)

---

### 053 — CLI: parallelize per-kernel benches
**Status:** ⚠️ feasible (risky)  
**Investigated:** 2026-05-18

**Result:** Serial bench flow could be parallelized with multiple queues, but DVFS pollution and SLC cache interference make results less reliable. Marginal wall-time win. [Details](ideas/053-parallel-bench-per-kernel.md)

---

### 054 — CLI: `tile bench --compare-against <baseline.json>`
**Status:** ⚠️ feasible (small UX)  
**Investigated:** 2026-05-18

**Result:** JSON save/load already exists. Inline diff against a baseline is a thin UX layer — one-day effort. [Details](ideas/054-bench-compare-against-baseline.md)

---

### 055 — Build: precompile `.metallib` per Apple GPU family
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** MetalTile generates MSL at runtime via `MslGenerator`. Build-time pre-compilation is infeasible due to JIT MSL + shape/dtype/fn_const combinatorics. [Details](ideas/055-precompile-metallib-gpu-family.md)

---

## Moonshot assessments (M1–M2)

### M1 — ML-driven autotuner
**Status:** ⚠️ feasible (project-scale)  
**Investigated:** 2026-05-18

**Result:** `TuneCache`, feature extraction (`compute_profiles`), and bench harness all exist. The missing pieces are: (1) exhaustive data collection (grid search doesn't exist yet — see #046), (2) model training, (3) wiring prediction into `lookup()`. Phase 1 should be "implement grid search + populate cache"; Phase 2 is "replace search with learned model". [Details](ideas/m1-ml-driven-autotuner.md)

---

### M2 — AMX / ANE offload for small-batch f16 GEMM
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** AMX has no public API — access is via private `libsystem_m.dylib` symbols or reverse-engineered bindings. CoreML/ANE has massive model-compilation overhead that dominates small-batch GEMM. MetalTile is GPU-first; MLX (Apple's own framework) also uses Metal exclusively. [Details](ideas/m2-amx-ane-offload.md)

---

### M3 — Persistent-kernel graph capture
**Status:** 🔴 blocked  
**Investigated:** 2026-05-18

**Result:** Metal has no graph capture API (unlike CUDA `cudaGraph`). `dispatch_chain` in `context.rs` already dispatches multiple kernels through a single command buffer with private intermediate buffers — eliminating most per-dispatch overhead. A persistent mega-kernel would require cross-kernel MSL fusion (M4) plus a device-memory work queue. [Details](ideas/m3-persistent-kernel-graph-capture.md)

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
