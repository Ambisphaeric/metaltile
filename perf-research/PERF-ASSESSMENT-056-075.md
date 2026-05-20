# MetalTile Performance Assessment — Ideas 056-075

> Branch: `dev-perf` · Generated: 2026-05-20
>
> Companion to `perf-ideas.md` (ideas 001-055) and `STATUS.md` (M1-M10 moonshots).
> These 20 new areas were identified by gap-analysis across the `crates/` tree,
> focusing on: (a) kernels still stubbed or unimplemented in the DSL, (b) codegen
> passes with untapped potential, and (c) runtime dispatch overhead not yet modeled.
> Each entry includes a feasibility assessment and a concrete bench / inspect path.

---

## Quick Reference

| # | Name | Category | Feasibility | Impact | Status |
|---|------|----------|-------------|--------|--------|
| 056 | Steel Attention prefill via simdgroup matmul | Kernel | 🔴 Blocked | **Critical** | not-started |
| 057 | Steel GEMM Split-K accumulator fusion | Kernel | ⚠️ Feasible | High | not-started |
| 058 | Steel GEMM Gather: indirect row indexing | Kernel | 🔴 Blocked | Medium | not-started |
| 059 | Steel GEMM Masked: block-level predication | Kernel | 🔴 Blocked | High | not-started |
| 060 | Steel GEMM Segmented: variable-K batched | Kernel | 🔴 Blocked | Medium | not-started |
| 061 | FFT radix-4/8 Cooley-Tukey DSL | Kernel | 🔴 Blocked | Medium | not-started |
| 062 | Conv2D Winograd F(2×2,3×3) + steel GEMM | Kernel | 🔴 Blocked | High | not-started |
| 063 | Ternary select: vectorized condition loads | Kernel | 🟢 Feasible | Low | not-started |
| 064 | Strided copy: auto-vectorize inner stride==1 | Codegen | 🟢 Feasible | Medium | not-started |
| 065 | Binary ops: fused binary→unary chain | Codegen | ⚠️ Feasible | Medium | not-started |
| 066 | Arange: function-constant start/step | Kernel | 🟢 Feasible | Low | not-started |
| 067 | RoPE: merge MLX vs FFAI dispatch heuristic | Kernel | 🟢 Feasible | Medium | not-started |
| 068 | Memory fence + atomic barrier DSL primitives | Kernel | 🔴 Blocked | Medium | not-started |
| 069 | Tile lowering: `GpuFamily`-aware dynamic schedule | Codegen | ⚠️ Feasible | Medium | not-started |
| 070 | Occupancy: model threadgroup bank conflicts | Codegen | ⚠️ Feasible | Medium | not-started |
| 071 | `dispatch_chain`: zero-copy in-place alias tracking | Runtime | ⚠️ Feasible | High | not-started |
| 072 | Resident buffer heap suballocation | Runtime | 🟢 Feasible | Medium | not-started |
| 073 | SLC flush: right-size scratch to actual SLC | Runtime | 🟢 Feasible | Low | not-started |
| 074 | Algebraic simplify: `pow(x,2)` → `x*x`, `sqrt(x*x)` → `abs(x)` | Codegen | 🟢 Feasible | Low | not-started |
| 075 | Bench runner: multi-variant encoding in one CB | Runtime | ⚠️ Feasible | Medium | not-started |

---

## 056 — Steel Attention prefill via simdgroup matmul

**Category:** Kernel · **Impact:** Critical (closes 32% → ~100% prefill gap)

### Hypothesis
`steel_attention.rs` is currently a stub — the DSL `Op::FlashAttention` lowers to an error placeholder. The scalar SDPA prefill kernel (`sdpa_prefill`) achieves only **32% of MLX** on M1 Max and **40%** on M5 Max. MLX's prefill uses `steel_attention` with simdgroup matrix ops for the Q×K^T and P×V products inside tiled attention blocks, plus online softmax across K tiles. Implementing this in the DSL would close the single largest perf regression in the entire bench suite.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/steel/attn/steel_attention.rs` (stub)
- **Codegen:** `crates/metaltile-codegen/src/passes/tile_lowering.rs` — `Op::FlashAttention` lowering
- **MSL emit:** `crates/metaltile-codegen/src/msl/fused.rs` — simdgroup matmul emission for attention tiles

### Measure
`tile bench -vv -f sdpa_prefill` — headline shape `B=1 H=32 T=512/512 D=128 gqa=4 bf16`

### Risk / Blockers
1. DSL has no `simdgroup_matrix_multiply` primitive (blocked on MSL emit support for 16×16×16 tiles).
2. Online softmax across K tiles requires a running `max` + `sum-exp` update in the tile loop — the DSL reduction model assumes one pass, not an iterative tile loop.
3. Mask application (causal / padding) inside the tile loop needs predicated simdgroup stores.
4. **Verdict:** Multi-week project. The structural blocker is `simdgroup_matrix` support in the DSL → MSL pipeline. Defer until steel GEMM (`steel_gemm_fused.rs`) is fully working end-to-end, then generalize.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 1/5 | `simdgroup_matrix` not in DSL |
| Bench testability | 2/5 | Would need new bench harness entry |
| Scope | 5/5 | Touches codegen + kernel + runner |
| Risk of regression | 2/5 | New kernel family, isolated from existing SDPA |
| **Overall** | **🔴 Blocked** | Unblock after steel GEMM matures |

---

## 057 — Steel GEMM Split-K accumulator fusion

**Category:** Kernel · **Impact:** High (eliminates second dispatch)

### Hypothesis
`steel_gemm_splitk.rs` requires a two-kernel dispatch: (1) partial sums across K splits, (2) accumulator reduction. For small split counts (K ≤ 4 splits), the accumulator can be fused into the tail of the first kernel — each threadgroup reduces its own partials before writing the final output tile. This eliminates the second kernel launch and temporary buffer, cutting dispatch overhead by ~30-50 µs per GEMM.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/steel/gemm/steel_gemm_splitk.rs`
- **Dispatch:** `crates/metaltile-std/src/run_spec.rs` — `run_gemm` split-K path

### Measure
`tile bench -vv -f gemm_splitk` (bench entry does not yet exist — would need to be added)

### Risk / Blockers
1. Large split counts (> 4) still need a separate reduction kernel to keep register pressure bounded.
2. The accumulator kernel in MLX supports α·X + β·Y fusion — our fused version must preserve that.
3. **Verdict:** Feasible for the small-split fast path. Scope is ~1 week.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 3/5 | steel GEMM must be working first |
| Bench testability | 2/5 | No split-K bench entry yet |
| Scope | 3/5 | One kernel + dispatch tweak |
| Risk of regression | 3/5 | Only affects split-K path |
| **Overall** | **⚠️ Feasible** | Wait for steel GEMM baseline |

---

## 058 — Steel GEMM Gather: indirect row indexing in tiled loads

**Category:** Kernel · **Impact:** Medium (embedding-table / sparse matmul)

### Hypothesis
`steel_gemm_gather.rs` implements GEMM where one operand is accessed via a gather index buffer (e.g., embedding lookup → matmul). The simdgroup tile loader needs to stage non-contiguous rows into threadgroup memory via indirect indices. Currently the DSL has no gather primitive compatible with tiled matmul staging.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/steel/gemm/steel_gemm_gather.rs`

### Measure
`tile bench -vv -f gather_gemm` (new bench entry)

### Risk / Blockers
1. Indirect indexing breaks vectorized loads — each row may be misaligned relative to tile boundaries.
2. Threadgroup staging with indices requires `threadgroup_array[idx[i]] = src[row[i]]` — divergence on load.
3. **Verdict:** Blocked on DSL gather/scatter primitive + alignment handling.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 1/5 | No gather primitive in DSL |
| Bench testability | 2/5 | New harness needed |
| Scope | 4/5 | New kernel type + codegen primitive |
| Risk of regression | 2/5 | Isolated to gather path |
| **Overall** | **🔴 Blocked** | Depends on gather primitive landing |

---

## 059 — Steel GEMM Masked: block-level output skip predication

**Category:** Kernel · **Impact:** High (sparse GEMM, MoE routing)

### Hypothesis
`steel_gemm_masked.rs` skips output blocks and/or operand blocks based on a block mask. When sparsity > 50%, early-exiting threadgroups before tile loads saves significant DRAM bandwidth. MLX uses this for MoE expert routing where only a subset of output tiles are active.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/steel/gemm/steel_gemm_masked.rs`

### Measure
`tile bench -vv -f gemm_masked` with synthetic 50% and 90% sparse masks

### Risk / Blockers
1. DSL has no block-level conditional dispatch — `if` inside a kernel is per-thread, not per-threadgroup.
2. Early-exit must be warp-uniform (all threads in a simdgroup agree) or divergence costs dominate.
3. **Verdict:** Blocked on per-threadgroup predicate primitive in DSL.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 1/5 | No block-level predication |
| Bench testability | 3/5 | Can use synthetic masks |
| Scope | 4/5 | New kernel + dispatch primitive |
| Risk of regression | 2/5 | Isolated path |
| **Overall** | **🔴 Blocked** | Needs DSL extension |

---

## 060 — Steel GEMM Segmented: variable-K batched dispatch

**Category:** Kernel · **Impact:** Medium (ragged batches, e.g., variable-length sequences)

### Hypothesis
`steel_gemm_segmented.rs` supports batched GEMM where each batch segment has a different K extent (ragged K). A segment descriptor buffer stores per-batch K offsets. This is essential for grouped-query attention with variable context lengths.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/steel/gemm/steel_gemm_segmented.rs`

### Measure
`tile bench -vv -f gemm_segmented` with 3 segments: K=128, 256, 512

### Risk / Blockers
1. DSL has no ragged/variable-K batched matmul abstraction.
2. Segment descriptors need a new buffer type or function-constant array.
3. **Verdict:** Blocked on DSL ragged-batch primitive.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 1/5 | No ragged batch support |
| Bench testability | 3/5 | Synthetic ragged batches easy |
| Scope | 4/5 | New kernel + buffer descriptor type |
| Risk of regression | 2/5 | Isolated path |
| **Overall** | **🔴 Blocked** | Needs DSL extension |

---

## 061 — FFT radix-4/8 Cooley-Tukey DSL implementation

**Category:** Kernel · **Impact:** Medium (signal processing, audio, spectrogram)

### Hypothesis
`fft.rs` is completely unimplemented — returns empty bench results. MLX implements radix-2/4/8 Cooley-Tukey FFT with bit-reversal permutation, butterfly operations with complex arithmetic, and sin/cos twiddle-factor tables. A direct O(N²) DFT is trivial in the DSL but meaningless vs MLX. The real goal is radix-4/8 with threadgroup staging for N up to 4096.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/fft.rs`

### Measure
`tile bench -vv -f fft` after implementation

### Risk / Blockers
1. DSL has no complex number type.
2. Bit-reversal permutation requires indirect indexing (`idx = bit_reverse(i)`), which the DSL lacks.
3. Multi-pass threadgroup synchronization (butterfly stages) needs stage barriers — the DSL barrier model is coarse (one `threadgroup_barrier` per kernel).
4. **Verdict:** Project-scale. Could start with a restricted N=power-of-2, fixed-radix kernel that precomputes twiddles as function constants.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 1/5 | No complex type, no indirect index |
| Bench testability | 1/5 | No harness exists |
| Scope | 5/5 | New domain entirely |
| Risk of regression | 1/5 | New kernel family |
| **Overall** | **🔴 Blocked** | Needs complex type + indirect index first |

---

## 062 — Conv2D Winograd F(2×2,3×3) + steel GEMM epilogue

**Category:** Kernel · **Impact:** High (CNN workloads, vision models)

### Hypothesis
`conv.rs` is unimplemented. Winograd convolution F(2×2,3×3) reduces a 3×3 conv to 4×4 GEMM tiles: input transform → GEMM → output transform. The GEMM step can reuse the steel GEMM pipeline. The transforms are small, constant matrices that can be unrolled into the DSL.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/conv.rs`
- **Depends on:** `steel_gemm_fused.rs` working end-to-end

### Measure
`tile bench -vv -f conv2d` with ResNet-style shapes (C=64, H=W=56, K=64)

### Risk / Blockers
1. Input/output transforms need 4×4 unrolled matrix multiplies — expressible in DSL but verbose.
2. Padding handling (same/valid) complicates the transform boundaries.
3. Depthwise conv is a separate kernel family.
4. **Verdict:** Project-scale. Could be a 2-week effort once steel GEMM is solid.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 2/5 | Needs steel GEMM + transform DSL patterns |
| Bench testability | 2/5 | New harness needed |
| Scope | 5/5 | New domain + steel dependency |
| Risk of regression | 1/5 | Completely new kernel |
| **Overall** | **🔴 Blocked** | Unblock after steel GEMM + FFT investigation |

---

## 063 — Ternary select: vectorized condition loads (`uchar4`/`uchar8`)

**Category:** Kernel · **Impact:** Low (elementwise, bandwidth-bound)

### Hypothesis
`ternary.rs` `mt_select` loads `u8` condition values one at a time. Metal supports `uchar4`/`uchar8` vector loads. Vectorizing the condition mask to 4-wide or 8-wide halves LSU pressure and may saturate DRAM bandwidth better on wide tensors.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/ternary.rs`
- **Codegen:** `crates/metaltile-codegen/src/passes/vectorize.rs` — ensure `u8` tensors vectorize to `uchar4`

### Measure
`tile bench -vv -f select` with N=1M, f32

### Risk / Blockers
1. The `vectorize.rs` pass may already handle this — verify with `tile inspect --kernel mt_select`.
2. If the pass misses `u8` → `uchar4`, it's a one-line fix in the vectorizer's type-width table.
3. **Verdict:** Likely a quick-win or a no-op. Verify first.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 5/5 | DSL already supports vectorization |
| Bench testability | 5/5 | Bench entry exists |
| Scope | 1/5 | One file or one pass tweak |
| Risk of regression | 1/5 | Elementwise kernel, isolated |
| **Overall** | **🟢 Feasible** | Verify in < 1 day |

---

## 064 — Strided copy: auto-vectorize when inner stride == 1

**Category:** Codegen · **Impact:** Medium (strided ops, layout transforms)

### Hypothesis
`strided.rs` `mt_strided_copy` uses the `#strided` attribute. The innermost axis may still have `stride == 1` (e.g., row-major tensor where only the outer dimension is strided). The `vectorize.rs` pass currently only vectorizes contiguous single-buffer stores. It should detect `stride == 1` on the fastest axis and emit vector loads/stores anyway.

### Target
- **Primary:** `crates/metaltile-codegen/src/passes/vectorize.rs`
- **Bench:** `crates/metaltile-std/src/mlx/strided.rs` `mt_strided_copy`

### Measure
`tile bench -vv -f strided_copy` with `pad=128` (outer strided, inner contiguous)

### Risk / Blockers
1. Strided tensors may have non-contiguous inner axes — the pass must read `strides` metadata, not just assume row-major.
2. Aliasing: if the strided source overlaps the destination, vector stores are unsafe. Need `no-alias` annotation.
3. **Verdict:** Feasible — extends existing vectorizer with a stride-check gate.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 4/5 | vectorize.rs exists, needs stride check |
| Bench testability | 4/5 | Bench entry exists |
| Scope | 2/5 | One pass tweak |
| Risk of regression | 3/5 | Could mis-vectorize non-unit-stride axes |
| **Overall** | **🟢 Feasible** | 1–2 day effort |

---

## 065 — Binary ops: fused binary→unary chain epilogue

**Category:** Codegen · **Impact:** Medium (fused activations after elementwise)

### Hypothesis
`binary.rs` implements `add`, `mul`, `sub`, `div`. In transformer graphs, `add_bias → ReLU/GELU/SiLU` is ubiquitous. The `fusion.rs` pass already fuses elementwise epilogues onto reductions. Extending it to fuse a unary op onto the output of a binary op would eliminate one kernel dispatch and one HBM round-trip.

### Target
- **Primary:** `crates/metaltile-codegen/src/passes/fusion.rs`
- **Bench:** Add a synthetic `add+relu` bench entry

### Measure
`tile bench -vv -f binary` with fused vs unfused `add+relu`

### Risk / Blockers
1. `fusion.rs` currently only fuses into `FusedElementwise` when the consumer is elementwise. Binary → unary is already elementwise → elementwise — the pass may just need to allow `Binary` as a fusion target.
2. Type promotion rules (`f16` add + `f32` relu) must be preserved.
3. **Verdict:** Feasible — likely a small scope extension to fusion heuristics.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 4/5 | fusion.rs exists |
| Bench testability | 3/5 | Need synthetic fused bench |
| Scope | 2/5 | Pass extension |
| Risk of regression | 3/5 | Could fuse incorrectly across type casts |
| **Overall** | **⚠️ Feasible** | 2–3 day effort |

---

## 066 — Arange: function-constant start/step instead of device load

**Category:** Kernel · **Impact:** Low (micro-optimization, dispatch overhead)

### Hypothesis
`arange.rs` `mt_arange` loads `start[0]` and `step[0]` from device memory every thread. These are scalar constants for the entire dispatch. Using MSL function constants (`function_constant(start)`) eliminates the load and one buffer binding. On small N (≤ 1024) the load cost is visible relative to the compute.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/arange.rs`
- **Runtime:** `crates/metaltile-runtime/src/context.rs` — fn constant plumbing

### Measure
`tile bench -vv -f arange` with N=1024 and N=1M

### Risk / Blockers
1. Bench harness must support function constants — `#[bench_kernel]` currently uses `device` buffers for all params.
2. If the harness change is large, the win may not justify the churn.
3. **Verdict:** Quick feasibility check: hand-edit kernel to use a literal constant, bench, decide if worth harness work.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 3/5 | Kernel edit is trivial; harness support is not |
| Bench testability | 5/5 | Bench entry exists |
| Scope | 1/5 | One kernel file |
| Risk of regression | 1/5 | Isolated |
| **Overall** | **🟢 Feasible** | Verify in < 1 day; full harness work = 1–2 days |

---

## 067 — RoPE: merge MLX `rope.rs` and FFAI `rope_llama.rs` into dispatch heuristic

**Category:** Kernel · **Impact:** Medium (one kernel to rule them all)

### Hypothesis
Two RoPE implementations exist:
- `mlx/rope.rs` — per-element, `program_id(0)` iterates over `d/2` positions, outer loop over `py` (sequence) and `pz` (head group).
- `ffai/rope_llama.rs` — groups 4 heads per dispatch, precomputes sin/cos per position.

They likely have different sweet spots: the MLX version may win at small `d` and short sequence; the FFAI version may win at large `d` and long sequence. A dispatch heuristic (or a single unified kernel that branches on `d` and `seq_len`) would pick the optimal path.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/rope.rs`, `crates/metaltile-std/src/ffai/rope_llama.rs`
- **Dispatch:** `crates/metaltile-std/src/run_spec.rs` — `run_rope` arm

### Measure
`tile bench -vv -f rope` across `d=64,128,256` and `seq=128,512,4096`

### Risk / Blockers
1. The two kernels use different grid layouts — switching requires dispatch-shape logic.
2. Numerical differences: one may use `metal::precise::sin`, the other `fast::sin`.
3. **Verdict:** Feasible — benchmark both, add a heuristic, no DSL changes needed.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 5/5 | Both kernels exist |
| Bench testability | 5/5 | Bench entry exists |
| Scope | 2/5 | Dispatch heuristic + bench sweep |
| Risk of regression | 2/5 | Only affects RoPE path |
| **Overall** | **🟢 Feasible** | 2–3 day effort |

---

## 068 — Memory fence + atomic barrier DSL primitives

**Category:** Kernel · **Impact:** Medium (multi-kernel pipelines, KV-cache sync)

### Hypothesis
`fence.rs` is unimplemented. Metal kernels for producer-consumer pipelines (e.g., KV-cache append → attention) need `metal::atomic_thread_fence` with `system` scope and `volatile coherent(system) device` memory qualifiers. The DSL has no atomic ops, no fences, and no `volatile`/`coherent` annotations.

### Target
- **Primary:** `crates/metaltile-std/src/mlx/fence.rs`
- **Codegen:** `crates/metaltile-codegen/src/msl/emit_block.rs` — memory qualifier emission

### Measure
`tile bench -vv -f fence` after implementation

### Risk / Blockers
1. Atomics are a large DSL surface addition (`Op::AtomicLoad`, `Op::AtomicStore`, `Op::AtomicAdd`, `Op::AtomicThreadFence`).
2. `coherent(system)` is only valid on `device` address space buffers — the DSL's `Tensor<T>` abstraction would need an opt-in coherence attribute.
3. **Verdict:** Blocked on atomic ops in DSL. However, a restricted `threadgroup_barrier` + `memory_scope::device` extension might suffice for most use cases.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 1/5 | No atomics in DSL |
| Bench testability | 1/5 | No harness |
| Scope | 4/5 | New DSL primitives + codegen |
| Risk of regression | 2/5 | New kernel family |
| **Overall** | **🔴 Blocked** | Needs atomic/fence DSL extension |

---

## 069 — Tile lowering: `GpuFamily`-aware dynamic schedule

**Category:** Codegen · **Impact:** Medium (right-size tiles per Apple Silicon generation)

### Hypothesis
`tile_lowering.rs` uses a hardcoded `Default` schedule: `tile_m=64, tile_n=64, tile_k=32, threads=(16,16,1)`. Apple7 (M1) has fixed 128-register allocation per thread — larger tiles push register pressure and reduce occupancy. Apple9+ (M3/M4/M5) has an OMU with dynamic register allocation, tolerating larger tiles. A `GpuFamily`-aware schedule selector would use smaller tiles on M1 and larger tiles on M4/M5.

### Target
- **Primary:** `crates/metaltile-codegen/src/passes/tile_lowering.rs`
- **Runtime:** `crates/metaltile-core/src/gpu_family.rs`

### Measure
`tile bench -vv -f steel_gemm_fused` (once steel GEMM is benched) with `tile_m=32` vs `64` on M1 Max

### Risk / Blockers
1. The `TileSchedule` is currently a `Default` constant — making it dynamic requires threading `GpuFamily` through the codegen pipeline.
2. Too many schedules = combinatorial explosion in PSO cache. Limit to 2–3 family buckets.
3. **Verdict:** Feasible — small codegen refactor with measurable win on M1.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 4/5 | `GpuFamily` detection exists |
| Bench testability | 3/5 | Need steel GEMM bench first |
| Scope | 2/5 | One pass + family plumbing |
| Risk of regression | 2/5 | Only affects tiled kernels |
| **Overall** | **⚠️ Feasible** | 2–3 day effort |

---

## 070 — Occupancy estimation: model threadgroup bank conflicts

**Category:** Codegen · **Impact:** Medium (reduce, softmax, scan)

### Hypothesis
`occupancy.rs` models register pressure and TG memory size but not **bank conflicts**. Apple GPUs have 32 banks of 4 bytes each. Strided access patterns (e.g., `threadgroup_store("tg_sum", lid, val)` followed by `threadgroup_load("tg_sum", lid + stride)`) hit bank conflicts when `stride` shares a bank with adjacent lanes. Adding a conflict model would let the autotuner avoid threadgroup layouts that serialize on shared memory.

### Target
- **Primary:** `crates/metaltile-codegen/src/passes/occupancy.rs`
- **Depends on:** `register_estimate.rs` for live-range analysis

### Measure
`tile bench -vv -f softmax` with `tpg=256` — bank conflicts show up as lower-than-expected GB/s despite high occupancy.

### Risk / Blockers
1. Bank conflict modeling requires knowing the exact access stride at compile time — dynamic strides (e.g., `lid * chunk`) are harder to analyze.
2. The model is heuristic; real Apple GPU banking may differ from the documented 32-bank layout.
3. **Verdict:** Feasible — add a simple stride-analysis pass to occupancy.rs. Low risk, incremental value.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 4/5 | occupancy.rs exists |
| Bench testability | 4/5 | Can measure on reduction kernels |
| Scope | 2/5 | Pass extension |
| Risk of regression | 2/5 | Only affects autotuner estimates |
| **Overall** | **⚠️ Feasible** | 2–3 day effort |

---

## 071 — `dispatch_chain`: zero-copy in-place alias tracking for mutable buffers

**Category:** Runtime · **Impact:** High (fused graphs, KV-cache, attention chains)

### Hypothesis
`context.rs` `dispatch_chain` auto-aliases buffers that are reused across consecutive passes (output of pass i → input of pass j). However, it does not track **in-place mutations** — e.g., `kv_cache_update` that writes into the same buffer it reads from. An in-place annotation (`BufferAccess::ReadWriteInPlace`) would skip allocation entirely for the output alias, eliminating one HBM round-trip per in-place op in a chain.

### Target
- **Primary:** `crates/metaltile-runtime/src/context.rs` — `DispatchSpec` alias logic
- **IR:** `crates/metaltile-core/src/ir.rs` — add `ParamKind::InPlace` or `BufferAccess` flag

### Measure
`tile bench -vv -f kv_cache` (once bench entry exists) with and without in-place tracking

### Risk / Blockers
1. In-place correctness depends on no other pass in the chain holding a reference to the same buffer. Need alias analysis.
2. `MTLStorageModePrivate` buffers can't be read by the CPU — in-place is only safe for GPU-only chains.
3. **Verdict:** Feasible — add an opt-in `in_place` flag to `ParamKind`, teach `context.rs` to skip the alias alloc.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 4/5 | Alias logic exists |
| Bench testability | 2/5 | Need kv_cache bench harness |
| Scope | 3/5 | Runtime + IR extension |
| Risk of regression | 3/5 | Could corrupt buffers if alias analysis is wrong |
| **Overall** | **⚠️ Feasible** | 3–4 day effort |

---

## 072 — Resident buffer heap suballocation

**Category:** Runtime · **Impact:** Medium (reduce `newBufferWithLength` overhead)

### Hypothesis
`context.rs` `ResidentBuffer` wraps full `MTLBuffer` allocations obtained via `newBufferWithLength`. For inference workloads that allocate many small temporaries (e.g., attention scores, cached softmax norms), each allocation hits the Metal driver allocator. Suballocating from pre-allocated `MTLHeap` slices reduces allocator pressure and improves cache locality.

### Target
- **Primary:** `crates/metaltile-runtime/src/context.rs` — `upload_resident` and `ResidentBuffer`
- **Depends on:** `buffer.rs` `BUF_POOL`

### Measure
Micro-bench: dispatch 1000 `arange` kernels with resident inputs, measure wall time.

### Risk / Blockers
1. `MTLHeap` requires iOS 13+ / macOS 10.15+ — compatible with all current targets.
2. Heap fragmentation: a bump allocator is fine for inference (predictable sizes), but training would need compaction.
3. **Verdict:** Feasible — replace `BUF_POOL` (size-bucket hashmap) with a heap-based arena per size class.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 4/5 | `BUF_POOL` exists as baseline |
| Bench testability | 4/5 | Micro-bench easy |
| Scope | 3/5 | Runtime refactor |
| Risk of regression | 2/5 | Only affects buffer allocation path |
| **Overall** | **🟢 Feasible** | 3–4 day effort |

---

## 073 — SLC flush: right-size scratch to actual SLC per device

**Category:** Runtime · **Impact:** Low (shaves µs off bench warm-up)

### Hypothesis
`runner.rs` uses a **128 MB** scratch buffer for SLC cache flush. The actual SLC sizes are:
- M1 Max: ~48 MB
- M4 Max: ~64 MB
- M5 Max: ~64 MB (documented, may differ)

Writing 128 MB is overkill — it works (guaranteed eviction) but wastes time. Right-sizing to `SLC_size + 20% margin` reduces the flush kernel dispatch time.

### Target
- **Primary:** `crates/metaltile-std/src/runner.rs` — `slc_kernel` scratch buffer size
- **Family data:** `crates/metaltile-core/src/gpu_family.rs` — `slc_label()` already has this info

### Measure
`time tile bench -f rms_norm` cold-start — compare 128 MB vs 64 MB vs 48 MB flush

### Risk / Blockers
1. `slc_label()` is a string heuristic, not a hard number. Need to convert to numeric bytes.
2. Future chips may have larger SLC — the margin must be generous enough to not break.
3. **Verdict:** Quick win. Add a `slc_bytes()` method to `GpuFamily` and use it in `runner.rs`.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 5/5 | `slc_label()` exists |
| Bench testability | 5/5 | Any bench triggers flush |
| Scope | 1/5 | Two-file change |
| Risk of regression | 1/5 | Only affects flush size |
| **Overall** | **🟢 Feasible** | 1 day effort |

---

## 074 — Algebraic simplify: strength-reduce `pow(x,2)` → `x*x` and `sqrt(x*x)` → `abs(x)`

**Category:** Codegen · **Impact:** Low (cleanup, marginal ALU savings)

### Hypothesis
`algebraic_simplify.rs` exists but its pattern set is minimal. Common patterns in normalization kernels:
- `pow(x, 2.0)` → `x * x` (avoids `log` + `exp` in `pow`)
- `sqrt(x * x)` → `abs(x)` (saves a `sqrt`)
- `(x / y) * y` → `x` (when `y != 0`, common in dequant scale/bias)

These appear in `rms_norm` (variance = `pow(x,2)` reduce) and `dequant_gemv` (scale then unscale). Each saves 1–2 ALU ops per element.

### Target
- **Primary:** `crates/metaltile-codegen/src/passes/algebraic_simplify.rs`

### Measure
`tile bench -vv -f rms_norm` and `tile bench -vv -f dequant_gemv` before/after

### Risk / Blockers
1. Floating-point `pow(x,2)` and `x*x` are not bit-exact equal (rounding difference in the `pow` path). For inference this is acceptable; for training it might not be.
2. `sqrt(x*x) → abs(x)` is exact for finite `x` but loses `NaN` propagation semantics (`sqrt(NaN*NaN)` is `NaN`; `abs(NaN)` is `NaN` — actually equivalent).
3. **Verdict:** Feasible — add a small pattern table to `algebraic_simplify.rs`. Low risk.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 5/5 | Pass exists |
| Bench testability | 4/5 | Multiple kernels exercise the patterns |
| Scope | 1/5 | Pass extension |
| Risk of regression | 2/5 | FP rounding differences only |
| **Overall** | **🟢 Feasible** | 1–2 day effort |

---

## 075 — Bench runner: encode multiple kernel variants in one command buffer for A/B testing

**Category:** Runtime · **Impact:** Medium (autotuner loop velocity)

### Hypothesis
`runner.rs` `measure()` creates a fresh `MTLCommandBuffer` per dispatch. For autotuner microbenchmarks that test 4–8 schedule variants of the same kernel, encoding all variants into a single CB + using `MTLCounterSampleBuffer` for per-dispatch timestamps eliminates driver overhead and improves measurement stability.

### Target
- **Primary:** `crates/metaltile-std/src/runner.rs` — `measure()` and `bench_gbps`
- **Depends on:** Idea #051 (pipelined sample collection)

### Measure
Wall time of `tile bench` with 8 variants of `rms_norm` — compare serial CBs vs one CB with counter samples.

### Risk / Blockers
1. `MTLCounterSampleBuffer` requires macOS 11+ — compatible.
2. Per-dispatch timer resolution must still distinguish microsecond-scale kernels. Verify counter granularity.
3. DVFS interactions: encoding many dispatches in one CB may heat the GPU differently than serial CBs.
4. **Verdict:** Feasible — extends idea #051 to the autotuner specifically.

### Feasibility Assessment
| Criterion | Score | Notes |
|---|---|---|
| Prerequisite readiness | 3/5 | Needs #051 or parallel work |
| Bench testability | 4/5 | Can micro-bench autotuner sweep |
| Scope | 3/5 | Runner refactor |
| Risk of regression | 2/5 | Only affects bench timing, not kernel correctness |
| **Overall** | **⚠️ Feasible** | 3–4 day effort |

---

## Summary & Prioritization

### Immediate (< 1 week)
- **063** Ternary select vectorization — verify if `vectorize.rs` already handles `u8`.
- **066** Arange function constants — quick feasibility check.
- **073** SLC flush right-sizing — one-day fix.
- **074** Algebraic simplify patterns — one-day pass extension.

### Short-term (1–2 weeks)
- **064** Strided copy auto-vectorize — codegen extension.
- **065** Binary→unary fusion — fusion.rs extension.
- **067** RoPE dispatch heuristic — bench comparison of two existing kernels.
- **069** Tile lowering family-aware schedule — codegen refactor.
- **070** Occupancy bank conflict model — pass extension.

### Medium-term (2–4 weeks)
- **057** Steel GEMM Split-K fusion — depends on steel GEMM baseline.
- **071** In-place alias tracking — runtime + IR extension.
- **072** Heap suballocation — runtime refactor.
- **075** Multi-variant CB encoding — runner refactor.

### Blocked on prerequisites
- **056** Steel Attention — blocked on `simdgroup_matrix` in DSL.
- **058–060** Steel Gather / Masked / Segmented — blocked on DSL primitives.
- **061** FFT — blocked on complex type + indirect index.
- **062** Conv2D — blocked on steel GEMM + transform patterns.
- **068** Memory fence/atomics — blocked on atomic ops in DSL.

---

*Document generated from gap-analysis of `crates/` on `dev-perf`.*