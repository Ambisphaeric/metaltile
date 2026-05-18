# MetalTile Perf Ideas — `bench, tweak, bench, tweak`

A hopper of independent, individually-bench-testable performance experiments for autoresearch-style loops. Each entry is scoped so one iteration = one tweak in one file, then `tile bench -vv -f <op>` re-runs in seconds.

**How to read each entry:**
- *Target* — primary file/area to edit.
- *Hypothesis* — the perf claim. If true, you'll see it in MT% / GB/s / GFLOP/s.
- *Measure* — minimal bench filter to confirm. Use `-vv` for `cv%`/p95/p99.
- *Risk* — what to watch for (correctness, regression on other shapes, register pressure).

Baseline: each tweak must keep `ok = ✓` on the correctness column and `cv% < 5%` on the headline.

---

## Quick-wins — single-file kernel knobs (1–15)

### 1. SDPA tile: bump `BLOCK_M` on f16/bf16
*Target*: `crates/metaltile-std/src/mlx/scaled_dot_product_attention.rs` (template constants).
*Hypothesis*: f16/bf16 halve register/threadgroup pressure vs f32, so the Q-tile rows can grow (16→32) without spilling. K/V load amortization scales with `BLOCK_M`.
*Measure*: `tile bench -vv -f sdpa` — watch `regs` and `MT%` for `H=32 N=4096 D=128 f16/bf16`.
*Risk*: spill at D=128 if registers exceed 128/thread; check `regs` column doesn't jump past ~120.

### 2. SDPA: `BLOCK_N` 64 → 128 on D=128
*Target*: same as #1.
*Hypothesis*: FlashAttention-2 paper shows BLOCK_N=128 wins on D=128 once K/V fits in threadgroup memory. M-series threadgroup mem (32 KB) easily fits 128×128 f16.
*Measure*: `tile bench -f sdpa`; expect uplift in `H=32 N=4096` cases.
*Risk*: smaller-D shapes (D=64) may regress — only flip the constant for D=128 path.

### 3. SDPA: split-K for low-occupancy H=8 shapes
*Target*: dispatcher in `crates/metaltile-std/src/run_spec.rs` (the `run_attention` arm), plus a split-K variant of the kernel.
*Hypothesis*: H=8 fills only 25% of M1 Max's cores. Splitting K-sequence into 4 chunks × 8 threadgroups = 32 threadgroups = full occupancy. Cost is a tiny second-stage merge.
*Measure*: `tile bench -f sdpa -vv` `H=8 N=2048 D=128 f32`. Target lifting from ~150 GB/s to ~250 GB/s.
*Risk*: epilogue merge is its own kernel — has to be small/cheap relative to first stage.

### 4. SDPA-vector decode: GQA-aware K/V reuse
*Target*: `crates/metaltile-std/src/mlx/sdpa_vector.rs` and `ffai/sdpa_decode.rs`.
*Hypothesis*: With `gqa=4`, every 4 Q-heads share the same K/V row. Currently each Q-head loads K/V independently. Load once per kv-group, `simd_shuffle` to each Q-head.
*Measure*: `tile bench -f sdpa_vector` — gqa=4 rows. Today MT% bf16 = 61%; aim 80%+.
*Risk*: alignment of head→kv-group within a threadgroup.

### 5. SDPA-vector: 8-wide vectorized loads on f16/bf16
*Target*: `mlx/sdpa_vector.rs`.
*Hypothesis*: currently loads vec4 of f16. Bumping to vec8 (`half8`) halves the LSU instruction count.
*Measure*: `tile bench -f sdpa_vector`.
*Risk*: D must be divisible by 8 (D=128 is fine); requires Metal 3.

### 6. RMS-norm: unroll 4 → 8
*Target*: `crates/metaltile-std/src/mlx/rms_norm.rs` (the DSL kernel in README is the model).
*Hypothesis*: 4-wide unrolls saturate ALU; 8-wide hides L1 latency better. README says already 104% of MLX on M4 — there may be headroom on M1.
*Measure*: `tile bench -f rms_norm`.
*Risk*: register pressure; verify `regs` doesn't push past 64 for f32.

### 7. Softmax: simdgroup reduce for small N (≤ 32)
*Target*: `mlx/softmax.rs`.
*Hypothesis*: for N ≤ 32 the two-pass threadgroup-memory reduction is overkill; use `simd_max` + `simd_sum` directly.
*Measure*: `tile bench -f softmax` with small N.
*Risk*: codegen may already do this — verify by inspecting MSL via `tile inspect`.

### 8. Softmax: float4 loads on f16/bf16 inner loop
*Target*: `mlx/softmax.rs`.
*Hypothesis*: load 4 elements as a vector, exp in lockstep — should saturate exp unit.
*Measure*: `tile bench -f softmax -vv`.
*Risk*: exp accuracy across vector lanes (no real risk on Metal).

### 9. LayerNorm: mirror RMS-norm tweaks
*Target*: `mlx/layer_norm.rs`.
*Hypothesis*: same structural improvements (unroll 8, simdgroup reduce) apply.
*Measure*: `tile bench -f layer_norm`.

### 10. GEMV: tune `simd_per_tg` per K dimension
*Target*: `mlx/gemv.rs`.
*Hypothesis*: large K (≥ 4096) wants 8 simdgroups per threadgroup for latency hiding; small K wants 2–4.
*Measure*: `tile bench -f gemv -vv`.
*Risk*: too many simdgroups → spill register file shared with kernel.

### 11. GEMV-masked: dense fallback above 50% density
*Target*: `mlx/gemv_masked.rs`.
*Hypothesis*: mask evaluation cost dominates when most rows are unmasked. Detect density at launch and route to dense `gemv`.
*Measure*: `tile bench -f gemv_masked` with synthetic dense-mask inputs.
*Risk*: classification overhead — must be cheaper than the savings.

### 12. all_reduce: two-stage simd→threadgroup
*Target*: `mlx/reduce.rs`.
*Hypothesis*: native simdgroup reduce intrinsics (Metal 3.1) eliminate one barrier.
*Measure*: `tile bench -f all_reduce`.
*Risk*: code path matters — `tile inspect` to confirm intrinsic was emitted.

### 13. row-reduce: rows-per-threadgroup when N is small
*Target*: `mlx/reduce.rs`.
*Hypothesis*: N < 256 means one row fits in one simdgroup. Pack multiple rows per threadgroup to improve occupancy.
*Measure*: `tile bench -f row_reduce`.
*Risk*: divergent stride — bench with both small/large N.

### 14. scan: prefer `simd_prefix_inclusive_sum`
*Target*: `mlx/scan.rs`.
*Hypothesis*: Metal 3.1 intrinsic should beat manual Kogge-Stone. If codegen already uses it, no-op — verify.
*Measure*: `tile bench -f scan`; `tile inspect --kernel mt_scan`.
*Risk*: zero (drop-in).

### 15. argmax: refuse to slow down 206%
*Target*: `ffai/arg_reduce.rs`.
*Hypothesis*: argmax already crushes MLX (206%). The interesting experiment is *can we hold this while shrinking register pressure*? Lowering regs frees occupancy for other kernels in fused graphs.
*Measure*: `tile profile mt_arg_reduce` for `regs`; bench for `MT%`.
*Risk*: regress 206% number — back out if MT% drops below 180%.

---

## Op-level structural changes (16–35)

### 16. RoPE: precompute sin/cos to threadgroup memory
*Target*: `ffai/rope_llama.rs`, `mlx/rope.rs`.
*Hypothesis*: sin/cos table for D=128 is 1 KB. Compute once per threadgroup, reuse across all heads in the tile.
*Measure*: `tile bench -f rope`.
*Risk*: extra barrier — only wins if N (seqlen) per threadgroup ≥ 32.

### 17. RoPE-into-QKV fusion
*Target*: bench harness for `qkv_proj + rope`; new fused kernel in `ffai/`.
*Hypothesis*: write Q/K from projection straight into rotated form, skipping a round-trip to global memory.
*Measure*: would need a new bench entry (`qkv_rope`); compare against MLX's `rope_apply_inplace`.
*Risk*: requires a small DSL extension to express the fusion idiomatically.

### 18. KV-cache append: vectorized aligned copy
*Target*: `ffai/kv_cache.rs`.
*Hypothesis*: append path is currently scalar/byte-wise; bump to vec4/vec8 with alignment guard.
*Measure*: `tile bench -f kv_cache` (add if missing).
*Risk*: alignment fallback for non-multiple-of-8 head dims.

### 19. Gather: prefetch-to-threadgroup for hot indices
*Target*: `ffai/gather.rs`, `mlx/strided.rs`.
*Hypothesis*: when indices show locality (KV-cache lookups, embedding tables), staging a window into threadgroup mem cuts DRAM round-trips.
*Measure*: `tile bench -f gather`.
*Risk*: pessimizes purely random gather — add a heuristic toggle.

### 20. Strided copy: emit vec types for stride-1 axes
*Target*: `mlx/copy.rs` + codegen `vectorize.rs`.
*Hypothesis*: contiguous inner axis should always vectorize. If codegen misses it, that's a codegen bug worth chasing.
*Measure*: `tile bench -f copy`; inspect MSL.
*Risk*: zero if guarded correctly.

### 21. FP4 dequant: packed bit ops
*Target*: `mlx/fp_quantized.rs`.
*Hypothesis*: unpack 8 fp4 → 8 half in one 32-bit shuffle with a precomputed LUT in const memory.
*Measure*: `tile bench -f fp_quantized`.
*Risk*: LUT in `constant` address space — make sure it fits in the 64 KB const segment.

### 22. Quantized int4 GEMV: simdgroup_matrix multiply
*Target*: `ffai/dequant_gemv.rs`.
*Hypothesis*: currently dequant→scalar mul. Dequant a tile into threadgroup memory then use `simdgroup_matrix_multiply` (16×16×16 tile).
*Measure*: `tile bench -f dequant_gemv`.
*Risk*: bigger kernel — only wins above some K threshold.

### 23. Quantized GEMV: int4 pack-of-2 lookup
*Target*: `mlx/quantized.rs`.
*Hypothesis*: pack `(int4_lo, int4_hi)` as uint8 index into a 256-entry half2 LUT — single load yields two dequanted values.
*Measure*: `tile bench -f quantized`.
*Risk*: LUT init cost (one-time, threadgroup-shared).

### 24. dequant_gather: skip dequant for cold misses
*Target*: `ffai/dequant_gather.rs`.
*Hypothesis*: rare — most lookups hit the L1; the kernel currently dequants unconditionally. Profile to confirm there's a measurable cold-miss frequency before chasing.
*Measure*: `tile profile mt_dequant_gather`.
*Risk*: premature; verify the assumption first.

### 25. Sort: 4-way bitonic merge
*Target*: `mlx/sort.rs`.
*Hypothesis*: stride-2 bitonic does N/2 compares per stage; stride-4 does N/4 with the same simdgroup width.
*Measure*: `tile bench -f sort` (warning: rarely benched — may need a small wrapper).
*Risk*: register usage doubles — `regs` column will tell you.

### 26. Sampling: radix-select top-k
*Target*: `ffai/sampling.rs`.
*Hypothesis*: current top-k probably sorts then slices; radix-select is O(N) for fixed k.
*Measure*: add a `tile bench -f sampling` case if missing.
*Risk*: only wins for k ≪ N (typical k ≤ 64).

### 27. SSM: scan with state vectorization
*Target*: `ffai/ssm.rs`.
*Hypothesis*: state vector update per token is the hot path; fuse the scan with the state-update mul.
*Measure*: `tile bench -f ssm`.
*Risk*: SSM math is delicate — keep correctness check tight.

### 28. logsumexp: fuse max + sum-exp
*Target*: `mlx/logsumexp.rs`.
*Hypothesis*: two-pass max-then-sum can collapse into one numerically-stable pass with a running update (same trick as online softmax).
*Measure*: `tile bench -f logsumexp`.
*Risk*: numerical accuracy — verify against CPU reference at fp32.

### 29. Reductions over short rows: cooperative groups
*Target*: `mlx/reduce.rs`.
*Hypothesis*: for N ≤ 32, one simdgroup does the whole row. No threadgroup memory, no barrier.
*Measure*: `tile bench -f all_reduce` with N=32.
*Risk*: dispatch shape must match — codegen specialization.

### 30. binary_two (fused add+mul): autovec test
*Target*: `mlx/binary_two.rs`.
*Hypothesis*: `fma` should auto-emit. If MT% lags MLX, codegen is missing it; inspect MSL.
*Measure*: `tile bench -f binary_two`.
*Risk*: zero (diagnostic).

### 31. Unary chains: emit single `metal::precise::*` calls
*Target*: `mlx/unary.rs` + codegen fusion.
*Hypothesis*: `sigmoid(x)` should not be `1 / (1 + exp(-x))` — Metal has `metal::precise::sigmoid` directly.
*Measure*: `tile bench -f sigmoid -vv`; `tile inspect`.
*Risk*: precision difference (fast vs precise) — likely fine for inference.

### 32. SwiGLU/GELU: fuse with downstream matmul write
*Target*: `mlx/unary.rs` + new fused emitter.
*Hypothesis*: activation right after a GEMM is everywhere in transformers. Fuse the epilogue so output never round-trips to HBM.
*Measure*: would need a new bench `gemm_silu`.
*Risk*: scope — touches GEMM epilogue codegen.

### 33. arg_reduce variants: argmin in same kernel
*Target*: `ffai/arg_reduce.rs`.
*Hypothesis*: argmax is great; argmin shares 90% of structure. Confirm both are equally tuned.
*Measure*: `tile bench -f argmin` (add if missing).
*Risk*: cheap to check.

### 34. softmax + attention epilogue fusion
*Target*: `mlx/softmax.rs` + new fused kernel in `ffai/`.
*Hypothesis*: standalone softmax bench is one number, but in real attention softmax + matmul-with-V is one operator. Fuse and bench against the two-kernel version.
*Measure*: add `softmax_v` bench.
*Risk*: scope.

### 35. random (xorshift32): use 64-bit state, vec4 generation
*Target*: `mlx/random.rs`.
*Hypothesis*: 32-bit xorshift wastes a register that could hold the high 32 bits; vec4 generation amortizes constant load.
*Measure*: `tile bench -f random`.
*Risk*: correctness of statistical properties — confirm with chi-square or just match MLX byte-for-byte.

---

## Codegen passes — repo-wide improvements (36–45)

### 36. `vectorize.rs`: 8-wide on f16/bf16
*Target*: `crates/metaltile-codegen/src/passes/vectorize.rs`.
*Hypothesis*: currently caps at vec4 (one MSL `*4` type). Metal supports `half8`/`bfloat8`. Emitting 8-wide doubles LSU throughput on bandwidth-bound kernels.
*Measure*: re-run *all* kernels with a single switch — that's the test.
*Risk*: a single bug regresses everything; gate behind a feature flag, A/B with `MTLT_VEC_WIDTH=8`.

### 37. `vectorize.rs`: detect strided-but-aligned stores
*Target*: same.
*Hypothesis*: stride==1 isn't the only vectorizable case — interleaved stores into different output buffers are also fair game.
*Measure*: `tile bench` aggregate.
*Risk*: aliasing; require explicit no-alias annotation on tensor inputs (already the convention).

### 38. `fusion.rs`: epilogue fusion onto reductions
*Target*: `crates/metaltile-codegen/src/passes/fusion.rs`.
*Hypothesis*: `softmax(x) * w` and `rms_norm(x) * w` are common; fuse the multiply into the reduction kernel.
*Measure*: bench fused vs unfused — would need a harness to express the chain.
*Risk*: type-check needs to allow the new fused op.

### 39. `fusion.rs`: multi-reduction in one pass
*Target*: same.
*Hypothesis*: variance computation reads x twice (mean, then mean^2). Read once, accumulate both — saves half the memory traffic.
*Measure*: `tile bench -f layer_norm` (layer-norm is the canonical case).
*Risk*: numerical stability with Welford's algorithm — well-known fix.

### 40. `unroll.rs`: register-pressure-aware unroll count
*Target*: `crates/metaltile-codegen/src/passes/unroll.rs` + `register_estimate.rs`.
*Hypothesis*: today's unroll factor is likely fixed-per-loop. Pick `unroll_count = max_regs / regs_per_iter`.
*Measure*: aggregate bench; `regs` column should fill closer to 80% of cap.
*Risk*: regress small kernels that don't have register headroom.

### 41. `schedule.rs`: software pipelining
*Target*: `crates/metaltile-codegen/src/passes/schedule.rs`.
*Hypothesis*: emit load(i+1) before compute(i) in the inner loop; classic 2-stage pipeline hides 50%+ of L1 latency.
*Measure*: any memory-bound kernel — `rms_norm`, `softmax`, `copy`.
*Risk*: register pressure; needs to compose with unroll heuristic from #40.

### 42. `licm.rs`: hoist gather indices when loop-invariant
*Target*: `crates/metaltile-codegen/src/passes/licm.rs`.
*Hypothesis*: `tensor[constant_idx]` inside an inner loop should be hoisted; verify currently is.
*Measure*: inspect MSL on a kernel known to have invariant indices.
*Risk*: easy diagnostic.

### 43. `cse.rs`: extend across simdgroup boundaries
*Target*: `crates/metaltile-codegen/src/passes/cse.rs`.
*Hypothesis*: shared subexpressions across simdgroup branches are likely missed today; broaden the scope.
*Measure*: code-size delta on MSL; runtime delta on bench.
*Risk*: aliasing across threads — be conservative with side-effecting ops.

### 44. `if_conversion.rs`: predicate tiny ifs in inner loops
*Target*: `crates/metaltile-codegen/src/passes/if_conversion.rs`.
*Hypothesis*: divergent simdgroup execution from `if (mask) { ... }` costs more than always-executing both sides for short bodies.
*Measure*: `tile bench -f gemv_masked`.
*Risk*: predicating ops with side effects (loads with OOB are bad on Metal).

### 45. `value_sink.rs`: sink threadgroup-memory stores
*Target*: `crates/metaltile-codegen/src/passes/value_sink.rs`.
*Hypothesis*: sinking a `threadgroup` store to right before the next barrier shortens the live range and frees a register.
*Measure*: aggregate bench + `regs` column.
*Risk*: barrier semantics — must preserve happens-before across threads.

---

## Runtime / dispatch / build (46–55)

### 46. Wire the autotuner `lookup()` (currently a placeholder)
*Target*: `crates/metaltile-runtime/src/autotune.rs:228` — the comment says "placeholder, see comment".
*Hypothesis*: a real lookup pipeline that selects pre-tuned schedules per (kernel, shape, dtype) tuple unlocks every other tweak in this list.
*Measure*: end-to-end speedup once even one kernel is plumbed.
*Risk*: highest-ROI moonshot-adjacent item — should be #1 if the loop allows broader refactors.

### 47. PSO disk cache
*Target*: `crates/metaltile-runtime/src/context.rs`.
*Hypothesis*: cold-start compile time dominates first-run latency; serialize compiled PSOs to `~/.cache/metaltile/pso/`.
*Measure*: `time tile bench` cold vs warm.
*Risk*: invalidation on toolchain bump — hash the MSL.

### 48. Heap-backed buffer pool
*Target*: `crates/metaltile-runtime/src/buffer.rs`.
*Hypothesis*: `newBufferWithBytes` allocations have non-zero cost; pre-allocate a heap, slice from it.
*Measure*: bench dispatch overhead — micro-bench a no-op kernel.
*Risk*: lifetime tracking; integrate carefully with Rust ownership.

### 49. Reuse command buffer across bench iterations
*Target*: `crates/metaltile-std/src/runner.rs` (`bench_gbps`).
*Hypothesis*: each iteration currently makes a fresh `MTLCommandBuffer`. Encoding N dispatches into one buffer cuts driver overhead.
*Measure*: tiny kernels (`copy`, `arange`) — should see GB/s rise.
*Risk*: changes the semantics of per-iter timing — GPU timestamps still work, but the model becomes "average per dispatch within one buffer". Document carefully.

### 50. Fast-math + disable shader-validation in release
*Target*: `crates/metaltile-runtime/src/context.rs` `MTLCompileOptions`.
*Hypothesis*: shader validation is on by default in debug; ensure release path disables it. Combine with `MTLLanguageVersion::Metal3_1` + `mathMode = fast`.
*Measure*: `tile bench` aggregate.
*Risk*: minor numerical drift in unary ops — already gated by `precise::` annotations where it matters.

### 51. Bench: pipelined sample collection
*Target*: `crates/metaltile-std/src/runner.rs`.
*Hypothesis*: today the SLC flush + warmup + 10 samples is serial per kernel. Encode all warmups + samples in one command buffer, read timestamps from `MTLCounterSampleBuffer`.
*Measure*: total bench wall time.
*Risk*: per-sample timer resolution — verify `gpu_time` deltas still match.

### 52. Multi-launch occupancy headroom: persistent threadgroups
*Target*: `crates/metaltile-runtime/src/context.rs`, optional.
*Hypothesis*: for ops dispatched in tight sequence, persistent threadgroups that pull work from a queue beat re-dispatching every step.
*Measure*: a chain of small ops — would need a microbench.
*Risk*: big refactor, mostly relevant once op fusion is in place.

### 53. CLI: parallelize per-kernel benches across non-overlapping shapes
*Target*: `crates/metaltile-cli/src/cmd/bench.rs`.
*Hypothesis*: dev-loop friction. *Doesn't change kernel perf*, but speeds the loop — same benches in less wall time means more tweak cycles per hour.
*Measure*: `time tile bench`.
*Risk*: cross-kernel DVFS pollution; need to keep `flush_slc` between independent kernels (already does).

### 54. CLI: `tile bench --compare-against <baseline.json>` inline
*Target*: `crates/metaltile-cli/src/cmd/bench.rs`.
*Hypothesis*: half of the autoresearch loop's value is "did my last tweak improve or regress?" Save the previous run automatically, diff inline.
*Measure*: loop velocity (qualitative).
*Risk*: minor UX.

### 55. Build: precompile `.metallib` per Apple GPU family
*Target*: `crates/metaltile-std/build.rs`.
*Hypothesis*: today the runtime compiles MSL on first dispatch. Pre-compiled per-family `.metallib` (Apple7, Apple8, Apple9) eliminates first-dispatch latency.
*Measure*: `time tile bench` cold-cache.
*Risk*: build-time cost (acceptable); fall back to JIT compile when family unknown.

---

## Moonshots (50+ already; here are 10 more swings)

### M1. ML-driven autotuner
Train a tiny gradient-boosted model on `(kernel, shape, dtype) → best_schedule` using features from `tile profile` (regs, occupancy, bytes/flop). One-time fit, zero per-launch cost. The autotuner cache becomes a learned predictor instead of an exhaustive sweep.

### M2. AMX / ANE offload for small-batch f16 GEMM
The Apple matrix coprocessor and Neural Engine sit idle during Metal kernel runs. Small-batch f16 GEMMs (≤ batch 4, ≤ 1024 dim) can be faster via AMX (CPU-side, through Accelerate's hidden APIs) or ANE (via CoreML). Worth measuring before designing.

### M3. Persistent-kernel graph capture
Replace the dispatch-per-op model with a "graph capture" mode: a stream of ops becomes one persistent Metal kernel that pulls work items from a producer-consumer queue. Eliminates dispatch overhead entirely for inference-loop hot paths.

### M4. Auto-fuse arbitrary elementwise DAGs at runtime
Build a runtime IR that captures `softmax(qk).matmul(v).rms_norm(g)` and JIT-compiles the whole chain. Same compiler infrastructure already exists for the `#[kernel]` macro — generalize it to runtime-constructed graphs.

### M5. Block-sparse SDPA exploiting real mask patterns
Sliding-window attention, sink-token, BigBird — all have known sparsity structure. A codegen path that takes mask metadata as a constexpr and emits a kernel skipping zero blocks could 4–8x decode throughput at long context.

### M6. KV-cache via Metal heaps + virtual remap
Append to KV cache currently means copy. With Metal heaps and `MTLBufferAccessUsage::TIER2`, you can carve a fresh slice off a pre-allocated heap each step and treat it as the new tail — zero copy, zero allocation.

### M7. Speculative-decode batched-K SDPA
Draft models propose multiple Q tokens at once; KV is shared. A batched-Q SDPA kernel (currently single-Q decode path) unlocks speculative decoding without splitting into N independent dispatches.

### M8. Codegen → Metal 3.2 tensor descriptors
Metal 3.2 (M4-era) exposes hardware tensor descriptors closer to NVIDIA's TMA. Once GA, the codegen layer can target it for D=128 GEMM/SDPA tiles, getting H/W async copy + autoswizzle for free.

### M9. CPU SIMD fallback codegen (NEON)
Same `#[kernel]` macro, second backend: NEON via Rust's `std::simd`. Unlocks unit-testing on CI (no Mac required), and gives CPU-only Macs (none ship now, but Linux ARM does) a coherent story.

### M10. Operator-cost predictor for op-fusion decisions
A learned cost model: given an op DAG and target hardware, predicts the runtime of every possible fusion partition. Drives an automatic fusion-planner during codegen. Pairs with M1 — the same features.

---

## Loop fit notes

- **Cheapest cycle**: items 1–15. Single constant flip, single bench rerun. Expect 5-min wall time per cycle.
- **One day of cycles**: items 16–35. Some require a new bench harness entry, but most are 1-file diffs.
- **Multi-day**: items 36–55. Touch codegen/runtime — broader test surface, longer correctness checks.
- **Project-scale**: moonshots. Each is a new sub-project.

## Methodology reminders for the loop

1. Run `tile bench -vv -f <op>` *twice* before claiming a regression — DVFS can take a run to stabilize after a code change forces recompile.
2. Always check the `ok` column. Speedup with a correctness regression is not a win.
3. Watch `cv%` — anything > 5% means the win is bench noise.
4. Per the runner: `min_us` is the GB/s metric (intentional), but `p95`/`p99`/`cv%` from `-vv` are the trust signals.
5. Save `tile snap -o results/<idea-N>.json` after each kept change so you can `tile diff` across the whole sweep at the end.
