# Session Kickoff Prompt — Moonshots M1–M10

> Copy-paste this into a fresh session. Moonshots are project-scale, not Quick-wins. Assessment is conceptual / scoping, not bench cycles.

## Context

We are working through the `perf-research/perf-ideas.md` hopper in the **metaltile** repo. Ideas 1–55 are fully or partially assessed with individual files in `perf-research/ideas/NNN-<name>.md`. STATUS.md and RESEARCH-LOG.md are kept current.

### Your task
Assess moonshots **M1 through M10** from `perf-ideas.md`. For each moonshot:
1. Read any relevant code to understand current capabilities
2. Scope the engineering effort (weeks? months? quarters?)
3. Identify prerequisites from earlier ideas
4. Create an individual file: `perf-research/ideas/MNN-<short-name>.md`
5. Update `perf-research/STATUS.md` and `perf-research/RESEARCH-LOG.md`
6. Commit to `dev`

### Moonshot file format
Same template as regular ideas, but with these additional sections:
- **Effort estimate**: rough calendar time (week / month / quarter)
- **Prerequisites**: which earlier ideas must land first
- **Current capabilities**: what already exists vs what's missing
- **Decision**: pursue / park / defer / reject

### Commit pattern
```bash
git add -A perf-research/ && git commit -m "perf-research: moonshot scoping M1–M10"
```

---

## Moonshot M1 — ML-driven autotuner

> Train a tiny gradient-boosted model on `(kernel, shape, dtype) → best_schedule` using features from `tile profile` (regs, occupancy, bytes/flop). One-time fit, zero per-launch cost. The autotuner cache becomes a learned predictor instead of an exhaustive sweep.

**Target:** `crates/metaltile-runtime/src/autotune.rs` (currently a placeholder) + new ML training pipeline.

**Current reality:** Idea 46 notes that `autotune.rs:228` has a comment saying "placeholder, see comment". The autotuner infrastructure exists but the `lookup()` function is stubbed.

**Key questions to answer:**
- What data already exists from bench runs that could be used for training?
- What features are available from `tile profile` (regs, occupancy, bytes/flop)?
- Is the training a one-time offline process or continuous?
- What model size is appropriate? ("tiny gradient-boosted model")

---

## Moonshot M2 — AMX / ANE offload for small-batch f16 GEMM

> The Apple matrix coprocessor and Neural Engine sit idle during Metal kernel runs. Small-batch f16 GEMMs (≤ batch 4, ≤ 1024 dim) can be faster via AMX (CPU-side, through Accelerate's hidden APIs) or ANE (via CoreML). Worth measuring before designing.

**Target:** New runtime backend, `crates/metaltile-runtime/src/`.

**Current reality:** All kernels go through Metal. There is no AMX or ANE dispatch path.

**Key questions:**
- Are there existing Rust bindings for AMX/ANE?
- What is the crossover point where Metal wins vs AMX/ANE?
- Is CoreML overhead small enough for single-op dispatch?

---

## Moonshot M3 — Persistent-kernel graph capture

> Replace the dispatch-per-op model with a "graph capture" mode: a stream of ops becomes one persistent Metal kernel that pulls work items from a producer-consumer queue. Eliminates dispatch overhead entirely for inference-loop hot paths.

**Target:** `crates/metaltile-runtime/src/context.rs` + new graph IR.

**Current reality:** Each `#[kernel]` generates a standalone MSL kernel. Each op dispatch creates a new `MTLCommandBuffer`.

**Key questions:**
- How would a persistent kernel pull work? Metal doesn't have CUDA-style persistent threads.
- What IR would represent the graph? The `#[kernel]` DSL only handles single kernels.
- How does this interact with the autotuner?

---

## Moonshot M4 — Auto-fuse arbitrary elementwise DAGs at runtime

> Build a runtime IR that captures `softmax(qk).matmul(v).rms_norm(g)` and JIT-compiles the whole chain. Same compiler infrastructure already exists for the `#[kernel]` macro — generalize it to runtime-constructed graphs.

**Target:** `crates/metaltile-codegen/src/` + new runtime graph IR.

**Current reality:** The `#[kernel]` macro is compile-time only. There is no runtime graph construction API.

**Key questions:**
- What does "runtime-constructed graphs" mean in this Rust codebase?
- How would the JIT compile a graph into a single MSL kernel?
- What are the memory management implications (intermediate buffers)?

---

## Moonshot M5 — Block-sparse SDPA exploiting real mask patterns

> Sliding-window attention, sink-token, BigBird — all have known sparsity structure. A codegen path that takes mask metadata as a constexpr and emits a kernel skipping zero blocks could 4–8x decode throughput at long context.

**Target:** `crates/metaltile-codegen/src/` + new SDPA kernel variant.

**Current reality:** SDPA kernels are dense. No block-sparse attention path exists.

**Key questions:**
- What mask metadata format would the kernel accept?
- How does the block size interact with head_dim and simdgroup size?
- Is this a new `#[kernel]` DSL extension or raw MSL?

---

## Moonshot M6 — KV-cache via Metal heaps + virtual remap

> Append to KV cache currently means copy. With Metal heaps and `MTLBufferAccessUsage::TIER2`, you can carve a fresh slice off a pre-allocated heap each step and treat it as the new tail — zero copy, zero allocation.

**Target:** `crates/metaltile-runtime/src/buffer.rs` + KV-cache allocator.

**Current reality:** `kv_cache_update` in `ffai/kv_cache.rs` copies one token at a time into a pre-allocated buffer. No Metal heap usage.

**Key questions:**
- Is `MTLBufferAccessUsage::TIER2` available on all target devices?
- What is the overhead of heap allocation vs direct buffer allocation?
- How does this interact with quantization (int4/int8 KV caches)?

---

## Moonshot M7 — Speculative-decode batched-K SDPA

> Draft models propose multiple Q tokens at once; KV is shared. A batched-Q SDPA kernel (currently single-Q decode path) unlocks speculative decoding without splitting into N independent dispatches.

**Target:** `crates/metaltile-std/src/ffai/sdpa_decode.rs` or new kernel.

**Current reality:** `sdpa_decode` processes one Q token per threadgroup. `sdpa_vector` (MLX bench) also single-Q.

**Key questions:**
- How would the dispatch grid change for batched-Q? `[n_kv_heads, n_draft_tokens, 1]`?
- How does online-softmax scale with multiple Q tokens?
- Is the bottleneck compute or memory bandwidth for batched-Q?

---

## Moonshot M8 — Codegen → Metal 3.2 tensor descriptors

> Metal 3.2 (M4-era) exposes hardware tensor descriptors closer to NVIDIA's TMA. Once GA, the codegen layer can target it for D=128 GEMM/SDPA tiles, getting H/W async copy + autoswizzle for free.

**Target:** `crates/metaltile-codegen/src/msl/`.

**Current reality:** Codegen targets Metal 3.1 via standard MSL.

**Key questions:**
- Is Metal 3.2 GA yet?
- What API changes does Metal 3.2 tensor descriptor require?
- How does TMA-style async copy map to the existing `#[kernel]` DSL?

---

## Moonshot M9 — CPU SIMD fallback codegen (NEON)

> Same `#[kernel]` macro, second backend: NEON via Rust's `std::simd`. Unlocks unit-testing on CI (no Mac required), and gives CPU-only Macs (none ship now, but Linux ARM does) a coherent story.

**Target:** `crates/metaltile-codegen/src/` + new NEON emitter.

**Current reality:** Codegen only emits MSL (Metal Shading Language). There is no CPU backend.

**Key questions:**
- How much of the codegen pipeline is MSL-specific vs target-agnostic?
- Can the IR (`metaltile_core::ir::Kernel`) lower to both MSL and NEON?
- What Rust SIMD crate is appropriate? (`std::simd` is nightly; `portable-simd`?)

---

## Moonshot M10 — Operator-cost predictor for op-fusion decisions

> A learned cost model: given an op DAG and target hardware, predicts the runtime of every possible fusion partition. Drives an automatic fusion-planner during codegen. Pairs with M1 — the same features.

**Target:** New module, pairs with M1 autotuner.

**Current reality:** No cost model exists. Fusion decisions are manual or heuristic.

**Key questions:**
- Is this the same model as M1 or a separate one?
- What hardware features does the cost model need? (regs, occupancy, memory bandwidth, compute units)
- How does it generalize across different Apple GPU families?

---

## Cross-cutting themes

| Theme | Moonshots | Prerequisite |
|-------|-----------|--------------|
| Autotuning / ML | M1, M10 | Idea 46 (stubbed autotuner) |
| New backends | M2 (AMX/ANE), M9 (NEON) | Runtime architecture changes |
| Graph-level | M3 (persistent), M4 (fusion), M7 (speculative) | Runtime graph IR |
| Sparse / efficient attention | M5 (block-sparse), M6 (heap KV) | New kernel variants |
| Hardware evolution | M8 (Metal 3.2) | Wait for GA |

## Assessment criteria for each moonshot

Rate each on these axes:
1. **Effort**: week / month / quarter / longer
2. **Prerequisites**: which earlier ideas must land first
3. **Risk**: technical (can we build it?), market (does hardware support it?), maintenance burden
4. **ROI**: throughput win, developer velocity win, or ecosystem win
5. **Current readiness**: is the infra already there or do we need to build foundations?

## Output expectation
At the end of the session, `ls perf-research/ideas/` should show files for **M1, M2, M3, M4, M5, M6, M7, M8, M9, M10**. STATUS.md updated with a "Moonshot" category. RESEARCH-LOG.md has a summary table. One commit.
