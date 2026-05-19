# Perf Idea M10 — Operator-cost predictor for op-fusion decisions

## Metadata
- **Number**: M10
- **Name**: operator-cost-predictor
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> A learned cost model: given an op DAG and target hardware, predicts the runtime of every possible fusion partition. Drives an automatic fusion-planner during codegen. Pairs with M1 — the same features.

## Target
- **Primary file(s)**: new runtime graph IR + learned model (does not exist)
- **Bench filter**: would need graph-level bench harness
- **Shapes / dtypes to watch**: transformer blocks (SDPA + MLP + norms)

## Assessment

### What the idea describes
An automatic fusion planner that:
1. Takes a DAG of ops (e.g., `matmul → add → silu → mul → matmul`).
2. Enumerates all possible fusion partitions (which ops to fuse into which kernels).
3. Predicts the runtime of each partition using a learned cost model.
4. Picks the partition with minimum predicted runtime.

This is the core optimization problem in **XLA** (TensorFlow) and **TVM**.

### Blockers

1. **No graph IR** (same as M4): MetalTile has no runtime IR that represents a graph of ops. Each `#[kernel]` is a standalone `Kernel`. Without a graph IR, there is no DAG to partition.

2. **No fusion beyond block-local** (same as M4): `fusion.rs` only fuses elementwise chains within a single block. It cannot fuse across reductions, barriers, or kernel boundaries. A fusion planner needs cross-block, cross-kernel fusion — which MetalTile does not support.

3. **No cost model data** (same as M1): A learned cost model needs training data: runtime measurements of many different fusion partitions. But:
   - MetalTile cannot generate arbitrary fusion partitions (no cross-kernel codegen).
   - The autotuner has no search implementation (see #046).
   - The bench harness measures single kernels, not fused chains.

4. **Hardware-specific costs**: The cost model must account for:
   - Register pressure (limits occupancy).
   - Threadgroup memory size (limits tile size).
   - Memory bandwidth vs. ALU throughput (determines if fusion helps or hurts).
   - Barrier cost (synchronization overhead).
   
   These features are already extracted by `compute_profiles()` in `bench.rs`, but they are kernel-level, not graph-level.

### What already exists that could be reused
- `register_estimate.rs` — static register pressure.
- `occupancy.rs` — occupancy prediction per kernel.
- `compute_profiles()` in `bench.rs` — extracts static features per kernel.
- `TuneCache` — stores per-kernel best configs.

### What would need to be built
1. **Graph IR**: A `Graph` type with `Node`s (ops) and `Edge`s (data flow).
2. **Partition enumerator**: Generate all valid fusion partitions of the graph.
3. **Cross-kernel codegen**: Generate a single `Kernel` from a fused subgraph (same blocker as M4).
4. **Cost model**: Train on measured runtimes of different partitions.
5. **Planner**: Search over partitions using the cost model (beam search, DP, or greedy).

### Effort estimate
- Graph IR: **multi-day**.
- Partition enumerator: **multi-day**.
- Cross-kernel codegen: **project-scale** (M4).
- Cost model + training: **multi-day** (M1).
- Planner: **multi-day**.
- **Total**: **project-scale** (months).

## Verdict

- **Outcome**: blocked — prerequisite missing (graph IR + cross-kernel codegen)
- **Why**: The cost model is the "brain" of a fusion planner, but the planner needs a body: a graph IR, a partition enumerator, and a cross-kernel code generator. None of these exist in MetalTile. M10 is essentially M4 + M1 combined.
- **Re-scope**: If M4 (auto-fuse DAGs) is ever implemented, M10 becomes the natural next step. Until then, it is not actionable.

## Risk Register
- Cost model accuracy: even with perfect data, predicting GPU runtime from static features is hard (DVFS, cache effects, driver scheduling).
- Search space explosion: the number of fusion partitions grows exponentially with graph size. Need pruning heuristics.
- Correctness: fused kernels must be numerically equivalent to unfused. Proving this for arbitrary fusion is hard.

## Notes for Next Person
- If you want to work on this, start with M4 (graph IR + cross-kernel codegen). M10 is the optimization layer on top of M4.
- XLA's fusion heuristics are well-documented. Study XLA's `fusion_queue` and `cost_model` for inspiration.
- A simpler interim step: hand-write fused kernels for the 3–4 most common transformer patterns (SDPA+norm, MLP, etc.) and measure the win. This gives data for a future cost model without building the planner.
