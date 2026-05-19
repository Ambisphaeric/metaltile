# 025 — Sort: 4-way bitonic merge

## Metadata
- **Number**: 025
- **Name**: sort-4-way-bitonic-merge
- **Source**: `perf-ideas.md` § Op-level structural changes — item 25
- **Status**: ⚠️ feasible / high risk
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> stride-2 bitonic does N/2 compares per stage; stride-4 does N/4 with the same simdgroup width.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/sort.rs`
- **Bench filter**: `tile bench -vv -f sort`
- **Shapes / dtypes**: `B=1024 N=1024`, f32/f16/bf16

## Current Code Reality Check

The target kernel `mt_sort` implements a **bitonic sort** for 1024 elements per block:
- `tpg=256`, each thread loads 4 elements into a `threadgroup_alloc("shared", 1024)` buffer.
- Bitonic stages `_k` run from 1 to 10 (log₂(1024)).
- For each stage `_k`, there are `_k` sub-stages (`_jb`).
- Each thread processes 4 elements per sub-stage (`_e` in 0..3).
- Each element does a compare-swap with a partner at distance `2^flip` where `flip = _k - _jb - 1`.
- A `threadgroup_barrier()` is inserted when `flip >= 7` (partner distance ≥ 128, crossing thread boundaries).

Per-thread operation count:
- Total compare-swaps per thread = 4 × Σ(k=1..10) k = 4 × 55 = **220 compare-swaps**.
- Each compare-swap = 2 `threadgroup_load` + 2 `threadgroup_store` = **4 tg memory ops**.
- Total tg memory ops per thread = **880**.

### Baseline numbers

```
$ tile bench -vv -f sort
B=1024 N=1024 f32  Ref=51.9 GB/s  MT=39.4 GB/s  MT%=76%  ok=✓  regs=117r  bottleneck=thread-limited
B=1024 N=1024 f16  Ref=29.0 GB/s  MT=22.4 GB/s  MT%=77%  ok=✓  regs=117r  bottleneck=thread-limited
B=1024 N=1024 bf16 Ref=27.5 GB/s  MT=21.3 GB/s  MT%=77%  ok=✓  regs=117r  bottleneck=thread-limited
```

The kernel is **thread-limited** at 117 registers. Occupancy is 100% because the thread count (256) × registers (117) = 29,952, which fits in the M1 Max register file (per-core limit ~64K–128K depending on config). However, being thread-limited means the ALU is busy but the memory unit may not be fully utilized.

### MLX reference: merge sort, not bitonic

MLX's `sort.metal` / `sort.h` uses **merge sort** (`BlockMergeSort`), not bitonic sort:
1. Each thread loads `N_PER_THREAD=4` elements.
2. Thread-level sort (odd-even sort on 4 elements).
3. Merge steps doubling width: 2→4→8→...→1024 threads.
4. Each merge step does `merge_partition` (binary search) + `merge_step` (k-way merge).

The MLX approach has fewer tg memory ops than bitonic sort for large blocks because merge sort is O(N log N) with better locality (each element is moved log N times with contiguous reads/writes), while bitonic sort's compare-swap pattern has scattered partners.

## What "4-way bitonic merge" means

The hypothesis proposes fusing **4 adjacent compare-swaps** into a single 4-way merge operation when the partner block is contiguous and at distance ≥ 4. Instead of:
```
for _e in 0..4:
  load(gi + _e)
  load(partner + _e)
  compare-swap
  store(gi + _e)
  store(partner + _e)
```
You would:
```
load 4 elements from gi block
load 4 elements from partner block
4-way bitonic merge the 8 elements
store lower 4 to gi block, upper 4 to partner block (or vice versa)
```

This reduces tg memory ops from 16 (4×4) to 8 per merge step when the partner block is contiguous. The catch: this only applies when `flip >= 2` and both the thread's block and the partner block are contiguous 4-element chunks.

## Risk: register pressure

The current kernel already uses **117 registers**. A 4-way merge would need to hold **8 live scalars** (4 local + 4 partner) plus temporaries for the merge network. The risk note says "register usage doubles" — from 117r to ~200r+. On M-series, spill typically begins around 120–128 registers per thread. At 200r, the kernel would spill to threadgroup or device memory, destroying performance.

The DSL has **no register-array type** for holding multiple scalars compactly. You'd need 8 separate scalar variables, each living in a physical register. The compiler may not be able to fold the merge network tightly enough to avoid live-range expansion.

### Can we mitigate?

- **Use `threadgroup` memory as spill** — explicitly stage the 8 values in tg memory instead of registers, then load them back. This defeats the purpose (more tg ops, not fewer).
- **Reduce elements per thread from 4 to 2** — this would lower register pressure but increase thread count requirements (need 512 threads for 1024 elements), which exceeds typical tpg limits.
- **Switch to merge sort** — this is what MLX does, but it's a full algorithm rewrite, not a stride tweak.

## Decision

The 4-way bitonic merge idea is **technically feasible** in the DSL but **high risk** due to register pressure. The current kernel is already at 117r and thread-limited. A register-doubling rewrite is very likely to spill, regressing performance.

The **real** 23% gap vs MLX is structural: MLX uses merge sort, not bitonic sort. A merge-sort port to the DSL would be a separate multi-day idea. The 4-way bitonic merge is a local optimization on the wrong algorithm.

## Risk Register
- **Register pressure doubles** — from 117r to ~200r+, likely spilling. (from perf-ideas.md)
- **Wrong algorithm family** — MLX uses merge sort, not bitonic sort. The 23% gap is algorithmic, not stride-related. (new finding)
- **No array types in DSL** — 8 live scalars require 8 separate variables; compiler may not fold them efficiently. (new finding)
- **Only applies to `flip >= 2` stages** — early bitonic stages (flip=0,1) have adjacent/interleaved partners where 4-way fusion doesn't apply cleanly. (new finding)

## Final Verdict
**⚠️ feasible / high risk / marginal value.**

A 4-way bitonic merge can be written in the DSL but is likely to spill registers given the current 117r baseline. The real performance gap is caused by algorithm choice (bitonic vs merge sort), not compare-stride width. Recommending **abandon** in favor of a potential future "port MLX merge sort to DSL" idea.

## Related Ideas
- **016–020** — Feasibility study (shows that structural changes need careful dispatch analysis).
- **006** — RMS-norm 8-wide unroll (register pressure exploded 9r→162r; same risk pattern).
