# 026 — Sampling: radix-select top-k

## Metadata
- **Number**: 026
- **Name**: sampling-radix-select-topk
- **Source**: `perf-ideas.md` § Op-level structural changes — item 26
- **Status**: 🔴 blocked
- **Worktree**: —
- **Assignee**: pi

## Hypothesis
> current top-k probably sorts then slices; radix-select is O(N) for fixed k.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/sampling.rs`
- **Bench filter**: `tile bench -vv -f sampling`
- **Shapes / dtypes**: N=152K vocab, k≤64

## Current Code Reality Check

The target file `sampling.rs` contains a **single** kernel:

**`softmax_categorical_sample`** — implements softmax normalization + categorical (inverse-CDF) sampling:
1. Cooperative `simd_max`-style tree reduction to find logit max (256 threads, 8 stages).
2. Cooperative `simd_sum`-style tree reduction to compute `sum(exp(logit - max))`.
3. Single-thread inverse CDF walk: iterate over all `n` elements, accumulate `exp(logit - max)`, stop when cumulative probability ≥ `uniform * total`.

This is **not** a top-k kernel. It is a random-sampling kernel used for temperature-based token selection. There is no sort, no slice, and no top-k selection anywhere in the file.

### Bench status

The kernel is already registered with `BenchSpec { op: "sampling", subop: "softmax_categorical_sample" }`. `tile bench -f sampling` works and runs this kernel. The idea says "add a `tile bench -f sampling` case if missing" — the case exists, but it is **categorical sampling**, not top-k.

### No top-k in MLX reference either

Searching MLX's cached Metal kernels reveals **no dedicated top-k Metal kernel**. MLX likely implements top-k at the framework level using existing GPU primitives (argmax, sort, or a custom op in Python/C++). There is no `.metal` top-k reference to port.

## Baseline
Not benched — analytical assessment only. The target file does not contain top-k.

## Risk Register
- **Target mismatch** — `sampling.rs` contains categorical sampling, not top-k. The idea assumes a top-k kernel exists in the target file. (new finding)
- **New kernel + new bench harness required** — implementing radix-select top-k is a new kernel body plus a new `run_spec.rs` dispatch arm and benchmark inputs. (from established patterns)
- **No MLX reference** — MLX does not ship a top-k Metal kernel, so there is no reference implementation to port or compare against. (new finding)
- **Radix-select complexity** — radix-select on GPU requires multi-pass histograms and prefix sums across threadgroups. It is a non-trivial kernel, not a single-file tweak. (from perf-ideas.md risk)

## Final Verdict
**Blocked / new kernel required.**

The target file does not contain a top-k kernel. It contains a categorical-sampling kernel (`softmax_categorical_sample`), which is a different operation entirely. Implementing radix-select top-k would require:
1. A new `#[kernel]` implementing radix-select or bitonic-top-k.
2. A new bench harness in `run_spec.rs` (top-k needs `logits` + `k` inputs, not `uniform` + `temperature`).
3. A correctness check against a CPU reference (sort-then-slice).

This is a **multi-day** effort, not a single-file tweak. MLX does not have a top-k Metal kernel either, confirming this is not a ported reference path.

## Related Ideas
- **015** — argmax (argmax is a building block for some top-k implementations, but not sufficient alone).
- **025** — Sort 4-way bitonic merge (sort is an alternative top-k building block — sort-then-slice).
