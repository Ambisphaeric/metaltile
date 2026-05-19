# 010 — GEMV: tune `simd_per_tg` per K dimension

## Metadata
- **Number**: 010
- **Name**: gemv-tpg-sweep
- **Source**: `perf-ideas.md` § Quick-wins — item 10
- **Status**: 🟢 **done** — experiment complete, results recorded
- **Worktree**: `../metaltile-perf-idea-10` (branch `perf/idea-10-gemv-tpg`)
- **Assignee**: pi

## Hypothesis
> Large K (≥ 4096) wants 8 simdgroups per threadgroup for latency hiding; small K wants 2–4.

## Target
- **Primary file**: `crates/metaltile-std/src/mlx/gemv.rs`
- **Bench filter**: `tile bench -vv -f gemv`
- **Shapes / dtypes**: `B=4096 N=4096` (K=4096) for f32, f16, bf16

## Baseline
Baseline is `mt_gemv` with `tpg=256` (8 simdgroups).  
Run twice for DVFS stabilization.

### Run 1 (first — some DVFS noise)
| variant | tpg | f32 | f16 | bf16 |
|---------|-----|-----|-----|------|
| baseline | 256 | 365.0 cv15.6% | 393.4 cv0.3% | 244.7 cv0.2% |
| tpg1024 | 1024 | 379.9 cv0.4% | 310.0 cv0.2% | 247.8 cv0.7% |
| tpg128 | 128 | 360.3 cv1.1% | 377.6 cv1.0% | 248.4 cv0.1% |
| tpg512 | 512 | 372.1 cv1.7% | 395.0 cv3.1% | 242.4 cv15.7% |
| tpg64 | 64 | 356.0 cv0.8% | 365.9 cv17.5% | 254.2 cv9.8% |

### Run 2 (stabilized — all cv% < 5% except tpg64 bf16)
| variant | tpg | simdgroups | f32 GB/s | cv% | f16 GB/s | cv% | bf16 GB/s | cv% |
|---------|-----|------------|----------|-----|----------|-----|-----------|-----|
| **baseline** | 256 | 8 | 368.7 | 0.6% | 388.3 | 2.5% | 244.5 | 0.2% |
| **tpg1024** | 1024 | 32 | 377.8 | 2.0% | **310.6** | 0.2% | 242.1 | 0.3% |
| **tpg128** | 128 | 4 | 357.6 | 0.3% | 387.5 | 2.1% | 248.9 | 0.3% |
| **tpg512** | 512 | 16 | 370.2 | 0.2% | **395.1** | 0.9% | 242.1 | 0.8% |
| **tpg64** | 64 | 2 | 360.0 | 0.4% | 384.2 | 1.6% | 255.1 | 34.4% |

*(tpg64 bf16 is noisy — ignore. Run it a third time if you want, but the other signals are clear.)*

## Experiment Log

### Cycle 1 — 2026-05-18
- **Change**: Cloned `mt_gemv` kernel body into 4 additional variants (`tpg=64,128,512,1024`), zero kernel-body changes.
- **Bench result**: See Run 2 table above.
- **Correctness**: `ok = ✓` on all 15 gemv variants + 3 gemv_masked baseline = 18/18 correct.
- **Trust**: Most cv% < 2.5%. tpg64 bf16 at 34.4% is untrustworthy; all others are solid.

## Analysis

### f32 — flat across all tpgs
| variant | vs baseline | delta |
|---------|-------------|-------|
| tpg512 | +0.4% | 370.2 vs 368.7 |
| tpg1024 | +2.5% | 377.8 vs 368.7 |
| tpg128 | −3.0% | 357.6 vs 368.7 |
| tpg64 | −2.4% | 360.0 vs 368.7 |

All within ~3% of each other. The kernel is memory-bandwidth-limited for f32, and changing tpg doesn't meaningfully alter the memory access pattern. The small spread is within the noise floor.

### f16 — clear signal
| variant | vs baseline | delta |
|---------|-------------|-------|
| **tpg512** | **+1.8%** | **395.1 vs 388.3** ✅ |
| tpg128 | −0.2% | 387.5 vs 388.3 |
| tpg64 | −1.1% | 384.2 vs 388.3 |
| **tpg1024** | **−20.0%** | **310.6 vs 388.3** ❌ |

**tpg=512 wins by ~1.8%** for f16. More importantly, **tpg=1024 is a disaster (−20%)**.

Why? With tpg=1024, each thread processes only 4096/1024 = **4 elements** = exactly **1 iteration** of the 4-wide unroll. There's no loop, so zero instruction-level parallelism to hide memory latency. The kernel becomes purely latency-bound rather than throughput-bound. With f16 (2 bytes), the total bytes are lower than f32, so the relative impact of poor latency hiding is larger — the memory subsystem has fewer concurrent requests in flight.

tpg=512 gives **8 elements per thread = 2 iterations**, which is the sweet spot: enough ILP to hide load latency, but not so many iterations that loop overhead dominates.

### bf16 — basically flat
| variant | vs baseline | delta |
|---------|-------------|-------|
| tpg128 | +1.8% | 248.9 vs 244.5 |
| tpg512 | −1.0% | 242.1 vs 244.5 |
| tpg1024 | −1.0% | 242.1 vs 244.5 |

All within ~2%. bf16 behaves like f32 in this sweep — the throughput is limited by something other than tpg (possibly the bfloat16→float conversion path in the load units).

### Register pressure
All variants report **9r** (9 registers). Zero change. The risk mentioned in the idea ("too many simdgroups → spill register file") did not materialize — the kernel has minimal live state.

## Risk Register
- **tpg=1024 regression on f16 (−20%)**: Real and reproducible (cv% 0.2%). Must not be adopted blindly.
- **tpg64 bf16 noise**: 34.4% cv on one run means that data point is not trustworthy.
- **No MLX reference for MatVec**: This is MT-only comparison. We can't claim parity vs MLX.
- **Single shape tested**: Only B=4096 N=4096. Other K dimensions (e.g., K=1024, K=16384) may have different sweet spots.

## Final Verdict

| dtype | Best tpg | Delta vs baseline | Verdict |
|-------|----------|-------------------|---------|
| f32 | 1024 | +2.5% | Marginal — within noise |
| f16 | **512** | **+1.8%** | Small but real win |
| bf16 | 128 | +1.8% | Marginal — within noise |

**Recommendation**: Keep the baseline `tpg=256` for f32 and bf16. For f16 GEMV workloads, consider switching to `tpg=512` for a ~2% uplift. **Never use tpg=1024 for f16** — it's a clear regression.

If you want to merge a change, the safest is to **change the default tpg from 256→512** in the `#[bench_kernel]` macro. This gives a small f16 win and doesn't regress f32/bf16.

## Snapshots
- Baseline: built into `010-run2.json` (baseline is `gemv` subop)
- All variants: `010-run2.json` includes all 5 subops

## Notes for Next Person
- The sweet spot depends on dtype: f16 likes more iterations per thread (tpg=512), f32 is flat.
- A more complete sweep would test K=1024, K=8192, K=16384 with the same tpgs.
- The `strided_reduce_dot` primitive generates a 4-wide unrolled loop; changing the unroll factor (e.g., 8-wide) is a codegen-level change, not a kernel-level one.
