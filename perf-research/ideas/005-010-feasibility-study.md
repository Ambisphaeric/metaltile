# Feasibility Study — Ideas 5 through 10

> Quick-wins from `perf-ideas.md`, assessed against current code, codegen, and bench infrastructure.  
> Each idea rated: ✅ feasible (single-file tweak, benchable today) / ⚠️ feasible-with-caveats / 🔴 blocked.

---

## 5. SDPA-vector: 8-wide vectorized loads on f16/bf16

**Target:** `crates/metaltile-std/src/mlx/sdpa_vector.rs`

### Claim
Bumping K/V loads from vec4 to vec8 halves LSU instruction count.

### Current reality
The kernel body loads 4 scalars per lane:
```rust
let d0 = lane * 4u32;
let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
... etc
```
Dispatch is `tpg=1024` (32 simdgroups × 32 lanes). Head dim is hardcoded to 128, so each lane owns exactly 4 elements (128 / 32 = 4).

### Why it's blocked
- **Geometry mismatch:** vec8 loads would need 8 elements per lane → head_dim = 256 (32 × 8). The current bench hardcodes `h=128`.
- **DSL has no vector-load primitive:** `load()` is scalar. There is no `load_vec8<T>()` or `half8` type in the `#[kernel]` DSL.
- **Codegen doesn't auto-vectorize:** `tile inspect mt_sdpa_vector` emits 4 independent scalar loads. The Metal driver *may* merge them at JIT, but there's no guarantee, and we can't force it from DSL source.
- **Metal 3 `half8` exists in MSL but not in DSL:** To use it, you'd either extend the DSL with vector types (multi-day project) or write raw MSL (defeats the purpose of the bench harness).

### What would need to change
| Layer | Change | Effort |
|-------|--------|--------|
| DSL | Add `Vec8<T>` tensor type or `load_vec4`/`load_vec8` intrinsics | Multi-day |
| Codegen | Lower vector loads to `half8`/`float4` MSL | Multi-day |
| Kernel | Rewrite lane mapping + head_dim assumptions | Medium |
| Bench | Add `h=256` shape or change dispatch | Low |

### Verdict
🔴 **Blocked.** Not a single-file tweak. The vector-load primitive doesn't exist. Moving to idea-space 16–35 (DSL/codegen extension).

---

## 6. RMS-norm: unroll 4 → 8

**Target:** `crates/metaltile-std/src/mlx/rms_norm.rs`

### Claim
8-wide unroll hides L1 latency better than 4-wide.

### Current reality
```rust
let x0 = load(x[base]).cast::<f32>();
let x1 = load(x[base + 1u32]).cast::<f32>();
let x2 = load(x[base + 2u32]).cast::<f32>();
let x3 = load(x[base + 3u32]).cast::<f32>();
let partial_ssq = x0*x0 + x1*x1 + x2*x2 + x3*x3;
```
Bench: `b=1024, n=4096, tpg=1024`.  
Each thread processes `4096 / 1024 = 4` elements. The 4-wide unroll is **exactly** the full per-thread workload — there is no loop at all.

### The arithmetic problem
If you go 8-wide while keeping `n=4096` and `tpg=1024`, each thread would need to process 8 elements, but the row only has 4 elements per thread. You'd need to change **either** `n` **or** `tpg`:

| Option | New params | Per-thread work | Loop? |
|--------|-----------|-----------------|-------|
| A. Double N | `n=8192, tpg=1024` | 8 elements | No loop |
| B. Halve TPG | `n=4096, tpg=512` | 8 elements | No loop |
| C. Keep both, add tail | `n=4096, tpg=1024` | 8 elements | Overflow — invalid |

### Register pressure check
Current registers in the kernel: `x0..x3`, `partial_ssq`, `tg_ssq`, `rms`, `w[col]..w[col+3]` → ~12 floats.  
Adding `x4..x7` → ~16 floats. Still comfortably under 64 (f32) or even 128 (f16 intermediates).

### What would need to change
The kernel body edit is trivial — copy-paste 4 more loads/accumulates/stores.  
The bench macro currently hardcodes `n=4096, tpg=1024`. To get a valid 8-wide dispatch, you must adjust one of them.

**Lowest-friction path:** Change `tpg` from 1024 → 512 in the `#[bench_kernel]` macro. This keeps the same `n=4096` workload but gives each thread 8 elements. The bench harness will automatically adjust grid and perf math.

### Risk
- README says RMS-norm is already at **104% of MLX on M4**. If you're already winning, there may be no headroom.
- On M1 (the user's target), there *may* be headroom, but it's speculative.

### Verdict
⚠️ **Feasible with caveat — BUT tested and abandoned.** The kernel edit is a 5-minute copy-paste, and the geometry works with `tpg=512`. However, the actual bench result showed **register pressure exploding from 9r → 162r**, occupancy dropping to **73%**, and the kernel becoming **register-limited**. Throughput regressed by **−50% to −80%** across dtypes. The 8-wide unroll holds too many live values for the Apple GPU register file.

**Abandoned** — see [006-rms-norm-unroll-8.md](006-rms-norm-unroll-8.md) for full data.

---

## 7. Softmax: simdgroup reduce for small N (≤ 32)

**Target:** `mlx/softmax.rs`

### Claim
For N ≤ 32, two-pass threadgroup reduction is overkill; use `simd_max` + `simd_sum` directly.

### Current reality
`tile inspect mt_softmax` shows the generated MSL already uses:
- `simd_max` for the per-simdgroup max
- `simd_sum` for the per-simdgroup sum
- Then a **second level** via threadgroup memory (`v_rm_sg[32]`, `v_rs_sum_sg[32]`) when `n_simd > 1`

The codegen (`crates/metaltile-codegen/src/msl/reduce.rs`) is already optimal: it emits a two-level reduction that **degrades** to single-simdgroup when `n_simd == 1`.

### Why it's mostly a no-op
The bench dispatch for softmax is:
```rust
b=1024, n=4096, tpg=256
```
With `tpg=256`, `n_simd = 8`. The second reduction level is **necessary** because 8 simdgroups each produce a partial max.

To eliminate the second level, you'd need to dispatch with `tpg=32` (exactly one simdgroup). But the bench doesn't test small N. For `n=4096`, even with `tpg=32`, you'd need `4096/32 = 128` iterations per thread to cover the row — the row is huge, the simdgroup count is irrelevant to the *row size*.

Wait — the idea says "for N ≤ 32". That means **row length ≤ 32**, not the current `n=4096`. The current bench doesn't exercise this at all.

### What would need to change
Add a small-N bench shape to the `#[bench_kernel]` macro or `ShapeSpec`. E.g.:
```rust
b=1024, n=32, tpg=32
```
With `tpg=32`, `n_simd=1`, and the codegen's second level still runs but `simd_lane < 1` means only lane 0 participates, and the two `threadgroup_barrier` calls are pure overhead (~10–20 cycles each).

### Risk
- The idea's premise is correct (two barriers are wasteful for 1-simdgroup), but the bench doesn't measure it.
- Adding a small-N shape is easy, but softmax at N=32 is not a realistic workload for LLM inference. It's a microbenchmark win.

### Verdict
⚠️ **Technically correct but bench gap.** The kernel already does `simd_max`/`simd_sum` at the simdgroup level. The "optimization" is avoiding the second level by using `tpg=32` for small rows. To bench it, you'd add a `n=32` shape. Effort: low. Expected win: small (only matters for tiny tensors).

---

## 8. Softmax: float4 loads on f16/bf16 inner loop

**Target:** `mlx/softmax.rs`

### Claim
Load 4 elements as a vector, `exp` in lockstep — saturates the exp unit.

### Current reality
Same as idea 5: the DSL `load()` is scalar. The generated MSL emits 4 scalar loads:
```metal
auto v18 = inp[v_base];
float v_v0 = static_cast<float>(v18);
auto v22 = inp[v_base + 1];
float v_v1 = static_cast<float>(v22);
...
```
For `T=f16`, these are `device half*` loads. Metal may auto-vectorize to `half4` if alignment permits, but we can't verify or force it.

### Why it's blocked
Identical blocker to idea 5: **no vector-load primitive in DSL.**

Even in raw MSL, a `float4` load requires pointer casting:
```metal
half4 v = *(const device half4*)(inp + v_base);
```
The DSL's `Tensor<T>` abstraction doesn't expose this.

### What would need to change
Same as idea 5: DSL vector type extension + codegen lowering.

### Verdict
🔴 **Blocked.** Same root cause as idea 5: DSL lacks vector loads.

---

## 9. LayerNorm: mirror RMS-norm tweaks

**Target:** `mlx/layer_norm.rs`

### Claim
Same structural improvements (unroll 8, simdgroup reduce) apply.

### Current reality
The kernel is structurally identical to RMS-norm but with two accumulators (`s` and `sq`) instead of one (`ssq`):
```rust
let v0 = load(x[base]).cast::<f32>();
... // 4-wide
s = s + v0 + v1 + v2 + v3;
sq = sq + v0*v0 + v1*v1 + v2*v2 + v3*v3;
```
Bench: `b=1024, n=4096, tpg=1024`.

### Feasibility assessment
Same arithmetic as idea 6:
- Current: 1024 threads × 4 elements = 4096 → exactly covers the row, no loop.
- 8-wide with same params: 1024 threads × 8 elements = 8196 → overflows 4096.
- Must adjust `tpg=512` or `n=8192`.

### Register pressure check
Current: `v0..v3`, `s`, `sq`, `mean`, `var`, `is`, `w[col]..w[col+3]`, `b[col]..b[col+3]` → ~18 floats.  
Adding `v4..v7` → ~22 floats. Still safe.

### Verdict
⚠️ **Feasible with caveat.** Same as idea 6: trivial kernel edit, but must adjust `tpg` or `n` to keep geometry valid.

---

## 10. GEMV: tune `simd_per_tg` per K dimension

**Target:** `mlx/gemv.rs`

### Claim
Large K wants 8 simdgroups per TG for latency hiding; small K wants 2–4.

### Current reality
Bench macro:
```rust
b=4096, n=4096, tpg=256
```
The kernel uses `strided_reduce_dot` which codegen lowers to a per-thread scalar loop with 4-wide inner unrolling, then `simd_sum` across simdgroups.

MLX reference pattern: `gemv_{tn}_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`
- BM=4, BN=1 → 4 simdgroups per TG
- SM=1, SN=32 → 32 threads per simdgroup
- Total threads: 4 × 32 = **128**

MetalTile uses **tpg=256** (8 simdgroups). That's already **more** simdgroups than MLX's instantiation.

### What the idea actually means
`tpg` in MetalTile = threads per threadgroup = `simdgroups × 32`.  
"simd_per_tg" = `tpg / 32`.

For K=4096:
- tpg=256 → 8 simdgroups → each thread processes 4096/256 = 16 elements (4 iterations of 4-wide loop)
- tpg=128 → 4 simdgroups → each thread processes 4096/128 = 32 elements (8 iterations)
- tpg=64 → 2 simdgroups → each thread processes 4096/64 = 64 elements (16 iterations)

More iterations per thread = better instruction-level parallelism (latency hiding). Fewer simdgroups = less parallelism across the row. For large K, the loop body dominates, so tpg=128 might actually win over tpg=256.

### How to test it
The `#[bench_kernel]` macro for `class=MatVec` generates a single `ShapeSpec` with one `tpg`. To compare multiple tpgs, you have two options:

**Option A — Duplicate kernel + macro (5 min):**
```rust
#[bench_kernel(... tpg=256 ...)]
#[kernel] pub fn mt_gemv_256<T>(...) { ... }

#[bench_kernel(... tpg=128 ...)]
#[kernel] pub fn mt_gemv_128<T>(...) { ... }
```
The `#[kernel]` bodies are identical; only the bench metadata differs. The bench runner will produce two subop rows (`gemv_256`, `gemv_128`) for direct comparison.

**Option B — Extend macro to accept `tpg` list (multi-day):**
Modify `bench_impl.rs` to accept `tpg=[256,128,64]` and emit multiple `BenchSpec`s. Not worth it for one experiment.

### Risk
- "Too many simdgroups → spill register file shared with kernel" — the generated MSL shows `v_result_sg[32]` (32 floats = 128 bytes) in threadgroup memory, not registers. Register usage is minimal (~5 floats: `v_acc`, loop indices). Spill risk is negligible.
- The win might be small or nonexistent if the bottleneck is memory bandwidth, not ALU latency.

### Verdict
✅ **Feasible.** This is the cleanest Quick-win in the whole set. No kernel body changes at all — just clone the `#[bench_kernel]` block with different `tpg` values and run `tile bench -f gemv -vv`. The bench harness handles everything.

---

## Summary table

| # | Idea | Verdict | Kernel edit? | Bench change? | Blocker |
|---|------|---------|--------------|---------------|---------|
| 5 | SDPA vec8 loads | 🔴 blocked | No (DSL lacks primitive) | — | No vector-load in `#[kernel]` DSL |
| 6 | RMS-norm unroll 8 | ⚠️ feasible | Yes (4 lines copy-paste) | Yes (tpg→512 or n→8192) | Geometry mismatch at current params |
| 7 | Softmax simd reduce small N | ⚠️ feasible | No (already optimal) | Yes (add n=32 shape) | Bench doesn't exercise small N |
| 8 | Softmax float4 loads | 🔴 blocked | No (DSL lacks primitive) | — | No vector-load in `#[kernel]` DSL |
| 9 | LayerNorm mirror #6 | ⚠️ feasible | Yes (copy-paste) | Yes (tpg→512 or n→8192) | Same geometry mismatch as #6 |
| 10 | GEMV tune tpg | ✅ feasible | No | Yes (clone macro with new tpg) | None — pure bench param sweep |

## Recommended next steps

1. **Immediate Quick-win:** Run idea 10. Copy `gemv.rs` to 2–3 variants with `tpg=64,128,256`. Bench all three. No kernel risk.
2. **If 10 shows signal:** Try ideas 6 and 9 together — edit `rms_norm.rs` and `layer_norm.rs` to 8-wide, change `tpg=512` in the macros, bench.
3. **Ideas 5 and 8:** Park as blocked. If vector-load DSL extension ever happens, revisit.
4. **Idea 7:** Add a `n=32` shape to softmax bench if you care about small-tensor perf. Otherwise skip.
