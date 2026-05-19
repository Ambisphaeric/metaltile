# 005 — SDPA-vector: 8-wide vectorized loads on f16/bf16

## Metadata
- **Number**: 005
- **Name**: sdpa-vec8-loads
- **Source**: `perf-ideas.md` § Quick-wins — item 5
- **Status**: 🔴 **blocked** — DSL lacks vector-load primitive
- **Worktree**: —
- **Assignee**: —

## Hypothesis
> Currently loads vec4 of f16. Bumping to vec8 (`half8`) halves the LSU instruction count.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/sdpa_vector.rs`
- **Bench filter**: `tile bench -vv -f sdpa_vector`
- **Shapes / dtypes to watch**: `H=32 N=4096 D=128 gqa=4`

## Current Code Reality Check
The kernel body loads 4 scalars per lane:
```rust
let d0 = lane * 4u32;
let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
```

Head dim is hardcoded to 128, so each lane owns exactly 4 elements (128 / 32 = 4). The bench hardcodes `h=128`.

### Why it's blocked
1. **Geometry mismatch:** vec8 loads would need 8 elements per lane → head_dim = 256 (32 × 8). The current bench hardcodes `h=128`.
2. **DSL has no vector-load primitive:** `load()` is scalar. There is no `load_vec8<T>()` or `half8` type in the `#[kernel]` DSL.
3. **Codegen doesn't auto-vectorize:** `tile inspect mt_sdpa_vector` emits 4 independent scalar loads. The Metal driver *may* merge them at JIT, but there's no guarantee, and we can't force it from DSL source.
4. **Metal 3 `half8` exists in MSL but not in DSL:** To use it, you'd either extend the DSL with vector types (multi-day project) or write raw MSL (defeats the purpose of the bench harness).

### What would need to change
| Layer | Change | Effort |
|-------|--------|--------|
| DSL | Add `Vec8<T>` tensor type or `load_vec4`/`load_vec8` intrinsics | Multi-day |
| Codegen | Lower vector loads to `half8`/`float4` MSL | Multi-day |
| Kernel | Rewrite lane mapping + head_dim assumptions | Medium |
| Bench | Add `h=256` shape or change dispatch | Low |

## Risk Register
- **D must be divisible by 8** (D=128 is fine). From original idea.
- **Requires Metal 3** — but we're already on Metal 3.1. From original idea.
- **Scope creep:** "halve LSU instruction count" assumes the driver doesn't already merge the 4 scalar loads. Need `tile inspect` + disassembly to confirm.

## Final Verdict
**Blocked.** Not a single-file tweak. The vector-load primitive doesn't exist in the DSL. Moving to idea-space 36–55 (DSL/codegen extension).

## Related Ideas
- **008** — Softmax float4 loads (same blocker)
- **018** — KV-cache vectorized copy (same blocker)
- **M1** — ML autotuner (would benefit from vector types once they exist)
