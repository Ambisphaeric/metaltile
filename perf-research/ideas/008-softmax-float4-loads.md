# 008 — Softmax: float4 loads on f16/bf16 inner loop

## Metadata
- **Number**: 008
- **Name**: softmax-float4-loads
- **Source**: `perf-ideas.md` § Quick-wins — item 8
- **Status**: 🔴 **blocked** — same root cause as idea #5
- **Worktree**: —
- **Assignee**: —

## Hypothesis
> Load 4 elements as a vector, exp in lockstep — should saturate exp unit.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/softmax.rs`
- **Bench filter**: `tile bench -vv -f softmax`
- **Shapes / dtypes to watch**: `B=1024 N=4096 f16/bf16`

## Current Code Reality Check
Same blocker as idea #5. The DSL `load()` is scalar. The generated MSL emits 4 scalar loads:
```metal
auto v18 = inp[v_base];
float v_v0 = static_cast<float>(v18);
auto v22 = inp[v_base + 1];
float v_v1 = static_cast<float>(v22);
...
```

For `T=f16`, these are `device half*` loads. Metal may auto-vectorize to `half4` if alignment permits, but we can't verify or force it.

### Why it's blocked
Identical blocker to idea #5: **no vector-load primitive in DSL.**

Even in raw MSL, a `float4` load requires pointer casting:
```metal
half4 v = *(const device half4*)(inp + v_base);
```
The DSL's `Tensor<T>` abstraction doesn't expose this.

### What would need to change
Same as idea #5: DSL vector type extension + codegen lowering.

## Risk Register
- **exp accuracy across vector lanes** — original idea says "no real risk on Metal". True, but irrelevant since we can't express vector loads.

## Final Verdict
**Blocked.** Same root cause as idea #5: DSL lacks vector loads.

## Related Ideas
- **005** — SDPA vec8 loads (same blocker)
- **018** — KV-cache vectorized copy (same blocker)
