# Perf Idea M7 — Speculative-decode batched-Q SDPA

## Metadata
- **Number**: M7
- **Name**: speculative-decode-batched-q-sdpa
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: ⚠️ feasible (high effort, high impact)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Draft models propose multiple Q tokens at once; KV is shared. A batched-Q SDPA kernel (currently single-Q decode path) unlocks speculative decoding without splitting into N independent dispatches.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/ffai/sdpa_decode.rs`
- **Bench filter**: `tile bench -f sdpa_decode` with batched-Q variant
- **Shapes / dtypes to watch**: batch_q=4, head_dim=128, n_kv=4096, f16

## Assessment

### Current `sdpa_decode` kernel
The kernel processes **one Q token per head**:
```rust
let q_head = tgid_x;      // one threadgroup per Q-head
let q_off = q_head * head_dim;
let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
```

Dispatch: `[n_q_heads, 1, 1]` threadgroups, each with 1024 threads (32 simdgroups × 32 lanes).

### What speculative decoding needs
Speculative decoding uses a draft model to propose K candidate tokens (e.g., K=4). The main model verifies all K tokens in parallel by running SDPA with a **batched Q**:
- Q shape: `[n_q_heads, K, head_dim]` instead of `[n_q_heads, 1, head_dim]`.
- KV cache is shared across all K Q positions (same prefix).
- The attention scores are computed for all K positions simultaneously.

### What a batched-Q kernel would look like
1. **Dispatch**: `[n_q_heads, K, 1]` or `[n_q_heads * K, 1, 1]`.
2. **Per-lane work**: Each lane computes dot products for one of the K Q vectors.
3. **Online softmax**: Each of the K positions maintains its own `(max, sum)` tuple.
4. **V accumulation**: Each of the K positions accumulates its own output vector.
5. **Shared KV loads**: All K Q positions load the same K/V cache entries. This is the key win — KV bandwidth is amortized across K Q positions.

### Implementation approach
Option A: Modify `sdpa_decode` to accept `batch_q` as a `#[constexpr]`:
```rust
#[kernel]
pub fn sdpa_decode_batched<T>(
    q: Tensor<T>,        // [n_q_heads, batch_q, head_dim]
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,      // [n_q_heads, batch_q, head_dim]
    #[constexpr] batch_q: u32,
    // ... rest same
)
```

- Threadgroup size stays 1024.
- Each simdgroup handles a subset of the K positions.
- The KV walk loop is shared — each KV position is loaded once and dot-producted against all K Q positions in the simdgroup.

### KV bandwidth amortization
Current: 1 Q position × N KV positions = N KV loads.  
Batched: K Q positions × N KV positions = N KV loads (shared).  
**K× reduction in KV memory traffic per Q position.**

### Effort estimate
- New kernel variant: **multi-day**.
- Dispatch model changes: `run_spec.rs` needs a new dispatch path for batched decode.
- Bench harness: new `BenchSpec` for batched decode.
- Correctness: verify that online softmax for K parallel streams is numerically stable.
- **Total**: **multi-day to project-scale**.

## Verdict

- **Outcome**: feasible — high effort, high impact for speculative decoding
- **Why**: The current kernel is strictly single-Q. Adding a `batch_q` dimension requires rethinking the dispatch, the per-lane work assignment, and the online-softmax state (K independent max/sum tuples). The KV bandwidth win is real and significant.
- **Note**: This is different from M5 (block-sparse) — they could be combined (batched-Q + block-sparse = maximum decode efficiency).

## Risk Register
- Register pressure: K independent softmax states + K output accumulators increases register usage K×. For K=4, this may push the kernel into register-limited territory.
- SIMD divergence: if K is not a multiple of simdgroup size, some lanes idle.
- Numerics: online softmax with K independent streams must be computed correctly. The rescale factor `exp(max_old - max_new)` is per-stream.

## Notes for Next Person
- Start with K=2 or K=4 (powers of 2 that divide evenly into 32-lane simdgroups).
- The threadgroup memory layout needs K× slots for `tg_max`/`tg_sum`/`tg_out`.
- Dispatch shape: `[n_q_heads, K / simdgroups_per_tg, 1]` if multiple Q positions share a threadgroup, or `[n_q_heads * K, 1, 1]` if each Q position gets its own threadgroup.
- Benchmark against K independent `sdpa_decode` dispatches to measure the win.
