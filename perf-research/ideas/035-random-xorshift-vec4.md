# Perf Idea 035 — random: 64-bit state, vec4 generation

## Metadata
- **Number**: 035
- **Name**: random-xorshift-vec4
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: 🔴 blocked / ill-formed
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> 32-bit xorshift wastes a register that could hold the high 32 bits; vec4 generation amortizes constant load.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/random.rs`
- **Bench filter**: `tile bench -f random`
- **Shapes / dtypes to watch**: n=1048576, tpg=1024

## Assessment

### Current `mt_random_hash` implementation
```rust
let gid = program_id::<0>();
let mut s = gid + 1u32;
s = s ^ (s << 13u32);
s = s ^ (s >> 17u32);
s = s ^ (s << 5u32);
store(out[gid], s);
```

This is a **toy hash function**, not xorshift32. It generates one `uint32` per thread from the thread ID. The PRNG state is the thread ID itself — there is no persistent state.

### Why the hypothesis is ill-formed
1. **"32-bit xorshift"**: The kernel does not use xorshift. It uses a 3-round hash of `gid`. There is no PRNG state to widen to 64 bits.
2. **"vec4 generation"**: The kernel stores a scalar `uint32`. Generating 4 values per thread would require 4 stores or a vector store. The DSL has no `uint4` type or vector store primitive (same blocker as ideas #5, #8).
3. **"amortizes constant load"**: There are no constants loaded — the only input is `gid`.

### MLX reference comparison
MLX's random generator (`random.metal`) uses **Threefry2x32**, a cryptographic-grade counter-based PRNG. It is a completely different algorithm from the toy hash in MetalTile's benchmark. The MLX kernel takes keys and counters as inputs, supports batched generation, and handles non-power-of-2 sizes.

MetalTile's `mt_random_hash` is a **benchmark stub**, not a production random generator. Its sole purpose is to measure the throughput of a trivial elementwise kernel.

### What a real random kernel would need
1. A proper PRNG algorithm (Threefry2x32, Philox, or LCG).
2. A key/counter input buffer.
3. Vector generation (4× or 8× per thread) for efficiency.
4. Non-power-of-2 handling.

This is a new kernel, not a tweak to the existing one.

## Verdict

- **Outcome**: blocked — hypothesis describes a different kernel than what exists
- **Why**: `mt_random_hash` is a toy hash benchmark, not a PRNG. The optimizations described (64-bit state, vec4) are irrelevant to the current code and would require a complete rewrite.
- **Re-scope**: A proper random number generator kernel is a one-day to multi-day effort, but it has nothing to do with the current `mt_random_hash`.

## Risk Register
- (not applicable — blocked by ill-formed hypothesis)

## Notes for Next Person
- If you need a production random generator, port MLX's Threefry2x32 (`random.metal`).
- If you just want to make the benchmark faster, the current kernel is already memory-bandwidth-bound (one load of `gid` implicit, one store). There's no ALU headroom to exploit.
