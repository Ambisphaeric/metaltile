# Perf Idea 030 — `binary_two`: FMA autovec diagnostic

## Metadata
- **Number**: 030
- **Name**: binary-two-fma-autovec
- **Source**: `perf-ideas.md` — Op-level structural changes (One-day)
- **Status**: ⚪ no-op / marginal
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> `fma` should auto-emit. If MT% lags MLX, codegen is missing it; inspect MSL.

## Target
- **Primary file(s)**: `crates/metaltile-std/src/mlx/binary_two.rs`
- **Bench filter**: `tile bench -f binary_two`
- **Shapes / dtypes to watch**: any — this is elementwise, bandwidth-bound

## Assessment

### Kernel structure
```rust
let x = load(a[idx]);
let y = load(b[idx]);
store(c[idx], x + y);
store(d[idx], x * y);
```

This computes **two independent results**: `a+b` and `a*b`, stored to two separate output tensors. There is **no FMA pattern** in this kernel. FMA (fused multiply-add) computes `a * b + c` in a single ALU instruction. The kernel computes addition and multiplication separately, with separate stores.

### Why FMA is irrelevant here
1. **No multiply-add chain**: The kernel has `x+y` and `x*y` as independent expressions. No expression combines both operations.
2. **Bandwidth-bound**: Two loads + two stores = 4 memory ops per element. Even if FMA were present, the kernel is memory-bandwidth-limited, not ALU-limited.
3. **MLX reference**: The MLX `binary_two` kernel has the same structure (two independent ops, two stores). There is no FMA to emit.

### The real question
The hypothesis seems to conflate `binary_two` (two-output elementwise) with a hypothetical fused multiply-add kernel. If the intent was "does MetalTile codegen emit `fma` when the IR contains `a*b+c`?", then the target kernel is wrong — `binary_two` never produces that IR.

A correct diagnostic would need a kernel like:
```rust
store(out[idx], a[idx] * b[idx] + c[idx]);
```

No such kernel exists in `binary_two.rs`.

## Verdict

- **Outcome**: no-op — target kernel does not contain an FMA pattern
- **Why**: `binary_two` computes `x+y` and `x*y` independently, not `x*y+z`. There is no `fma` opportunity. The kernel is bandwidth-bound regardless.
- **Note**: If someone wants to verify whether MetalTile codegen emits `fma` for `a*b+c`, they need a different test kernel.

## Risk Register
- (none — nothing to change)

## Notes for Next Person
- Don't chase FMA in bandwidth-bound kernels. FMA only matters when ALU is the bottleneck (GEMM, GEMV, dense matmul).
- If you want to test codegen FMA emission, write a minimal `#[kernel]` with `a*b+c` and `tile inspect` the MSL.
