# Perf Idea 045 — `value_sink.rs`: sink threadgroup-memory stores

## Metadata
- **Number**: 045
- **Name**: value-sink-threadgroup-stores
- **Source**: `perf-ideas.md` — Codegen passes (Multi-day)
- **Status**: 🔴 blocked
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Sinking a `threadgroup` store to right before the next barrier shortens the live range and frees a register.

## Target
- **Primary file(s)**: `crates/metaltile-codegen/src/passes/value_sink.rs`
- **Bench filter**: aggregate bench + `regs` column
- **Shapes / dtypes to watch**: kernels with `Op::ThreadgroupStore` followed by `Op::Barrier`

## Assessment

The `value_sink.rs` pass **explicitly does not and cannot** sink `Op::ThreadgroupStore`.

Three reasons:

1. **Side effects**: `is_sinkable` requires `remap::is_cheap_alu(op) && !remap::has_side_effects(op)`. `Op::ThreadgroupStore` is classified as having side effects (`remap::has_side_effects` returns `true` for it). The pass rejects all side-effecting ops.

2. **Not cheap ALU**: `is_cheap_alu` covers `BinOp`, `UnaryOp`, `Cast`, `Select`, `Const`, `ProgramId`. Memory stores are never "cheap ALU" — they access threadgroup address space and have ordering semantics visible to other threads.

3. **Register misconception**: Threadgroup stores do not "hold registers". The store instruction writes a value to threadgroup memory; the value being stored may live in a register, but sinking the store instruction itself does not change the live range of that value. If the value is a single-use cheap ALU op, `value_sink.rs` already sinks that ALU op closer to the `ThreadgroupStore`, which is the correct way to shorten the value's live range.

Moving a `ThreadgroupStore` across other instructions would be **unsafe** because it changes the happens-before relationship with respect to other threads. The only valid motion for a threadgroup store is toward a barrier, and even that requires proving no other thread reads the written location before the barrier — a whole-program analysis that `value_sink.rs` is not designed to perform.

## Verdict

- **Outcome**: blocked — target pass is not the right mechanism; hypothesis describes an unsafe transformation
- **Why**: `value_sink.rs` sinks cheap ALU ops, not memory stores. Threadgroup-store motion requires a separate, barrier-aware pass with inter-thread alias analysis. The hypothesized benefit (freeing a register) is also incorrect — the pass already shortens register live ranges by sinking the ALU ops that *feed* threadgroup stores.

## Risk Register
- Barrier semantics — must preserve happens-before across threads. Correctly identified in perf-ideas.md, but the risk makes this impossible in `value_sink.rs`.
- A new pass for threadgroup-store scheduling would need:
  - Alias analysis on threadgroup memory indices.
  - Proof that no intervening op between the store and the barrier touches the same location.
  - Handling of divergent control flow (what if the store is inside an `Op::If`?).

## Notes for Next Person
- If register pressure around threadgroup stores is a real problem, the fix is almost certainly in the MSL generator or in a new, specialized pass — not in `value_sink.rs`.
- Check whether the MSL generator already places `threadgroup_store` instructions close to their consumers. Since MetalTile emits `auto` variables and the Metal compiler does its own scheduling, the driver may already handle this.
