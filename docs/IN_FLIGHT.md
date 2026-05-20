# MetalTile In-Flight Snapshot

Captured: 2026-05-19 · Branch: `dev` · Repo state: dev is ~30 commits ahead of `main`

This is a point-in-time inventory of open PRs, open issues, and recently
merged work, used as input to `RELEASE_PLAN.md`. It will go stale fast —
treat it as a snapshot, not a living document. The companion live source is
`gh pr list` / `gh issue list`.

---

## Open PRs (7)

Sorted by readiness for the v0.1.0 cut. Status reflects CI rollup at capture
time.

### Ready to land — v0.1.0 candidates

| PR | Title | Author | CI | LOC | Notes |
|---:|---|---|---|---|---|
| #59 | `perf(codegen): compile-time single-simdgroup specialization for Reduction-mode reduce` | ekryski | in progress (typos/format/clippy/tests/kernels running, title passed) | +359 / −48 | Eliminates threadgroup barriers when TPG ≤ 32; shrinks compiled kernel size for small-N reductions like `rms_norm_small`. Land if CI completes green. |
| #56 | `perf(mlx): mt_qmm_bm2 — BM=2 W-reuse + selector` | TheTom | all green | +1,079 / −2 | 1.2–1.3× over v2 baseline across M=2..32. Independent landing; #57 stacks on top. |
| #42 | `feat(cli): bench dirty-tree guard + auto-diff vs target-branch baseline` | Ambisphaeric | all green | +715 / −182 | CLI ergonomics for `tile bench`. Useful for the release reproducer flow described in `RELEASE_PLAN.md`. |
| #37 | `perf: non-consecutive store vectorization via Pack op` | 0xClandestine | all green | +733 / −591 | New `Pack` op lets LLVM vectorize scattered stores. Codegen-level perf win across multiple kernels. |

### Conditional — land in v0.1.0 only if cleanup happens before cut

| PR | Title | Author | CI | LOC | Notes |
|---:|---|---|---|---|---|
| #57 | `perf(mlx): mt_qmm_bm4 — BM=4 hand-unroll closes M3+ MLX gap` | TheTom | all green, but draft | +1,816 / −2 | Stacks on #56. If #56 lands and #57 goes ready + green, include. Otherwise defer to first patch release or v0.2. |

### Defer to v0.2.0

| PR | Title | Author | CI | LOC | Reason for defer |
|---:|---|---|---|---|---|
| #58 | `feat: metaltile-model — TOML-based model definition system` | 0xClandestine | format + commits FAILING; others green; draft | +2,244 / −0 | New crate, large surface area, single example model (`llama_decode.toml`), failing required checks. Belongs to the v0.2 "production-quality preview" cut where it can ship with more models and polish. |
| #46 | `refactor(bench): collapse BenchDispatch, extract fn-ptr hooks, single generic runner` | 0xClandestine | not run; draft | +715 / −2,185 | Delete-heavy refactor with no CI signal. Useful work but not on the v0.1 critical path; risk of bench breakage near the release cut is not worth it. |

---

## Open issues (2)

Issues are sparse — no `release-blocker`, `v0.1`, or `breaking` labels in
the repo, no GitHub milestones defined.

- **#55** — *Tier 1 follow-up: mt_qmm v3 (BM-tile W-reuse) + M2 f16 SDPA
  prefill tuning.* Quantization perf follow-on to PRs #56/#57. Planned as
  v0.2 workstream-5 in `RELEASE_PLAN.md`.
- **#53** — *Codegen: brittle per-Op match dispatch in `fused.rs` /
  `helpers.rs` — maintainable alternative.* Maintenance / structural cleanup
  in the codegen crate. Planned as v0.2 workstream-4.

---

## Recently merged (last 30 days)

27 PRs merged 2026-05-17 through 2026-05-19 — extremely high recent
velocity. Categorized:

### Kernel features and perf (10)

#52, #51, #50, #47, #44, #43, #35, #34, #30, #15.
Highlights:

- **#47, #51, #52** — Flash-Attention-2 prefill kernels (`mt_sdpa_prefill`)
  beating MLX on the 6 measured prefill cells; subsequent kernel-side B>1
  and long-T coverage.
- **#50** — sliding-window + sink-token SDPA decode specialization,
  4–8× at long sequences.
- **#43, #44** — `sdpa_vector` and `mt_qmv<T>` beating MLX across
  Qwen3 shapes; bf16 +191% on vector decode.
- **#34, #35** — Apple GPU family runtime detection + production-grade
  `sdpa_decode` 2-pass with `dispatch_chain`.

### Codegen / runtime perf (9)

#54, #48, #45, #40, #39, #38, #36, #20, #14.
Highlights: PSO cache key cleanup, FxHashMap migrations across CSE / fusion
/ dead-store-elim passes, large bench-output redesign moving GPU runner to
`metaltile-std` and CLI to `clap`.

### Tests & CI (6)

#41, #33, #31, #29, #28, #27.
Highlights: ratchet-blocking codecov gates, kernel-job split into its own
path-filtered workflow, trybuild compile-fail tests for `#[kernel]`,
expanded coverage on `constexpr`/`dtype`/`shape`/`buffer`/`autotune`.

### Docs / chore (12)

#25, #24, #23, #22, #21, #19, #18, #17, #16, #10, #9, #8, #7.
Highlights: MSL golden snapshots via `insta`, cargo-llvm-cov workflow,
M5 Max baseline + cross-dev baselines dir, canonical per-crate READMEs,
deletion of the dead `metaltile-interp` crate, commit-message hygiene
workflow.

---

## Branch & release state

- `dev` is the integration branch and is ahead of `main` by ~30 commits.
  Per `CONTRIBUTING.md`, the release workflow is `dev → main` PR, then tag.
- `main` has 2 commits not in `dev` (legacy FFAI cleanup); rebase / merge
  cleanup is part of the pre-tag work in `v0.1.0_CHECKLIST.md`.
- No git tags exist (`git tag -l` is empty).
- No GitHub releases exist.
- Working tree on `dev` has 4 untracked paths:
  `.ralph/`, `crates/metaltile-bench/`, `metaltile-planning/`, `results/`.
  Disposition of each is an item in `v0.1.0_CHECKLIST.md`.

---

## Active contributors (last 30 days)

| Account | Merged PRs | Area |
|---|---:|---|
| TheTom | 16 | Kernel design, MLX-parity perf, codegen optimization |
| 0xClandestine | 7 | Architecture, infra refactoring, model definition |
| Ambisphaeric | 3 | Perf tuning, benchmarking harness |
| ekryski | 1 | Kernel specialization |

Total: 4 active contributors, no external/community PRs in the window.
This bears on release messaging: v0.1.0 is the first public artifact from a
small core team, not a project with an existing public-contributor base.
