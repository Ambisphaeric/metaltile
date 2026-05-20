# MetalTile Release Plan — v0.1.0 → v0.2.0

Owner: maintainers · Last updated: 2026-05-19 · Status: proposal

This document defines the scope, themes, and exit criteria for the first two
public releases of MetalTile. Companion files:

- `IN_FLIGHT.md` — snapshot of open PRs and issues at the time of writing.
- `v0.1.0_CHECKLIST.md` — concrete pre-tag checklist for the v0.1.0 cut.

---

## Project positioning at v0.1.0

MetalTile is a Rust-embedded DSL that lowers tile-level kernels to Apple Metal
Shading Language. Today, the workspace declares version `0.1.0` for every
publishable crate (`Cargo.toml`), but no tag, no GitHub release, and no
`crates.io` publish has happened yet. The repo has high merge velocity
(27 PRs in the three days before this plan was written), a small but active
contributor base of four, and a 241-row benchmark suite that is numerically
verified vs. MLX on both M4 Max and M5 Max.

**v0.1.0 is positioned as a "preview" release**, not a stable API commitment.
It is the first public, taggable, install-from-`crates.io` artifact of the
project. The README already states "Early development — APIs are not yet
stable" (`README.md:52`), and v0.1.0 will keep that posture explicit.

**v0.2.0 is positioned as the "production-quality preview"**: a still-pre-1.0
release that closes the hot-path performance gaps, ships the model-definition
layer, and turns the autotuner from a stub into a working component. v0.2.0 is
the version we would ask third-party users to take seriously for evaluation.

A 1.0 commitment is explicitly out of scope for both releases.

---

## v0.1.0 — "First taggable cut"

### Theme

Get what already works in front of users without overpromising. The bulk of
the work is **finishing**, not building — landing in-flight PRs that have
already passed CI, aligning metadata so `cargo publish` works, and writing a
release that is honest about its perf cliffs and its API instability.

### What ships

Published to `crates.io` (in dependency order):

1. `metaltile-core`
2. `metaltile-macros`
3. `metaltile-codegen`
4. `metaltile-runtime`
5. `metaltile-std`
6. `metaltile` (facade re-export)

Plus the `tile` CLI binary, distributed as:

- A `cargo install metaltile-cli` path (crate kept `publish = false` until
  v0.2 — see `crates/metaltile-cli/Cargo.toml`).
- A GitHub Release with a prebuilt `tile` binary for macOS arm64.

### Scope inclusions

- **In-flight perf PRs that have passed CI** — see `IN_FLIGHT.md`. Concretely
  this means landing #37 (Pack-op vectorization), #42 (bench dirty-tree guard
  + auto-diff), #56 (BM=2 qmm with kernel selector), and #59 (single-simdgroup
  reduce specialization, contingent on green CI).
- **Baselines refreshed** on M4 Max and M5 Max as the final pre-tag step, so
  the release README links to a baseline JSON captured from the release SHA.
- **Honest performance documentation**. Headline number stays
  "~110% of MLX average on M4 Max." The README and a new
  `docs/PERFORMANCE.md` (optional, can defer) explicitly call out the M5 Max
  SDPA+GQA+bf16 regression (31% of MLX) and the softmax bf16 regression
  (29% of MLX) as known structural issues targeted for v0.2.
- **Release machinery proven end-to-end**: `git tag v0.1.0` → `cargo publish`
  → GitHub Release with binary → install-from-crates-io smoke test on a
  clean macOS box.

### Scope exclusions (deferred to v0.2.0)

- **PR #58 `metaltile-model`** — 2,244-line new crate for TOML-based model
  definitions. Compelling, but it is currently a draft with failing format /
  commit-hygiene checks and only one model file. Shipping it in v0.1 would
  add a second large surface area to "preview." Defer and let it bake.
- **PR #46 bench dispatch refactor** — draft, no CI run, large delete-heavy
  diff. Defer.
- **PR #57 BM=4 qmm** — depends on #56, currently draft. Land in v0.1 only if
  it goes ready and green before the cut date; otherwise defer.
- **Autotuner activation** — `metaltile-runtime/src/autotune.rs` carries a
  literal `TODO: actually run tuning` and the README/CONTRIBUTING already
  describe the autotuner as v0.2 work.
- **Hot-path SDPA/softmax bf16 fixes** — structural (per-shape block size
  tuning); not a one-PR change. Owned by v0.2.
- **MSRV policy** — currently nightly-only for edition=2024. A stable-toolchain
  MSRV is a v0.2 conversation.
- **No new public APIs.** v0.1.0 freezes the API surface that already exists
  in `dev`. Anything that would expand the public surface (e.g. new crate,
  new top-level macro) belongs in v0.2.

### Exit criteria

1. All v0.1-scope PRs merged to `main` via `dev → main` per `CONTRIBUTING.md`.
2. `cargo test --workspace`, `make clippy`, and `make typos` all green on the
   release commit. The kernels job on macOS is green (no regressed cells vs.
   the baseline).
3. README, CONTRIBUTING, and crate-level Cargo.toml metadata (repository,
   homepage, keywords, categories, description) are consistent and point at
   the actual canonical GitHub URL.
4. `CHANGELOG.md` exists at the repo root, with a populated `## v0.1.0`
   section. (See `v0.1.0_CHECKLIST.md` for content rules.)
5. `cargo publish --dry-run` succeeds for every publishable crate from a
   clean clone of the release tag.
6. A reproducer transcript exists in the release notes: "fresh macOS box →
   `cargo install metaltile-cli` → `tile bench --filter rms_norm` → verified
   matches a published cell." This is the trust-but-verify gate before the
   actual `cargo publish` push.

### Non-goals (explicit)

- We are **not** waiting for the M5 Max SDPA regression to be fixed.
- We are **not** waiting for the autotuner to become functional.
- We are **not** promising API stability. A breaking change between 0.1.x
  and 0.2.0 is expected and explicitly allowed by semver pre-1.0.

---

## v0.2.0 — "Production-quality preview"

### Theme

Close the gaps that make the v0.1.0 release "preview-only" and ship the
model-definition story. After v0.2.0, MetalTile should be a credible
"please evaluate this" tool, even if 1.0 is still out.

### Workstreams

Five parallel-friendly workstreams, in rough priority order:

1. **Hot-path perf parity with MLX on M5 Max** — the headline regression
   is SDPA + GQA + bf16 at 31% of MLX. This is structural: MLX uses
   `sdpa_vector_2pass` with per-shape block tuning; MetalTile uses a fixed
   single-pass kernel. The fix is per-shape dispatch (likely a follow-on
   to issue #55), and is the highest-impact perf work in the project.
   Secondary targets: softmax bf16 (29% MLX), quantized affine bits=3–4
   (24–41% MLX).

2. **Autotuner v1** — turn `metaltile-runtime/src/autotune.rs::lookup`
   from a stub into a real component. Per `perf-research/STATUS.md`, this
   is Idea 046, called out as "highest ROI" in the research planning.
   v0.2 scope is **rules-based + persistent disk cache** (Idea 047), not
   ML-driven (M1 is a v0.3+ moonshot).

3. **`metaltile-model` crate (PR #58)** — TOML-based model definition,
   constexpr buffer slot assignment, execution plan compilation. v0.2
   scope: clean up PR #58, add 2–3 more example models beyond
   `llama_decode.toml`, document the schema as a non-stable preview.

4. **Codegen expansion** — Ideas 041–045 in `perf-research/STATUS.md`
   (software pipelining, LICM, CSE polish, if-conversion, value sink),
   plus the brittle-match cleanup from issue #53. Reduces both perf gaps
   and maintenance load on the codegen crate.

5. **Quantization perf** — qmm v3 (issue #55, BM-tile W-reuse) plus
   FP4 packed kernel work. Pairs with workstream #1 because quantized
   ops are the second-worst category vs. MLX.

### Bigger surface decisions for v0.2

- **Shape algebra**: `README.md` mentions "type-level shape algebra" as a
  v0.2 item. Scope it concretely in a follow-up design doc once v0.1 is
  out the door — it is not blocking the v0.2 tag, but is on the radar.
- **MSRV decision**: pick a stable Rust version and commit to it, or
  document an explicit "nightly until edition 2024 is stable" policy.
- **API-stability statement**: v0.2.0 should soften the
  "APIs are not yet stable" line in the README to something like
  "Pre-1.0 — breaking changes allowed at minor versions, called out in
  CHANGELOG." Same posture, more legible to evaluators.
- **Examples directory**: ship 2–3 standalone examples under
  `crates/metaltile/examples/` (rms_norm, sdpa, gemv) that compile from
  a fresh `cargo new`. Currently the only "examples" are the bench
  kernels under `metaltile-std`, which is not the right entry point for
  a new user.

### Exit criteria

1. M5 Max baseline shows SDPA + GQA + bf16 at ≥ 80% of MLX on the headline
   shapes (Qwen3, Llama-style). Softmax bf16 ≥ 80% of MLX. Average MT% on
   M5 Max stays ≥ 130%.
2. `Autotuner::lookup` returns a tuned config on cache hit and runs a real
   tuning loop on cache miss, with a documented cache location.
3. `metaltile-model` ships with at least three models compiled into MSL
   end-to-end, with a `tile model` CLI subcommand demonstrating the flow.
4. CHANGELOG entry exists, breaking changes are explicitly listed, and a
   short migration note covers any renamed/removed public symbols.
5. Reproducer transcript: same as v0.1, plus a `tile model build
   examples/models/llama_decode.toml` end-to-end demo.

### Non-goals

- No 1.0 stability commitment. The "API stability" pass is a v0.3+ exercise.
- No CPU/SIMD fallback (Idea M9 moonshot — out of scope).
- No ML-driven autotuner (M1 moonshot).
- No Metal 3.2 tensor descriptor codegen (M8 — blocked on hardware
  availability and is project-scale).

---

## Cadence and decision-making

- **v0.1.0 target**: aim for a 1–2 week cut from this document landing. The
  scope is mostly "land in-flight + tighten metadata," so the elapsed time
  is dominated by the release-machinery dry-run, not by new code.
- **v0.2.0 target**: 4–8 weeks after v0.1.0. The dominant cost is the
  SDPA bf16 fix and the autotuner — both are real engineering, not
  janitorial.
- **Patch releases**: v0.1.x and v0.2.x are reserved for fixes only.
  Anything that would be a breaking change goes to the next minor.
- **Decision authority**: scope changes to either version should be PR-able
  against this file with a one-paragraph rationale. The maintainer set is
  small enough that a Slack/issue thread suffices for sign-off.

---

## Open questions parked here

These intentionally do not block v0.1, but should be resolved before v0.2
tags:

1. **Repository canonical URL.** `Cargo.toml` says `github.com/metaltile/metaltile`,
   the GitHub Pages planning site is `wafflehaus.github.io`, and contributor
   accounts (`0xClandestine`) suggest the real org may differ. Pick one,
   update everywhere.
2. **License headers in source files.** LICENSE is Apache-2.0 at root; source
   files don't currently carry SPDX headers. Decide whether to add them or
   document the root-license-only posture.
3. **`metaltile-bench` crate**. Present in `crates/` but untracked, not in
   workspace members. Decide whether to commit it, gitignore it, or delete
   it before v0.1 to avoid confusing first-time clones.
4. **`metaltile-planning/` and `perf-research/` directories**. These are
   internal-planning artifacts that currently live in the repo (the planning
   dir is committed; the perf-research dir is partly tracked). For an
   open-source release, decide whether to keep them in-repo, move to a
   separate `metaltile-planning` repo, or scrub before publishing.
