# Perf Idea M1 — ML-driven autotuner

## Metadata
- **Number**: M1
- **Name**: ml-driven-autotuner
- **Source**: `perf-ideas.md` — Moonshots
- **Status**: ⚠️ feasible (project-scale)
- **Worktree**: — (analytical assessment, no worktree)
- **Assignee**: (self)

## Hypothesis (from perf-ideas.md)
> Train a tiny gradient-boosted model on `(kernel, shape, dtype) → best_schedule` using features from `tile profile` (regs, occupancy, bytes/flop). One-time fit, zero per-launch cost. The autotuner cache becomes a learned predictor instead of an exhaustive sweep.

## Target
- **Primary file(s)**: `crates/metaltile-runtime/src/autotune.rs`, `crates/metaltile-cli/src/cmd/bench.rs` (feature extraction)
- **Bench filter**: end-to-end autotune accuracy — % of predictions that match the true best config
- **Shapes / dtypes to watch**: all kernels across the full shape sweep

## Assessment

### Current autotuner state
The `Autotuner` infrastructure has:
- `TuneCache` with `entries: BTreeMap<String, TuneEntry>` persisted to `~/.cache/metaltile/<chip>/<kernel_hash>.json`.
- `TuneEntry` contains `bucket: Vec<ShapeBucket>` and `best_config: TuneConfig`.
- `TuneConfig` has `tile_dims`, `threads`, `unroll_factor`, `use_simd_matrix`, `use_async_copy`.
- `lookup()` is a placeholder returning `None` (see idea #046).

The current search strategy is described as "grid search with exponential backoff" but **no search code is visible** in `autotune.rs`. The file only has cache save/load and a placeholder lookup.

### What already exists for feature extraction
`bench.rs` `compute_profiles()` already extracts static features per kernel:
- `regs_per_thread` from `register_estimate::estimate_registers()`.
- `occ_pct` (occupancy percentage) and `bottleneck` (`ThreadLimited` / `RegisterLimited` / `MemoryLimited`) from `occupancy::best_threadgroup_size()`.

These are CPU-only, fast calculations that don't require GPU execution.

### What the ML autotuner would need
1. **Feature vector per kernel+shape+dtype**:
   - Static features: `regs_per_thread`, `max_live`, `occ_pct`, `bottleneck`, `kernel.mode` (Reduction/Elementwise/etc.), `num_params`, `num_constexprs`.
   - Shape features: `n_elements`, `n_rows`, `n_cols`, `bytes_per_element`.
   - Derived features: `bytes/flop` (memory intensity), `arithmetic intensity`.

2. **Label**: The `TuneConfig` that produces the best throughput (GB/s or GFLOP/s) from an actual bench run.

3. **Model**: A tiny gradient-boosted decision tree (e.g., `lightgbm` or `xgboost` via Rust bindings, or a simple decision tree in pure Rust). The model predicts the best `TuneConfig` (or a ranking of configs) given the feature vector.

4. **Training data**: Run the full bench suite across a grid of shapes for each kernel, recording the best config per shape. This is a one-time cost (minutes to hours).

5. **Inference**: At dispatch time, compute features → model predicts config → `lookup()` returns the predicted entry without any GPU search.

### Why this is a moonshot, not a quick win
- **Data collection**: Need to run the exhaustive sweep to generate training labels. The current autotuner has no search implementation, so this sweep doesn't exist.
- **Model integration**: Adding a gradient-boosted model as a dependency (`lightgbm-rs`, `linfa`) adds build complexity and binary size.
- **Cold-start problem**: New kernels/shapes not in the training set need a fallback (either the current grid search or a default config).
- **Hardware variation**: The model is trained on a specific chip (Apple7/8/9). Different chips have different register files, threadgroup memory sizes, and ALU widths. A model trained on M1 may mispredict on M4.

### Practical path
A simpler interim step: build a **lookup table** (exact match on hashed features → config) rather than a learned model. This is what `TuneCache` already is — it just needs the training data. The ML model is an optimization on top of the lookup table, not a replacement for it.

## Verdict

- **Outcome**: feasible — project-scale, but the infra pieces exist
- **Why**: The autotuner cache, feature extraction (`compute_profiles`), and bench harness are all in place. The missing pieces are: (1) exhaustive data collection, (2) model training, (3) wiring the prediction into `lookup()`. This is a genuine multi-day to project-scale effort, not a quick tweak.
- **Re-scope**: Phase 1 should be "implement exhaustive grid search + populate `TuneCache`" (essentially what idea #046 describes). Phase 2 is "replace search with learned model".

## Risk Register
- Training data scarcity: each new kernel requires a sweep. The model generalizes poorly to unseen kernels.
- Hardware-specific models: need separate models per chip family, or hardware features as inputs.
- Model drift: codegen changes (new passes, new intrinsics) invalidate training data.

## Notes for Next Person
- Before building a model, verify that the current autotuner actually has a search implementation. As of the code I read, `autotune.rs` has cache save/load but no search grid.
- Start with a small decision tree (e.g., `if occ_pct < 50 { use more threads }`) as a proof of concept. A full gradient-boosted model is overkill until the data pipeline is proven.
