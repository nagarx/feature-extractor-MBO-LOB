# Test Inventory: Feature Extractor MBO-LOB

> **Purpose**: Complete map of all tests in the repository. An LLM coder should know exactly which test file validates which module, whether it requires real data, and what invariants it checks.
> **Last Updated**: March 5, 2026

---

## Summary

| Category | Count | Runtime |
|----------|-------|---------|
| Unit tests (`cargo test --lib`) | ~730 | ~30s |
| Integration tests (39 files) | 389 | ~3-5 min (CI) |
| **Total** | **~1119** | |

**CI command**: `cargo test --features "parallel,databento"` (excludes `extended_validation`)
**Toolchain**: Rust 1.94.0

---

## Test Architecture: 4-Level Validation Pyramid

All core validation targets one canonical day (Feb 3, 2025 NVIDIA MBO data) processed deeply.

```
                    Level 4: Export Round-Trip
                       3 tests, synthetic, ~1s
                  Level 3: Full-Day Statistical Invariants
                    6 tests, 1 canonical day, ~60-90s
               Level 2: Golden Snapshot (500 vectors)
                 1 test, per-group checksums, ~15-30s
            Level 1: Per-Formula Transformation Tracing
              6 tests, first 50K events, ~10s
```

---

## CI Configuration (`.github/workflows/ci.yml`)

| Job | Command | What it checks |
|-----|---------|----------------|
| test | `cargo test --features "parallel,databento"` | >= 700 tests pass |
| test (no defaults) | `cargo test --no-default-features` | No-feature compilation |
| fmt | `cargo fmt --all -- --check` | Formatting |
| clippy | `cargo clippy --features "parallel,databento" -- -D warnings` | Lints |
| docs | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` | Doc warnings |
| bench | `cargo bench --no-run` | Benchmark compilation |

---

## Shared Test Helpers (`tests/common/`)

### `mod.rs` (105 lines)

| Export | Description |
|--------|-------------|
| `HOT_STORE_DIR` | `"../data/hot_store"` -- canonical path to decompressed test data |
| `COMPRESSED_DIR` | `"../data/compressed"` |
| `MBP10_DIR` | `"../data/mbp10"` |
| `find_hot_store_file()` | Find a `.dbn` file in the hot store |
| `find_compressed_file()` | Find a `.dbn.zst` file |
| `find_mbo_file()` | Find any MBO data file |
| `find_mbp10_file()` | Find MBP-10 data file |
| `has_test_data()` | Check if data directory exists and has files |
| `skip_if_no_data!()` | Macro: skip test gracefully if no real data available |

### `assertions.rs` (197 lines)

| Function | Description |
|----------|-------------|
| `assert_f64_eq(a, b, msg)` | Compare with `contract::FLOAT_CMP_EPS` |
| `assert_f64_approx(a, b, tol, msg)` | Compare with custom tolerance |
| `assert_features_finite(features)` | No NaN or Inf in feature vector |
| `assert_feature_layout(features, expected_count)` | Correct length |
| `assert_signal_basics(features)` | Signal index 92 (book_valid) is 0 or 1 |
| `compare_vectors(a, b, abs_tol)` | Element-wise comparison with tolerance |
| `pearson_correlation(x, y)` | Pearson r for statistical correlation |
| `assert_feature_has_variance(features, idx, name)` | Feature is not constant |

### `fixtures.rs` (56 lines)

| Function | Description |
|----------|-------------|
| `load_or_generate_fixture<T, F>(path, gen)` | Load JSON fixture or generate and save |
| `fixture_path(filename)` | Resolve `tests/fixtures/{filename}` |

### Golden Fixtures (`tests/fixtures/`)

| File | Purpose |
|------|---------|
| `golden_feb3_500.json` | 500 post-warmup feature vectors for regression testing |
| `golden_feb3_100.json` | 100 vectors for smaller tests |
| `pipeline_contract.toml` | Mirror of `contracts/pipeline_contract.toml` for when workspace TOML unavailable |

---

## Integration Test Files (39 files, 389 tests)

### Level 1: Transformation Tracing

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `transformation_tracing.rs` | 6 | Yes | LOB reconstruction, raw features, derived features, MBO features, signals, 98-feature composition | Per-formula correctness at 7 checkpoints (100, 500, 1K, 5K, 10K, 25K, 50K events) |

### Level 2: Golden Snapshot

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `golden_snapshot.rs` | 1 | Yes | Full pipeline (98 features) | 500 vectors x 98 features bit-exact match at `FLOAT_CMP_EPS`. Per-group checksums (LOB, Derived, MBO, Signal). First run generates fixture; subsequent runs validate. |

### Level 3: Full-Day Statistical Invariants

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `phase3_real_data_validation.rs` | 6 | Yes | Full pipeline, export, cross-day isolation | Determinism (two runs identical), NVIDIA statistical invariants (price range, spread, OFI two-sided, book_valid > 95%), cross-day state isolation, signal spot-check, MBO feature statistics, export end-to-end |

### Level 4: Export Round-Trip

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `export_roundtrip.rs` | 3 | No | AlignedBatchExporter, NPY export | Write-then-read round trip, shapes match, no NaN/Inf |

### Feature Extraction Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `feature_count_validation.rs` | 14 | No | FeatureConfig, feature_count() | All config combos produce correct counts (40/48/76/84/98/116) |
| `comprehensive_validation.rs` | 6 | Yes | Full pipeline, MBO features | Feature finiteness, variance, statistical properties on real data |
| `granular_mbo_mbp_validation.rs` | 4 | No | MBO feature extraction | Granular MBO vs MBP feature comparison |
| `mbp_mbo_validation.rs` | 4 | No | LOB reconstruction | MBO vs MBP-10 price accuracy |

### Signal Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `signal_extraction_test.rs` | 9 | No | Signal computation, indices | Synthetic signal extraction, index bounds |
| `signal_layer_integration.rs` | 4 | Yes | OfiComputer, signals pipeline | OFI warmup, signal ranges on real data |
| `signal_layer_comprehensive_validation.rs` | 9 | Yes | Full signal layer | Statistical properties of all 14 signals |
| `signal_level_validation_test.rs` | 5 | No | Signal validation | Signal boundary conditions |
| `sign_convention_validation.rs` | 2 | Yes | Sign conventions (RULE.md section 10) | Directional features have correct bullish/bearish signs, pearson correlation between signals and price direction |

### Pipeline Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `pipeline_tests.rs` | 25 | No | Pipeline, PipelineBuilder | Pipeline construction, config validation, streaming sequences |
| `pipeline_integration_tests.rs` | 21 | No | Pipeline end-to-end | Full pipeline with synthetic data |
| `pipeline_refactoring_tests.rs` | 3 | Yes | process() vs process_messages() equivalence | Both APIs produce identical output at `FLOAT_CMP_EPS` tolerance |
| `source_abstraction_tests.rs` | 10 | No | Pipeline source abstraction | Different input sources produce consistent results |
| `state_leak_test.rs` | 9 | No | Pipeline.reset() | No state leakage across resets, deterministic output |

### MBO Feature Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `mbo_features_real_data_validation.rs` | 2 | Yes | MBO features (36) | Feature ranges, statistical properties on NVIDIA data |
| `real_nvidia_validation.rs` | 8 | Yes | Full pipeline on NVIDIA | NVIDIA-specific feature statistics |
| `tick_size_mbo_test.rs` | 7 | No | MBO features, tick_size propagation | Tick size correctly affects depth_ticks features |
| `tick_size_propagation_test.rs` | 3 | No | Tick size config | Tick size flows from config through to feature extraction |

### Labeling Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `labeling_tests.rs` | 18 | No | TlobLabelGenerator, LabelConfig | Formula correctness, edge cases, class balance |
| `multi_horizon_label_tests.rs` | 25 | No | MultiHorizonLabelGenerator, ThresholdStrategy | All threshold variants, multi-horizon consistency |
| `opportunity_labeling_tests.rs` | 13 | No | OpportunityLabelGenerator | Peak return detection, conflict resolution, edge cases |
| `triple_barrier_integration_tests.rs` | 15 | No | TripleBarrierLabeler, TimeoutStrategy | Barrier hits, timeout strategies, risk/reward |

### Export Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `aligned_exporter_tests.rs` | 26 | No | AlignedBatchExporter | Alignment, normalization, NPY output shapes |
| `export_tests.rs` | 5 | No | Export legacy, TensorFormatter | Basic export functionality |
| `dataset_config_integration.rs` | 16 | Yes | DatasetConfig TOML parsing | Config validation, pipeline config generation |
| `fair_validation_with_warnings.rs` | 2 | No | Export validation | Warning-level vs error-level validation |

### Contract Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `contract_validation_test.rs` | 19 | No | contract.rs vs pipeline_contract.toml | All 116 feature indices unique and correct, schema version, label encodings, normalization rules, TOML structural fingerprint |

### Normalization Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `normalization_integration_tests.rs` | 35 | No | All 7 normalizers | Per-normalizer formula correctness, edge cases (NaN, zero variance), reset semantics |

### Sequence Builder Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `multiscale_reset_leak_test.rs` | 4 | No | MultiScaleWindow reset | No cross-day state leakage in multi-scale sequences |
| `multiscale_streaming_test.rs` | 3 | No | MultiScaleWindow streaming | Streaming mode produces correct sequences |

### Sampling Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `sampling_interval_units_test.rs` | 4 | No | SamplingConfig units | Nanosecond vs millisecond interval consistency |

### Parallel Processing Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `parallel_processing_tests.rs` | 25 | Yes | BatchProcessor, CancellationToken | Parallel == sequential output, graceful cancellation, error modes |

### Validation & Phase Tests

| File | Tests | Real Data | Module(s) Tested | Key Invariants |
|------|-------|-----------|------------------|----------------|
| `phase1_tests.rs` | 5 | No | Phase 1 pipeline features | Basic pipeline construction and output |
| `real_data_validation.rs` | 3 | No | LOB reconstruction validation | Price accuracy, size accuracy |
| `integration_tests.rs` | 20 | Yes | Mixed integration | Various cross-module integration checks |
| `outlier_investigation.rs` | 3 | No | Outlier detection | Extreme value handling |

---

## Unit Tests (~730 tests, `cargo test --lib`)

Key modules with inline `#[cfg(test)] mod tests`:

| Source File | Approximate Tests | What They Cover |
|-------------|-------------------|-----------------|
| `features/mbo_features/mod.rs` | ~50 | All 36 MBO features: synthetic formula test, sign conventions, edge cases |
| `features/signals/ofi.rs` | ~15 | OFI accumulation, warmup, sample_and_reset |
| `features/signals/compute.rs` | ~10 | Signal vector computation |
| `features/signals/time_regime.rs` | ~8 | UTC-to-ET conversion, DST handling |
| `labeling/multi_horizon.rs` | ~45 | ThresholdStrategy variants, multi-horizon generation |
| `labeling/triple_barrier.rs` | ~30 | Barrier hit detection, timeout strategies |
| `labeling/opportunity.rs` | ~25 | Peak return detection, conflict resolution |
| `labeling/magnitude.rs` | ~20 | Return type computation |
| `preprocessing/normalization.rs` | ~29 | All 7 normalizers: formula, edge cases, reset |
| `preprocessing/adaptive_sampling.rs` | ~15 | Volatility-adaptive threshold computation |
| `sequence_builder/builder.rs` | ~23 | Circular buffer, push/build, stride |
| `sequence_builder/multiscale.rs` | ~26 | Multi-scale windowing, decimation |
| `export/config/labels.rs` | ~39 | ExportLabelConfig TOML parsing, validation |
| `export_aligned/mod.rs` | ~20 | Export alignment, normalization |
| `features/extractor.rs` | ~15 | Feature composition, config validation |
| `pipeline.rs` | ~10 | Pipeline construction, process flow |
| `batch.rs` | ~9 | Batch processing, cancellation |

---

## Test Data Requirements

| Data Source | Path | Used By |
|-------------|------|---------|
| NVIDIA MBO (Feb 3, 2025) | `../data/hot_store/dbn/GLBX.MDP3-2023-02-03.dbn` | Golden snapshot, transformation tracing, phase3, most real-data tests |
| NVIDIA MBO (compressed) | `../data/compressed/*.dbn.zst` | Parallel processing tests |
| MBP-10 | `../data/mbp10/*.dbn` | MBP validation tests |

**CI behavior**: Tests requiring real data use `skip_if_no_data!()` and skip gracefully when data is absent.

---

*Last updated: March 5, 2026*
