# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Zero-Allocation Feature Extraction API (Phase 1)**
  - `FeatureExtractor::extract_into()` - Write features to pre-allocated buffer
  - `FeatureExtractor::extract_arc()` - Extract and wrap in `Arc<Vec<f64>>`
  - Enables buffer reuse across samples, eliminating per-sample allocations
  - Full documentation with usage examples

- **Arc-Native Sequence Building API**
  - `SequenceBuilder::push_arc()` - Accept `Arc<Vec<f64>>` directly
  - `MultiScaleWindow::push_arc()` - Zero-copy sharing across all scales
  - Original `push()` methods remain for backward compatibility
  - Multi-scale memory savings: ~66% reduction (Arc clone vs Vec clone)

- **Zero-Copy Sequence Building**
  - `FeatureVec` type alias (`Arc<Vec<f64>>`) for shared feature storage
  - `Sequence.features` now `Vec<Arc<Vec<f64>>>` instead of `Vec<Vec<f64>>`
  - Eliminates 67KB deep-copy per sequence (100 snapshots × 84 features × 8 bytes)
  - **3.2 million sequences/sec** throughput in release mode

- **Comprehensive Test Coverage**
  - 19 new tests for `extract_into` and `extract_arc`
  - 6 new tests for `SequenceBuilder::push_arc`
  - 6 new tests for `MultiScaleWindow::push_arc`
  - 7 new pipeline integration tests for zero-allocation path
  - Numerical precision tests with edge cases (Pi, E, 1e-15, 1e15)
  - Multi-builder sharing isolation tests

### Changed

- **Pipeline Hot Path Optimization (Phase 2)**
  - `Pipeline.process()` now uses `extract_into()` with reusable feature buffer
  - Features wrapped in `Arc` once, then shared via `push_arc()` to sequence builders
  - Multi-scale path uses `push_arc()` for zero-copy sharing across all scales
  - Memory allocation per sample: 1 Vec + Arc wrap (vs 3 Vec allocations before)

- `Pipeline.process()` uses `process_message_into()` from mbo-lob-reconstructor
- Reuses single `LobState` buffer across all messages (zero allocation in hot loop)
- Export functions updated to handle `Arc<Vec<f64>>` → `Vec<f64>` conversion
- `push()` methods now delegate to `push_arc()` internally

### Performance

- **Pipeline throughput**: Uses O(1) PriceLevel caching from reconstructor
- **Feature extraction**: Buffer reuse eliminates per-sample Vec allocation
- **Sequence building**: 67.2 KB saved per sequence via Arc sharing
- **Multi-scale sharing**: 16 bytes (2 Arc clones) vs 1,344 bytes (2 Vec clones)
- **Validated**: 500K+ messages, 7140 sequences with 0 numerical mismatches

## [0.1.1] - 2025-12-04

### Added

- **PipelineBuilder API**
  - `PipelineBuilder` - Fluent API for configuring and building pipelines
  - `PipelineBuilder::from_preset(Preset)` - Create from paper-aligned presets
  - Auto-computed feature count (no manual calculation required)
  - Research presets: DeepLOB, TLOB, FI-2010, TransLOB, LiT, Minimal, Full

- **Prelude Module**
  - `feature_extractor::prelude::*` - Single import for all common types
  - Re-exports both feature-extractor and mbo-lob-reconstructor types

- **FeatureConfig Improvements**
  - `FeatureConfig::feature_count()` - Auto-compute total feature count
  - `SequenceConfig::from_feature_config()` - Auto-sync feature count

### Fixed

- **Critical: Streaming Sequence Generation**
  - Fixed sequence loss bug where ~98.8% of sequences were lost due to buffer eviction
  - Pipeline now accumulates sequences during streaming instead of batch generation at end
  - Before: 96 sequences from 79,834 features (1.2% efficiency)
  - After: 7,979 sequences from 79,834 features (100% efficiency)

- Fixed rustdoc broken intra-doc links
- Fixed clippy warnings (unnecessary parentheses, unused variables)
- Fixed test assertions for feature count validation

### Changed

- `Pipeline::process()` now uses streaming mode for sequence generation
- MSRV updated to 1.83 (required by transitive dependency `pest`)

## [0.1.0] - 2025-12-01

### Added

- **Core Features**
  - Raw LOB feature extraction (40 features for 10 levels)
  - Derived feature computation (mid-price, spread, volume imbalance, etc.)
  - MBO feature aggregation (36 features)
  - FI-2010 handcrafted features (80 features)
  
- **Order Flow Analysis**
  - Order Flow Imbalance (OFI) computation
  - Multi-Level OFI (MLOFI) across 10 levels
  - Queue imbalance and depth imbalance
  - Trade imbalance tracking
  
- **Market Impact**
  - Pre-trade market impact estimation
  - VWAP calculation
  - Slippage estimation in basis points
  
- **Normalization**
  - Z-score normalization (online and windowed)
  - Bilinear normalization (TLOB paper)
  - Min-Max normalization
  - Percentage change normalization
  - Rolling Z-score (multi-day statistics)
  - Global Z-score (snapshot-level)
  - Per-feature normalization
  
- **Sampling Strategies**
  - Volume-based sampling
  - Event-based sampling
  - Adaptive volume threshold (volatility-aware)
  
- **Sequence Building**
  - Configurable sequence builder
  - Horizon-aware configuration
  - Multi-scale windowing (fast/medium/slow)
  
- **Data Validation**
  - LOB state validation (crossed quotes, price ordering)
  - Feature validation (NaN/Inf detection, range checks)
  - Timestamp monotonicity validation
  
- **Export**
  - NumPy export for PyTorch/TensorFlow
  - Aligned export (sequences + labels)
  - Batch export with train/val/test splits
  
- **Schema System**
  - Feature definitions with paper references
  - Paper-aligned presets (DeepLOB, TLOB, FI-2010, TransLOB, LiT, Full)
  - Feature categories (Raw, Derived, OrderFlow, Handcrafted)

### Documentation

- Comprehensive README with usage examples
- Architecture documentation (ARCHITECTURE.md)
- Usage guide (docs/USAGE_GUIDE.md)
- Inline documentation for all public APIs
- Doc tests for key functions

### Testing

- 300+ unit tests
- 100+ integration tests (including real-data validation)
- 50+ doc tests
- Benchmark suite using Criterion
- Validated against 151M+ MBO messages (21 days NVIDIA data)
- 99.42% price accuracy against MBP-10 ground truth

[Unreleased]: https://github.com/nagarx/feature-extractor-MBO-LOB/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nagarx/feature-extractor-MBO-LOB/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nagarx/feature-extractor-MBO-LOB/releases/tag/v0.1.0
