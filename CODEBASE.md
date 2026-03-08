# Feature-Extractor-MBO-LOB: Codebase Technical Reference

> **Purpose**: This document provides complete technical details for LLMs and developers to understand, modify, and extend the codebase without prior context.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Module Architecture](#2-module-architecture)
3. [Core Data Flow](#3-core-data-flow)
4. [Feature Extraction System](#4-feature-extraction-system)
5. [Sequence Building](#5-sequence-building)
6. [Sampling Strategies](#6-sampling-strategies)
7. [Normalization System](#7-normalization-system)
8. [Label Generation](#8-label-generation)
9. [Export Pipeline](#9-export-pipeline)
10. [Dataset Configuration System](#10-dataset-configuration-system)
11. [Parallel Batch Processing](#11-parallel-batch-processing) *(feature-gated)*
12. [Configuration System](#12-configuration-system)
13. [Validation](#13-validation)
14. [Zero-Allocation APIs](#14-zero-allocation-apis)
15. [Testing Patterns](#15-testing-patterns)
16. [Performance Considerations](#16-performance-considerations)
17. [Integration with MBO-LOB-Reconstructor](#17-integration-with-mbo-lob-reconstructor)
18. [Common Patterns and Idioms](#18-common-patterns-and-idioms)
19. [Known Limitations](#19-known-limitations)

---

## 1. Project Overview

### Purpose

High-performance feature extraction library for Limit Order Book (LOB) and Market-By-Order (MBO) data, designed for training transformer models (TLOB, DeepLOB, PatchTST).

### Core Dependencies

```toml
[dependencies]
mbo-lob-reconstructor = { git = "https://github.com/..." }  # LOB reconstruction
ndarray = "0.15"           # NumPy-like arrays
ndarray-npy = "0.8"        # NumPy file export
ahash = "0.8"              # Fast hashing for order tracking
serde = "1.0"              # Serialization
toml = "0.8"               # Config files
chrono = "0.4"             # Timestamps
```

### Research Paper Compliance

The library implements features from:
- **TLOB**: Transformer model with dual attention
- **DeepLOB**: CNN for LOB data
- **FI-2010**: Benchmark dataset methodology
- **LOBFrame/LOBench**: Normalization approaches

---

## 2. Module Architecture

```
build.rs                     # Compile-time git hash capture for provenance
src/
├── lib.rs                    # Public API exports
├── prelude.rs                # Convenience re-exports
├── builder.rs                # PipelineBuilder fluent API
├── pipeline.rs               # Main Pipeline orchestrator
├── config.rs                 # PipelineConfig, SamplingConfig, MultiScaleSamplingConfig
├── contract.rs               # Pipeline contract constants (SCHEMA_VERSION, feature counts, eps)
├── validation.rs             # Data quality validation
├── batch.rs                  # Parallel batch processing (feature-gated: parallel)
│
├── features/
│   ├── mod.rs                # Thin re-export layer: FeatureConfig, FeatureExtractor, SignalContext
│   ├── config.rs             # FeatureConfig struct, builder, validation, feature_count()
│   ├── extractor.rs          # FeatureExtractor: orchestrates LOB/MBO/signal extraction
│   ├── lob_features.rs       # Raw LOB features (40 features)
│   ├── derived_features.rs   # Derived metrics (8 features)
│   ├── mbo_features/          # MBO aggregated features (36 features) — directory module
│   │   ├── mod.rs             # MboAggregator orchestrator, public API, integration tests
│   │   ├── event.rs           # MboEvent struct + from_mbo_message + to_mbo_message
│   │   ├── window.rs          # MboWindow rolling buffer with O(1) incremental statistics
│   │   ├── order_tracker.rs   # OrderInfo + OrderTracker (lifecycle state, eviction)
│   │   ├── flow_features.rs   # 12 order flow features (indices 48-59)
│   │   ├── size_features.rs   # 8 size distribution features (indices 60-67)
│   │   ├── queue_features.rs  # 6 queue & depth features (indices 68-73)
│   │   ├── institutional_features.rs  # 4 institutional detection features (indices 74-77)
│   │   └── lifecycle_features.rs      # 6 core MBO metrics (indices 78-83)
│   ├── signals/              # Trading signals (14 features) — directory module
│   │   ├── mod.rs            # SignalContext, re-exports
│   │   ├── time_regime.rs    # TimeRegime enum, compute_time_regime(), ET offset estimation
│   │   ├── book_valid.rs     # is_book_valid(), is_book_valid_from_lob()
│   │   ├── ofi.rs            # OfiComputer, OfiSample, streaming OFI accumulation
│   │   ├── compute.rs        # SignalVector, compute_signals(), compute_signals_with_book_valid()
│   │   └── indices.rs        # Signal index constants (TRUE_OFI through SCHEMA_VERSION)
│   ├── order_flow.rs         # Order flow imbalance
│   ├── fi2010.rs             # FI-2010 benchmark features
│   ├── market_impact.rs      # Market impact estimation
│   └── experimental/         # Opt-in experimental features (Schema 3.0+)
│       ├── mod.rs            # ExperimentalConfig, group registry, feature_count()
│       ├── institutional_v2.rs  # Enhanced whale detection (8 features, indices 98-105)
│       ├── volatility.rs     # Realized vol & regime (6 features, indices 106-111)
│       └── seasonality.rs    # Time-of-day features (4 features, indices 112-115)
│
├── sequence_builder/
│   ├── mod.rs                # Module exports, FeatureVec type alias
│   ├── builder.rs            # SequenceBuilder, Sequence (Arc-based), push_arc()
│   ├── horizon_aware.rs      # Label-aware sequence building
│   └── multiscale.rs         # MultiScaleWindow, push_arc()
│
├── preprocessing/
│   ├── mod.rs                # Module exports
│   ├── sampling.rs           # VolumeBasedSampler, EventBasedSampler
│   ├── normalization.rs      # All normalizer implementations
│   ├── adaptive_sampling.rs  # AdaptiveVolumeThreshold
│   └── volatility.rs         # VolatilityEstimator
│
├── labeling/
│   ├── mod.rs                # LabelConfig, TrendLabel, LabelStats, re-exports
│   ├── tlob.rs               # TlobLabelGenerator - ✅ Export integrated
│   ├── deeplob.rs            # DeepLobLabelGenerator - ✅ Export integrated
│   ├── multi_horizon.rs      # MultiHorizonLabelGenerator - ✅ Export integrated
│   ├── opportunity.rs        # OpportunityLabelGenerator - ✅ Export integrated
│   ├── triple_barrier.rs     # TripleBarrierLabeler - ✅ Export integrated (Schema 2.4+)
│   └── magnitude.rs          # MagnitudeGenerator (regression) - ⚠️ API only, export pending
│
├── schema/
│   ├── mod.rs                # Module exports
│   ├── presets.rs            # Preset feature configurations
│   └── feature_def.rs        # Feature definitions
│
├── export/                   # Export configuration
│   ├── mod.rs                # Module declarations, config re-exports
│   ├── config/               # NormalizationConfig, DatasetConfig, FeatureNormStrategy, etc.
│   ├── tensor_format.rs      # TensorFormat, TensorFormatter, TensorOutput
│   └── dataset_config.rs     # DatasetConfig, LabelingStrategy, ExportLabelConfig, etc.
│
├── export_aligned/           # Production exporter (modular directory)
│   ├── mod.rs                # AlignedBatchExporter struct, builder, dispatch, export_day_common()
│   ├── types.rs              # LabelEncoding, NormalizationStrategy, NormalizationParams, AlignedDayExport, LabelingResult
│   ├── metadata.rs           # build_provenance(), build_normalization_metadata(), build_processing_metadata()
│   ├── alignment.rs          # generate_labels(), build_multi_horizon_label_matrix(), align_sequences_with_multi_labels()
│   ├── validation.rs         # verify_raw_spreads(), validate_label_alignment(), validate_class_balance()
│   ├── normalization.rs      # normalize_sequences(), compute_feature_statistics(), apply_normalization()
│   ├── npy_export.rs         # export_sequences(), export_labels(), export_multi_horizon_labels(), tensor format exports
│   └── strategies/           # Per-strategy label generation → LabelingResult
│       ├── mod.rs            # Sub-module declarations
│       ├── tlob.rs           # labeling_single_horizon_tlob(), labeling_multi_horizon_tlob()
│       ├── opportunity.rs    # labeling_opportunity(), build_opportunity_label_matrix()
│       └── triple_barrier.rs # labeling_triple_barrier(), compute_daily_volatility(), build_triple_barrier_label_matrix()
│
├── tools/                    # CLI tools (binaries)
│   ├── export_dataset.rs     # Configuration-driven export CLI
│   └── calibrate_triple_barrier.py  # Per-day volatility analysis & barrier calibration
│
└── configs/                  # Sample configuration files
    ├── nvda_98feat.toml      # 98-feature NVIDIA export configuration
    ├── nvda_98feat_v2.toml   # Updated 98-feature config
    ├── nvda_98feat_full.toml # Full 98-feature config (all features)
    ├── nvda_84feat_baseline.toml # 84-feature baseline configuration
    ├── nvda_116feat_full_analysis.toml # 116-feature config (includes experimental)
    ├── nvda_triple_barrier.toml       # Triple Barrier labeling config
    ├── nvda_11month_triple_barrier.toml           # 11-month Triple Barrier export
    ├── nvda_11month_triple_barrier_calibrated.toml # Calibrated per-horizon barriers
    ├── nvda_11month_triple_barrier_volscaled.toml  # Volatility-scaled barriers
    ├── nvda_11month_complete.toml     # Complete 11-month dataset export
    ├── nvda_bigmove_detection.toml    # Opportunity/big-move detection config
    ├── nvda_multi_horizon.toml        # Multi-horizon TLOB config
    ├── nvda_extended_multi_horizon.toml # Extended multi-horizon config
    ├── nvda_balanced.toml             # Balanced class distribution config
    ├── nvda_spread_adaptive.toml      # Spread-adaptive threshold config
    ├── nvda_tlob_raw_v2.toml          # Raw TLOB (no normalization)
    ├── nvda_tlob_repo_v1.toml         # TLOB repo-compatible v1
    ├── nvda_tlob_repo_v2.toml         # TLOB repo-compatible v2
    ├── template_multi_symbol.toml     # Multi-symbol export template
    └── examples/                      # Example configs
        ├── nvda_tlob_repo.toml
        └── nvda_tlob_raw.toml
```

---

## 3. Core Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STREAMING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

    MboMessage           LobState              Features           Sequences
        │                   │                     │                   │
        ▼                   ▼                     ▼                   ▼
┌─────────────┐     ┌─────────────────┐   ┌─────────────────┐   ┌──────────┐
│ LobRecon-   │────▶│ Feature         │──▶│ Sequence        │──▶│ Pipeline │
│ structor    │     │ Extractor       │   │ Builder         │   │ Output   │
│ (external)  │     │                 │   │                 │   │          │
│             │     │ ┌─────────────┐ │   │ ┌─────────────┐ │   │sequences │
│ process_    │     │ │ LOB (40)    │ │   │ │ Circular    │ │   │mid_prices│
│ message()   │     │ │ Derived (8) │ │   │ │ Buffer      │ │   │stats     │
│             │     │ │ MBO (36)    │ │   │ │ max=1000    │ │   │          │
└─────────────┘     │ └─────────────┘ │   │ └─────────────┘ │   └──────────┘
                    └─────────────────┘   └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │    Sampling       │
                    │  (should_sample)  │
                    │                   │
                    │ Volume-based      │
                    │ Event-based       │
                    │ Adaptive          │
                    └───────────────────┘

CRITICAL: Sequences are accumulated DURING streaming via try_build_sequence()
          after each feature push, NOT at the end with generate_all_sequences()
```

### Pipeline Processing Methods

The Pipeline provides three processing methods (all produce identical results):

```rust
// Method 1: Process a file path (most common)
pub fn process<P: AsRef<Path>>(&mut self, input_path: P) -> Result<PipelineOutput> {
    let loader = DbnLoader::new(input_path)?;
    self.process_messages(loader.iter_messages()?)  // Delegates to process_messages
}

// Method 2: Process a MarketDataSource (supports hot store)
pub fn process_source<S: MarketDataSource>(&mut self, source: S) -> Result<PipelineOutput> {
    let messages = source.messages()?;
    self.process_messages(messages)  // Delegates to process_messages
    }
    
// Method 3: Process an iterator of messages (core implementation)
pub fn process_messages<I>(&mut self, messages: I) -> Result<PipelineOutput>
where
    I: Iterator<Item = MboMessage>,
{
    // Single source of truth for processing logic:
    // 1. For each message: Update LOB state (zero-allocation)
    // 2. Check sampling condition (volume/event based)
    // 3. Extract features into reusable buffer
    // 4. Wrap in Arc for zero-copy sharing
    // 5. Push to sequence builder(s)
    // 6. Try build sequence immediately (streaming mode)
    // 7. Track mid-price for labeling
}
```

**Note**: `process()` delegates to `process_messages()` (DRY principle). This ensures
a single code path for all processing methods, making bug fixes apply universally.

### PipelineOutput

Output container from pipeline processing with methods for post-processing:

```rust
pub struct PipelineOutput {
    pub sequences: Vec<Sequence>,       // Generated sequences [N_sequences, window_size, features]
    pub mid_prices: Vec<f64>,           // Mid-prices for labeling
    pub messages_processed: usize,      // Total MBO messages
    pub features_extracted: usize,      // Sampled LOB snapshots
    pub sequences_generated: usize,     // Complete sequences
    pub stride: usize,                  // Stride used for sequences
    pub window_size: usize,             // Window size used
    pub multiscale_sequences: Option<MultiScaleSequence>,  // Multi-scale (if enabled)
    pub adaptive_stats: Option<AdaptiveSamplingStats>,     // Adaptive stats (if enabled)
}

impl PipelineOutput {
    /// Format sequences for model-specific tensor shapes
    pub fn format_sequences(&self, formatter: &TensorFormatter) -> Result<TensorOutput>;
    
    /// Convenience method for formatting with format and mapping
    pub fn format_as(&self, format: TensorFormat, mapping: FeatureMapping) -> Result<TensorOutput>;
    
    /// Generate labels for multiple prediction horizons
    pub fn generate_multi_horizon_labels(&self, config: MultiHorizonConfig) -> Result<MultiHorizonLabels>;
    
    /// Get feature count per timestep (if sequences available)
    pub fn feature_count(&self) -> Option<usize>;
    
    /// Get features as flat 2D array for export
    pub fn features_flat(&self) -> Vec<Vec<f64>>;
}
```

---

## 4. Feature Extraction System

### Feature Count Calculation

The total feature count depends on configuration:

| Configuration | Formula | Count |
|---------------|---------|-------|
| Raw LOB only | `levels × 4` | 40 (10 levels) |
| LOB + Derived | `levels × 4 + 8` | 48 |
| LOB + MBO | `levels × 4 + 36` | 76 |
| LOB + Derived + MBO | `levels × 4 + 8 + 36` | 84 |
| LOB + Derived + MBO + Signals | `levels × 4 + 8 + 36 + 14` | 98 |
| + Experimental (all groups) | `98 + 8 + 6 + 4` | 116 |

### FeatureConfig

```rust
pub struct FeatureConfig {
    pub lob_levels: usize,        // Default: 10
    pub tick_size: f64,           // Default: 0.01
    pub include_derived: bool,    // Default: false
    pub include_mbo: bool,        // Default: false
    pub mbo_window_size: usize,   // Default: 1000
    pub include_signals: bool,    // Default: false (14 trading signals)
}

impl FeatureConfig {
    pub const DERIVED_FEATURE_COUNT: usize = 8;
    pub const MBO_FEATURE_COUNT: usize = 36;
    pub const SIGNAL_FEATURE_COUNT: usize = 14;
    
    // AUTHORITATIVE feature count calculation
    pub fn feature_count(&self) -> usize {
        let base = self.lob_levels * 4;
        let derived = if self.include_derived { 8 } else { 0 };
        let mbo = if self.include_mbo { 36 } else { 0 };
        let signals = if self.include_signals { 14 } else { 0 };
        base + derived + mbo + signals
    }
    
    // Builder methods
    pub fn with_derived(self, enabled: bool) -> Self;
    pub fn with_mbo(self, enabled: bool) -> Self;
    pub fn with_signals(self, enabled: bool) -> Self;
}
```

### LOB Features (40 for 10 levels)

Extracted in `lob_features.rs`:

```
Index Range | Feature Type | Description
------------|--------------|------------
0-9         | Ask Prices   | Best ask to level 10 (in dollars)
10-19       | Ask Sizes    | Volume at each ask level (shares)
20-29       | Bid Prices   | Best bid to level 10 (in dollars)
30-39       | Bid Sizes    | Volume at each bid level (shares)
```

**Price Conversion**: Prices stored as fixed-point `i64` (divide by 1e9 for dollars)

```rust
// From lob_features.rs
pub fn extract_raw_features(lob_state: &LobState, levels: usize, output: &mut Vec<f64>) {
    // Ask prices (converted from fixed-point)
    for i in 0..levels {
        let price = if lob_state.ask_prices[i] > 0 {
            lob_state.ask_prices[i] as f64 / 1e9  // Fixed-point to dollars
        } else {
            0.0
        };
        output.push(price);
    }
    // ... ask sizes, bid prices, bid sizes
}
```

### Derived Features (8)

Computed in `derived_features.rs`:

| Index | Feature | Formula |
|-------|---------|---------|
| 0 | Mid-price | `(best_bid + best_ask) / 2` |
| 1 | Spread | `best_ask - best_bid` |
| 2 | Spread (bps) | `spread / mid_price × 10000` |
| 3 | Total Bid Volume | `Σ bid_sizes` |
| 4 | Total Ask Volume | `Σ ask_sizes` |
| 5 | Volume Imbalance | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` |
| 6 | Weighted Mid-price | Volume-weighted mid-price |
| 7 | Price Impact | Estimated impact of market order |

### MBO Features (36)

Extracted in `features/mbo_features/` directory module using multi-timescale windows.

**Module Architecture** (`src/features/mbo_features/`):

| File | Responsibility | Lines |
|------|---------------|-------|
| `mod.rs` | `MboAggregator` orchestrator, public API (`MboEvent`, `OrderInfo` re-exports), integration tests | ~990 |
| `event.rs` | `MboEvent` struct, `from_mbo_message()`, `to_mbo_message()` | ~106 |
| `window.rs` | `MboWindow` rolling buffer with O(1) incremental counters, lazy percentile/stats recomputation | ~272 |
| `order_tracker.rs` | `OrderInfo` struct, `OrderTracker` (active orders map, completed order buffers, eviction) | ~329 |
| `flow_features.rs` | 12 order flow features: net flow, cancel flow, trade flow, event rates, volatility, regime indicator | ~274 |
| `size_features.rs` | 8 size distribution features: percentiles, z-score, skewness, large order ratio, concentration | ~177 |
| `queue_features.rs` | 6 queue & depth features: queue position, size ahead, orders per level, concentration, depth ticks | ~251 |
| `institutional_features.rs` | 4 institutional detection features: large order frequency/imbalance, modification score, iceberg proxy | ~127 |
| `lifecycle_features.rs` | 6 core MBO metrics: avg order age, median lifetime, avg fill ratio, time to first fill, cancel-to-add ratio | ~209 |

**Window Sizes**:
- Fast: 100 messages (~2 seconds)
- Medium: 1000 messages (~20 seconds) - **Primary extraction source**
- Slow: 5000 messages (~100 seconds)

**Feature Categories** (feature-to-file mapping):

| Range | Category | Count | Source File | Description |
|-------|----------|-------|------------|-------------|
| 0-11 | Order Flow | 12 | `flow_features.rs` | Event rates, net flows, imbalances, regime indicator |
| 12-19 | Size Distribution | 8 | `size_features.rs` | Percentiles, z-scores, skewness, concentration |
| 20-25 | Queue & Depth | 6 | `queue_features.rs` | Queue position, depth ticks, level concentration |
| 26-29 | Institutional | 4 | `institutional_features.rs` | Large order detection, modification score |
| 30-35 | Core MBO | 6 | `lifecycle_features.rs` | Order lifecycle metrics |

Each feature sub-module exports an `extract()` function returning a fixed-size array `[f64; N]`
to avoid heap allocations in the high-frequency path. The `MboAggregator::extract_features()`
method in `mod.rs` calls each `extract()` and concatenates the results into a `Vec<f64>` of length 36.

**Data Flow**:

```
MboMessage → MboEvent (event.rs)
                │
                ├──► MboWindow.push() (window.rs) — 3 windows: fast/medium/slow
                └──► OrderTracker.process_event() (order_tracker.rs) — lifecycle state
                          │
                          ▼
                MboAggregator.extract_features(lob) → Vec<f64> [36]
                          │
                ┌─────────┼─────────┬─────────┬─────────┐
                ▼         ▼         ▼         ▼         ▼
          flow_features  size_    queue_   instit.   lifecycle_
          ::extract()    features features features  features
          [f64; 12]     ::extract ::extract ::extract ::extract
                        [f64; 8] [f64; 6] [f64; 4]  [f64; 6]
```

> **Fixed in v2.2**: `median_order_lifetime` (index 31 within MBO, absolute index 79)
> now returns the true median lifetime of completed orders. Implementation uses rolling
> buffers in `OrderTracker` (`completed_lifetimes`, `completed_fill_ratios`, `completed_modifications`)
> to track order outcomes.

**MboAggregator Memory**: ~8 MB per symbol (bounded via order eviction)

**Order Eviction** (in `order_tracker.rs`): Orders are evicted from tracking after 1 hour
(`MAX_ORDER_AGE_NS`) or when tracker exceeds 50K orders (`MAX_ORDER_TRACKER_SIZE`) to prevent
unbounded memory growth. Evicted orders are recorded in completed buffers to preserve statistics.

**Tick Size Configuration**:

The `depth_ticks_bid` and `depth_ticks_ask` features (indices 24-25 within MBO, computed in
`queue_features.rs`) require correct tick size configuration for accurate depth measurements:

```rust
// Default: $0.01 (US equities)
let aggregator = MboAggregator::new();

// Crypto ($0.001 tick)
let aggregator = MboAggregator::new().with_tick_size(0.001);

// Forex ($0.0001 pip)
let aggregator = MboAggregator::new().with_tick_size(0.0001);
```

When using `FeatureExtractor`, tick_size is automatically propagated from `FeatureConfig`:
```rust
let config = FeatureConfig::new(10)
    .with_tick_size(0.001)  // Crypto tick size
    .with_mbo(true);
let extractor = FeatureExtractor::with_config(config);
```

**WARNING**: Using incorrect tick_size causes integer division truncation in depth calculations.
A $0.001 tick instrument with default $0.01 configuration will report depth_ticks = 0.

**Test Distribution** (72 tests total):

| File | Test Count | Scope |
|------|-----------|-------|
| `mod.rs` | 33 | Integration tests: sign conventions, queue tracking, eviction, full extraction |
| `flow_features.rs` | 7 | Unit: net flows, symmetry, volatility, regime indicator |
| `size_features.rs` | 7 | Unit: skewness, concentration, insufficient data |
| `queue_features.rs` | 7 | Unit: level concentration, depth ticks (bid/ask) |
| `lifecycle_features.rs` | 7 | Unit: cancel-to-add ratio, median lifetime, fill ratio |
| `order_tracker.rs` | 4 | Unit: fill ratio, add/cancel, full fill, eviction |
| `institutional_features.rs` | 3 | Unit: modification score edge cases |
| `window.rs` | 2 | Unit: push, eviction |
| `event.rs` | 1 | Unit: event creation |

### Trading Signals (14 for indices 84-97)

Computed in `signals/` directory module using streaming OFI and base features.

**CONSTRAINT**: Signals require **exactly 10 LOB levels** (`lob_levels == 10`).
The signal indices are hardcoded for the 10-level layout:
- `derived_indices::MID_PRICE = 40` (assumes 10 × 4 = 40 raw features)
- `mbo_indices::CANCEL_RATE_BID = 50` (assumes MBO starts at 48)

Configurations with `include_signals: true` and `lob_levels ≠ 10` will fail validation.

**Implementation**:

| File | Module |
|------|--------|
| `src/features/signals/` | Directory module: `OfiComputer`, `compute_signals()`, `TimeRegime` |
| `src/features/signals/ofi.rs` | `OfiComputer`, `OfiSample`, streaming OFI accumulation |
| `src/features/signals/compute.rs` | `SignalVector`, `compute_signals()`, `compute_signals_with_book_valid()` |
| `src/features/signals/time_regime.rs` | `TimeRegime` enum, `compute_time_regime()`, ET offset estimation |
| `src/features/signals/book_valid.rs` | `is_book_valid()`, `is_book_valid_from_lob()` |
| `src/features/signals/indices.rs` | Signal index constants (TRUE_OFI=0 through SCHEMA_VERSION=13) |

**Research Foundation**:
- OFI: Cont, Kukanov & Stoikov (2014) "The Price Impact of Order Book Events"
- Microprice: Stoikov (2018) "The Micro-Price"
- Time regimes: Cont et al. §3.3 (intraday price impact patterns)

**OfiComputer (Streaming OFI)**:

```rust
pub struct OfiComputer {
    // State tracking for OFI calculation
    prev_best_bid: Option<i64>,
    prev_best_ask: Option<i64>,
    prev_best_bid_size: u32,
    prev_best_ask_size: u32,
    
    // OFI accumulators (since last sample)
    ofi_bid: i64,           // Σ bid_size_change
    ofi_ask: i64,           // Σ ask_size_change
    depth_sum: u64,         // For average depth
    depth_count: u64,
    
    // Warmup tracking (MIN_WARMUP_STATE_CHANGES = 100)
    state_changes_since_reset: u64,  // Must reach MIN_WARMUP_STATE_CHANGES (100)
    last_sample_timestamp: i64,
}

/// Minimum effective state changes before OFI is considered "warm".
/// This counts ACTUAL LOB state transitions, not raw messages.
pub const MIN_WARMUP_STATE_CHANGES: u64 = 100;

impl OfiComputer {
    /// Update on EVERY LOB state transition (critical for accuracy)
    pub fn update(&mut self, lob_state: &LobState);
    
    /// Sample OFI and reset accumulators (at sampling points)
    pub fn sample_and_reset(&mut self, timestamp: i64) -> OfiSample;
    
    /// Check if warmup complete (≥MIN_WARMUP_STATE_CHANGES state changes)
    pub fn is_warm(&self) -> bool;
    
    /// Reset on Action::Clear or day boundary
    pub fn reset_on_clear(&mut self);
}
```

**Signal Index Mapping**:

| Index | Signal | Description | Range |
|-------|--------|-------------|-------|
| 84 | `true_ofi` | Raw OFI per Cont et al. | unbounded |
| 85 | `depth_norm_ofi` | OFI / avg_depth | unbounded |
| 86 | `executed_pressure` | trade_rate_ask - trade_rate_bid | unbounded |
| 87 | `signed_mp_delta_bps` | Microprice delta in basis points | ~[-100, 100] |
| 88 | `trade_asymmetry` | Normalized trade imbalance | [-1, 1] |
| 89 | `cancel_asymmetry` | Normalized cancel imbalance | [-1, 1] |
| 90 | `fragility_score` | `level_conc / ln(avg_depth)` | [0, ∞) |
| 91 | `depth_asymmetry` | Depth position asymmetry ¹ | [-1, 1] |
| 92 | `book_valid` | Valid book flag | {0, 1} |
| 93 | `time_regime` | Market session | {0, 1, 2, 3, 4} |
| 94 | `mbo_ready` | Warmup status | {0, 1} |
| 95 | `dt_seconds` | Time since last sample | [0, ∞) |
| 96 | `invalidity_delta` | Quote anomalies since last sample | [0, ∞) |
| 97 | `schema_version` | Always 2.2 | {2.2} |

**Signal Index Constants Module**:

For programmatic access to signal indices, use the `signals::indices` module:

```rust
use feature_extractor::features::signals::indices;

// Access specific signal indices
let ofi_idx = indices::TRUE_OFI;           // 84
let regime_idx = indices::TIME_REGIME;      // 93
let valid_idx = indices::BOOK_VALID;        // 92

// Use in feature selection or analysis
let ofi_value = features[indices::TRUE_OFI];
let is_valid = features[indices::BOOK_VALID] > 0.5;
```

**Available constants**:
- `TRUE_OFI`, `DEPTH_NORM_OFI`, `EXECUTED_PRESSURE` — Direction signals
- `SIGNED_MP_DELTA_BPS`, `TRADE_ASYMMETRY`, `CANCEL_ASYMMETRY` — Confirmation signals
- `FRAGILITY_SCORE`, `DEPTH_ASYMMETRY` — Impact signals
- `BOOK_VALID`, `MBO_READY` — Safety gates
- `TIME_REGIME`, `DT_SECONDS`, `INVALIDITY_DELTA`, `SCHEMA_VERSION` — Meta signals

**TimeRegime Enum**:

```rust
pub enum TimeRegime {
    Open   = 0,  // 9:30-9:45 ET - Highest volatility
    Early  = 1,  // 9:45-10:30 ET - Settling period
    Midday = 2,  // 10:30-15:30 ET - Most stable
    Close  = 3,  // 15:30-16:00 ET - Position squaring
    Closed = 4,  // Outside market hours
}
```

¹ `depth_asymmetry` uses `depth_ticks_*` (volume-weighted avg distance from BBO), NOT raw volume.

**Signal Categories**:

| Category | Signals | Purpose |
|----------|---------|---------|
| Safety Gates | `book_valid`, `mbo_ready` | Must pass before trading |
| Direction | `true_ofi`, `depth_norm_ofi`, `executed_pressure` | Predict price movement |
| Confirmation | `trade_asymmetry`, `cancel_asymmetry` | Validate direction |
| Impact | `fragility_score`, `depth_asymmetry` | Market stability |
| Timing | `signed_mp_delta_bps`, `time_regime` | When to trade |
| Meta | `dt_seconds`, `invalidity_delta`, `schema_version` | Data quality |

**Usage**:

```rust
// Enable signals in pipeline
let pipeline = PipelineBuilder::new()
    .lob_levels(10)
    .with_derived_features()  // Required for signals
    .with_mbo_features()      // Required for signals
    .with_trading_signals()   // Adds 14 signals (indices 84-97)
    .build()?;

assert_eq!(pipeline.config().features.feature_count(), 98);
```

### Experimental Features (18 features, indices 98-115)

Opt-in features for analysis and experimentation. Enabled via `include_experimental` in `FeatureConfig`
or `include_experimental = true` in TOML. Subject to change without schema version bumps.

**Groups:**

| Group | Count | Indices | Features |
|-------|-------|---------|----------|
| `institutional_v2` | 8 | 98-105 | round_lot_ratio, odd_lot_ratio, size_clustering, price_clustering, mod_before_cancel, sweep_ratio, fill_patience_bid, fill_patience_ask |
| `volatility` | 6 | 106-111 | realized_vol_fast, realized_vol_slow, vol_ratio, vol_momentum, return_autocorr, vol_of_vol |
| `seasonality` | 4 | 112-115 | minutes_since_open, minutes_until_close, session_progress, time_bucket |

**Configuration:**

```rust
let config = ExperimentalConfig::new()
    .with_all_groups();  // Enable all 18 features

let config = ExperimentalConfig::new()
    .with_groups(vec!["volatility".into(), "seasonality".into()]);  // Selective
```

```toml
[features]
include_experimental = true
experimental_groups = ["institutional_v2", "volatility", "seasonality"]
```

**Promotion Path**: Features that prove valuable in analysis can be promoted to the main schema
(requires schema version bump, documentation update, and Python contract update).

---

## 5. Sequence Building

### SequenceConfig

```rust
pub struct SequenceConfig {
    pub window_size: usize,      // Snapshots per sequence (default: 100)
    pub stride: usize,           // Skip between sequences (default: 1)
    pub max_buffer_size: usize,  // Max buffer capacity (default: 1000)
    pub feature_count: usize,    // MUST match FeatureConfig.feature_count()
}
```

### Circular Buffer Architecture

```rust
pub struct SequenceBuilder {
    buffer: VecDeque<Snapshot>,   // Circular buffer of snapshots
    config: SequenceConfig,
    total_pushed: u64,
    total_sequences: u64,
    last_sequence_pos: usize,     // For stride tracking
}
```

### Sequence Generation

**Streaming Mode** (Recommended):
```rust
// Called after each feature push
if let Some(seq) = builder.try_build_sequence() {
    sequences.push(seq);
}
```

**Batch Mode** (Limited use - may lose data):
```rust
// Only use for final batch, buffer size limits data
let sequences = builder.generate_all_sequences();
```

### Sequence Output

```rust
/// Type alias for zero-copy feature sharing
pub type FeatureVec = Arc<Vec<f64>>;

pub struct Sequence {
    /// Feature vectors wrapped in Arc for zero-copy sharing
    /// [window_size × feature_count]
    pub features: Vec<FeatureVec>,  // Arc<Vec<f64>> - NOT Vec<Vec<f64>>
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ns: u64,
    pub length: usize,
}
```

**IMPORTANT**: Features are stored as `Arc<Vec<f64>>` to enable zero-copy sharing between:
- Multiple sequences with overlapping windows
- Multi-scale builders (fast/medium/slow)
- Parallel processing threads

---

## 6. Sampling Strategies

### VolumeBasedSampler (Recommended)

```rust
pub struct VolumeBasedSampler {
    target_volume: u64,         // Target shares per sample (default: 1000)
    accumulated_volume: u64,    // Current accumulation
    min_time_interval_ns: u64,  // Min time between samples (default: 1ms = 1_000_000 ns)
    last_sample_time: u64,
}

impl VolumeBasedSampler {
    /// Create a new volume-based sampler.
    ///
    /// # Arguments
    /// * `target_volume` - Target volume per sample (shares)
    /// * `min_time_interval_ns` - Minimum **nanoseconds** between samples (e.g., 1_000_000 = 1ms)
    ///
    /// # Units
    /// The `min_time_interval_ns` is in nanoseconds to match:
    /// - `SamplingConfig.min_time_interval_ns` (pipeline config)
    /// - MBO message timestamps (nanoseconds since epoch)
    pub fn new(target_volume: u64, min_time_interval_ns: u64) -> Self;
    
    // Returns true when volume threshold AND time threshold met
    pub fn should_sample(&mut self, event_volume: u32, timestamp_ns: u64) -> bool {
        self.accumulated_volume += event_volume as u64;
        
        let volume_ok = self.accumulated_volume >= self.target_volume;
        let time_ok = timestamp_ns - self.last_sample_time >= self.min_time_interval_ns;
        
        if volume_ok && time_ok {
            self.accumulated_volume = 0;
            self.last_sample_time = timestamp_ns;
            true
        } else {
            false
        }
    }
    
    /// Dynamically update the volume threshold at runtime.
    ///
    /// This enables adaptive sampling strategies that adjust to market conditions:
    /// - **High volatility**: Increase threshold to avoid over-sampling
    /// - **Low volatility**: Decrease threshold to maintain data density
    ///
    /// # Example
    /// ```rust
    /// let mut sampler = VolumeBasedSampler::new(1000, 1_000_000);
    /// 
    /// // Adaptive adjustment based on volatility
    /// if realized_volatility > high_threshold {
    ///     sampler.set_threshold(1500);  // Wider threshold in volatile markets
    /// } else if realized_volatility < low_threshold {
    ///     sampler.set_threshold(500);   // Tighter threshold in quiet markets
    /// }
    /// ```
    pub fn set_threshold(&mut self, new_threshold: u64) {
        self.target_volume = new_threshold;
    }
}
```

### EventBasedSampler (Legacy)

```rust
pub struct EventBasedSampler {
    sample_interval: u64,  // Sample every N events
    event_count: u64,
}
```

### AdaptiveVolumeThreshold (Phase 1)

Adjusts volume threshold based on realized volatility:

```rust
pub struct AdaptiveVolumeThreshold {
    volatility_estimator: VolatilityEstimator,
    base_threshold: u64,
    min_multiplier: f64,  // 0.5 = 50% in quiet markets
    max_multiplier: f64,  // 2.0 = 200% in volatile markets
}
```

### TimeBasedSampler (⚠️ NOT YET IMPLEMENTED)

The `SamplingStrategy::TimeBased` enum variant is declared but **not implemented**.
Attempting to use it will result in an explicit error:

```rust
// This will return an error at runtime:
let config = SamplingConfig {
    strategy: SamplingStrategy::TimeBased,
    min_time_interval_ns: Some(100_000_000),  // 100ms
    ..Default::default()
};
// Error: "TimeBased sampling strategy is not yet implemented.
//         Please use VolumeBased or EventBased sampling instead."
```

**Workaround**: Use `SamplingStrategy::VolumeBased` (recommended) or `SamplingStrategy::EventBased`.

**Tracking**: See `TODO.md` for implementation status.

### Critical Design Constraints

The preprocessing module enforces several constraints to prevent numerical bugs:

| Component | Constraint | Reason |
|-----------|------------|--------|
| `VolatilityEstimator` | `window_size >= 2` | Variance calculation requires n≥2. Welford's reverse formula uses `(n-1)` denominator. |
| `RollingZScoreNormalizer` | Eager computation | Rolling stats are computed immediately on `add_day_stats()`, not lazily on `normalize()`. Prevents stale-cache bugs. |
| `AdaptiveVolumeThreshold` | `total_cmp()` for f64 sorting | Handles NaN/Inf safely. Filters non-finite values with diagnostic logging per RULE.md. |
| `AdaptiveVolumeThreshold` | `volatility_window >= 2` | Propagated from `VolatilityEstimator` constraint. |

**Example: VolatilityEstimator constraint**
```rust
// Panics - window_size=1 causes division by zero in variance removal
let estimator = VolatilityEstimator::new(1);  // PANICS

// Valid - minimum window size is 2
let estimator = VolatilityEstimator::new(2);  // OK
```

---

## 7. Normalization System

### Overview

The normalization system provides **per-feature-group configuration** for flexible preprocessing
that matches requirements from different research papers (TLOB, DeepLOB, LOBench, FI-2010).

### Feature Groups (98-feature mode)

| Group | Indices | Count | Description |
|-------|---------|-------|-------------|
| LOB Prices | 0-9, 20-29 | 20 | Ask/Bid prices at 10 levels |
| LOB Sizes | 10-19, 30-39 | 20 | Ask/Bid sizes at 10 levels |
| Derived | 40-47 | 8 | Mid-price, spread, imbalance, etc. |
| MBO | 48-83 | 36 | Order flow microstructure features |
| Signals | 84-97 | 14 | Trading signals (**includes categoricals - NEVER normalize**) |

### NormalizationConfig

Configuration-driven normalization for each feature group:

```rust
use feature_extractor::export::dataset_config::{NormalizationConfig, FeatureNormStrategy};

// For TLOB paper (raw export - model handles normalization via BiN)
let config = NormalizationConfig::raw();  // or NormalizationConfig::tlob_paper()

// For official TLOB repository preprocessing
let config = NormalizationConfig::tlob_repo();  // Global Z-score before BiN

// For DeepLOB
let config = NormalizationConfig::deeplob();  // Per-feature Z-score

// Custom configuration
let config = NormalizationConfig::default()
    .with_lob_prices(FeatureNormStrategy::GlobalZScore)
    .with_lob_sizes(FeatureNormStrategy::ZScore)
    .with_derived(FeatureNormStrategy::None)
    .with_signals(FeatureNormStrategy::None);  // CRITICAL: Never normalize categoricals
```

### FeatureNormStrategy Options

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `None` | Raw value | TLOB paper (BiN handles normalization) |
| `ZScore` | `(x - μ_i) / σ_i` per feature | DeepLOB, per-feature independence |
| `GlobalZScore` | `(x - μ) / σ` shared across group | TLOB repo, LOBench (preserves relationships) |
| `MarketStructure` | Ask+Bid share stats per level | Preserves ask > bid ordering |
| `PercentageChange` | `(x - ref) / ref` | HLOB, cross-instrument generalization |
| `MinMax` | `(x - min) / (max - min)` | Bounded output required |
| `Bilinear` | `(price - mid) / (k × tick)` | LOB structure |

### Research Paper Presets

| Preset | LOB Prices | LOB Sizes | Derived | MBO | Signals |
|--------|------------|-----------|---------|-----|---------|
| `raw()` / `tlob_paper()` | None | None | None | None | None |
| `tlob_repo()` | GlobalZScore | GlobalZScore | None | None | None |
| `deeplob()` | ZScore | ZScore | None | None | None |
| `lobench()` | GlobalZScore | GlobalZScore | GlobalZScore | GlobalZScore | None |
| `fi2010()` | ZScore | ZScore | ZScore | None | None |

### TOML Configuration

```toml
[normalization]
# Option 1: Use defaults (all None = raw export for TLOB paper)
# Just omit the section or leave fields empty

# Option 2: Explicit configuration
lob_prices = "global_z_score"  # or: none, z_score, market_structure, percentage_change, min_max, bilinear
lob_sizes = "z_score"
derived = "none"
mbo = "none"
signals = "none"  # CRITICAL: Always "none" for signals (categoricals)

# Additional options
reference_price = "mid_price"  # For percentage_change: mid_price, first_ask, first_bid
bilinear_scale_factor = 50.0   # For bilinear normalization
```

### Categorical Signal Protection

Signals at indices 92, 93, 94, 97 are **categorical** and MUST NOT be normalized:

| Index | Signal | Type | Values |
|-------|--------|------|--------|
| 92 | book_valid | Binary | 0, 1 |
| 93 | time_regime | Categorical | 0, 1, 2, 3, 4 |
| 94 | mbo_ready | Binary | 0, 1 |
| 97 | schema_version | Constant | 2.2 |

The system always skips normalization for signals regardless of configuration.

### Legacy Normalizer Trait

For streaming normalization (per-value updates):

```rust
pub trait Normalizer: Send + Sync {
    fn update(&mut self, value: f64);
    fn normalize(&self, value: f64) -> f64;
    fn normalize_batch(&self, values: &[f64]) -> Vec<f64>;
    fn reset(&mut self);
    fn is_ready(&self) -> bool;
}
```

### Available Streaming Normalizers

| Normalizer | Formula | Use Case |
|------------|---------|----------|
| `PercentageChangeNormalizer` | `(x - ref) / ref` | Prices |
| `ZScoreNormalizer` | `(x - mean) / std` | Volumes, spreads |
| `RollingZScoreNormalizer` | Multi-day rolling Z-score | Cross-day normalization |
| `BilinearNormalizer` | `(x - mid) / (k × tick)` | LOB structure |
| `MinMaxNormalizer` | `(x - min) / (max - min)` | Bounded features |
| `GlobalZScoreNormalizer` | All features together | LOBench method |
| `PerFeatureNormalizer` | Separate stats per feature | Multi-feature |

---

## 8. Label Generation

### LabelConfig

```rust
pub struct LabelConfig {
    pub horizon: usize,           // Steps ahead to predict (default: 10)
    pub smoothing_window: usize,  // Prices to average (default: 5)
    pub threshold: f64,           // Classification threshold (default: 0.002)
}

impl LabelConfig {
    pub fn hft() -> Self { Self { horizon: 10, smoothing_window: 5, threshold: 0.0002 } }
    pub fn short_term() -> Self { Self { horizon: 50, smoothing_window: 10, threshold: 0.002 } }
    pub fn medium_term() -> Self { Self { horizon: 100, smoothing_window: 20, threshold: 0.005 } }
    
    // Minimum prices needed for TLOB method
    pub fn min_prices_required(&self) -> usize {
        self.smoothing_window + self.horizon + self.smoothing_window + 1
    }
}
```

### TrendLabel

```rust
pub enum TrendLabel {
    Down = -1,   // Price decreased by > threshold
    Stable = 0,  // Price change within threshold
    Up = 1,      // Price increased by > threshold
}

impl TrendLabel {
    pub fn as_int(&self) -> i8;           // -1, 0, 1
    pub fn as_class_index(&self) -> usize; // 0, 1, 2 (for softmax)
}
```

### Labeling Strategies Overview

The library provides multiple labeling strategies for different trading objectives:

| Strategy | Classes | Export CLI | Use Case |
|----------|---------|------------|----------|
| **TLOB** | Down/Stable/Up | ✅ Integrated | Trend following, DeepLOB reproduction |
| **Multi-Horizon** | Down/Stable/Up × N | ✅ Integrated | FI-2010 benchmarks, multi-task learning |
| **Opportunity** | BigDown/NoOpp/BigUp | ✅ Integrated | Big move detection, peak returns |
| **Triple Barrier** | StopLoss/Timeout/ProfitTarget | ✅ Integrated | Risk-managed trading, volatility-scaled barriers |
| **Magnitude** | Continuous returns | ⚠️ API only | Regression, position sizing |

> **Note**: "API only" means the labeler is fully implemented and tested in Rust, but NOT yet
> integrated into `export_dataset` CLI. Use the Rust API directly or wait for export integration.
> See [docs/LABELING_STRATEGIES.md](docs/LABELING_STRATEGIES.md) for detailed documentation.

### LabelingStrategy Enum (TOML `strategy` field)

The `LabelingStrategy` enum controls which labeling method is used during export:

```rust
pub enum LabelingStrategy {
    Tlob,           // Default: smoothed average trend labels
    Opportunity,    // Peak return based big-move detection
    TripleBarrier,  // Trade outcome based on profit/stop barriers
}
```

**TOML usage**: `strategy = "tlob"`, `strategy = "opportunity"`, `strategy = "triple_barrier"`

### LabelEncoding (Unified Validation Contract)

The `LabelEncoding` enum defines the expected label range and class names for each strategy,
serving as the single source of truth for label validation across the entire export pipeline:

```rust
pub enum LabelEncoding {
    SignedTrend,              // TLOB/Multi-horizon: {-1=Down, 0=Stable, 1=Up}
    SignedOpportunity,        // Opportunity: {-1=BigDown, 0=NoOpportunity, 1=BigUp}
    TripleBarrierClassIndex,  // Triple Barrier: {0=StopLoss, 1=Timeout, 2=ProfitTarget}
}

impl LabelEncoding {
    pub fn valid_range(&self) -> (i8, i8);           // (-1,1) or (0,2)
    pub fn num_classes(&self) -> usize;              // Always 3
    pub fn class_names(&self) -> Vec<&'static str>;  // Strategy-specific names
    pub fn strategy_name(&self) -> &'static str;     // "TLOB", "Opportunity", "Triple Barrier"
    pub fn expected_range_description(&self) -> String; // Human-readable for error messages
}
```

All labeling strategies produce a `LabelingResult` struct (defined in `export_aligned/types.rs`),
which is consumed by the unified `export_day_common()` pipeline. Each strategy function
(`labeling_single_horizon_tlob`, `labeling_multi_horizon_tlob`, `labeling_opportunity`,
`labeling_triple_barrier`) returns a `LabelingResult` containing label indices, label matrix,
encoding, strategy metadata, and distribution. The common pipeline handles alignment,
validation via `validate_label_alignment()` parameterized by `LabelEncoding`, normalization,
NPY export, and metadata generation.

### TLOB Labeling Method

```
w+(t,h,k) = (1/(k+1)) × Σ(i=0 to k) p(t+h-i)   // Future smoothed
w-(t,h,k) = (1/(k+1)) × Σ(i=0 to k) p(t-i)     // Past smoothed
l(t,h,k) = (w+ - w-) / w-                       // Percentage change

Label = Up     if l > θ
        Down   if l < -θ
        Stable otherwise
```

### Multi-Horizon Label Generation

For benchmark reproduction, labels are needed at multiple prediction horizons simultaneously.

#### MultiHorizonConfig

```rust
pub struct MultiHorizonConfig {
    pub horizons: Vec<usize>,              // Prediction horizons (e.g., [10, 20, 50, 100])
    pub smoothing_window: usize,           // Smoothing window size (default: 5)
    pub threshold_strategy: ThresholdStrategy,  // Classification threshold strategy
}

impl MultiHorizonConfig {
    pub fn fi2010() -> Self;     // Horizons: [10, 20, 30, 50, 100]
    pub fn deeplob() -> Self;    // Horizons: [10, 20, 50, 100]
    pub fn tlob() -> Self;       // Horizons: [1, 3, 5, 10, 30, 50]
}
```

#### ThresholdStrategy

```rust
pub enum ThresholdStrategy {
    /// Fixed percentage threshold (default: 0.002 = 0.2%)
    Fixed(f64),
    
    /// Rolling spread-based threshold (adaptive to market conditions)
    RollingSpread {
        window_size: usize,   // Rolling window size
        multiplier: f64,      // Multiplier for average spread
        fallback: f64,        // Fallback threshold when insufficient data
    },
    
    /// Quantile-based threshold for balanced class distribution
    Quantile {
        target_proportion: f64,  // Target proportion for Up/Down (e.g., 0.33)
        window_size: usize,      // Rolling window for quantile computation
        fallback: f64,           // Fallback when insufficient data
    },
}
```

**Per-Horizon Quantile Threshold Computation**:

When using `ThresholdStrategy::Quantile`, the threshold is computed **per-horizon** from the actual
smoothed price change distribution `l(t,h,k) = (w⁺ - w⁻) / w⁻` for that specific horizon.

```rust
// compute_quantile_threshold_for_horizon(horizon) logic:
// 1. Compute all smoothed percentage changes for THIS horizon
let smoothed_changes: Vec<f64> = (start..end)
    .map(|t| {
        let past_smooth = smoothed_past(t, k);        // w⁻(t,h,k)
        let future_smooth = smoothed_future(t, h, k); // w⁺(t,h,k)
        (future_smooth - past_smooth) / past_smooth   // l(t,h,k)
    })
    .collect();

// 2. Compute threshold from quantile of |l(t,h,k)|
// target_proportion=0.33 → ~33% Up, ~33% Down, ~34% Stable
```

**Why Per-Horizon Matters**:
- 1-step price changes are tiny (e.g., 0.001%)
- Multi-step smoothed changes are larger (scale with horizon)
- Using 1-step threshold for h=100 causes severe class imbalance (~95% Stable)
- Per-horizon thresholds match the actual distribution being classified

#### MultiHorizonLabelGenerator

```rust
pub struct MultiHorizonLabelGenerator {
    config: MultiHorizonConfig,
    prices: Vec<f64>,          // Single buffer shared across all horizons
}

impl MultiHorizonLabelGenerator {
    pub fn new(config: MultiHorizonConfig) -> Self;
    pub fn add_prices(&mut self, prices: &[f64]);    // Batch add
    pub fn add_price(&mut self, price: f64);         // Single add
    pub fn generate_labels(&self) -> Result<MultiHorizonLabels>;
    pub fn compute_stats(&self) -> BTreeMap<usize, LabelStats>;
}
```

#### MultiHorizonLabels (Output)

```rust
pub struct MultiHorizonLabels {
    horizons: Vec<usize>,
    labels: BTreeMap<usize, Vec<(usize, TrendLabel, f64)>>,  // horizon → [(index, label, change)]
}

impl MultiHorizonLabels {
    pub fn horizons(&self) -> &[usize];                        // List all horizons
    pub fn labels_for_horizon(&self, h: usize) -> Option<&Vec<...>>;  // Get labels for horizon
    pub fn summary(&self) -> MultiHorizonSummary;              // Aggregate statistics
}
```

#### Research Reference

| Paper | Horizons | Unit |
|-------|----------|------|
| FI-2010 | 10, 20, 30, 50, 100 | Events |
| DeepLOB | 10, 20, 50, 100 | Ticks |
| TLOB | 1, 3, 5, 10, 30, 50 | Seconds |

---

## 8b. Pipeline Contract Constants (`contract.rs`)

Single Rust-side source of truth for pipeline-wide constants. All values **must** match
`contracts/pipeline_contract.toml` and are validated at `cargo test` time by
`tests/contract_validation_test.rs` (11 golden tests).

| Constant | Value | Source in TOML |
|----------|-------|----------------|
| `SCHEMA_VERSION` | `2.2` (f64) | `[contract].schema_version` |
| `SCHEMA_VERSION_STR` | `"2.2"` | `[contract].schema_version` |
| `STABLE_FEATURE_COUNT` | `98` | `[features].stable_count` |
| `EXPERIMENTAL_FEATURE_COUNT` | `18` | `[features].experimental_count` |
| `FULL_FEATURE_COUNT` | `116` | `[features].full_count` |
| `LOB_LEVELS` | `10` | `[features].lob_levels` |
| `SIGNAL_COUNT` | `14` | count of `[features.signals]` entries |
| `CATEGORICAL_INDICES` | `[92, 93, 94, 97]` | `[features.categorical].indices` |
| `FLOAT_CMP_EPS` | `1e-10` | Floating-point comparison threshold |
| `DIVISION_GUARD_EPS` | `1e-8` | Division-by-zero guard for ratio features |

**Two schema version systems exist (do NOT merge them):**

1. `contract::SCHEMA_VERSION` = `2.2` (f64) -- the **pipeline contract** version, embedded at feature index 97, used in export metadata
2. `schema::SCHEMA_VERSION` = `"1.0.0"` (str) -- the **feature definition** schema version, used only by paper presets in `schema/feature_def.rs`

---

## 9. Export Pipeline

### AlignedBatchExporter

The production exporter for labeled datasets with perfect 1:1 sequence-label alignment.
Located in `src/export_aligned/` as a modular directory with single-responsibility sub-modules.

#### Module Architecture

```
export_aligned/
  mod.rs            ← AlignedBatchExporter struct, builder, export_day() dispatch, export_day_common()
  types.rs          ← LabelEncoding, NormalizationStrategy, NormalizationParams, AlignedDayExport, LabelingResult
  metadata.rs       ← build_provenance(), build_normalization_metadata(), build_processing_metadata()
  alignment.rs      ← generate_labels(), build_multi_horizon_label_matrix(), align_sequences_with_multi_labels()
  validation.rs     ← verify_raw_spreads(), validate_label_alignment(), validate_class_balance()
  normalization.rs  ← normalize_sequences(), compute_feature_statistics(), apply_normalization()
  npy_export.rs     ← export_sequences(), export_labels(), export_multi_horizon_labels(), tensor format exports
  strategies/       ← Per-strategy label generation (each returns LabelingResult)
    tlob.rs         ← labeling_single_horizon_tlob(), labeling_multi_horizon_tlob()
    opportunity.rs  ← labeling_opportunity(), build_opportunity_label_matrix()
    triple_barrier.rs ← labeling_triple_barrier(), compute_daily_volatility(), build_triple_barrier_label_matrix()
```

#### Deduplication Pattern: LabelingResult

All four labeling strategies (single-horizon TLOB, multi-horizon TLOB, opportunity, triple barrier)
produce a `LabelingResult` struct, which the unified `export_day_common()` pipeline consumes:

```rust
pub(super) struct LabelingResult {
    pub label_indices: Vec<usize>,            // Indices into mid_prices where labels exist
    pub label_matrix: Vec<Vec<i8>>,           // Labels (single = Vec<[1]>, multi = Vec<[N_horizons]>)
    pub encoding: LabelEncoding,              // Strategy-specific encoding for validation
    pub strategy_name: String,                // "TLOB", "Opportunity", "Triple Barrier"
    pub strategy_metadata: serde_json::Value, // Strategy-specific metadata for JSON export
    pub distribution: serde_json::Value,      // Label class distribution
    pub is_multi_horizon: bool,               // Controls 1D vs 2D label export
    pub horizons_config: Option<serde_json::Value>, // Horizon config for {day}_horizons.json
}
```

This pattern eliminated ~1000 lines of duplicated pipeline logic.

#### Public API

```rust
pub struct AlignedBatchExporter {
    output_dir: PathBuf,
    label_config: LabelConfig,
    window_size: usize,
    stride: usize,
    tensor_format: Option<TensorFormat>,
    feature_mapping: Option<FeatureMapping>,
    multi_horizon_config: Option<MultiHorizonConfig>,
    opportunity_configs: Option<Vec<OpportunityConfig>>,
    triple_barrier_configs: Option<(Vec<TripleBarrierConfig>, Vec<usize>)>,
    volatility_scaling: Option<(f64, f64, f64)>,  // (reference_vol, floor, cap)
    normalization_config: NormalizationConfig,
    config_hash: Option<String>,  // For provenance tracking
}

impl AlignedBatchExporter {
    pub fn new(output_dir: P, label_config: LabelConfig, window_size: usize, stride: usize) -> Self;

    // Builder methods
    pub fn with_tensor_format(self, format: TensorFormat) -> Self;
    pub fn with_feature_mapping(self, mapping: FeatureMapping) -> Self;
    pub fn with_multi_horizon_labels(self, config: MultiHorizonConfig) -> Self;
    pub fn with_opportunity_labels(self, configs: Vec<OpportunityConfig>) -> Self;
    pub fn with_triple_barrier_labels(self, configs: Vec<TripleBarrierConfig>, horizons: Vec<usize>) -> Self;
    pub fn with_volatility_scaling(self, reference_vol: f64, floor: f64, cap: f64) -> Self;
    pub fn with_normalization(self, config: NormalizationConfig) -> Self;
    pub fn with_config_hash(self, hash: String) -> Self;

    // Inspection
    pub fn normalization_config(&self) -> &NormalizationConfig;
    pub fn is_opportunity_labeling(&self) -> bool;
    pub fn is_triple_barrier_labeling(&self) -> bool;
    pub fn is_multi_horizon(&self) -> bool;
    pub fn is_tensor_formatted(&self) -> bool;
    pub fn tensor_format(&self) -> Option<&TensorFormat>;
    pub fn multi_horizon_config(&self) -> Option<&MultiHorizonConfig>;

    // Export: creates {day}_sequences.npy, {day}_labels.npy, {day}_metadata.json, {day}_normalization.json
    // For multi-horizon: also creates {day}_horizons.json
    pub fn export_day(&self, day_name: &str, output: &PipelineOutput) -> Result<AlignedDayExport>;
}
```

### ExportLabelConfig (Multi-Strategy, Multi-Horizon)

The `ExportLabelConfig` supports all three labeling strategies with multi-horizon generation:

```rust
pub struct ExportLabelConfig {
    // Strategy selection (Schema 2.3+)
    pub strategy: LabelingStrategy,                     // tlob (default), opportunity, triple_barrier
    pub conflict_priority: Option<ExportConflictPriority>, // For opportunity only

    // Horizon configuration
    pub horizon: usize,                    // Single horizon (default: 50)
    pub horizons: Vec<usize>,             // Multiple horizons (alias: max_horizons for Triple Barrier)
    pub smoothing_window: usize,           // Smoothing window size (default: 10)

    // Threshold configuration
    pub threshold: f64,                    // Classification threshold (default: 0.0008)
    pub threshold_strategy: Option<ExportThresholdStrategy>, // fixed, quantile, rolling_spread, tlob_dynamic

    // Triple Barrier fields (Schema 2.4+, only when strategy = "triple_barrier")
    pub profit_target_pct: Option<f64>,    // Upper barrier (e.g., 0.005 = 50 bps)
    pub stop_loss_pct: Option<f64>,        // Lower barrier (e.g., 0.003 = 30 bps)
    pub timeout_strategy: Option<ExportTimeoutStrategy>,
    pub min_holding_period: Option<usize>,

    // Per-horizon barrier overrides (Schema 3.2+)
    pub profit_targets: Option<Vec<f64>>,  // Per-horizon profit targets
    pub stop_losses: Option<Vec<f64>>,     // Per-horizon stop losses

    // Volatility-adaptive scaling (Schema 3.3+)
    pub volatility_scaling: Option<bool>,
    pub volatility_reference: Option<f64>,
    pub volatility_floor: Option<f64>,     // Default: 0.3
    pub volatility_cap: Option<f64>,       // Default: 3.0
}

impl ExportLabelConfig {
    pub fn is_multi_horizon(&self) -> bool;
    pub fn effective_horizons(&self) -> Vec<usize>;
    pub fn max_horizon(&self) -> usize;
    pub fn is_volatility_scaling(&self) -> bool;
    pub fn volatility_scaling_params(&self) -> Option<(f64, f64, f64)>;

    // Constructors
    pub fn single(horizon: usize, smoothing: usize, threshold: f64) -> Self;
    pub fn multi(horizons: Vec<usize>, smoothing: usize, threshold: f64) -> Self;
    pub fn fi2010() -> Self;
    pub fn deeplob() -> Self;
    pub fn triple_barrier(horizons: Vec<usize>, profit_targets: Vec<f64>, stop_losses: Vec<f64>) -> Self;
}
```

**TOML Configuration Examples:**

```toml
# TLOB single-horizon (backward compatible)
[labels]
horizon = 200
smoothing_window = 10
threshold = 0.0008

# TLOB multi-horizon
[labels]
horizons = [10, 20, 50, 100, 200]
smoothing_window = 10
threshold = 0.0008

# Opportunity detection
[labels]
strategy = "opportunity"
horizons = [50, 100, 200]
threshold = 0.005
conflict_priority = "larger_magnitude"

# Triple Barrier with per-horizon barriers
[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_targets = [0.0028, 0.0039, 0.0059]
stop_losses = [0.0019, 0.0026, 0.0040]

# Triple Barrier with volatility-adaptive scaling
[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_targets = [0.0020, 0.0028, 0.0042]
stop_losses = [0.0013, 0.0019, 0.0028]
volatility_scaling = true
volatility_reference = 0.00015
volatility_floor = 0.3
volatility_cap = 3.0
```

**Output Shapes:**
- Single-horizon: `labels.npy` shape `(N_seq,)` int8
- Multi-horizon: `labels.npy` shape `(N_seq, num_horizons)` int8

### Export Process (Aligned)

1. Generate labels from mid-prices FIRST
2. Align sequences with labels at sequence endpoints (1:1 mapping)
3. Apply market-structure preserving normalization
4. Write 3D sequences `[N_seq, window_size, n_features]` to NumPy
5. Write 1D labels `[N_seq]` to NumPy
6. Write metadata JSON with validation info
7. Write normalization params JSON

### TensorFormatter (Model-Specific Shapes)

Different deep learning models expect LOB data in different tensor shapes.
The `TensorFormatter` provides efficient reshaping for model-specific formats.

#### TensorFormat

```rust
pub enum TensorFormat {
    /// Flat features: (T, F) - for TLOB, LSTM, MLP
    Flat,
    
    /// DeepLOB format: (T, 4, L) - channels [ask_p, ask_v, bid_p, bid_v]
    DeepLOB { levels: usize },
    
    /// HLOB format: (T, L, 4) - level-first ordering
    HLOB { levels: usize },
    
    /// Image format: (T, C, H, W) - for CNN models
    Image { channels: usize, height: usize, width: usize },
}

impl TensorFormat {
    pub fn output_shape(&self, seq_len: usize, n_features: usize) -> Vec<usize>;
}
```

#### FeatureMapping

Maps feature indices to tensor positions:

```rust
pub struct FeatureMapping {
    pub ask_price_start: usize,   // Start index of ask prices (default: 0)
    pub ask_volume_start: usize,  // Start index of ask volumes (default: 10)
    pub bid_price_start: usize,   // Start index of bid prices (default: 20)
    pub bid_volume_start: usize,  // Start index of bid volumes (default: 30)
    pub levels: usize,            // Number of levels
}

impl FeatureMapping {
    pub fn standard_lob(levels: usize) -> Self;     // Standard 40-feature LOB
    pub fn with_derived(levels: usize) -> Self;     // LOB + 8 derived features
}
```

#### TensorFormatter

```rust
pub struct TensorFormatter {
    format: TensorFormat,
    mapping: FeatureMapping,
}

impl TensorFormatter {
    pub fn new(format: TensorFormat, mapping: FeatureMapping) -> Self;
    pub fn deeplob(levels: usize) -> Self;     // Convenience constructor
    pub fn hlob(levels: usize) -> Self;        // Convenience constructor
    
    pub fn format_sequence(&self, features: &[Vec<f64>]) -> Result<TensorOutput>;
    pub fn format_batch(&self, sequences: &[Vec<Vec<f64>>]) -> Result<TensorOutput>;
}
```

#### TensorOutput

Type-safe output container:

```rust
pub enum TensorOutput {
    Flat2D(Array2<f64>),      // (T, F)
    Channel3D(Array3<f64>),   // (T, C, L) or (T, L, C)
    Image4D(Array4<f64>),     // (T, C, H, W)
}

impl TensorOutput {
    pub fn shape(&self) -> Vec<usize>;
    pub fn as_flat(&self) -> Option<&Array2<f64>>;
    pub fn as_channel(&self) -> Option<&Array3<f64>>;
    pub fn as_image(&self) -> Option<&Array4<f64>>;
}
```

#### Format Comparison

| Format | Shape | Models | When to Use |
|--------|-------|--------|-------------|
| Flat | (T, F) | TLOB, LSTM, MLP | Sequence models, attention |
| DeepLOB | (T, 4, L) | DeepLOB, CNN-LSTM | CNN with channel semantics |
| HLOB | (T, L, 4) | HLOB | Level-aware architectures |
| Image | (T, C, H, W) | CNN | Image-processing pipelines |

#### Usage via PipelineOutput

```rust
// Method 1: With pre-configured formatter
let formatter = TensorFormatter::deeplob(10);
let tensor = output.format_sequences(&formatter)?;

// Method 2: Convenience method
let tensor = output.format_as(
    TensorFormat::DeepLOB { levels: 10 },
    FeatureMapping::standard_lob(10)
)?;
```

### Normalization Output

The `AlignedBatchExporter` exports normalization parameters alongside sequences,
enabling Python-side denormalization and validation.

#### Normalization Strategy

```rust
/// Normalization strategy used for feature export.
pub enum NormalizationStrategy {
    None,                    // Raw values
    PerFeatureZScore,        // Independent z-score per feature
    MarketStructureZScore,   // Shared price stats (ask+bid per level)
    GlobalZScore,            // All features share mean/std
    Bilinear,                // (price - mid) / (k * tick)
}
```

**Default: `MarketStructureZScore`** — Preserves market structure (ask > bid ordering).

#### Normalization Params Structure

```rust
pub struct NormalizationParams {
    pub strategy: NormalizationStrategy,
    pub price_means: Vec<f64>,    // 10 values (per level, shared ask+bid)
    pub price_stds: Vec<f64>,     // 10 values
    pub size_means: Vec<f64>,     // 20 values (10 ask + 10 bid)
    pub size_stds: Vec<f64>,      // 20 values
    pub sample_count: usize,       // Samples used for statistics
    pub feature_layout: String,    // "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10"
    pub levels: usize,             // Number of LOB levels
}
```

#### Export Output Files

```
{day}/
├── {day}_sequences.npy       # Normalized features [N, T, F]
├── {day}_labels.npy          # Labels [N] or [N, H]
├── {day}_normalization.json  # Normalization params (NEW)
└── {day}_metadata.json       # Includes normalization section
```

#### Example `{day}_normalization.json`

```json
{
  "strategy": "market_structure_z_score",
  "price_means": [120.50, 120.51, 120.52, ...],
  "price_stds": [0.05, 0.06, 0.07, ...],
  "size_means": [150.0, 145.0, ...],
  "size_stds": [50.0, 48.0, ...],
  "sample_count": 10000,
  "feature_layout": "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10",
  "levels": 10
}
```

#### Example `{day}_metadata.json` (normalization section)

```json
{
  "normalization": {
    "strategy": "market_structure_zscore",
    "applied": true,
    "levels": 10,
    "sample_count": 10000,
    "feature_layout": "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10",
    "params_file": "2025-02-03_normalization.json"
  }
}
```

#### Python Usage

```python
import json
import numpy as np

# Load normalization params
with open("2025-02-03_normalization.json") as f:
    norm_params = json.load(f)

# Denormalize prices (for interpretability)
def denormalize_prices(normalized, means, stds, levels=10):
    result = normalized.copy()
    for level in range(levels):
        # Ask prices (0-9)
        result[:, :, level] = normalized[:, :, level] * stds[level] + means[level]
        # Bid prices (20-29)
        result[:, :, 20 + level] = normalized[:, :, 20 + level] * stds[level] + means[level]
    return result
```

### Export Metadata Contract

Every `{day}_metadata.json` now includes standardized fields enforced by the pipeline contract:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Pipeline schema version (e.g., "2.2") |
| `contract_version` | string | Same as schema_version |
| `label_strategy` | string | "tlob", "opportunity", or "triple_barrier" |
| `label_encoding` | object | Strategy-specific class name mapping |
| `normalization` | object | Strategy, applied flag, params_file |
| `provenance` | object | Git commit, dirty flag, extractor version, config hash |
| `processing` | object | Messages processed, features extracted, sequences stats |
| `export_timestamp` | string | ISO 8601 timestamp (UTC) |

#### Provenance Block

Embedded in every metadata JSON and `dataset_manifest.json`:

```json
{
  "provenance": {
    "extractor_version": "0.10.0",
    "git_commit": "abc123def...",
    "git_dirty": false,
    "config_hash": "d41d8cd98f00...",
    "contract_version": "2.2",
    "export_timestamp_utc": "2026-03-06T12:00:00Z"
  }
}
```

The git hash and dirty status are captured at compile time via `build.rs`. The config hash is an MD5 digest of the serialized `DatasetConfig`, ensuring every exported dataset is traceable to its exact configuration.

#### Build-Time Provenance (`build.rs`)

```rust
// Captured at compile time, available as:
env!("GIT_COMMIT_HASH")  // Full SHA-1 hash
env!("GIT_DIRTY")        // "true" or "false"
```

---

## 10. Dataset Configuration System

The pipeline provides a configuration-driven, symbol-agnostic export system via `DatasetConfig`.

### Purpose

- **Model-agnostic**: Exports features, not trading decisions
- **Symbol-agnostic**: Works for any instrument (NVDA, AAPL, etc.)
- **Configuration-driven**: All parameters via TOML/JSON, no hard-coding
- **Flexible feature sets**: Support 40, 48, 76, 84, or 98 features
- **Reproducible**: Serializable configurations for experiment tracking

### DatasetConfig Structure

```rust
pub struct DatasetConfig {
    pub symbol: SymbolConfig,          // Symbol name, exchange, filename pattern
    pub data: DataPathConfig,          // Input/output directories, hot store
    pub dates: DateRangeConfig,        // Date range with weekend exclusion
    pub features: FeatureSetConfig,    // Feature selection (derived, MBO, signals)
    pub sampling: ExportSamplingConfig, // Event-based or volume-based sampling
    pub sequence: ExportSequenceConfig, // Window size, stride, buffer
    pub labels: ExportLabelConfig,     // Horizon, smoothing, threshold
    pub split: SplitConfig,            // Train/val/test ratios
    pub processing: ProcessingConfig,  // Threads, error mode
}
```

### Configuration Components

| Component | Purpose | Key Fields |
|-----------|---------|------------|
| `SymbolConfig` | Symbol definition | `name`, `exchange`, `filename_pattern`, `tick_size` |
| `DataPathConfig` | Data locations | `input_dir`, `output_dir`, `hot_store_dir` |
| `DateRangeConfig` | Date selection | `start_date`, `end_date`, `exclude_weekends` |
| `FeatureSetConfig` | Feature flags | `include_derived`, `include_mbo`, `include_signals` |
| `ExportSamplingConfig` | Sampling strategy | `strategy`, `target_volume` or `event_count` |
| `ExportSequenceConfig` | Sequence building | `window_size`, `stride`, `max_buffer` |
| `ExportLabelConfig` | Label generation | `horizon`, `horizons`, `smoothing_window`, `threshold` |
| `SplitConfig` | Data splits | `train_ratio`, `val_ratio`, `test_ratio` |
| `ProcessingConfig` | Parallelism | `num_threads`, `error_mode` |

### Feature Count Configurations

| Configuration | Count | FeatureSetConfig |
|---------------|-------|------------------|
| Raw LOB only | 40 | `include_derived: false, include_mbo: false, include_signals: false` |
| + Derived | 48 | `include_derived: true` |
| + MBO | 76 | `include_mbo: true` |
| + Derived + MBO | 84 | `include_derived: true, include_mbo: true` |
| + Signals | 98 | `include_derived: true, include_mbo: true, include_signals: true` |

### TOML Configuration Example

```toml
# configs/nvda_98feat.toml

[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
tick_size = 0.01  # Minimum price increment (default: 0.01 for US stocks)

[data]
input_dir = "../data/databento/NVDA"
output_dir = "../data/exports/nvda_98feat"
hot_store_dir = "../data/hot_store/NVDA"

[dates]
start_date = "2025-02-03"
end_date = "2025-02-28"
exclude_weekends = true

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true

[sampling]
strategy = "VolumeBased"
target_volume = 1000

[sequence]
window_size = 100
stride = 10

[labels]
horizon = 50
smoothing_window = 10
threshold = 0.0008

[split]
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

[processing]
num_threads = 8
error_mode = "CollectErrors"
```

### CLI Export Tool

```bash
# Generate a template configuration
cargo run --release --bin export_dataset -- --generate-config configs/my_dataset.toml

# Execute export using configuration
cargo run --release --features parallel --bin export_dataset -- --config configs/nvda_98feat.toml
```

### Programmatic Usage

```rust
use feature_extractor::export::DatasetConfig;

// Load and validate configuration
let config = DatasetConfig::load_toml("configs/nvda_98feat.toml")?;
config.validate()?;

// Convert to PipelineConfig (feature counts and tick_size auto-propagated from SymbolConfig)
let pipeline_config = config.to_pipeline_config();

// Use with BatchProcessor
let batch_config = config.processing.to_batch_config(&config.data.hot_store_dir);
let processor = BatchProcessor::new(pipeline_config, batch_config);
```

---

## 11. Parallel Batch Processing

> **Feature Gate**: Requires `--features parallel` to enable.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BatchProcessor                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Rayon Thread Pool                          ││
│  │                                                              ││
│  │  Thread 1        Thread 2        Thread N                   ││
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                  ││
│  │  │Pipeline │    │Pipeline │    │Pipeline │   (each owns)    ││
│  │  │   #1    │    │   #2    │    │   #N    │                  ││
│  │  └────┬────┘    └────┬────┘    └────┬────┘                  ││
│  │       │              │              │                        ││
│  │  Day1.dbn       Day2.dbn       DayN.dbn                     ││
│  │       │              │              │                        ││
│  │       ▼              ▼              ▼                        ││
│  │  DayResult      DayResult      DayResult                    ││
│  └──────────────────────┬───────────────────────────────────────┘│
│                         ▼                                        │
│                   BatchOutput                                    │
│                   (was_cancelled, skipped_count)                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Types

```rust
/// Thread-safe cancellation token
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self;
    pub fn cancel(&self);           // Signal cancellation
    pub fn is_cancelled(&self) -> bool;
    pub fn reset(&self);            // For reuse
}

/// Batch processing configuration
pub struct BatchConfig {
    pub num_threads: Option<usize>,  // None = Rayon default
    pub error_mode: ErrorMode,       // FailFast or CollectErrors
    pub report_progress: bool,
    pub stack_size: Option<usize>,   // Advanced: per-thread stack
    pub hot_store_dir: Option<PathBuf>,  // Hot store for decompressed files
}

/// Error handling strategy
pub enum ErrorMode {
    FailFast,      // Stop on first error
    CollectErrors, // Continue, collect all errors
}

/// Result from processing a single file
#[derive(Clone)]
pub struct DayResult {
    pub day: String,
    pub file_path: String,
    pub output: PipelineOutput,  // Clone via Arc internally
    pub elapsed: Duration,
    pub thread_id: usize,
}

/// Aggregated results from batch processing
pub struct BatchOutput {
    pub results: Vec<DayResult>,
    pub errors: Vec<FileError>,
    pub elapsed: Duration,
    pub threads_used: usize,
    pub was_cancelled: bool,    // True if cancelled before completion
    pub skipped_count: usize,   // Files skipped due to cancellation
}
```

### BatchProcessor

```rust
pub struct BatchProcessor {
    pipeline_config: Arc<PipelineConfig>,
    batch_config: BatchConfig,
    progress_callback: Option<Arc<dyn ProgressCallback>>,
    cancellation_token: CancellationToken,
    hot_store_manager: Option<Arc<HotStoreManager>>,  // Auto-created from config
}

impl BatchProcessor {
    // Auto-creates HotStoreManager if batch_config.hot_store_dir is set
    pub fn new(pipeline_config: PipelineConfig, batch_config: BatchConfig) -> Self;
    
    // Builder methods
    pub fn with_progress_callback(self, callback: Box<dyn ProgressCallback>) -> Self;
    pub fn with_cancellation_token(self, token: CancellationToken) -> Self;
    pub fn with_hot_store(self, hot_store: HotStoreManager) -> Self;  // Override config
    
    // Cancellation
    pub fn cancel(&self);
    pub fn is_cancelled(&self) -> bool;
    pub fn cancellation_token(&self) -> CancellationToken;
    
    // Processing (uses local thread pool for correct parallel execution)
    pub fn process_files<P: AsRef<Path> + Sync>(&self, files: &[P]) -> Result<BatchOutput>;
}
```

### Usage Pattern

```rust
use feature_extractor::prelude::*;
use feature_extractor::batch::{BatchProcessor, BatchConfig, CancellationToken, ErrorMode};

// Configure
let pipeline_config = PipelineBuilder::new()
    .lob_levels(10)
    .event_sampling(1000)
    .window(100, 10)
    .build_config()?;

let batch_config = BatchConfig::new()
    .with_threads(8)                          // Use 8 threads
    .with_error_mode(ErrorMode::CollectErrors) // Continue on errors
    .with_hot_store_dir("data/hot_store/");   // Use decompressed files (~30% faster)

// Create processor with cancellation support
// HotStoreManager is auto-created from batch_config.hot_store_dir
let token = CancellationToken::new();
let processor = BatchProcessor::new(pipeline_config, batch_config)
    .with_cancellation_token(token.clone());

// Process files in parallel (prefers decompressed versions if available)
let files = vec!["day1.dbn.zst", "day2.dbn.zst", "day3.dbn.zst"];
let output = processor.process_files(&files)?;

// Check results
println!("Processed: {}", output.successful_count());
println!("Failed: {}", output.failed_count());
println!("Cancelled: {}", output.was_cancelled);
println!("Throughput: {:.2} msg/sec", output.throughput_msg_per_sec());
```

### Cancellation Pattern

```rust
use std::thread;
use std::time::Duration;

let token = CancellationToken::new();
let processor = BatchProcessor::new(config, batch_config)
    .with_cancellation_token(token.clone());

// Cancel from another thread
let cancel_token = token.clone();
thread::spawn(move || {
    thread::sleep(Duration::from_secs(30));
    cancel_token.cancel();
});

let output = processor.process_files(&files)?;

if output.was_cancelled {
    println!("Cancelled after {} files", output.successful_count());
    println!("Skipped {} files", output.skipped_count);
}
```

### Thread Isolation

**CRITICAL**: Each thread creates its OWN Pipeline instance from the shared config:

```rust
// Inside process_files() - each thread does:
let mut pipeline = Pipeline::from_config((*self.pipeline_config).clone())?;
let output = pipeline.process(&file_path)?;
```

This ensures:
- No shared mutable state between threads
- No mutex contention
- BIT-LEVEL identical results to sequential processing

### Convenience Functions

```rust
// Process with default settings
let output = process_files_parallel(&config, &files)?;

// Process with specific thread count
let output = process_files_with_threads(&config, &files, 8)?;
```

---

## 12. Configuration System

### PipelineConfig

```rust
pub struct PipelineConfig {
    pub features: FeatureConfig,
    pub sequence: SequenceConfig,
    pub sampling: Option<SamplingConfig>,
    pub metadata: Option<ExperimentMetadata>,
}

impl PipelineConfig {
    // Auto-sync feature count between configs
    pub fn with_features(mut self, config: FeatureConfig) -> Self {
        let count = config.feature_count();
        self.features = config;
        self.sequence.feature_count = count;  // Auto-sync!
        self
    }
    
    pub fn validate(&self) -> Result<(), String>;
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    pub fn load_toml<P: AsRef<Path>>(path: P) -> Result<Self>;
}
```

### SamplingConfig

```rust
pub struct SamplingConfig {
    pub strategy: SamplingStrategy,
    pub volume_threshold: Option<u64>,
    pub min_time_interval_ns: Option<u64>,
    pub event_count: Option<usize>,
    pub adaptive: Option<AdaptiveSamplingConfig>,
    pub multiscale: Option<MultiScaleSamplingConfig>,
}

pub enum SamplingStrategy {
    VolumeBased,
    EventBased,
    TimeBased,
    MultiScale,
}
```

### PipelineBuilder (Fluent API)

```rust
use feature_extractor::prelude::*;

let pipeline = PipelineBuilder::new()
    .with_lob_levels(10)
    .with_derived_features(true)
    .with_mbo_features(true)
    .with_volume_sampling(1000, 1)  // 1000 shares, 1ms min interval
    .with_window(100, 10)           // window=100, stride=10
    .build()?;
```

---

## 13. Validation

### FeatureValidator

```rust
pub struct FeatureValidator {
    config: ValidationConfig,
}

impl FeatureValidator {
    pub fn validate_lob(&self, lob: &LobState) -> ValidationResult;
    pub fn validate_features(&self, features: &[f64]) -> ValidationResult;
}
```

### Validation Checks

| Check | Type | Description |
|-------|------|-------------|
| Crossed quotes | Error | `bid > ask` |
| Locked quotes | Warning | `bid == ask` |
| Price ordering | Error | Bids not decreasing, asks not increasing |
| NaN/Inf | Error | Invalid float values |
| Price ranges | Warning | Outside configured bounds |
| Volume ranges | Warning | Exceeds maximum |
| Spread | Warning/Error | Too wide or negative |

---

## 14. Zero-Allocation APIs

The library provides zero-allocation APIs for high-performance hot paths.

### FeatureExtractor Methods

```rust
impl FeatureExtractor {
    // === Point-in-Time Extraction (base features only) ===
    
    /// Extract base features into a pre-allocated buffer (zero allocation)
    /// Returns 40-84 features depending on config (LOB + derived + MBO)
    pub fn extract_into(&mut self, lob_state: &LobState, output: &mut Vec<f64>) -> Result<()>;
    
    /// Extract and wrap in Arc for zero-copy sharing
    pub fn extract_arc(&mut self, lob_state: &LobState) -> Result<Arc<Vec<f64>>>;
    
    // === Signal Extraction (streaming, requires OFI warmup) ===
    
    /// Update OFI state on every LOB transition (required for signals)
    pub fn update_ofi(&mut self, lob: &LobState);
    
    /// Extract all features including signals (98 features for full config)
    /// Requires: include_signals=true, call update_ofi() on every LOB transition
    pub fn extract_with_signals(
        &mut self, lob: &LobState, ctx: &SignalContext, output: &mut Vec<f64>
    ) -> Result<()>;
    
    // === Feature Counts ===
    
    /// Base features (LOB + derived + MBO): what extract_into() produces
    pub fn base_feature_count(&self) -> usize;  // 40-84
    
    /// Total features (including signals): what extract_with_signals() produces
    pub fn feature_count(&self) -> usize;       // 40-98
    
    /// Check if OFI is warmed up for signal computation
    pub fn is_signals_warm(&self) -> bool;
}
```

### Signal Extraction Pattern

Signals are "streaming" features that require OFI accumulation across LOB transitions:

```rust
// 1. Configure with signals
let config = FeatureConfig::new(10)
    .with_derived(true)
    .with_mbo(true)
    .with_signals(true);
let mut extractor = FeatureExtractor::with_config(config);

// 2. On EVERY LOB transition (before sampling):
extractor.update_ofi(&lob_state);

// 3. At sampling points (e.g., after volume threshold):
let ctx = SignalContext::new(timestamp_ns, invalidity_delta);
let mut output = Vec::new();
extractor.extract_with_signals(&lob_state, &ctx, &mut output)?;
assert_eq!(output.len(), 98); // 40 + 8 + 36 + 14

// 4. On day/session boundaries:
extractor.reset(); // Clears OFI warmup state
```

### Feature Set Flexibility

The pipeline supports multiple feature set configurations without being tied to any specific model:

| Config | Features | Use Case |
|--------|----------|----------|
| `FeatureConfig::new(10)` | 40 | Raw LOB only (DeepLOB baseline) |
| `.with_derived(true)` | 48 | + market structure metrics |
| `.with_mbo(true)` | 76-84 | + MBO microstructure features |
| `.with_signals(true)` | 98 | Full signal layer for HFT |
```

### SequenceBuilder Arc-Native Methods

```rust
/// Type alias for zero-copy feature vectors
pub type FeatureVec = Arc<Vec<f64>>;

impl SequenceBuilder {
    /// Push features wrapped in Arc (zero-copy)
    /// 
    /// The Arc is cloned (8 bytes) instead of the Vec (672 bytes for 84 features).
    pub fn push_arc(&mut self, timestamp: u64, features: FeatureVec) -> Result<(), SequenceError>;
    
    /// Legacy method (wraps in Arc internally, then calls push_arc)
    pub fn push(&mut self, timestamp: u64, features: Vec<f64>) -> Result<(), SequenceError>;
}
```

### MultiScaleWindow Arc-Native Methods

```rust
impl MultiScaleWindow {
    /// Push to all scales with Arc (zero-copy sharing)
    /// 
    /// The same Arc is shared across fast/medium/slow builders.
    /// Memory savings: 16 bytes (2 Arc clones) vs 1,344 bytes (2 Vec clones).
    ///
    /// STREAMING MODE: Sequences are automatically built during push_arc().
    /// Each scale's try_build_sequence() is called after push, and built
    /// sequences are accumulated internally.
    pub fn push_arc(&mut self, timestamp: u64, features: FeatureVec);
    
    /// Legacy method (wraps in Arc, then calls push_arc)
    pub fn push(&mut self, timestamp: u64, features: Vec<f64>);
    
    /// Return all accumulated sequences from streaming mode.
    /// 
    /// Returns ALL sequences built during push_arc() calls, plus any
    /// final sequences that can be built from remaining buffer data.
    /// Returns None if no sequences were built at any scale.
    pub fn try_build_all(&mut self) -> Option<MultiScaleSequence>;
}
```

### Pipeline Hot Path (Optimized)

```rust
// Current implementation in Pipeline::process()
pub fn process<P: AsRef<Path>>(&mut self, path: P) -> Result<PipelineOutput> {
    let mut lob_state = LobState::new(self.levels);        // Reused buffer
    let mut feature_buffer: Vec<f64> = Vec::with_capacity(feature_count);
    
    for msg in loader.iter_messages()? {
        // Zero-allocation LOB update
        self.reconstructor.process_message_into(&msg, &mut lob_state)?;
        
        if self.should_sample(&msg) {
            // Zero-allocation feature extraction
            self.feature_extractor.extract_into(&lob_state, &mut feature_buffer)?;
            
            // Wrap once in Arc, share everywhere
            let features = Arc::new(std::mem::take(&mut feature_buffer));
            feature_buffer = Vec::with_capacity(feature_count);
            
            // Zero-copy sharing to sequence builders
            self.sequence_builder.push_arc(timestamp, features.clone())?;
            
            if let Some(ref mut ms) = self.multiscale_window {
                ms.push_arc(timestamp, features);  // Shares same Arc
            }
        }
    }
}
```

### Memory Savings Summary

| Operation | Before | After | Savings |
|-----------|--------|-------|---------|
| Per-sample feature extraction | 1 Vec alloc | 0 alloc (buffer reuse) | 100% |
| Per-sequence feature storage | 67.2 KB clone | 8 byte Arc clone | 99.99% |
| Multi-scale sharing | 2 × 672 byte clones | 2 × 8 byte clones | 98.8% |

---

## 15. Testing Patterns

### Test Infrastructure

#### Centralized Test Helpers (`tests/common/mod.rs`)

All integration tests import shared constants and helpers:

```rust
pub const HOT_STORE_DIR: &str = "...";    // Hot store decompressed MBO files
pub const COMPRESSED_DIR: &str = "...";    // Compressed .dbn.zst files
pub const MBP10_DIR: &str = "...";         // MBP-10 files

pub fn find_mbo_file(date: &str) -> Option<PathBuf>;  // Finds MBO file for YYYYMMDD
pub fn has_test_data() -> bool;                         // Checks data directory exists

macro_rules! skip_if_no_data { ... }  // Gracefully skips tests when data unavailable
```

#### Unit Tests (729 tests, `cargo test --lib`)

Each module contains `#[cfg(test)] mod tests` with:
- Formula validation against hand-calculated expected values
- Edge cases: `0.0`, `NaN`, `Inf`, near-zero inputs
- Shape/dimension assertions for feature vectors
- Sign convention checks (positive = bullish, negative = bearish)

#### Contract Validation (`tests/contract_validation_test.rs`)

11 assertions verifying `src/contract.rs` constants match `contracts/pipeline_contract.toml`:
- `SCHEMA_VERSION`, `STABLE_FEATURE_COUNT`, `SIGNAL_COUNT`
- `CATEGORICAL_INDICES`, `LOB_LEVELS`
- Feature group boundaries (LOB, derived, MBO, signal ranges)

#### Real-Data Validation (`tests/phase3_real_data_validation.rs`)

6 tests running against real NVIDIA MBO data (requires `data/` directory):

| Test | What it validates |
|------|-------------------|
| `test_determinism_full_pipeline` | Two runs of the same day produce identical output (excluding known non-deterministic features) |
| `test_feature_layout_validation` | Feature vector width matches `STABLE_FEATURE_COUNT`, all values finite |
| `test_cross_day_state_isolation` | Processing day A then B gives same result for B as processing B alone |
| `test_signal_spot_check` | Signal features have correct sign conventions and reasonable ranges |
| `test_mbo_feature_statistics` | MBO features show non-trivial variance (excludes config-dependent zero features) |
| `test_export_end_to_end` | Full export pipeline: .npy shapes, metadata JSON, no NaN/Inf, valid label ranges |

#### Golden Snapshot Regression (`tests/golden_snapshot.rs`)

- Fixture: `tests/fixtures/golden_feb3_100.json` (100 post-warmup feature vectors from Feb 3, 2025)
- First run: generates fixture from live pipeline output
- Subsequent runs: compares new output against fixture with tolerances
- Deterministic features: `ABS_TOL = 1e-10`
- Non-deterministic features (indices 70, 78-81, 83): `REL_TOL = 0.01` with warnings

### Running Tests

```bash
# Unit tests only (no data required)
cargo test --lib --features "parallel,databento"

# All tests including integration (requires data/)
cargo test --features "parallel,databento" -- --test-threads=4

# Specific integration test
cargo test --test phase3_real_data_validation --features "parallel,databento"

# Golden snapshot regression
cargo test --test golden_snapshot --features "parallel,databento"
```

---

## 16. Performance Considerations

### Hot Paths

| Function | Target | Notes |
|----------|--------|-------|
| `VolumeBasedSampler::should_sample()` | ~2-3 cycles | Zero allocations |
| `extract_raw_features()` | ~50 ns | Pre-allocated output vector |
| `MboWindow::push()` | O(1) amortized | Incremental counters |
| `SequenceBuilder::try_build_sequence()` | O(window_size) | Single allocation |

### Memory Budget

| Component | Memory | Notes |
|-----------|--------|-------|
| LOB features (10 levels) | 320 bytes | 40 × 8 bytes |
| Sequence buffer (1000) | ~320 KB | 1000 × 40 × 8 bytes |
| MBO aggregator | ~8 MB | Per symbol |
| Order tracker | ~2 MB | ~22K orders |

### Optimization Guidelines

1. **Pre-allocate vectors**: Use `Vec::with_capacity()`
2. **Reuse buffers**: Clear and reuse instead of reallocating
3. **Lazy evaluation**: MBO percentiles computed only when accessed
4. **Streaming sequences**: Build sequences during processing, not at end

---

## 17. Integration with MBO-LOB-Reconstructor

### Dependency Setup

```toml
[dependencies]
mbo-lob-reconstructor = { git = "https://github.com/nagarx/MBO-LOB-reconstructor.git" }
```

### Simplest Integration (Recommended)

The `Pipeline` handles all complexity internally - LOB reconstruction, sampling, feature extraction,
and sequence building are all managed automatically:

```rust
use feature_extractor::prelude::*;

fn process_file(path: &str) -> Result<PipelineOutput> {
    // Build pipeline with desired configuration
    let mut pipeline = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()      // +8 features (48 total)
        .with_mbo_features()          // +36 features (84 total)
        .window(100, 10)              // 100 snapshots, stride 10
        .volume_sampling(1000)        // Sample every 1000 shares
        .build()?;
    
    // Process entire file - returns complete output
    let output = pipeline.process(path)?;

    println!("Processed {} messages", output.messages_processed);
    println!("Generated {} sequences", output.sequences_generated);
    
    Ok(output)
}
```

### With Hot Store (Faster)

For ~30% faster processing, use pre-decompressed files:

```rust
use feature_extractor::prelude::*;
use mbo_lob_reconstructor::{DbnSource, HotStoreManager};

fn process_with_hot_store(path: &str, hot_store_dir: &str) -> Result<PipelineOutput> {
    let mut pipeline = PipelineBuilder::new()
        .with_derived_features()
        .volume_sampling(1000)
        .build()?;

    // Create source that prefers decompressed files
    let hot_store = HotStoreManager::new(hot_store_dir)?;
    let source = DbnSource::with_hot_store(path, &hot_store)?;

    // Process through hot store
    pipeline.process_source(source)
}
```

### Multi-Day Processing

```rust
use feature_extractor::prelude::*;

fn process_multiple_days(paths: &[&str]) -> Result<()> {
    let mut pipeline = PipelineBuilder::new()
        .with_derived_features()
        .volume_sampling(1000)
        .build()?;

    let exporter = BatchExporter::new("output/", Some(LabelConfig::short_term()));
    
    for (i, path) in paths.iter().enumerate() {
        // IMPORTANT: Reset between days to prevent state leakage
        pipeline.reset();

        // Process the day
        let output = pipeline.process(path)?;

        // Export with labels
        exporter.export_day(&format!("day_{}", i), &output)?;
    }
    
    Ok(())
}
```

### Parallel Batch Processing

For multi-day processing on multi-core machines (requires `parallel` feature):

```rust
use feature_extractor::prelude::*;
use feature_extractor::batch::{BatchProcessor, BatchConfig, ErrorMode};

fn process_parallel(paths: &[&str]) -> Result<BatchOutput> {
    // Configure pipeline
    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()
        .volume_sampling(1000)
        .build_config()?;

    // Configure batch processing
    let batch_config = BatchConfig::new()
        .with_threads(8)                          // 8 parallel workers
        .with_error_mode(ErrorMode::CollectErrors) // Continue on errors
        .with_hot_store_dir("data/hot_store/");   // ~30% faster

    // Process all files in parallel
    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let output = processor.process_files(paths)?;

    println!("Processed {} days in {:?}", output.successful_count(), output.elapsed);
    println!("Throughput: {:.2} msg/sec", output.throughput_msg_per_sec());

    Ok(output)
}
```

### Custom Message Processing

For advanced use cases where you need control over the message stream:

```rust
use feature_extractor::prelude::*;
use mbo_lob_reconstructor::MboMessage;

fn process_custom_messages(messages: impl Iterator<Item = MboMessage>) -> Result<PipelineOutput> {
    let mut pipeline = PipelineBuilder::new()
        .with_derived_features()
        .volume_sampling(1000)
        .build()?;

    // Process custom iterator
    pipeline.process_messages(messages)
}

---

## 18. Common Patterns and Idioms

### Feature Count Synchronization

**Problem**: `SequenceConfig.feature_count` must match `FeatureConfig.feature_count()`

**Solution**: Use `PipelineConfig::with_features()` which auto-syncs:

```rust
// CORRECT: Auto-synced
let config = PipelineConfig::default()
    .with_features(FeatureConfig::default().with_derived(true));
// config.sequence.feature_count is automatically set to 48

// WRONG: Manual mismatch
let mut config = PipelineConfig::default();
config.features.include_derived = true;
// config.sequence.feature_count is still 40! Will fail validation.
```

### Streaming vs Batch Sequence Generation

**Streaming** (Recommended - used by Pipeline internally):
```rust
// Pipeline.process() uses streaming mode internally:
// - After each feature push, try_build_sequence() is called
// - Complete sequences are immediately accumulated
// - No data loss due to buffer eviction
let output = pipeline.process("data.dbn.zst")?;
// output.sequences contains all generated sequences
```

**Batch** (Direct SequenceBuilder use - limited buffer):
```rust
// WARNING: Only generates sequences from buffer (max 1000 snapshots)
// If you use SequenceBuilder directly (not via Pipeline):
let sequences = sequence_builder.generate_all_sequences();
// May lose data if more than buffer_size snapshots were pushed!
```

### Reset Semantics

All stateful components follow the same reset contract: after `reset()`, the component
behaves identically to a freshly constructed instance. This is critical for multi-day
processing to prevent cross-day state leakage.

```rust
// LobReconstructor
reconstructor.reset();       // Clears book, PRESERVES stats
reconstructor.full_reset();  // Clears EVERYTHING

// Pipeline  
pipeline.reset();            // Clears ALL state (calls reset on all sub-components)

// FeatureExtractor
extractor.reset();           // Clears MBO aggregator state

// SequenceBuilder
builder.reset();             // Clears buffer and counters

// MultiScaleWindow
ms_window.reset();           // Clears builders, counters, AND accumulated sequences
                             // CRITICAL: Prevents cross-day sequence leakage

// OfiComputer
ofi.reset_on_clear();        // Clears OFI accumulators and warmup state
```

**Multi-Day Processing Pattern**:
```rust
for day_file in data_files {
    pipeline.reset();  // CRITICAL: Clear all state before each day
    let output = pipeline.process(&day_file)?;
    // Export day's sequences...
}
```

---

## 19. Known Limitations

### 1. MBO Features Initial NaN

MBO features may return NaN until sufficient events processed:
```rust
// Solution: Check window population before using features
if aggregator.medium_window.len() >= 100 {
    let features = aggregator.extract_features(&lob_state);
}
```

### 2. Fixed Feature Count

Feature count is determined at pipeline creation and cannot change:
```rust
// Cannot change feature count after creation
let pipeline = Pipeline::from_config(config)?;
// pipeline.add_mbo_features(); // Does not exist!
```

### 3. Single-Symbol Pipeline

Each pipeline instance handles one symbol:
```rust
// For multi-symbol: create one pipeline per symbol
let pipelines: HashMap<String, Pipeline> = symbols
    .iter()
    .map(|s| (s.clone(), Pipeline::new()))
    .collect();
```

### 4. Label Generation Lag

Labels require future data, creating a lag:
```rust
// With horizon=10, smoothing=5: need 21 prices for first label
// min_prices_required() = k + h + k + 1 = 5 + 10 + 5 + 1 = 21
```

### 5. Buffer Size vs Data Loss (Regular SequenceBuilder)

Regular sequence builder has fixed buffer; old data evicted:
```rust
// If processing 10,000 snapshots with buffer_size=1000
// and using generate_all_sequences() at end: 9,000 snapshots LOST
// SOLUTION: Use streaming mode (try_build_sequence() during processing)
```

**Note**: `MultiScaleWindow` now builds sequences automatically during `push_arc()`,
accumulating them internally. Call `try_build_all()` at the end to retrieve all
accumulated sequences (not just one per scale).

### 6. Signals Require 10 LOB Levels

Trading signals (`include_signals: true`) require exactly 10 LOB levels:
```rust
// Valid: 10 levels with signals
let config = FeatureConfig::new(10).with_signals(true); // OK

// Invalid: 5 levels with signals - validation fails
let config = FeatureConfig {
    lob_levels: 5,
    include_signals: true,
    ..Default::default()
};
assert!(config.validate().is_err()); // Error: requires 10 levels
```

Signal indices are hardcoded for the 10-level layout. Non-10-level configurations
with signals enabled will fail validation.

### 7. Magnitude Export Not Integrated

The `MagnitudeGenerator` (regression targets) is fully implemented and tested but NOT integrated
into `export_dataset`. Use the Rust API directly.

> **Note**: Triple Barrier labeling IS fully integrated into the export pipeline (Schema 2.4+),
> including per-horizon barriers, volatility-adaptive scaling, and TOML configuration support.

**Tracking**: See roadmap in [docs/LABELING_STRATEGIES.md](docs/LABELING_STRATEGIES.md#roadmap).

### 8. OrderTracker HashMap Non-Determinism

**Root cause**: `OrderTracker` in `src/features/mbo_features/order_tracker.rs` uses `HashMap<u64, OrderInfo>` with Rust's default `RandomState` hasher. The hasher seed is randomized per process, causing iteration order to vary across runs. Features that aggregate over `HashMap` values (sums, means, counts) produce different results due to floating-point non-associativity.

**Affected feature indices** (MBO block, offsets from MBO base index 48):

| Index | Feature | Why non-deterministic |
|-------|---------|----------------------|
| 70 | `orders_per_level` | Aggregation over HashMap values |
| 78 | `avg_order_age` | Mean computed over HashMap iteration |
| 79 | `median_order_lifetime` | Order of collection from HashMap |
| 80 | `avg_fill_ratio` | Mean over HashMap values |
| 81 | `avg_time_to_first_fill` | Mean over HashMap values |
| 83 | `active_order_count` | Count derived from HashMap state |

**Impact**: Different f64 values across runs for the same input data. Differences are small (typically < 1e-6 relative) but fail strict bitwise equality.

**Current workaround**: Tests in `phase3_real_data_validation.rs` and `golden_snapshot.rs` use a `KNOWN_NONDETERMINISTIC` array to exclude these indices from strict equality, applying relaxed tolerance (`REL_TOL = 0.01`) with logged warnings instead.

**Fix plan**: Replace `HashMap<u64, OrderInfo>` with `BTreeMap<u64, OrderInfo>` (deterministic iteration order) or switch to `FxHashMap` with a fixed seed. This requires verifying no performance regression in the hot path, as `OrderTracker` processes every MBO event.

---

## Quick Reference

### Feature Counts

| Config | Count | Breakdown |
|--------|-------|-----------|
| Default | 40 | 10 levels × 4 |
| +Derived | 48 | 40 + 8 |
| +MBO | 76 | 40 + 36 |
| +Derived +MBO | 84 | 40 + 8 + 36 |
| +Derived +MBO +Signals | 98 | 40 + 8 + 36 + 14 |
| +Experimental (all) | 116 | 98 + 8 + 6 + 4 |

### Key Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| LOB levels | 10 | Standard for research |
| Window size | 100 | TLOB paper |
| Stride | 1 | Maximum overlap |
| Volume threshold | 1000 | Shares per sample |
| Min time interval | 1 ms | Prevents over-sampling |
| Label horizon | 10 | Steps ahead |
| Label threshold | 0.002 | 20 bps |

### Import Patterns

```rust
// Recommended: Full prelude (includes all common types)
use feature_extractor::prelude::*;

// Minimal (just for processing)
use feature_extractor::{Pipeline, PipelineBuilder, PipelineConfig};

// Specific components
use feature_extractor::{
    features::{FeatureConfig, FeatureExtractor},
    features::signals::{OfiComputer, OfiSample, TimeRegime, compute_signals},
    sequence_builder::{SequenceBuilder, SequenceConfig, FeatureVec},
    preprocessing::{VolumeBasedSampler, Normalizer},
    labeling::{LabelConfig, TlobLabelGenerator, MultiHorizonConfig},
    export_aligned::{AlignedBatchExporter, NormalizationStrategy},
    contract::{SCHEMA_VERSION, STABLE_FEATURE_COUNT, FULL_FEATURE_COUNT},
};

// Parallel processing (requires "parallel" feature)
#[cfg(feature = "parallel")]
use feature_extractor::batch::{BatchProcessor, BatchConfig, CancellationToken};
```

---

## Related Projects

### lob-dataset-analyzer (Python)

Python library for statistical analysis of exported datasets. Located at `../lob-dataset-analyzer/`.

**Key Capabilities (v0.4.0)**:
- **Unified Analyzer Protocol**: All analyzers inherit from `BaseAnalyzer` with consistent `.run()` interface
- **Full 98-Feature Analysis**: Analyze predictive power of ALL features, not just 8 signals
- **Centralized Configuration**: `FullAnalysisConfig` with JSON/YAML serialization for reproducibility
- **6 Metrics Per Feature**: Pearson, Spearman, Mutual Information, F-score, Kruskal-Wallis H, Consensus Rank

**Analyzer Classes**:
```python
from lobanalyzer.analysis import (
    PredictivePowerAnalyzer,      # Multi-metric predictive power
    SignalCorrelationAnalyzer,    # Cross-category correlations
    MultiHorizonAnalyzer,         # Horizon-aware analysis
    TemporalDynamicsAnalyzer,     # Autocorrelation, lead-lag
    GeneralizationAnalyzer,       # Walk-forward validation
    IntradaySeasonalityAnalyzer,  # Regime analysis
)

# All use consistent interface
config = PredictivePowerAnalyzer.default_config(Path("data"))
report = PredictivePowerAnalyzer(config=config).run()
print(report.summary())  # Human-readable
data = report.to_dict()  # With version metadata
```

**Documentation**: `../lob-dataset-analyzer/CODEBASE.md`, `../lob-dataset-analyzer/README.md`

---

*Last updated: March 6, 2026*
