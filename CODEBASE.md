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
10. [Parallel Batch Processing](#10-parallel-batch-processing) *(feature-gated)*
11. [Configuration System](#11-configuration-system)
12. [Validation](#12-validation)
13. [Zero-Allocation APIs](#13-zero-allocation-apis)
14. [Testing Patterns](#14-testing-patterns)
15. [Performance Considerations](#15-performance-considerations)
16. [Integration with MBO-LOB-Reconstructor](#16-integration-with-mbo-lob-reconstructor)
17. [Common Patterns and Idioms](#17-common-patterns-and-idioms)
18. [Known Limitations](#18-known-limitations)

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
src/
├── lib.rs                    # Public API exports
├── prelude.rs                # Convenience re-exports
├── builder.rs                # PipelineBuilder fluent API
├── pipeline.rs               # Main Pipeline orchestrator
├── config.rs                 # PipelineConfig, SamplingConfig
├── validation.rs             # Data quality validation
├── batch.rs                  # Parallel batch processing (feature-gated: parallel)
│
├── features/
│   ├── mod.rs                # FeatureConfig, FeatureExtractor, extract_into(), extract_arc()
│   ├── lob_features.rs       # Raw LOB features (40 features)
│   ├── derived_features.rs   # Derived metrics (8 features)
│   ├── mbo_features.rs       # MBO aggregated (36 features)
│   ├── signals.rs            # Trading signals (14 features): OfiComputer, TimeRegime
│   ├── order_flow.rs         # Order flow imbalance
│   ├── fi2010.rs             # FI-2010 benchmark features
│   └── market_impact.rs      # Market impact estimation
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
│   ├── mod.rs                # LabelConfig, TrendLabel, LabelStats
│   ├── tlob.rs               # TlobLabelGenerator
│   ├── deeplob.rs            # DeepLobLabelGenerator
│   └── multi_horizon.rs      # MultiHorizonLabelGenerator, ThresholdStrategy
│
├── schema/
│   ├── mod.rs                # Module exports
│   ├── presets.rs            # Preset feature configurations
│   └── feature_def.rs        # Feature definitions
│
├── export/                   # Export module (directory)
│   ├── mod.rs                # NumpyExporter, BatchExporter, re-exports
│   └── tensor_format.rs      # TensorFormat, TensorFormatter, TensorOutput
└── export_aligned.rs         # Aligned feature/label export
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

Extracted in `mbo_features.rs` using multi-timescale windows:

**Window Sizes**:
- Fast: 100 messages (~2 seconds)
- Medium: 1000 messages (~20 seconds) - **Primary extraction source**
- Slow: 5000 messages (~100 seconds)

**Feature Categories**:

| Range | Category | Count | Description |
|-------|----------|-------|-------------|
| 0-11 | Order Flow | 12 | Event rates, imbalances |
| 12-19 | Size Distribution | 8 | Percentiles, z-scores |
| 20-25 | Queue & Depth | 6 | Queue position, concentration |
| 26-29 | Institutional | 4 | Large order detection |
| 30-35 | Core MBO | 6 | Lifecycle metrics |

**MboAggregator Memory**: ~8 MB per symbol

### Trading Signals (14 for indices 84-97)

Computed in `signals.rs` using streaming OFI and base features:

**Implementation**:

| File | Module |
|------|--------|
| `src/features/signals.rs` | `OfiComputer`, `compute_signals()`, `TimeRegime` |

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
    
    // Warmup tracking
    state_changes_since_reset: u64,  // Must reach 100
    last_sample_timestamp: i64,
}

impl OfiComputer {
    /// Update on EVERY LOB state transition (critical for accuracy)
    pub fn update(&mut self, lob_state: &LobState);
    
    /// Sample OFI and reset accumulators (at sampling points)
    pub fn sample_and_reset(&mut self, timestamp: i64) -> OfiSample;
    
    /// Check if warmup complete (≥100 state changes)
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
| 90 | `fragility_score` | Book fragility indicator | [0, 1] |
| 91 | `depth_asymmetry` | Volume imbalance | [-1, 1] |
| 92 | `book_valid` | Valid book flag | {0, 1} |
| 93 | `time_regime` | Market session | {0, 1, 2, 3, 4} |
| 94 | `mbo_ready` | Warmup status | {0, 1} |
| 95 | `dt_seconds` | Time since last sample | [0, ∞) |
| 96 | `invalidity_delta` | Quote anomalies since last sample | [0, ∞) |
| 97 | `schema_version` | Always 2.0 | {2.0} |

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
    min_time_interval_ns: u64,  // Min time between samples (default: 1ms)
    last_sample_time: u64,
}

impl VolumeBasedSampler {
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

---

## 7. Normalization System

### Normalizer Trait

```rust
pub trait Normalizer: Send + Sync {
    fn update(&mut self, value: f64);
    fn normalize(&self, value: f64) -> f64;
    fn normalize_batch(&self, values: &[f64]) -> Vec<f64>;
    fn reset(&mut self);
    fn is_ready(&self) -> bool;
}
```

### Available Normalizers

| Normalizer | Formula | Use Case |
|------------|---------|----------|
| `PercentageChangeNormalizer` | `(x - ref) / ref` | Prices |
| `ZScoreNormalizer` | `(x - mean) / std` | Volumes, spreads |
| `RollingZScoreNormalizer` | Multi-day rolling Z-score | Cross-day normalization |
| `BilinearNormalizer` | `(x - mid) / (k × tick)` | LOB structure |
| `MinMaxNormalizer` | `(x - min) / (max - min)` | Bounded features |
| `GlobalZScoreNormalizer` | All features together | LOBench method |
| `PerFeatureNormalizer` | Separate stats per feature | Multi-feature |

### GlobalZScoreNormalizer (LOBench)

Normalizes ALL features in a snapshot together, preserving LOB constraints:

```rust
pub fn normalize_snapshot(&self, features: &[f64]) -> Vec<f64> {
    let mean: f64 = features.iter().sum::<f64>() / features.len() as f64;
    let variance: f64 = features.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt().max(self.min_std);
    
    features.iter().map(|&x| (x - mean) / std).collect()
}
```

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
}
```

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

## 9. Export Pipeline

### NumpyExporter

```rust
pub struct NumpyExporter {
    output_dir: PathBuf,
}

impl NumpyExporter {
    // Creates: features.npy, mid_prices.npy, metadata.json
    pub fn export(&self, output: &PipelineOutput) -> Result<()>;
}
```

### BatchExporter

For multi-day processing with optional labeling:

```rust
pub struct BatchExporter {
    output_dir: PathBuf,
    label_config: Option<LabelConfig>,
}

impl BatchExporter {
    // Creates: {day}_features.npy, {day}_labels.npy, {day}_metadata.json
    pub fn export_day(&self, day_name: &str, output: &PipelineOutput) -> Result<DayExportResult>;
}
```

### Export Process

1. Flatten sequences to `[N_samples × N_features]`
2. Apply z-score normalization (Stage 2 normalization)
3. Write to NumPy format
4. Generate labels from mid-prices (if configured)
5. Write metadata JSON

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

---

## 10. Parallel Batch Processing

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

## 11. Configuration System

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
    pub multiscale: Option<MultiScaleConfig>,
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

## 12. Validation

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

## 13. Zero-Allocation APIs

The library provides zero-allocation APIs for high-performance hot paths.

### FeatureExtractor Zero-Allocation Methods

```rust
impl FeatureExtractor {
    /// Extract features into a pre-allocated buffer (zero allocation)
    /// 
    /// The output buffer is cleared and reused, avoiding per-sample allocation.
    pub fn extract_into(&mut self, lob_state: &LobState, output: &mut Vec<f64>) -> Result<()>;
    
    /// Extract and wrap in Arc for zero-copy sharing
    /// 
    /// Convenience method that calls extract_into() then wraps in Arc.
    pub fn extract_arc(&mut self, lob_state: &LobState) -> Result<Arc<Vec<f64>>>;
    
    /// Legacy method (still works, uses extract_into internally)
    #[deprecated(note = "Use extract_into() for better performance")]
    pub fn extract_all_features(&mut self, lob_state: &LobState) -> Result<Vec<f64>>;
}
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
    pub fn push_arc(&mut self, timestamp: u64, features: FeatureVec);
    
    /// Legacy method (wraps in Arc, then calls push_arc)
    pub fn push(&mut self, timestamp: u64, features: Vec<f64>);
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

## 14. Testing Patterns

### Unit Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper to create realistic test LOB
    fn create_test_lob_state() -> LobState {
        let mut state = LobState::new(10);
        // Set realistic prices and sizes
        state.bid_prices[0] = 100_000_000_000;  // $100.00
        state.ask_prices[0] = 100_010_000_000;  // $100.01
        // ...
        state
    }
    
    #[test]
    fn test_feature_count_consistency() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::with_config(config.clone());
        let state = create_test_lob_state();
        
        let features = extractor.extract_lob_features(&state).unwrap();
        assert_eq!(features.len(), config.feature_count());
    }
}
```

### Integration Test Pattern

```rust
#[test]
fn test_full_pipeline() {
    // 1. Create pipeline
    let pipeline = Pipeline::from_config(PipelineConfig::default()).unwrap();
    
    // 2. Process messages
    for msg in test_messages {
        let lob_state = reconstructor.process_message(&msg).unwrap();
        pipeline.process(&msg, &lob_state);
    }
    
    // 3. Get output
    let output = pipeline.finalize();
    
    // 4. Verify
    assert!(output.sequences.len() > 0);
    assert_eq!(output.sequences[0].features[0].len(), pipeline.config().features.feature_count());
}
```

---

## 15. Performance Considerations

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

## 16. Integration with MBO-LOB-Reconstructor

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

## 17. Common Patterns and Idioms

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

```rust
// LobReconstructor
reconstructor.reset();       // Clears book, PRESERVES stats
reconstructor.full_reset();  // Clears EVERYTHING

// Pipeline  
pipeline.reset();            // Clears all state

// FeatureExtractor
extractor.reset();           // Clears MBO aggregator state

// SequenceBuilder
builder.reset();             // Clears buffer and counters
```

---

## 18. Known Limitations

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

### 5. Buffer Size vs Data Loss

Sequence builder has fixed buffer; old data evicted:
```rust
// If processing 10,000 snapshots with buffer_size=1000
// and using generate_all_sequences() at end: 9,000 snapshots LOST
// SOLUTION: Use streaming mode (try_build_sequence() during processing)
```

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
    features::signals::{OfiComputer, OfiSample, TimeRegime, compute_signals, SIGNAL_COUNT},
    sequence_builder::{SequenceBuilder, SequenceConfig, FeatureVec},
    preprocessing::{VolumeBasedSampler, Normalizer},
    labeling::{LabelConfig, TlobLabelGenerator, MultiHorizonConfig},
    export::{NumpyExporter, BatchExporter},
};

// Parallel processing (requires "parallel" feature)
#[cfg(feature = "parallel")]
use feature_extractor::batch::{BatchProcessor, BatchConfig, CancellationToken};
```

---

*Last updated: December 19, 2025*
