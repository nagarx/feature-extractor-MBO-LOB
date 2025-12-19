# Feature Extractor for MBO-LOB Data

High-performance feature extraction library for Limit Order Book (LOB) and Market-by-Order (MBO) data, designed for deep learning model training in high-frequency trading.

[![Rust](https://img.shields.io/badge/rust-1.83%2B-blue.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/nagarx/feature-extractor-MBO-LOB/workflows/CI/badge.svg)](https://github.com/nagarx/feature-extractor-MBO-LOB/actions)

## Overview

This library provides a modular, research-aligned feature extraction pipeline for LOB data. It works in conjunction with [MBO-LOB-reconstructor](https://github.com/nagarx/MBO-LOB-reconstructor) to transform raw Market-by-Order data into ML-ready features.

### Key Features

- **Fluent Builder API**: Simple, readable pipeline configuration
- **Configuration-Driven Exports**: TOML/JSON configs for reproducible experiments
- **Symbol-Agnostic**: Works for any instrument (NVDA, AAPL, etc.)
- **Paper-Aligned Presets**: DeepLOB, TLOB, FI-2010, TransLOB, LiT configurations
- **Auto-Computed Feature Count**: No manual calculation required
- **Comprehensive Feature Set**: Up to 98 features across multiple categories
- **Trading Signals**: 14 research-backed signals (OFI, microprice, time regime)
- **Label Generation**: TLOB and DeepLOB labeling methods for supervised learning
- **Multi-Horizon Labels**: Generate labels for multiple prediction horizons (FI-2010, DeepLOB presets)
- **TensorFormatter**: Model-specific tensor shapes (DeepLOB, HLOB, Flat, Image formats)
- **Multiple Normalization Strategies**: Z-score, Rolling Z-score, Global Z-score, Bilinear
- **Multi-Scale Sequences**: Fast/Medium/Slow temporal resolution
- **High Performance**: Single-pass computation, zero-allocation hot paths
- **Parallel Processing**: Multi-threaded batch processing with Rayon (optional)
- **Graceful Cancellation**: Cancel long-running jobs from any thread
- **Hot Store Support**: Pre-decompress data for ~30% faster processing
- **CLI Export Tool**: `export_dataset` binary for command-line exports

## Quick Start

### Using the Prelude (Recommended)

```rust
use feature_extractor::prelude::*;

fn main() -> Result<()> {
    // Build pipeline with fluent API
    let mut pipeline = PipelineBuilder::new()
        .lob_levels(10)           // 10 price levels (40 raw features)
        .with_derived_features()  // +8 derived features
        .window(100, 10)          // 100 snapshots per sequence, stride 10
        .event_sampling(1000)     // Sample every 1000 events
        .build()?;

    // Process MBO data
    let output = pipeline.process("data/SYMBOL.mbo.dbn.zst")?;

    println!("Processed {} messages", output.messages_processed);
    println!("Generated {} sequences", output.sequences_generated);
    println!("Features per snapshot: {}", output.sequences[0].features[0].len());

    // Export to NumPy for Python/PyTorch
    let exporter = NumpyExporter::new("output/");
    exporter.export_day("2025-02-03", &output)?;

    Ok(())
}
```

### Using Research Paper Presets

```rust
use feature_extractor::prelude::*;

// DeepLOB configuration (40 features)
let pipeline = PipelineBuilder::from_preset(Preset::DeepLOB)
    .volume_sampling(1000)
    .build()?;

// TLOB configuration (40 features)
let pipeline = PipelineBuilder::from_preset(Preset::TLOB)
    .event_sampling(1000)
    .build()?;

// Full feature set (84 features)
let pipeline = PipelineBuilder::from_preset(Preset::Full)
    .build()?;
```

## Pipeline Builder API

The `PipelineBuilder` provides a fluent API for configuring the entire pipeline:

```rust
let mut pipeline = PipelineBuilder::new()
    // Feature configuration
    .lob_levels(10)              // Number of LOB levels (default: 10)
    .with_derived_features()     // Enable 8 derived features
    .with_mbo_features()         // Enable 36 MBO features
    .mbo_window(1000)            // MBO aggregation window
    .tick_size(0.01)             // Tick size for US stocks
    
    // Sequence configuration
    .window(100, 10)             // Window size and stride
    .max_buffer(2000)            // Maximum buffer capacity
    
    // Sampling configuration
    .event_sampling(1000)        // Or: .volume_sampling(1000)
    .min_sample_interval_ns(1_000_000)
    .with_adaptive_sampling()    // Volatility-based adjustment
    .with_multiscale()           // Multi-scale windowing
    
    // Metadata
    .experiment("name", "description")
    
    .build()?;
```

### Feature Count Auto-Computation

The builder automatically computes the correct feature count:

```rust
// 40 features (raw LOB only)
let pipeline = PipelineBuilder::new().build()?;

// 48 features (40 raw + 8 derived)
let pipeline = PipelineBuilder::new()
    .with_derived_features()
    .build()?;

// 76 features (40 raw + 36 MBO)
let pipeline = PipelineBuilder::new()
    .with_mbo_features()
    .build()?;

// 84 features (40 raw + 8 derived + 36 MBO)
let pipeline = PipelineBuilder::new()
    .with_derived_features()
    .with_mbo_features()
    .build()?;

// 98 features (40 raw + 8 derived + 36 MBO + 14 signals)
let pipeline = PipelineBuilder::new()
    .with_derived_features()
    .with_mbo_features()
    .with_trading_signals()
    .build()?;
```

## Feature Categories

| Category | Count | Indices | Description |
|----------|-------|---------|-------------|
| Raw LOB | 40 | 0-39 | Prices and volumes at 10 levels |
| Derived | 8 | 40-47 | Mid-price, spread, imbalance, etc. |
| MBO Features | 36 | 48-83 | Order flow dynamics |
| Trading Signals | 14 | 84-97 | OFI, microprice, time regime, safety gates |
| **Total** | **98** | 0-97 | Full feature set with all options enabled |

### Additional Feature Sets (standalone usage)

| Category | Count | Description |
|----------|-------|-------------|
| Order Flow | 8 | OFI, queue imbalance, trade flow |
| Multi-Level OFI | 10 | OFI at each LOB level |
| FI-2010 Time-Insensitive | 20 | Spread, mid-price, price diffs |
| FI-2010 Time-Sensitive | 20 | Derivatives, intensity measures |
| FI-2010 Depth | 40 | Accumulated volumes and diffs |
| Market Impact | 8 | Slippage, execution quality |

## Trading Signals (14 Features)

Enable research-backed trading signals as **input features for ML models**:

> **Design Philosophy**: These signals are **model-agnostic features**, not a trading system.
> They provide high-frequency market microstructure information that models can learn to use.
> Entry/exit thresholds are NOT hard-coded—let your model learn what works.

```rust
let pipeline = PipelineBuilder::new()
    .lob_levels(10)
    .with_derived_features()  // Required
    .with_mbo_features()      // Required
    .with_trading_signals()   // +14 signals (indices 84-97)
    .build()?;

assert_eq!(pipeline.config().features.feature_count(), 98);
```

### Use Cases

| Use Case | Recommended Approach |
|----------|----------------------|
| **ML Training** | Use all 98 features; let model learn signal importance |
| **Experimentation** | Compare different feature subsets |
| **Data Quality** | Use `book_valid`, `mbo_ready` as input features for the model |
| **Rule-Based** | Optional—see plan docs for threshold examples (may overfit) |

### Signal Categories

| Category | Signals | Purpose |
|----------|---------|---------|
| Safety/Quality | `book_valid`, `mbo_ready`, `invalidity_delta` | Data quality indicators |
| Direction | `true_ofi`, `depth_norm_ofi`, `executed_pressure` | Order flow imbalance |
| Conviction | `trade_asymmetry`, `cancel_asymmetry` | Flow confirmation |
| Impact | `fragility_score`, `depth_asymmetry` | Market stability |
| Timing | `signed_mp_delta_bps`, `time_regime` | Microstructure context |
| Meta | `dt_seconds`, `schema_version` | Sample metadata |

### Research Foundation

| Signal | Research Paper |
|--------|----------------|
| `true_ofi` | Cont et al. (2014) "The Price Impact of Order Book Events" |
| `depth_norm_ofi` | Cont et al. (2014) - OFI normalized by average depth |
| `signed_mp_delta_bps` | Stoikov (2018) "The Micro-Price" |
| `time_regime` | Cont et al. §3.3 - Intraday price impact patterns |

### Time Regime

Automatic UTC → Eastern Time conversion with DST handling:

```rust
use feature_extractor::features::signals::TimeRegime;

// TimeRegime values:
// 0 = Open (9:30-9:45 ET) - Highest volatility
// 1 = Early (9:45-10:30 ET) - Settling period
// 2 = Midday (10:30-15:30 ET) - Most stable
// 3 = Close (15:30-16:00 ET) - Position squaring
// 4 = Closed - Outside market hours
```

## Sampling Strategies

### Volume-Based Sampling

Sample after a specified volume of shares has traded:

```rust
let pipeline = PipelineBuilder::new()
    .volume_sampling(1000)  // Sample every 1000 shares
    .build()?;
```

### Event-Based Sampling

Sample after a specified number of MBO events:

```rust
let pipeline = PipelineBuilder::new()
    .event_sampling(500)    // Sample every 500 events
    .build()?;
```

### Adaptive Sampling

Automatically adjust threshold based on market volatility:

```rust
let pipeline = PipelineBuilder::new()
    .volume_sampling(1000)
    .with_adaptive_sampling()
    .build()?;
```

## Label Generation

### TLOB Method (Recommended)

```rust
use feature_extractor::prelude::*;

let config = LabelConfig {
    horizon: 50,          // Predict 50 steps ahead
    smoothing_window: 10, // Smooth over 10 prices
    threshold: 0.002,     // 0.2% threshold
};

let mut generator = TlobLabelGenerator::new(config);
generator.add_prices(&output.mid_prices);

let labels = generator.generate_labels()?;
let stats = generator.compute_stats(&labels);

println!("Class balance: Up={:.1}%, Stable={:.1}%, Down={:.1}%",
    stats.up_count as f64 / stats.total as f64 * 100.0,
    stats.stable_count as f64 / stats.total as f64 * 100.0,
    stats.down_count as f64 / stats.total as f64 * 100.0,
);
```

### Configuration Presets

```rust
// HFT (short-term, tight threshold)
let config = LabelConfig::hft();  // h=10, k=5, threshold=0.02%

// Short-term trading
let config = LabelConfig::short_term();  // h=50, k=10, threshold=0.2%

// Medium-term trading
let config = LabelConfig::medium_term();  // h=100, k=20, threshold=0.5%

// FI-2010 benchmark (k = h)
let config = LabelConfig::fi2010(50);
```

### Multi-Horizon Label Generation

For benchmark reproduction, generate labels at multiple prediction horizons:

```rust
use feature_extractor::prelude::*;

// FI-2010 preset: horizons [10, 20, 30, 50, 100]
let config = MultiHorizonConfig::fi2010();

// Generate labels from pipeline output
let labels = output.generate_multi_horizon_labels(config)?;

// Access per-horizon labels
for horizon in labels.horizons() {
    let horizon_labels = labels.labels_for_horizon(*horizon).unwrap();
    println!("Horizon {}: {} labels", horizon, horizon_labels.len());
}

// Get summary statistics
let summary = labels.summary();
println!("Total labels: {}, Horizons: {}", summary.total_labels, summary.num_horizons);
```

#### Threshold Strategies

```rust
// Fixed threshold (default: 0.2%)
let config = MultiHorizonConfig::new(
    vec![10, 20, 50, 100],
    5,
    ThresholdStrategy::Fixed(0.002)
);

// Rolling spread-based (adaptive to market conditions)
let config = MultiHorizonConfig::new(
    vec![10, 20, 50, 100],
    5,
    ThresholdStrategy::rolling_spread(100, 1.0, 0.002)
);
```

## Tensor Formatting

Format sequences for specific model architectures:

```rust
use feature_extractor::prelude::*;

// DeepLOB format: (T, 4, L) with channels [ask_p, ask_v, bid_p, bid_v]
let formatter = TensorFormatter::deeplob(10);
let tensor = output.format_sequences(&formatter)?;

// HLOB format: (T, L, 4) level-first ordering
let formatter = TensorFormatter::hlob(10);
let tensor = output.format_sequences(&formatter)?;

// Flat format: (T, F) for LSTM/Transformer models
let tensor = output.format_as(TensorFormat::Flat, FeatureMapping::standard_lob(10))?;
```

#### Supported Formats

| Format | Shape | Models | Description |
|--------|-------|--------|-------------|
| `Flat` | (T, F) | TLOB, LSTM | Standard flat features |
| `DeepLOB` | (T, 4, L) | DeepLOB, CNN-LSTM | Channel-first format |
| `HLOB` | (T, L, 4) | HLOB | Level-first format |
| `Image` | (T, C, H, W) | CNN | Image-like representation |

## Export to NumPy

### Configuration-Driven Export (Recommended)

Use the `export_dataset` CLI tool with TOML configuration:

```bash
# Generate a template configuration
cargo run --release --bin export_dataset -- --generate-config configs/my_dataset.toml

# Edit the configuration file, then export
cargo run --release --features parallel --bin export_dataset -- --config configs/nvda_98feat.toml
```

Sample configuration (`configs/nvda_98feat.toml`):

```toml
[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"

[data]
input_dir = "../data/databento/NVDA"
output_dir = "../data/exports/nvda_98feat"

[dates]
start_date = "2025-02-03"
end_date = "2025-02-28"
exclude_weekends = true

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true  # Full 98-feature set

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

### Programmatic Export

```rust
use feature_extractor::export::DatasetConfig;

// Load and validate configuration
let config = DatasetConfig::load_toml("configs/nvda_98feat.toml")?;
config.validate()?;

// Convert to PipelineConfig
let pipeline_config = config.to_pipeline_config()?;

// Process files...
```

### Single Day Export (Legacy)

```rust
let exporter = NumpyExporter::new("output/");
exporter.export_day("2025-02-03", &output)?;
```

### Batch Export with Labels (Legacy)

```rust
let label_config = LabelConfig::short_term();
let exporter = BatchExporter::new("output/", Some(label_config));

for day in trading_days {
    pipeline.reset();  // Clear state between days
    let output = pipeline.process(&format!("data/{}.dbn.zst", day))?;
    exporter.export_day(&day, &output)?;
}
```

## Normalization Strategies

| Strategy | Use Case | Source |
|----------|----------|--------|
| `ZScoreNormalizer` | Standard ML preprocessing | DeepLOB |
| `RollingZScoreNormalizer` | Non-stationary data | LOBFrame |
| `GlobalZScoreNormalizer` | Preserve LOB constraints | LOBench |
| `BilinearNormalizer` | LOB structure preservation | TLOB |
| `PercentageChangeNormalizer` | Cross-instrument training | HLOB |

```rust
use feature_extractor::preprocessing::{GlobalZScoreNormalizer, Normalizer};

let normalizer = GlobalZScoreNormalizer::new();
let normalized = normalizer.normalize_snapshot(&features);
```

## Multi-Day Processing

```rust
let mut pipeline = PipelineBuilder::new()
    .with_derived_features()
    .event_sampling(1000)
    .build()?;

for day in trading_days {
    pipeline.reset();  // Critical: Reset state between days
    
    let path = format!("data/{}.dbn.zst", day);
    let output = pipeline.process(&path)?;
    
    // Export...
}
```

## Parallel Batch Processing

Enable the `parallel` feature for multi-threaded processing:

```bash
cargo build --features parallel
```

### Basic Usage

```rust
use feature_extractor::prelude::*;
use feature_extractor::batch::{BatchProcessor, BatchConfig, ErrorMode};

// Configure for 8-core machine
let batch_config = BatchConfig::new()
    .with_threads(8)
    .with_error_mode(ErrorMode::CollectErrors);

let pipeline_config = PipelineBuilder::new()
    .lob_levels(10)
    .event_sampling(1000)
    .build_config()?;

// Process multiple files in parallel
let processor = BatchProcessor::new(pipeline_config, batch_config);
let files = vec!["day1.dbn.zst", "day2.dbn.zst", "day3.dbn.zst"];

let output = processor.process_files(&files)?;

println!("Processed {} days in {:?}", output.successful_count(), output.elapsed);
println!("Throughput: {:.2} msg/sec", output.throughput_msg_per_sec());
```

### With Cancellation Support

```rust
use feature_extractor::batch::CancellationToken;
use std::thread;

let token = CancellationToken::new();
let processor = BatchProcessor::new(config, batch_config)
    .with_cancellation_token(token.clone());

// Cancel from another thread
let cancel_token = token.clone();
thread::spawn(move || {
    thread::sleep(std::time::Duration::from_secs(30));
    cancel_token.cancel();
});

let output = processor.process_files(&files)?;

if output.was_cancelled {
    println!("Cancelled after {} files", output.successful_count());
    println!("Skipped {} files", output.skipped_count);
}
```

### Convenience Functions

```rust
// Quick parallel processing with defaults
let output = process_files_parallel(&config, &files)?;

// Specify thread count
let output = process_files_with_threads(&config, &files, 8)?;
```

## Performance

The library is optimized for HFT environments:

- **Single-pass computation**: Features extracted in one LOB traversal
- **Zero-allocation hot paths**: `extract_into()` and `push_arc()` APIs
- **Arc-based sequences**: Zero-copy feature sharing (8-byte clone vs 672-byte)
- **Welford's algorithm**: Numerically stable running statistics
- **PriceLevel O(1) caching**: Constant-time size queries in reconstructor
- **Parallel processing**: ~64K msg/sec with multi-threading

### Benchmark Results

| Metric | Value |
|--------|-------|
| Sequential throughput | ~42K msg/sec |
| Parallel throughput (2 threads) | ~64K msg/sec |
| Memory per sequence | 8 bytes (Arc) vs 67.2 KB (Vec clone) |
| Feature extraction | 0 allocations (buffer reuse) |

> **Note**: The primary bottleneck is **zstd decompression** (single-threaded per file stream).
> Parallel processing helps by processing multiple files simultaneously, but throughput
> scales sub-linearly beyond 2-4 threads due to I/O saturation. For maximum throughput,
> consider pre-decompressing `.dbn.zst` files to uncompressed `.dbn` format.

Run benchmarks:

```bash
cargo bench
```

## Testing

```bash
# Run all tests
cargo test

# Run library tests only
cargo test --lib

# Run with verbose output
cargo test -- --nocapture

# Run parallel processing tests (requires --features parallel)
cargo test --features parallel --test parallel_processing_tests
```

## Documentation

- [Usage Guide](docs/USAGE_GUIDE.md) - Comprehensive usage documentation
- [Architecture](ARCHITECTURE.md) - Design documentation

## Research Papers

This library implements features from:

- DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
- TLOB: Transformer Model with Dual Attention for Price Trend Prediction
- FI-2010: Benchmark Dataset for Mid-Price Forecasting
- LOBench: Representation Learning of Limit Order Book
- ViT-LOB: Vision Transformer for Stock Price Trend Prediction
- **Price Impact: The Price Impact of Order Book Events (Cont et al. 2014)** - OFI signals
- **Micro-Price: A High Frequency Estimator of Future Prices (Stoikov 2018)** - Microprice delta
- Queue Imbalance: Queue Imbalance as a One-Tick-Ahead Price Predictor

## License

Proprietary - All Rights Reserved. See [LICENSE](LICENSE) for details.

No permission is granted to use, copy, modify, or distribute this software.
