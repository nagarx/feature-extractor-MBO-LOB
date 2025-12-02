# Feature Extractor for MBO-LOB Data

High-performance feature extraction library for Limit Order Book (LOB) and Market-by-Order (MBO) data, designed for deep learning model training in high-frequency trading.

[![Rust](https://img.shields.io/badge/rust-1.83%2B-blue.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/nagarx/feature-extractor-MBO-LOB/workflows/CI/badge.svg)](https://github.com/nagarx/feature-extractor-MBO-LOB/actions)

## Overview

This library provides a modular, research-aligned feature extraction pipeline for LOB data. It works in conjunction with [MBO-LOB-reconstructor](https://github.com/nagarx/MBO-LOB-reconstructor) to transform raw Market-by-Order data into ML-ready features.

### Key Features

- Fluent Builder API: Simple, readable pipeline configuration
- Paper-Aligned Presets: DeepLOB, TLOB, FI-2010, TransLOB, LiT configurations
- Auto-Computed Feature Count: No manual calculation required
- Comprehensive Feature Set: 200+ features across multiple categories
- Label Generation: TLOB and DeepLOB labeling methods for supervised learning
- Multiple Normalization Strategies: Z-score, Rolling Z-score, Global Z-score, Bilinear
- Multi-Scale Sequences: Fast/Medium/Slow temporal resolution
- High Performance: Single-pass computation, pre-allocated buffers

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
```

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Raw LOB | 40 | Prices and volumes at 10 levels |
| Derived | 8 | Mid-price, spread, imbalance, etc. |
| MBO Features | 36 | Order flow dynamics |
| Order Flow | 8 | OFI, queue imbalance, trade flow |
| Multi-Level OFI | 10 | OFI at each LOB level |
| FI-2010 Time-Insensitive | 20 | Spread, mid-price, price diffs |
| FI-2010 Time-Sensitive | 20 | Derivatives, intensity measures |
| FI-2010 Depth | 40 | Accumulated volumes and diffs |
| Market Impact | 8 | Slippage, execution quality |

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

## Export to NumPy

### Single Day Export

```rust
let exporter = NumpyExporter::new("output/");
exporter.export_day("2025-02-03", &output)?;
```

### Batch Export with Labels

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

## Performance

The library is optimized for HFT environments:

- Single-pass computation: Features extracted in one LOB traversal
- Pre-allocated buffers: No allocations in hot paths
- Welford's algorithm: Numerically stable running statistics
- Zero-copy normalization: In-place operations where possible

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
- Price Impact: The Price Impact of Order Book Events (Cont et al.)
- Queue Imbalance: Queue Imbalance as a One-Tick-Ahead Price Predictor

## License

Proprietary - All Rights Reserved. See [LICENSE](LICENSE) for details.

No permission is granted to use, copy, modify, or distribute this software.
