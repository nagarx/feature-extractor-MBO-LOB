# Feature Extractor for MBO-LOB Data

High-performance feature extraction library for Limit Order Book (LOB) and Market-by-Order (MBO) data, designed for deep learning model training in high-frequency trading.

[![Rust](https://img.shields.io/badge/rust-1.83%2B-blue.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/nagarx/feature-extractor-MBO-LOB/workflows/CI/badge.svg)](https://github.com/nagarx/feature-extractor-MBO-LOB/actions)

## Overview

This library provides a modular, research-aligned feature extraction pipeline for LOB data. It works in conjunction with [MBO-LOB-reconstructor](https://github.com/nagarx/MBO-LOB-reconstructor) to transform raw Market-by-Order data into ML-ready features.

### Key Features

- Paper-Aligned Presets: DeepLOB, TLOB, FI-2010, TransLOB, LiT configurations
- Comprehensive Feature Set: 200+ features across multiple categories
- Label Generation: TLOB and DeepLOB labeling methods for supervised learning
- Multiple Normalization Strategies: Z-score, Rolling Z-score, Global Z-score (LOBench), Bilinear
- Multi-Scale Sequences: Fast/Medium/Slow temporal resolution
- Validation Module: Crossed quote detection, feature range checks
- High Performance: Single-pass computation, pre-allocated buffers

## Quick Start

```rust
use feature_extractor::{
    FeatureExtractor, FeatureConfig, FI2010Extractor, FI2010Config,
    GlobalZScoreNormalizer, FeatureValidator,
};
use mbo_lob_reconstructor::LobState;

// Basic LOB feature extraction
let extractor = FeatureExtractor::new(10); // 10 LOB levels
let features = extractor.extract_lob_features(&lob_state)?;

// FI-2010 handcrafted features
let mut fi2010 = FI2010Extractor::new(FI2010Config::default());
let handcrafted = fi2010.extract(&lob_state, timestamp)?;

// Global Z-score normalization (LOBench paper)
let normalizer = GlobalZScoreNormalizer::new();
let normalized = normalizer.normalize_snapshot(&features);

// Validate LOB data quality
let validator = FeatureValidator::new();
let result = validator.validate_lob(&lob_state);
if result.has_errors() {
    eprintln!("Validation errors: {:?}", result.errors());
}
```

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Raw LOB | 40 | Prices and volumes at 10 levels |
| Order Flow | 8 | OFI, queue imbalance, trade flow |
| Multi-Level OFI | 10 | OFI at each LOB level |
| FI-2010 Time-Insensitive | 20 | Spread, mid-price, price diffs |
| FI-2010 Time-Sensitive | 20 | Derivatives, intensity measures |
| FI-2010 Depth | 40 | Accumulated volumes and diffs |
| Derived | 8 | Microprice, VWAP, imbalance |
| Market Impact | 8 | Slippage, execution quality |
| MBO Features | 36 | Order lifecycle patterns |

## Paper-Aligned Presets

```rust
use feature_extractor::schema::{Preset, FeatureSchema};

// DeepLOB: 40 features, Z-score normalization
let schema = FeatureSchema::from_preset(Preset::DeepLOB);

// FI-2010: 120 features (40 raw + 80 handcrafted)
let schema = FeatureSchema::from_preset(Preset::FI2010);

// Full: All available features
let schema = FeatureSchema::from_preset(Preset::Full);
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
use feature_extractor::preprocessing::{
    ZScoreNormalizer, RollingZScoreNormalizer, GlobalZScoreNormalizer,
    BilinearNormalizer, Normalizer,
};

// Standard Z-score
let mut zscore = ZScoreNormalizer::new();
zscore.update(100.0);
let normalized = zscore.normalize(101.0);

// Rolling Z-score (multi-day)
let mut rolling = RollingZScoreNormalizer::new(5); // 5-day window
rolling.add_day_stats(1000, 100.0, 10.0); // n, mean, std
let normalized = rolling.normalize(105.0);

// Global Z-score (LOBench)
let global = GlobalZScoreNormalizer::new();
let normalized = global.normalize_snapshot(&features);
```

## Order Flow Features

```rust
use feature_extractor::features::order_flow::{OrderFlowTracker, MultiLevelOfiTracker};

// Basic order flow
let mut tracker = OrderFlowTracker::new();
tracker.update(&lob_state);
let features = tracker.features();
// Returns: OFI, queue imbalance, trade imbalance, depth imbalance, arrival rates

// Multi-level OFI
let mut mlofi = MultiLevelOfiTracker::new(10);
mlofi.update(&lob_state);
let ofi_by_level = mlofi.ofi_by_level();
```

## FI-2010 Features

```rust
use feature_extractor::features::fi2010::{FI2010Extractor, FI2010Config};

let config = FI2010Config {
    levels: 10,
    include_time_insensitive: true,
    include_time_sensitive: true,
    include_depth: true,
    time_delta_ns: 1_000_000, // 1ms
};

let mut extractor = FI2010Extractor::new(config);

// Extract features (80 total)
let features = extractor.extract(&lob_state, timestamp)?;
```

## Validation

```rust
use feature_extractor::validation::{FeatureValidator, ValidationConfig};

let config = ValidationConfig {
    max_spread_bps: 1000.0,
    check_crossed_quotes: true,
    check_locked_quotes: true,
    check_price_ordering: true,
    ..Default::default()
};

let validator = FeatureValidator::with_config(config);
let result = validator.validate_lob(&lob_state);

if !result.is_valid() {
    for warning in result.warnings() {
        println!("Warning: {}", warning);
    }
    for error in result.errors() {
        println!("Error: {}", error);
    }
}
```

## Multi-Scale Sequences

```rust
use feature_extractor::sequence_builder::{
    MultiScaleWindow, MultiScaleConfig, ScaleConfig,
};

let config = MultiScaleConfig::default();
let mut window = MultiScaleWindow::new(config, 40); // 40 features

// Push events
for (ts, features) in events {
    window.push(ts, features);
}

// Get multi-scale sequences
if let Some(multiscale) = window.try_build_all() {
    let fast = multiscale.fast();   // High resolution
    let medium = multiscale.medium(); // Medium resolution
    let slow = multiscale.slow();   // Low resolution (context)
}
```

## Label Generation

The library includes two labeling strategies from research papers:

### TLOB Method (Recommended)

Decouples the prediction horizon from the smoothing window:

```rust
use feature_extractor::{LabelConfig, TlobLabelGenerator, TrendLabel};

// Configure labeling parameters
let config = LabelConfig {
    horizon: 10,           // Predict 10 steps ahead
    smoothing_window: 5,   // Average 5 prices for smoothing
    threshold: 0.002,      // 0.2% threshold for Up/Down
};

let mut generator = TlobLabelGenerator::new(config);

// Add mid-prices
generator.add_prices(&mid_prices);

// Generate labels
let labels = generator.generate_labels()?;
for (idx, label, pct_change) in &labels {
    match label {
        TrendLabel::Up => println!("t={}: Up ({:.2}%)", idx, pct_change * 100.0),
        TrendLabel::Down => println!("t={}: Down ({:.2}%)", idx, pct_change * 100.0),
        TrendLabel::Stable => println!("t={}: Stable", idx),
    }
}

// Get statistics
let stats = generator.compute_stats(&labels);
println!("Class balance: Up={:.1}%, Stable={:.1}%, Down={:.1}%",
    stats.up_count as f64 / stats.total as f64 * 100.0,
    stats.stable_count as f64 / stats.total as f64 * 100.0,
    stats.down_count as f64 / stats.total as f64 * 100.0,
);
```

### DeepLOB Method

Simpler method where smoothing window equals horizon (k = h):

```rust
use feature_extractor::{LabelConfig, DeepLobLabelGenerator, DeepLobMethod};

// FI-2010 style configuration (k = h)
let config = LabelConfig::fi2010(50);

// Method 1: Compare to current price
let mut gen = DeepLobLabelGenerator::new(config.clone());

// Method 2: Compare to past average (like TLOB)
let mut gen = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);
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

# Run doc tests
cargo test --doc

# Run with verbose output
cargo test -- --nocapture
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

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
