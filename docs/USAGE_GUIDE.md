# Usage Guide: MBO-LOB Feature Extraction Pipeline

This guide explains how to use the `MBO-LOB-reconstructor` and `feature-extractor-MBO-LOB` libraries together to preprocess Market-by-Order data for deep learning models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Pipeline Builder API](#pipeline-builder-api)
4. [Configuration Options](#configuration-options)
5. [Feature Sets](#feature-sets)
6. [Sampling Strategies](#sampling-strategies)
7. [Label Generation](#label-generation)
8. [Export to NumPy](#export-to-numpy)
9. [Advanced Usage](#advanced-usage)
10. [Performance Optimization](#performance-optimization)

## Installation

Add both libraries to your `Cargo.toml`:

```toml
[dependencies]
mbo-lob-reconstructor = { git = "https://github.com/nagarx/MBO-LOB-reconstructor.git", features = ["databento"] }
feature-extractor = { git = "https://github.com/nagarx/feature-extractor-MBO-LOB.git", features = ["databento"] }
```

## Quick Start

The simplest way to use the libraries is through the `prelude` and `PipelineBuilder`:

```rust
use feature_extractor::prelude::*;

fn main() -> Result<()> {
    // Build pipeline with fluent API
    let mut pipeline = PipelineBuilder::new()
        .lob_levels(10)           // 10 price levels (40 raw features)
        .window(100, 10)          // 100 snapshots per sequence, stride 10
        .event_sampling(1000)     // Sample every 1000 events
        .build()?;

    // Process MBO data
    let output = pipeline.process("data/SYMBOL.mbo.dbn.zst")?;

    println!("Processed {} messages", output.messages_processed);
    println!("Generated {} sequences", output.sequences_generated);

    // Export to NumPy for Python/PyTorch
    let exporter = NumpyExporter::new("output/");
    exporter.export_day("2025-02-03", &output)?;

    Ok(())
}
```

## Pipeline Builder API

The `PipelineBuilder` provides a fluent API for configuring the entire pipeline:

### Basic Configuration

```rust
let pipeline = PipelineBuilder::new()
    .lob_levels(10)              // Number of LOB levels (default: 10)
    .tick_size(0.01)             // Tick size for US stocks
    .window(100, 10)             // Window size and stride
    .build()?;
```

### Feature Selection

```rust
// Raw LOB only (40 features)
let pipeline = PipelineBuilder::new().build()?;

// With derived features (48 features = 40 + 8)
let pipeline = PipelineBuilder::new()
    .with_derived_features()
    .build()?;

// With MBO features (76 features = 40 + 36)
let pipeline = PipelineBuilder::new()
    .with_mbo_features()
    .build()?;

// Full feature set (84 features = 40 + 8 + 36)
let pipeline = PipelineBuilder::new()
    .with_derived_features()
    .with_mbo_features()
    .build()?;
```

### Sampling Configuration

```rust
// Volume-based sampling (sample after N shares traded)
let pipeline = PipelineBuilder::new()
    .volume_sampling(1000)       // Every 1000 shares
    .build()?;

// Event-based sampling (sample after N MBO events)
let pipeline = PipelineBuilder::new()
    .event_sampling(500)         // Every 500 events
    .build()?;

// With minimum time interval
let pipeline = PipelineBuilder::new()
    .event_sampling(1000)
    .min_sample_interval_ns(1_000_000)  // At least 1ms between samples
    .build()?;
```

### Research Paper Presets

```rust
// DeepLOB configuration
let pipeline = PipelineBuilder::from_preset(Preset::DeepLOB)
    .volume_sampling(1000)
    .build()?;

// TLOB configuration
let pipeline = PipelineBuilder::from_preset(Preset::TLOB)
    .event_sampling(1000)
    .build()?;

// Full feature set
let pipeline = PipelineBuilder::from_preset(Preset::Full)
    .build()?;
```

### Experiment Metadata

```rust
let pipeline = PipelineBuilder::new()
    .experiment("nvda_baseline_v1", "Initial NVDA experiment with default settings")
    .build()?;
```

## Configuration Options

### FeatureConfig

Controls which features are extracted:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lob_levels` | `usize` | 10 | Number of LOB price levels |
| `tick_size` | `f64` | 0.01 | Minimum price increment |
| `include_derived` | `bool` | false | Include 8 derived features |
| `include_mbo` | `bool` | false | Include 36 MBO features |
| `mbo_window_size` | `usize` | 1000 | MBO aggregation window |

### SequenceConfig

Controls sequence generation:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | `usize` | 100 | Snapshots per sequence |
| `stride` | `usize` | 1 | Step between sequences |
| `max_buffer_size` | `usize` | 1000 | Maximum buffer capacity |
| `feature_count` | `usize` | auto | Auto-computed from FeatureConfig |

### SamplingConfig

Controls when to sample LOB snapshots:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `VolumeBased` | Sample after N shares traded | Activity-weighted sampling |
| `EventBased` | Sample after N MBO events | Uniform event-time sampling |
| `TimeBased` | Sample at fixed intervals | Calendar-time analysis |

## Feature Sets

### Raw LOB Features (40)

For each of 10 levels:
- Ask price, Ask size, Bid price, Bid size

### Derived Features (8)

- Mid-price
- Spread (absolute)
- Spread (basis points)
- Total bid volume
- Total ask volume
- Volume imbalance
- Weighted mid-price
- Price impact estimate

### MBO Features (36)

Order flow dynamics across fast/medium/slow windows:
- Order arrival rates
- Trade intensity
- Order flow imbalance
- Cancellation rates
- Fill ratios
- Queue position metrics

## Sampling Strategies

### Volume-Based Sampling

Best for capturing market activity:

```rust
let pipeline = PipelineBuilder::new()
    .volume_sampling(1000)  // Sample every 1000 shares
    .build()?;
```

### Event-Based Sampling

Best for uniform coverage in event time:

```rust
let pipeline = PipelineBuilder::new()
    .event_sampling(500)    // Sample every 500 events
    .build()?;
```

### Adaptive Sampling

Adjusts threshold based on market volatility:

```rust
let pipeline = PipelineBuilder::new()
    .volume_sampling(1000)
    .with_adaptive_sampling()  // Enable volatility-based adjustment
    .build()?;
```

### Multi-Scale Windowing

Maintains three temporal resolutions:

```rust
let pipeline = PipelineBuilder::new()
    .event_sampling(1000)
    .with_multiscale()  // Fast/Medium/Slow windows
    .build()?;
```

## Label Generation

### TLOB Method

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

println!("Up: {:.1}%", stats.up_count as f64 / stats.total as f64 * 100.0);
println!("Stable: {:.1}%", stats.stable_count as f64 / stats.total as f64 * 100.0);
println!("Down: {:.1}%", stats.down_count as f64 / stats.total as f64 * 100.0);
```

### DeepLOB Method

```rust
let config = LabelConfig::fi2010(50);  // k = h = 50
let mut generator = DeepLobLabelGenerator::new(config);
```

### Configuration Presets

```rust
// HFT (ultra short-term)
let config = LabelConfig::hft();  // h=10, k=5, threshold=0.02%

// Short-term trading
let config = LabelConfig::short_term();  // h=50, k=10, threshold=0.2%

// Medium-term trading
let config = LabelConfig::medium_term();  // h=100, k=20, threshold=0.5%
```

## Export to NumPy

### Single Day Export

```rust
let exporter = NumpyExporter::new("output/");
exporter.export_day("2025-02-03", &output)?;

// Creates:
// - output/2025-02-03_features.npy
// - output/2025-02-03_labels.npy (if labels generated)
// - output/2025-02-03_metadata.json
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

### Aligned Export

Ensures features and labels are properly aligned:

```rust
let exporter = AlignedBatchExporter::new("output/", label_config);
let result = exporter.export_day("2025-02-03", &output)?;

println!("Exported {} sequences with {} labels", result.n_sequences, result.n_labels);
```

## Advanced Usage

### Manual Component Control

For fine-grained control, use individual components:

```rust
use feature_extractor::prelude::*;

// Initialize components
let lob_config = LobConfig::new(10)
    .with_crossed_quote_policy(CrossedQuotePolicy::UseLastValid);
let mut reconstructor = LobReconstructor::with_config(lob_config);

let feature_config = FeatureConfig::default().with_derived(true);
let extractor = FeatureExtractor::with_config(feature_config.clone());

let seq_config = SequenceConfig::from_feature_config(100, 10, &feature_config);
let mut sequence_builder = SequenceBuilder::with_config(seq_config);

// Process messages - use streaming mode to avoid buffer eviction
let loader = DbnLoader::new("data/SYMBOL.mbo.dbn.zst")?;
let mut sequences = Vec::new();

for msg in loader.iter_messages()? {
    // Skip system messages (order_id=0, etc.)
    // Note: LobReconstructor does this by default if skip_system_messages=true
    
    let lob_state = reconstructor.process_message(&msg)?;
    
    // Skip if LOB not ready
    if lob_state.mid_price().is_none() {
        continue;
    }
    
    let features = extractor.extract_lob_features(&lob_state)?;
    sequence_builder.push(msg.timestamp.unwrap_or(0) as u64, features)?;
    
    // IMPORTANT: Build sequences incrementally (streaming mode)
    // This prevents data loss from buffer eviction
    if let Some(seq) = sequence_builder.try_build_sequence() {
        sequences.push(seq);
    }
}

println!("Generated {} sequences", sequences.len());
```

> **Important**: Always use `try_build_sequence()` after each push for streaming mode.
> Using `generate_all_sequences()` at the end will only return sequences from the
> buffer's current contents (default 1000 snapshots), losing earlier data.

### Multi-Day Processing

```rust
let mut pipeline = PipelineBuilder::new()
    .with_derived_features()
    .event_sampling(1000)
    .build()?;

for day in trading_days {
    pipeline.reset();  // Critical: Reset state between days
    
    let path = format!("data/{}.dbn.zst", day);
    let output = pipeline.process(&path)?;
    
    // Process output...
}
```

### Validation

```rust
let validator = FeatureValidator::new();

for seq in &output.sequences {
    for features in &seq.features {
        let result = validator.validate_features(features);
        if !result.is_valid() {
            eprintln!("Invalid features: {:?}", result.errors());
        }
    }
}
```

## Performance Optimization

### Release Mode

Always use release mode for production:

```bash
cargo run --release --example your_pipeline
```

### Buffer Sizing

Adjust buffer size based on available memory:

```rust
let pipeline = PipelineBuilder::new()
    .window(100, 10)
    .max_buffer(5000)  // Larger buffer for batch processing
    .build()?;
```

### Streaming Processing

For very large datasets, process in streaming fashion:

```rust
let mut pipeline = PipelineBuilder::new().build()?;

// Process produces sequences incrementally
let output = pipeline.process("large_file.dbn.zst")?;

// Sequences are available as they're generated
for seq in output.sequences {
    // Process each sequence immediately
}
```

### Memory Efficiency

Reset pipeline between days to free memory:

```rust
for day in days {
    pipeline.reset();  // Clears all internal buffers
    let output = pipeline.process(&path)?;
    // Export immediately to free memory
    exporter.export_day(&day, &output)?;
}
```

## Example: Complete Pipeline

```rust
use feature_extractor::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Configure pipeline
    let mut pipeline = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()
        .window(100, 10)
        .event_sampling(1000)
        .experiment("nvda_experiment", "NVDA feature extraction")
        .build()?;

    // Configure labeling
    let label_config = LabelConfig::short_term();
    let exporter = BatchExporter::new("output/", Some(label_config));

    // Process multiple days
    let days = vec!["2025-02-03", "2025-02-04", "2025-02-05"];
    
    for day in days {
        let start = Instant::now();
        
        // Reset pipeline state
        pipeline.reset();
        
        // Process data
        let path = format!("data/xnas-itch-{}.mbo.dbn.zst", day.replace("-", ""));
        let output = pipeline.process(&path)?;
        
        // Export
        let result = exporter.export_day(day, &output)?;
        
        println!("{}: {} sequences, {} labels in {:.1}s",
            day,
            result.n_sequences,
            result.n_labels,
            start.elapsed().as_secs_f64()
        );
    }

    println!("Pipeline complete!");
    Ok(())
}
```

## Troubleshooting

### Feature Count Mismatch

If you see a feature count mismatch error, ensure your sequence config matches your feature config:

```rust
// Wrong: Manual feature count
let seq_config = SequenceConfig::new(100, 10)
    .with_feature_count(40);  // May not match!

// Right: Auto-computed feature count
let seq_config = SequenceConfig::from_feature_config(100, 10, &feature_config);

// Or use PipelineBuilder which handles this automatically
let pipeline = PipelineBuilder::new()
    .with_derived_features()  // Feature count auto-synced
    .build()?;
```

### No Sequences Generated

If no sequences are generated:

1. Check sampling threshold (may be too high)
2. Ensure enough data for window size
3. Verify data file exists and is valid

```rust
// Lower threshold for testing
let pipeline = PipelineBuilder::new()
    .volume_sampling(100)  // Lower threshold
    .window(50, 5)         // Smaller window
    .build()?;
```

### State Leakage Between Days

Always reset pipeline between processing different days:

```rust
for day in days {
    pipeline.reset();  // Critical!
    let output = pipeline.process(&path)?;
}
```

### Sequence Loss (Streaming vs Batch)

If you see far fewer sequences than expected, you may be using batch mode incorrectly:

```rust
// WRONG: generate_all_sequences() only returns sequences from current buffer
for msg in messages {
    sequence_builder.push(ts, features)?;
}
let sequences = sequence_builder.generate_all_sequences();  // May lose data!

// RIGHT: Use streaming mode (try_build_sequence after each push)
let mut sequences = Vec::new();
for msg in messages {
    sequence_builder.push(ts, features)?;
    if let Some(seq) = sequence_builder.try_build_sequence() {
        sequences.push(seq);  // Capture immediately
    }
}
```

The `PipelineBuilder` handles this automatically.

### System Messages

If you see "Invalid order ID: 0" errors, system messages aren't being filtered:

```rust
// LobReconstructor filters system messages by default
let config = LobConfig::new(10)
    .with_skip_system_messages(true);  // Default

// Check stats after processing
let stats = reconstructor.stats();
println!("System messages skipped: {}", stats.system_messages_skipped);
```

System messages (~6% of real data) have:
- `order_id = 0` (heartbeats, status)
- `size = 0` (invalid)
- `price <= 0` (invalid)

