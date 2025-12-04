# Codebase Technical Reference

This document provides technical context for developers and AI assistants working on this codebase.

## Repository Overview

This repository (`feature-extractor-MBO-LOB`) is the second stage of a two-library pipeline:

```
MBO Data (.dbn.zst) 
    -> [MBO-LOB-reconstructor] 
    -> LOB Snapshots 
    -> [feature-extractor-MBO-LOB] 
    -> ML-Ready Features/Sequences
    -> PyTorch/TensorFlow
```

## Related Repository

- **MBO-LOB-reconstructor**: https://github.com/nagarx/MBO-LOB-reconstructor
  - Converts Market-By-Order events to Limit Order Book snapshots
  - Provides `LobReconstructor`, `DbnLoader`, `LobState`, `MboMessage`
  - This library depends on it via git dependency

## Critical Implementation Details

### 1. Streaming Sequence Generation (Fixed in v0.1.1)

**Problem**: The `SequenceBuilder` has a bounded buffer (default 1000). If you push features and only call `generate_all_sequences()` at the end, most features are lost due to eviction.

**Solution**: The `Pipeline::process()` method now calls `try_build_sequence()` after each `push()`:

```rust
// In src/pipeline.rs
if let Err(e) = self.sequence_builder.push(ts, features) {
    log::error!("Sequence builder push failed: {}", e);
} else {
    // Streaming: try to build sequence after each push
    if let Some(seq) = self.sequence_builder.try_build_sequence() {
        accumulated_sequences.push(seq);
    }
}
```

**Impact**: Without this fix, you lose ~98.8% of sequences.

### 2. Feature Count Auto-Computation

Feature count is computed from `FeatureConfig`:

```rust
// In src/features/mod.rs
impl FeatureConfig {
    pub fn feature_count(&self) -> usize {
        let lob_features = self.lob_levels * 4;  // 4 per level
        let derived = if self.include_derived { 8 } else { 0 };
        let mbo = if self.include_mbo { 36 } else { 0 };
        lob_features + derived + mbo
    }
}
```

This is auto-synced to `SequenceConfig` via `PipelineConfig::sync_feature_count()`.

### 3. System Message Filtering

System messages (order_id=0, size=0, price<=0) are filtered by `LobReconstructor` by default:

```rust
// In mbo-lob-reconstructor/src/lob/reconstructor.rs
if self.config.skip_system_messages
    && (msg.order_id == 0 || msg.size == 0 || msg.price <= 0)
{
    self.stats.system_messages_skipped += 1;
    return Ok(self.get_lob_state());
}
```

This is ~6% of real NVIDIA data.

### 4. Pipeline Duplicate Filtering

The `Pipeline` also filters system messages (in case `skip_system_messages` is disabled):

```rust
// In src/pipeline.rs
if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
    continue;
}
```

### 5. Reset Methods

**LobReconstructor has two reset methods:**
- `reset()`: Clears book state but **preserves stats** (for Action::Clear messages)
- `full_reset()`: Clears everything including stats (for new day/fresh start)

**Pipeline reset:**
- `Pipeline::reset()`: Calls `sequence_builder.reset()` and clears internal state

## Key Files to Understand

| File | Purpose | Key Types |
|------|---------|-----------|
| `src/lib.rs` | Public API, re-exports | Module structure |
| `src/prelude.rs` | Convenience imports | Re-exports common types |
| `src/builder.rs` | Fluent API | `PipelineBuilder` |
| `src/pipeline.rs` | Orchestrator | `Pipeline`, `PipelineOutput` |
| `src/config.rs` | Configuration | `PipelineConfig`, `SamplingConfig` |
| `src/features/mod.rs` | Feature extraction | `FeatureExtractor`, `FeatureConfig` |
| `src/sequence_builder/builder.rs` | Sequence generation | `SequenceBuilder`, `Sequence` |
| `src/labeling/tlob.rs` | Label generation | `TlobLabelGenerator` |
| `src/export.rs` | NumPy export | `NumpyExporter`, `BatchExporter` |

## Common Patterns

### Creating a Pipeline

```rust
// Simple (recommended)
let pipeline = PipelineBuilder::new()
    .with_derived_features()
    .event_sampling(1000)
    .build()?;

// From preset
let pipeline = PipelineBuilder::from_preset(Preset::DeepLOB)
    .build()?;

// Manual (advanced)
let config = PipelineConfig::default()
    .with_features(FeatureConfig::default().with_derived(true));
let pipeline = Pipeline::from_config(config)?;
```

### Processing Data

```rust
// Single file
let output = pipeline.process("path/to/file.dbn.zst")?;

// Multiple days (reset between)
for day in days {
    pipeline.reset();  // Critical!
    let output = pipeline.process(&path)?;
    exporter.export_day(&day, &output)?;
}
```

### Label Generation

```rust
let config = LabelConfig::short_term();  // h=50, k=10, threshold=0.2%
let mut generator = TlobLabelGenerator::new(config);
generator.add_prices(&output.mid_prices);
let labels = generator.generate_labels()?;
```

## Testing Guidelines

### Running Tests

```bash
# All tests (some require data files)
cargo test --all-features

# Skip tests requiring data
cargo test --all-features --lib

# With output
cargo test -- --nocapture
```

### Test Data Handling

Integration tests check for data file existence:

```rust
fn require_test_data() -> bool {
    let path = PathBuf::from(DATA_PATH);
    if !path.exists() {
        println!("[SKIP] Data file not found");
        return false;
    }
    true
}
```

## Error Handling

The library uses `mbo_lob_reconstructor::Result<T>` which is `Result<T, TlobError>`.

Common error patterns:
- Feature count mismatch: Auto-compute via `FeatureConfig::feature_count()`
- No sequences generated: Lower sampling threshold or check window size
- Message validation: Configure `LobConfig::with_validation(false)` for testing

## Performance Considerations

1. **Always use release mode**: `cargo run --release`
2. **Buffer sizing**: `max_buffer` should be >= `window_size + stride`
3. **Streaming**: Sequences are built incrementally, no memory spike
4. **Reset between days**: Prevents state leakage and frees memory

## CI/CD

GitHub Actions workflow runs:
- `cargo fmt --check`
- `cargo clippy --all-features -- -D warnings`
- `cargo test --all-features`
- `cargo doc --no-deps --all-features`
- MSRV check (Rust 1.83)

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Feature count mismatch | Manual count wrong | Use `FeatureConfig::feature_count()` |
| No sequences | High threshold | Lower `volume_sampling()` or `event_sampling()` |
| State leakage | Missing reset | Call `pipeline.reset()` between days |
| CI doc failure | Unescaped brackets | Use `\[text\]` in doc comments |
| Invalid order ID | System message | Enable `skip_system_messages` (default) |

## Versioning

- **Rust Edition**: 2021
- **MSRV**: 1.83 (due to `pest` crate)
- **Semantic Versioning**: Yes

## File Locations

- **Data files**: Typically in `data/` directory (gitignored)
- **Output**: Typically in `output/` directory (gitignored)
- **Benchmarks**: `target/criterion/` (gitignored)

