# Feature Extractor Architecture

## Design Philosophy

Based on analysis of 20+ research papers (DeepLOB, TLOB, FI-2010, TransLOB, LiT, HLOB, LOBench, ViT-LOB, etc.), this library follows these principles:

1. **Feature Quality > Model Complexity**: Research consistently shows feature engineering matters more than architecture
2. **Paper-Aligned Presets**: Easy reproduction of published results
3. **Separation of Concerns**: Raw features -> Derived features -> Normalization -> Sequences
4. **Modular & Composable**: Each component works independently
5. **Versioned Schema**: Track feature definitions for reproducibility
6. **Speed & Accuracy**: Optimized for HFT environments with nanosecond precision
7. **Streaming First**: Process data incrementally to avoid memory issues

## Module Structure

```
feature_extractor/
├── src/
│   ├── lib.rs                    # Public API and re-exports
│   ├── prelude.rs                # Convenience re-exports for common use
│   │
│   ├── builder.rs                # PipelineBuilder fluent API
│   ├── config.rs                 # Pipeline configuration
│   ├── pipeline.rs               # High-level pipeline orchestrator
│   ├── batch.rs                  # Parallel batch processing (feature: parallel)
│   │
│   ├── schema/                   # Feature schema definitions
│   │   ├── mod.rs               # Schema registry and version
│   │   ├── feature_def.rs       # FeatureDef, FeatureCategory, FeatureSchema
│   │   └── presets.rs           # Paper-aligned presets (DeepLOB, TLOB, FI-2010, etc.)
│   │
│   ├── features/                 # Feature extraction (raw computation only)
│   │   ├── mod.rs               # FeatureExtractor, extract_into(), extract_arc()
│   │   ├── lob_features.rs      # Raw LOB features (prices, volumes) - 40 features
│   │   ├── derived_features.rs  # Derived analytics (spread, microprice, etc.) - 8 features
│   │   ├── mbo_features.rs      # MBO-specific features - 36 features
│   │   ├── signals.rs           # Trading signals (OFI, TimeRegime, etc.) - 14 features
│   │   ├── order_flow.rs        # OFI, queue imbalance, trade flow
│   │   ├── fi2010.rs            # FI-2010 handcrafted features (80)
│   │   └── market_impact.rs     # Market impact estimation (slippage, VWAP)
│   │
│   ├── labeling/                 # Label generation for supervised learning
│   │   ├── mod.rs               # TrendLabel, LabelConfig, LabelStats
│   │   ├── tlob.rs              # TLOB labeling (decoupled h/k)
│   │   ├── deeplob.rs           # DeepLOB labeling (k=h)
│   │   └── multi_horizon.rs     # Multi-horizon labels (FI-2010, DeepLOB presets)
│   │
│   ├── preprocessing/            # Normalization and sampling
│   │   ├── mod.rs               # Module exports
│   │   ├── normalization.rs     # All normalizers (Z-score, Bilinear, Global, Rolling, etc.)
│   │   ├── sampling.rs          # Volume/Event-based sampling
│   │   ├── adaptive_sampling.rs # Volatility-adaptive thresholds
│   │   └── volatility.rs        # Realized volatility estimation (Welford)
│   │
│   ├── sequence_builder/         # Sequence building for transformers
│   │   ├── mod.rs               # Module exports, FeatureVec type alias
│   │   ├── builder.rs           # SequenceBuilder, Sequence (Arc-based), push_arc()
│   │   ├── horizon_aware.rs     # Horizon-aware windowing
│   │   └── multiscale.rs        # MultiScaleWindow, push_arc()
│   │
│   ├── validation.rs             # Feature validation (crossed quotes, NaN checks)
│   ├── export/                   # Export module
│   │   ├── mod.rs               # NumpyExporter, BatchExporter, DatasetConfig re-exports
│   │   ├── tensor_format.rs     # TensorFormatter, TensorFormat, TensorOutput
│   │   └── dataset_config.rs    # DatasetConfig, SymbolConfig, FeatureSetConfig, etc.
│   └── export_aligned.rs         # Aligned batch export
│
├── tools/                        # CLI tools
│   └── export_dataset.rs        # Configuration-driven export CLI
│
├── configs/                      # Sample configuration files
│   ├── nvda_98feat.toml         # 98-feature NVIDIA export config
│   ├── nvda_84feat_baseline.toml # 84-feature baseline config
│   └── template_multi_symbol.toml # Multi-symbol template
│
├── benches/
│   └── feature_extraction.rs     # Criterion benchmarks
│
├── examples/                      # Usage examples
│   ├── builder_validation.rs     # PipelineBuilder with real data
│   ├── full_pipeline.rs          # Complete pipeline example
│   └── ...
│
├── tests/                         # Integration tests
│   ├── comprehensive_validation.rs
│   ├── pipeline_tests.rs
│   ├── parallel_processing_tests.rs  # Batch processing tests
│   └── ...
│
└── docs/
    └── USAGE_GUIDE.md            # Comprehensive usage documentation
```

## Key Components

### 1. PipelineBuilder (Recommended Entry Point)

Fluent API for configuring and building pipelines:

```rust
use feature_extractor::prelude::*;

let pipeline = PipelineBuilder::new()
    .lob_levels(10)           // 10 levels -> 40 raw features
    .with_derived_features()  // +8 derived features
    .window(100, 10)          // 100 snapshots, stride 10
    .event_sampling(1000)     // Sample every 1000 events
    .build()?;
```

### 2. Feature Count Auto-Computation

Feature count is automatically computed:

| Configuration | Feature Count | Formula |
|--------------|---------------|---------|
| Raw LOB only | 40 | 10 levels × 4 |
| + Derived | 48 | 40 + 8 |
| + MBO | 76 | 40 + 36 |
| + Derived + MBO | 84 | 40 + 8 + 36 |
| + Derived + MBO + Signals | 98 | 40 + 8 + 36 + 14 |

### 3. Streaming Sequence Generation

The pipeline uses streaming mode to avoid buffer eviction:

```
Messages -> LOB Reconstruction -> Feature Extraction -> Sequence Building
                                                              ↓
                                              try_build_sequence() after each push
                                                              ↓
                                              Accumulated sequences (no data loss)
```

### 4. Prelude Module

Single import for all common types:

```rust
use feature_extractor::prelude::*;

// Now you have access to:
// - Pipeline, PipelineBuilder, PipelineConfig, PipelineOutput
// - FeatureExtractor, FeatureConfig
// - SequenceBuilder, Sequence, SequenceConfig, FeatureVec
// - LabelGenerator, LabelConfig, TrendLabel
// - MultiHorizonLabelGenerator, MultiHorizonConfig, ThresholdStrategy
// - TensorFormatter, TensorFormat, TensorOutput, FeatureMapping
// - All normalizers
// - All mbo-lob-reconstructor types (LobReconstructor, DbnLoader, etc.)
// - Trading signals: OfiComputer, OfiSample, TimeRegime, compute_signals
```

## Feature Categories

| Category | Count | Indices | Source | Description |
|----------|-------|---------|--------|-------------|
| Raw LOB | 40 | 0-39 | All papers | (P_ask, V_ask, P_bid, V_bid) × 10 levels |
| Derived | 8 | 40-47 | TLOB, DeepLOB | Microprice, spread, imbalance |
| MBO Features | 36 | 48-83 | MBO Paper | Order lifecycle patterns |
| Trading Signals | 14 | 84-97 | Cont et al., Stoikov | OFI, microprice, time regime, safety gates |
| **Total (full)** | **98** | 0-97 | - | All features enabled |

### Additional Feature Sets (standalone modules)

| Category | Count | Source | Description |
|----------|-------|--------|-------------|
| Order Flow | 8 | Cont et al. | OFI, queue imbalance, trade flow |
| Multi-Level OFI | 10 | LOB-feature-analysis | OFI at each LOB level |
| FI-2010 | 80 | FI-2010 benchmark | Handcrafted features |
| Market Impact | 8 | OrderBook-rs | Slippage, VWAP |

## Normalization Strategies

| Strategy | Source | Use Case |
|----------|--------|----------|
| Z-Score | DeepLOB, FI-2010 | Standard ML preprocessing |
| Rolling Z-Score | LOBFrame | Non-stationary data |
| Global Z-Score | LOBench | Preserve LOB constraints |
| Bilinear | TLOB | LOB structure preservation |
| Percentage Change | HLOB | Cross-instrument training |

## Paper-Aligned Presets

```rust
pub enum Preset {
    DeepLOB,   // 40 raw + Z-score, seq_len=100
    TLOB,      // 40 raw + bilinear normalization
    FI2010,    // 120 features (40 raw + 80 handcrafted)
    TransLOB,  // 40 raw + multi-horizon
    LiT,       // 80 features (20 levels × 4)
    Minimal,   // 40 raw LOB only
    Full,      // All available features (84)
}
```

## Performance Characteristics

- **Single-pass computation**: Features extracted in one LOB traversal
- **Streaming sequences**: No buffer eviction, 100% sequence efficiency
- **Pre-allocated buffers**: No allocations in hot paths (extract_into, push_arc)
- **Welford's algorithm**: Numerically stable running statistics
- **O(1) updates**: Constant time for rolling statistics
- **Arc-based sharing**: Zero-copy feature vectors across sequences
- **Parallel processing**: Multi-threaded batch processing with Rayon
- **Graceful cancellation**: CancellationToken for long-running jobs

### Memory Optimizations

| Optimization | Impact |
|-------------|--------|
| `extract_into()` buffer reuse | 0 allocations per sample |
| `FeatureVec = Arc<Vec<f64>>` | 8-byte clone vs 672-byte clone |
| Multi-scale Arc sharing | 98.8% memory reduction |
| `PriceLevel` O(1) cache | Eliminates O(n) sum per level |

## Test Coverage

- **300+ unit tests** covering all modules
- **100+ integration tests** with real NVIDIA MBO data
- **50+ doc tests** with working examples
- **Comprehensive validation**: 151M+ messages, 21 days data
- **99.42% price accuracy** against MBP-10 ground truth

## Dependencies

- `mbo-lob-reconstructor`: LOB reconstruction from MBO data
- `serde`: Serialization for configuration
- `ndarray-npy`: NumPy export
- `criterion` (dev): Benchmarking

## Roadmap

### Completed
- [x] Raw LOB features (40)
- [x] Order Flow features (OFI, MLOFI, queue imbalance)
- [x] FI-2010 features (80)
- [x] Market Impact estimation
- [x] All normalization strategies
- [x] Multi-scale sequence building
- [x] Validation module
- [x] Benchmark suite
- [x] Labeling module (TLOB + DeepLOB methods)
- [x] Documentation (README, ARCHITECTURE, USAGE_GUIDE)
- [x] Standalone repository
- [x] CI workflow (GitHub Actions)
- [x] PipelineBuilder fluent API
- [x] Prelude module
- [x] Feature count auto-computation
- [x] Streaming sequence generation fix
- [x] Comprehensive real-data validation (151M+ messages, 21 days)
- [x] **Zero-allocation APIs** (extract_into, push_arc)
- [x] **Arc-based sequence storage** (FeatureVec = Arc<Vec<f64>>)
- [x] **Parallel batch processing** (BatchProcessor, Rayon)
- [x] **Graceful cancellation** (CancellationToken)
- [x] **PriceLevel O(1) caching** (in mbo-lob-reconstructor)
- [x] **Multi-Horizon Label Generator** (FI-2010, DeepLOB, TLOB presets)
- [x] **TensorFormatter** (DeepLOB, HLOB, Flat, Image formats)
- [x] **Pipeline Integration** (format_sequences, generate_multi_horizon_labels)
- [x] **Hot Store Integration** (decompressed file caching, ~30% faster)
- [x] **Trading Signals Module** (14 signals: OFI, microprice, time regime)
- [x] **OfiComputer** (streaming OFI with warmup tracking per Cont et al.)
- [x] **TimeRegime** (UTC→ET conversion with DST handling)
- [x] **Safety Gates** (book_valid, mbo_ready)
- [x] **Comprehensive Signal Validation** (1.7M+ samples, NVIDIA real data)
- [x] **DatasetConfig Export System** (symbol-agnostic, configuration-driven exports)
- [x] **Export CLI Tool** (`export_dataset` binary for TOML-driven exports)
- [x] **Sample Configurations** (NVDA 98-feat, 84-feat baseline, multi-symbol templates)

### Pending
- [ ] crates.io publication
- [ ] Additional paper presets (ViT-LOB)
- [ ] Weighted MLOFI (depth-aware OFI scalar per Xu et al.)
