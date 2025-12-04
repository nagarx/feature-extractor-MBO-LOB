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
│   │
│   ├── schema/                   # Feature schema definitions
│   │   ├── mod.rs               # Schema registry and version
│   │   ├── feature_def.rs       # FeatureDef, FeatureCategory, FeatureSchema
│   │   └── presets.rs           # Paper-aligned presets (DeepLOB, TLOB, FI-2010, etc.)
│   │
│   ├── features/                 # Feature extraction (raw computation only)
│   │   ├── mod.rs               # FeatureExtractor, FeatureConfig
│   │   ├── lob_features.rs      # Raw LOB features (prices, volumes)
│   │   ├── derived_features.rs  # Derived analytics (spread, microprice, etc.)
│   │   ├── order_flow.rs        # OFI, queue imbalance, trade flow
│   │   ├── fi2010.rs            # FI-2010 handcrafted features (80)
│   │   ├── market_impact.rs     # Market impact estimation (slippage, VWAP)
│   │   └── mbo_features.rs      # MBO-specific features
│   │
│   ├── labeling/                 # Label generation for supervised learning
│   │   ├── mod.rs               # TrendLabel, LabelConfig, LabelStats
│   │   ├── tlob.rs              # TLOB labeling (decoupled h/k)
│   │   └── deeplob.rs           # DeepLOB labeling (k=h)
│   │
│   ├── preprocessing/            # Normalization and sampling
│   │   ├── mod.rs               # Module exports
│   │   ├── normalization.rs     # All normalizers (Z-score, Bilinear, Global, Rolling, etc.)
│   │   ├── sampling.rs          # Volume/Event-based sampling
│   │   ├── adaptive_sampling.rs # Volatility-adaptive thresholds
│   │   └── volatility.rs        # Realized volatility estimation (Welford)
│   │
│   ├── sequence_builder/         # Sequence building for transformers
│   │   ├── mod.rs               # Module exports
│   │   ├── builder.rs           # Core sequence builder
│   │   ├── horizon_aware.rs     # Horizon-aware windowing
│   │   └── multiscale.rs        # Multi-scale sequences (fast/medium/slow)
│   │
│   ├── validation.rs             # Feature validation (crossed quotes, NaN checks)
│   ├── export.rs                 # NumPy export
│   └── export_aligned.rs         # Aligned batch export
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
| + Both | 84 | 40 + 8 + 36 |

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
// - Pipeline, PipelineBuilder, PipelineConfig
// - FeatureExtractor, FeatureConfig
// - SequenceBuilder, Sequence, SequenceConfig
// - LabelGenerator, LabelConfig, TrendLabel
// - All normalizers
// - All mbo-lob-reconstructor types (LobReconstructor, DbnLoader, etc.)
```

## Feature Categories

| Category | Count | Source | Description |
|----------|-------|--------|-------------|
| Raw LOB | 40 | All papers | (P_ask, V_ask, P_bid, V_bid) × 10 levels |
| Derived | 8 | TLOB, DeepLOB | Microprice, spread, imbalance |
| Order Flow | 8 | Cont et al. | OFI, queue imbalance, trade flow |
| Multi-Level OFI | 10 | LOB-feature-analysis | OFI at each LOB level |
| FI-2010 | 80 | FI-2010 benchmark | Handcrafted features |
| Market Impact | 8 | OrderBook-rs | Slippage, VWAP |
| MBO Features | 36 | MBO Paper | Order lifecycle patterns |

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
- **Pre-allocated buffers**: No allocations in hot paths
- **Welford's algorithm**: Numerically stable running statistics
- **O(1) updates**: Constant time for rolling statistics

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

### Pending
- [ ] crates.io publication
- [ ] Statistical validation tests (OFI vs ΔP correlation)
- [ ] Additional paper presets (HLOB, ViT-LOB)
