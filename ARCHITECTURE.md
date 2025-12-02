# Feature Extractor Architecture

## Design Philosophy

Based on analysis of 20+ research papers (DeepLOB, TLOB, FI-2010, TransLOB, LiT, HLOB, LOBench, ViT-LOB, etc.), this library follows these principles:

1. **Feature Quality > Model Complexity**: Research consistently shows feature engineering matters more than architecture
2. **Paper-Aligned Presets**: Easy reproduction of published results
3. **Separation of Concerns**: Raw features → Derived features → Normalization → Sequences
4. **Modular & Composable**: Each component works independently
5. **Versioned Schema**: Track feature definitions for reproducibility
6. **Speed & Accuracy**: Optimized for HFT environments with nanosecond precision

## Current Module Structure

```
feature_extractor/
├── src/
│   ├── lib.rs                    # Public API and re-exports
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
│   │   ├── order_flow.rs        # OFI, MLOFI, queue imbalance, trade flow
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
│   ├── config.rs                 # Pipeline configuration
│   ├── pipeline.rs               # High-level pipeline orchestrator
│   ├── export.rs                 # NumPy export
│   └── export_aligned.rs         # Aligned batch export
│
├── benches/
│   └── feature_extraction.rs     # Criterion benchmarks
│
├── examples/                      # Usage examples
└── tests/                         # Integration tests
```

## Feature Categories

| Category | Count | Source | Description |
|----------|-------|--------|-------------|
| Raw LOB | 40 | All papers | (P_ask, V_ask, P_bid, V_bid) × 10 levels |
| Order Flow | 8 | Cont et al. | OFI, queue imbalance, trade flow, arrival rates |
| Multi-Level OFI | 10 | LOB-feature-analysis | OFI at each LOB level |
| FI-2010 Time-Insensitive | 20 | FI-2010 | Spread, mid-price, price/volume diffs |
| FI-2010 Time-Sensitive | 20 | FI-2010 | Derivatives, intensity measures |
| FI-2010 Depth | 40 | FI-2010 | Accumulated volumes and price diffs |
| Derived | 8 | TLOB, DeepLOB | Microprice, VWAP, imbalance |
| Market Impact | 8 | OrderBook-rs | Slippage, VWAP, levels consumed |
| MBO Features | 36 | MBO Paper | Order lifecycle, institutional patterns |

## Normalization Strategies

| Strategy | Source | Use Case |
|----------|--------|----------|
| Z-Score | DeepLOB, FI-2010 | Standard ML preprocessing |
| Rolling Z-Score | LOBFrame | Non-stationary data (multi-day) |
| Global Z-Score | LOBench | Preserve LOB constraints (bid < ask) |
| Bilinear | TLOB, BiN-CTABL | LOB structure preservation |
| Percentage Change | HLOB | Cross-instrument training |
| Min-Max | General | Bounded features |
| Per-Feature | General | Feature-specific normalization |

## Paper-Aligned Presets

```rust
pub enum Preset {
    /// DeepLOB: 40 raw + Z-score, seq_len=100, k=10/20/50
    DeepLOB,
    
    /// TLOB: 40 raw + bilinear normalization, dual attention
    TLOB,
    
    /// FI-2010: 120 features (40 raw + 80 handcrafted)
    FI2010,
    
    /// TransLOB: 40 raw + multi-horizon, transformer-ready
    TransLOB,
    
    /// LiT: 80 features (20 levels × 4), patched input
    LiT,
    
    /// Minimal: 40 raw LOB only
    Minimal,
    
    /// Full: All available features
    Full,
}
```

## Key Components

### 1. FI2010Extractor

Implements the 80 handcrafted features from the FI-2010 benchmark paper:

```rust
let mut extractor = FI2010Extractor::new(FI2010Config::default());
let features = extractor.extract(&lob_state, timestamp)?;
// Returns 80 features: 20 time-insensitive + 20 time-sensitive + 40 depth
```

### 2. OrderFlowTracker

Computes Order Flow Imbalance (OFI) and related features:

```rust
let mut tracker = OrderFlowTracker::new();
tracker.update(&lob_state);
let features = tracker.features();
// OFI, queue imbalance, trade imbalance, depth imbalance, arrival rates
```

### 3. MultiLevelOfiTracker

Computes OFI at each LOB level:

```rust
let mut tracker = MultiLevelOfiTracker::new(10);
tracker.update(&lob_state);
let ofi_levels = tracker.ofi_by_level();
// 10 OFI values, one per level
```

### 4. GlobalZScoreNormalizer (LOBench)

Normalizes all features within a snapshot together:

```rust
let normalizer = GlobalZScoreNormalizer::new();
let normalized = normalizer.normalize_snapshot(&features);
// Preserves bid < ask constraint, handles scale disparity
```

### 5. FeatureValidator

Validates LOB data quality:

```rust
let validator = FeatureValidator::new();
let result = validator.validate_lob(&lob_state);
if result.has_errors() {
    // Handle crossed quotes, invalid prices, etc.
}
```

### 6. TlobLabelGenerator

Generates labels using the TLOB paper method (decoupled horizon and smoothing):

```rust
let config = LabelConfig {
    horizon: 10,           // Predict 10 steps ahead
    smoothing_window: 5,   // Average 5 prices for smoothing
    threshold: 0.002,      // 0.2% threshold
};
let mut generator = TlobLabelGenerator::new(config);
generator.add_prices(&mid_prices);
let labels = generator.generate_labels()?;
// Returns Vec<(index, TrendLabel, pct_change)>
```

### 7. DeepLobLabelGenerator

Generates labels using the simpler DeepLOB method (k = horizon):

```rust
let config = LabelConfig::fi2010(50);  // k = h = 50
let mut generator = DeepLobLabelGenerator::new(config);
generator.add_prices(&mid_prices);
let labels = generator.generate_labels()?;
```

## Test Coverage

- **298+ unit tests** covering all modules
- **49 doc tests** with working examples
- **Integration tests** with real NVIDIA MBO data
- **Labeling integration tests** (18 tests)
- **Benchmark suite** for performance tracking

## Performance Characteristics

- Single-pass feature computation
- Pre-allocated buffers in hot paths
- Zero-copy where possible
- Welford's algorithm for numerical stability
- O(1) updates for rolling statistics

## Dependencies

- `mbo-lob-reconstructor`: LOB reconstruction from MBO data
- `serde`: Serialization for configuration
- `criterion` (dev): Benchmarking

## Roadmap

### Completed ✅
- [x] Raw LOB features (40)
- [x] Order Flow features (OFI, MLOFI, queue imbalance)
- [x] FI-2010 features (80)
- [x] Market Impact estimation
- [x] All normalization strategies
- [x] Multi-scale sequence building
- [x] Validation module
- [x] Benchmark suite
- [x] Labeling module (TLOB + DeepLOB methods)
- [x] README.md and documentation
- [x] Standalone extraction to separate repository
- [x] Comprehensive real-data validation (151M+ messages, 21 days NVIDIA data)

### Pending 📋
- [ ] CI workflow (GitHub Actions)
- [ ] crates.io publication
- [ ] Statistical validation tests (OFI vs ΔP correlation)
