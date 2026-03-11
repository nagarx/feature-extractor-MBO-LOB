# Configuration Reference: DatasetConfig and TOML Configs

> **Purpose**: Reference for the TOML-driven export system. An LLM coder should understand every config field, all available configs, and how to create new ones.
> **Last Updated**: March 5, 2026

---

## `export_dataset` CLI

The primary production export path:

```bash
# Export using a TOML config
cargo run --release --features parallel --bin export_dataset -- --config configs/nvda_98feat.toml

# Generate a config template
cargo run --release --features parallel --bin export_dataset -- --generate-config configs/new_config.toml
```

**Source**: `tools/export_dataset.rs`

---

## DatasetConfig Schema

`DatasetConfig` is parsed from TOML by `DatasetConfig::load_toml(path)`.
It is converted to internal types via `config.to_pipeline_config()`.

**Source**: `src/export/config/mod.rs`, `src/export/config/*.rs`

### `[experiment]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | String | Yes | Experiment name for tracking |
| `description` | String | No | Human-readable description |
| `version` | String | No | Config version for reproducibility |
| `tags` | Vec\<String\> | No | Tags for filtering/search |

### `[symbol]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | String | Yes | Symbol ticker (e.g., "NVDA") |
| `exchange` | String | Yes | Exchange code (e.g., "XNAS") |
| `filename_pattern` | String | Yes | Pattern with `{date}` placeholder (e.g., `"xnas-itch-{date}.mbo.dbn.zst"`) |
| `tick_size` | f64 | Yes | Minimum price increment in dollars (e.g., 0.01 for US equities) |

### `[data]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_dir` | String | Yes | Directory containing raw DBN files |
| `output_dir` | String | Yes | Directory for exported NPY/JSON files |
| `hot_store_dir` | String | No | Optional decompressed file cache (~30% faster) |

### `[dates]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `start_date` | String | Yes (unless `explicit_dates`) | Start date (YYYY-MM-DD). Weekends auto-excluded |
| `end_date` | String | Yes (unless `explicit_dates`) | End date (YYYY-MM-DD) |
| `exclude_dates` | Vec\<String\> | No | Specific dates to skip (holidays) |
| `explicit_dates` | Vec\<String\> | No | Explicit date list (overrides start/end) |

### `[features]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lob_levels` | usize | 10 | Number of LOB price levels |
| `include_derived` | bool | false | +8 derived features (indices 40-47) |
| `include_mbo` | bool | false | +36 MBO features (indices 48-83) |
| `include_signals` | bool | false | +14 signals (indices 84-97). Requires `lob_levels == 10` |
| `mbo_window_size` | usize | 1000 | MBO aggregation rolling window size |

#### `[features.experimental]` (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable +18 experimental features (indices 98-115) |
| `groups` | Vec\<String\> | all | Which groups: `"institutional_v2"`, `"volatility"`, `"seasonality"` |

### `[sampling]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | String | `"event_based"` | `"event_based"` or `"volume_based"` |
| `event_count` | usize | 1000 | Events between samples (if event_based) |
| `volume_threshold` | usize | 1000 | Shares between samples (if volume_based) |
| `min_time_interval_ns` | u64 | 1000000 | Minimum 1ms between samples |

### `[sequence]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | usize | 100 | Snapshots per sequence |
| `stride` | usize | 10 | Step between sequences |
| `max_buffer_size` | usize | 50000 | Maximum circular buffer capacity |

### `[labels]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | String | `"tlob"` | `"tlob"`, `"opportunity"`, `"triple_barrier"` |
| `horizon` | usize | 50 | Single-horizon look-ahead |
| `horizons` | Vec\<usize\> | \[\] | Multi-horizon list (overrides horizon) |
| `smoothing_window` | usize | 10 | TLOB smoothing window (k) |
| `threshold` | f64 | 0.0008 | Classification threshold |

#### TLOB Threshold Strategy (optional)

```toml
[labels.threshold_strategy]
type = "fixed"           # or "rolling_spread", "quantile", "tlob_dynamic"
value = 0.001            # for Fixed
```

```toml
[labels.threshold_strategy]
type = "quantile"
target_proportion = 0.33
window_size = 5000
fallback = 0.002
```

```toml
[labels.threshold_strategy]
type = "tlob_dynamic"
fallback = 0.0008
divisor = 2.0
```

#### Opportunity Labels

```toml
[labels]
strategy = "opportunity"
horizon = 50
threshold = 0.005
conflict_priority = "larger_magnitude"  # or "up_priority", "down_priority", "ambiguous"
```

#### Triple Barrier Labels

```toml
[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_target_pct = 0.005
stop_loss_pct = 0.003
timeout_strategy = "label_as_timeout"  # or "use_return_sign", "use_fractional_threshold"

# Per-horizon barriers (overrides single values)
profit_targets = [0.0028, 0.0039, 0.0059]
stop_losses = [0.0019, 0.0026, 0.0040]

# Volatility-adaptive scaling
volatility_scaling = true
volatility_reference = 0.00015
volatility_floor = 0.3
volatility_cap = 3.0
```

### `[normalization]` (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | String | `"raw"` | `"raw"`, `"z_score"`, `"market_structure_zscore"` |

### `[split]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_ratio` | f64 | 0.7 | Training set proportion |
| `val_ratio` | f64 | 0.15 | Validation set proportion |
| `test_ratio` | f64 | 0.15 | Test set proportion |

**Invariant**: `train_ratio + val_ratio + test_ratio == 1.0`

### `[processing]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `threads` | usize | `num_cpus - 2` | Number of parallel threads |
| `error_mode` | String | `"fail_fast"` | `"fail_fast"` or `"collect_errors"` |
| `verbose` | bool | false | Enable verbose logging |

---

## Config Inventory

### Standard Configs

| Config File | Features | Labeling | Threshold | Sampling | Use Case |
|-------------|----------|----------|-----------|----------|----------|
| `nvda_98feat.toml` | 98 | TLOB | Fixed 0.0008 | Event 1000 | Reference 98-feature config |
| `nvda_98feat_v2.toml` | 98 | TLOB | TlobDynamic | Event 1000 | Dynamic threshold per TLOB paper |
| `nvda_98feat_full.toml` | 98 | TLOB | Fixed | Event 1000 | Full 98-feature dataset |
| `nvda_84feat_baseline.toml` | 84 | TLOB | Fixed | Event 1000 | Baseline without signals |
| `nvda_116feat_full_analysis.toml` | 116 | TLOB | Fixed | Event 1000 | Full with experimental features |

### Multi-Horizon Configs

| Config File | Features | Labeling | Horizons | Threshold | Use Case |
|-------------|----------|----------|----------|-----------|----------|
| `nvda_multi_horizon.toml` | 98 | Multi-Horizon TLOB | \[10, 20, 50, 100, 200\] | Fixed | Signal decay analysis |
| `nvda_extended_multi_horizon.toml` | 98 | Multi-Horizon TLOB | Extended set | Fixed | Extended horizon analysis |
| `nvda_balanced.toml` | 98 | Multi-Horizon TLOB | \[10, 20, 50, 100, 200\] | Quantile 0.33 | Balanced class distribution |
| `nvda_spread_adaptive.toml` | 98 | Multi-Horizon TLOB | Multiple | RollingSpread | Transaction-cost-aware thresholds |

### Triple Barrier Configs

| Config File | Features | Barriers | Volatility Scaling | Use Case |
|-------------|----------|----------|--------------------|----------|
| `nvda_triple_barrier.toml` | 98 | Fixed | No | Basic triple barrier labeling |
| `nvda_11month_triple_barrier.toml` | 98 | Fixed | No | Full dataset, triple barrier |
| `nvda_11month_triple_barrier_calibrated.toml` | 98 | Per-horizon calibrated | No | Data-driven barrier calibration |
| `nvda_11month_triple_barrier_volscaled.toml` | 98 | Per-horizon | Yes | Volatility-adaptive barriers |

### Opportunity Configs

| Config File | Features | Threshold | Use Case |
|-------------|----------|-----------|----------|
| `nvda_bigmove_detection.toml` | 98 | 50+ bps, high imbalance | Big move / whale detection |

### Long-Run Configs

| Config File | Features | Date Range | Use Case |
|-------------|----------|-----------|----------|
| `nvda_11month_complete.toml` | 98 | Feb 2025 - Jan 2026 (233 days) | Full production dataset |
| `nvda_tlob_raw_v2.toml` | 98 | 233 days | Raw LOB for TLOB (BiN-compatible) |
| `nvda_tlob_repo_v1.toml` | 98 | Various | TLOB repo-style global Z-score + BiN |
| `nvda_tlob_repo_v2.toml` | 98 | Various | Updated TLOB repo-style config |
| `nvda_arcx_98feat.toml` | 98 | 233 days | ARCX (NYSE Arca) MBO, TLOB dynamic threshold |
| `nvda_arcx_98feat_multi_horizon.toml` | 98 | 233 days | ARCX multi-horizon TLOB labels |

### Templates and Examples

| Config File | Purpose |
|-------------|---------|
| `template_multi_symbol.toml` | Generic template for multi-symbol configs |
| `configs/examples/nvda_tlob_raw.toml` | Example: TLOB raw export |
| `configs/examples/nvda_tlob_repo.toml` | Example: TLOB official repo preprocessing |

---

## TOML-to-Pipeline Mapping

The transformation from TOML to internal types happens in `DatasetConfig::to_pipeline_config()`:

```
[features]  →  FeatureConfig {
    lob_levels, include_derived, include_mbo, include_signals,
    mbo_window_size, tick_size (from [symbol])
}

[sequence]  →  SequenceConfig {
    window_size, stride, max_buffer_size,
    feature_count (auto-computed from FeatureConfig)
}

[sampling]  →  SamplingConfig {
    strategy (VolumeBased | EventBased),
    volume_threshold, event_count, min_time_interval_ns
}

[labels]  →  ExportLabelConfig → LabelConfig | MultiHorizonConfig |
             OpportunityConfig | TripleBarrierConfig
```

---

## Creating a New Config

1. Copy `configs/nvda_98feat.toml` as template
2. Modify `[symbol]` for your instrument (name, exchange, tick_size, filename_pattern)
3. Modify `[data]` paths for your environment
4. Modify `[dates]` for your date range
5. Adjust `[labels]` for your labeling strategy
6. Run: `cargo run --release --features parallel --bin export_dataset -- --config configs/your_config.toml`

---

*Last updated: March 5, 2026*
