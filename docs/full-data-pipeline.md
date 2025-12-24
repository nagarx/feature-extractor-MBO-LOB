# Full Data Pipeline Specification: MBO → LOB Feature Dataset

> **Purpose**: Single source of truth for all data transformations from raw Databento MBO files to model-ready LOB feature datasets. This document describes the preprocessing pipeline capabilities independent of any specific model architecture.  
> **Audience**: LLMs and developers needing exact technical details for debugging, extending, or reproducing the pipeline.  
> **Last Updated**: 2025-12-19 (DatasetConfig export system added)

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage 1: DBN File Loading](#2-stage-1-dbn-file-loading)
3. [Stage 2: LOB Reconstruction](#3-stage-2-lob-reconstruction)
4. [Stage 3: Sampling](#4-stage-3-sampling)
5. [Stage 4: Feature Extraction](#5-stage-4-feature-extraction)
   - [4A: Raw LOB Features (40)](#5a-raw-lob-features-40-features)
   - [4B: Derived Features (8)](#5b-derived-features-8-features)
   - [4C: MBO Features (36)](#5c-mbo-features-36-features)
   - [4D: Feature Configuration](#5d-feature-configuration)
6. [Stage 5: Sequence Building](#6-stage-5-sequence-building)
7. [Stage 6: Label Generation](#7-stage-6-label-generation)
   - [6A: TLOB Labeling Method](#7a-tlob-labeling-method)
   - [6B: DeepLOB Labeling Method](#7b-deeplob-labeling-method)
   - [6C: Threshold Strategies](#7c-threshold-strategies)
8. [Stage 7: Sequence-Label Alignment](#8-stage-7-sequence-label-alignment)
9. [Stage 8: Per-Day Normalization](#9-stage-8-per-day-normalization)
10. [Stage 9: Export Artifacts](#10-stage-9-export-artifacts)
11. [Stage 10: Python Data Loading](#11-stage-10-python-data-loading)
12. [Stage 11: Feature Layout Transformation](#12-stage-11-feature-layout-transformation)
13. [Stage 12: Train-Stats Renormalization (P0.2)](#13-stage-12-train-stats-renormalization-p02)
14. [Data Type Summary](#14-data-type-summary)
15. [Known Issues & Contracts](#15-known-issues--contracts)
16. [File Reference Index](#16-file-reference-index)
17. [Appendix A: LobState Analytics](#17-appendix-a-lobstate-analytics)
18. [Appendix B: Feature Set Summary](#18-appendix-b-feature-set-summary)

---

## 1. Pipeline Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RUST PREPROCESSING                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  Databento DBN (.dbn.zst)
       │
       ▼ [Stage 1: DbnLoader + DbnBridge]
  MboMessage { order_id, action, side, price(i64), size(u32), timestamp }
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
  [Stage 2: LobReconstructor]           [MboAggregator (optional)]
  LobState { bid/ask prices & sizes }   MboWindow { rolling event stats }
       │                                      │
       ▼                                      │
  [Stage 3: Sampling]                         │
  ├─ EventBasedSampler (every N events)       │
  └─ VolumeBasedSampler (every N shares)      │
       │                                      │
       ▼                                      │
  Sampled LobState + mid_price (f64)          │
       │                                      │
       ▼ [Stage 4: Feature Extraction] ◄──────┘
       ├─ extract_raw_features()      → 40 features (raw LOB)
       ├─ compute_derived_features()  → 8 features (optional)
       └─ MboAggregator.extract()     → 36 features (optional)
       │
       ▼
  Vec<f64> [40-84 features, GROUPED layout]
       │
       ▼ [Stage 5: SequenceBuilder]
  Sequence { features: Vec<Arc<Vec<f64>>>, length: 100 }
       │
       ▼ [Stage 6: Label Generation]
       ├─ TlobLabelGenerator (smoothed past vs smoothed future)
       └─ DeepLobLabelGenerator (alternative method)
       │
       ▼ [Stage 7: align_sequences_with_labels()]
  (aligned_sequences, aligned_labels) - 1:1 mapping
       │
       ▼ [Stage 8: normalize_sequences() - market_structure_zscore]
  Normalized sequences (mean≈0, std≈1 per day) + NormalizationParams
       │
       ▼ [Stage 9: Export]
  {day}_sequences.npy, {day}_labels.npy, {day}_normalization.json

┌─────────────────────────────────────────────────────────────────────────────┐
│                           PYTHON DATA LOADING                                │
└─────────────────────────────────────────────────────────────────────────────┘

  .npy files + manifest
       │
       ▼ [Stage 10: AlignedExportSource / DataModule]
  PyTorch tensors (float32) + label class indices {0,1,2}
       │
       ▼ [Stage 11: FeatureReorderTransform (optional)]
  GROUPED → INTERLEAVED layout (if model requires)
       │
       ▼ [Stage 12: RenormTransform (optional)]
  Per-day normalized → Train-stats normalized
       │
       ▼
  Model Input (TLOB, DeepLOB, Transformer, etc.)
```

### Key Numerical Invariants

| Quantity | Value | Source |
|----------|-------|--------|
| LOB levels | 10 | Pipeline config |
| Raw LOB features per snapshot | 40 | 10 levels × 4 feature types |
| Derived features (optional) | 8 | Spread, imbalance, etc. |
| MBO features (optional) | 36 | Order flow, institutional patterns |
| Trading signals (optional) | 14 | OFI, time regime, safety gates |
| **Total possible features** | **98** | 40 + 8 + 36 + 14 |
| Sequence length | 100 | `window_size` |
| Stride | 10 | Sequences overlap by 90 snapshots |
| Sample interval (event-based) | 1000 events | EventBasedSampler |
| Sample interval (volume-based) | configurable | VolumeBasedSampler |
| Prediction horizons | [10, 20, 50, 100, 200] | Multi-horizon config (configurable) |
| Label threshold | 0.0008 (8 bps) | LabelConfig |
| Smoothing window | 10 | LabelConfig |

---

## 2. Stage 1: DBN File Loading

### Input

- **File**: `xnas-itch-YYYYMMDD.mbo.dbn.zst` (zstd-compressed Databento DBN)
- **Location**: `/data/databento/NVDA/` (convention)

### Implementation

| File | Module |
|------|--------|
| `MBO-LOB-reconstructor/src/loader.rs` | `DbnLoader` |
| `MBO-LOB-reconstructor/src/dbn_bridge.rs` | `DbnBridge::convert()` |

### Transformation

**Databento `dbn::MboMsg`** → **Internal `MboMessage`**

```rust
// From MBO-LOB-reconstructor/src/dbn_bridge.rs
pub fn convert(msg: &dbn::MboMsg) -> Result<MboMessage> {
    Ok(MboMessage {
        order_id: msg.order_id,
        action: convert_action(msg.action as u8)?,   // b'A' → Action::Add, etc.
        side: convert_side(msg.side as u8)?,         // b'B' → Side::Bid, etc.
        price: msg.price,                            // i64, fixed-point (nanodollars)
        size: msg.size,                              // u32, shares
        timestamp: Some(msg.hd.ts_event as i64),     // nanoseconds since epoch
    })
}
```

### Output Schema: `MboMessage`

```rust
pub struct MboMessage {
    pub order_id: u64,         // 0 = system message (skip)
    pub action: Action,        // Add | Modify | Cancel | Trade | Fill | Clear | None
    pub side: Side,            // Bid | Ask | None
    pub price: i64,            // Fixed-point: dollars = price / 1e9
    pub size: u32,             // Shares
    pub timestamp: Option<i64>, // Nanoseconds since epoch
}
```

### Price Conversion Formula

```
price_dollars = price_i64 / 1_000_000_000.0
```

**Example**: `price = 120_500_000_000` → `$120.50`

### System Message Filtering

Messages with `order_id == 0` or `size == 0` or `price <= 0` are **system messages** (heartbeats, status updates). These are skipped by default:

```rust
// LobReconstructor.process_message()
if self.config.skip_system_messages && (msg.order_id == 0 || msg.size == 0 || msg.price <= 0) {
    self.stats.system_messages_skipped += 1;
    return Ok(self.get_lob_state());  // Return unchanged state
}
```

---

## 3. Stage 2: LOB Reconstruction

### Implementation

| File | Module |
|------|--------|
| `MBO-LOB-reconstructor/src/lob/reconstructor.rs` | `LobReconstructor` |
| `MBO-LOB-reconstructor/src/types.rs` | `LobState` |

### Internal State

```rust
pub struct LobReconstructor {
    bids: BTreeMap<i64, PriceLevel>,     // price → orders at that price
    asks: BTreeMap<i64, PriceLevel>,     // price → orders at that price
    orders: AHashMap<u64, Order>,        // order_id → (side, price, size)
    best_bid: Option<i64>,               // Cached for O(1) access
    best_ask: Option<i64>,               // Cached for O(1) access
    // ...
}
```

### Action Processing

| Action | Behavior |
|--------|----------|
| `Add` | Insert order into price level, track in orders map |
| `Modify` | Remove old order, add new (handles price changes) |
| `Cancel` | Reduce size or remove order; supports partial cancels |
| `Trade` | Reduce size or remove order (execution) |
| `Clear` | Call `reset()`, clear entire book |
| `None` | No-op |

### Output Schema: `LobState`

```rust
pub struct LobState {
    // Core LOB data (stack-allocated arrays, MAX_LOB_LEVELS = 20)
    pub bid_prices: [i64; 20],   // Highest to lowest (index 0 = best bid)
    pub bid_sizes: [u32; 20],    // Aggregated shares at each price
    pub ask_prices: [i64; 20],   // Lowest to highest (index 0 = best ask)
    pub ask_sizes: [u32; 20],    // Aggregated shares at each price
    pub best_bid: Option<i64>,   // Cached best prices
    pub best_ask: Option<i64>,
    pub levels: usize,           // Number of tracked levels (≤ 20)
    pub timestamp: Option<i64>,  // Nanoseconds
    pub sequence: u64,           // Message sequence number
    // Temporal fields for FI-2010 features (not used in current TLOB export)
    pub delta_ns: u64,
    pub triggering_action: Option<Action>,
    pub triggering_side: Option<Side>,
}
```

### Mid-Price Calculation

```rust
// From MBO-LOB-reconstructor/src/types.rs
pub fn mid_price(&self) -> Option<f64> {
    match (self.best_bid, self.best_ask) {
        (Some(bid), Some(ask)) => Some((bid as f64 + ask as f64) / 2.0 / 1e9),
        _ => None,
    }
}
```

**Formula**: `mid_price = (best_bid + best_ask) / 2 / 1e9` (in dollars)

### LobState Analytics Methods

The `LobState` struct provides additional analytical methods beyond mid-price:

| Method | Formula | Description |
|--------|---------|-------------|
| `spread()` | `(best_ask - best_bid) / 1e9` | Bid-ask spread in dollars |
| `microprice()` | `(bid × ask_vol + ask × bid_vol) / (ask_vol + bid_vol) / 1e9` | Volume-weighted mid-price |
| `depth_imbalance(n)` | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` | Normalized volume imbalance across top n levels |
| `vwap_bid(n)` | `Σ(price_i × size_i) / Σ(size_i)` | Volume-weighted average price (bid side) |
| `vwap_ask(n)` | `Σ(price_i × size_i) / Σ(size_i)` | Volume-weighted average price (ask side) |
| `total_bid_volume(n)` | `Σ(bid_size_i)` | Total shares on bid side |
| `total_ask_volume(n)` | `Σ(ask_size_i)` | Total shares on ask side |
| `active_bid_levels()` | count of `bid_prices[i] > 0` | Number of active bid levels |
| `active_ask_levels()` | count of `ask_prices[i] > 0` | Number of active ask levels |

These methods can be used for advanced feature engineering or real-time analytics.

---

## 4. Stage 3: Sampling

### Implementation

| File | Module |
|------|--------|
| `feature-extractor-MBO-LOB/src/preprocessing/sampling.rs` | `EventBasedSampler`, `VolumeBasedSampler` |
| `feature-extractor-MBO-LOB/src/pipeline.rs` | `Pipeline.should_sample()` |

### Sampling Strategies

The pipeline supports two sampling strategies, each with different trade-offs:

#### Strategy 1: Event-Based Sampling

Samples the order book after a fixed number of MBO events. Simple and predictable.

```rust
pub struct EventBasedSampler {
    sample_interval: u64,  // e.g., 1000
    event_count: u64,
    sample_count: u64,
}

impl EventBasedSampler {
    pub fn should_sample(&mut self) -> bool {
        self.event_count += 1;
        if self.event_count >= self.sample_interval {
            self.event_count = 0;
            self.sample_count += 1;
            return true;
        }
        false
    }
}
```

**Configuration (NVDA Export)**:
```rust
SamplingConfig {
    strategy: SamplingStrategy::EventBased,
    event_count: Some(1000),  // Sample every 1000 valid MBO events
}
```

**Characteristics**:
- ✅ Simple, deterministic
- ✅ Consistent sample count across similar-activity days
- ⚠️ Can over/under-sample during high/low activity periods

#### Strategy 2: Volume-Based Sampling (Recommended by TLOB Paper)

Samples the order book after a predetermined volume of shares has been traded. Aligns with market activity and provides better signal quality for capturing institutional order flow.

```rust
pub struct VolumeBasedSampler {
    target_volume: u64,           // e.g., 10000 shares
    accumulated_volume: u64,
    min_time_interval_ms: u64,    // Minimum time between samples
    last_sample_ts: u64,
}

impl VolumeBasedSampler {
    pub fn should_sample(&mut self, event_volume: u32, timestamp_ns: u64) -> bool {
        self.accumulated_volume += event_volume as u64;
        
        // Check minimum time interval
        let elapsed_ms = (timestamp_ns - self.last_sample_ts) / 1_000_000;
        if elapsed_ms < self.min_time_interval_ms {
            return false;
        }
        
        if self.accumulated_volume >= self.target_volume {
            self.accumulated_volume = 0;
            self.last_sample_ts = timestamp_ns;
            return true;
        }
        false
    }
    
    /// Dynamically update the volume threshold at runtime.
    /// Enables adaptive sampling strategies that adjust to market conditions.
    pub fn set_threshold(&mut self, new_threshold: u64) {
        self.target_volume = new_threshold;
    }
}
```

**Configuration Example**:
```rust
SamplingConfig {
    strategy: SamplingStrategy::VolumeBased,
    target_volume: Some(10000),      // Sample every 10K shares traded
    min_time_interval_ms: Some(100), // Minimum 100ms between samples
}
```

**Adaptive Sampling Example**:
```rust
let mut sampler = VolumeBasedSampler::new(1000, 1_000_000);

// Adapt to market conditions during processing
if realized_volatility > high_threshold {
    sampler.set_threshold(1500);  // Wider threshold in volatile markets
} else if realized_volatility < low_threshold {
    sampler.set_threshold(500);   // Tighter threshold in quiet markets
}
```

**Characteristics**:
- ✅ Adapts to market activity (more samples during high activity)
- ✅ Better captures institutional trading patterns
- ✅ Recommended by TLOB paper for signal quality
- ⚠️ Variable sample count per day

### Strategy Comparison

| Aspect | Event-Based | Volume-Based |
|--------|-------------|--------------|
| Trigger | Every N messages | Every N shares traded |
| Market adaptation | None | Automatic |
| High activity periods | Under-samples information | Samples proportionally |
| Low activity periods | Over-samples noise | Fewer samples |
| Sample count predictability | High | Variable |
| Research recommendation | Standard baseline | TLOB paper preferred |

### Output

For each sample:
- `LobState` (current order book snapshot)
- `mid_price: f64` (dollars, for labeling)
- `timestamp: u64` (nanoseconds since epoch)

**Typical day statistics**:
- Event-based (1000 events): ~10K-18K samples
- Volume-based (10K shares): ~5K-25K samples (depends on trading activity)

---

## 5. Stage 4: Feature Extraction

The pipeline supports three categories of features, totaling up to 84 features per snapshot:

| Category | Features | Source |
|----------|----------|--------|
| Raw LOB | 40 | `extract_raw_features()` |
| Derived | 8 | `compute_derived_features()` |
| MBO | 36 | `MboAggregator.extract_features()` |
| Trading Signals | 14 | `signals::compute_signals()` |
| **Total** | **98** | Configurable via `FeatureConfig` |

---

### 5A. Raw LOB Features (40 features)

#### Implementation

| File | Function |
|------|----------|
| `feature-extractor-MBO-LOB/src/features/lob_features.rs` | `extract_raw_features()` |

#### Transformation: LobState → Raw Features

```rust
pub fn extract_raw_features(lob_state: &LobState, levels: usize, output: &mut Vec<f64>) {
    let levels = levels.min(lob_state.levels);  // Typically 10

    // Ask prices (levels 0-9): convert from nanodollars to dollars
    for i in 0..levels {
        let price = if lob_state.ask_prices[i] > 0 {
            lob_state.ask_prices[i] as f64 / 1e9
        } else {
            0.0
        };
        output.push(price);
    }

    // Ask sizes (levels 10-19): raw shares as f64
    for i in 0..levels {
        output.push(lob_state.ask_sizes[i] as f64);
    }

    // Bid prices (levels 20-29): convert from nanodollars to dollars
    for i in 0..levels {
        let price = if lob_state.bid_prices[i] > 0 {
            lob_state.bid_prices[i] as f64 / 1e9
        } else {
            0.0
        };
        output.push(price);
    }

    // Bid sizes (levels 30-39): raw shares as f64
    for i in 0..levels {
        output.push(lob_state.bid_sizes[i] as f64);
    }
}
```

#### Output Layout: GROUPED (40 features)

```
Index Range | Feature Type    | Unit    | Description
------------|-----------------|---------|---------------------------
0-9         | ask_prices      | dollars | Best ask to level 10
10-19       | ask_sizes       | shares  | Volume at each ask level
20-29       | bid_prices      | dollars | Best bid to level 10
30-39       | bid_sizes       | shares  | Volume at each bid level
```

**Layout name**: `GROUPED` (features grouped by type, then by level)

**Alternative layout**: `INTERLEAVED` (features ordered by level, then by type):
```
[ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...]
```

The Rust exporter outputs **GROUPED**. Some models (TLOB/DeepLOB) expect **INTERLEAVED** (see Stage 11).

#### Alternative: Normalized Feature Extraction

For multi-instrument training (e.g., training on multiple stocks simultaneously), there's an alternative extraction function that normalizes prices relative to mid-price:

```rust
pub fn extract_normalized_features(lob_state: &LobState, levels: usize, output: &mut Vec<f64>) {
    let mid = lob_state.mid_price().unwrap_or(0.0);
    let total_volume = lob_state.total_bid_volume(levels) + lob_state.total_ask_volume(levels);
    
    // Prices normalized relative to mid-price (scale-invariant)
    for i in 0..levels {
        let price = lob_state.ask_prices[i] as f64 / 1e9;
        output.push(if mid > 0.0 { (price - mid) / mid } else { 0.0 });
    }
    
    // Sizes normalized relative to total volume
    for i in 0..levels {
        output.push(lob_state.ask_sizes[i] as f64 / total_volume.max(1) as f64);
    }
    // ... (same pattern for bid side)
}
```

This produces features in the range [-1, 1] for prices and [0, 1] for sizes, enabling cross-instrument training without scale differences.

---

### 5B. Derived Features (8 features)

#### Implementation

| File | Function |
|------|----------|
| `feature-extractor-MBO-LOB/src/features/derived_features.rs` | `compute_derived_features()` |

#### Feature Definitions

```rust
pub fn compute_derived_features(lob_state: &LobState, levels: usize) -> Result<[f64; 8]> {
    let best_bid_f64 = best_bid as f64 / 1e9;
    let best_ask_f64 = best_ask as f64 / 1e9;

    // 1. Mid-price
    let mid_price = (best_bid_f64 + best_ask_f64) / 2.0;

    // 2. Spread (absolute, dollars)
    let spread = best_ask_f64 - best_bid_f64;

    // 3. Spread (basis points)
    let spread_bps = (spread / mid_price) * 10_000.0;

    // 4. Total bid volume
    let total_bid_volume: f64 = (0..levels).map(|i| lob_state.bid_sizes[i] as f64).sum();

    // 5. Total ask volume
    let total_ask_volume: f64 = (0..levels).map(|i| lob_state.ask_sizes[i] as f64).sum();

    // 6. Volume imbalance
    let volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume);

    // 7. Weighted mid-price (microprice)
    let best_bid_size = lob_state.bid_sizes[0] as f64;
    let best_ask_size = lob_state.ask_sizes[0] as f64;
    let weighted_mid_price = (best_bid_f64 * best_ask_size + best_ask_f64 * best_bid_size) 
                            / (best_bid_size + best_ask_size);

    // 8. Price impact (difference between mid and microprice)
    let price_impact = (mid_price - weighted_mid_price).abs();

    Ok([mid_price, spread, spread_bps, total_bid_volume, total_ask_volume, 
        volume_imbalance, weighted_mid_price, price_impact])
}
```

#### Feature Index Mapping

| Index | Feature | Formula | Unit |
|-------|---------|---------|------|
| 0 | mid_price | `(best_bid + best_ask) / 2` | dollars |
| 1 | spread | `best_ask - best_bid` | dollars |
| 2 | spread_bps | `spread / mid_price × 10000` | basis points |
| 3 | total_bid_volume | `Σ bid_sizes[0..L]` | shares |
| 4 | total_ask_volume | `Σ ask_sizes[0..L]` | shares |
| 5 | volume_imbalance | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` | ratio [-1, 1] |
| 6 | weighted_mid_price | `(bid×ask_vol + ask×bid_vol) / (bid_vol + ask_vol)` | dollars |
| 7 | price_impact | `|mid_price - weighted_mid_price|` | dollars |

#### Additional Depth Features

```rust
pub fn compute_depth_features(lob_state: &LobState, levels: usize, tick_size: f64) -> [f64; 4] {
    // Returns:
    // [0] active_bid_levels: number of levels with orders
    // [1] active_ask_levels: number of levels with orders
    // [2] avg_bid_depth_ticks: average depth in price ticks
    // [3] avg_ask_depth_ticks: average depth in price ticks
}
```

---

### 5C. MBO Features (36 features)

#### Implementation

| File | Module |
|------|--------|
| `feature-extractor-MBO-LOB/src/features/mbo_features.rs` | `MboAggregator`, `MboWindow`, `OrderInfo` |

#### Architecture

The MBO feature extractor uses a multi-timescale rolling window architecture:

```rust
pub struct MboAggregator {
    fast_window: MboWindow,    // 100 events (~2s) - immediate signals
    medium_window: MboWindow,  // 1000 events (~20s) - primary features
    slow_window: MboWindow,    // 5000 events (~100s) - long-term trends
    order_tracker: AHashMap<u64, OrderInfo>,  // Order lifecycle tracking
}
```

**Memory Usage**: ~8 MB per symbol (bounded, no leaks)

#### Feature Categories (36 total)

**Category 1: Order Flow Statistics (12 features)**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | add_rate_bid | Bid add orders per second |
| 1 | add_rate_ask | Ask add orders per second |
| 2 | cancel_rate_bid | Bid cancels per second |
| 3 | cancel_rate_ask | Ask cancels per second |
| 4 | trade_rate_bid | Bid trades per second |
| 5 | trade_rate_ask | Ask trades per second |
| 6 | net_order_flow | `(adds_bid - adds_ask) / total_adds` |
| 7 | net_cancel_flow | `(cancels_ask - cancels_bid) / total_cancels` ¹ |
| 8 | net_trade_flow | `(trades_ask - trades_bid) / total_trades` ¹ |

¹ **Fixed in v2.1**: Formula uses `(ask - bid)` to follow standard sign convention where `> 0` = BULLISH.
| 9 | aggressive_order_ratio | `trades / (adds + trades)` |
| 10 | order_flow_volatility | Std dev of net flow in sub-windows |
| 11 | flow_regime_indicator | `fast_flow / slow_flow` ratio |

**Category 2: Size Distribution (8 features)**

| Index | Feature | Description |
|-------|---------|-------------|
| 12 | size_p25 | 25th percentile order size |
| 13 | size_p50 | Median order size |
| 14 | size_p75 | 75th percentile order size |
| 15 | size_p90 | 90th percentile order size |
| 16 | size_zscore | Z-score of last order size |
| 17 | large_order_ratio | Proportion of orders > p90 |
| 18 | size_skewness | Distribution skewness |
| 19 | size_concentration | Herfindahl index of sizes |

**Category 3: Queue Position & Depth (6 features)**

| Index | Feature | Description |
|-------|---------|-------------|
| 20 | avg_queue_position | Average position in price level |
| 21 | queue_size_ahead | Volume ahead of tracked orders |
| 22 | orders_per_level | `active_orders / active_levels` |
| 23 | level_concentration | Volume in top N levels ratio |
| 24 | depth_ticks_bid | Weighted depth on bid side |
| 25 | depth_ticks_ask | Weighted depth on ask side |

**Category 4: Institutional Detection (4 features)**

| Index | Feature | Description |
|-------|---------|-------------|
| 26 | large_order_frequency | Large orders per second |
| 27 | large_order_imbalance | `(large_bid - large_ask) / total_large` |
| 28 | modification_score | Average modifications per order |
| 29 | iceberg_proxy | `fill_ratio × modification_score` |

**Category 5: Core MBO Metrics (6 features)**

| Index | Feature | Description |
|-------|---------|-------------|
| 30 | avg_order_age | Average age of active orders (seconds) |
| 31 | median_order_lifetime | **Placeholder (always 0.0)** - pending implementation |
| 32 | avg_fill_ratio | Average `filled_size / original_size` |
| 33 | avg_time_to_first_fill | Average seconds to first execution |
| 34 | cancel_to_add_ratio | `cancels / adds` ratio |
| 35 | active_order_count | Number of tracked active orders |

#### Usage Pattern

```rust
let mut aggregator = MboAggregator::new();

// Process each MBO event
for msg in mbo_messages {
    let event = MboEvent::from_mbo_message(&msg);
    aggregator.process_event(event);  // O(1) amortized
}

// Extract features (typically at sampling points)
let mbo_features: Vec<f64> = aggregator.extract_features(&lob_state);
assert_eq!(mbo_features.len(), 36);
```

---

### 5D. Trading Signals (14 features)

> **Design Philosophy**: These signals are **model-agnostic input features**, not a trading system.
> They provide high-frequency market microstructure information that models can learn to use.
> Entry/exit thresholds are NOT hard-coded—let the model learn what works for each symbol/regime.

#### Purpose

The 14 trading signals serve as **rich input features** for ML experimentation:

| Use Case | Description |
|----------|-------------|
| **Feature Engineering** | Higher-level features derived from base LOB/MBO data |
| **Model Input** | Directional and safety signals for any model architecture |
| **Experimentation** | Enable comparison of different feature combinations |
| **NOT** | A prescriptive trading system with fixed thresholds |

**Recommended approach**: Train models on all 98 features and let the model learn:
- Which features are predictive for your symbol
- What thresholds work for your time horizon
- How to combine signals optimally

#### Implementation

| File | Module |
|------|--------|
| `feature-extractor-MBO-LOB/src/features/signals.rs` | `OfiComputer`, `compute_signals()`, `TimeRegime` |
| `feature-extractor-MBO-LOB/src/pipeline.rs` | Signal integration into pipeline |

#### Research Foundation

The trading signals are based on peer-reviewed research:

| Signal | Research Paper | Key Insight |
|--------|----------------|-------------|
| `true_ofi`, `depth_norm_ofi` | Cont et al. (2014) "The Price Impact of Order Book Events" | OFI = Σ(bid_change - ask_change); β ∝ 1/avg_depth |
| `signed_mp_delta_bps` | Stoikov (2018) "The Micro-Price" | Volume-weighted mid-price as high-frequency estimator |
| `time_regime` | Cont et al. §3.3 | Price impact is 5× higher at open vs close |

#### Architecture

The signal layer is a **thin computation layer** that:
1. Receives the 84 base features (LOB + Derived + MBO) as input
2. Maintains streaming state via `OfiComputer` (updated on every LOB transition)
3. Computes 14 higher-level trading signals at sampling points
4. Outputs a fixed-size signal vector appended to the feature buffer

```rust
// In Pipeline::process_messages()
// 1. Update OFI on EVERY LOB state transition (not just at sampling)
if let Some(ref mut ofi) = self.ofi_computer {
    ofi.update(&lob_state);
}

// 2. At sampling points, sample OFI and compute all signals
if should_sample {
    let ofi_sample = ofi.sample_and_reset(timestamp);
    let signals = signals::compute_signals(&features, &ofi_sample, timestamp, invalidity_delta);
    feature_buffer.extend_from_slice(signals.as_slice());
}
```

#### OfiComputer (Streaming OFI)

```rust
pub struct OfiComputer {
    // State tracking
    prev_best_bid: Option<i64>,
    prev_best_ask: Option<i64>,
    prev_best_bid_size: u32,
    prev_best_ask_size: u32,
    
    // OFI accumulators (since last sample)
    ofi_bid: i64,        // Σ bid_size_change
    ofi_ask: i64,        // Σ ask_size_change
    depth_sum: u64,      // For average depth calculation
    depth_count: u64,
    
    // Warmup tracking
    state_changes_since_reset: u64,  // Must reach MIN_WARMUP_STATE_CHANGES (100)
    
    // Timing
    last_sample_timestamp: i64,
}

impl OfiComputer {
    /// Update on every LOB state transition (not just at sampling)
    pub fn update(&mut self, lob_state: &LobState);
    
    /// Sample current OFI and reset accumulators
    pub fn sample_and_reset(&mut self, timestamp: i64) -> OfiSample;
    
    /// Check if warmup period is complete
    pub fn is_warm(&self) -> bool;
    
    /// Reset on Action::Clear or day boundary
    pub fn reset_on_clear(&mut self);
}
```

**Critical**: `update()` must be called on **every** LOB state transition, not just at sampling points. This is essential for accurate OFI accumulation per Cont et al.

#### OfiSample Output

```rust
pub struct OfiSample {
    pub ofi_bid: i64,          // Bid size change since last sample
    pub ofi_ask: i64,          // Ask size change since last sample
    pub true_ofi: i64,         // ofi_bid - ofi_ask (Cont definition)
    pub avg_depth: f64,        // Average total depth over interval
    pub depth_norm_ofi: f64,   // true_ofi / avg_depth
    pub dt_seconds: f64,       // Time since last sample
    pub is_warm: bool,         // Warmup status
}
```

#### Time Regime (UTC → ET Conversion)

```rust
pub enum TimeRegime {
    Open   = 0,  // 9:30-9:45 ET  - Highest volatility
    Early  = 1,  // 9:45-10:30 ET - Settling period
    Midday = 2,  // 10:30-15:30 ET - Most stable
    Close  = 3,  // 15:30-16:00 ET - Position squaring
    Closed = 4,  // Outside market hours
}
```

Handles DST using calendar-based approximation:
- DST begins: ~March 10-14 (second Sunday)
- DST ends: ~November 1-7 (first Sunday)

#### Signal Index Mapping (indices 84-97)

| Index | Signal | Formula | Range | Category |
|-------|--------|---------|-------|----------|
| 84 | `true_ofi` | `Σ(bid_change - ask_change)` | unbounded | Direction |
| 85 | `depth_norm_ofi` | `true_ofi / avg_depth` | unbounded | Direction |
| 86 | `executed_pressure` | `trade_rate_ask - trade_rate_bid` | unbounded | Direction |
| 87 | `signed_mp_delta_bps` | `(microprice - mid) / mid × 10000` | ~[-100, 100] | Timing |
| 88 | `trade_asymmetry` | `(buys - sells) / (buys + sells)` | [-1, 1] | Confirmation |
| 89 | `cancel_asymmetry` | `(cancel_ask - cancel_bid) / total` | [-1, 1] | Confirmation |
| 90 | `fragility_score` | `level_concentration / ln(avg_depth)` ¹ | [0, ∞) | Impact |
| 91 | `depth_asymmetry` | `(depth_ticks_bid - depth_ticks_ask) / total` ² | [-1, 1] | Impact |
| 92 | `book_valid` | `bid > 0 && ask > 0 && bid < ask` | {0, 1} | Safety |
| 93 | `time_regime` | `compute_time_regime(timestamp)` | {0, 1, 2, 3, 4} | Timing |
| 94 | `mbo_ready` | `state_changes >= 100` | {0, 1} | Safety |
| 95 | `dt_seconds` | Time since last sample | [0, ∞) | Meta |
| 96 | `invalidity_delta` | `crossed_quotes_delta + locked_quotes_delta` | [0, ∞) | Meta |
| 97 | `schema_version` | Always `2.1` | {2.1} | Meta |

¹ `fragility_score`: Falls back to `level_concentration` when `avg_depth ≤ 1`. Higher values = more fragile book.

² `depth_asymmetry`: Uses `depth_ticks_*` = volume-weighted average distance from BBO in ticks (NOT raw volume). Measures liquidity positioning, not volume imbalance.

#### Signal Categories and Usage

**Safety Gates (must pass before any trading)**:
- `book_valid`: Binary flag - book has valid bid/ask with bid < ask
- `mbo_ready`: Binary flag - OFI warmup complete (≥100 state changes)

**Direction Signals (predict price movement)**:
- `true_ofi`: Raw order flow imbalance (+ = buy pressure)
- `depth_norm_ofi`: OFI normalized by depth (cross-stock comparable)
- `executed_pressure`: Trade-based directional signal

**Confirmation Signals (validate direction)**:
- `trade_asymmetry`: Trade count imbalance
- `cancel_asymmetry`: Cancellation imbalance

**Impact Signals (measure market stability)**:
- `fragility_score`: Book fragility (concentration / ln(depth))
- `depth_asymmetry`: Liquidity positioning asymmetry (NOT volume)

**Timing Signals (when to trade)**:
- `signed_mp_delta_bps`: Microprice momentum
- `time_regime`: Market session for threshold adjustment

**Meta Signals (data quality)**:
- `dt_seconds`: For rate normalization
- `invalidity_delta`: Feed quality indicator
- `schema_version`: API versioning

#### Enabling Trading Signals

```rust
use feature_extractor::prelude::*;

let pipeline = PipelineBuilder::new()
    .lob_levels(10)
    .with_derived_features()  // Required
    .with_mbo_features()      // Required
    .with_trading_signals()   // Adds 14 signals
    .build()?;

assert_eq!(pipeline.config().features.feature_count(), 98);
```

**Note**: Trading signals require both derived and MBO features because:
- `signed_mp_delta_bps` uses `weighted_mid_price` (derived index 46)
- `trade_asymmetry` uses `trade_rate_*` (MBO indices 52-53)
- `cancel_asymmetry` uses `cancel_rate_*` (MBO indices 50-51)
- `fragility_score` uses `level_concentration` (MBO index 71)

---

### 5E. Feature Configuration

#### Implementation

| File | Struct |
|------|--------|
| `feature-extractor-MBO-LOB/src/features/mod.rs` | `FeatureConfig` |

#### Configuration Options

```rust
pub struct FeatureConfig {
    pub lob_levels: usize,       // LOB depth (default: 10)
    pub tick_size: f64,          // Price tick size (default: 0.01)
    pub include_derived: bool,   // Add 8 derived features
    pub include_mbo: bool,       // Add 36 MBO features
    pub mbo_window_size: usize,  // MBO aggregation window (default: 1000)
    pub include_signals: bool,   // Add 14 trading signals
}

impl FeatureConfig {
    pub const DERIVED_FEATURE_COUNT: usize = 8;
    pub const MBO_FEATURE_COUNT: usize = 36;
    pub const SIGNAL_FEATURE_COUNT: usize = 14;

    pub fn feature_count(&self) -> usize {
        let base = self.lob_levels * 4;  // 40 raw LOB features
        let derived = if self.include_derived { 8 } else { 0 };
        let mbo = if self.include_mbo { 36 } else { 0 };
        let signals = if self.include_signals { 14 } else { 0 };
        base + derived + mbo + signals
    }
}
```

#### Feature Configurations

| Configuration | Features | Use Case |
|---------------|----------|----------|
| `FeatureConfig::default()` | 40 | Standard DeepLOB/TLOB input |
| `with_derived(true)` | 48 | LOB + market microstructure |
| `with_mbo(true)` | 76 | LOB + order flow analysis |
| `with_derived(true).with_mbo(true)` | 84 | Full base feature set |
| `with_derived(true).with_mbo(true).with_signals(true)` | 98 | Full feature set + trading signals |

#### Combined Feature Layout

When all features are enabled (98 total):

```
Index Range | Feature Type           | Count
------------|------------------------|-------
0-9         | ask_prices (raw)       | 10
10-19       | ask_sizes (raw)        | 10
20-29       | bid_prices (raw)       | 10
30-39       | bid_sizes (raw)        | 10
40-47       | derived features       | 8
48-83       | MBO features           | 36
84-97       | trading signals        | 14
```

---

## 6. Stage 5: Sequence Building

### Implementation

| File | Module |
|------|--------|
| `feature-extractor-MBO-LOB/src/sequence_builder/builder.rs` | `SequenceBuilder` |

### Configuration (NVDA Export)

```rust
SequenceConfig {
    window_size: 100,      // Snapshots per sequence
    stride: 10,            // Skip between sequences
    max_buffer_size: 50000, // Large buffer for full-day processing
    feature_count: 40,     // Must match FeatureConfig
}
```

### Behavior

Maintains a circular buffer of feature vectors. For each new sample:

1. Push `Arc<Vec<f64>>` to buffer (zero-copy sharing)
2. Call `try_build_sequence()` to check if a new sequence is ready
3. If buffer has ≥ `window_size` samples since last emit, build sequence

### Sequence Generation Formula

```
Sequence i uses samples at buffer positions:
  [i * stride, i * stride + 1, ..., i * stride + window_size - 1]

With stride=10, window_size=100:
  Sequence 0: samples [0, 1, ..., 99]
  Sequence 1: samples [10, 11, ..., 109]
  Sequence 2: samples [20, 21, ..., 119]
  ...
```

### Output Schema: `Sequence`

```rust
pub struct Sequence {
    pub features: Vec<Arc<Vec<f64>>>,  // [window_size × feature_count]
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ns: u64,
    pub length: usize,  // Always == window_size
}
```

### Typical Day Output

- ~10K-18K samples per day
- → ~1K-1.7K sequences per day (with stride=10)

---

## 7. Stage 6: Label Generation

The pipeline supports two labeling methods, each with different characteristics:

| Method | Source | Horizon Bias | Use Case |
|--------|--------|--------------|----------|
| TLOB | `TlobLabelGenerator` | No | Production, research |
| DeepLOB | `DeepLobLabelGenerator` | Yes | Benchmarking, simplicity |

---

### 7A. TLOB Labeling Method

#### Implementation

| File | Module |
|------|--------|
| `feature-extractor-MBO-LOB/src/labeling/tlob.rs` | `TlobLabelGenerator` |
| `feature-extractor-MBO-LOB/src/labeling/multi_horizon.rs` | `MultiHorizonLabelGenerator` |

#### Configuration (NVDA Export)

```rust
LabelConfig {
    horizon: 50,           // Predict 50 samples ahead
    smoothing_window: 10,  // Average over 10 samples
    threshold: 0.0008,     // 8 basis points
}

MultiHorizonConfig {
    horizons: vec![10, 20, 50, 100, 200],  // Configurable via ExportLabelConfig
    smoothing_window: 10,
    threshold_strategy: ThresholdStrategy::Fixed(0.0008),
}
```

#### Formula

Given mid-prices `p(t)` at sample index `t`:

**1. Past smoothed mid-price** (centered at `t`):
```
w⁻(t,h,k) = (1/(k+1)) × Σᵢ₌₀ᵏ p(t-i)
```

**2. Future smoothed mid-price** (centered at `t+h`):
```
w⁺(t,h,k) = (1/(k+1)) × Σᵢ₌₀ᵏ p(t+h-i)
```

**3. Percentage change**:
```
l(t,h,k) = (w⁺(t,h,k) - w⁻(t,h,k)) / w⁻(t,h,k)
```

**4. Classification**:
```
label = Up     if l > θ
        Down   if l < -θ
        Stable otherwise
```

Where:
- `h` = horizon (e.g., 50 samples)
- `k` = smoothing_window (e.g., 10 samples, independent of h)
- `θ` = threshold (e.g., 0.0008 = 8 bps)

#### Valid Label Range

```
min_required = k + h + k + 1 = 10 + 50 + 10 + 1 = 71 samples

First valid label: index k = 10
Last valid label: index (N - h - 1) where N = total samples
```

---

### 7B. DeepLOB Labeling Method

#### Implementation

| File | Module |
|------|--------|
| `feature-extractor-MBO-LOB/src/labeling/deeplob.rs` | `DeepLobLabelGenerator` |

#### Key Difference from TLOB

In DeepLOB, the smoothing window size **equals** the prediction horizon (`k = h`), creating "horizon bias" where longer prediction horizons use more smoothing.

#### Method Variants

```rust
pub enum DeepLobMethod {
    /// Method 1: Compare future average to current price
    /// l_t = (m+(t) - p_t) / p_t
    VsCurrentPrice,  // Default

    /// Method 2: Compare future average to past average (like FI-2010)
    /// l_t = (m+(t) - m-(t)) / m-(t)
    VsPastAverage,
}
```

#### Formulas

**Method 1 (VsCurrentPrice)** - Original DeepLOB paper:
```
Future average: m⁺(t) = (1/k) × Σᵢ₌₁ᵏ p(t+i)
Percentage change: l_t = (m⁺(t) - p_t) / p_t
```

**Method 2 (VsPastAverage)** - Similar to FI-2010:
```
Past average: m⁻(t) = (1/k) × Σᵢ₌₀ᵏ⁻¹ p(t-i)
Future average: m⁺(t) = (1/k) × Σᵢ₌₁ᵏ p(t+i)
Percentage change: l_t = (m⁺(t) - m⁻(t)) / m⁻(t)
```

#### Valid Label Range

| Method | Minimum Prices | First Valid Index | Last Valid Index |
|--------|----------------|-------------------|------------------|
| VsCurrentPrice | k + 1 | 0 | N - k - 1 |
| VsPastAverage | 2k | k - 1 | N - k - 1 |

#### Usage Example

```rust
let config = LabelConfig {
    horizon: 10,       // k = 10
    smoothing_window: 10,  // Ignored in DeepLOB (uses horizon)
    threshold: 0.002,
};

// Method 1 (default)
let gen1 = DeepLobLabelGenerator::new(config.clone());

// Method 2
let gen2 = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);

gen1.add_prices(&prices);
let labels = gen1.generate_labels()?;
```

#### Comparison: TLOB vs DeepLOB

| Aspect | TLOB | DeepLOB |
|--------|------|---------|
| Parameters | h (horizon), k (smoothing) | k only (horizon = smoothing) |
| Horizon bias | No | Yes |
| Flexibility | High | Low |
| Complexity | More complex | Simpler |
| Recommended use | Production | Benchmarking |

---

### 7C. Threshold Strategies

#### Implementation

| File | Enum |
|------|------|
| `feature-extractor-MBO-LOB/src/labeling/multi_horizon.rs` | `ThresholdStrategy` |

#### Available Strategies

```rust
pub enum ThresholdStrategy {
    /// Fixed threshold (e.g., 8 bps)
    Fixed(f64),
    
    /// Rolling spread-based threshold
    /// threshold = spread × multiplier (adapts to market conditions)
    RollingSpread {
        window_size: usize,    // Rolling window for spread calculation
        multiplier: f64,       // e.g., 0.5 × spread
        fallback: f64,         // Used when spread unavailable
    },
    
    /// Quantile-based threshold
    /// threshold set so target_proportion of labels are non-stationary
    Quantile {
        target_proportion: f64,  // e.g., 0.33 for ~33% Up + Down
        window_size: usize,
        fallback: f64,
    },
}
```

#### Strategy Selection Guide

| Strategy | When to Use | Characteristics |
|----------|-------------|-----------------|
| `Fixed` | Known threshold, reproducibility | Simple, consistent |
| `RollingSpread` | Adaptive to volatility | Wider threshold in volatile markets |
| `Quantile` | Balanced class distribution | Targets specific Up/Down proportion |

#### Configuration Examples

```rust
// Fixed threshold (standard)
MultiHorizonConfig::new(vec![10, 20, 50, 100], 10, ThresholdStrategy::Fixed(0.0008));

// Spread-adaptive threshold
MultiHorizonConfig::new(vec![10, 20, 50, 100], 10, ThresholdStrategy::RollingSpread {
    window_size: 100,
    multiplier: 0.5,
    fallback: 0.0008,
});

// Quantile-based (33% Up + Down)
MultiHorizonConfig::new(vec![10, 20, 50, 100], 10, ThresholdStrategy::Quantile {
    target_proportion: 0.33,
    window_size: 1000,
    fallback: 0.0008,
});
```

---

### Output Schema

```rust
// Single-horizon
Vec<(usize, TrendLabel, f64)>  // (sample_index, label, pct_change)

// Multi-horizon
labels: BTreeMap<usize, Vec<(usize, TrendLabel, f64)>>  // horizon → [(index, label, change)]
```

### Label Encoding

| TrendLabel | i8 value | class_index (PyTorch) |
|------------|----------|----------------------|
| Down | -1 | 0 |
| Stable | 0 | 1 |
| Up | 1 | 2 |

---

## 8. Stage 7: Sequence-Label Alignment

### Implementation

| File | Function |
|------|----------|
| `feature-extractor-MBO-LOB/src/export_aligned.rs` | `align_sequences_with_labels()` |

### Alignment Formula

**Critical invariant**: A sequence predicts the future from its **ending snapshot**.

```rust
// For sequence at index seq_idx:
ending_idx = seq_idx * stride + window_size - 1

// Example with stride=10, window_size=100:
//   Sequence 0: snapshots [0..99]   → ending_idx = 0*10 + 100 - 1 = 99
//   Sequence 1: snapshots [10..109] → ending_idx = 1*10 + 100 - 1 = 109
//   Sequence 2: snapshots [20..119] → ending_idx = 2*10 + 100 - 1 = 119
```

The label for sequence `i` is the label computed at `ending_idx`.

### Dropped Sequences

Sequences are dropped if:
1. Their `ending_idx` has no valid label (near end of day)
2. The label generator couldn't compute a label (insufficient future data)

Typical drop rate: 5-15% of sequences at day boundaries.

---

## 9. Stage 8: Per-Day Normalization

### Implementation

| File | Function |
|------|----------|
| `feature-extractor-MBO-LOB/src/export_aligned.rs` | `normalize_sequences()` |

### Strategy: `market_structure_zscore`

This normalization preserves the market structure invariant that `ask_price > bid_price` at each level.

### Algorithm

**For prices** (features 0-9 and 20-29):
- Compute mean and std for each level by **pooling ask and bid prices together**
- Apply the same mean/std to both ask and bid at that level

```rust
// For each level L in 0..10:
values = [all ask_price_L across all samples] ∪ [all bid_price_L across all samples]
mean_L = mean(values)
std_L = std(values)

// Normalize:
ask_price_L_normalized = (ask_price_L - mean_L) / std_L
bid_price_L_normalized = (bid_price_L - mean_L) / std_L
```

**For sizes** (features 10-19 and 30-39):
- Each size feature has its own independent mean/std

```rust
// For each size feature i in 0..20:
values = [all size_i across all samples]
mean_i = mean(values)
std_i = std(values)

size_i_normalized = (size_i - mean_i) / std_i
```

### Epsilon Handling

```rust
let epsilon = 1e-8;
let std = if variance > epsilon { variance.sqrt() } else { 1.0 };
```

### Output: `NormalizationParams`

```json
{
  "strategy": "market_structure_zscore",
  "price_means": [120.50, 120.51, 120.52, ...],   // 10 values (per level, shared ask+bid)
  "price_stds": [0.05, 0.06, 0.07, ...],          // 10 values
  "size_means": [150.0, 145.0, ...],              // 20 values (10 ask + 10 bid)
  "size_stds": [50.0, 48.0, ...],                 // 20 values
  "sample_count": 180000,                          // n_sequences × window_size
  "feature_layout": "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10",
  "levels": 10
}
```

### Post-Normalization Invariants (per day)

- Each feature has mean ≈ 0, std ≈ 1 within that day
- `ask_price_L_norm > bid_price_L_norm` is preserved (both shifted by same amount)

---

## 10. Stage 9: Export Artifacts

### Configuration-Driven Export System

The pipeline now supports a **configuration-driven export system** via `DatasetConfig`. This enables:

- **Symbol-agnostic exports**: Works for any instrument (NVDA, AAPL, etc.)
- **Flexible feature sets**: Export 40, 48, 76, 84, or 98 features
- **TOML/JSON configuration**: Reproducible experiment configurations
- **Date range handling**: Automatic weekend exclusion, date list generation
- **Train/val/test splits**: Configurable split ratios

### DatasetConfig Structure

```rust
pub struct DatasetConfig {
    pub symbol: SymbolConfig,          // Symbol name, exchange, filename pattern
    pub data: DataPathConfig,          // Input/output directories, hot store
    pub dates: DateRangeConfig,        // Date range with weekend exclusion
    pub features: FeatureSetConfig,    // Feature selection (derived, MBO, signals)
    pub sampling: ExportSamplingConfig, // Event-based or volume-based sampling
    pub sequence: ExportSequenceConfig, // Window size, stride, buffer
    pub labels: ExportLabelConfig,     // Horizon, smoothing, threshold
    pub split: SplitConfig,            // Train/val/test ratios
    pub processing: ProcessingConfig,  // Threads, error mode
}
```

### Sample Configuration (TOML)

```toml
# configs/nvda_98feat.toml - 98-feature export for NVIDIA

[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
tick_size = 0.01

[data]
input_dir = "../data/databento/NVDA"
output_dir = "../data/exports/nvda_98feat"
hot_store_dir = "../data/hot_store/NVDA"

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

### CLI Export Tool

```bash
# Generate template configuration
cargo run --release --bin export_dataset -- --generate-config configs/my_dataset.toml

# Execute export
cargo run --release --features parallel --bin export_dataset -- --config configs/nvda_98feat.toml
```

### Directory Structure

```
data/exports/nvda_98feat/
├── dataset_manifest.json           # Split lists, config, horizons
├── train/
│   ├── 2025-02-03_sequences.npy    # (N, 100, 98) float32
│   ├── 2025-02-03_labels.npy       # (N, num_horizons) int8 for multi-horizon
│   ├── 2025-02-03_horizons.json    # Horizon values [10, 20, 50, 100, 200]
│   ├── 2025-02-03_normalization.json
│   ├── 2025-02-03_metadata.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### File Schemas

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `*_sequences.npy` | `(N, 100, F)` | `float32` | Normalized features (F=40-98 based on config) |
| `*_labels.npy` | `(N,)` or `(N, H)` | `int8` | Labels: 1D single-horizon, 2D multi-horizon |
| `*_horizons.json` | — | JSON | Horizon values (multi-horizon exports only) |
| `*_normalization.json` | — | JSON | Normalization parameters |
| `*_metadata.json` | — | JSON | Export metadata, horizons, validation info |

### dataset_manifest.json Schema

```json
{
  "version": "1.0",
  "symbol": "NVDA",
  "split": {
    "train_days": ["2025-02-03", "2025-02-04", ...],
    "val_days": ["2025-04-14", "2025-04-15", ...],
    "test_days": ["2025-05-01", "2025-05-02", ...]
  },
  "sequence_length": 100,
  "stride": 10,
  "n_features": 40,
  "n_levels": 10,
  "normalization": "market_structure_zscore",
  "horizons": [10, 20, 50, 100, 200],
  "num_horizons": 5,
  "labeling": {
    "method": "tlob",
    "smoothing_window": 10,
    "threshold": 0.0008
  }
}
```

---

## 11. Stage 10: Python Data Loading

### Implementation

| File | Class |
|------|-------|
| `lob-model-trainer/src/lob_trainer/data/sources/aligned_source.py` | `AlignedExportSource` |
| `lob-model-trainer/src/lob_trainer/data/sources/multi_day_source.py` | `MultiDayAlignedSource` |
| `lob-model-trainer/src/lob_trainer/data/datamodule.py` | `DataModule` |

### Loading Sequences

```python
sequences = np.load(f"{day}_sequences.npy", mmap_mode="r")
# Shape: (N, 100, 40), dtype: float32
```

### Loading Labels

```python
labels = np.load(f"{day}_labels.npy", mmap_mode="r")
# Shape depends on export mode:
# - Single-horizon: (N,) int8
# - Multi-horizon: (N, num_horizons) int8
# Values: {-1, 0, 1}

# Check if multi-horizon from metadata
with open(f"{day}_metadata.json") as f:
    meta = json.load(f)
is_multi_horizon = 'horizons' in meta and meta.get('num_horizons', 1) > 1

# For multi-horizon, select specific horizon
if is_multi_horizon:
    horizons = meta['horizons']  # e.g., [10, 20, 50, 100, 200]
    horizon_idx = horizons.index(50)  # Get index for horizon=50
    labels_h50 = labels[:, horizon_idx]  # Shape: (N,)
```

### Python DayData Multi-Horizon Support

The Python data loaders (`lobtrainer.data.dataset.DayData` and `lobtrainer.analysis.streaming.DayData`) support multi-horizon labels:

```python
# DayData properties
day.is_multi_horizon  # bool: True if labels are 2D
day.num_horizons      # int: Number of horizons (1 for single-horizon)
day.horizons          # List[int] or None: Horizon values from metadata

# Get labels for specific horizon (works with both single/multi-horizon)
labels = day.get_labels(horizon_idx=0)  # First horizon (or all labels if single-horizon)
labels = day.get_labels(horizon_idx=None)  # All horizons (returns 2D array for multi-horizon)
```

### Label Conversion to Class Index

```python
# Convert {-1, 0, 1} → {0, 1, 2} for PyTorch CrossEntropyLoss
class_indices = labels.astype(np.int64) + 1
# -1 (Down)   → 0
#  0 (Stable) → 1
#  1 (Up)     → 2
```

### DataModule Output

```python
# train_dataloader() yields:
x: torch.Tensor  # shape (batch, 100, 40), dtype float32
y: torch.Tensor  # shape (batch,), dtype int64, values in {0, 1, 2}
```

---

## 12. Stage 11: Feature Layout Transformation

### Implementation

| File | Class |
|------|-------|
| `lob-model-trainer/src/lob_trainer/data/transforms/feature_reorder.py` | `FeatureReorderTransform` |

### When Needed

Models declare their `required_layout` in the registry:
- TLOB/TLOBv2: `FeatureLayout.INTERLEAVED`
- Our Rust export: `FeatureLayout.GROUPED`

If mismatch, `ExperimentRunner` applies `FeatureReorderTransform`.

### GROUPED → INTERLEAVED Transformation

```python
def reorder_features_to_deeplob(x, levels=10):
    """
    GROUPED:     [ask_p(10), ask_v(10), bid_p(10), bid_v(10)]
    INTERLEAVED: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...]
    """
    # Reshape: (..., 40) → (..., 4, 10)
    x_reshaped = x.reshape(*x.shape[:-1], 4, levels)
    # Transpose: (..., 4, 10) → (..., 10, 4)
    x_transposed = np.swapaxes(x_reshaped, -2, -1)
    # Flatten: (..., 10, 4) → (..., 40)
    return x_transposed.reshape(*x.shape[:-1], 40)
```

### Feature Index Mapping

| GROUPED Index | Feature | INTERLEAVED Index |
|---------------|---------|-------------------|
| 0 | ask_price_1 | 0 |
| 1 | ask_price_2 | 4 |
| ... | ... | ... |
| 10 | ask_size_1 | 1 |
| 11 | ask_size_2 | 5 |
| ... | ... | ... |
| 20 | bid_price_1 | 2 |
| 21 | bid_price_2 | 6 |
| ... | ... | ... |
| 30 | bid_size_1 | 3 |
| 31 | bid_size_2 | 7 |
| ... | ... | ... |

---

## 13. Stage 12: Train-Stats Renormalization (P0.2)

### Purpose

The Rust exporter normalizes each day independently. The original TLOB repo normalizes all data using train-set global statistics, creating distribution shifts that BiN adapts to.

### Implementation

| File | Class |
|------|-------|
| `lob-model-trainer/src/lob_trainer/data/normalization.py` | `NormalizationManager`, `RenormTransform` |

### Renormalization Formula

Convert from per-day normalized to train-stats normalized:

```
x_train_norm = x_per_day × (std_day / std_train) + (mean_day - mean_train) / std_train
```

Where:
- `x_per_day`: Data with per-day normalization (mean≈0, std≈1)
- `mean_day`, `std_day`: Per-day normalization parameters
- `mean_train`, `std_train`: Aggregated train-set statistics

### Resolved Issues (P0.2.1, P0.2.2) — Fixed 2025-12-15

**P0.2.1: Pooled variance aggregation — ✅ FIXED**

Now uses correct pooled variance formula that includes between-day mean drift:

```python
# CORRECT (now implemented in NormalizationManager._aggregate()):
pooled_var = (Σ(n_day × var_day) + Σ(n_day × (mean_day - mean_global)²)) / Σ(n_day)
```

**P0.2.2: Transform composition — ✅ FIXED**

Transforms are now correctly composed using `ComposeTransform`:

```python
# ExperimentRunner now composes transforms:
dataset.transform = compose_transforms(existing_renorm, layout_transform)

# Order enforced:
# 1. RenormTransform (operates on GROUPED layout, uses GROUPED stats)
# 2. FeatureReorderTransform (GROUPED → INTERLEAVED)
```

---

## 14. Data Type Summary

### Rust Types

| Stage | Type | Memory |
|-------|------|--------|
| DBN decode | `dbn::MboMsg` | ~120 bytes |
| Internal message | `MboMessage` | 32 bytes |
| LOB state | `LobState` | ~560 bytes (stack) |
| Feature vector (40 feat) | `Arc<Vec<f64>>` | 8 + 40×8 = 328 bytes |
| Feature vector (84 feat) | `Arc<Vec<f64>>` | 8 + 84×8 = 680 bytes |
| Feature vector (98 feat) | `Arc<Vec<f64>>` | 8 + 98×8 = 792 bytes |
| Sequence (40 feat) | `Sequence` | 100 × 8 + metadata ≈ 848 bytes |
| MBO aggregator | `MboAggregator` | ~8 MB per symbol (3 windows + tracker) |
| OFI computer | `OfiComputer` | ~200 bytes (streaming state) |

### Numpy Types

| Array | Dtype | Shape | Size per day |
|-------|-------|-------|--------------|
| sequences (40 features) | `float32` | `(N, 100, 40)` | N × 16 KB |
| sequences (84 features) | `float32` | `(N, 100, 84)` | N × 33.6 KB |
| sequences (98 features) | `float32` | `(N, 100, 98)` | N × 39.2 KB |
| labels | `int8` | `(N, 4)` | N × 4 bytes |

### PyTorch Types

| Tensor | Dtype | Shape |
|--------|-------|-------|
| features (40) | `torch.float32` | `(batch, 100, 40)` |
| features (84) | `torch.float32` | `(batch, 100, 84)` |
| features (98) | `torch.float32` | `(batch, 100, 98)` |
| labels | `torch.int64` | `(batch,)` |

---

## 15. Known Issues & Contracts

### Critical Contracts

1. **Alignment**: `len(sequences) == len(labels)` — enforced by export validation
2. **Feature count**: Sequence `shape[-1]` must equal model `num_features` (40)
3. **Label values**: Must be in `{-1, 0, 1}` (Rust export) or `{0, 1, 2}` (Python class index)
4. **Layout**: Model's `required_layout` must match data layout (or transform applied)
5. **BiN + renorm**: If `use_train_stats_normalization=True`, then `bypass_bin` must be `False`

### Resolved Bugs (tracked in TLOB_WORK.md)

| ID | Issue | Status | Fixed |
|----|-------|--------|-------|
| P0.2.1 | Train std aggregation ignores between-day drift | ✅ Fixed | 2025-12-15 |
| P0.2.2 | Transform composition can drop renorm | ✅ Fixed | 2025-12-15 |

### Validation Checks (export-time)

1. No NaN/Inf in sequences
2. Labels subset of `{-1, 0, 1}`
3. Spread integrity: `ask_price - bid_price > 0` for >99% of samples
4. Shape consistency within day

---

## 16. File Reference Index

### MBO-LOB-reconstructor

| Path | Purpose |
|------|---------|
| `src/loader.rs` | `DbnLoader` - DBN file streaming |
| `src/dbn_bridge.rs` | `DbnBridge::convert()` - DBN → MboMessage |
| `src/types.rs` | `MboMessage`, `LobState`, `Action`, `Side`, analytics methods |
| `src/lob/reconstructor.rs` | `LobReconstructor` - LOB reconstruction |
| `src/lob/price_level.rs` | `PriceLevel` - Order aggregation at price |

### feature-extractor-MBO-LOB

| Path | Purpose |
|------|---------|
| `src/features/lob_features.rs` | `extract_raw_features()`, `extract_normalized_features()` |
| `src/features/derived_features.rs` | `compute_derived_features()`, `compute_depth_features()` |
| `src/features/mbo_features.rs` | `MboAggregator`, `MboWindow`, `MboEvent`, `OrderInfo` |
| `src/features/signals.rs` | `OfiComputer`, `compute_signals()`, `TimeRegime`, `OfiSample` |
| `src/features/mod.rs` | `FeatureConfig` - feature set configuration |
| `src/preprocessing/sampling.rs` | `EventBasedSampler`, `VolumeBasedSampler` |
| `src/sequence_builder/builder.rs` | `SequenceBuilder`, `SequenceConfig`, `Sequence` |
| `src/labeling/tlob.rs` | `TlobLabelGenerator` |
| `src/labeling/deeplob.rs` | `DeepLobLabelGenerator`, `DeepLobMethod` |
| `src/labeling/multi_horizon.rs` | `MultiHorizonLabelGenerator`, `ThresholdStrategy` |
| `src/export/mod.rs` | `NumpyExporter`, `BatchExporter`, `DatasetConfig` re-exports |
| `src/export/dataset_config.rs` | `DatasetConfig`, `SymbolConfig`, `FeatureSetConfig`, etc. |
| `src/export_aligned.rs` | `AlignedBatchExporter`, `normalize_sequences()` |
| `tools/export_dataset.rs` | CLI tool for configuration-driven exports |
| `configs/nvda_98feat.toml` | Sample 98-feature NVIDIA configuration |
| `configs/nvda_84feat_baseline.toml` | Sample 84-feature baseline configuration |

### lob-model-trainer

| Path | Purpose |
|------|---------|
| `src/lob_trainer/data/sources/aligned_source.py` | `AlignedExportSource` |
| `src/lob_trainer/data/sources/multi_day_source.py` | `MultiDayAlignedSource` |
| `src/lob_trainer/data/datamodule.py` | `DataModule` |
| `src/lob_trainer/data/transforms/feature_reorder.py` | `FeatureReorderTransform` |
| `src/lob_trainer/data/normalization.py` | `NormalizationManager`, `RenormTransform` |
| `src/lob_trainer/data/validation.py` | `DataValidator`, `ValidationReport` |
| `src/lob_trainer/evaluation/baselines.py` | `BaselineComparator` |
| `src/lob_trainer/experiments/runner.py` | `ExperimentRunner` |

---

## 17. Appendix A: LobState Analytics

### Full Method Reference

The `LobState` struct (`MBO-LOB-reconstructor/src/types.rs`) provides rich analytics beyond basic price/size arrays:

#### Price Analytics

| Method | Signature | Description |
|--------|-----------|-------------|
| `mid_price()` | `Option<f64>` | `(best_bid + best_ask) / 2 / 1e9` |
| `spread()` | `Option<f64>` | `(best_ask - best_bid) / 1e9` (dollars) |
| `spread_bps()` | `Option<f64>` | Spread in basis points |
| `microprice()` | `Option<f64>` | Volume-weighted mid: `(bid×ask_vol + ask×bid_vol) / (ask_vol + bid_vol)` |

#### Volume Analytics

| Method | Signature | Description |
|--------|-----------|-------------|
| `total_bid_volume(n)` | `u64` | Sum of bid sizes across n levels |
| `total_ask_volume(n)` | `u64` | Sum of ask sizes across n levels |
| `depth_imbalance(n)` | `f64` | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` |
| `vwap_bid(n)` | `Option<f64>` | Volume-weighted average bid price |
| `vwap_ask(n)` | `Option<f64>` | Volume-weighted average ask price |
| `weighted_mid(n)` | `Option<f64>` | VWAP-based mid-price |

#### Level Analytics

| Method | Signature | Description |
|--------|-----------|-------------|
| `active_bid_levels()` | `usize` | Count of levels with `price > 0` |
| `active_ask_levels()` | `usize` | Count of levels with `price > 0` |
| `is_valid()` | `bool` | `best_bid < best_ask` |
| `is_crossed()` | `bool` | `best_bid > best_ask` |
| `is_locked()` | `bool` | `best_bid == best_ask` |

#### Temporal Analytics

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `Option<i64>` | Current snapshot timestamp (ns) |
| `previous_timestamp` | `Option<i64>` | Previous snapshot timestamp |
| `delta_ns` | `u64` | Time since last snapshot |
| `triggering_action` | `Option<Action>` | What caused this snapshot |
| `triggering_side` | `Option<Side>` | Which side was affected |
| `sequence` | `u64` | Message sequence number |

---

## 18. Appendix B: Feature Set Summary

### Complete Feature Inventory

#### Raw LOB Features (40)

| Index | Name | Formula | Unit |
|-------|------|---------|------|
| 0-9 | `ask_price_{1-10}` | `ask_prices[i] / 1e9` | USD |
| 10-19 | `ask_size_{1-10}` | `ask_sizes[i]` | shares |
| 20-29 | `bid_price_{1-10}` | `bid_prices[i] / 1e9` | USD |
| 30-39 | `bid_size_{1-10}` | `bid_sizes[i]` | shares |

#### Derived Features (8)

| Index | Name | Formula | Unit/Range |
|-------|------|---------|------------|
| 40 | `mid_price` | `(bid + ask) / 2` | USD |
| 41 | `spread` | `ask - bid` | USD |
| 42 | `spread_bps` | `spread / mid × 10000` | bps |
| 43 | `total_bid_volume` | `Σ bid_sizes` | shares |
| 44 | `total_ask_volume` | `Σ ask_sizes` | shares |
| 45 | `volume_imbalance` | `(bid_vol - ask_vol) / total` | [-1, 1] |
| 46 | `weighted_mid_price` | Volume-weighted mid | USD |
| 47 | `price_impact` | `|mid - weighted_mid|` | USD |

#### MBO Features (36)

**Order Flow (12)**

| Index | Name | Description |
|-------|------|-------------|
| 48 | `add_rate_bid` | Bid adds per second |
| 49 | `add_rate_ask` | Ask adds per second |
| 50 | `cancel_rate_bid` | Bid cancels per second |
| 51 | `cancel_rate_ask` | Ask cancels per second |
| 52 | `trade_rate_bid` | Bid trades per second |
| 53 | `trade_rate_ask` | Ask trades per second |
| 54 | `net_order_flow` | Normalized bid-ask add imbalance |
| 55 | `net_cancel_flow` | Normalized bid-ask cancel imbalance |
| 56 | `net_trade_flow` | Normalized bid-ask trade imbalance |
| 57 | `aggressive_order_ratio` | Market order proportion |
| 58 | `order_flow_volatility` | Flow stability measure |
| 59 | `flow_regime_indicator` | Fast/slow flow ratio |

**Size Distribution (8)**

| Index | Name | Description |
|-------|------|-------------|
| 60 | `size_p25` | 25th percentile |
| 61 | `size_p50` | Median |
| 62 | `size_p75` | 75th percentile |
| 63 | `size_p90` | 90th percentile |
| 64 | `size_zscore` | Last order z-score |
| 65 | `large_order_ratio` | Proportion > p90 |
| 66 | `size_skewness` | Distribution asymmetry |
| 67 | `size_concentration` | Herfindahl index |

**Queue & Depth (6)**

| Index | Name | Description |
|-------|------|-------------|
| 68 | `avg_queue_position` | Mean position in level |
| 69 | `queue_size_ahead` | Volume ahead |
| 70 | `orders_per_level` | Order density |
| 71 | `level_concentration` | Top-level volume ratio |
| 72 | `depth_ticks_bid` | Weighted bid depth |
| 73 | `depth_ticks_ask` | Weighted ask depth |

**Institutional Detection (4)**

| Index | Name | Description |
|-------|------|-------------|
| 74 | `large_order_frequency` | Large orders/sec |
| 75 | `large_order_imbalance` | Large order side bias |
| 76 | `modification_score` | Avg mods per order |
| 77 | `iceberg_proxy` | Hidden order indicator |

**Core MBO (6)**

| Index | Name | Description |
|-------|------|-------------|
| 78 | `avg_order_age` | Mean order lifetime |
| 79 | `median_order_lifetime` | **Placeholder (always 0.0)** |
| 80 | `avg_fill_ratio` | Execution rate |
| 81 | `avg_time_to_first_fill` | Execution speed |
| 82 | `cancel_to_add_ratio` | Cancellation behavior |
| 83 | `active_order_count` | Current order book size |

#### Trading Signals (14)

| Index | Name | Formula/Description | Range |
|-------|------|---------------------|-------|
| 84 | `true_ofi` | Σ(bid_change - ask_change) per Cont et al. | unbounded |
| 85 | `depth_norm_ofi` | `true_ofi / avg_depth` | unbounded |
| 86 | `executed_pressure` | `trade_rate_ask - trade_rate_bid` | unbounded |
| 87 | `signed_mp_delta_bps` | `(microprice - mid) / mid × 10000` | ~[-100, 100] |
| 88 | `trade_asymmetry` | `(buys - sells) / (buys + sells)` | [-1, 1] |
| 89 | `cancel_asymmetry` | `(cancel_ask - cancel_bid) / total` | [-1, 1] |
| 90 | `fragility_score` | `level_concentration / ln(avg_depth)` | [0, ∞) |
| 91 | `depth_asymmetry` | `(depth_ticks_bid - depth_ticks_ask) / total` ¹ | [-1, 1] |
| 92 | `book_valid` | `1` if valid book, `0` otherwise | {0, 1} |
| 93 | `time_regime` | Market session (Open/Early/Midday/Close/Closed) | {0, 1, 2, 3, 4} |
| 94 | `mbo_ready` | `1` if warmup complete, `0` otherwise | {0, 1} |
| 95 | `dt_seconds` | Time since last sample | [0, ∞) |
| 96 | `invalidity_delta` | Quote anomaly count since last sample | [0, ∞) |
| 97 | `schema_version` | Always `2.1` | {2.1} |

**Signal Index Constants Module**:

For programmatic access to signal indices, use `signals::indices`:

```rust
use feature_extractor::features::signals::indices;

let ofi_idx = indices::TRUE_OFI;       // 84
let regime_idx = indices::TIME_REGIME;  // 93
let valid_idx = indices::BOOK_VALID;    // 92
```

¹ `depth_ticks_*` = volume-weighted average distance from BBO in ticks. Measures liquidity positioning, NOT raw volume.

### Feature Configuration Presets

| Preset | Features | Index Range | Use Case |
|--------|----------|-------------|----------|
| DeepLOB Standard | 40 | 0-39 | Original DeepLOB paper |
| TLOB Standard | 40 | 0-39 | Original TLOB paper |
| LOB + Microstructure | 48 | 0-47 | + spread, imbalance |
| LOB + Order Flow | 76 | 0-39, 48-83 | + MBO patterns |
| Full Base Set | 84 | 0-83 | All base features |
| Full + Signals | 98 | 0-97 | Maximum information + trading signals |

### Normalization Considerations

| Feature Type | Normalization Strategy | Notes |
|--------------|----------------------|-------|
| Prices | `market_structure_zscore` | Shared mean/std per level (preserves spread) |
| Sizes | Independent z-score | Each level normalized separately |
| Derived | Standard z-score | Per-feature normalization |
| MBO ratios | Already normalized | Range typically [-1, 1] or [0, 1] |
| MBO rates | Log-transform recommended | Can vary by orders of magnitude |
| Signal asymmetries | Already normalized | Range [-1, 1] (trade/cancel/depth asymmetry) |
| Signal binaries | No normalization needed | {0, 1} (book_valid, mbo_ready) |
| Signal OFI | Optional z-score | Unbounded, varies by stock |

---

*This document reflects the actual implementation as of 2025-12-24. All formulas and code snippets are derived from source files in `MBO-LOB-reconstructor` and `feature-extractor-MBO-LOB`.*
