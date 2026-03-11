# Feature Reference: Complete 116-Feature Index

> **Purpose**: Authoritative single source of truth for all features extracted by the pipeline.
> Look up any feature by index or name to find its exact formula, units, range, normalization behavior, and source file.
> **Schema Version**: 2.2 (defined in `src/contract.rs`)

---

## Contract Constants

| Constant | Value | Source |
|----------|-------|--------|
| `STABLE_FEATURE_COUNT` | 98 | `src/contract.rs` |
| `EXPERIMENTAL_FEATURE_COUNT` | 18 | `src/contract.rs` |
| `FULL_FEATURE_COUNT` | 116 | `src/contract.rs` |
| `LOB_LEVELS` | 10 | `src/contract.rs` |
| `SIGNAL_COUNT` | 14 | `src/contract.rs` |
| `CATEGORICAL_INDICES` | \[92, 93, 94, 97\] | `src/contract.rs` -- must NOT be normalized |
| `FLOAT_CMP_EPS` | 1e-10 | Absolute tolerance for f64 comparisons |
| `DIVISION_GUARD_EPS` | 1e-8 | Added to denominators to prevent division by zero |
| `SCHEMA_VERSION` | 2.2 (f64) | Emitted at index 97 |

---

## Feature Composition Order

Features are assembled in `src/features/extractor.rs`:

1. `extract_into(&lob_state, &mut buf)` appends: **LOB raw** (0-39), then **Derived** (40-47) if enabled, then **MBO** (48-83) if enabled.
2. `extract_with_signals()` calls `extract_into()` then appends: **Signals** (84-97).
3. Experimental features (98-115) are appended by `extract_experimental_into()` on top of the 98-feature vector.

```
[0..39] LOB raw  →  [40..47] Derived  →  [48..83] MBO  →  [84..97] Signals  →  [98..115] Experimental
```

---

## Sign Convention (RULE.md Section 10)

All directional features follow:

| Convention | Meaning |
|------------|---------|
| `> 0` | Bullish / Buy pressure |
| `< 0` | Bearish / Sell pressure |
| `= 0` | Neutral / No signal |

---

## 1. Raw LOB Features (40 features, indices 0-39)

**Source**: `src/features/lob_features.rs`
**Units**: Prices in dollars (converted from nanodollars: `i64 / 1e9`), Sizes in shares (u32 cast to f64)

| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0-9 | `ask_price_L1` .. `ask_price_L10` | USD | Ask price at level 1 through 10 |
| 10-19 | `ask_size_L1` .. `ask_size_L10` | shares | Ask size at level 1 through 10 |
| 20-29 | `bid_price_L1` .. `bid_price_L10` | USD | Bid price at level 1 through 10 |
| 30-39 | `bid_size_L1` .. `bid_size_L10` | shares | Bid size at level 1 through 10 |

**Layout invariant**: `ask_price_L1 >= bid_price_L1` (validated by `FeatureValidator`).
Empty levels have price = 0.0 and size = 0.

---

## 2. Derived Features (8 features, indices 40-47)

**Source**: `src/features/derived_features.rs`

| Index | Name | Formula | Unit | Range |
|-------|------|---------|------|-------|
| 40 | `mid_price` | `(best_bid + best_ask) / 2` | USD | > 0 |
| 41 | `spread` | `best_ask - best_bid` | USD | >= 0 |
| 42 | `spread_bps` | `(spread / mid_price) * 10000` | bps | >= 0 |
| 43 | `total_bid_volume` | `sum(bid_sizes[0..levels])` | shares | >= 0 |
| 44 | `total_ask_volume` | `sum(ask_sizes[0..levels])` | shares | >= 0 |
| 45 | `volume_imbalance` | `(total_bid - total_ask) / (total_bid + total_ask)` | ratio | \[-1, 1\] |
| 46 | `weighted_mid_price` | `(best_bid * best_ask_size + best_ask * best_bid_size) / (best_bid_size + best_ask_size)` | USD | > 0 |
| 47 | `price_impact` | `\|mid_price - weighted_mid_price\|` | USD | >= 0 |

**Note**: All prices are converted from nanodollars (`i64 / 1e9`) at the boundary.

---

## 3. MBO Features (36 features, indices 48-83)

**Source**: `src/features/mbo_features/` (directory module)

The MBO aggregator (`MboAggregator`) maintains three rolling windows:
- **fast** (default 5 events): rapid micro-changes
- **medium** (default 500 events): primary computation window
- **slow** (default 5000 events): long-term trends

An `OrderTracker` (BTreeMap-backed, deterministic iteration) tracks active orders for lifecycle features.

### 3.1 Flow Features (12 features, indices 48-59)

**Source**: `src/features/mbo_features/flow_features.rs`

| Index | Name | Formula | Unit | Range |
|-------|------|---------|------|-------|
| 48 | `add_rate_bid` | `medium.add_count_bid / duration_seconds` | events/sec | >= 0 |
| 49 | `add_rate_ask` | `medium.add_count_ask / duration_seconds` | events/sec | >= 0 |
| 50 | `cancel_rate_bid` | `medium.cancel_count_bid / duration_seconds` | events/sec | >= 0 |
| 51 | `cancel_rate_ask` | `medium.cancel_count_ask / duration_seconds` | events/sec | >= 0 |
| 52 | `trade_rate_bid` | `medium.trade_count_bid / duration_seconds` | events/sec | >= 0 |
| 53 | `trade_rate_ask` | `medium.trade_count_ask / duration_seconds` | events/sec | >= 0 |
| 54 | `net_order_flow` | `(add_bid - add_ask) / (add_bid + add_ask + EPS)` | ratio | \[-1, 1\] |
| 55 | `net_cancel_flow` | `(cancel_ask - cancel_bid) / (total_cancel + EPS)` | ratio | \[-1, 1\] |
| 56 | `net_trade_flow` | `(trade_bid - trade_ask) / (total_trade + EPS)` | ratio | \[-1, 1\] |
| 57 | `aggressive_order_ratio` | `total_trades / (total_adds + EPS)` | ratio | \[0, 1\] |
| 58 | `order_flow_volatility` | `std(rolling_net_order_flow)` (requires >= 50 events) | ratio | >= 0 |
| 59 | `flow_regime_indicator` | `fast_net_flow / max(abs(slow_net_flow), flow_min_denom)`, clamped \[-10, 10\] | ratio | \[-10, 10\] |

**Sign convention**: `net_order_flow > 0` = more bid adds = BULLISH.
`net_cancel_flow > 0` = more ask cancels (sellers pulling) = BULLISH.

### 3.2 Size Features (8 features, indices 60-67)

**Source**: `src/features/mbo_features/size_features.rs`

| Index | Name | Formula | Unit | Range |
|-------|------|---------|------|-------|
| 60 | `size_p25` | 25th percentile of event sizes in medium window | shares | >= 0 |
| 61 | `size_p50` | 50th percentile (median) | shares | >= 0 |
| 62 | `size_p75` | 75th percentile | shares | >= 0 |
| 63 | `size_p90` | 90th percentile | shares | >= 0 |
| 64 | `size_zscore` | `(last_event_size - mean) / (std + EPS)` | dimensionless | unbounded |
| 65 | `large_order_ratio` | `count(size > 3 * mean) / total_events` | ratio | \[0, 1\] |
| 66 | `size_skewness` | `(1/n) * sum((x_i - mean)^3 / std^3)` | dimensionless | unbounded |
| 67 | `size_concentration` | HHI = `sum((size_i / total_size)^2)` | ratio | (0, 1\] |

### 3.3 Queue Features (6 features, indices 68-73)

**Source**: `src/features/mbo_features/queue_features.rs`

| Index | Name | Formula | Unit | Range |
|-------|------|---------|------|-------|
| 68 | `avg_queue_position` | average queue position of tracked orders (0.0 if tracking disabled) | ratio | >= 0 |
| 69 | `queue_volume_ahead` | average volume ahead of tracked orders (0.0 if tracking disabled) | shares | >= 0 |
| 70 | `orders_per_level` | `active_orders / max(active_levels, 1)` | count | >= 0 |
| 71 | `level_concentration` | HHI over all non-zero bid + ask level sizes | ratio | (0, 1\] |
| 72 | `depth_ticks_bid` | `(best_bid - worst_bid_price) / tick_size` | ticks | >= 0 |
| 73 | `depth_ticks_ask` | `(worst_ask_price - best_ask) / tick_size` | ticks | >= 0 |

**Note**: Features 68-69 are 0.0 by default because `queue_tracking` is disabled for performance.

### 3.4 Institutional Features (4 features, indices 74-77)

**Source**: `src/features/mbo_features/institutional_features.rs`

| Index | Name | Formula | Unit | Range |
|-------|------|---------|------|-------|
| 74 | `large_order_frequency` | `count(size > threshold) / duration_seconds` in medium window | events/sec | >= 0 |
| 75 | `large_order_imbalance` | `(large_bid - large_ask) / (total_large + EPS)` | ratio | \[-1, 1\] |
| 76 | `modification_score` | mean modification count of completed orders | count | >= 0 |
| 77 | `iceberg_proxy` | `avg_fill_ratio * min(mod_score / 10, 1)` | ratio | \[0, 1\] |

### 3.5 Lifecycle Features (6 features, indices 78-83)

**Source**: `src/features/mbo_features/lifecycle_features.rs`

| Index | Name | Formula | Unit | Range |
|-------|------|---------|------|-------|
| 78 | `avg_order_age` | mean age of active orders | seconds | >= 0 |
| 79 | `median_order_lifetime` | median of completed order lifetimes (buffer of 1000) | seconds | >= 0 |
| 80 | `avg_fill_ratio` | mean fill ratio of completed orders | ratio | \[0, 1\] |
| 81 | `avg_time_to_first_fill` | mean time from add to first fill for active orders | seconds | >= 0 |
| 82 | `cancel_to_add_ratio` | `total_cancels / (total_adds + EPS)`, clamped \[0, 10\] | ratio | \[0, 10\] |
| 83 | `active_order_count` | count of tracked active orders | count | >= 0 |

---

## 4. Trading Signals (14 features, indices 84-97)

**Source**: `src/features/signals/` (directory module)
**Requirement**: `lob_levels == 10` (signal indices assume 40 raw LOB features)

| Index | Constant | Name | Formula | Range | Category |
|-------|----------|------|---------|-------|----------|
| 84 | `TRUE_OFI` | Streaming OFI | `sum(delta_bid_size_L1 - delta_ask_size_L1)` (Cont et al., 2014) | unbounded | Continuous |
| 85 | `DEPTH_NORM_OFI` | Depth-normalized OFI | `ofi / (total_bid_vol + total_ask_vol + EPS)` | unbounded | Continuous |
| 86 | `EXECUTED_PRESSURE` | Execution pressure | `trade_rate_ask - trade_rate_bid` | unbounded | Continuous |
| 87 | `SIGNED_MP_DELTA_BPS` | Microprice delta | `(weighted_mid - prev_weighted_mid) / mid * 10000` | unbounded | Continuous |
| 88 | `TRADE_ASYMMETRY` | Trade asymmetry | `(trade_bid - trade_ask) / (trade_bid + trade_ask + EPS)` | \[-1, 1\] | Continuous |
| 89 | `CANCEL_ASYMMETRY` | Cancel asymmetry | `(cancel_ask - cancel_bid) / (cancel_bid + cancel_ask + EPS)` | \[-1, 1\] | Continuous |
| 90 | `FRAGILITY_SCORE` | Book fragility | `1 - level_concentration * (depth_bid + depth_ask + EPS) / 100` | \[0, 1\] | Continuous |
| 91 | `DEPTH_ASYMMETRY` | Depth asymmetry | `(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + EPS)` | \[-1, 1\] | Continuous |
| 92 | `BOOK_VALID` | Book validity | 1.0 if bid < ask and both > 0, else 0.0 | {0, 1} | **Categorical** |
| 93 | `TIME_REGIME` | Market session | 0=pre-market, 1=opening, 2=morning, 3=afternoon, 4=closing | {0,1,2,3,4} | **Categorical** |
| 94 | `MBO_READY` | MBO warmup | 1.0 if `ofi.is_warm()` (>= 100 state changes), else 0.0 | {0, 1} | **Categorical** |
| 95 | `DT_SECONDS` | Sample interval | `(current_ts - prev_ts) / 1e9` | >= 0 | Continuous |
| 96 | `INVALIDITY_DELTA` | Invalidity count | Number of crossed/locked book events since last sample | >= 0 | Continuous |
| 97 | `SCHEMA_VERSION` | Schema version | Always `contract::SCHEMA_VERSION` (2.2) | {2.2} | **Categorical** |

**Categorical indices \[92, 93, 94, 97\]** must be excluded from normalization.

**OFI Warmup**: `OfiComputer` requires `MIN_WARMUP_STATE_CHANGES = 100` LOB transitions before producing valid OFI values. Before warmup, `MBO_READY = 0.0` and OFI signals are 0.0.

**Signal computation source files**:
- `signals/ofi.rs` -- `OfiComputer`, streaming OFI accumulation
- `signals/time_regime.rs` -- `TimeRegime`, UTC-to-ET conversion with DST
- `signals/compute.rs` -- `compute_signals()`, assembles all 14 signals
- `signals/indices.rs` -- constant definitions for all 14 indices
- `signals/book_valid.rs` -- `is_book_valid()` check

---

## 5. Experimental Features (18 features, indices 98-115)

**Source**: `src/features/experimental/`
**Activation**: `[features.experimental] enabled = true` in DatasetConfig TOML

### 5.1 Institutional V2 (8 features, indices 98-105)

**Source**: `src/features/experimental/institutional_v2.rs`

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 98 | `round_lot_ratio` | Fraction of orders that are round lots (100-share multiples) | \[0, 1\] |
| 99 | `odd_lot_ratio` | Fraction of orders < 100 shares | \[0, 1\] |
| 100 | `size_clustering` | HHI of size buckets (how concentrated sizes are) | (0, 1\] |
| 101 | `price_clustering` | HHI of price tick offsets | (0, 1\] |
| 102 | `mod_before_cancel` | Rate of orders modified before being cancelled | \[0, 1\] |
| 103 | `sweep_ratio` | Fraction of trades that look like sweeps (multi-level fills) | \[0, 1\] |
| 104 | `fill_patience_bid` | Avg time to fill for bid orders | seconds |
| 105 | `fill_patience_ask` | Avg time to fill for ask orders | seconds |

### 5.2 Volatility (6 features, indices 106-111)

**Source**: `src/features/experimental/volatility.rs`

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 106 | `realized_vol_fast` | Short-window realized volatility | >= 0 |
| 107 | `realized_vol_slow` | Long-window realized volatility | >= 0 |
| 108 | `vol_ratio` | `fast_vol / (slow_vol + EPS)` | >= 0 |
| 109 | `vol_momentum` | Change in fast vol between snapshot intervals | unbounded |
| 110 | `return_autocorr` | Autocorrelation of returns | \[-1, 1\] |
| 111 | `vol_of_vol` | Volatility of volatility (std of fast_vol history) | >= 0 |

### 5.3 Seasonality (4 features, indices 112-115)

**Source**: `src/features/experimental/seasonality.rs`

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 112 | `minutes_since_open` | Minutes elapsed since market open (9:30 ET) | \[0, 390\] |
| 113 | `minutes_until_close` | Minutes remaining until close (16:00 ET) | \[0, 390\] |
| 114 | `session_progress` | Fraction of trading session elapsed | \[0, 1\] |
| 115 | `time_bucket` | 30-minute bucket index (0-12) | {0..12} |

---

## Feature Count Summary

| Configuration | Count | Formula |
|---------------|-------|---------|
| Raw LOB only | 40 | 10 levels x 4 |
| + Derived | 48 | 40 + 8 |
| + MBO | 76 | 40 + 36 |
| + Derived + MBO | 84 | 40 + 8 + 36 |
| + Derived + MBO + Signals | **98** | 40 + 8 + 36 + 14 |
| + Experimental (all groups) | **116** | 98 + 8 + 6 + 4 |

---

## Normalization Behavior

| Feature Group | Indices | Default Strategy | Notes |
|--------------|---------|-----------------|-------|
| Ask prices | 0-9 | `market_structure_zscore` (shared mean/std per level pair) | Preserves ask > bid ordering |
| Ask sizes | 10-19 | `z_score` (per-feature) | Independent normalization |
| Bid prices | 20-29 | `market_structure_zscore` (shared with corresponding ask level) | |
| Bid sizes | 30-39 | `z_score` (per-feature) | |
| Derived | 40-47 | `z_score` (per-feature) | |
| MBO ratios | 48-83 | Already normalized or `z_score` | Most are bounded [-1, 1] or [0, 1] |
| Signal continuous | 84-91, 95-96 | `z_score` or raw | OFI may benefit from z-score |
| **Categorical** | **92, 93, 94, 97** | **No normalization** | Must be excluded from all normalizers |
| Experimental | 98-115 | `z_score` or raw | Depends on feature nature |

---

*This document is derived from the source code in `src/features/`, `src/contract.rs`, and `contracts/pipeline_contract.toml`. Last updated: March 5, 2026.*
