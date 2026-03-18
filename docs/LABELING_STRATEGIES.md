# Labeling Strategies for ML Trading Models

This document describes the labeling strategies available in the `feature-extractor` library for training machine learning models on limit order book (LOB) data.

## Overview

Choosing the right labeling strategy is crucial for training effective trading models. Different strategies optimize for different objectives:

| Strategy | Type | Best For | Key Advantage | Export Status |
|----------|------|----------|---------------|---------------|
| TLOB | Classification | Trend following | Simple, widely used | ✅ Integrated |
| Multi-Horizon | Classification | Adaptive strategies | Multiple timeframes | ✅ Integrated |
| Opportunity | Classification | Big move detection | Peak return focus | ✅ Integrated |
| Triple Barrier | Classification | Risk-managed trading | Explicit risk/reward, vol-adaptive | ✅ **Integrated** |
| Magnitude | Regression | Position sizing | Continuous targets | ✅ **Integrated** |

> **All labeling strategies are fully integrated** into the `export_dataset` CLI tool. See [Integration Status](#integration-status) for details.

## 1. TLOB Labeling (Original DeepLOB)

The original labeling method from the DeepLOB paper (Zhang et al., 2019).

### Method

Computes smoothed mid-price returns and classifies based on fixed thresholds:

```
smoothed_return = avg(price[t+1:t+h+1]) / avg(price[t-k:t+1]) - 1

if smoothed_return > threshold:
    label = UP
elif smoothed_return < -threshold:
    label = DOWN
else:
    label = STABLE
```

### Configuration

```rust
use feature_extractor::{LabelConfig, TlobLabelGenerator};

let config = LabelConfig::new(
    10,     // horizon (h): forward prediction window
    5,      // smoothing_window (k): past/future smoothing
    0.0002, // threshold: classification boundary
);
let mut labeler = TlobLabelGenerator::new(config);
```

### When to Use

- Baseline experiments
- Comparison with DeepLOB paper
- Simple trend classification

### Limitations

- Fixed threshold may not adapt to volatility
- Smoothing can blur sharp moves
- Equal treatment of all directions

---

## 2. Multi-Horizon Labeling

Generates labels for multiple prediction horizons simultaneously.

### Method

Computes returns at multiple horizons with configurable threshold strategies:

```
for each horizon h in [10, 50, 100, 200]:
    return[h] = price[t+h] / price[t] - 1
    label[h] = classify(return[h], threshold[h])
```

### Configuration

```rust
use feature_extractor::{MultiHorizonConfig, MultiHorizonLabelGenerator, ThresholdStrategy};

let config = MultiHorizonConfig::new(
    vec![10, 50, 100, 200],
    5,  // smoothing_window
    ThresholdStrategy::quantile(0.33, 5000, 0.002),  // target_proportion, window_size, fallback
);

let mut labeler = MultiHorizonLabelGenerator::new(config);
```

### Threshold Strategies

| Strategy | Constructor | Description | Use Case |
|----------|------------|-------------|----------|
| `Fixed(f64)` | `ThresholdStrategy::fixed(0.001)` | Same threshold for all horizons | Simple, predictable |
| `RollingSpread { window_size, multiplier, fallback }` | `ThresholdStrategy::rolling_spread(5000, 1.5, 0.002)` | avg_spread x multiplier | Cost-aware |
| `Quantile { target_proportion, window_size, fallback }` | `ThresholdStrategy::quantile(0.33, 5000, 0.002)` | Adaptive balanced classes | Equal class distribution |
| `TlobDynamic { fallback, divisor }` | `ThresholdStrategy::tlob_dynamic(0.0008, 2.0)` | `mean(\|pct_change\|) / divisor` | Match official TLOB repo |

#### TLOB Dynamic Threshold (New)

The `TlobDynamic` strategy computes the threshold from the **entire** price series as:

```
alpha = mean(|percentage_change|) / divisor
```

This matches the official TLOB repository's `labeling()` function:

```python
# From TLOB/utils/utils_data.py
alpha = np.abs(percentage_change).mean() / 2
labels = np.where(percentage_change < -alpha, 2,
                  np.where(percentage_change > alpha, 0, 1))
```

**Configuration:**

```rust
use feature_extractor::labeling::ThresholdStrategy;

// Match official TLOB repository
let strategy = ThresholdStrategy::tlob_dynamic_default(0.0008); // fallback = 0.0008

// Custom divisor
let strategy = ThresholdStrategy::tlob_dynamic(0.0008, 3.0); // divisor = 3.0
```

**TOML Configuration:**

```toml
[labels.threshold_strategy]
type = "tlob_dynamic"
fallback = 0.0008    # Used when insufficient data
divisor = 2.0        # Default per TLOB paper
```

### When to Use

- Multi-task learning
- Ensemble predictions
- Signal persistence analysis

---

## 3. Opportunity Labeling (Big Move Detection)

**New in Phase 1.5**

Designed for detecting significant price movements ("big moves") by focusing on peak returns within a horizon.

### Method

Unlike TLOB which uses smoothed endpoint returns, Opportunity labeling finds the **maximum** and **minimum** returns within the entire horizon window:

```
max_return = max((price[t+1:t+h+1] - price[t]) / price[t])
min_return = min((price[t+1:t+h+1] - price[t]) / price[t])

if max_return > up_threshold:
    label = BIG_UP
elif min_return < -down_threshold:
    label = BIG_DOWN
else:
    label = NO_OPPORTUNITY
```

### Key Insight

The peak return approach captures opportunities that TLOB might miss:

```
Price path: 100 → 103 → 101 → 100

TLOB (endpoint): return = 0% → STABLE
Opportunity (peak): max_return = 3% → BIG_UP
```

### Configuration

```rust
use feature_extractor::{OpportunityConfig, OpportunityLabelGenerator, ConflictPriority};

let mut config = OpportunityConfig::new(
    50,    // horizon: look-ahead window
    0.005, // threshold: 0.5% for big moves (symmetric by default)
);
config.conflict_priority = ConflictPriority::LargerMagnitude;
// Optional: asymmetric thresholds
config.down_threshold = Some(0.003); // Separate down threshold

let mut labeler = OpportunityLabelGenerator::new(config);
```

### Conflict Resolution

When both thresholds are exceeded within the horizon:

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `LargerMagnitude` | Pick the larger move | Capture dominant direction (default) |
| `UpPriority` | Always pick BigUp | Long-only strategies |
| `DownPriority` | Always pick BigDown | Short-only strategies |
| `Ambiguous` | Label as NoOpportunity | Conservative approach |

### When to Use

- "Whale detection" - finding big opportunities
- Low-frequency trading strategies
- When you want to trade less but win more

---

## 4. Triple Barrier Labeling

> ✅ **Fully Integrated into Export Pipeline (Schema 2.4+)**
>
> The Triple Barrier labeler is fully integrated into the `export_dataset` CLI tool.
> Use `strategy = "triple_barrier"` in your TOML config. Features include:
> - Per-horizon barriers (different profit/stop-loss per horizon)
> - Volatility-adaptive scaling (per-day barrier adjustment based on realized volatility)
> - Configurable timeout strategies and conflict resolution
> - Calibration tool: `tools/calibrate_triple_barrier.py`

Implements the Triple Barrier method from López de Prado's "Advances in Financial Machine Learning" (2018).

### Method

Places three barriers around each entry point and labels based on which barrier is hit first:

```
         ▲ Upper Barrier (Profit Target: +2%)
         │
Entry ───┼─────────────────────────► Time
         │                           │
         ▼ Lower Barrier (Stop-Loss: -1%)
                                     │
                        Vertical Barrier (Max Horizon)
```

### Label Assignment

| First Barrier Hit | Label (Class Index) | Meaning |
|-------------------|---------------------|---------|
| Lower (Stop-Loss) | 0 | Losing trade |
| Vertical (Time Limit) | 1 | Inconclusive / Timeout |
| Upper (Profit Target) | 2 | Winning trade |

> **Note**: Triple Barrier uses class indices `{0, 1, 2}` (not signed `{-1, 0, 1}` like TLOB/Opportunity).
> This is enforced by `LabelEncoding::TripleBarrierClassIndex` in the validation contract.

### Configuration

```rust
use feature_extractor::{TripleBarrierConfig, TripleBarrierLabeler, TimeoutStrategy};

// Asymmetric barriers: 2:1 reward/risk
let config = TripleBarrierConfig::new(
    0.02,   // profit_target_pct: 2%
    0.01,   // stop_loss_pct: 1%
    100,    // max_horizon: events
);

// Or use presets
let config = TripleBarrierConfig::intraday();  // 0.5% / 0.3% / 100
let config = TripleBarrierConfig::scalping();  // 0.2% / 0.15% / 20
let config = TripleBarrierConfig::swing();     // 1.0% / 0.5% / 500
```

### Risk/Reward Analysis

```rust
let config = TripleBarrierConfig::new(0.02, 0.01, 100);

assert_eq!(config.risk_reward_ratio(), 2.0);      // 2:1 R:R
assert!((config.breakeven_win_rate() - 0.33).abs() < 0.01); // Need 33% win rate
```

### TOML Export Configuration

```toml
# Basic Triple Barrier
[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_target_pct = 0.005
stop_loss_pct = 0.003

# Per-horizon barriers (calibrated to each horizon's volatility)
[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_targets = [0.0028, 0.0039, 0.0059]
stop_losses = [0.0019, 0.0026, 0.0040]

# Volatility-adaptive scaling (barriers adjust per-day)
[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_targets = [0.0020, 0.0028, 0.0042]
stop_losses = [0.0013, 0.0019, 0.0028]
volatility_scaling = true
volatility_reference = 0.00015
volatility_floor = 0.3
volatility_cap = 3.0
```

### Calibration Tool

Use `tools/calibrate_triple_barrier.py` to compute data-driven barrier thresholds:

```bash
python tools/calibrate_triple_barrier.py \
    --data-dir ../data/exports/my_export \
    --horizons 50 100 200 \
    --target-pt-rate 0.25 --target-sl-rate 0.25

# Per-day volatility analysis
python tools/calibrate_triple_barrier.py \
    --data-dir ../data/exports/my_export \
    --per-day --horizons 50 100 200
```

### Timeout Strategies

| Strategy | Behavior |
|----------|----------|
| `LabelAsTimeout` | Always label as 1 (standard) |
| `UseReturnSign` | Label based on final return sign |
| `UseFractionalThreshold` | Use partial threshold (50%) |

### When to Use

- Training models that directly inform trading decisions
- When risk management is a priority
- Backtesting with realistic entry/exit logic

---

## 5. Magnitude Generation (Regression)

> ✅ **Fully Integrated into Export Pipeline**
>
> Use `strategy = "regression"` in your TOML config. Features include:
> - Multiple return types: `smoothed_return`, `point_return`, `peak_return`, `mean_return`, `dominant_return`
> - Configurable via `return_type` field in TOML
> - Multi-horizon and single-horizon support
> - Pure regression mode (no classification labels generated)

Generates continuous return values instead of discrete labels, enabling regression-based approaches.

### Method

Computes various return metrics for each entry point:

```rust
ReturnData {
    point_return,   // (price[t+h] - price[t]) / price[t]
    max_return,     // max return within horizon
    min_return,     // min return within horizon
    mean_return,    // average return
    return_std,     // volatility within horizon
    time_to_max,    // when peak occurred
    time_to_min,    // when trough occurred
}
```

### Configuration

```rust
use feature_extractor::{MagnitudeConfig, MagnitudeGenerator, ReturnType};

// Single horizon point return
let config = MagnitudeConfig::point_return(50);

// Peak returns for opportunity sizing
let config = MagnitudeConfig::peak_returns(100);

// Multi-horizon for signal decay analysis
let config = MagnitudeConfig::multi_horizon(vec![10, 50, 100, 200]);
```

### Return Types

| Type | Formula | Use Case |
|------|---------|----------|
| `PointReturn` | price[t+h] / price[t] - 1 | Standard prediction |
| `MaxReturn` | max(prices[t+1:t+h+1]) / price[t] - 1 | Long opportunities |
| `MinReturn` | min(prices[t+1:t+h+1]) / price[t] - 1 | Short opportunities |
| `DominantReturn` | Larger of max/min by magnitude | Direction detection |
| `MeanReturn` | mean(prices[t+1:t+h+1]) / price[t] - 1 | Smoothed target |

### TOML Export Configuration

```toml
# Basic regression (smoothed returns, default)
[labels]
strategy = "regression"
horizons = [10, 60, 300]
smoothing_window = 10
threshold = 0.0008

# Point returns (no smoothing)
[labels]
strategy = "regression"
horizons = [10, 60, 300]
smoothing_window = 10
return_type = "point_return"

# Peak returns (max opportunity in horizon)
[labels]
strategy = "regression"
horizons = [10, 60, 300]
smoothing_window = 10
return_type = "peak_return"
```

### When to Use

- Position sizing based on predicted magnitude
- Training regression models (XGBoost, Neural Nets)
- Applying thresholds at inference time instead of training
- Alpha research and signal analysis

---

## Choosing a Strategy

### Decision Tree

```
What's your trading objective?
│
├─► High-frequency scalping → TLOB or Triple Barrier (scalping preset)
│
├─► Swing trading / big moves → Opportunity or Triple Barrier (swing preset)
│
├─► Risk-managed strategies → Triple Barrier
│
├─► Position sizing → Magnitude + classification ensemble
│
├─► Research / exploration → Multi-Horizon or Magnitude
│
└─► Baseline / comparison → TLOB (matches DeepLOB paper)
```

### Comparison Table

| Feature | TLOB | Multi-Horizon | Opportunity | Triple Barrier | Magnitude |
|---------|------|---------------|-------------|----------------|-----------|
| Output Type | 3-class | Multi 3-class | 3/4-class | 3-class | Continuous |
| Peak Detection | ❌ | ❌ | ✅ | Via barriers | ✅ |
| Risk/Reward | ❌ | ❌ | ❌ | ✅ | Via post-processing |
| Path Dependent | Via smoothing | ❌ | ✅ | ✅ | ✅ |
| Configurable | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multi-timeframe | ❌ | ✅ | ❌ | ❌ | ✅ |

---

## Best Practices

### 1. Match Strategy to Trading Style

Don't train with TLOB labels if you plan to use Triple Barrier-style exits in production.

### 2. Check Class Balance

```rust
let stats = labeler.compute_stats(&labels);
println!("Class distribution: {:?}", stats.class_balance());
```

### 3. Test Multiple Strategies

Export data with different labeling strategies and compare model performance:

```toml
# experiment_1.toml
[labels]
strategy = "opportunity"
horizon = 50
up_threshold = 0.005

# experiment_2.toml
[labels]
strategy = "triple_barrier"
profit_target_pct = 0.005
stop_loss_pct = 0.003
```

### 4. Consider Regime Dependence

What works in trending markets may fail in choppy markets. Consider:
- Using `ThresholdStrategy::RollingSpread` or `ThresholdStrategy::TlobDynamic` for adaptive thresholds
- Training separate models for different regimes
- Including regime indicators as features

---

## API Reference

### Common Pattern

All labelers follow a similar pattern:

```rust
// 1. Create config
let config = XxxConfig::new(...);

// 2. Create labeler
let mut labeler = XxxLabeler::new(config);

// 3. Add prices
labeler.add_prices(&mid_prices);

// 4. Generate labels
let labels = labeler.generate_labels()?;

// 5. Compute statistics
let stats = labeler.compute_stats(&labels);
```

### Error Handling

All generators return `Result<T>` and validate data requirements:

```rust
match labeler.generate_labels() {
    Ok(labels) => { /* process */ },
    Err(e) => eprintln!("Error: {}", e),
}
```

---

---

## Integration Status

This section clarifies which labeling strategies are fully integrated into the export pipeline (`export_dataset` CLI tool) vs. available only through the Rust API.

### Fully Integrated (Export Pipeline)

These strategies can be used via TOML config and `export_dataset`:

| Strategy | Config Field | Classes | Label Encoding | Notes |
|----------|--------------|---------|----------------|-------|
| **TLOB** | `strategy = "tlob"` | Down/Stable/Up | `{-1, 0, 1}` (SignedTrend) | Default, most tested |
| **Multi-Horizon** | `horizons = [10, 50, 100]` | Down/Stable/Up × N | `{-1, 0, 1}` (SignedTrend) | Multiple horizons |
| **Opportunity** | `strategy = "opportunity"` | BigDown/NoOpp/BigUp | `{-1, 0, 1}` (SignedOpportunity) | Peak return based |
| **Triple Barrier** | `strategy = "triple_barrier"` | StopLoss/Timeout/ProfitTarget | `{0, 1, 2}` (TripleBarrierClassIndex) | Vol-adaptive, per-horizon barriers |
| **Magnitude** | `strategy = "regression"` | Regression (continuous) | continuous_bps, float64 | Label file: `{day}_regression_labels.npy` |

### Using API-Only Strategies

All strategies are now fully integrated into the export pipeline. Use TOML configuration as documented above for each strategy.

### Roadmap

- [x] Integrate `TripleBarrierLabeler` into `AlignedBatchExporter` ✅ (Schema 2.4+)
- [x] Add `profit_target_pct`, `stop_loss_pct` fields to `ExportLabelConfig` ✅ (Schema 2.4+)
- [x] Per-horizon barrier overrides ✅ (Schema 3.2+)
- [x] Volatility-adaptive barrier scaling ✅ (Schema 3.3+)
- [x] Unified `LabelEncoding` validation contract ✅
- [x] Integrate `MagnitudeGenerator` into export pipeline ✅
- [x] Add regression label export support ✅

---

## Research References

1. **TLOB**: Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. IEEE TNNLS.

2. **Triple Barrier**: López de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 3.

3. **Feature Engineering for Finance**: de Prado, M. L. (2020). Machine Learning for Asset Managers. Cambridge University Press.

