# Research Gap Analysis: Feature-Extractor-MBO-LOB

> **Purpose**: Comprehensive technical analysis of missing components required to build state-of-the-art LOB prediction models based on 22 research papers.
>
> **Scope**: Preprocessing, feature extraction, normalization, labeling, and data export for multi-model support.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Implementation Status](#current-implementation-status)
3. [High Priority Gaps](#high-priority-gaps)
4. [Medium Priority Gaps](#medium-priority-gaps)
5. [Model-Specific Requirements](#model-specific-requirements)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

The feature-extractor-MBO-LOB library is **~70% complete** for supporting state-of-the-art models. Critical gaps exist in:

1. **Normalization**: Missing rolling/dynamic z-score (required by DeepLOB, TLOB)
2. **FI-2010 Features**: Missing time-sensitive features (u6-u9)
3. **Labeling**: Missing spread-based thresholds (TLOB)
4. **Data Format**: Missing multi-scale output (HLOB)

### Priority Matrix

| Gap | Impact | Effort | Models Affected |
|-----|--------|--------|-----------------|
| Rolling Z-Score Normalization | 🔴 Critical | Medium | DeepLOB, TLOB, FI-2010 |
| FI-2010 Time-Sensitive Features | 🔴 Critical | High | FI-2010 benchmark |
| Spread-Based Labeling | 🟡 High | Low | TLOB, practical trading |
| Multi-Horizon Labels | 🟡 High | Low | All models |
| HLOB Topology Features | 🟡 High | High | HLOB |
| Input Tensor Formatting | 🟡 High | Medium | All models |

---

## Current Implementation Status

### ✅ Fully Implemented

| Component | Features | Research Reference |
|-----------|----------|-------------------|
| Raw LOB features | 40 (10 levels × 4) | DeepLOB, TLOB |
| Derived features | 8 (spread, imbalance, etc.) | Standard |
| MBO features | 36 | MBO Paper |
| Volume-based sampling | Adaptive threshold | FI-2010 |
| Event-based sampling | N-event aggregation | Standard |
| TLOB labeling | Decoupled h/k | TLOB Paper |
| DeepLOB labeling | k=h method | DeepLOB Paper |
| Sequence building | Sliding window | All models |
| NumPy export | .npy files | Standard |

### ⚠️ Partially Implemented

| Component | Current State | Gap |
|-----------|--------------|-----|
| Z-Score normalization | Static (full dataset) | Missing rolling/dynamic |
| FI-2010 features | u1-u5 only | Missing u6-u9 |
| Label thresholds | Fixed percentage | Missing spread-based |
| Multi-scale | Window support | Missing proper export |

---

## High Priority Gaps

### 1. Rolling Z-Score Normalization (Dynamic)

**Research Reference**: DeepLOB (Section 3.2), TLOB, FI-2010

**Why Critical**: All benchmark papers use **rolling z-score** normalization, not static:

> "We use z-score normalization where mean and std are computed from a rolling window of the past 5 trading days" — DeepLOB

**Current Implementation**:
```rust
// Current: Static z-score (computed once over entire dataset)
pub struct GlobalZScoreNormalizer {
    means: Vec<f64>,  // Fixed after initialization
    stds: Vec<f64>,   // Fixed after initialization
}
```

**Required Implementation**:

```rust
/// Rolling Z-Score Normalizer (DeepLOB-style)
/// 
/// Computes mean and std from a rolling window of past N trading days.
/// Updates statistics at each day boundary.
/// 
/// # Research Alignment
/// 
/// From DeepLOB paper:
/// - Window: Past 5 trading days
/// - Update: At market open each day
/// - Features: All 40+ features normalized independently
/// 
/// # Mathematical Formulation
/// 
/// For feature x at time t:
/// ```text
/// μ_t = mean(x over past N days)
/// σ_t = std(x over past N days)
/// x_normalized = (x - μ_t) / σ_t
/// ```
#[derive(Debug, Clone)]
pub struct RollingZScoreNormalizer {
    /// Number of days in the rolling window
    window_days: usize,
    
    /// Per-feature statistics storage
    /// Each entry: (running_sum, running_sum_sq, count)
    day_stats: VecDeque<DayFeatureStats>,
    
    /// Current normalization parameters (updated at day boundaries)
    current_means: Vec<f64>,
    current_stds: Vec<f64>,
    
    /// Minimum std to prevent division by zero
    min_std: f64,
    
    /// Number of features
    num_features: usize,
}

#[derive(Debug, Clone)]
struct DayFeatureStats {
    /// Sum of values for each feature
    sums: Vec<f64>,
    
    /// Sum of squared values for each feature  
    sum_squares: Vec<f64>,
    
    /// Count of samples
    count: usize,
    
    /// Trading day identifier
    day_index: u32,
}

impl RollingZScoreNormalizer {
    /// Create a new rolling z-score normalizer
    /// 
    /// # Arguments
    /// 
    /// * `window_days` - Number of past days to include (typically 5)
    /// * `num_features` - Number of features to normalize
    /// * `min_std` - Minimum standard deviation (default: 1e-8)
    pub fn new(window_days: usize, num_features: usize, min_std: f64) -> Self {
        Self {
            window_days,
            day_stats: VecDeque::with_capacity(window_days + 1),
            current_means: vec![0.0; num_features],
            current_stds: vec![1.0; num_features],
            min_std,
            num_features,
        }
    }
    
    /// Update with a new feature vector (call for each sample)
    pub fn update(&mut self, features: &[f64]) {
        // Add to current day's running statistics
        // ...
    }
    
    /// Trigger day boundary update
    /// 
    /// Call this when a new trading day starts.
    /// Recomputes mean/std from the rolling window.
    pub fn on_day_boundary(&mut self) {
        // 1. Finalize current day stats
        // 2. Add to deque, remove oldest if > window_days
        // 3. Recompute means and stds from all days in window
        
        let mut new_means = vec![0.0; self.num_features];
        let mut new_stds = vec![0.0; self.num_features];
        
        let total_count: usize = self.day_stats.iter().map(|d| d.count).sum();
        
        if total_count > 0 {
            for i in 0..self.num_features {
                let sum: f64 = self.day_stats.iter().map(|d| d.sums[i]).sum();
                let sum_sq: f64 = self.day_stats.iter().map(|d| d.sum_squares[i]).sum();
                
                let mean = sum / total_count as f64;
                let variance = (sum_sq / total_count as f64) - mean * mean;
                let std = variance.sqrt().max(self.min_std);
                
                new_means[i] = mean;
                new_stds[i] = std;
            }
        }
        
        self.current_means = new_means;
        self.current_stds = new_stds;
    }
    
    /// Normalize a feature vector using current rolling statistics
    pub fn normalize(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .zip(self.current_means.iter())
            .zip(self.current_stds.iter())
            .map(|((x, mean), std)| (x - mean) / std)
            .collect()
    }
    
    /// Get current normalization parameters
    pub fn get_params(&self) -> (&[f64], &[f64]) {
        (&self.current_means, &self.current_stds)
    }
}
```

**Integration with Pipeline**:

```rust
/// Normalization strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationStrategy {
    /// No normalization
    None,
    
    /// Static z-score (mean/std from full dataset)
    /// Use for: Post-hoc analysis, small datasets
    StaticZScore,
    
    /// Rolling z-score (mean/std from past N days)
    /// Use for: Production, online learning, benchmark reproduction
    RollingZScore {
        window_days: usize,  // Typically 5
        min_std: f64,        // Typically 1e-8
    },
    
    /// Percentage change from previous value
    /// Use for: Time series, returns-based features
    PercentageChange,
    
    /// Min-max scaling to [0, 1]
    /// Use for: Neural networks with sigmoid activation
    MinMax,
    
    /// Bilinear normalization (separate for buy/sell)
    /// Use for: Order flow features
    Bilinear,
}
```

**Impact**: 
- Reproduces DeepLOB benchmark results
- Required for TLOB paper replication
- Essential for online/streaming inference

---

### 2. FI-2010 Time-Sensitive Features (u6-u9)

**Research Reference**: Ntakaris et al. "Feature Engineering for Mid-Price Prediction"

**Why Critical**: The FI-2010 benchmark uses **144 total features**, of which 64 are time-sensitive:

| Group | Features | Description | Status |
|-------|----------|-------------|--------|
| u1 | 40 | Basic (price, volume) | ✅ |
| u2 | 40 | Spread, mid-price | ✅ |
| u3-u5 | Variable | Statistical (mean, std) | ⚠️ Partial |
| u6 | 20 | Price derivatives | ❌ Missing |
| u7 | 20 | Volume derivatives | ❌ Missing |
| u8 | 6 | Event arrival intensity | ❌ Missing |
| u9 | 18 | Relative intensity | ❌ Missing |

**Required Implementation**:

```rust
/// FI-2010 Time-Sensitive Feature Calculator
/// 
/// Computes features u6-u9 from the FI-2010 benchmark paper.
/// Requires temporal information (timestamps, message types).
/// 
/// # Feature Groups
/// 
/// ## u6: Price Derivatives (20 features)
/// - dP_ask(i)/dt for i in 1..10 (10 features)
/// - dP_bid(i)/dt for i in 1..10 (10 features)
/// 
/// ## u7: Volume Derivatives (20 features)
/// - dV_ask(i)/dt for i in 1..10 (10 features)
/// - dV_bid(i)/dt for i in 1..10 (10 features)
/// 
/// ## u8: Event Intensity (6 features)
/// - λ_limit_buy: Limit buy order arrival rate
/// - λ_limit_sell: Limit sell order arrival rate
/// - λ_market_buy: Market buy arrival rate
/// - λ_market_sell: Market sell arrival rate
/// - λ_cancel_buy: Cancel buy rate
/// - λ_cancel_sell: Cancel sell rate
/// 
/// ## u9: Relative Intensity (18 features)
/// - Intensity ratios between event types
/// - Intensity differences
/// - Combined intensity metrics
#[derive(Debug)]
pub struct Fi2010TimeSensitiveExtractor {
    /// Number of LOB levels (typically 10)
    lob_levels: usize,
    
    /// Time window for derivative calculation (nanoseconds)
    derivative_window_ns: u64,
    
    /// Time window for intensity calculation (nanoseconds)
    intensity_window_ns: u64,
    
    /// Recent price/volume history for derivative calculation
    price_history: PriceVolumeHistory,
    
    /// Event counters for intensity calculation
    event_counters: EventCounters,
    
    /// Minimum time delta to prevent division by zero
    min_delta_ns: u64,
}

#[derive(Debug)]
struct PriceVolumeHistory {
    /// Circular buffer of (timestamp, ask_prices, ask_volumes, bid_prices, bid_volumes)
    history: VecDeque<(u64, Vec<i64>, Vec<u64>, Vec<i64>, Vec<u64>)>,
    max_size: usize,
}

#[derive(Debug)]
struct EventCounters {
    /// (timestamp, event_type, side) for each event
    events: VecDeque<(u64, EventType, Side)>,
    max_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum EventType {
    LimitOrder,
    MarketOrder,
    Cancel,
    Modify,
}

impl Fi2010TimeSensitiveExtractor {
    /// Create a new extractor
    pub fn new(lob_levels: usize) -> Self {
        Self {
            lob_levels,
            derivative_window_ns: 1_000_000_000, // 1 second default
            intensity_window_ns: 60_000_000_000, // 1 minute default
            price_history: PriceVolumeHistory::new(100),
            event_counters: EventCounters::new(10000),
            min_delta_ns: 1_000, // 1 microsecond
        }
    }
    
    /// Update with new LOB state and event information
    pub fn update(
        &mut self,
        timestamp: u64,
        lob_state: &LobState,
        event_type: EventType,
        side: Side,
    ) {
        // Update price/volume history
        self.price_history.push(
            timestamp,
            lob_state.asks.iter().map(|l| l.price).collect(),
            lob_state.asks.iter().map(|l| l.size).collect(),
            lob_state.bids.iter().map(|l| l.price).collect(),
            lob_state.bids.iter().map(|l| l.size).collect(),
        );
        
        // Update event counters
        self.event_counters.push(timestamp, event_type, side);
    }
    
    /// Extract u6: Price derivatives (20 features)
    /// 
    /// dP/dt = (P(t) - P(t-Δ)) / Δ
    pub fn extract_u6(&self, current_ts: u64) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.lob_levels * 2);
        
        // Find oldest price within window
        let window_start = current_ts.saturating_sub(self.derivative_window_ns);
        
        if let Some((old_ts, old_ask_p, _, old_bid_p, _)) = 
            self.price_history.get_at_or_before(window_start) 
        {
            let dt_seconds = (current_ts - old_ts).max(self.min_delta_ns) as f64 / 1e9;
            
            if let Some((_, curr_ask_p, _, curr_bid_p, _)) = self.price_history.latest() {
                // Ask price derivatives
                for i in 0..self.lob_levels {
                    let dp = (curr_ask_p[i] - old_ask_p[i]) as f64;
                    features.push(dp / dt_seconds);
                }
                // Bid price derivatives
                for i in 0..self.lob_levels {
                    let dp = (curr_bid_p[i] - old_bid_p[i]) as f64;
                    features.push(dp / dt_seconds);
                }
            }
        }
        
        // Pad with zeros if insufficient history
        while features.len() < self.lob_levels * 2 {
            features.push(0.0);
        }
        
        features
    }
    
    /// Extract u7: Volume derivatives (20 features)
    pub fn extract_u7(&self, current_ts: u64) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.lob_levels * 2);
        
        let window_start = current_ts.saturating_sub(self.derivative_window_ns);
        
        if let Some((old_ts, _, old_ask_v, _, old_bid_v)) = 
            self.price_history.get_at_or_before(window_start) 
        {
            let dt_seconds = (current_ts - old_ts).max(self.min_delta_ns) as f64 / 1e9;
            
            if let Some((_, _, curr_ask_v, _, curr_bid_v)) = self.price_history.latest() {
                // Ask volume derivatives
                for i in 0..self.lob_levels {
                    let dv = curr_ask_v[i] as f64 - old_ask_v[i] as f64;
                    features.push(dv / dt_seconds);
                }
                // Bid volume derivatives
                for i in 0..self.lob_levels {
                    let dv = curr_bid_v[i] as f64 - old_bid_v[i] as f64;
                    features.push(dv / dt_seconds);
                }
            }
        }
        
        while features.len() < self.lob_levels * 2 {
            features.push(0.0);
        }
        
        features
    }
    
    /// Extract u8: Event arrival intensity (6 features)
    /// 
    /// λ = count(events in window) / window_duration
    pub fn extract_u8(&self, current_ts: u64) -> Vec<f64> {
        let window_start = current_ts.saturating_sub(self.intensity_window_ns);
        let window_seconds = self.intensity_window_ns as f64 / 1e9;
        
        let mut counts = [0u64; 6]; // limit_buy, limit_sell, market_buy, market_sell, cancel_buy, cancel_sell
        
        for (ts, event_type, side) in self.event_counters.events_since(window_start) {
            let idx = match (event_type, side) {
                (EventType::LimitOrder, Side::Bid) => 0,
                (EventType::LimitOrder, Side::Ask) => 1,
                (EventType::MarketOrder, Side::Bid) => 2,
                (EventType::MarketOrder, Side::Ask) => 3,
                (EventType::Cancel, Side::Bid) => 4,
                (EventType::Cancel, Side::Ask) => 5,
                _ => continue,
            };
            counts[idx] += 1;
        }
        
        counts.iter().map(|c| *c as f64 / window_seconds).collect()
    }
    
    /// Extract u9: Relative intensity (18 features)
    /// 
    /// Ratios and differences between event intensities
    pub fn extract_u9(&self, current_ts: u64) -> Vec<f64> {
        let u8 = self.extract_u8(current_ts);
        let mut features = Vec::with_capacity(18);
        
        // Ratios (with epsilon to prevent division by zero)
        let eps = 1e-10;
        
        // Limit order imbalance: λ_limit_buy / (λ_limit_buy + λ_limit_sell)
        features.push(u8[0] / (u8[0] + u8[1] + eps));
        
        // Market order imbalance
        features.push(u8[2] / (u8[2] + u8[3] + eps));
        
        // Cancel imbalance
        features.push(u8[4] / (u8[4] + u8[5] + eps));
        
        // Limit vs market ratio (buy side)
        features.push(u8[0] / (u8[2] + eps));
        
        // Limit vs market ratio (sell side)
        features.push(u8[1] / (u8[3] + eps));
        
        // Limit vs cancel ratio (buy side)
        features.push(u8[0] / (u8[4] + eps));
        
        // Limit vs cancel ratio (sell side)
        features.push(u8[1] / (u8[5] + eps));
        
        // Total intensity
        let total_intensity: f64 = u8.iter().sum();
        features.push(total_intensity);
        
        // Intensity differences
        features.push(u8[0] - u8[1]); // Limit imbalance
        features.push(u8[2] - u8[3]); // Market imbalance
        features.push(u8[4] - u8[5]); // Cancel imbalance
        
        // Normalized intensities
        if total_intensity > eps {
            for i in 0..6 {
                features.push(u8[i] / total_intensity);
            }
        } else {
            for _ in 0..6 {
                features.push(0.0);
            }
        }
        
        // Net order flow intensity
        features.push((u8[0] + u8[2]) - (u8[1] + u8[3])); // Buy - Sell
        
        features
    }
    
    /// Extract all time-sensitive features (u6-u9)
    pub fn extract_all(&self, current_ts: u64) -> Vec<f64> {
        let mut features = self.extract_u6(current_ts);
        features.extend(self.extract_u7(current_ts));
        features.extend(self.extract_u8(current_ts));
        features.extend(self.extract_u9(current_ts));
        features // Total: 20 + 20 + 6 + 18 = 64 features
    }
}
```

**Impact**:
- Enables FI-2010 benchmark reproduction
- Adds 64 time-sensitive features
- Captures market dynamics that static features miss

---

### 3. Spread-Based Labeling Threshold

**Research Reference**: TLOB Paper (Section 4.1.3)

**Why Critical**: The TLOB paper recommends using **spread-based thresholds** instead of fixed percentages:

> "We also propose using the average spread as a percentage of the mid-price as a more market-adaptive threshold" — TLOB

**Current Implementation**:
```rust
// Fixed threshold (not adaptive)
pub struct LabelConfig {
    pub threshold: f64,  // e.g., 0.002 (0.2%)
}
```

**Required Implementation**:

```rust
/// Threshold strategy for label generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdStrategy {
    /// Fixed percentage threshold (e.g., 0.002 = 0.2%)
    /// Simple but not adaptive to market conditions
    Fixed(f64),
    
    /// Rolling average spread as percentage of mid-price
    /// Adapts to liquidity conditions
    RollingSpread {
        /// Window size for rolling average (number of samples)
        window_size: usize,
        /// Multiplier for the average spread (typically 1.0)
        multiplier: f64,
    },
    
    /// Quantile-based threshold
    /// Sets threshold such that P(up) = P(down) = p, P(stable) = 1-2p
    Quantile {
        /// Target proportion for up/down classes (e.g., 0.3)
        target_proportion: f64,
        /// Window for quantile estimation
        window_size: usize,
    },
    
    /// ATR-based threshold (Average True Range)
    /// Uses volatility measure for threshold
    ATR {
        /// ATR period
        period: usize,
        /// Multiplier for ATR
        multiplier: f64,
    },
}

/// Enhanced label configuration with adaptive thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLabelConfig {
    /// Prediction horizon (h)
    pub horizon: usize,
    
    /// Smoothing window (k)
    pub smoothing_window: usize,
    
    /// Threshold strategy
    pub threshold_strategy: ThresholdStrategy,
}

impl AdaptiveLabelConfig {
    /// Create with rolling spread threshold (TLOB recommended)
    pub fn tlob_recommended(horizon: usize, smoothing_window: usize) -> Self {
        Self {
            horizon,
            smoothing_window,
            threshold_strategy: ThresholdStrategy::RollingSpread {
                window_size: 1000,
                multiplier: 1.0,
            },
        }
    }
}

/// Spread-based threshold calculator
#[derive(Debug)]
pub struct SpreadThresholdCalculator {
    /// Rolling window of spreads as percentage of mid-price
    spread_window: VecDeque<f64>,
    window_size: usize,
    multiplier: f64,
}

impl SpreadThresholdCalculator {
    pub fn new(window_size: usize, multiplier: f64) -> Self {
        Self {
            spread_window: VecDeque::with_capacity(window_size),
            window_size,
            multiplier,
        }
    }
    
    /// Update with new spread observation
    pub fn update(&mut self, spread: f64, mid_price: f64) {
        if mid_price > 0.0 {
            let spread_pct = spread / mid_price;
            self.spread_window.push_back(spread_pct);
            if self.spread_window.len() > self.window_size {
                self.spread_window.pop_front();
            }
        }
    }
    
    /// Get current adaptive threshold
    pub fn current_threshold(&self) -> f64 {
        if self.spread_window.is_empty() {
            return 0.002; // Default fallback
        }
        
        let avg_spread_pct: f64 = 
            self.spread_window.iter().sum::<f64>() / self.spread_window.len() as f64;
        
        avg_spread_pct * self.multiplier
    }
}
```

**Impact**:
- More robust labels for different markets/assets
- Adapts to liquidity conditions
- Aligns with TLOB paper methodology

---

### 4. Multi-Horizon Label Generation

**Research Reference**: FI-2010, DeepLOB, TLOB

**Why Critical**: All benchmark papers evaluate models at **multiple horizons**:

| Paper | Horizons (k) | Meaning |
|-------|-------------|---------|
| FI-2010 | 10, 20, 30, 50, 100 | Number of events |
| DeepLOB | 10, 20, 50, 100 | Tick time steps |
| TLOB | 1, 3, 5, 10, 30, 50 | Seconds |

**Current Implementation**: Single-horizon labeling

**Required Implementation**:

```rust
/// Multi-horizon label generator
/// 
/// Generates labels for multiple prediction horizons simultaneously.
/// Efficient implementation using a single pass over price data.
#[derive(Debug)]
pub struct MultiHorizonLabelGenerator {
    /// Configurations for each horizon
    configs: Vec<LabelConfig>,
    
    /// Price buffer (shared across all horizons)
    prices: VecDeque<f64>,
    
    /// Maximum horizon to determine buffer size
    max_horizon: usize,
    
    /// Maximum smoothing window
    max_smoothing: usize,
}

impl MultiHorizonLabelGenerator {
    /// Create generator for multiple horizons
    /// 
    /// # Arguments
    /// 
    /// * `horizons` - List of prediction horizons
    /// * `smoothing_window` - Smoothing window (shared or per-horizon)
    /// * `threshold_strategy` - Threshold strategy
    pub fn new(
        horizons: &[usize],
        smoothing_window: usize,
        threshold_strategy: ThresholdStrategy,
    ) -> Self {
        let configs: Vec<_> = horizons
            .iter()
            .map(|&h| LabelConfig {
                horizon: h,
                smoothing_window,
                threshold: 0.002, // Will be updated if adaptive
            })
            .collect();
        
        let max_horizon = *horizons.iter().max().unwrap_or(&10);
        
        Self {
            configs,
            prices: VecDeque::with_capacity(max_horizon + smoothing_window * 2 + 1),
            max_horizon,
            max_smoothing: smoothing_window,
        }
    }
    
    /// Add a new mid-price observation
    pub fn add_price(&mut self, price: f64) {
        self.prices.push_back(price);
        
        let required_size = self.max_horizon + self.max_smoothing * 2 + 1;
        while self.prices.len() > required_size {
            self.prices.pop_front();
        }
    }
    
    /// Generate labels for all horizons at current position
    /// 
    /// Returns: Vec of (horizon, label) pairs, or None if insufficient data
    pub fn generate_labels(&self) -> Option<Vec<(usize, TrendLabel)>> {
        // Check if we have enough data for all horizons
        let required = self.max_horizon + self.max_smoothing * 2 + 1;
        if self.prices.len() < required {
            return None;
        }
        
        let mut labels = Vec::with_capacity(self.configs.len());
        
        for config in &self.configs {
            // Compute label for this horizon
            let label = self.compute_label_for_horizon(config)?;
            labels.push((config.horizon, label));
        }
        
        Some(labels)
    }
    
    fn compute_label_for_horizon(&self, config: &LabelConfig) -> Option<TrendLabel> {
        let k = config.smoothing_window;
        let h = config.horizon;
        
        // Position: current is at index (prices.len() - h - k - 1)
        let current_idx = self.prices.len() - h - k - 1;
        
        // Compute past smoothed (w-)
        let past_sum: f64 = (0..=k)
            .map(|i| self.prices[current_idx - i])
            .sum();
        let w_minus = past_sum / (k + 1) as f64;
        
        // Compute future smoothed (w+)
        let future_sum: f64 = (0..=k)
            .map(|i| self.prices[current_idx + h - i])
            .sum();
        let w_plus = future_sum / (k + 1) as f64;
        
        // Compute label
        let pct_change = (w_plus - w_minus) / w_minus;
        
        Some(if pct_change > config.threshold {
            TrendLabel::Up
        } else if pct_change < -config.threshold {
            TrendLabel::Down
        } else {
            TrendLabel::Stable
        })
    }
    
    /// Get horizons being tracked
    pub fn horizons(&self) -> Vec<usize> {
        self.configs.iter().map(|c| c.horizon).collect()
    }
}
```

**Export Format**:

```rust
/// Multi-horizon label export structure
#[derive(Debug, Serialize)]
pub struct MultiHorizonLabels {
    /// Shape: (num_samples, num_horizons)
    pub labels: Array2<i8>,
    
    /// Horizon values for each column
    pub horizons: Vec<usize>,
    
    /// Threshold used for each horizon
    pub thresholds: Vec<f64>,
}

impl NumpyExporter {
    /// Export features with multi-horizon labels
    pub fn export_multi_horizon(
        &self,
        features: &Array2<f64>,
        labels: &MultiHorizonLabels,
        path: &Path,
    ) -> Result<()> {
        // Save features as X.npy
        // Save labels as Y.npy (shape: samples x horizons)
        // Save metadata as config.json
    }
}
```

**Impact**:
- Enables comprehensive model evaluation
- Reproduces benchmark paper results
- Efficient single-pass implementation

---

### 5. Input Tensor Formatting for Models

**Research Reference**: DeepLOB (Section 3.1), TLOB, HLOB

**Why Critical**: Different models expect different input formats:

| Model | Input Shape | Feature Order |
|-------|-------------|---------------|
| DeepLOB | (T, 4, L) | price/vol separated |
| TLOB | (T, F) | flattened |
| HLOB | (T, L, 4) | level-first |
| CNN-based | (T, C, H, W) | image-like |

**Current Implementation**: Flat feature vector only

**Required Implementation**:

```rust
/// Input tensor formatter for different model architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorFormat {
    /// Flat feature vector: (T, F)
    /// Used by: TLOB, MLP, basic LSTM
    Flat,
    
    /// DeepLOB format: (T, 4, L)
    /// Channels: [ask_price, ask_vol, bid_price, bid_vol]
    /// L: number of levels
    DeepLOB { levels: usize },
    
    /// HLOB format: (T, L, 4)
    /// Level-first ordering for topological features
    HLOB { levels: usize },
    
    /// Image format: (T, C, H, W)
    /// For CNN-based models treating LOB as image
    Image { 
        channels: usize,
        height: usize,
        width: usize,
    },
    
    /// Multi-scale format: Dict[scale -> (T, F)]
    /// For models processing multiple time scales
    MultiScale { scales: Vec<usize> },
}

/// Tensor formatter implementation
pub struct TensorFormatter {
    format: TensorFormat,
    lob_levels: usize,
}

impl TensorFormatter {
    pub fn new(format: TensorFormat, lob_levels: usize) -> Self {
        Self { format, lob_levels }
    }
    
    /// Format a sequence of flat feature vectors into model input
    pub fn format(&self, sequence: &Array2<f64>) -> TensorOutput {
        match &self.format {
            TensorFormat::Flat => TensorOutput::Array2(sequence.clone()),
            
            TensorFormat::DeepLOB { levels } => {
                // Reshape from (T, F) to (T, 4, L)
                let (t, f) = sequence.dim();
                assert!(f >= levels * 4, "Insufficient features for DeepLOB format");
                
                let mut tensor = Array3::zeros((t, 4, *levels));
                for i in 0..t {
                    for l in 0..*levels {
                        tensor[[i, 0, l]] = sequence[[i, l]];          // ask_price
                        tensor[[i, 1, l]] = sequence[[i, levels + l]]; // ask_vol
                        tensor[[i, 2, l]] = sequence[[i, 2*levels + l]]; // bid_price
                        tensor[[i, 3, l]] = sequence[[i, 3*levels + l]]; // bid_vol
                    }
                }
                TensorOutput::Array3(tensor)
            }
            
            TensorFormat::HLOB { levels } => {
                // Reshape from (T, F) to (T, L, 4)
                let (t, f) = sequence.dim();
                assert!(f >= levels * 4, "Insufficient features for HLOB format");
                
                let mut tensor = Array3::zeros((t, *levels, 4));
                for i in 0..t {
                    for l in 0..*levels {
                        tensor[[i, l, 0]] = sequence[[i, l]];          // ask_price
                        tensor[[i, l, 1]] = sequence[[i, levels + l]]; // ask_vol
                        tensor[[i, l, 2]] = sequence[[i, 2*levels + l]]; // bid_price
                        tensor[[i, l, 3]] = sequence[[i, 3*levels + l]]; // bid_vol
                    }
                }
                TensorOutput::Array3(tensor)
            }
            
            TensorFormat::Image { channels, height, width } => {
                // Reshape to image format
                let (t, _) = sequence.dim();
                let mut tensor = Array4::zeros((t, *channels, *height, *width));
                // Implementation depends on specific model requirements
                TensorOutput::Array4(tensor)
            }
            
            TensorFormat::MultiScale { scales } => {
                // Would require multi-scale features as input
                unimplemented!("Multi-scale formatting requires MultiScaleWindow output")
            }
        }
    }
}

pub enum TensorOutput {
    Array2(Array2<f64>),
    Array3(Array3<f64>),
    Array4(Array4<f64>),
    MultiScale(HashMap<usize, Array2<f64>>),
}
```

**Impact**:
- Direct compatibility with DeepLOB, HLOB architectures
- Reduces Python preprocessing code
- Enables efficient batch processing

---

## Medium Priority Gaps

### 6. HLOB Topological Features

**Research Reference**: "HLOB – Information Persistence and Structure in Limit Order Books"

**Technical Complexity**: High (requires Information Filtering Network)

```rust
/// HLOB-style topological features
/// 
/// Computes features based on Triangulated Maximally Filtered Graph (TMFG)
/// of volume correlations across price levels.
/// 
/// # Research Background
/// 
/// HLOB uses Information Filtering Networks to unveil deeper dependency
/// structures among volume levels. The TMFG reveals:
/// - Cross-level correlations
/// - Volume clustering patterns
/// - Information flow between levels
pub struct HlobTopologyExtractor {
    /// Number of LOB levels
    levels: usize,
    
    /// Window for correlation calculation
    correlation_window: usize,
    
    /// Volume history for correlation
    volume_history: VecDeque<Vec<f64>>,
}

impl HlobTopologyExtractor {
    /// Extract TMFG-based features
    /// 
    /// Returns: Topological features derived from volume correlation structure
    pub fn extract_features(&self) -> Vec<f64> {
        // 1. Build correlation matrix from volume history
        // 2. Construct TMFG (simplified: use correlation thresholding)
        // 3. Extract graph features: centrality, clustering, etc.
        unimplemented!("Requires graph algorithm implementation")
    }
}
```

**Recommendation**: Consider using Python bindings for NetworkX or implement simplified version.

---

### 7. Order Flow Imbalance (OFI) Enhanced

**Research Reference**: Cont et al. "The Price Impact of Order Book Events"

**Current State**: Basic OFI implemented

**Enhancement Needed**:

```rust
/// Enhanced OFI calculation (Cont et al.)
/// 
/// # Definition
/// 
/// OFI captures the net order flow pressure:
/// 
/// OFI_n = Σ_{i=1}^{n} (e_i^B - e_i^A)
/// 
/// Where:
/// - e_i^B: Bid-side event contribution
/// - e_i^A: Ask-side event contribution
/// 
/// # Event Contributions
/// 
/// For price increase at best bid:
///   e^B = V^B_new - V^B_old
/// 
/// For price decrease at best bid:
///   e^B = -V^B_old
/// 
/// (Symmetric for ask side)
pub struct ContOfiCalculator {
    /// Previous best bid price
    prev_best_bid_price: Option<i64>,
    /// Previous best bid volume
    prev_best_bid_vol: Option<u64>,
    /// Previous best ask price
    prev_best_ask_price: Option<i64>,
    /// Previous best ask volume
    prev_best_ask_vol: Option<u64>,
    
    /// Accumulated OFI
    accumulated_ofi: f64,
    
    /// OFI history for multi-scale analysis
    ofi_history: VecDeque<f64>,
}

impl ContOfiCalculator {
    /// Update OFI with new LOB state
    pub fn update(&mut self, state: &LobState) -> f64 {
        let bid_contribution = self.compute_bid_contribution(state);
        let ask_contribution = self.compute_ask_contribution(state);
        
        let delta_ofi = bid_contribution - ask_contribution;
        self.accumulated_ofi += delta_ofi;
        self.ofi_history.push_back(delta_ofi);
        
        // Update previous state
        self.prev_best_bid_price = state.best_bid.map(|b| b.price);
        self.prev_best_bid_vol = state.best_bid.map(|b| b.size);
        self.prev_best_ask_price = state.best_ask.map(|a| a.price);
        self.prev_best_ask_vol = state.best_ask.map(|a| a.size);
        
        delta_ofi
    }
    
    fn compute_bid_contribution(&self, state: &LobState) -> f64 {
        // Implementation following Cont et al. methodology
        match (self.prev_best_bid_price, state.best_bid) {
            (Some(prev_price), Some(curr)) => {
                if curr.price > prev_price {
                    // Price increased: new volume
                    curr.size as f64
                } else if curr.price < prev_price {
                    // Price decreased: lost previous volume
                    -(self.prev_best_bid_vol.unwrap_or(0) as f64)
                } else {
                    // Same price: volume change
                    curr.size as f64 - self.prev_best_bid_vol.unwrap_or(0) as f64
                }
            }
            _ => 0.0,
        }
    }
    
    fn compute_ask_contribution(&self, state: &LobState) -> f64 {
        // Symmetric to bid side
        // ...
        0.0
    }
}
```

---

### 8. Data Augmentation Support

**Research Reference**: LOBCAST Benchmark (robustness testing)

```rust
/// Data augmentation strategies for LOB data
/// 
/// Improves model robustness by generating synthetic variations.
#[derive(Debug, Clone)]
pub enum AugmentationStrategy {
    /// Add Gaussian noise to features
    GaussianNoise { std: f64 },
    
    /// Random scaling of volumes
    VolumeScaling { min_scale: f64, max_scale: f64 },
    
    /// Time warping (stretch/compress sequences)
    TimeWarp { sigma: f64, knot_count: usize },
    
    /// Dropout levels (randomly zero out some levels)
    LevelDropout { probability: f64 },
    
    /// Synthetic spread widening
    SpreadWidening { max_ticks: i64 },
}

pub struct DataAugmentor {
    strategies: Vec<AugmentationStrategy>,
    rng: rand::rngs::StdRng,
}

impl DataAugmentor {
    /// Apply augmentation to a sequence
    pub fn augment(&mut self, sequence: &Array2<f64>) -> Array2<f64> {
        let mut result = sequence.clone();
        
        for strategy in &self.strategies {
            result = self.apply_strategy(&result, strategy);
        }
        
        result
    }
}
```

---

## Model-Specific Requirements

### DeepLOB Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| 10-level LOB features | ✅ | |
| Rolling z-score normalization | ❌ | **Priority 1** |
| Input shape (T, 4, L) | ❌ | Add TensorFormatter |
| Multi-horizon labels | ⚠️ | Need multi-horizon generator |
| Sequence length T=100 | ✅ | Configurable |

### TLOB Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| 10-level LOB features | ✅ | |
| Spread-based threshold | ❌ | **Priority 3** |
| Decoupled h/k labeling | ✅ | |
| Multiple horizons | ⚠️ | Need batch generation |
| z-score normalization | ❌ | Need rolling |

### FI-2010 Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| u1-u5 basic features | ⚠️ | Partial |
| u6-u9 time features | ❌ | **Priority 2** |
| 10-event sampling | ✅ | EventBasedSampler |
| Multi-horizon (10,20,30,50,100) | ⚠️ | Need multi-gen |
| z-score normalization | ❌ | Need rolling |

### HLOB Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Volume-level features | ✅ | |
| TMFG topology | ❌ | Complex, medium priority |
| Input shape (T, L, 4) | ❌ | Add TensorFormatter |
| Multi-scale support | ✅ | MultiScaleWindow |

### MBO Paper Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| MBO features (36) | ✅ | |
| Order lifecycle | ❌ | Reconstructor gap |
| Trade aggregation | ❌ | Reconstructor gap |
| Tick-based normalization | ✅ | |

---

## Implementation Roadmap

### Phase 1: Foundation (Essential for Training)

**Week 1-2: Normalization & Labeling**

1. **Rolling Z-Score Normalizer**
   - Implement `RollingZScoreNormalizer`
   - Add day boundary integration
   - Test with NVIDIA dataset
   - **Effort**: 6-8 hours

2. **Multi-Horizon Label Generator**
   - Implement `MultiHorizonLabelGenerator`
   - Add to pipeline builder
   - Export format support
   - **Effort**: 4-6 hours

3. **Spread-Based Threshold**
   - Implement `SpreadThresholdCalculator`
   - Add `ThresholdStrategy` enum
   - Integrate with label generators
   - **Effort**: 3-4 hours

### Phase 2: Feature Completeness (Benchmark Reproduction)

**Week 3-4: FI-2010 Features**

4. **FI-2010 Time-Sensitive Features**
   - Implement `Fi2010TimeSensitiveExtractor`
   - Add u6 (price derivatives)
   - Add u7 (volume derivatives)
   - Add u8 (event intensity)
   - Add u9 (relative intensity)
   - **Effort**: 8-12 hours

5. **Input Tensor Formatter**
   - Implement `TensorFormatter`
   - Support DeepLOB, HLOB, Flat formats
   - Add to export pipeline
   - **Effort**: 4-6 hours

### Phase 3: Advanced Features (Model-Specific)

**Week 5-6: Enhancements**

6. **Enhanced OFI (Cont et al.)**
   - Implement `ContOfiCalculator`
   - Multi-scale OFI
   - Integration with feature extractor
   - **Effort**: 4-6 hours

7. **HLOB Topology Features** (Optional)
   - Simplified correlation-based approach
   - Graph feature extraction
   - **Effort**: 8-12 hours

8. **Data Augmentation**
   - Implement `DataAugmentor`
   - Standard augmentation strategies
   - **Effort**: 4-6 hours

---

## Appendix: Feature Count Summary

| Category | Current | After Phase 1 | After Phase 2 | Total |
|----------|---------|---------------|---------------|-------|
| Raw LOB (u1) | 40 | 40 | 40 | 40 |
| Derived | 8 | 8 | 8 | 8 |
| MBO | 36 | 36 | 36 | 36 |
| FI-2010 u2-u5 | 0 | 0 | 40+ | 40+ |
| FI-2010 u6-u9 | 0 | 0 | 64 | 64 |
| Enhanced OFI | 0 | 0 | 0 | 8 |
| HLOB Topology | 0 | 0 | 0 | 20+ |
| **Total** | **84** | **84** | **188+** | **216+** |

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Based on analysis of 22 research papers*

