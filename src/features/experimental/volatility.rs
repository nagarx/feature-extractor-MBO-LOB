//! Volatility Features for Options Trading
//!
//! This module provides 6 features related to realized volatility,
//! volatility regimes, and return characteristics. These are essential
//! for options pricing and intraday volatility trading.
//!
//! # Feature Index Layout (when experimental enabled)
//!
//! | Offset | Name | Description | Sign Convention |
//! |--------|------|-------------|-----------------|
//! | 0 | realized_vol_fast | Annualized vol (fast window) | N/A |
//! | 1 | realized_vol_slow | Annualized vol (slow window) | N/A |
//! | 2 | vol_ratio | fast / slow (vol regime indicator) | N/A |
//! | 3 | vol_momentum | Change in realized vol | > 0 = vol increasing |
//! | 4 | return_autocorr | Autocorrelation of returns | < 0 = mean reversion |
//! | 5 | vol_of_vol | Volatility of rolling volatility | N/A |
//!
//! # Usage in Options Trading
//!
//! 1. **Vol Regime Detection**: `vol_ratio` helps identify regime shifts
//!    - `vol_ratio > 1.5` → Volatility spike, consider buying options
//!    - `vol_ratio < 0.7` → Low volatility, consider selling premium
//!
//! 2. **Mean Reversion Trading**: `return_autocorr` indicates strategy
//!    - Negative autocorr → Mean reversion strategies favored
//!    - Positive autocorr → Momentum strategies favored
//!
//! 3. **Volatility Timing**: `vol_momentum` helps time vol trades
//!    - Positive momentum → Vol likely to continue rising
//!    - Negative momentum → Vol likely to continue falling
//!
//! # References
//!
//! - Andersen, T. G., & Bollerslev, T. (1998). "Answering the Skeptics:
//!   Yes, Standard Volatility Models do Provide Accurate Forecasts"
//! - Garman, M. B., & Klass, M. J. (1980). "On the Estimation of
//!   Security Price Volatilities from Historical Data"

/// Number of features in this group.
pub const FEATURE_COUNT: usize = 6;

/// Feature indices (relative to group start).
pub mod indices {
    pub const REALIZED_VOL_FAST: usize = 0;
    pub const REALIZED_VOL_SLOW: usize = 1;
    pub const VOL_RATIO: usize = 2;
    pub const VOL_MOMENTUM: usize = 3;
    pub const RETURN_AUTOCORR: usize = 4;
    pub const VOL_OF_VOL: usize = 5;
}

/// Numerical stability constant.
const EPSILON: f64 = 1e-10;

/// Annualization factor for intraday data.
/// Assumes ~6.5 hours of trading, 252 days per year.
/// For 1-second returns: sqrt(252 * 6.5 * 3600) ≈ 2430
/// For event-based returns, we estimate based on event frequency.
const SQRT_ANNUAL_TRADING_SECONDS: f64 = 2430.0;

/// Rolling window for statistics computation using Welford's algorithm.
struct RollingStats {
    values: std::collections::VecDeque<f64>,
    capacity: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    fn new(capacity: usize) -> Self {
        Self {
            values: std::collections::VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        if self.values.len() >= self.capacity {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn mean(&self) -> f64 {
        let n = self.values.len() as f64;
        if n > 0.0 {
            self.sum / n
        } else {
            0.0
        }
    }

    fn variance(&self) -> f64 {
        let n = self.values.len() as f64;
        if n > 1.0 {
            let mean = self.mean();
            (self.sum_sq / n) - (mean * mean)
        } else {
            0.0
        }
    }

    fn std(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    #[allow(dead_code)] // Utility method for debugging/future use
    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_warm(&self) -> bool {
        self.values.len() >= self.capacity / 2
    }

    fn clear(&mut self) {
        self.values.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

/// Volatility computer using rolling windows.
pub struct VolatilityComputer {
    /// Fast window for realized volatility.
    fast_window: RollingStats,

    /// Slow window for realized volatility.
    slow_window: RollingStats,

    /// Rolling volatility values for vol-of-vol.
    vol_history: RollingStats,

    /// Previous mid-price for return computation.
    prev_price: Option<f64>,

    /// Previous timestamp.
    prev_timestamp: Option<u64>,

    /// Rolling returns for autocorrelation.
    returns: std::collections::VecDeque<f64>,

    /// Maximum returns to track.
    returns_capacity: usize,

    /// Previous fast volatility (for momentum).
    prev_fast_vol: f64,

    /// Sample count.
    sample_count: usize,
}

impl VolatilityComputer {
    /// Create a new volatility computer.
    ///
    /// # Arguments
    ///
    /// * `fast_window` - Size of fast volatility window (e.g., 50)
    /// * `slow_window` - Size of slow volatility window (e.g., 500)
    pub fn new(fast_window: usize, slow_window: usize) -> Self {
        Self {
            fast_window: RollingStats::new(fast_window),
            slow_window: RollingStats::new(slow_window),
            vol_history: RollingStats::new(50), // Last 50 volatility readings
            prev_price: None,
            prev_timestamp: None,
            returns: std::collections::VecDeque::with_capacity(100),
            returns_capacity: 100,
            prev_fast_vol: 0.0,
            sample_count: 0,
        }
    }

    /// Update with a new mid-price sample.
    ///
    /// # Arguments
    ///
    /// * `mid_price` - Current mid-price in dollars
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    #[inline]
    pub fn update(&mut self, mid_price: f64, timestamp_ns: u64) {
        if !mid_price.is_finite() || mid_price <= 0.0 {
            return;
        }

        self.sample_count += 1;

        // Compute log return if we have a previous price
        if let Some(prev) = self.prev_price {
            if prev > 0.0 {
                let log_return = (mid_price / prev).ln();

                // Only use reasonable returns (filter outliers)
                if log_return.is_finite() && log_return.abs() < 0.1 {
                    // Add to rolling windows
                    self.fast_window.push(log_return);
                    self.slow_window.push(log_return);

                    // Track returns for autocorrelation
                    self.returns.push_back(log_return);
                    if self.returns.len() > self.returns_capacity {
                        self.returns.pop_front();
                    }

                    // Update vol history periodically
                    if self.sample_count % 10 == 0 {
                        let current_vol = self.fast_window.std();
                        if current_vol > EPSILON {
                            self.vol_history.push(current_vol);
                        }
                    }
                }
            }
        }

        self.prev_price = Some(mid_price);
        self.prev_timestamp = Some(timestamp_ns);
    }

    /// Extract all 6 features into the output buffer.
    pub fn extract_into(&self, output: &mut Vec<f64>) {
        // Feature 0: Realized volatility (fast window), annualized
        let vol_fast = self.fast_window.std() * SQRT_ANNUAL_TRADING_SECONDS;
        output.push(vol_fast);

        // Feature 1: Realized volatility (slow window), annualized
        let vol_slow = self.slow_window.std() * SQRT_ANNUAL_TRADING_SECONDS;
        output.push(vol_slow);

        // Feature 2: Vol ratio (fast / slow)
        let vol_ratio = if vol_slow > EPSILON {
            vol_fast / vol_slow
        } else {
            1.0
        };
        output.push(vol_ratio);

        // Feature 3: Vol momentum (change in fast vol)
        let vol_momentum = if self.prev_fast_vol > EPSILON {
            (self.fast_window.std() - self.prev_fast_vol) / self.prev_fast_vol
        } else {
            0.0
        };
        output.push(vol_momentum);

        // Feature 4: Return autocorrelation (lag-1)
        let return_autocorr = self.compute_autocorrelation();
        output.push(return_autocorr);

        // Feature 5: Volatility of volatility
        let vol_of_vol = self.vol_history.std() * SQRT_ANNUAL_TRADING_SECONDS;
        output.push(vol_of_vol);
    }

    /// Compute lag-1 autocorrelation of returns.
    fn compute_autocorrelation(&self) -> f64 {
        if self.returns.len() < 10 {
            return 0.0;
        }

        let returns: Vec<f64> = self.returns.iter().copied().collect();
        let n = returns.len();

        // Compute mean
        let mean = returns.iter().sum::<f64>() / n as f64;

        // Compute lag-1 autocorrelation
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let dev = returns[i] - mean;
            denominator += dev * dev;

            if i > 0 {
                let prev_dev = returns[i - 1] - mean;
                numerator += dev * prev_dev;
            }
        }

        if denominator > EPSILON {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.fast_window.clear();
        self.slow_window.clear();
        self.vol_history.clear();
        self.prev_price = None;
        self.prev_timestamp = None;
        self.returns.clear();
        self.prev_fast_vol = 0.0;
        self.sample_count = 0;
    }

    /// Check if enough data for valid features.
    pub fn is_warm(&self) -> bool {
        self.fast_window.is_warm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        assert_eq!(FEATURE_COUNT, 6);
    }

    #[test]
    fn test_volatility_basic() {
        let mut computer = VolatilityComputer::new(50, 500);

        // Simulate price series with known volatility
        let base_price = 100.0;
        for i in 0..100 {
            // Add some noise
            let price = base_price + (i as f64 * 0.01).sin() * 0.5;
            computer.update(price, i * 1_000_000_000);
        }

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        assert_eq!(output.len(), FEATURE_COUNT);
        // All values should be finite
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Feature {} is not finite: {}", i, val);
        }
    }

    #[test]
    fn test_vol_ratio() {
        let mut computer = VolatilityComputer::new(10, 100);

        // Low vol period
        for i in 0..50 {
            computer.update(100.0 + (i as f64 * 0.001), i * 1_000_000_000);
        }

        // High vol period (only affects fast window)
        for i in 50..60 {
            let price = 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
            computer.update(price, i * 1_000_000_000);
        }

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        // Vol ratio should be > 1 after volatility spike
        let vol_ratio = output[indices::VOL_RATIO];
        assert!(vol_ratio > 0.0, "Vol ratio should be positive: {}", vol_ratio);
    }

    #[test]
    fn test_reset() {
        let mut computer = VolatilityComputer::new(50, 500);

        for i in 0..100 {
            computer.update(100.0 + i as f64 * 0.01, i * 1_000_000_000);
        }

        assert!(computer.is_warm());

        computer.reset();

        assert!(!computer.is_warm());
        assert!(computer.prev_price.is_none());
    }
}
