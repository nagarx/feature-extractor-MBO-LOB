//! Rolling Realized Volatility Estimation
//!
//! This module provides numerically stable volatility estimation using Welford's
//! online algorithm for running variance calculation. Critical for adaptive sampling
//! in financial data processing where precision matters.
//!
//! # Mathematical Foundation
//!
//! **Log Returns**:
//! ```text
//! r_t = ln(P_t / P_{t-1})
//! ```
//!
//! **Realized Volatility** (Welford's Algorithm):
//! ```text
//! For each new return r_i:
//!   n = n + 1
//!   delta = r_i - mean
//!   mean = mean + delta / n
//!   M2 = M2 + delta × (r_i - mean)
//!   
//! variance = M2 / (n - 1)
//! volatility = sqrt(variance)
//! ```
//!
//! # Key Features
//!
//! - **Numerical Stability**: Uses Welford's algorithm (no catastrophic cancellation)
//! - **Rolling Window**: Fixed-size FIFO buffer for recent data
//! - **High Precision**: f64 throughout, careful handling of edge cases
//! - **Zero-Allocation Hot Path**: Pre-allocated VecDeque
//!
//! # Example
//!
//! ```
//! use feature_extractor::preprocessing::VolatilityEstimator;
//!
//! let mut estimator = VolatilityEstimator::new(1000);
//!
//! // Feed mid-prices (need at least 3 prices for 2 returns)
//! for price in &[100.0, 100.5, 101.0, 100.8, 101.2, 100.9, 101.5] {
//!     let volatility = estimator.update(*price);
//!     if let Some(vol) = volatility {
//!         println!("Current volatility: {:.6}", vol);
//!     }
//! }
//!
//! assert!(estimator.is_ready());
//! ```

use std::collections::VecDeque;

/// Rolling realized volatility estimator using Welford's online algorithm.
///
/// Maintains a fixed-size rolling window of log returns and computes running
/// statistics with numerical stability. This is the foundation for adaptive
/// volume-based sampling.
///
/// # Numerical Stability
///
/// Uses Welford's algorithm to avoid catastrophic cancellation that can occur
/// with naive variance calculations (sum of squares - square of sum).
///
/// # Performance
///
/// - Update: O(1) amortized (VecDeque operations)
/// - Memory: O(window_size)
/// - Zero allocations in hot path after initial construction
#[derive(Debug, Clone)]
pub struct VolatilityEstimator {
    /// Maximum number of returns to keep in the rolling window
    window_size: usize,

    /// Rolling buffer of log returns (FIFO)
    returns: VecDeque<f64>,

    /// Running mean of returns (Welford's algorithm)
    mean: f64,

    /// Running M2 for variance calculation (Welford's algorithm)
    /// M2 = sum of squared deviations from the mean
    m2: f64,

    /// Number of returns currently in the window
    count: usize,

    /// Last observed price (for calculating next return)
    last_price: Option<f64>,

    /// Minimum standard deviation to avoid division by zero
    /// Set to 1e-10 (effectively zero volatility)
    min_std: f64,
}

impl VolatilityEstimator {
    /// Create a new volatility estimator with specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of returns to keep in rolling window (e.g., 1000)
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// // Keep last 1000 returns for volatility calculation
    /// let estimator = VolatilityEstimator::new(1000);
    /// ```
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");

        Self {
            window_size,
            returns: VecDeque::with_capacity(window_size),
            mean: 0.0,
            m2: 0.0,
            count: 0,
            last_price: None,
            min_std: 1e-10,
        }
    }

    /// Update with a new mid-price observation.
    ///
    /// Calculates the log return from the last price, updates the rolling window,
    /// and recomputes volatility statistics using Welford's algorithm.
    ///
    /// # Arguments
    ///
    /// * `mid_price` - Current mid-price (must be > 0)
    ///
    /// # Returns
    ///
    /// - `Some(volatility)` if at least 2 samples have been seen
    /// - `None` if insufficient data (first sample, or invalid price)
    ///
    /// # Panics
    ///
    /// Panics if `mid_price` <= 0 (invalid for log returns).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// let mut estimator = VolatilityEstimator::new(100);
    ///
    /// assert_eq!(estimator.update(100.0), None);  // First sample
    /// assert_eq!(estimator.update(100.5), None);  // Second sample (1 return, need 2)
    /// assert!(estimator.update(101.0).is_some()); // Third sample - can calculate now
    /// ```
    pub fn update(&mut self, mid_price: f64) -> Option<f64> {
        assert!(mid_price > 0.0, "mid_price must be > 0");

        // Calculate log return if we have a previous price
        let log_return = match self.last_price {
            Some(last) => {
                // r_t = ln(P_t / P_{t-1})
                (mid_price / last).ln()
            }
            None => {
                // First price, no return yet
                self.last_price = Some(mid_price);
                return None;
            }
        };

        // Update last price for next iteration
        self.last_price = Some(mid_price);

        // Add to rolling window with eviction if necessary
        if self.returns.len() >= self.window_size {
            // Window full - evict oldest return and recalculate statistics
            if let Some(old_return) = self.returns.pop_front() {
                self.remove_return(old_return);
            }
        }

        // Add new return and update statistics
        self.returns.push_back(log_return);
        self.add_return(log_return);

        // Return volatility if we have enough samples
        self.volatility()
    }

    /// Get current realized volatility (standard deviation of returns).
    ///
    /// # Returns
    ///
    /// - `Some(volatility)` if at least 2 samples available
    /// - `None` if insufficient data
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// let mut estimator = VolatilityEstimator::new(100);
    /// estimator.update(100.0);
    /// estimator.update(100.5);
    /// estimator.update(101.0);  // Need 3 prices for 2 returns
    ///
    /// let vol = estimator.volatility().unwrap();
    /// assert!(vol > 0.0);
    /// ```
    pub fn volatility(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }

        // Calculate sample standard deviation
        // std = sqrt(M2 / (n - 1))
        let variance = self.m2 / (self.count - 1) as f64;
        let std = variance.sqrt();

        // Return max of calculated std and minimum (avoid zero)
        Some(std.max(self.min_std))
    }

    /// Get current mean return.
    ///
    /// # Returns
    ///
    /// - `Some(mean)` if at least 1 sample available
    /// - `None` if no data
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// let mut estimator = VolatilityEstimator::new(100);
    /// estimator.update(100.0);
    /// estimator.update(101.0);
    ///
    /// let mean = estimator.mean_return().unwrap();
    /// assert!(mean > 0.0); // Positive drift in this example
    /// ```
    pub fn mean_return(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    /// Get number of returns currently in the window.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// let mut estimator = VolatilityEstimator::new(100);
    /// assert_eq!(estimator.sample_count(), 0);
    ///
    /// estimator.update(100.0);
    /// estimator.update(100.5);
    /// assert_eq!(estimator.sample_count(), 1); // 1 return from 2 prices
    /// ```
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.count
    }

    /// Check if estimator has sufficient samples for reliable volatility estimate.
    ///
    /// Returns true if at least 2 returns are available (minimum for variance).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// let mut estimator = VolatilityEstimator::new(100);
    /// assert!(!estimator.is_ready());
    ///
    /// estimator.update(100.0);
    /// assert!(!estimator.is_ready()); // First price, no returns yet
    ///
    /// estimator.update(100.5);
    /// assert!(!estimator.is_ready()); // 1 return, need 2
    ///
    /// estimator.update(101.0);
    /// assert!(estimator.is_ready());  // 2 returns, now ready
    /// ```
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.count >= 2
    }

    /// Get the configured window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Reset the estimator to initial state.
    ///
    /// Clears all data and statistics. Useful when starting a new trading session
    /// or when market regime changes dramatically.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolatilityEstimator;
    ///
    /// let mut estimator = VolatilityEstimator::new(100);
    /// estimator.update(100.0);
    /// estimator.update(100.5);
    ///
    /// estimator.reset();
    /// assert_eq!(estimator.sample_count(), 0);
    /// assert!(!estimator.is_ready());
    /// ```
    pub fn reset(&mut self) {
        self.returns.clear();
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
        self.last_price = None;
    }

    /// Add a return to the statistics (Welford's algorithm).
    ///
    /// Updates running mean and M2 with the new return value.
    fn add_return(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Remove a return from the statistics (Welford's algorithm in reverse).
    ///
    /// When a return leaves the rolling window, we need to update the running
    /// statistics. This is the inverse of `add_return`.
    fn remove_return(&mut self, value: f64) {
        if self.count == 0 {
            return;
        }

        let delta = value - self.mean;
        self.mean = (self.count as f64 * self.mean - value) / (self.count - 1) as f64;
        let delta2 = value - self.mean;
        self.m2 -= delta * delta2;
        self.count -= 1;

        // Numerical safety: ensure M2 doesn't go negative due to floating point errors
        if self.m2 < 0.0 {
            self.m2 = 0.0;
        }
    }
}

impl Default for VolatilityEstimator {
    /// Create estimator with default window size of 1000.
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-8;

    #[test]
    fn test_new() {
        let estimator = VolatilityEstimator::new(100);
        assert_eq!(estimator.window_size(), 100);
        assert_eq!(estimator.sample_count(), 0);
        assert!(!estimator.is_ready());
        assert_eq!(estimator.volatility(), None);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_new_zero_window() {
        VolatilityEstimator::new(0);
    }

    #[test]
    fn test_first_update() {
        let mut estimator = VolatilityEstimator::new(100);
        let result = estimator.update(100.0);

        assert_eq!(result, None); // No return yet
        assert_eq!(estimator.sample_count(), 0);
        assert!(!estimator.is_ready());
    }

    #[test]
    fn test_second_update() {
        let mut estimator = VolatilityEstimator::new(100);
        estimator.update(100.0);
        let result = estimator.update(100.5);

        // We have 1 return, but need 2 for variance (volatility returns None)
        assert_eq!(result, None);
        assert_eq!(estimator.sample_count(), 1);
        assert!(!estimator.is_ready()); // Need at least 2 returns
    }

    #[test]
    fn test_third_update() {
        let mut estimator = VolatilityEstimator::new(100);
        estimator.update(100.0);
        estimator.update(100.5);
        let result = estimator.update(101.0);

        assert!(result.is_some());
        assert_eq!(estimator.sample_count(), 2);
        assert!(estimator.is_ready()); // Now ready
    }

    #[test]
    #[should_panic(expected = "mid_price must be > 0")]
    fn test_invalid_price_zero() {
        let mut estimator = VolatilityEstimator::new(100);
        estimator.update(0.0);
    }

    #[test]
    #[should_panic(expected = "mid_price must be > 0")]
    fn test_invalid_price_negative() {
        let mut estimator = VolatilityEstimator::new(100);
        estimator.update(-100.0);
    }

    #[test]
    fn test_constant_prices_zero_volatility() {
        let mut estimator = VolatilityEstimator::new(100);

        for _ in 0..10 {
            estimator.update(100.0);
        }

        // Constant prices → zero returns → zero volatility
        let vol = estimator.volatility().unwrap();
        assert!(vol < EPSILON, "Expected near-zero volatility, got {vol}");
    }

    #[test]
    fn test_simple_volatility_calculation() {
        let mut estimator = VolatilityEstimator::new(100);

        // Prices: 100, 101, 99, 102, 98
        // Returns: ln(101/100), ln(99/101), ln(102/99), ln(98/102)
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];

        for &price in &prices {
            estimator.update(price);
        }

        let vol = estimator.volatility().unwrap();

        // Calculate expected volatility manually
        let returns: Vec<f64> = vec![
            (101.0_f64 / 100.0_f64).ln(),
            (99.0_f64 / 101.0_f64).ln(),
            (102.0_f64 / 99.0_f64).ln(),
            (98.0_f64 / 102.0_f64).ln(),
        ];

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let expected_vol = variance.sqrt();

        assert!(
            (vol - expected_vol).abs() < EPSILON,
            "Expected {expected_vol}, got {vol}"
        );
    }

    #[test]
    fn test_rolling_window_eviction() {
        let mut estimator = VolatilityEstimator::new(3); // Small window for testing

        // Add 5 prices (4 returns), window should keep last 3 returns
        estimator.update(100.0);
        estimator.update(101.0); // return 1
        estimator.update(102.0); // return 2
        estimator.update(103.0); // return 3
        estimator.update(104.0); // return 4, evicts return 1

        assert_eq!(estimator.sample_count(), 3);
        assert_eq!(estimator.returns.len(), 3);
    }

    #[test]
    fn test_mean_return() {
        let mut estimator = VolatilityEstimator::new(100);

        assert_eq!(estimator.mean_return(), None);

        estimator.update(100.0);
        estimator.update(102.0); // 2% return

        let mean = estimator.mean_return().unwrap();
        let expected = (102.0_f64 / 100.0_f64).ln();

        assert!((mean - expected).abs() < EPSILON);
    }

    #[test]
    fn test_welford_numerical_stability() {
        // Test Welford's algorithm with large numbers (potential overflow)
        let mut estimator = VolatilityEstimator::new(100);

        let base_price = 1e6; // $1 million
        let prices: Vec<f64> = (0..100)
            .map(|i| base_price * (1.0 + (i as f64) * 0.0001))
            .collect();

        for &price in &prices {
            estimator.update(price);
        }

        let vol = estimator.volatility();
        assert!(vol.is_some());
        assert!(vol.unwrap().is_finite());
        assert!(vol.unwrap() > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut estimator = VolatilityEstimator::new(100);

        for i in 1..=10 {
            estimator.update(100.0 + i as f64);
        }

        assert!(estimator.is_ready());
        assert!(estimator.volatility().is_some());

        estimator.reset();

        assert_eq!(estimator.sample_count(), 0);
        assert!(!estimator.is_ready());
        assert_eq!(estimator.volatility(), None);
        assert_eq!(estimator.mean_return(), None);
    }

    #[test]
    fn test_default() {
        let estimator = VolatilityEstimator::default();
        assert_eq!(estimator.window_size(), 1000);
    }

    #[test]
    fn test_extreme_volatility() {
        let mut estimator = VolatilityEstimator::new(100);

        // Extreme price swings
        let prices = vec![100.0, 150.0, 80.0, 200.0, 50.0];

        for &price in &prices {
            estimator.update(price);
        }

        let vol = estimator.volatility().unwrap();
        assert!(vol > 0.0);
        assert!(vol.is_finite());
        // High volatility expected
        assert!(vol > 0.3, "Expected high volatility, got {vol}");
    }

    #[test]
    fn test_gradual_price_increase() {
        let mut estimator = VolatilityEstimator::new(100);

        // Steady 1% increase
        let mut price = 100.0;
        for _ in 0..50 {
            estimator.update(price);
            price *= 1.01;
        }

        let _vol = estimator.volatility().unwrap();
        let mean = estimator.mean_return().unwrap();

        // Should have positive mean (upward drift)
        assert!(mean > 0.0);

        // Volatility should be low (steady trend)
        let expected_return = 1.01_f64.ln();
        assert!(
            (mean - expected_return).abs() < 0.001,
            "Expected mean return ~{expected_return}, got {mean}"
        );
    }
}
