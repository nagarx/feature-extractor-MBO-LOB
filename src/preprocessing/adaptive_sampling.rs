//! Adaptive Volume Threshold for Market-Regime-Aware Sampling
//!
//! This module provides dynamic volume threshold adjustment based on realized
//! market volatility. Critical for NVDA and other volatile stocks where fixed
//! thresholds lead to inconsistent information density.
//!
//! # Problem Statement
//!
//! Fixed volume thresholds (e.g., 500 shares) work poorly across different
//! market regimes:
//! - **Quiet periods**: Too many samples (low information content)
//! - **Volatile periods**: Too few samples (miss important moves)
//!
//! # Solution
//!
//! Adapt the threshold based on current realized volatility:
//! ```text
//! threshold(t) = base_threshold × (RV(t) / baseline_RV)
//!
//! Where:
//!   RV(t) = realized volatility at time t
//!   baseline_RV = median volatility over calibration period
//!   base_threshold = nominal threshold (e.g., 500 shares for NVDA)
//! ```
//!
//! # Mathematical Foundation
//!
//! **Calibration Phase** (first N samples):
//! ```text
//! 1. Collect volatilities: [RV_1, RV_2, ..., RV_N]
//! 2. Calculate baseline: baseline_RV = median(volatilities)
//! 3. Store baseline for adaptive scaling
//! ```
//!
//! **Adaptive Phase**:
//! ```text
//! For each new price:
//!   1. Update volatility: RV(t) = VolatilityEstimator.update(price)
//!   2. Calculate multiplier: m = RV(t) / baseline_RV
//!   3. Enforce bounds: m = clamp(m, min_multiplier, max_multiplier)
//!   4. Calculate threshold: threshold = base_threshold × m
//! ```
//!
//! # Key Features
//!
//! - **Auto-Calibration**: Learns baseline from initial samples
//! - **Bounded Adaptation**: Min/max multipliers prevent extremes
//! - **Numerical Stability**: Uses VolatilityEstimator's Welford algorithm
//! - **Market-Aware**: Samples more in volatile periods, less in quiet periods
//!
//! # Example
//!
//! ```
//! use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
//!
//! // Create adaptive threshold for NVDA (base: 500 shares)
//! let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
//!
//! // Auto-calibrate on first 100 samples (small for example)
//! adaptive.set_calibration_size(100);
//!
//! // Feed mid-prices during trading (need many samples to calibrate)
//! for i in 0..150 {
//!     let price = 500.0 + (i as f64 * 0.1);
//!     let _threshold = adaptive.update(price);
//! }
//!
//! // Check if calibration complete
//! if adaptive.is_calibrated() {
//!     if let Some(vol) = adaptive.baseline_volatility() {
//!         println!("Baseline volatility: {:.6}", vol);
//!     }
//!     println!("Current threshold: {}", adaptive.current_threshold());
//! }
//! ```

use super::volatility::VolatilityEstimator;

/// Adaptive volume threshold calculator based on market volatility.
///
/// Automatically adjusts the volume sampling threshold based on realized
/// market volatility to maintain consistent information density across
/// different market regimes.
///
/// # Lifecycle
///
/// 1. **Calibration Phase**: Collects initial samples to establish baseline
/// 2. **Adaptive Phase**: Dynamically adjusts threshold based on current volatility
///
/// # Thread Safety
///
/// This struct is NOT thread-safe (uses interior mutability).
/// Wrap in `Mutex` if sharing across threads.
#[derive(Debug, Clone)]
pub struct AdaptiveVolumeThreshold {
    /// Base (nominal) volume threshold in shares
    base_threshold: u64,

    /// Volatility estimator for current market state
    volatility_estimator: VolatilityEstimator,

    /// Baseline volatility (calibrated from initial samples)
    /// None until calibration is complete
    baseline_volatility: Option<f64>,

    /// Minimum multiplier (prevents threshold from going too low)
    /// Default: 0.2 (allows threshold to drop to 20% of base)
    min_multiplier: f64,

    /// Maximum multiplier (prevents threshold from going too high)
    /// Default: 5.0 (allows threshold to rise to 500% of base)
    max_multiplier: f64,

    /// Number of samples needed for calibration
    /// Default: 10,000 (about 1-2 hours of NVDA data at 500 share threshold)
    calibration_size: usize,

    /// Volatility samples collected during calibration
    calibration_volatilities: Vec<f64>,

    /// Current adaptive threshold (cached for efficiency)
    current_threshold: u64,
}

impl AdaptiveVolumeThreshold {
    /// Create a new adaptive volume threshold calculator.
    ///
    /// # Arguments
    ///
    /// * `base_threshold` - Nominal volume threshold in shares (e.g., 500 for NVDA)
    /// * `volatility_window` - Rolling window size for volatility calculation (e.g., 1000)
    ///
    /// # Panics
    ///
    /// Panics if `base_threshold` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// // 500 share base, 1000-event volatility window
    /// let adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// ```
    pub fn new(base_threshold: u64, volatility_window: usize) -> Self {
        assert!(base_threshold > 0, "base_threshold must be > 0");

        Self {
            base_threshold,
            volatility_estimator: VolatilityEstimator::new(volatility_window),
            baseline_volatility: None,
            min_multiplier: 0.2,
            max_multiplier: 5.0,
            calibration_size: 10_000,
            calibration_volatilities: Vec::with_capacity(10_000),
            current_threshold: base_threshold,
        }
    }

    /// Set minimum threshold multiplier.
    ///
    /// Prevents threshold from dropping too low during quiet periods.
    ///
    /// # Arguments
    ///
    /// * `multiplier` - Minimum multiplier (e.g., 0.2 = 20% of base)
    ///
    /// # Panics
    ///
    /// Panics if `multiplier` <= 0 or > `max_multiplier`.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// adaptive.set_min_multiplier(0.5); // Don't go below 250 shares
    /// ```
    pub fn set_min_multiplier(&mut self, multiplier: f64) {
        assert!(multiplier > 0.0, "min_multiplier must be > 0");
        assert!(
            multiplier <= self.max_multiplier,
            "min_multiplier must be <= max_multiplier"
        );
        self.min_multiplier = multiplier;
    }

    /// Set maximum threshold multiplier.
    ///
    /// Prevents threshold from rising too high during volatile periods.
    ///
    /// # Arguments
    ///
    /// * `multiplier` - Maximum multiplier (e.g., 5.0 = 500% of base)
    ///
    /// # Panics
    ///
    /// Panics if `multiplier` <= 0 or < `min_multiplier`.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// adaptive.set_max_multiplier(3.0); // Don't exceed 1500 shares
    /// ```
    pub fn set_max_multiplier(&mut self, multiplier: f64) {
        assert!(multiplier > 0.0, "max_multiplier must be > 0");
        assert!(
            multiplier >= self.min_multiplier,
            "max_multiplier must be >= min_multiplier"
        );
        self.max_multiplier = multiplier;
    }

    /// Set calibration sample size.
    ///
    /// Number of samples to collect before calculating baseline volatility.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples (e.g., 10,000)
    ///
    /// # Panics
    ///
    /// Panics if `size` < 100 (need sufficient data for reliable baseline).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// adaptive.set_calibration_size(5000); // Faster calibration
    /// ```
    pub fn set_calibration_size(&mut self, size: usize) {
        assert!(size >= 100, "calibration_size must be >= 100");
        self.calibration_size = size;
        self.calibration_volatilities.reserve(size);
    }

    /// Update with a new mid-price and return the current adaptive threshold.
    ///
    /// During calibration phase, collects volatility samples. After calibration,
    /// dynamically adjusts threshold based on current vs baseline volatility.
    ///
    /// # Arguments
    ///
    /// * `mid_price` - Current mid-price (must be > 0)
    ///
    /// # Returns
    ///
    /// Current volume threshold in shares (u64).
    ///
    /// # Panics
    ///
    /// Panics if `mid_price` <= 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    ///
    /// // During calibration, returns base threshold
    /// let threshold = adaptive.update(500.0);
    /// assert_eq!(threshold, 500);
    ///
    /// // After calibration, adapts to volatility
    /// for price in 500..=600 {
    ///     let threshold = adaptive.update(price as f64);
    ///     // Threshold adjusts based on volatility
    /// }
    /// ```
    pub fn update(&mut self, mid_price: f64) -> u64 {
        assert!(mid_price > 0.0, "mid_price must be > 0");

        // Update volatility estimator
        let current_volatility = self.volatility_estimator.update(mid_price);

        // Check if we're in calibration phase
        if self.baseline_volatility.is_none() {
            return self.calibration_update(current_volatility);
        }

        // Adaptive phase: adjust threshold based on current volatility
        if let Some(vol) = current_volatility {
            self.calculate_adaptive_threshold(vol);
        }

        self.current_threshold
    }

    /// Handle updates during calibration phase.
    ///
    /// Collects volatility samples until we have enough to calculate baseline.
    fn calibration_update(&mut self, current_volatility: Option<f64>) -> u64 {
        // Collect volatility samples
        if let Some(vol) = current_volatility {
            self.calibration_volatilities.push(vol);

            // Check if calibration is complete
            if self.calibration_volatilities.len() >= self.calibration_size {
                self.finalize_calibration();
            }
        }

        // During calibration, use base threshold
        self.current_threshold = self.base_threshold;
        self.current_threshold
    }

    /// Finalize calibration by calculating baseline volatility.
    ///
    /// Uses median (more robust than mean) to establish baseline.
    fn finalize_calibration(&mut self) {
        if self.calibration_volatilities.is_empty() {
            return;
        }

        // Calculate median volatility as baseline (robust to outliers)
        let mut sorted_vols = self.calibration_volatilities.clone();
        sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let baseline = if sorted_vols.len() % 2 == 0 {
            let mid = sorted_vols.len() / 2;
            (sorted_vols[mid - 1] + sorted_vols[mid]) / 2.0
        } else {
            sorted_vols[sorted_vols.len() / 2]
        };

        self.baseline_volatility = Some(baseline);

        // Clear calibration data (no longer needed)
        self.calibration_volatilities.clear();
        self.calibration_volatilities.shrink_to_fit();
    }

    /// Calculate adaptive threshold based on current vs baseline volatility.
    ///
    /// Formula: threshold = base × clamp(current_vol / baseline_vol, min, max)
    fn calculate_adaptive_threshold(&mut self, current_volatility: f64) {
        let baseline = self.baseline_volatility.expect("Must be calibrated");

        // Avoid division by zero (should never happen, but guard)
        if baseline < 1e-10 {
            self.current_threshold = self.base_threshold;
            return;
        }

        // Calculate multiplier
        let multiplier = current_volatility / baseline;

        // Enforce bounds
        let bounded_multiplier = multiplier.clamp(self.min_multiplier, self.max_multiplier);

        // Calculate new threshold (ensure it's at least 1)
        let new_threshold = (self.base_threshold as f64 * bounded_multiplier).round() as u64;
        self.current_threshold = new_threshold.max(1);
    }

    /// Get the current adaptive threshold.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// assert_eq!(adaptive.current_threshold(), 500); // Before calibration
    /// ```
    #[inline]
    pub fn current_threshold(&self) -> u64 {
        self.current_threshold
    }

    /// Get the base threshold.
    #[inline]
    pub fn base_threshold(&self) -> u64 {
        self.base_threshold
    }

    /// Get the baseline volatility (if calibrated).
    ///
    /// # Returns
    ///
    /// - `Some(baseline)` if calibration is complete
    /// - `None` if still calibrating
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// assert_eq!(adaptive.baseline_volatility(), None); // Not calibrated yet
    /// ```
    #[inline]
    pub fn baseline_volatility(&self) -> Option<f64> {
        self.baseline_volatility
    }

    /// Get the current market volatility.
    ///
    /// # Returns
    ///
    /// - `Some(volatility)` if enough samples collected
    /// - `None` if insufficient data
    #[inline]
    pub fn current_volatility(&self) -> Option<f64> {
        self.volatility_estimator.volatility()
    }

    /// Check if calibration is complete.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// assert!(!adaptive.is_calibrated());
    /// ```
    #[inline]
    pub fn is_calibrated(&self) -> bool {
        self.baseline_volatility.is_some()
    }

    /// Get calibration progress (samples collected / samples needed).
    ///
    /// # Returns
    ///
    /// Value between 0.0 (just started) and 1.0 (complete).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// adaptive.set_calibration_size(100);
    ///
    /// assert_eq!(adaptive.calibration_progress(), 0.0);
    ///
    /// for i in 0..50 {
    ///     adaptive.update(100.0 + i as f64);
    /// }
    ///
    /// // Should be around 50%
    /// assert!(adaptive.calibration_progress() > 0.4);
    /// assert!(adaptive.calibration_progress() < 0.6);
    /// ```
    pub fn calibration_progress(&self) -> f64 {
        if self.is_calibrated() {
            return 1.0;
        }

        self.calibration_volatilities.len() as f64 / self.calibration_size as f64
    }

    /// Get the current multiplier (current_vol / baseline_vol).
    ///
    /// Returns None if not calibrated or no current volatility.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// assert_eq!(adaptive.current_multiplier(), None); // Not calibrated
    /// ```
    pub fn current_multiplier(&self) -> Option<f64> {
        let baseline = self.baseline_volatility?;
        let current = self.current_volatility()?;

        if baseline < 1e-10 {
            return Some(1.0); // Avoid division by zero
        }

        Some(current / baseline)
    }

    /// Reset the adaptive threshold to initial state.
    ///
    /// Clears all calibration data and volatility history.
    /// Useful when starting a new trading session.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    ///
    /// // ... collect data ...
    ///
    /// adaptive.reset(); // Start fresh
    /// assert!(!adaptive.is_calibrated());
    /// assert_eq!(adaptive.current_threshold(), 500);
    /// ```
    pub fn reset(&mut self) {
        self.volatility_estimator.reset();
        self.baseline_volatility = None;
        self.calibration_volatilities.clear();
        self.current_threshold = self.base_threshold;
    }

    /// Manually set baseline volatility (skip calibration).
    ///
    /// Useful when baseline is known from prior analysis or configuration.
    ///
    /// # Arguments
    ///
    /// * `baseline` - Baseline volatility value (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `baseline` <= 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::AdaptiveVolumeThreshold;
    ///
    /// let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
    /// adaptive.set_baseline(0.01); // Use known baseline
    ///
    /// assert!(adaptive.is_calibrated());
    /// ```
    pub fn set_baseline(&mut self, baseline: f64) {
        assert!(baseline > 0.0, "baseline must be > 0");
        self.baseline_volatility = Some(baseline);
        self.calibration_volatilities.clear();
        self.calibration_volatilities.shrink_to_fit();
    }
}

impl Default for AdaptiveVolumeThreshold {
    /// Create with default settings (base: 500 shares, window: 1000).
    fn default() -> Self {
        Self::new(500, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const EPSILON: f64 = 1e-8;

    #[test]
    fn test_new() {
        let adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        assert_eq!(adaptive.base_threshold(), 500);
        assert_eq!(adaptive.current_threshold(), 500);
        assert!(!adaptive.is_calibrated());
        assert_eq!(adaptive.calibration_progress(), 0.0);
    }

    #[test]
    #[should_panic(expected = "base_threshold must be > 0")]
    fn test_new_zero_threshold() {
        AdaptiveVolumeThreshold::new(0, 1000);
    }

    #[test]
    fn test_set_multipliers() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);

        adaptive.set_min_multiplier(0.5);
        adaptive.set_max_multiplier(3.0);

        assert_eq!(adaptive.min_multiplier, 0.5);
        assert_eq!(adaptive.max_multiplier, 3.0);
    }

    #[test]
    #[should_panic(expected = "min_multiplier must be > 0")]
    fn test_invalid_min_multiplier_zero() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_min_multiplier(0.0);
    }

    #[test]
    #[should_panic(expected = "min_multiplier must be <= max_multiplier")]
    fn test_invalid_min_multiplier_too_large() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_min_multiplier(10.0); // Exceeds default max of 5.0
    }

    #[test]
    #[should_panic(expected = "max_multiplier must be >= min_multiplier")]
    fn test_invalid_max_multiplier_too_small() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_min_multiplier(0.5);
        adaptive.set_max_multiplier(0.1); // Less than min
    }

    #[test]
    fn test_set_calibration_size() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(5000);
        assert_eq!(adaptive.calibration_size, 5000);
    }

    #[test]
    #[should_panic(expected = "calibration_size must be >= 100")]
    fn test_invalid_calibration_size() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(50);
    }

    #[test]
    fn test_calibration_phase() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(100);

        // During calibration, should return base threshold
        // Note: First 2 prices generate no volatility, so we need 102 total prices
        // to get 100 volatility samples
        for i in 0..101 {
            let threshold = adaptive.update(100.0 + i as f64 * 0.1);
            assert_eq!(threshold, 500);
            if i < 100 {
                assert!(!adaptive.is_calibrated());
            }
        }

        // 102nd sample should complete calibration (100 volatilities collected)
        let threshold = adaptive.update(110.0);
        assert_eq!(threshold, 500); // Still base initially
        assert!(adaptive.is_calibrated());
        assert_eq!(adaptive.calibration_progress(), 1.0);
    }

    #[test]
    fn test_manual_baseline() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);

        adaptive.set_baseline(0.01);

        assert!(adaptive.is_calibrated());
        assert_eq!(adaptive.baseline_volatility(), Some(0.01));
        assert_eq!(adaptive.calibration_progress(), 1.0);
    }

    #[test]
    fn test_adaptive_threshold_high_volatility() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_baseline(0.001); // Very low baseline (manually set, skip calibration)

        // Feed volatile prices with VARYING returns (baseline already set)
        let prices = vec![
            100.0, 105.0, 103.0, 108.0, 106.0, // Varying returns
            111.0, 108.0, 114.0, 112.0, 118.0, 115.0, 121.0, 119.0, 125.0, 123.0, 129.0, 127.0,
            134.0, 132.0, 139.0,
        ];

        for &price in &prices {
            adaptive.update(price);
        }

        // High volatility → higher threshold
        let threshold = adaptive.current_threshold();
        assert!(threshold > 500, "Expected threshold > 500, got {threshold}");
    }

    #[test]
    fn test_adaptive_threshold_low_volatility() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_baseline(0.05); // High baseline (manually set, skip calibration)

        // Feed stable prices (baseline already set)
        for i in 0..20 {
            adaptive.update(100.0 + (i as f64) * 0.01); // Tiny changes
        }

        // Low volatility → lower threshold
        let threshold = adaptive.current_threshold();
        assert!(threshold < 500, "Expected threshold < 500, got {threshold}");
    }

    #[test]
    fn test_multiplier_bounds_enforced() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_min_multiplier(0.5);
        adaptive.set_max_multiplier(2.0);
        adaptive.set_baseline(0.01);

        // Extreme volatility should be clamped
        let mut price = 100.0;
        for _ in 0..50 {
            price *= 1.1; // Extreme jumps
            adaptive.update(price);
        }

        let threshold = adaptive.current_threshold();

        // Should be clamped to max (2.0 × 500 = 1000)
        assert!(
            threshold <= 1000,
            "Expected threshold <= 1000, got {threshold}"
        );
    }

    #[test]
    fn test_constant_prices_neutral_multiplier() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(100);

        // Feed constant prices (zero volatility)
        // First 2 prices generate no volatility
        // Next prices generate zero volatilities
        // Need 102 prices for 100 volatilities (all zero)
        for _ in 0..110 {
            adaptive.update(100.0);
        }

        // Should handle zero volatility gracefully
        let threshold = adaptive.current_threshold();
        assert_eq!(threshold, 500); // Should stay at base
    }

    #[test]
    fn test_reset() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(100);

        // Calibrate (need 102 prices for 100 volatilities)
        for i in 0..102 {
            adaptive.update(100.0 + i as f64);
        }

        assert!(adaptive.is_calibrated());

        // Reset
        adaptive.reset();

        assert!(!adaptive.is_calibrated());
        assert_eq!(adaptive.current_threshold(), 500);
        assert_eq!(adaptive.calibration_progress(), 0.0);
    }

    #[test]
    fn test_calibration_progress() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(100);

        assert_eq!(adaptive.calibration_progress(), 0.0);

        // Need 52 prices for 50 volatilities
        for i in 1..=52 {
            adaptive.update(100.0 + i as f64);
        }

        let progress = adaptive.calibration_progress();
        assert!(
            progress > 0.47 && progress < 0.53,
            "Expected ~0.5, got {progress}"
        );

        // Feed 50 more prices to reach 100 volatilities
        for i in 53..=102 {
            adaptive.update(100.0 + i as f64);
        }

        assert_eq!(adaptive.calibration_progress(), 1.0);
    }

    #[test]
    fn test_current_multiplier() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);

        // Before calibration
        assert_eq!(adaptive.current_multiplier(), None);

        // Set baseline and add data
        adaptive.set_baseline(0.01);

        for i in 0..20 {
            adaptive.update(100.0 + i as f64);
        }

        // After calibration, should have multiplier
        let multiplier = adaptive.current_multiplier();
        assert!(multiplier.is_some());
        assert!(multiplier.unwrap() > 0.0);
    }

    #[test]
    fn test_default() {
        let adaptive = AdaptiveVolumeThreshold::default();
        assert_eq!(adaptive.base_threshold(), 500);
        assert_eq!(adaptive.current_threshold(), 500);
    }

    #[test]
    fn test_median_calculation_odd_size() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(101); // Odd calibration size

        // Need 103 prices to get 101 volatilities (first 2 prices generate 0 volatilities)
        for i in 0..103 {
            adaptive.update(100.0 + (i as f64) * 0.1);
        }

        assert!(adaptive.is_calibrated());
        assert!(adaptive.baseline_volatility().is_some());
    }

    #[test]
    fn test_median_calculation_even_size() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_calibration_size(100); // Even calibration size

        // Need 102 prices to get 100 volatilities
        for i in 0..102 {
            adaptive.update(100.0 + (i as f64) * 0.1);
        }

        assert!(adaptive.is_calibrated());
        assert!(adaptive.baseline_volatility().is_some());
    }

    #[test]
    fn test_threshold_never_zero() {
        let mut adaptive = AdaptiveVolumeThreshold::new(500, 1000);
        adaptive.set_min_multiplier(0.001); // Very low multiplier
        adaptive.set_baseline(1.0); // High baseline

        // Feed very stable prices
        for _ in 0..20 {
            adaptive.update(100.0);
        }

        // Threshold should be at least 1
        assert!(adaptive.current_threshold() >= 1);
    }
}
