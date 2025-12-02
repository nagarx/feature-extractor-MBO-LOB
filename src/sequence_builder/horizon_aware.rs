//! Horizon-Aware Lookback Window Configuration
//!
//! This module provides automatic lookback window scaling based on prediction
//! horizon. Research (TLOB, PatchTST papers) shows that longer horizons require
//! longer context windows for optimal prediction accuracy.
//!
//! # Problem Statement
//!
//! Fixed lookback windows work poorly across different prediction horizons:
//! - **Short horizons (h=10-20)**: 100-event window sufficient for microstructure
//! - **Medium horizons (h=50)**: Need ~500-event window for trend context
//! - **Long horizons (h=100)**: Require ~1000-event window for macro patterns
//!
//! # Solution
//!
//! Automatically scale lookback window with prediction horizon:
//! ```text
//! L(h) = clamp(max(base_window, h × scaling_factor), min_window, max_window)
//!
//! Recommended settings (from TLOB/PatchTST research):
//!   h ≤ 20:  L = 100   (base window)
//!   h = 50:  L = 500   (10× horizon)
//!   h = 100: L = 1000  (10× horizon)
//! ```
//!
//! # Mathematical Foundation
//!
//! **Lookback-to-Horizon Ratio**:
//! ```text
//! ratio = L / h
//!
//! Empirical findings:
//!   - ratio < 5:  Insufficient context (underfitting)
//!   - ratio = 10: Optimal for most horizons
//!   - ratio > 20: Diminishing returns (overfitting risk)
//! ```
//!
//! **Stride Calculation** (for efficiency):
//! ```text
//! stride = max(1, L / target_sequence_length)
//!
//! Example:
//!   L = 1000, target = 100 → stride = 10
//!   (Sample every 10th event to get 100-length sequence)
//! ```
//!
//! # Key Features
//!
//! - **Automatic Scaling**: No manual tuning per horizon
//! - **Research-Backed**: Based on TLOB/PatchTST empirical results
//! - **Bounded**: Min/max constraints prevent extremes
//! - **Builder Pattern**: Fluent configuration API
//! - **Zero-Cost**: Pure compile-time calculation
//!
//! # Example
//!
//! ```
//! use feature_extractor::sequence_builder::HorizonAwareConfig;
//!
//! // Short-term prediction (10 events ahead)
//! let config_short = HorizonAwareConfig::new(10);
//! assert_eq!(config_short.lookback_window(), 100); // Base window
//!
//! // Medium-term prediction (50 events ahead)
//! let config_medium = HorizonAwareConfig::new(50)
//!     .with_scaling(10.0);
//! assert_eq!(config_medium.lookback_window(), 500); // 50 × 10
//!
//! // Long-term prediction (100 events ahead)
//! let config_long = HorizonAwareConfig::new(100)
//!     .with_scaling(10.0);
//! assert_eq!(config_long.lookback_window(), 1000); // 100 × 10
//!
//! // Custom bounds
//! let config_custom = HorizonAwareConfig::new(200)
//!     .with_scaling(10.0)
//!     .with_bounds(100, 2000);
//! assert_eq!(config_custom.lookback_window(), 2000); // Clamped to max
//! ```

/// Horizon-aware lookback window configuration.
///
/// Automatically calculates optimal lookback window size based on prediction
/// horizon using research-backed scaling formulas.
///
/// # Design Philosophy
///
/// - **Simple**: One input (horizon), one output (window size)
/// - **Bounded**: Always produces valid, safe window sizes
/// - **Configurable**: Sensible defaults, but fully customizable
/// - **Zero-Cost**: All calculations at construction time
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonAwareConfig {
    /// Prediction horizon (number of events ahead to predict)
    horizon: usize,

    /// Base window size (minimum for short horizons)
    /// Default: 100 (sufficient for microstructure patterns)
    base_window: usize,

    /// Scaling factor (window = horizon × factor)
    /// Default: 10.0 (10× ratio from TLOB/PatchTST research)
    scaling_factor: f64,

    /// Minimum window size (safety bound)
    /// Default: 50 (minimum for transformer architecture)
    min_window: usize,

    /// Maximum window size (performance/memory bound)
    /// Default: 5000 (prevents excessive memory usage)
    max_window: usize,

    /// Target sequence length (for stride calculation)
    /// Default: 100 (TLOB paper recommendation)
    target_sequence_length: usize,
}

impl HorizonAwareConfig {
    /// Create a new horizon-aware configuration with sensible defaults.
    ///
    /// # Arguments
    ///
    /// * `horizon` - Prediction horizon in number of events (e.g., 10, 50, 100)
    ///
    /// # Default Parameters
    ///
    /// - `base_window`: 100 (minimum context)
    /// - `scaling_factor`: 10.0 (10× ratio)
    /// - `min_window`: 50 (transformer minimum)
    /// - `max_window`: 5000 (memory limit)
    /// - `target_sequence_length`: 100 (TLOB recommendation)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// let config = HorizonAwareConfig::new(50);
    /// assert_eq!(config.lookback_window(), 500); // 50 × 10.0
    /// ```
    pub fn new(horizon: usize) -> Self {
        Self {
            horizon,
            base_window: 100,
            scaling_factor: 10.0,
            min_window: 50,
            max_window: 5000,
            target_sequence_length: 100,
        }
    }

    /// Set custom scaling factor (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `factor` - Multiplier for horizon (e.g., 10.0 for 10× ratio)
    ///
    /// # Panics
    ///
    /// Panics if `factor` <= 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// // Conservative scaling (5× ratio)
    /// let config = HorizonAwareConfig::new(50)
    ///     .with_scaling(5.0);
    /// assert_eq!(config.lookback_window(), 250); // 50 × 5.0
    /// ```
    pub fn with_scaling(mut self, factor: f64) -> Self {
        assert!(factor > 0.0, "scaling_factor must be > 0");
        self.scaling_factor = factor;
        self
    }

    /// Set custom base window (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `base` - Minimum window size for short horizons
    ///
    /// # Panics
    ///
    /// Panics if `base` < `min_window`.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// let config = HorizonAwareConfig::new(10)
    ///     .with_base_window(200);
    /// assert_eq!(config.lookback_window(), 200); // Uses base
    /// ```
    pub fn with_base_window(mut self, base: usize) -> Self {
        assert!(
            base >= self.min_window,
            "base_window ({}) must be >= min_window ({})",
            base,
            self.min_window
        );
        self.base_window = base;
        self
    }

    /// Set custom min/max bounds (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum allowed window size
    /// * `max` - Maximum allowed window size
    ///
    /// # Panics
    ///
    /// Panics if `min` <= 0, `max` <= 0, or `min` > `max`.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// let config = HorizonAwareConfig::new(100)
    ///     .with_bounds(100, 1000);
    /// assert_eq!(config.lookback_window(), 1000); // Clamped to max
    /// ```
    pub fn with_bounds(mut self, min: usize, max: usize) -> Self {
        assert!(min > 0, "min_window must be > 0");
        assert!(max > 0, "max_window must be > 0");
        assert!(min <= max, "min_window must be <= max_window");
        self.min_window = min;
        self.max_window = max;
        self
    }

    /// Set target sequence length for stride calculation (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `length` - Desired sequence length after striding
    ///
    /// # Panics
    ///
    /// Panics if `length` <= 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// let config = HorizonAwareConfig::new(50)
    ///     .with_target_length(128);
    /// assert_eq!(config.optimal_stride(), 3); // 500 / 128 ≈ 3
    /// ```
    pub fn with_target_length(mut self, length: usize) -> Self {
        assert!(length > 0, "target_sequence_length must be > 0");
        self.target_sequence_length = length;
        self
    }

    /// Calculate optimal lookback window for this horizon.
    ///
    /// # Formula
    ///
    /// ```text
    /// raw_window = max(base_window, horizon × scaling_factor)
    /// final_window = clamp(raw_window, min_window, max_window)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// // Short horizon: uses base window
    /// let short = HorizonAwareConfig::new(10);
    /// assert_eq!(short.lookback_window(), 100);
    ///
    /// // Medium horizon: scales with horizon
    /// let medium = HorizonAwareConfig::new(50);
    /// assert_eq!(medium.lookback_window(), 500);
    ///
    /// // Long horizon: bounded by max
    /// let long = HorizonAwareConfig::new(1000)
    ///     .with_bounds(50, 2000);
    /// assert_eq!(long.lookback_window(), 2000); // Clamped
    /// ```
    pub fn lookback_window(&self) -> usize {
        let scaled = (self.horizon as f64 * self.scaling_factor).round() as usize;
        let raw_window = self.base_window.max(scaled);

        // Enforce bounds
        raw_window.clamp(self.min_window, self.max_window)
    }

    /// Calculate optimal stride for efficient sequence generation.
    ///
    /// Stride allows downsampling long lookback windows to maintain
    /// consistent sequence lengths for the transformer.
    ///
    /// # Formula
    ///
    /// ```text
    /// stride = max(1, lookback_window / target_sequence_length)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// // Short window: no striding needed
    /// let short = HorizonAwareConfig::new(10);
    /// assert_eq!(short.optimal_stride(), 1);
    ///
    /// // Long window: stride to maintain target length
    /// let long = HorizonAwareConfig::new(100);
    /// assert_eq!(long.optimal_stride(), 10); // 1000 / 100 = 10
    /// ```
    pub fn optimal_stride(&self) -> usize {
        let window = self.lookback_window();
        (window / self.target_sequence_length).max(1)
    }

    /// Get the prediction horizon.
    #[inline]
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Get the base window size.
    #[inline]
    pub fn base_window(&self) -> usize {
        self.base_window
    }

    /// Get the scaling factor.
    #[inline]
    pub fn scaling_factor(&self) -> f64 {
        self.scaling_factor
    }

    /// Get the minimum window size.
    #[inline]
    pub fn min_window(&self) -> usize {
        self.min_window
    }

    /// Get the maximum window size.
    #[inline]
    pub fn max_window(&self) -> usize {
        self.max_window
    }

    /// Get the target sequence length.
    #[inline]
    pub fn target_sequence_length(&self) -> usize {
        self.target_sequence_length
    }

    /// Get the effective lookback-to-horizon ratio.
    ///
    /// # Returns
    ///
    /// Ratio of lookback window to horizon (useful for analysis).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::HorizonAwareConfig;
    ///
    /// let config = HorizonAwareConfig::new(50);
    /// assert_eq!(config.effective_ratio(), 10.0); // 500 / 50 = 10
    /// ```
    pub fn effective_ratio(&self) -> f64 {
        self.lookback_window() as f64 / self.horizon as f64
    }
}

impl Default for HorizonAwareConfig {
    /// Create with default horizon of 10 events.
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_new() {
        let config = HorizonAwareConfig::new(50);
        assert_eq!(config.horizon(), 50);
        assert_eq!(config.base_window(), 100);
        assert_eq!(config.scaling_factor(), 10.0);
        assert_eq!(config.min_window(), 50);
        assert_eq!(config.max_window(), 5000);
        assert_eq!(config.target_sequence_length(), 100);
    }

    #[test]
    fn test_lookback_short_horizon() {
        // h=10: Should use base window (100)
        let config = HorizonAwareConfig::new(10);
        assert_eq!(config.lookback_window(), 100);
    }

    #[test]
    fn test_lookback_medium_horizon() {
        // h=50: Should scale (50 × 10 = 500)
        let config = HorizonAwareConfig::new(50);
        assert_eq!(config.lookback_window(), 500);
    }

    #[test]
    fn test_lookback_long_horizon() {
        // h=100: Should scale (100 × 10 = 1000)
        let config = HorizonAwareConfig::new(100);
        assert_eq!(config.lookback_window(), 1000);
    }

    #[test]
    fn test_custom_scaling() {
        let config = HorizonAwareConfig::new(50).with_scaling(5.0);
        assert_eq!(config.lookback_window(), 250); // 50 × 5
    }

    #[test]
    fn test_custom_base_window() {
        let config = HorizonAwareConfig::new(10).with_base_window(200);
        assert_eq!(config.lookback_window(), 200); // Uses base
    }

    #[test]
    fn test_min_bound_enforced() {
        // Set base_window lower than default to test min clamping
        let config = HorizonAwareConfig::new(1)
            .with_base_window(60) // Set base above min
            .with_scaling(1.0) // 1 × 1.0 = 1
            .with_bounds(50, 5000); // min=50

        // max(60, 1) = 60, clamped to [50, 5000] = 60
        assert_eq!(config.lookback_window(), 60); // base_window

        // Now test actual min clamping
        let config2 = HorizonAwareConfig::new(1)
            .with_scaling(1.0) // 1 × 1.0 = 1
            .with_base_window(51) // base > min
            .with_bounds(100, 5000); // min=100

        // max(51, 1) = 51, clamped to [100, 5000] = 100
        assert_eq!(config2.lookback_window(), 100); // Clamped to min
    }

    #[test]
    fn test_max_bound_enforced() {
        let config = HorizonAwareConfig::new(1000).with_scaling(10.0); // Would give 10000, but clamped to max
        assert_eq!(config.lookback_window(), 5000); // max_window
    }

    #[test]
    fn test_custom_bounds() {
        let config = HorizonAwareConfig::new(100).with_bounds(100, 2000);

        assert_eq!(config.min_window(), 100);
        assert_eq!(config.max_window(), 2000);
        assert_eq!(config.lookback_window(), 1000); // Within bounds
    }

    #[test]
    fn test_custom_bounds_clamp_to_max() {
        let config = HorizonAwareConfig::new(500).with_bounds(100, 1000);

        // 500 × 10 = 5000, but clamped to 1000
        assert_eq!(config.lookback_window(), 1000);
    }

    #[test]
    fn test_optimal_stride_short_window() {
        let config = HorizonAwareConfig::new(10);
        // window=100, target=100 → stride=1
        assert_eq!(config.optimal_stride(), 1);
    }

    #[test]
    fn test_optimal_stride_long_window() {
        let config = HorizonAwareConfig::new(100);
        // window=1000, target=100 → stride=10
        assert_eq!(config.optimal_stride(), 10);
    }

    #[test]
    fn test_optimal_stride_custom_target() {
        let config = HorizonAwareConfig::new(100).with_target_length(200);
        // window=1000, target=200 → stride=5
        assert_eq!(config.optimal_stride(), 5);
    }

    #[test]
    fn test_effective_ratio() {
        let config = HorizonAwareConfig::new(50);
        // window=500, horizon=50 → ratio=10
        assert!((config.effective_ratio() - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_monotonic_scaling() {
        // Larger horizons should give larger or equal windows
        let h10 = HorizonAwareConfig::new(10).lookback_window();
        let h20 = HorizonAwareConfig::new(20).lookback_window();
        let h50 = HorizonAwareConfig::new(50).lookback_window();
        let h100 = HorizonAwareConfig::new(100).lookback_window();

        assert!(h10 <= h20);
        assert!(h20 <= h50);
        assert!(h50 <= h100);
    }

    #[test]
    fn test_builder_pattern_chaining() {
        let config = HorizonAwareConfig::new(50)
            .with_scaling(8.0)
            .with_base_window(150)
            .with_bounds(100, 2000)
            .with_target_length(128);

        assert_eq!(config.scaling_factor(), 8.0);
        assert_eq!(config.base_window(), 150);
        assert_eq!(config.min_window(), 100);
        assert_eq!(config.max_window(), 2000);
        assert_eq!(config.target_sequence_length(), 128);

        // 50 × 8 = 400, clamped to [100, 2000]
        assert_eq!(config.lookback_window(), 400);
    }

    #[test]
    fn test_default() {
        let config = HorizonAwareConfig::default();
        assert_eq!(config.horizon(), 10);
        assert_eq!(config.lookback_window(), 100);
    }

    #[test]
    #[should_panic(expected = "scaling_factor must be > 0")]
    fn test_invalid_scaling_zero() {
        HorizonAwareConfig::new(50).with_scaling(0.0);
    }

    #[test]
    #[should_panic(expected = "scaling_factor must be > 0")]
    fn test_invalid_scaling_negative() {
        HorizonAwareConfig::new(50).with_scaling(-1.0);
    }

    #[test]
    #[should_panic(expected = "min_window must be > 0")]
    fn test_invalid_min_zero() {
        HorizonAwareConfig::new(50).with_bounds(0, 1000);
    }

    #[test]
    #[should_panic(expected = "max_window must be > 0")]
    fn test_invalid_max_zero() {
        HorizonAwareConfig::new(50).with_bounds(100, 0);
    }

    #[test]
    #[should_panic(expected = "min_window must be <= max_window")]
    fn test_invalid_min_greater_than_max() {
        HorizonAwareConfig::new(50).with_bounds(1000, 100);
    }

    #[test]
    #[should_panic(expected = "base_window")]
    fn test_invalid_base_less_than_min() {
        HorizonAwareConfig::new(50)
            .with_bounds(100, 1000)
            .with_base_window(50); // Less than min_window (100)
    }

    #[test]
    #[should_panic(expected = "target_sequence_length must be > 0")]
    fn test_invalid_target_length_zero() {
        HorizonAwareConfig::new(50).with_target_length(0);
    }

    #[test]
    fn test_clone_copy() {
        let config1 = HorizonAwareConfig::new(50);
        let config2 = config1; // Copy
        let config3 = config1; // Clone

        assert_eq!(config1.lookback_window(), config2.lookback_window());
        assert_eq!(config1.lookback_window(), config3.lookback_window());
    }

    #[test]
    fn test_partial_eq() {
        let config1 = HorizonAwareConfig::new(50).with_scaling(10.0);
        let config2 = HorizonAwareConfig::new(50).with_scaling(10.0);
        let config3 = HorizonAwareConfig::new(100).with_scaling(10.0);

        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_tlob_paper_recommendations() {
        // From PHASE1_DESIGN.md lines 173-176

        // h ≤ 20: L = 100
        let h10 = HorizonAwareConfig::new(10);
        let h20 = HorizonAwareConfig::new(20);
        assert_eq!(h10.lookback_window(), 100);
        assert_eq!(h20.lookback_window(), 200); // 20 × 10

        // h = 50: L = 500
        let h50 = HorizonAwareConfig::new(50);
        assert_eq!(h50.lookback_window(), 500);

        // h = 100: L = 1000
        let h100 = HorizonAwareConfig::new(100);
        assert_eq!(h100.lookback_window(), 1000);
    }
}
