//! TLOB label generation method.
//!
//! Implements the labeling method from:
//! "TLOB: A Novel Transformer Model with Dual Attention for Price Trend
//! Prediction with Limit Order Book Data" (Berti & Kasneci, 2025)
//!
//! # Key Innovation
//!
//! The TLOB method separates:
//! - Smoothing window size (k) - controls noise reduction
//! - Prediction horizon (h) - the lookahead period
//!
//! This removes the "horizon bias" present in previous labeling methods
//! where k = h (like in DeepLOB/FI-2010).
//!
//! # Mathematical Formulation
//!
//! Given mid-prices p(t) at time t:
//!
//! 1. Past smoothed mid-price (centered at t):
//!    ```text
//!    w-(t,h,k) = (1/(k+1)) * Σ(i=0 to k) p(t-i)
//!    ```
//!
//! 2. Future smoothed mid-price (centered at t+h):
//!    ```text
//!    w+(t,h,k) = (1/(k+1)) * Σ(i=0 to k) p(t+h-i)
//!    ```
//!
//! 3. Percentage change:
//!    ```text
//!    l(t,h,k) = (w+(t,h,k) - w-(t,h,k)) / w-(t,h,k)
//!    ```
//!
//! 4. Classification:
//!    - Upward: l > θ
//!    - Downward: l < -θ
//!    - Stable: -θ ≤ l ≤ θ
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{LabelConfig, TlobLabelGenerator, TrendLabel};
//!
//! let config = LabelConfig {
//!     horizon: 5,
//!     smoothing_window: 2,
//!     threshold: 0.01,
//! };
//!
//! let mut generator = TlobLabelGenerator::new(config);
//!
//! // Simulate upward trending prices
//! let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 0.5).collect();
//! generator.add_prices(&prices);
//!
//! let labels = generator.generate_labels().unwrap();
//! assert!(!labels.is_empty());
//!
//! // With upward trend, most labels should be Up
//! let up_count = labels.iter().filter(|(_, l, _)| *l == TrendLabel::Up).count();
//! assert!(up_count > labels.len() / 2);
//! ```

use super::{LabelConfig, LabelStats, TrendLabel};
use mbo_lob_reconstructor::Result;

/// TLOB label generator.
///
/// Implements the labeling strategy from the TLOB paper with decoupled
/// horizon and smoothing window parameters.
///
/// # Design
///
/// - Maintains a buffer of mid-prices for smoothing
/// - Computes past and future smoothed averages
/// - Classifies based on percentage change relative to threshold
///
/// # Performance
///
/// - O(k) for each label generation (k = smoothing window)
/// - Memory: O(T) where T = sequence length
/// - Optimized for batch processing
///
/// # Thread Safety
///
/// Not thread-safe. Use separate instances for parallel processing.
pub struct TlobLabelGenerator {
    config: LabelConfig,
    prices: Vec<f64>,
}

impl TlobLabelGenerator {
    /// Create a new TLOB label generator with specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Label configuration (horizon, smoothing_window, threshold)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::{LabelConfig, TlobLabelGenerator};
    ///
    /// let config = LabelConfig::default();
    /// let generator = TlobLabelGenerator::new(config);
    /// assert!(generator.is_empty());
    /// ```
    pub fn new(config: LabelConfig) -> Self {
        Self {
            config,
            prices: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity for better performance.
    ///
    /// # Arguments
    ///
    /// * `config` - Label configuration
    /// * `capacity` - Expected number of prices
    pub fn with_capacity(config: LabelConfig, capacity: usize) -> Self {
        Self {
            config,
            prices: Vec::with_capacity(capacity),
        }
    }

    /// Add a mid-price to the buffer.
    ///
    /// Call this for each LOB snapshot as it arrives.
    ///
    /// # Arguments
    ///
    /// * `mid_price` - The mid-price at this timestep
    #[inline]
    pub fn add_price(&mut self, mid_price: f64) {
        self.prices.push(mid_price);
    }

    /// Add multiple prices at once.
    ///
    /// More efficient than calling `add_price` repeatedly.
    ///
    /// # Arguments
    ///
    /// * `prices` - Slice of mid-prices to add
    pub fn add_prices(&mut self, prices: &[f64]) {
        self.prices.extend_from_slice(prices);
    }

    /// Get the number of prices in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Clear the price buffer.
    pub fn clear(&mut self) {
        self.prices.clear();
    }

    /// Get the current configuration.
    pub fn config(&self) -> &LabelConfig {
        &self.config
    }

    /// Check if we have enough prices to generate at least one label.
    pub fn can_generate(&self) -> bool {
        self.prices.len() >= self.config.min_prices_required()
    }

    /// Generate labels for all valid timesteps.
    ///
    /// Returns a vector of (index, label, percentage_change) tuples.
    /// Only generates labels for timesteps where both past and future
    /// smoothing windows are complete.
    ///
    /// # Valid Range
    ///
    /// Labels can be generated for t in [k, T - h - k] where:
    /// - k = smoothing_window
    /// - h = horizon
    /// - T = total number of prices
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<(usize, TrendLabel, f64)>)` - Vector of (index, label, pct_change)
    /// * `Err` - If insufficient data
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::{LabelConfig, TlobLabelGenerator};
    ///
    /// let config = LabelConfig {
    ///     horizon: 2,
    ///     smoothing_window: 1,
    ///     threshold: 0.01,
    /// };
    /// let mut gen = TlobLabelGenerator::new(config);
    ///
    /// // Need at least k + h + k + 1 = 1 + 2 + 1 + 1 = 5 prices
    /// gen.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
    ///
    /// let labels = gen.generate_labels().unwrap();
    /// assert!(!labels.is_empty());
    /// ```
    pub fn generate_labels(&self) -> Result<Vec<(usize, TrendLabel, f64)>> {
        let k = self.config.smoothing_window;
        let h = self.config.horizon;
        let total = self.prices.len();

        // Need at least k past prices, h future steps, and k future smoothing
        let min_length = self.config.min_prices_required();

        if total < min_length {
            return Err(mbo_lob_reconstructor::TlobError::generic(format!(
                "Need at least {min_length} prices for labeling (have {total})"
            )));
        }

        let mut labels = Vec::with_capacity(total - min_length + 1);

        // Generate labels for valid range: [k, total - h - k)
        // This ensures we have k prices before t and k prices after t+h
        for t in k..(total - h - k) {
            // Compute past smoothed mid-price: w-(t,h,k)
            let past_smooth = self.smoothed_past(t, k);

            // Compute future smoothed mid-price: w+(t,h,k)
            let future_smooth = self.smoothed_future(t, h, k);

            // Compute percentage change: l(t,h,k)
            let pct_change = (future_smooth - past_smooth) / past_smooth;

            // Classify based on threshold
            let label = self.classify(pct_change);

            labels.push((t, label, pct_change));
        }

        Ok(labels)
    }

    /// Generate labels and return only the labels (without indices and changes).
    ///
    /// Useful when you only need the label sequence for training.
    pub fn generate_label_sequence(&self) -> Result<Vec<TrendLabel>> {
        let labels = self.generate_labels()?;
        Ok(labels.into_iter().map(|(_, label, _)| label).collect())
    }

    /// Generate labels and return as class indices (0, 1, 2).
    ///
    /// Useful for direct use with cross-entropy loss.
    pub fn generate_class_indices(&self) -> Result<Vec<usize>> {
        let labels = self.generate_labels()?;
        Ok(labels
            .into_iter()
            .map(|(_, label, _)| label.as_class_index())
            .collect())
    }

    /// Compute past smoothed mid-price: w-(t,h,k).
    ///
    /// Averages prices from t-k to t (inclusive), giving k+1 terms.
    #[inline]
    fn smoothed_past(&self, t: usize, k: usize) -> f64 {
        let start = t.saturating_sub(k);
        let end = t + 1;

        let sum: f64 = self.prices[start..end].iter().sum();
        sum / (k + 1) as f64
    }

    /// Compute future smoothed mid-price: w+(t,h,k).
    ///
    /// Averages prices from (t+h-k) to (t+h) (inclusive), giving k+1 terms.
    #[inline]
    fn smoothed_future(&self, t: usize, h: usize, k: usize) -> f64 {
        let center = t + h;
        let start = center.saturating_sub(k);
        let end = center + 1;

        let sum: f64 = self.prices[start..end].iter().sum();
        sum / (k + 1) as f64
    }

    /// Classify percentage change into trend label.
    #[inline]
    fn classify(&self, pct_change: f64) -> TrendLabel {
        let theta = self.config.threshold;

        if pct_change > theta {
            TrendLabel::Up
        } else if pct_change < -theta {
            TrendLabel::Down
        } else {
            TrendLabel::Stable
        }
    }

    /// Compute label statistics for analysis.
    ///
    /// Provides comprehensive statistics about the generated labels
    /// for quality assurance and class balance analysis.
    ///
    /// # Arguments
    ///
    /// * `labels` - Labels generated by `generate_labels()`
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::{LabelConfig, TlobLabelGenerator};
    ///
    /// let config = LabelConfig {
    ///     horizon: 2,
    ///     smoothing_window: 1,
    ///     threshold: 0.01,
    /// };
    /// let mut gen = TlobLabelGenerator::new(config);
    /// gen.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
    ///
    /// let labels = gen.generate_labels().unwrap();
    /// let stats = gen.compute_stats(&labels);
    ///
    /// assert_eq!(stats.total, labels.len());
    /// assert_eq!(stats.up_count + stats.down_count + stats.stable_count, stats.total);
    /// ```
    pub fn compute_stats(&self, labels: &[(usize, TrendLabel, f64)]) -> LabelStats {
        if labels.is_empty() {
            return LabelStats::default();
        }

        let mut up_count = 0;
        let mut down_count = 0;
        let mut stable_count = 0;

        let changes: Vec<f64> = labels
            .iter()
            .map(|(_, label, change)| {
                match label {
                    TrendLabel::Up => up_count += 1,
                    TrendLabel::Down => down_count += 1,
                    TrendLabel::Stable => stable_count += 1,
                }
                *change
            })
            .collect();

        let total = labels.len();
        let avg_change = changes.iter().sum::<f64>() / total as f64;

        let variance = changes
            .iter()
            .map(|&x| (x - avg_change).powi(2))
            .sum::<f64>()
            / total as f64;
        let std_change = variance.sqrt();

        let min_change = changes.iter().copied().fold(f64::INFINITY, f64::min);
        let max_change = changes.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        LabelStats {
            total,
            up_count,
            down_count,
            stable_count,
            avg_change,
            std_change,
            min_change,
            max_change,
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_generator_creation() {
        let config = LabelConfig::default();
        let generator = TlobLabelGenerator::new(config);
        assert_eq!(generator.len(), 0);
        assert!(generator.is_empty());
        assert!(!generator.can_generate());
    }

    #[test]
    fn test_with_capacity() {
        let config = LabelConfig::default();
        let generator = TlobLabelGenerator::with_capacity(config, 1000);
        assert_eq!(generator.len(), 0);
        assert!(generator.is_empty());
    }

    #[test]
    fn test_add_prices() {
        let config = LabelConfig::default();
        let mut generator = TlobLabelGenerator::new(config);

        generator.add_price(100.0);
        generator.add_price(101.0);
        generator.add_price(102.0);

        assert_eq!(generator.len(), 3);
        assert!(!generator.is_empty());
    }

    #[test]
    fn test_add_prices_batch() {
        let config = LabelConfig::default();
        let mut generator = TlobLabelGenerator::new(config);

        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0]);
        assert_eq!(generator.len(), 5);
    }

    #[test]
    fn test_clear() {
        let config = LabelConfig::default();
        let mut generator = TlobLabelGenerator::new(config);

        generator.add_prices(&[100.0, 101.0, 102.0]);
        assert_eq!(generator.len(), 3);

        generator.clear();
        assert_eq!(generator.len(), 0);
        assert!(generator.is_empty());
    }

    #[test]
    fn test_smoothed_past() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 2, // k=2: average of 3 prices
            threshold: 0.002,
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Prices: [100, 101, 102, 103, 104]
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0]);

        // At t=2, past smoothed = avg(100, 101, 102) = 101.0
        let past = generator.smoothed_past(2, 2);
        assert!((past - 101.0).abs() < 1e-10);

        // At t=4, past smoothed = avg(102, 103, 104) = 103.0
        let past = generator.smoothed_past(4, 2);
        assert!((past - 103.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothed_future() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1, // k=1: average of 2 prices
            threshold: 0.002,
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Prices: [100, 101, 102, 103, 104]
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0]);

        // At t=0, h=2, future smoothed = avg(p[2-1], p[2]) = avg(101, 102) = 101.5
        let future = generator.smoothed_future(0, 2, 1);
        assert!((future - 101.5).abs() < 1e-10);
    }

    #[test]
    fn test_classify() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 5,
            threshold: 0.002,
        };
        let generator = TlobLabelGenerator::new(config);

        assert_eq!(generator.classify(0.003), TrendLabel::Up);
        assert_eq!(generator.classify(0.001), TrendLabel::Stable);
        assert_eq!(generator.classify(-0.003), TrendLabel::Down);
        assert_eq!(generator.classify(0.0), TrendLabel::Stable);
        assert_eq!(generator.classify(0.002), TrendLabel::Stable); // At threshold
        assert_eq!(generator.classify(-0.002), TrendLabel::Stable); // At threshold
        assert_eq!(generator.classify(0.00201), TrendLabel::Up); // Just above
        assert_eq!(generator.classify(-0.00201), TrendLabel::Down); // Just below
    }

    #[test]
    fn test_generate_labels_simple() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01, // 1%
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Upward trend: prices increasing
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        let labels = generator.generate_labels().unwrap();

        // Should generate labels for valid range
        assert!(!labels.is_empty());

        // Check that we get reasonable labels
        for (idx, label, change) in &labels {
            assert!(!change.is_nan());
            assert!(change.is_finite());
            assert!(*idx < generator.len());
            // With upward trend, should see Up labels
            assert!(
                *label == TrendLabel::Up || *label == TrendLabel::Stable,
                "Expected Up or Stable for upward trend, got {:?}",
                label
            );
        }
    }

    #[test]
    fn test_generate_labels_downward_trend() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01,
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Downward trend: prices decreasing
        generator.add_prices(&[105.0, 104.0, 103.0, 102.0, 101.0, 100.0]);

        let labels = generator.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // With downward trend, should see Down labels
        let down_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Down)
            .count();
        assert!(
            down_count > 0,
            "Expected some Down labels for downward trend"
        );
    }

    #[test]
    fn test_generate_labels_stable() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01, // 1%
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Stable: prices barely moving (< 1% change)
        generator.add_prices(&[100.0, 100.001, 100.002, 100.001, 100.0, 100.001]);

        let labels = generator.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // With stable prices, should see mostly Stable labels
        let stable_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Stable)
            .count();
        assert!(
            stable_count == labels.len(),
            "Expected all Stable labels for stable prices"
        );
    }

    #[test]
    fn test_insufficient_data() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 5,
            threshold: 0.002,
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Only 5 prices - not enough (need 21 for h=10, k=5)
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0]);

        let result = generator.generate_labels();
        assert!(result.is_err());
    }

    #[test]
    fn test_can_generate() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01,
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Need k + h + k + 1 = 1 + 2 + 1 + 1 = 5 prices
        assert!(!generator.can_generate());

        generator.add_prices(&[100.0, 101.0, 102.0, 103.0]);
        assert!(!generator.can_generate()); // Still not enough

        generator.add_price(104.0);
        assert!(generator.can_generate()); // Now we have enough
    }

    #[test]
    fn test_generate_label_sequence() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01,
        };
        let mut generator = TlobLabelGenerator::new(config);
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        let labels = generator.generate_label_sequence().unwrap();
        assert!(!labels.is_empty());
        assert!(labels
            .iter()
            .all(|l| matches!(l, TrendLabel::Up | TrendLabel::Down | TrendLabel::Stable)));
    }

    #[test]
    fn test_generate_class_indices() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01,
        };
        let mut generator = TlobLabelGenerator::new(config);
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        let indices = generator.generate_class_indices().unwrap();
        assert!(!indices.is_empty());
        assert!(indices.iter().all(|&i| i <= 2)); // Valid class indices
    }

    #[test]
    fn test_label_stats() {
        let config = LabelConfig::default();
        let generator = TlobLabelGenerator::new(config);

        let labels = vec![
            (0, TrendLabel::Up, 0.003),
            (1, TrendLabel::Down, -0.003),
            (2, TrendLabel::Stable, 0.001),
            (3, TrendLabel::Up, 0.005),
        ];

        let stats = generator.compute_stats(&labels);

        assert_eq!(stats.total, 4);
        assert_eq!(stats.up_count, 2);
        assert_eq!(stats.down_count, 1);
        assert_eq!(stats.stable_count, 1);

        let (up_pct, stable_pct, down_pct) = stats.class_balance();
        assert!((up_pct - 0.5).abs() < 1e-10);
        assert!((stable_pct - 0.25).abs() < 1e-10);
        assert!((down_pct - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_label_stats_empty() {
        let config = LabelConfig::default();
        let generator = TlobLabelGenerator::new(config);
        let stats = generator.compute_stats(&[]);

        assert_eq!(stats.total, 0);
        assert_eq!(stats.up_count, 0);
        assert_eq!(stats.down_count, 0);
        assert_eq!(stats.stable_count, 0);
    }

    #[test]
    fn test_deterministic_output() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01,
        };

        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];

        let mut gen1 = TlobLabelGenerator::new(config.clone());
        gen1.add_prices(&prices);
        let labels1 = gen1.generate_labels().unwrap();

        let mut gen2 = TlobLabelGenerator::new(config);
        gen2.add_prices(&prices);
        let labels2 = gen2.generate_labels().unwrap();

        assert_eq!(labels1.len(), labels2.len());
        for (l1, l2) in labels1.iter().zip(labels2.iter()) {
            assert_eq!(l1.0, l2.0); // Same index
            assert_eq!(l1.1, l2.1); // Same label
            assert!((l1.2 - l2.2).abs() < 1e-15); // Same change
        }
    }

    #[test]
    fn test_no_nan_or_inf() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 1,
            threshold: 0.01,
        };
        let mut generator = TlobLabelGenerator::new(config);

        // Various price patterns
        generator.add_prices(&[100.0, 100.0, 100.0, 100.0, 100.0, 100.0]); // Flat
        let labels = generator.generate_labels().unwrap();

        for (_, _, change) in &labels {
            assert!(!change.is_nan(), "Found NaN in percentage change");
            assert!(!change.is_infinite(), "Found Inf in percentage change");
        }
    }
}
