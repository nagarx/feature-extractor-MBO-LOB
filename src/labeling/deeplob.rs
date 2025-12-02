//! DeepLOB label generation method.
//!
//! Implements the simpler labeling method from:
//! "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
//! (Zhang et al., 2019)
//!
//! # Overview
//!
//! The DeepLOB method is simpler than TLOB but has "horizon bias" because
//! the smoothing window size equals the prediction horizon (k = h).
//!
//! # Mathematical Formulation
//!
//! Given mid-prices p(t) at time t:
//!
//! 1. Future average mid-price:
//!    ```text
//!    m+(t) = (1/k) * Σ(i=1 to k) p(t+i)
//!    ```
//!
//! 2. Percentage change (Method 1 - vs current price):
//!    ```text
//!    l_t = (m+(t) - p_t) / p_t
//!    ```
//!
//! 3. Percentage change (Method 2 - vs past average):
//!    ```text
//!    m-(t) = (1/k) * Σ(i=0 to k-1) p(t-i)
//!    l_t = (m+(t) - m-(t)) / m-(t)
//!    ```
//!
//! 4. Classification:
//!    - Upward: l > α
//!    - Downward: l < -α
//!    - Stable: -α ≤ l ≤ α
//!
//! # Comparison with TLOB
//!
//! | Aspect | DeepLOB | TLOB |
//! |--------|---------|------|
//! | Parameters | k (horizon = smoothing) | h (horizon), k (smoothing) |
//! | Horizon bias | Yes | No |
//! | Complexity | Simpler | More flexible |
//! | Use case | Benchmarking | Production |
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{LabelConfig, DeepLobLabelGenerator, TrendLabel};
//!
//! // Note: For DeepLOB, smoothing_window is ignored (uses horizon as k)
//! let config = LabelConfig {
//!     horizon: 10,
//!     smoothing_window: 10,  // Should equal horizon for DeepLOB
//!     threshold: 0.002,
//! };
//!
//! let mut generator = DeepLobLabelGenerator::new(config);
//!
//! // Add prices
//! let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.1).collect();
//! generator.add_prices(&prices);
//!
//! let labels = generator.generate_labels().unwrap();
//! assert!(!labels.is_empty());
//! ```

use super::{LabelConfig, LabelStats, TrendLabel};
use mbo_lob_reconstructor::Result;

/// Labeling method variant for DeepLOB.
///
/// The original DeepLOB paper presents two methods:
/// - Method 1: Compare future average to current price
/// - Method 2: Compare future average to past average (like FI-2010)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeepLobMethod {
    /// l_t = (m+(t) - p_t) / p_t
    ///
    /// Compares future average to current price.
    /// Used in the original DeepLOB paper (Equation 3).
    #[default]
    VsCurrentPrice,

    /// l_t = (m+(t) - m-(t)) / m-(t)
    ///
    /// Compares future average to past average.
    /// Similar to FI-2010 method (Equation 4).
    VsPastAverage,
}

/// DeepLOB label generator.
///
/// Implements the simpler labeling strategy from the DeepLOB paper
/// where k = horizon (no separate smoothing window).
///
/// # Design
///
/// - Uses horizon as both prediction target and smoothing window
/// - Supports two methods: vs current price or vs past average
/// - Simpler but has horizon bias
///
/// # Performance
///
/// - O(k) for each label generation
/// - Memory: O(T) where T = sequence length
pub struct DeepLobLabelGenerator {
    config: LabelConfig,
    method: DeepLobMethod,
    prices: Vec<f64>,
}

impl DeepLobLabelGenerator {
    /// Create a new DeepLOB label generator.
    ///
    /// Uses the default method (VsCurrentPrice).
    ///
    /// # Arguments
    ///
    /// * `config` - Label configuration (horizon used as k, smoothing_window ignored)
    pub fn new(config: LabelConfig) -> Self {
        Self {
            config,
            method: DeepLobMethod::default(),
            prices: Vec::new(),
        }
    }

    /// Create with a specific labeling method.
    ///
    /// # Arguments
    ///
    /// * `config` - Label configuration
    /// * `method` - Which DeepLOB method to use
    pub fn with_method(config: LabelConfig, method: DeepLobMethod) -> Self {
        Self {
            config,
            method,
            prices: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(config: LabelConfig, capacity: usize) -> Self {
        Self {
            config,
            method: DeepLobMethod::default(),
            prices: Vec::with_capacity(capacity),
        }
    }

    /// Add a mid-price to the buffer.
    #[inline]
    pub fn add_price(&mut self, mid_price: f64) {
        self.prices.push(mid_price);
    }

    /// Add multiple prices at once.
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

    /// Get the labeling method.
    pub fn method(&self) -> DeepLobMethod {
        self.method
    }

    /// Calculate minimum required prices.
    ///
    /// For DeepLOB:
    /// - Method 1 (VsCurrentPrice): k + 1 (current + k future)
    /// - Method 2 (VsPastAverage): 2k (k past + k future)
    pub fn min_prices_required(&self) -> usize {
        let k = self.config.horizon;
        match self.method {
            DeepLobMethod::VsCurrentPrice => k + 1,
            DeepLobMethod::VsPastAverage => 2 * k,
        }
    }

    /// Check if we have enough prices to generate at least one label.
    pub fn can_generate(&self) -> bool {
        self.prices.len() >= self.min_prices_required()
    }

    /// Generate labels for all valid timesteps.
    ///
    /// Returns a vector of (index, label, percentage_change) tuples.
    ///
    /// # Valid Range
    ///
    /// - Method 1: t in [0, T - k)
    /// - Method 2: t in [k-1, T - k)
    pub fn generate_labels(&self) -> Result<Vec<(usize, TrendLabel, f64)>> {
        let k = self.config.horizon;
        let total = self.prices.len();
        let min_required = self.min_prices_required();

        if total < min_required {
            return Err(mbo_lob_reconstructor::TlobError::generic(format!(
                "Need at least {min_required} prices for DeepLOB labeling (have {total})"
            )));
        }

        let mut labels = Vec::new();

        match self.method {
            DeepLobMethod::VsCurrentPrice => {
                // l_t = (m+(t) - p_t) / p_t
                // Valid range: [0, T - k)
                for t in 0..(total - k) {
                    let current_price = self.prices[t];
                    let future_avg = self.future_average(t, k);
                    let pct_change = (future_avg - current_price) / current_price;
                    let label = self.classify(pct_change);
                    labels.push((t, label, pct_change));
                }
            }
            DeepLobMethod::VsPastAverage => {
                // l_t = (m+(t) - m-(t)) / m-(t)
                // Valid range: [k-1, T - k)
                // Need k-1 prices before t for past average
                let start = k.saturating_sub(1);
                for t in start..(total - k) {
                    let past_avg = self.past_average(t, k);
                    let future_avg = self.future_average(t, k);
                    let pct_change = (future_avg - past_avg) / past_avg;
                    let label = self.classify(pct_change);
                    labels.push((t, label, pct_change));
                }
            }
        }

        Ok(labels)
    }

    /// Generate labels and return only the labels.
    pub fn generate_label_sequence(&self) -> Result<Vec<TrendLabel>> {
        let labels = self.generate_labels()?;
        Ok(labels.into_iter().map(|(_, label, _)| label).collect())
    }

    /// Generate labels and return as class indices.
    pub fn generate_class_indices(&self) -> Result<Vec<usize>> {
        let labels = self.generate_labels()?;
        Ok(labels
            .into_iter()
            .map(|(_, label, _)| label.as_class_index())
            .collect())
    }

    /// Compute future average: m+(t) = (1/k) * Σ(i=1 to k) p(t+i)
    #[inline]
    fn future_average(&self, t: usize, k: usize) -> f64 {
        let start = t + 1;
        let end = t + k + 1;
        let sum: f64 = self.prices[start..end].iter().sum();
        sum / k as f64
    }

    /// Compute past average: m-(t) = (1/k) * Σ(i=0 to k-1) p(t-i)
    #[inline]
    fn past_average(&self, t: usize, k: usize) -> f64 {
        let start = t.saturating_sub(k - 1);
        let end = t + 1;
        let sum: f64 = self.prices[start..end].iter().sum();
        sum / k as f64
    }

    /// Classify percentage change into trend label.
    #[inline]
    fn classify(&self, pct_change: f64) -> TrendLabel {
        let alpha = self.config.threshold;

        if pct_change > alpha {
            TrendLabel::Up
        } else if pct_change < -alpha {
            TrendLabel::Down
        } else {
            TrendLabel::Stable
        }
    }

    /// Compute label statistics.
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
    fn test_deeplob_creation() {
        let config = LabelConfig::default();
        let generator = DeepLobLabelGenerator::new(config);
        assert!(generator.is_empty());
        assert_eq!(generator.method(), DeepLobMethod::VsCurrentPrice);
    }

    #[test]
    fn test_with_method() {
        let config = LabelConfig::default();
        let gen = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);
        assert_eq!(gen.method(), DeepLobMethod::VsPastAverage);
    }

    #[test]
    fn test_min_prices_method1() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 10,
            threshold: 0.002,
        };
        let gen = DeepLobLabelGenerator::new(config);
        // Method 1: k + 1 = 11
        assert_eq!(gen.min_prices_required(), 11);
    }

    #[test]
    fn test_min_prices_method2() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 10,
            threshold: 0.002,
        };
        let gen = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);
        // Method 2: 2k = 20
        assert_eq!(gen.min_prices_required(), 20);
    }

    #[test]
    fn test_future_average() {
        let config = LabelConfig {
            horizon: 3,
            smoothing_window: 3,
            threshold: 0.01,
        };
        let mut gen = DeepLobLabelGenerator::new(config);
        gen.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0]);

        // At t=0, k=3: avg(p[1], p[2], p[3]) = avg(101, 102, 103) = 102.0
        let avg = gen.future_average(0, 3);
        assert!((avg - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_past_average() {
        let config = LabelConfig {
            horizon: 3,
            smoothing_window: 3,
            threshold: 0.01,
        };
        let mut gen = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);
        gen.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        // At t=2, k=3: avg(p[0], p[1], p[2]) = avg(100, 101, 102) = 101.0
        let avg = gen.past_average(2, 3);
        assert!((avg - 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_labels_method1() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 2,
            threshold: 0.01,
        };
        let mut gen = DeepLobLabelGenerator::new(config);

        // Upward trend
        gen.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0]);

        let labels = gen.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // Check all labels are valid
        for (idx, label, change) in &labels {
            assert!(*idx < gen.len() - 2);
            assert!(!change.is_nan());
            assert!(!change.is_infinite());
            assert!(matches!(
                label,
                TrendLabel::Up | TrendLabel::Down | TrendLabel::Stable
            ));
        }
    }

    #[test]
    fn test_generate_labels_method2() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 2,
            threshold: 0.01,
        };
        let mut gen = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);

        // Upward trend
        gen.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        let labels = gen.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // Should detect upward trend
        let up_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Up)
            .count();
        assert!(up_count > 0);
    }

    #[test]
    fn test_insufficient_data() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 10,
            threshold: 0.002,
        };
        let mut gen = DeepLobLabelGenerator::new(config);
        gen.add_prices(&[100.0, 101.0, 102.0]);

        let result = gen.generate_labels();
        assert!(result.is_err());
    }

    #[test]
    fn test_classify() {
        let config = LabelConfig {
            horizon: 10,
            smoothing_window: 10,
            threshold: 0.002,
        };
        let gen = DeepLobLabelGenerator::new(config);

        assert_eq!(gen.classify(0.003), TrendLabel::Up);
        assert_eq!(gen.classify(-0.003), TrendLabel::Down);
        assert_eq!(gen.classify(0.001), TrendLabel::Stable);
        assert_eq!(gen.classify(0.0), TrendLabel::Stable);
    }

    #[test]
    fn test_label_stats() {
        let config = LabelConfig::default();
        let gen = DeepLobLabelGenerator::new(config);

        let labels = vec![
            (0, TrendLabel::Up, 0.003),
            (1, TrendLabel::Down, -0.003),
            (2, TrendLabel::Stable, 0.001),
        ];

        let stats = gen.compute_stats(&labels);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.up_count, 1);
        assert_eq!(stats.down_count, 1);
        assert_eq!(stats.stable_count, 1);
    }

    #[test]
    fn test_deterministic() {
        let config = LabelConfig {
            horizon: 2,
            smoothing_window: 2,
            threshold: 0.01,
        };
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];

        let mut gen1 = DeepLobLabelGenerator::new(config.clone());
        gen1.add_prices(&prices);
        let labels1 = gen1.generate_labels().unwrap();

        let mut gen2 = DeepLobLabelGenerator::new(config);
        gen2.add_prices(&prices);
        let labels2 = gen2.generate_labels().unwrap();

        assert_eq!(labels1.len(), labels2.len());
        for (l1, l2) in labels1.iter().zip(labels2.iter()) {
            assert_eq!(l1.0, l2.0);
            assert_eq!(l1.1, l2.1);
            assert!((l1.2 - l2.2).abs() < 1e-15);
        }
    }

    #[test]
    fn test_comparison_with_tlob() {
        // When using same parameters, TLOB and DeepLOB Method 2 should be similar
        // (but not identical due to different smoothing formulas)
        let config = LabelConfig {
            horizon: 5,
            smoothing_window: 5,
            threshold: 0.005, // Lower threshold to detect the trend
        };

        // Stronger upward trend to ensure detection
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();

        let mut deeplob =
            DeepLobLabelGenerator::with_method(config.clone(), DeepLobMethod::VsPastAverage);
        deeplob.add_prices(&prices);
        let deeplob_labels = deeplob.generate_labels().unwrap();

        let mut tlob = super::super::TlobLabelGenerator::new(config);
        tlob.add_prices(&prices);
        let tlob_labels = tlob.generate_labels().unwrap();

        // Both should generate some labels
        assert!(!deeplob_labels.is_empty(), "DeepLOB should generate labels");
        assert!(!tlob_labels.is_empty(), "TLOB should generate labels");

        // Both should detect upward trend (at least some Up labels)
        let deeplob_up = deeplob_labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Up)
            .count();
        let tlob_up = tlob_labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Up)
            .count();

        assert!(deeplob_up > 0, "DeepLOB should detect upward trend");
        assert!(tlob_up > 0, "TLOB should detect upward trend");
    }
}
