//! Label Generation for LOB Price Trend Prediction
//!
//! This module provides various labeling strategies for supervised learning
//! with Limit Order Book (LOB) data, aligned with research papers.
//!
//! # Overview
//!
//! Labeling is a critical preprocessing step that converts raw mid-prices into
//! classification targets (Up/Down/Stable) for training deep learning models.
//!
//! # Available Strategies
//!
//! - [`TlobLabelGenerator`]: TLOB paper method with decoupled horizon and smoothing window
//! - [`DeepLobLabelGenerator`]: DeepLOB paper method (simpler, horizon = smoothing window)
//!
//! # Research References
//!
//! - **TLOB**: "TLOB: A Novel Transformer Model with Dual Attention for Price Trend
//!   Prediction with Limit Order Book Data" (Berti & Kasneci, 2025)
//! - **DeepLOB**: "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
//!   (Zhang et al., 2019)
//! - **FI-2010**: "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data
//!   with Machine Learning Methods" (Ntakaris et al., 2018)
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{LabelConfig, TlobLabelGenerator, TrendLabel};
//!
//! // Configure labeling parameters
//! let config = LabelConfig {
//!     horizon: 10,           // Predict 10 steps ahead
//!     smoothing_window: 5,   // Average over 5 prices for smoothing
//!     threshold: 0.002,      // 0.2% threshold for Up/Down classification
//! };
//!
//! let mut generator = TlobLabelGenerator::new(config);
//!
//! // Add mid-prices as they arrive
//! let mid_prices = vec![100.0, 100.1, 100.2, 100.15, 100.3];
//! generator.add_prices(&mid_prices);
//!
//! // Generate labels when enough data is available
//! // let labels = generator.generate_labels()?;
//! ```
//!
//! # Choosing a Labeling Strategy
//!
//! | Strategy | Use Case | Pros | Cons |
//! |----------|----------|------|------|
//! | TLOB | Research, production | Decoupled h/k, less bias | More parameters |
//! | DeepLOB | Benchmarking, comparison | Simple, standard | Horizon bias |
//!
//! # Mathematical Background
//!
//! ## TLOB Method (Recommended)
//!
//! The TLOB method separates the prediction horizon `h` from the smoothing window `k`:
//!
//! ```text
//! w+(t,h,k) = (1/(k+1)) * Σ(i=0 to k) p(t+h-i)   // Future smoothed
//! w-(t,h,k) = (1/(k+1)) * Σ(i=0 to k) p(t-i)     // Past smoothed
//! l(t,h,k) = (w+ - w-) / w-                       // Percentage change
//! ```
//!
//! ## DeepLOB Method
//!
//! The DeepLOB method uses the same value for horizon and smoothing:
//!
//! ```text
//! m+(t) = (1/k) * Σ(i=1 to k) p(t+i)   // Future average
//! l_t = (m+(t) - p_t) / p_t             // Percentage change vs current
//! ```

pub mod deeplob;
pub mod multi_horizon;
pub mod tlob;

// Re-exports for convenience
pub use deeplob::{DeepLobLabelGenerator, DeepLobMethod};
pub use multi_horizon::{
    MultiHorizonConfig, MultiHorizonLabelGenerator, MultiHorizonLabels, MultiHorizonSummary,
    ThresholdStrategy,
};
pub use tlob::TlobLabelGenerator;

// Backward compatibility alias
pub use TlobLabelGenerator as LabelGenerator;

use mbo_lob_reconstructor::Result;
use serde::{Deserialize, Serialize};

// ============================================================================
// Core Types
// ============================================================================

/// Trend label classification for price movement prediction.
///
/// Represents the predicted direction of mid-price movement over a horizon.
///
/// # Variants
///
/// - `Up`: Price is predicted to increase by more than threshold
/// - `Down`: Price is predicted to decrease by more than threshold
/// - `Stable`: Price change is within the threshold (no significant movement)
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::TrendLabel;
///
/// let label = TrendLabel::Up;
/// assert_eq!(label.as_int(), 1);
/// assert_eq!(label.as_class_index(), 2);  // For softmax output
///
/// // Convert from integer
/// let recovered = TrendLabel::from_int(1).unwrap();
/// assert_eq!(recovered, TrendLabel::Up);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrendLabel {
    /// Downward price trend (l < -θ)
    Down = -1,

    /// Stable price (minimal movement, -θ ≤ l ≤ θ)
    Stable = 0,

    /// Upward price trend (l > θ)
    Up = 1,
}

impl TrendLabel {
    /// Convert to integer representation for ML models.
    ///
    /// Returns: -1 (Down), 0 (Stable), 1 (Up)
    #[inline]
    pub fn as_int(&self) -> i8 {
        *self as i8
    }

    /// Convert to class index for softmax output (0-indexed).
    ///
    /// Returns: 0 (Down), 1 (Stable), 2 (Up)
    ///
    /// This is useful for cross-entropy loss where classes are 0-indexed.
    #[inline]
    pub fn as_class_index(&self) -> usize {
        match self {
            TrendLabel::Down => 0,
            TrendLabel::Stable => 1,
            TrendLabel::Up => 2,
        }
    }

    /// Create from integer representation.
    ///
    /// # Arguments
    ///
    /// * `value` - Integer value: -1 (Down), 0 (Stable), 1 (Up)
    ///
    /// # Returns
    ///
    /// `Some(TrendLabel)` if valid, `None` otherwise
    pub fn from_int(value: i8) -> Option<Self> {
        match value {
            -1 => Some(TrendLabel::Down),
            0 => Some(TrendLabel::Stable),
            1 => Some(TrendLabel::Up),
            _ => None,
        }
    }

    /// Create from class index (0-indexed).
    ///
    /// # Arguments
    ///
    /// * `index` - Class index: 0 (Down), 1 (Stable), 2 (Up)
    ///
    /// # Returns
    ///
    /// `Some(TrendLabel)` if valid, `None` otherwise
    pub fn from_class_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(TrendLabel::Down),
            1 => Some(TrendLabel::Stable),
            2 => Some(TrendLabel::Up),
            _ => None,
        }
    }

    /// Get the string name of this label.
    pub fn name(&self) -> &'static str {
        match self {
            TrendLabel::Down => "Down",
            TrendLabel::Stable => "Stable",
            TrendLabel::Up => "Up",
        }
    }
}

impl std::fmt::Display for TrendLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for label generation.
///
/// This configuration is used by all labeling strategies and defines
/// the core parameters for trend classification.
///
/// # Parameters
///
/// - `horizon`: How many steps ahead to predict (h)
/// - `smoothing_window`: How many prices to average for smoothing (k)
/// - `threshold`: Percentage change threshold for Up/Down classification (θ)
///
/// # Common Configurations
///
/// | Use Case | Horizon | Smoothing | Threshold | Notes |
/// |----------|---------|-----------|-----------|-------|
/// | HFT (seconds) | 10 | 5 | 0.0002 | Very short-term |
/// | Short-term (minutes) | 50 | 10 | 0.002 | Standard |
/// | Medium-term (hours) | 100 | 20 | 0.005 | Longer horizon |
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::LabelConfig;
///
/// // Standard configuration (from TLOB paper)
/// let config = LabelConfig::default();
/// assert_eq!(config.horizon, 10);
/// assert_eq!(config.smoothing_window, 5);
/// assert_eq!(config.threshold, 0.002);
///
/// // HFT configuration
/// let hft_config = LabelConfig::hft();
/// assert_eq!(hft_config.horizon, 10);
/// assert_eq!(hft_config.threshold, 0.0002);
///
/// // Custom configuration
/// let custom = LabelConfig {
///     horizon: 50,
///     smoothing_window: 10,
///     threshold: 0.003,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelConfig {
    /// Prediction horizon (h): number of steps ahead to predict.
    ///
    /// Common values: 10, 20, 50, 100 (in tick time)
    pub horizon: usize,

    /// Smoothing window size (k): number of prices to average.
    ///
    /// Recommended: 5-10 for noise reduction without over-smoothing.
    /// For TLOB method, this is independent of horizon.
    /// For DeepLOB method, this equals horizon.
    pub smoothing_window: usize,

    /// Classification threshold (θ): percentage change threshold.
    ///
    /// Common values:
    /// - 0.0002 (2 bps) for HFT
    /// - 0.002 (20 bps) for standard (TLOB paper default)
    /// - 0.005 (50 bps) for longer horizons
    ///
    /// Alternative: Use average spread as percentage of mid-price
    pub threshold: f64,
}

impl Default for LabelConfig {
    /// Default configuration from TLOB paper.
    fn default() -> Self {
        Self {
            horizon: 10,
            smoothing_window: 5,
            threshold: 0.002, // 0.2% - commonly used in TLOB paper
        }
    }
}

impl LabelConfig {
    /// Create a new label configuration.
    pub fn new(horizon: usize, smoothing_window: usize, threshold: f64) -> Self {
        Self {
            horizon,
            smoothing_window,
            threshold,
        }
    }

    /// HFT (High-Frequency Trading) configuration.
    ///
    /// Short horizon with tight threshold for rapid trading.
    pub fn hft() -> Self {
        Self {
            horizon: 10,
            smoothing_window: 5,
            threshold: 0.0002, // 2 bps
        }
    }

    /// Short-term trading configuration.
    ///
    /// Medium horizon for minute-scale predictions.
    pub fn short_term() -> Self {
        Self {
            horizon: 50,
            smoothing_window: 10,
            threshold: 0.002, // 20 bps
        }
    }

    /// Medium-term trading configuration.
    ///
    /// Longer horizon for hour-scale predictions.
    pub fn medium_term() -> Self {
        Self {
            horizon: 100,
            smoothing_window: 20,
            threshold: 0.005, // 50 bps
        }
    }

    /// FI-2010 benchmark configuration.
    ///
    /// Standard configuration used in FI-2010 dataset papers.
    pub fn fi2010(horizon: usize) -> Self {
        Self {
            horizon,
            smoothing_window: horizon, // FI-2010 uses k = h
            threshold: 0.002,
        }
    }

    /// Calculate minimum required prices for label generation.
    ///
    /// For TLOB method: k + h + k + 1 = 2k + h + 1
    /// For DeepLOB method: k + 1 (simpler)
    pub fn min_prices_required(&self) -> usize {
        // TLOB method requires: past smoothing + horizon + future smoothing + 1
        self.smoothing_window + self.horizon + self.smoothing_window + 1
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "Horizon must be > 0",
            ));
        }
        if self.smoothing_window == 0 {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "Smoothing window must be > 0",
            ));
        }
        if self.threshold <= 0.0 {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "Threshold must be > 0",
            ));
        }
        if self.threshold >= 1.0 {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "Threshold must be < 1.0 (100%)",
            ));
        }
        Ok(())
    }
}

/// Label statistics for validation and analysis.
///
/// Provides comprehensive statistics about generated labels for
/// quality assurance and class balance analysis.
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::LabelStats;
///
/// let stats = LabelStats {
///     total: 1000,
///     up_count: 350,
///     down_count: 300,
///     stable_count: 350,
///     avg_change: 0.0001,
///     std_change: 0.002,
///     min_change: -0.01,
///     max_change: 0.01,
/// };
///
/// let (up_pct, stable_pct, down_pct) = stats.class_balance();
/// assert!(stats.is_balanced());
/// ```
#[derive(Debug, Clone)]
pub struct LabelStats {
    /// Total number of labels generated
    pub total: usize,

    /// Number of Up labels
    pub up_count: usize,

    /// Number of Down labels
    pub down_count: usize,

    /// Number of Stable labels
    pub stable_count: usize,

    /// Average percentage change across all labels
    pub avg_change: f64,

    /// Standard deviation of percentage change
    pub std_change: f64,

    /// Minimum percentage change observed
    pub min_change: f64,

    /// Maximum percentage change observed
    pub max_change: f64,
}

impl LabelStats {
    /// Calculate class balance metrics.
    ///
    /// Returns (up_percentage, stable_percentage, down_percentage).
    /// All values are in range [0.0, 1.0].
    pub fn class_balance(&self) -> (f64, f64, f64) {
        if self.total == 0 {
            return (0.0, 0.0, 0.0);
        }
        let total = self.total as f64;
        (
            self.up_count as f64 / total,
            self.stable_count as f64 / total,
            self.down_count as f64 / total,
        )
    }

    /// Check if classes are reasonably balanced (no class > 50%).
    ///
    /// A balanced dataset is important for training deep learning models
    /// to avoid bias toward the majority class.
    pub fn is_balanced(&self) -> bool {
        if self.total == 0 {
            return false;
        }
        let (up_pct, stable_pct, down_pct) = self.class_balance();
        up_pct < 0.5 && stable_pct < 0.5 && down_pct < 0.5
    }

    /// Check if any class is severely underrepresented (< 10%).
    pub fn has_minority_class(&self) -> bool {
        if self.total == 0 {
            return true;
        }
        let (up_pct, stable_pct, down_pct) = self.class_balance();
        up_pct < 0.1 || stable_pct < 0.1 || down_pct < 0.1
    }

    /// Get the majority class.
    pub fn majority_class(&self) -> TrendLabel {
        if self.up_count >= self.down_count && self.up_count >= self.stable_count {
            TrendLabel::Up
        } else if self.down_count >= self.up_count && self.down_count >= self.stable_count {
            TrendLabel::Down
        } else {
            TrendLabel::Stable
        }
    }

    /// Calculate imbalance ratio (max_class / min_class).
    ///
    /// A ratio close to 1.0 indicates good balance.
    /// A ratio > 3.0 suggests significant imbalance.
    pub fn imbalance_ratio(&self) -> f64 {
        let counts = [self.up_count, self.down_count, self.stable_count];
        let max_count = *counts.iter().max().unwrap_or(&1) as f64;
        let min_count = *counts.iter().filter(|&&c| c > 0).min().unwrap_or(&1) as f64;
        if min_count == 0.0 {
            f64::INFINITY
        } else {
            max_count / min_count
        }
    }
}

impl Default for LabelStats {
    fn default() -> Self {
        Self {
            total: 0,
            up_count: 0,
            down_count: 0,
            stable_count: 0,
            avg_change: 0.0,
            std_change: 0.0,
            min_change: 0.0,
            max_change: 0.0,
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
    fn test_trend_label_conversion() {
        assert_eq!(TrendLabel::Up.as_int(), 1);
        assert_eq!(TrendLabel::Stable.as_int(), 0);
        assert_eq!(TrendLabel::Down.as_int(), -1);

        assert_eq!(TrendLabel::from_int(1), Some(TrendLabel::Up));
        assert_eq!(TrendLabel::from_int(0), Some(TrendLabel::Stable));
        assert_eq!(TrendLabel::from_int(-1), Some(TrendLabel::Down));
        assert_eq!(TrendLabel::from_int(2), None);
        assert_eq!(TrendLabel::from_int(-2), None);
    }

    #[test]
    fn test_trend_label_class_index() {
        assert_eq!(TrendLabel::Down.as_class_index(), 0);
        assert_eq!(TrendLabel::Stable.as_class_index(), 1);
        assert_eq!(TrendLabel::Up.as_class_index(), 2);

        assert_eq!(TrendLabel::from_class_index(0), Some(TrendLabel::Down));
        assert_eq!(TrendLabel::from_class_index(1), Some(TrendLabel::Stable));
        assert_eq!(TrendLabel::from_class_index(2), Some(TrendLabel::Up));
        assert_eq!(TrendLabel::from_class_index(3), None);
    }

    #[test]
    fn test_trend_label_name() {
        assert_eq!(TrendLabel::Up.name(), "Up");
        assert_eq!(TrendLabel::Down.name(), "Down");
        assert_eq!(TrendLabel::Stable.name(), "Stable");
    }

    #[test]
    fn test_trend_label_display() {
        assert_eq!(format!("{}", TrendLabel::Up), "Up");
        assert_eq!(format!("{}", TrendLabel::Down), "Down");
        assert_eq!(format!("{}", TrendLabel::Stable), "Stable");
    }

    #[test]
    fn test_label_config_default() {
        let config = LabelConfig::default();
        assert_eq!(config.horizon, 10);
        assert_eq!(config.smoothing_window, 5);
        assert_eq!(config.threshold, 0.002);
    }

    #[test]
    fn test_label_config_presets() {
        let hft = LabelConfig::hft();
        assert_eq!(hft.horizon, 10);
        assert_eq!(hft.threshold, 0.0002);

        let short = LabelConfig::short_term();
        assert_eq!(short.horizon, 50);
        assert_eq!(short.smoothing_window, 10);

        let medium = LabelConfig::medium_term();
        assert_eq!(medium.horizon, 100);
        assert_eq!(medium.smoothing_window, 20);

        let fi2010 = LabelConfig::fi2010(50);
        assert_eq!(fi2010.horizon, 50);
        assert_eq!(fi2010.smoothing_window, 50); // k = h for FI-2010
    }

    #[test]
    fn test_label_config_min_prices() {
        let config = LabelConfig::default(); // h=10, k=5
                                             // TLOB needs: k + h + k + 1 = 5 + 10 + 5 + 1 = 21
        assert_eq!(config.min_prices_required(), 21);
    }

    #[test]
    fn test_label_config_validation() {
        let valid = LabelConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_horizon = LabelConfig::new(0, 5, 0.002);
        assert!(invalid_horizon.validate().is_err());

        let invalid_window = LabelConfig::new(10, 0, 0.002);
        assert!(invalid_window.validate().is_err());

        let invalid_threshold_zero = LabelConfig::new(10, 5, 0.0);
        assert!(invalid_threshold_zero.validate().is_err());

        let invalid_threshold_high = LabelConfig::new(10, 5, 1.5);
        assert!(invalid_threshold_high.validate().is_err());
    }

    #[test]
    fn test_label_stats_class_balance() {
        let stats = LabelStats {
            total: 100,
            up_count: 40,
            down_count: 30,
            stable_count: 30,
            avg_change: 0.001,
            std_change: 0.002,
            min_change: -0.01,
            max_change: 0.01,
        };

        let (up_pct, stable_pct, down_pct) = stats.class_balance();
        assert!((up_pct - 0.4).abs() < 1e-10);
        assert!((stable_pct - 0.3).abs() < 1e-10);
        assert!((down_pct - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_label_stats_is_balanced() {
        let balanced = LabelStats {
            total: 100,
            up_count: 35,
            down_count: 35,
            stable_count: 30,
            ..Default::default()
        };
        assert!(balanced.is_balanced());

        let imbalanced = LabelStats {
            total: 100,
            up_count: 60,
            down_count: 20,
            stable_count: 20,
            ..Default::default()
        };
        assert!(!imbalanced.is_balanced());
    }

    #[test]
    fn test_label_stats_minority_class() {
        let no_minority = LabelStats {
            total: 100,
            up_count: 35,
            down_count: 35,
            stable_count: 30,
            ..Default::default()
        };
        assert!(!no_minority.has_minority_class());

        let has_minority = LabelStats {
            total: 100,
            up_count: 5,
            down_count: 50,
            stable_count: 45,
            ..Default::default()
        };
        assert!(has_minority.has_minority_class());
    }

    #[test]
    fn test_label_stats_majority_class() {
        let stats = LabelStats {
            total: 100,
            up_count: 50,
            down_count: 30,
            stable_count: 20,
            ..Default::default()
        };
        assert_eq!(stats.majority_class(), TrendLabel::Up);
    }

    #[test]
    fn test_label_stats_imbalance_ratio() {
        let balanced = LabelStats {
            total: 99,
            up_count: 33,
            down_count: 33,
            stable_count: 33,
            ..Default::default()
        };
        assert!((balanced.imbalance_ratio() - 1.0).abs() < 0.01);

        let imbalanced = LabelStats {
            total: 100,
            up_count: 60,
            down_count: 20,
            stable_count: 20,
            ..Default::default()
        };
        assert!((imbalanced.imbalance_ratio() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_label_stats_empty() {
        let empty = LabelStats::default();
        assert_eq!(empty.class_balance(), (0.0, 0.0, 0.0));
        assert!(!empty.is_balanced());
        assert!(empty.has_minority_class());
    }
}
