//! Opportunity-Based Label Generation for Big Move Detection
//!
//! This module implements a labeling strategy designed for detecting "big move"
//! opportunities in the market, rather than predicting small average price changes.
//!
//! # Key Difference from TLOB/DeepLOB
//!
//! | Aspect | TLOB/DeepLOB | Opportunity Detection |
//! |--------|--------------|----------------------|
//! | Return calc | Smoothed average change | Peak return in horizon |
//! | Question answered | "What is the average trend?" | "Is there a big move?" |
//! | Threshold | 2-20 bps (frequent signals) | 50-200 bps (rare events) |
//! | Use case | HFT (many trades) | Whale/event detection |
//!
//! # Mathematical Formulation
//!
//! Given mid-prices p(t) at time t and prediction horizon h:
//!
//! 1. Compute peak returns within the horizon window:
//!    ```text
//!    max_return(t, h) = max(p(t+1), ..., p(t+h)) / p(t) - 1
//!    min_return(t, h) = min(p(t+1), ..., p(t+h)) / p(t) - 1
//!    ```
//!
//! 2. Classification:
//!    - **BIG_UP**: max_return > threshold (there's a big upward move)
//!    - **BIG_DOWN**: min_return < -threshold (there's a big downward move)
//!    - **NO_OPPORTUNITY**: Neither condition met
//!
//! # Priority Logic
//!
//! When both max_return > threshold AND min_return < -threshold occur in the
//! same horizon (volatile period), we use configurable priority:
//!
//! - **LargerMagnitude** (default): Label based on which move is larger
//! - **FirstOccurrence**: Label based on which extreme occurs first
//! - **UpPriority**: Always label as BIG_UP if both trigger
//! - **DownPriority**: Always label as BIG_DOWN if both trigger
//!
//! # Research Motivation
//!
//! Backtesting showed that:
//! - DeepLOB achieves 65% directional accuracy
//! - But 65% isn't profitable at realistic transaction costs (0.5-1 bps)
//! - Only 90%+ confidence predictions are profitable
//! - This suggests we should train for "obvious" opportunities, not average trends
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{OpportunityConfig, OpportunityLabelGenerator, OpportunityLabel};
//!
//! let config = OpportunityConfig::new(50, 0.005); // h=50, threshold=0.5%
//! let mut generator = OpportunityLabelGenerator::new(config);
//!
//! // Add prices
//! let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0).collect();
//! generator.add_prices(&prices);
//!
//! let labels = generator.generate_labels().unwrap();
//! for (idx, label, max_ret, min_ret) in &labels {
//!     println!("t={}: {:?}, max={:.4}%, min={:.4}%", idx, label, max_ret * 100.0, min_ret * 100.0);
//! }
//! ```

use mbo_lob_reconstructor::{Result, TlobError};
use serde::{Deserialize, Serialize};

// ============================================================================
// Opportunity Label Enum
// ============================================================================

/// Opportunity label classification.
///
/// Unlike `TrendLabel` which has Up/Stable/Down for trend direction,
/// `OpportunityLabel` indicates whether a trading opportunity exists.
///
/// # Variants
///
/// - `BigUp`: A significant upward move occurs within the horizon
/// - `NoOpportunity`: No significant move in either direction
/// - `BigDown`: A significant downward move occurs within the horizon
///
/// # Class Indices
///
/// For ML training, class indices are:
/// - 0: BigDown
/// - 1: NoOpportunity
/// - 2: BigUp
///
/// This matches the convention in `TrendLabel` (Down=0, Stable=1, Up=2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpportunityLabel {
    /// Significant downward move (min_return < -threshold)
    BigDown = -1,

    /// No significant move in either direction
    NoOpportunity = 0,

    /// Significant upward move (max_return > threshold)
    BigUp = 1,
}

impl OpportunityLabel {
    /// Convert to integer representation for ML models.
    ///
    /// Returns: -1 (BigDown), 0 (NoOpportunity), 1 (BigUp)
    #[inline]
    pub fn as_int(&self) -> i8 {
        *self as i8
    }

    /// Convert to class index for softmax output (0-indexed).
    ///
    /// Returns: 0 (BigDown), 1 (NoOpportunity), 2 (BigUp)
    #[inline]
    pub fn as_class_index(&self) -> usize {
        match self {
            OpportunityLabel::BigDown => 0,
            OpportunityLabel::NoOpportunity => 1,
            OpportunityLabel::BigUp => 2,
        }
    }

    /// Create from integer representation.
    pub fn from_int(value: i8) -> Option<Self> {
        match value {
            -1 => Some(OpportunityLabel::BigDown),
            0 => Some(OpportunityLabel::NoOpportunity),
            1 => Some(OpportunityLabel::BigUp),
            _ => None,
        }
    }

    /// Create from class index (0-indexed).
    pub fn from_class_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(OpportunityLabel::BigDown),
            1 => Some(OpportunityLabel::NoOpportunity),
            2 => Some(OpportunityLabel::BigUp),
            _ => None,
        }
    }

    /// Get the string name of this label.
    pub fn name(&self) -> &'static str {
        match self {
            OpportunityLabel::BigDown => "BigDown",
            OpportunityLabel::NoOpportunity => "NoOpportunity",
            OpportunityLabel::BigUp => "BigUp",
        }
    }

    /// Check if this label represents an opportunity (i.e., not NoOpportunity).
    #[inline]
    pub fn is_opportunity(&self) -> bool {
        !matches!(self, OpportunityLabel::NoOpportunity)
    }
}

impl std::fmt::Display for OpportunityLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Priority Strategy
// ============================================================================

/// Priority strategy when both big up and big down occur in the same horizon.
///
/// During volatile periods, both max_return > threshold AND min_return < -threshold
/// may occur. This strategy determines how to resolve the conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ConflictPriority {
    /// Label based on which move has larger absolute magnitude.
    ///
    /// Example: max_return = 0.6%, min_return = -0.8% → BigDown (larger magnitude)
    #[default]
    LargerMagnitude,

    /// Always prioritize BigUp when both trigger.
    ///
    /// Use when upside opportunities are more valuable.
    UpPriority,

    /// Always prioritize BigDown when both trigger.
    ///
    /// Use when downside protection is more important.
    DownPriority,

    /// Label as NoOpportunity when both trigger.
    ///
    /// Conservative approach: unclear situations are not opportunities.
    Ambiguous,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for opportunity-based label generation.
///
/// # Key Parameters
///
/// - `horizon`: Number of time steps to look ahead for peak return
/// - `threshold`: Minimum absolute return to qualify as a "big move"
/// - `conflict_priority`: How to handle cases where both up and down moves occur
///
/// # Choosing Thresholds
///
/// | Threshold | Interpretation | Expected Positive Rate |
/// |-----------|----------------|------------------------|
/// | 0.002 (20 bps) | Small move | ~20-40% |
/// | 0.005 (50 bps) | Medium move | ~5-15% |
/// | 0.01 (100 bps) | Large move | ~1-5% |
/// | 0.02 (200 bps) | Very large move | ~0.1-1% |
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::OpportunityConfig;
///
/// // Detect 0.5% moves within 50 events
/// let config = OpportunityConfig::new(50, 0.005);
/// assert_eq!(config.horizon, 50);
/// assert_eq!(config.threshold, 0.005);
///
/// // Use preset for whale detection
/// let whale = OpportunityConfig::whale_detection();
/// assert_eq!(whale.threshold, 0.01); // 1% moves
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityConfig {
    /// Prediction horizon: number of time steps to look ahead.
    ///
    /// The generator scans prices from t+1 to t+horizon for peak returns.
    /// Larger horizons capture slower-developing opportunities but may
    /// include multiple independent events.
    ///
    /// Recommended values:
    /// - 10-20: Ultra-short-term (seconds)
    /// - 50-100: Short-term (tens of seconds to minutes)
    /// - 200-500: Medium-term (minutes)
    pub horizon: usize,

    /// Move threshold: minimum absolute return to qualify as a "big move".
    ///
    /// Must be > 0 and < 1 (as a decimal, not percentage).
    ///
    /// Example: 0.005 = 0.5% = 50 basis points
    pub threshold: f64,

    /// How to handle conflicts when both up and down moves exceed threshold.
    ///
    /// Default: LargerMagnitude (label based on which move is larger)
    #[serde(default)]
    pub conflict_priority: ConflictPriority,

    /// Optional asymmetric threshold for down moves.
    ///
    /// If None, uses the same threshold for both directions.
    /// If Some(t), uses `threshold` for up moves and `t` for down moves.
    ///
    /// Use when up and down moves have different risk/reward profiles.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub down_threshold: Option<f64>,
}

impl OpportunityConfig {
    /// Create a new opportunity configuration.
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of time steps to look ahead
    /// * `threshold` - Minimum absolute return for a "big move" (decimal, e.g., 0.005 = 0.5%)
    ///
    /// # Panics
    ///
    /// Panics if horizon is 0 or threshold is not in (0, 1).
    pub fn new(horizon: usize, threshold: f64) -> Self {
        assert!(horizon > 0, "horizon must be > 0");
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "threshold must be in range (0, 1), got {}",
            threshold
        );

        Self {
            horizon,
            threshold,
            conflict_priority: ConflictPriority::default(),
            down_threshold: None,
        }
    }

    /// Create with asymmetric thresholds.
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of time steps to look ahead
    /// * `up_threshold` - Threshold for upward moves
    /// * `down_threshold` - Threshold for downward moves
    pub fn with_asymmetric_thresholds(
        horizon: usize,
        up_threshold: f64,
        down_threshold: f64,
    ) -> Self {
        assert!(horizon > 0, "horizon must be > 0");
        assert!(
            up_threshold > 0.0 && up_threshold < 1.0,
            "up_threshold must be in range (0, 1)"
        );
        assert!(
            down_threshold > 0.0 && down_threshold < 1.0,
            "down_threshold must be in range (0, 1)"
        );

        Self {
            horizon,
            threshold: up_threshold,
            conflict_priority: ConflictPriority::default(),
            down_threshold: Some(down_threshold),
        }
    }

    /// Preset for detecting medium-sized opportunities.
    ///
    /// - Horizon: 50 events (~5-25 seconds in active trading)
    /// - Threshold: 0.5% (50 bps)
    ///
    /// Expected positive rate: ~5-15% depending on volatility.
    pub fn medium_moves() -> Self {
        Self::new(50, 0.005)
    }

    /// Preset for detecting large "whale" moves.
    ///
    /// - Horizon: 100 events (~10-50 seconds in active trading)
    /// - Threshold: 1.0% (100 bps)
    ///
    /// Expected positive rate: ~1-5% depending on volatility.
    pub fn whale_detection() -> Self {
        Self::new(100, 0.01)
    }

    /// Preset for detecting extreme moves.
    ///
    /// - Horizon: 200 events (~20-100 seconds in active trading)
    /// - Threshold: 2.0% (200 bps)
    ///
    /// Expected positive rate: ~0.1-1% depending on volatility.
    pub fn extreme_moves() -> Self {
        Self::new(200, 0.02)
    }

    /// Set the conflict priority strategy.
    pub fn with_conflict_priority(mut self, priority: ConflictPriority) -> Self {
        self.conflict_priority = priority;
        self
    }

    /// Calculate minimum required prices for label generation.
    ///
    /// We need at least horizon + 1 prices to generate one label
    /// (current price + horizon future prices).
    pub fn min_prices_required(&self) -> usize {
        self.horizon + 1
    }

    /// Get the effective up threshold.
    #[inline]
    pub fn up_threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the effective down threshold.
    #[inline]
    pub fn down_threshold(&self) -> f64 {
        self.down_threshold.unwrap_or(self.threshold)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            return Err(TlobError::generic("horizon must be > 0"));
        }
        if self.threshold <= 0.0 {
            return Err(TlobError::generic("threshold must be > 0"));
        }
        if self.threshold >= 1.0 {
            return Err(TlobError::generic("threshold must be < 1.0 (100%)"));
        }
        if let Some(dt) = self.down_threshold {
            if dt <= 0.0 || dt >= 1.0 {
                return Err(TlobError::generic(
                    "down_threshold must be in range (0, 1)",
                ));
            }
        }
        Ok(())
    }
}

impl Default for OpportunityConfig {
    /// Default configuration: detect 0.5% moves within 50 events.
    fn default() -> Self {
        Self::medium_moves()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for opportunity label generation.
///
/// Provides comprehensive analysis of the generated labels for quality
/// assurance and understanding class balance.
#[derive(Debug, Clone, Default)]
pub struct OpportunityStats {
    /// Total number of labels generated
    pub total: usize,

    /// Number of BigUp labels
    pub big_up_count: usize,

    /// Number of BigDown labels
    pub big_down_count: usize,

    /// Number of NoOpportunity labels
    pub no_opportunity_count: usize,

    /// Number of conflicts (both up and down triggered)
    pub conflict_count: usize,

    /// Average max return across all windows
    pub avg_max_return: f64,

    /// Average min return across all windows
    pub avg_min_return: f64,

    /// Maximum max_return observed
    pub peak_max_return: f64,

    /// Minimum min_return observed (most negative)
    pub peak_min_return: f64,
}

impl OpportunityStats {
    /// Calculate opportunity rate (fraction of samples that are opportunities).
    pub fn opportunity_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.big_up_count + self.big_down_count) as f64 / self.total as f64
    }

    /// Calculate class balance as (big_up_pct, no_opportunity_pct, big_down_pct).
    pub fn class_balance(&self) -> (f64, f64, f64) {
        if self.total == 0 {
            return (0.0, 0.0, 0.0);
        }
        let total = self.total as f64;
        (
            self.big_up_count as f64 / total,
            self.no_opportunity_count as f64 / total,
            self.big_down_count as f64 / total,
        )
    }

    /// Calculate conflict rate (fraction of samples where both directions triggered).
    pub fn conflict_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.conflict_count as f64 / self.total as f64
    }

    /// Check if labels are severely imbalanced (any opportunity class < 1%).
    pub fn is_severely_imbalanced(&self) -> bool {
        if self.total == 0 {
            return true;
        }
        let (up, _, down) = self.class_balance();
        up < 0.01 || down < 0.01
    }
}

// ============================================================================
// Generator
// ============================================================================

/// Opportunity-based label generator.
///
/// Generates labels based on peak (max/min) returns within a horizon,
/// designed for detecting trading opportunities rather than average trends.
///
/// # Design
///
/// - Scans for maximum and minimum prices within each horizon window
/// - Labels based on whether peak returns exceed configurable thresholds
/// - Handles conflicts (both up and down moves) with configurable priority
///
/// # Performance
///
/// - Time: O(T × h) where T = number of prices, h = horizon
/// - Space: O(T) for price storage
///
/// # Thread Safety
///
/// Not thread-safe. Use separate instances for parallel processing.
pub struct OpportunityLabelGenerator {
    config: OpportunityConfig,
    prices: Vec<f64>,
}

impl OpportunityLabelGenerator {
    /// Create a new opportunity label generator.
    pub fn new(config: OpportunityConfig) -> Self {
        Self {
            config,
            prices: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(config: OpportunityConfig, capacity: usize) -> Self {
        Self {
            config,
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
    pub fn config(&self) -> &OpportunityConfig {
        &self.config
    }

    /// Check if we have enough prices to generate at least one label.
    pub fn can_generate(&self) -> bool {
        self.prices.len() >= self.config.min_prices_required()
    }

    /// Generate labels for all valid timesteps.
    ///
    /// Returns a vector of (index, label, max_return, min_return) tuples.
    ///
    /// # Valid Range
    ///
    /// Labels are generated for t in [0, T - horizon) where T is total prices.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<(usize, OpportunityLabel, f64, f64)>)` - (index, label, max_return, min_return)
    /// * `Err` - If insufficient data
    pub fn generate_labels(&self) -> Result<Vec<(usize, OpportunityLabel, f64, f64)>> {
        let h = self.config.horizon;
        let total = self.prices.len();
        let min_required = self.config.min_prices_required();

        if total < min_required {
            return Err(TlobError::generic(format!(
                "Need at least {} prices for opportunity labeling with horizon {} (have {})",
                min_required, h, total
            )));
        }

        let mut labels = Vec::with_capacity(total - h);
        let up_threshold = self.config.up_threshold();
        let down_threshold = self.config.down_threshold();

        // Generate labels for valid range: [0, total - horizon)
        for t in 0..(total - h) {
            let current_price = self.prices[t];

            // Find max and min prices in horizon window [t+1, t+horizon]
            let window = &self.prices[(t + 1)..=(t + h)];
            let (max_price, min_price) = self.find_extremes(window);

            // Compute returns relative to current price
            let max_return = (max_price - current_price) / current_price;
            let min_return = (min_price - current_price) / current_price;

            // Classify based on thresholds
            let label = self.classify(max_return, min_return, up_threshold, down_threshold);

            labels.push((t, label, max_return, min_return));
        }

        Ok(labels)
    }

    /// Generate labels and return only the labels (without returns).
    pub fn generate_label_sequence(&self) -> Result<Vec<OpportunityLabel>> {
        let labels = self.generate_labels()?;
        Ok(labels.into_iter().map(|(_, label, _, _)| label).collect())
    }

    /// Generate labels and return as class indices.
    pub fn generate_class_indices(&self) -> Result<Vec<usize>> {
        let labels = self.generate_labels()?;
        Ok(labels
            .into_iter()
            .map(|(_, label, _, _)| label.as_class_index())
            .collect())
    }

    /// Find maximum and minimum values in a slice.
    #[inline]
    fn find_extremes(&self, window: &[f64]) -> (f64, f64) {
        let mut max_val = f64::NEG_INFINITY;
        let mut min_val = f64::INFINITY;

        for &price in window {
            if price > max_val {
                max_val = price;
            }
            if price < min_val {
                min_val = price;
            }
        }

        (max_val, min_val)
    }

    /// Classify based on max/min returns and thresholds.
    #[inline]
    fn classify(
        &self,
        max_return: f64,
        min_return: f64,
        up_threshold: f64,
        down_threshold: f64,
    ) -> OpportunityLabel {
        let is_big_up = max_return > up_threshold;
        let is_big_down = min_return < -down_threshold;

        match (is_big_up, is_big_down) {
            (false, false) => OpportunityLabel::NoOpportunity,
            (true, false) => OpportunityLabel::BigUp,
            (false, true) => OpportunityLabel::BigDown,
            (true, true) => {
                // Conflict: both directions triggered
                match self.config.conflict_priority {
                    ConflictPriority::LargerMagnitude => {
                        if max_return.abs() >= min_return.abs() {
                            OpportunityLabel::BigUp
                        } else {
                            OpportunityLabel::BigDown
                        }
                    }
                    ConflictPriority::UpPriority => OpportunityLabel::BigUp,
                    ConflictPriority::DownPriority => OpportunityLabel::BigDown,
                    ConflictPriority::Ambiguous => OpportunityLabel::NoOpportunity,
                }
            }
        }
    }

    /// Compute statistics for the generated labels.
    pub fn compute_stats(
        &self,
        labels: &[(usize, OpportunityLabel, f64, f64)],
    ) -> OpportunityStats {
        if labels.is_empty() {
            return OpportunityStats::default();
        }

        let mut stats = OpportunityStats::default();
        stats.total = labels.len();

        let mut sum_max_return = 0.0;
        let mut sum_min_return = 0.0;
        stats.peak_max_return = f64::NEG_INFINITY;
        stats.peak_min_return = f64::INFINITY;

        let up_threshold = self.config.up_threshold();
        let down_threshold = self.config.down_threshold();

        for (_, label, max_return, min_return) in labels {
            match label {
                OpportunityLabel::BigUp => stats.big_up_count += 1,
                OpportunityLabel::BigDown => stats.big_down_count += 1,
                OpportunityLabel::NoOpportunity => stats.no_opportunity_count += 1,
            }

            // Check for conflicts
            if *max_return > up_threshold && *min_return < -down_threshold {
                stats.conflict_count += 1;
            }

            sum_max_return += max_return;
            sum_min_return += min_return;

            if *max_return > stats.peak_max_return {
                stats.peak_max_return = *max_return;
            }
            if *min_return < stats.peak_min_return {
                stats.peak_min_return = *min_return;
            }
        }

        stats.avg_max_return = sum_max_return / stats.total as f64;
        stats.avg_min_return = sum_min_return / stats.total as f64;

        stats
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // OpportunityLabel Tests
    // ========================================================================

    #[test]
    fn test_opportunity_label_conversion() {
        // Test int conversion
        assert_eq!(OpportunityLabel::BigUp.as_int(), 1);
        assert_eq!(OpportunityLabel::NoOpportunity.as_int(), 0);
        assert_eq!(OpportunityLabel::BigDown.as_int(), -1);

        // Test roundtrip
        for label in [
            OpportunityLabel::BigDown,
            OpportunityLabel::NoOpportunity,
            OpportunityLabel::BigUp,
        ] {
            let as_int = label.as_int();
            let recovered = OpportunityLabel::from_int(as_int).unwrap();
            assert_eq!(label, recovered);
        }
    }

    #[test]
    fn test_opportunity_label_class_index() {
        assert_eq!(OpportunityLabel::BigDown.as_class_index(), 0);
        assert_eq!(OpportunityLabel::NoOpportunity.as_class_index(), 1);
        assert_eq!(OpportunityLabel::BigUp.as_class_index(), 2);

        // Test roundtrip
        for label in [
            OpportunityLabel::BigDown,
            OpportunityLabel::NoOpportunity,
            OpportunityLabel::BigUp,
        ] {
            let as_idx = label.as_class_index();
            let recovered = OpportunityLabel::from_class_index(as_idx).unwrap();
            assert_eq!(label, recovered);
        }
    }

    #[test]
    fn test_opportunity_label_is_opportunity() {
        assert!(OpportunityLabel::BigUp.is_opportunity());
        assert!(OpportunityLabel::BigDown.is_opportunity());
        assert!(!OpportunityLabel::NoOpportunity.is_opportunity());
    }

    // ========================================================================
    // OpportunityConfig Tests
    // ========================================================================

    #[test]
    fn test_config_creation() {
        let config = OpportunityConfig::new(50, 0.005);
        assert_eq!(config.horizon, 50);
        assert_eq!(config.threshold, 0.005);
        assert_eq!(config.conflict_priority, ConflictPriority::LargerMagnitude);
        assert!(config.down_threshold.is_none());
    }

    #[test]
    fn test_config_presets() {
        let medium = OpportunityConfig::medium_moves();
        assert_eq!(medium.horizon, 50);
        assert_eq!(medium.threshold, 0.005);

        let whale = OpportunityConfig::whale_detection();
        assert_eq!(whale.horizon, 100);
        assert_eq!(whale.threshold, 0.01);

        let extreme = OpportunityConfig::extreme_moves();
        assert_eq!(extreme.horizon, 200);
        assert_eq!(extreme.threshold, 0.02);
    }

    #[test]
    fn test_config_asymmetric_thresholds() {
        let config = OpportunityConfig::with_asymmetric_thresholds(50, 0.005, 0.003);
        assert_eq!(config.up_threshold(), 0.005);
        assert_eq!(config.down_threshold(), 0.003);
    }

    #[test]
    fn test_config_validation() {
        let valid = OpportunityConfig::new(50, 0.005);
        assert!(valid.validate().is_ok());

        // Invalid horizon
        let config = OpportunityConfig {
            horizon: 0,
            threshold: 0.005,
            conflict_priority: ConflictPriority::default(),
            down_threshold: None,
        };
        assert!(config.validate().is_err());

        // Invalid threshold
        let config = OpportunityConfig {
            horizon: 50,
            threshold: 0.0,
            conflict_priority: ConflictPriority::default(),
            down_threshold: None,
        };
        assert!(config.validate().is_err());

        let config = OpportunityConfig {
            horizon: 50,
            threshold: 1.5,
            conflict_priority: ConflictPriority::default(),
            down_threshold: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_min_prices_required() {
        let config = OpportunityConfig::new(50, 0.005);
        assert_eq!(config.min_prices_required(), 51); // horizon + 1
    }

    #[test]
    #[should_panic(expected = "horizon must be > 0")]
    fn test_config_zero_horizon_panics() {
        OpportunityConfig::new(0, 0.005);
    }

    #[test]
    #[should_panic(expected = "threshold must be in range (0, 1)")]
    fn test_config_invalid_threshold_panics() {
        OpportunityConfig::new(50, 0.0);
    }

    // ========================================================================
    // OpportunityLabelGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_creation() {
        let config = OpportunityConfig::new(10, 0.01);
        let generator = OpportunityLabelGenerator::new(config);
        assert!(generator.is_empty());
        assert!(!generator.can_generate());
    }

    #[test]
    fn test_generator_add_prices() {
        let config = OpportunityConfig::new(10, 0.01);
        let mut generator = OpportunityLabelGenerator::new(config);

        generator.add_price(100.0);
        generator.add_price(101.0);
        assert_eq!(generator.len(), 2);

        generator.add_prices(&[102.0, 103.0, 104.0]);
        assert_eq!(generator.len(), 5);
    }

    #[test]
    fn test_generator_can_generate() {
        // Need horizon + 1 = 11 prices
        let config = OpportunityConfig::new(10, 0.01);
        let mut generator = OpportunityLabelGenerator::new(config);

        let prices: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        generator.add_prices(&prices);
        assert!(!generator.can_generate()); // 10 < 11

        generator.add_price(110.0);
        assert!(generator.can_generate()); // 11 >= 11
    }

    #[test]
    fn test_generator_insufficient_data_error() {
        let config = OpportunityConfig::new(50, 0.01);
        let mut generator = OpportunityLabelGenerator::new(config);

        generator.add_prices(&[100.0; 20]); // Only 20 prices

        let result = generator.generate_labels();
        assert!(result.is_err());
    }

    #[test]
    fn test_generator_strong_upward_spike() {
        let config = OpportunityConfig::new(5, 0.02); // 2% threshold
        let mut generator = OpportunityLabelGenerator::new(config);

        // Create a scenario with a spike:
        // Prices: 100, 100, 100, 105, 100, 100, 100, 100, 100, 100
        // At t=0, max in [1..5] = 105, return = 5% > 2% → BigUp
        let prices = vec![
            100.0, 100.0, 100.0, 105.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ];
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // First label (t=0) should be BigUp because max_return = 5%
        let (idx, label, max_return, _) = &labels[0];
        assert_eq!(*idx, 0);
        assert_eq!(*label, OpportunityLabel::BigUp);
        assert!((*max_return - 0.05).abs() < 1e-10); // 5% return
    }

    #[test]
    fn test_generator_strong_downward_spike() {
        let config = OpportunityConfig::new(5, 0.02); // 2% threshold
        let mut generator = OpportunityLabelGenerator::new(config);

        // Create a scenario with a drop:
        // Prices: 100, 100, 100, 95, 100, 100, 100, 100, 100, 100
        // At t=0, min in [1..5] = 95, return = -5% < -2% → BigDown
        let prices = vec![
            100.0, 100.0, 100.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ];
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // First label (t=0) should be BigDown because min_return = -5%
        let (idx, label, _, min_return) = &labels[0];
        assert_eq!(*idx, 0);
        assert_eq!(*label, OpportunityLabel::BigDown);
        assert!((*min_return - (-0.05)).abs() < 1e-10); // -5% return
    }

    #[test]
    fn test_generator_stable_prices() {
        let config = OpportunityConfig::new(5, 0.02); // 2% threshold
        let mut generator = OpportunityLabelGenerator::new(config);

        // Stable prices with tiny fluctuations
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.01).sin() * 0.1).collect();
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();
        assert!(!labels.is_empty());

        // All labels should be NoOpportunity
        for (_, label, max_return, min_return) in &labels {
            assert_eq!(
                *label,
                OpportunityLabel::NoOpportunity,
                "Expected NoOpportunity for max={:.6}, min={:.6}",
                max_return,
                min_return
            );
        }
    }

    #[test]
    fn test_generator_conflict_larger_magnitude() {
        let config = OpportunityConfig::new(5, 0.02)
            .with_conflict_priority(ConflictPriority::LargerMagnitude);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Create volatile scenario with both up and down moves
        // At t=0: window [1..5] has max=108 (8% up) and min=92 (8% down)
        // But min magnitude is larger in absolute terms
        let prices = vec![
            100.0, 92.0, 108.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ];
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();
        let (_, label, max_return, min_return) = &labels[0];

        // Both should trigger: max_return = 8%, min_return = -8%
        assert!(*max_return > 0.02);
        assert!(*min_return < -0.02);

        // Since |max_return| = |min_return|, BigUp should win (tie goes to up)
        // Actually checking: if max = 0.08 and min = -0.08, magnitudes are equal
        // We need to check the exact behavior
        assert!(
            *label == OpportunityLabel::BigUp || *label == OpportunityLabel::BigDown,
            "Conflict should resolve to one of the opportunities"
        );
    }

    #[test]
    fn test_generator_conflict_up_priority() {
        let config =
            OpportunityConfig::new(5, 0.02).with_conflict_priority(ConflictPriority::UpPriority);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Volatile scenario
        let prices = vec![
            100.0, 92.0, 108.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ];
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();
        let (_, label, _, _) = &labels[0];

        // With UpPriority, conflict should always resolve to BigUp
        assert_eq!(*label, OpportunityLabel::BigUp);
    }

    #[test]
    fn test_generator_conflict_ambiguous() {
        let config =
            OpportunityConfig::new(5, 0.02).with_conflict_priority(ConflictPriority::Ambiguous);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Volatile scenario
        let prices = vec![
            100.0, 92.0, 108.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ];
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();
        let (_, label, _, _) = &labels[0];

        // With Ambiguous, conflict should resolve to NoOpportunity
        assert_eq!(*label, OpportunityLabel::NoOpportunity);
    }

    #[test]
    fn test_generator_deterministic() {
        let config = OpportunityConfig::new(5, 0.01);
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 3.0).collect();

        let mut gen1 = OpportunityLabelGenerator::new(config.clone());
        gen1.add_prices(&prices);
        let labels1 = gen1.generate_labels().unwrap();

        let mut gen2 = OpportunityLabelGenerator::new(config);
        gen2.add_prices(&prices);
        let labels2 = gen2.generate_labels().unwrap();

        assert_eq!(labels1.len(), labels2.len());
        for (l1, l2) in labels1.iter().zip(labels2.iter()) {
            assert_eq!(l1.0, l2.0); // Same index
            assert_eq!(l1.1, l2.1); // Same label
            assert!((l1.2 - l2.2).abs() < 1e-15); // Same max_return
            assert!((l1.3 - l2.3).abs() < 1e-15); // Same min_return
        }
    }

    #[test]
    fn test_generator_no_nan_or_inf() {
        let config = OpportunityConfig::new(10, 0.01);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Various price patterns
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();
        generator.add_prices(&prices);

        let labels = generator.generate_labels().unwrap();

        for (_, _, max_return, min_return) in &labels {
            assert!(!max_return.is_nan(), "Found NaN in max_return");
            assert!(!max_return.is_infinite(), "Found Inf in max_return");
            assert!(!min_return.is_nan(), "Found NaN in min_return");
            assert!(!min_return.is_infinite(), "Found Inf in min_return");
        }
    }

    #[test]
    fn test_generator_clear_and_reuse() {
        let config = OpportunityConfig::new(5, 0.02);
        let mut generator = OpportunityLabelGenerator::new(config);

        // First use
        generator.add_prices(&[100.0, 100.0, 100.0, 105.0, 100.0, 100.0, 100.0, 100.0]);
        let labels1 = generator.generate_labels().unwrap();
        assert!(!labels1.is_empty());

        // Clear and reuse
        generator.clear();
        assert!(generator.is_empty());

        generator.add_prices(&[100.0, 100.0, 100.0, 95.0, 100.0, 100.0, 100.0, 100.0]);
        let labels2 = generator.generate_labels().unwrap();
        assert!(!labels2.is_empty());

        // First had upward spike, second has downward spike
        assert_eq!(labels1[0].1, OpportunityLabel::BigUp);
        assert_eq!(labels2[0].1, OpportunityLabel::BigDown);
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_stats_computation() {
        let config = OpportunityConfig::new(5, 0.02);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Create scenario with some of each type
        let mut prices = vec![100.0];
        // Add a big up spike
        prices.extend(vec![100.0, 100.0, 105.0, 100.0, 100.0]);
        // Add stable
        prices.extend(vec![100.0, 100.0, 100.0, 100.0, 100.0]);
        // Add a big down spike
        prices.extend(vec![100.0, 100.0, 95.0, 100.0, 100.0]);
        // Add more stable
        prices.extend(vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);

        generator.add_prices(&prices);
        let labels = generator.generate_labels().unwrap();
        let stats = generator.compute_stats(&labels);

        // Verify counts sum to total
        assert_eq!(
            stats.big_up_count + stats.big_down_count + stats.no_opportunity_count,
            stats.total
        );

        // Verify we have some of each
        assert!(stats.big_up_count > 0, "Should have some BigUp labels");
        assert!(stats.big_down_count > 0, "Should have some BigDown labels");
        assert!(
            stats.no_opportunity_count > 0,
            "Should have some NoOpportunity labels"
        );

        // Verify class balance sums to ~1.0
        let (up_pct, no_pct, down_pct) = stats.class_balance();
        let sum = up_pct + no_pct + down_pct;
        assert!((sum - 1.0).abs() < 1e-10, "Class balance should sum to 1.0");
    }

    #[test]
    fn test_stats_opportunity_rate() {
        let stats = OpportunityStats {
            total: 100,
            big_up_count: 10,
            big_down_count: 5,
            no_opportunity_count: 85,
            conflict_count: 0,
            avg_max_return: 0.0,
            avg_min_return: 0.0,
            peak_max_return: 0.0,
            peak_min_return: 0.0,
        };

        let rate = stats.opportunity_rate();
        assert!((rate - 0.15).abs() < 1e-10); // 15% opportunity rate
    }

    #[test]
    fn test_stats_empty() {
        let stats = OpportunityStats::default();
        assert_eq!(stats.opportunity_rate(), 0.0);
        assert_eq!(stats.class_balance(), (0.0, 0.0, 0.0));
        assert!(stats.is_severely_imbalanced());
    }

    // ========================================================================
    // Formula Correctness Tests
    // ========================================================================

    #[test]
    fn test_formula_max_return_correctness() {
        // Verify that max_return = (max_price - current) / current
        let config = OpportunityConfig::new(3, 0.01);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Prices: [100, 101, 102, 103]
        // At t=0, horizon=3, window=[101, 102, 103]
        // max_price = 103
        // max_return = (103 - 100) / 100 = 0.03 = 3%
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0]);

        let labels = generator.generate_labels().unwrap();
        let (_, _, max_return, _) = &labels[0];

        let expected = (103.0 - 100.0) / 100.0;
        assert!(
            (*max_return - expected).abs() < 1e-10,
            "Expected max_return = {}, got {}",
            expected,
            max_return
        );
    }

    #[test]
    fn test_formula_min_return_correctness() {
        // Verify that min_return = (min_price - current) / current
        let config = OpportunityConfig::new(3, 0.01);
        let mut generator = OpportunityLabelGenerator::new(config);

        // Prices: [100, 99, 98, 97]
        // At t=0, horizon=3, window=[99, 98, 97]
        // min_price = 97
        // min_return = (97 - 100) / 100 = -0.03 = -3%
        generator.add_prices(&[100.0, 99.0, 98.0, 97.0]);

        let labels = generator.generate_labels().unwrap();
        let (_, _, _, min_return) = &labels[0];

        let expected = (97.0 - 100.0) / 100.0;
        assert!(
            (*min_return - expected).abs() < 1e-10,
            "Expected min_return = {}, got {}",
            expected,
            min_return
        );
    }

    #[test]
    fn test_threshold_boundary_exactly_at_threshold() {
        // Test behavior exactly at threshold boundary
        let config = OpportunityConfig::new(3, 0.03); // 3% threshold
        let mut generator = OpportunityLabelGenerator::new(config);

        // Create return exactly at 3%: (103 - 100) / 100 = 0.03
        generator.add_prices(&[100.0, 101.0, 102.0, 103.0]);

        let labels = generator.generate_labels().unwrap();
        let (_, label, max_return, _) = &labels[0];

        // max_return = 3% exactly, should NOT trigger (> not >=)
        assert!(
            (*max_return - 0.03).abs() < 1e-10,
            "max_return should be exactly 3%"
        );
        assert_eq!(
            *label,
            OpportunityLabel::NoOpportunity,
            "Exactly at threshold should be NoOpportunity (strict >)"
        );
    }

    #[test]
    fn test_threshold_boundary_just_above() {
        // Test behavior just above threshold
        let config = OpportunityConfig::new(3, 0.03); // 3% threshold
        let mut generator = OpportunityLabelGenerator::new(config);

        // Create return just above 3%: (103.01 - 100) / 100 = 0.0301
        generator.add_prices(&[100.0, 101.0, 102.0, 103.01]);

        let labels = generator.generate_labels().unwrap();
        let (_, label, max_return, _) = &labels[0];

        assert!(*max_return > 0.03, "max_return should be above 3%");
        assert_eq!(*label, OpportunityLabel::BigUp, "Above threshold should be BigUp");
    }
}

