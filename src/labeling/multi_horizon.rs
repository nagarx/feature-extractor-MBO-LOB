//! Multi-Horizon Label Generation for LOB Price Trend Prediction
//!
//! Generates labels for multiple prediction horizons simultaneously,
//! as required by benchmark papers (FI-2010, DeepLOB, TLOB).
//!
//! # Research Reference
//!
//! All major benchmark papers evaluate at multiple horizons:
//!
//! | Paper | Horizons | Unit |
//! |-------|----------|------|
//! | FI-2010 | 10, 20, 30, 50, 100 | Events |
//! | DeepLOB | 10, 20, 50, 100 | Ticks |
//! | TLOB | 1, 3, 5, 10, 30, 50 | Seconds |
//!
//! # Design
//!
//! - Uses TLOB method (decoupled horizon/smoothing) for each horizon
//! - Single price buffer shared across all horizons (memory efficient)
//! - Single-pass label generation (O(T × H) where H = number of horizons)
//! - Configurable threshold strategies (Fixed, RollingSpread)
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{
//!     MultiHorizonConfig, MultiHorizonLabelGenerator, ThresholdStrategy,
//! };
//!
//! // Configure for FI-2010 horizons
//! let config = MultiHorizonConfig::fi2010();
//!
//! let mut generator = MultiHorizonLabelGenerator::new(config);
//!
//! // Add prices
//! let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64) * 0.01).collect();
//! generator.add_prices(&prices);
//!
//! // Generate labels for all horizons
//! let result = generator.generate_labels().unwrap();
//!
//! // Access per-horizon labels
//! for horizon in result.horizons() {
//!     let labels = result.labels_for_horizon(*horizon).unwrap();
//!     println!("Horizon {}: {} labels", horizon, labels.len());
//! }
//! ```

use super::{LabelConfig, LabelStats, TrendLabel};
use mbo_lob_reconstructor::{Result, TlobError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ============================================================================
// Threshold Strategy
// ============================================================================

/// Threshold strategy for label classification.
///
/// Different strategies for determining the Up/Down/Stable threshold.
///
/// # Research Reference
///
/// - **Fixed**: Standard fixed percentage (DeepLOB default: 0.002)
/// - **RollingSpread**: Adaptive based on spread (TLOB paper Section 4.1.3)
/// - **Quantile**: Balanced class distribution via rolling quantiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdStrategy {
    /// Fixed percentage threshold.
    ///
    /// Simple and reproducible. Use for benchmark comparison.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::ThresholdStrategy;
    ///
    /// let strategy = ThresholdStrategy::Fixed(0.002); // 0.2%
    /// assert_eq!(strategy.get_threshold(None), 0.002);
    /// ```
    Fixed(f64),

    /// Rolling spread-based threshold.
    ///
    /// Uses average spread as percentage of mid-price.
    /// More adaptive to market conditions.
    ///
    /// # Parameters
    ///
    /// - `window_size`: Number of samples for rolling average
    /// - `multiplier`: Multiplier for the average spread (typically 1.0)
    /// - `fallback`: Fallback threshold when insufficient data
    RollingSpread {
        window_size: usize,
        multiplier: f64,
        fallback: f64,
    },

    /// Quantile-based threshold for balanced class distribution.
    ///
    /// Computes the threshold from the quantile of absolute percentage changes
    /// in a rolling window. This ensures roughly equal proportions of
    /// Up/Down/Stable labels regardless of market volatility.
    ///
    /// # Parameters
    ///
    /// - `target_proportion`: Target proportion for up/down classes (e.g., 0.3 for 30%)
    /// - `window_size`: Number of samples for quantile computation
    /// - `fallback`: Fallback threshold when insufficient data
    ///
    /// # Research Reference
    ///
    /// Balanced class distribution is critical for training:
    /// - Imbalanced datasets lead to biased models
    /// - Quantile-based thresholds adapt to market conditions
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::ThresholdStrategy;
    ///
    /// // Target 30% up, 30% down, 40% stable
    /// let strategy = ThresholdStrategy::Quantile {
    ///     target_proportion: 0.3,
    ///     window_size: 5000,
    ///     fallback: 0.002,
    /// };
    /// ```
    Quantile {
        /// Target proportion for up/down classes (0.0 to 0.5)
        /// Example: 0.3 means ~30% Up, ~30% Down, ~40% Stable
        target_proportion: f64,
        /// Number of samples for quantile computation
        window_size: usize,
        /// Fallback threshold when insufficient data
        fallback: f64,
    },

    /// TLOB paper dynamic threshold (global computation).
    ///
    /// Computes threshold from the **entire** price series as:
    /// `alpha = mean(|percentage_change|) / 2`
    ///
    /// This is a two-pass approach:
    /// 1. First pass: Compute all smoothed percentage changes l(t,h,k)
    /// 2. Compute alpha from the full distribution
    /// 3. Second pass: Apply threshold for classification
    ///
    /// # Research Reference
    ///
    /// TLOB repository: `utils/utils_data.py::labeling()`
    /// ```python
    /// alpha = np.abs(percentage_change).mean() / 2
    /// labels = np.where(percentage_change < -alpha, 2,
    ///                   np.where(percentage_change > alpha, 0, 1))
    /// ```
    ///
    /// # Parameters
    ///
    /// - `fallback`: Fallback threshold when insufficient data
    /// - `divisor`: Divisor for mean (default: 2.0 per TLOB paper)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::ThresholdStrategy;
    ///
    /// // Match official TLOB repository labeling
    /// let strategy = ThresholdStrategy::TlobDynamic {
    ///     fallback: 0.0008,
    ///     divisor: 2.0,
    /// };
    /// ```
    TlobDynamic {
        /// Fallback threshold when insufficient data
        fallback: f64,
        /// Divisor for mean absolute change (default: 2.0 per TLOB paper)
        /// alpha = mean(|percentage_change|) / divisor
        divisor: f64,
    },
}

impl ThresholdStrategy {
    /// Get the threshold value.
    ///
    /// # Arguments
    ///
    /// * `rolling_value` - Rolling value for adaptive strategies:
    ///   - For `RollingSpread`: rolling spread as percentage
    ///   - For `Quantile`: computed quantile threshold
    ///   - For `TlobDynamic`: computed alpha from mean(|pct_change|) / divisor
    pub fn get_threshold(&self, rolling_value: Option<f64>) -> f64 {
        match self {
            ThresholdStrategy::Fixed(threshold) => *threshold,
            ThresholdStrategy::RollingSpread {
                multiplier,
                fallback,
                ..
            } => rolling_value.map(|s| s * multiplier).unwrap_or(*fallback),
            ThresholdStrategy::Quantile { fallback, .. } => rolling_value.unwrap_or(*fallback),
            ThresholdStrategy::TlobDynamic { fallback, .. } => rolling_value.unwrap_or(*fallback),
        }
    }

    /// Create a fixed threshold strategy.
    pub fn fixed(threshold: f64) -> Self {
        ThresholdStrategy::Fixed(threshold)
    }

    /// Create a rolling spread threshold strategy.
    pub fn rolling_spread(window_size: usize, multiplier: f64, fallback: f64) -> Self {
        ThresholdStrategy::RollingSpread {
            window_size,
            multiplier,
            fallback,
        }
    }

    /// Create a quantile-based threshold strategy for balanced classes.
    ///
    /// # Arguments
    ///
    /// * `target_proportion` - Target proportion for up/down classes (0.0 to 0.5)
    /// * `window_size` - Number of samples for quantile computation
    /// * `fallback` - Fallback threshold when insufficient data
    ///
    /// # Panics
    ///
    /// Panics if `target_proportion` is not in range (0.0, 0.5].
    pub fn quantile(target_proportion: f64, window_size: usize, fallback: f64) -> Self {
        assert!(
            target_proportion > 0.0 && target_proportion <= 0.5,
            "target_proportion must be in range (0.0, 0.5], got {}",
            target_proportion
        );
        ThresholdStrategy::Quantile {
            target_proportion,
            window_size,
            fallback,
        }
    }

    /// Create a TLOB paper dynamic threshold strategy.
    ///
    /// Computes threshold from the entire price series as:
    /// `alpha = mean(|percentage_change|) / divisor`
    ///
    /// # Arguments
    ///
    /// * `fallback` - Fallback threshold when insufficient data
    /// * `divisor` - Divisor for mean (default: 2.0 per TLOB paper)
    ///
    /// # Research Reference
    ///
    /// TLOB repository: `utils/utils_data.py::labeling()`
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is <= 0.
    pub fn tlob_dynamic(fallback: f64, divisor: f64) -> Self {
        assert!(divisor > 0.0, "divisor must be > 0.0, got {}", divisor);
        ThresholdStrategy::TlobDynamic { fallback, divisor }
    }

    /// Create a TLOB paper dynamic threshold with default divisor (2.0).
    ///
    /// # Arguments
    ///
    /// * `fallback` - Fallback threshold when insufficient data
    pub fn tlob_dynamic_default(fallback: f64) -> Self {
        Self::tlob_dynamic(fallback, 2.0)
    }

    /// Check if this strategy requires rolling computation.
    pub fn needs_rolling_computation(&self) -> bool {
        matches!(
            self,
            ThresholdStrategy::RollingSpread { .. } | ThresholdStrategy::Quantile { .. }
        )
    }

    /// Check if this strategy requires global (full dataset) computation.
    ///
    /// Returns `true` for strategies that need all data before threshold can be computed.
    pub fn needs_global_computation(&self) -> bool {
        matches!(self, ThresholdStrategy::TlobDynamic { .. })
    }

    /// Get the window size for rolling strategies, if applicable.
    pub fn window_size(&self) -> Option<usize> {
        match self {
            ThresholdStrategy::Fixed(_) => None,
            ThresholdStrategy::RollingSpread { window_size, .. } => Some(*window_size),
            ThresholdStrategy::Quantile { window_size, .. } => Some(*window_size),
            ThresholdStrategy::TlobDynamic { .. } => None, // Global, not windowed
        }
    }

    /// Get the target proportion for Quantile strategy, if applicable.
    pub fn target_proportion(&self) -> Option<f64> {
        match self {
            ThresholdStrategy::Quantile {
                target_proportion, ..
            } => Some(*target_proportion),
            _ => None,
        }
    }

    /// Get the divisor for TlobDynamic strategy, if applicable.
    pub fn tlob_divisor(&self) -> Option<f64> {
        match self {
            ThresholdStrategy::TlobDynamic { divisor, .. } => Some(*divisor),
            _ => None,
        }
    }

    /// Get the fallback threshold for adaptive strategies.
    pub fn fallback(&self) -> Option<f64> {
        match self {
            ThresholdStrategy::Fixed(_) => None,
            ThresholdStrategy::RollingSpread { fallback, .. } => Some(*fallback),
            ThresholdStrategy::Quantile { fallback, .. } => Some(*fallback),
            ThresholdStrategy::TlobDynamic { fallback, .. } => Some(*fallback),
        }
    }
}

impl Default for ThresholdStrategy {
    fn default() -> Self {
        ThresholdStrategy::Fixed(0.002) // 0.2% - standard in TLOB/DeepLOB
    }
}

// ============================================================================
// Multi-Horizon Configuration
// ============================================================================

/// Configuration for multi-horizon label generation.
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::{MultiHorizonConfig, ThresholdStrategy};
///
/// // Custom configuration
/// let config = MultiHorizonConfig::new(
///     vec![10, 20, 50, 100],
///     5,
///     ThresholdStrategy::Fixed(0.002),
/// );
///
/// // Or use presets
/// let fi2010 = MultiHorizonConfig::fi2010();
/// let deeplob = MultiHorizonConfig::deeplob();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHorizonConfig {
    /// Prediction horizons (sorted ascending).
    ///
    /// Each horizon h specifies how many steps ahead to predict.
    horizons: Vec<usize>,

    /// Smoothing window size (k).
    ///
    /// Shared across all horizons. Controls noise reduction.
    /// Recommended: 5-10 for most applications.
    pub smoothing_window: usize,

    /// Threshold strategy for classification.
    pub threshold_strategy: ThresholdStrategy,
}

impl MultiHorizonConfig {
    /// Create a new multi-horizon configuration.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Prediction horizons (will be sorted)
    /// * `smoothing_window` - Smoothing window size
    /// * `threshold_strategy` - Strategy for threshold determination
    ///
    /// # Panics
    ///
    /// Panics if horizons is empty or smoothing_window is 0.
    pub fn new(
        mut horizons: Vec<usize>,
        smoothing_window: usize,
        threshold_strategy: ThresholdStrategy,
    ) -> Self {
        assert!(!horizons.is_empty(), "Horizons cannot be empty");
        assert!(smoothing_window > 0, "Smoothing window must be > 0");

        // Sort and deduplicate
        horizons.sort_unstable();
        horizons.dedup();

        // Remove zero horizons
        horizons.retain(|&h| h > 0);

        Self {
            horizons,
            smoothing_window,
            threshold_strategy,
        }
    }

    /// FI-2010 benchmark configuration.
    ///
    /// Horizons: [10, 20, 30, 50, 100] events
    /// Smoothing: 5 (standard)
    /// Threshold: 0.002 (0.2%)
    pub fn fi2010() -> Self {
        Self::new(
            vec![10, 20, 30, 50, 100],
            5,
            ThresholdStrategy::Fixed(0.002),
        )
    }

    /// DeepLOB paper configuration.
    ///
    /// Horizons: [10, 20, 50, 100] ticks
    /// Smoothing: 5 (standard)
    /// Threshold: 0.002 (0.2%)
    pub fn deeplob() -> Self {
        Self::new(vec![10, 20, 50, 100], 5, ThresholdStrategy::Fixed(0.002))
    }

    /// TLOB paper configuration.
    ///
    /// Horizons: [1, 3, 5, 10, 30, 50] seconds (converted to ticks by user)
    /// Smoothing: 5 (standard)
    /// Threshold: 0.002 (0.2%)
    pub fn tlob() -> Self {
        Self::new(
            vec![1, 3, 5, 10, 30, 50],
            5,
            ThresholdStrategy::Fixed(0.002),
        )
    }

    /// HFT (High-Frequency Trading) configuration.
    ///
    /// Short horizons with tight threshold.
    /// Horizons: [5, 10, 20] ticks
    /// Threshold: 0.0002 (2 bps)
    pub fn hft() -> Self {
        Self::new(vec![5, 10, 20], 3, ThresholdStrategy::Fixed(0.0002))
    }

    /// Get the horizons.
    pub fn horizons(&self) -> &[usize] {
        &self.horizons
    }

    /// Get the maximum horizon.
    pub fn max_horizon(&self) -> usize {
        *self.horizons.last().unwrap_or(&0)
    }

    /// Get the minimum horizon.
    pub fn min_horizon(&self) -> usize {
        *self.horizons.first().unwrap_or(&0)
    }

    /// Calculate minimum required prices.
    ///
    /// Formula: k + max_horizon + k + 1 = 2k + max_h + 1
    pub fn min_prices_required(&self) -> usize {
        let k = self.smoothing_window;
        let max_h = self.max_horizon();
        k + max_h + k + 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.horizons.is_empty() {
            return Err(TlobError::generic("Horizons cannot be empty"));
        }
        if self.smoothing_window == 0 {
            return Err(TlobError::generic("Smoothing window must be > 0"));
        }
        if let ThresholdStrategy::Fixed(t) = self.threshold_strategy {
            if t <= 0.0 || t >= 1.0 {
                return Err(TlobError::generic(
                    "Fixed threshold must be in range (0, 1)",
                ));
            }
        }
        Ok(())
    }

    /// Convert to single-horizon LabelConfig for a specific horizon.
    ///
    /// Useful for compatibility with existing single-horizon code.
    pub fn to_label_config(&self, horizon: usize) -> LabelConfig {
        let threshold = self.threshold_strategy.get_threshold(None);
        LabelConfig {
            horizon,
            smoothing_window: self.smoothing_window,
            threshold,
        }
    }
}

impl Default for MultiHorizonConfig {
    fn default() -> Self {
        Self::fi2010()
    }
}

// ============================================================================
// Multi-Horizon Labels Result
// ============================================================================

/// Result container for multi-horizon label generation.
///
/// Contains labels for each horizon, indexed by horizon value.
///
/// # Example
///
/// ```ignore
/// let result = generator.generate_labels()?;
///
/// // Get all horizons
/// for horizon in result.horizons() {
///     let labels = result.labels_for_horizon(*horizon).unwrap();
///     let stats = result.stats_for_horizon(*horizon).unwrap();
///     println!("Horizon {}: {} labels, {:.1}% up",
///         horizon, labels.len(), stats.class_balance().0 * 100.0);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MultiHorizonLabels {
    /// Labels indexed by horizon.
    /// Each entry: (index, TrendLabel, percentage_change)
    labels: BTreeMap<usize, Vec<(usize, TrendLabel, f64)>>,

    /// Statistics per horizon.
    stats: BTreeMap<usize, LabelStats>,

    /// Configuration used for generation.
    config: MultiHorizonConfig,

    /// Total prices processed.
    total_prices: usize,
}

impl MultiHorizonLabels {
    /// Create a new multi-horizon labels container.
    fn new(config: MultiHorizonConfig, total_prices: usize) -> Self {
        Self {
            labels: BTreeMap::new(),
            stats: BTreeMap::new(),
            config,
            total_prices,
        }
    }

    /// Get the horizons in sorted order.
    pub fn horizons(&self) -> Vec<&usize> {
        self.labels.keys().collect()
    }

    /// Get labels for a specific horizon.
    pub fn labels_for_horizon(&self, horizon: usize) -> Option<&Vec<(usize, TrendLabel, f64)>> {
        self.labels.get(&horizon)
    }

    /// Get only the TrendLabel sequence for a horizon.
    pub fn label_sequence(&self, horizon: usize) -> Option<Vec<TrendLabel>> {
        self.labels
            .get(&horizon)
            .map(|v| v.iter().map(|(_, l, _)| *l).collect())
    }

    /// Get class indices (0, 1, 2) for a horizon.
    pub fn class_indices(&self, horizon: usize) -> Option<Vec<usize>> {
        self.labels
            .get(&horizon)
            .map(|v| v.iter().map(|(_, l, _)| l.as_class_index()).collect())
    }

    /// Get statistics for a specific horizon.
    pub fn stats_for_horizon(&self, horizon: usize) -> Option<&LabelStats> {
        self.stats.get(&horizon)
    }

    /// Get the configuration used.
    pub fn config(&self) -> &MultiHorizonConfig {
        &self.config
    }

    /// Get total prices that were processed.
    pub fn total_prices(&self) -> usize {
        self.total_prices
    }

    /// Get count of labels for each horizon.
    pub fn label_counts(&self) -> BTreeMap<usize, usize> {
        self.labels.iter().map(|(h, v)| (*h, v.len())).collect()
    }

    /// Check if a horizon exists.
    pub fn has_horizon(&self, horizon: usize) -> bool {
        self.labels.contains_key(&horizon)
    }

    /// Get the number of horizons.
    pub fn num_horizons(&self) -> usize {
        self.labels.len()
    }

    /// Check if any labels were generated.
    pub fn is_empty(&self) -> bool {
        self.labels.values().all(|v| v.is_empty())
    }

    /// Get summary statistics across all horizons.
    pub fn summary(&self) -> MultiHorizonSummary {
        let mut total_labels = 0;
        let mut total_up = 0;
        let mut total_down = 0;
        let mut total_stable = 0;

        for stats in self.stats.values() {
            total_labels += stats.total;
            total_up += stats.up_count;
            total_down += stats.down_count;
            total_stable += stats.stable_count;
        }

        MultiHorizonSummary {
            num_horizons: self.labels.len(),
            total_labels,
            total_up,
            total_down,
            total_stable,
            total_prices: self.total_prices,
        }
    }
}

/// Summary statistics across all horizons.
#[derive(Debug, Clone)]
pub struct MultiHorizonSummary {
    /// Number of horizons.
    pub num_horizons: usize,
    /// Total labels across all horizons.
    pub total_labels: usize,
    /// Total Up labels.
    pub total_up: usize,
    /// Total Down labels.
    pub total_down: usize,
    /// Total Stable labels.
    pub total_stable: usize,
    /// Total prices processed.
    pub total_prices: usize,
}

// ============================================================================
// Multi-Horizon Label Generator
// ============================================================================

/// Multi-horizon label generator.
///
/// Generates labels for multiple prediction horizons in a single pass.
/// Uses the TLOB method (decoupled horizon/smoothing) for each horizon.
///
/// # Performance
///
/// - Time: O(T × H × k) where T = prices, H = horizons, k = smoothing
/// - Space: O(T + H × L) where L = labels per horizon
///
/// # Thread Safety
///
/// Not thread-safe. Use separate instances for parallel processing.
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::{MultiHorizonConfig, MultiHorizonLabelGenerator};
///
/// let config = MultiHorizonConfig::fi2010();
/// let mut generator = MultiHorizonLabelGenerator::new(config);
///
/// // Add prices (need at least 2k + max_h + 1 prices)
/// let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64) * 0.01).collect();
/// generator.add_prices(&prices);
///
/// let result = generator.generate_labels().unwrap();
/// assert!(!result.is_empty());
/// ```
pub struct MultiHorizonLabelGenerator {
    config: MultiHorizonConfig,
    prices: Vec<f64>,

    /// Rolling spread tracker (for RollingSpread threshold)
    spread_history: Vec<f64>,

    /// Rolling percentage change tracker (for Quantile threshold)
    /// Note: Currently unused - reserved for future Quantile strategy implementation
    #[allow(dead_code)]
    pct_change_history: Vec<f64>,
}

impl MultiHorizonLabelGenerator {
    /// Create a new multi-horizon label generator.
    ///
    /// # Arguments
    ///
    /// * `config` - Multi-horizon configuration
    pub fn new(config: MultiHorizonConfig) -> Self {
        Self {
            config,
            prices: Vec::new(),
            spread_history: Vec::new(),
            pct_change_history: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `config` - Multi-horizon configuration
    /// * `capacity` - Expected number of prices
    pub fn with_capacity(config: MultiHorizonConfig, capacity: usize) -> Self {
        Self {
            config,
            prices: Vec::with_capacity(capacity),
            spread_history: Vec::new(),
            pct_change_history: Vec::new(),
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

    /// Add a spread observation (for RollingSpread threshold).
    ///
    /// Only needed when using `ThresholdStrategy::RollingSpread`.
    pub fn add_spread(&mut self, spread_pct: f64) {
        self.spread_history.push(spread_pct);
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

    /// Clear all buffers.
    pub fn clear(&mut self) {
        self.prices.clear();
        self.spread_history.clear();
    }

    /// Get the configuration.
    pub fn config(&self) -> &MultiHorizonConfig {
        &self.config
    }

    /// Check if we have enough prices to generate labels.
    pub fn can_generate(&self) -> bool {
        self.prices.len() >= self.config.min_prices_required()
    }

    /// Get the current threshold value.
    fn current_threshold(&self) -> f64 {
        let rolling_value = match &self.config.threshold_strategy {
            ThresholdStrategy::Fixed(_) => None,
            ThresholdStrategy::RollingSpread { .. } => self.compute_rolling_spread(),
            ThresholdStrategy::Quantile { .. } => self.compute_quantile_threshold(),
            // TlobDynamic uses horizon-specific computation, not a single global value
            // The fallback is used here; actual threshold is computed per-horizon in generate_labels()
            ThresholdStrategy::TlobDynamic { .. } => None,
        };
        self.config.threshold_strategy.get_threshold(rolling_value)
    }

    /// Compute rolling spread average (for RollingSpread threshold).
    fn compute_rolling_spread(&self) -> Option<f64> {
        if let ThresholdStrategy::RollingSpread { window_size, .. } =
            &self.config.threshold_strategy
        {
            if self.spread_history.len() >= *window_size {
                let start = self.spread_history.len() - window_size;
                let sum: f64 = self.spread_history[start..].iter().sum();
                return Some(sum / *window_size as f64);
            }
        }
        None
    }

    /// Compute quantile threshold for a specific horizon.
    ///
    /// This computes the threshold from the quantile of absolute smoothed percentage
    /// changes (the actual `l(t,h,k)` values used for labeling), NOT from 1-step changes.
    ///
    /// For example, with target_proportion = 0.33:
    /// - We want ~33% Up, ~33% Down, ~34% Stable
    /// - Threshold = (1 - target_proportion)th quantile of |l(t,h,k)|
    /// - This ensures ~(1 - 2*target_proportion) samples are within ±threshold (Stable)
    ///
    /// # Arguments
    ///
    /// * `horizon` - The prediction horizon to compute threshold for
    ///
    /// # Returns
    ///
    /// The quantile threshold for this horizon, or None if insufficient data.
    ///
    /// # Research Reference
    ///
    /// Using horizon-specific thresholds computed from the actual l(t,h,k) distribution
    /// ensures balanced class distribution at each horizon. This addresses the issue
    /// where 1-step changes don't represent the scale of multi-step smoothed changes.
    fn compute_quantile_threshold_for_horizon(&self, horizon: usize) -> Option<f64> {
        if let ThresholdStrategy::Quantile {
            target_proportion,
            window_size,
            ..
        } = &self.config.threshold_strategy
        {
            let k = self.config.smoothing_window;
            let total = self.prices.len();

            // Valid range: [k, total - horizon - k)
            let start = k;
            let end = total.saturating_sub(horizon + k);

            if start >= end {
                return None;
            }

            // Compute all smoothed percentage changes l(t,h,k) for this horizon
            // This is the ACTUAL distribution we're classifying, not 1-step changes
            let smoothed_changes: Vec<f64> = (start..end)
                .map(|t| {
                    let past_smooth = self.smoothed_past(t, k);
                    let future_smooth = self.smoothed_future(t, horizon, k);
                    (future_smooth - past_smooth) / past_smooth
                })
                .collect();

            if smoothed_changes.len() < *window_size {
                // Fall back to using all available changes if less than window_size
                if smoothed_changes.is_empty() {
                    return None;
                }
            }

            // Take the most recent window (or all if fewer than window_size)
            let window_start = smoothed_changes.len().saturating_sub(*window_size);
            let mut abs_changes: Vec<f64> = smoothed_changes[window_start..]
                .iter()
                .map(|x| x.abs())
                .collect();

            // Sort for quantile computation
            abs_changes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Compute quantile position
            // We want (1 - 2*target_proportion) of samples to be Stable
            // So the threshold is at the (1 - target_proportion) quantile
            // Example: target_proportion=0.33 → quantile=0.67 → ~34% Stable
            let quantile_pos = (1.0 - *target_proportion) * (abs_changes.len() - 1) as f64;
            let lower_idx = quantile_pos.floor() as usize;
            let upper_idx = (lower_idx + 1).min(abs_changes.len() - 1);
            let frac = quantile_pos - lower_idx as f64;

            // Linear interpolation between adjacent values
            let threshold = abs_changes[lower_idx] * (1.0 - frac) + abs_changes[upper_idx] * frac;

            return Some(threshold);
        }
        None
    }

    /// Compute quantile threshold (legacy method for backwards compatibility).
    ///
    /// **DEPRECATED**: Use `compute_quantile_threshold_for_horizon` instead.
    /// This method computes threshold from 1-step changes which doesn't match
    /// the scale of multi-step smoothed changes used for labeling.
    ///
    /// Kept for RollingSpread strategy and backwards compatibility.
    fn compute_quantile_threshold(&self) -> Option<f64> {
        // For backwards compatibility, compute for the median horizon
        let horizons = &self.config.horizons;
        if horizons.is_empty() {
            return None;
        }
        let median_horizon = horizons[horizons.len() / 2];
        self.compute_quantile_threshold_for_horizon(median_horizon)
    }

    /// Compute TLOB dynamic threshold for a specific horizon.
    ///
    /// This implements the TLOB paper's dynamic threshold formula:
    /// `alpha = mean(|l(t,h,k)|) / divisor`
    ///
    /// Where l(t,h,k) is the smoothed percentage change at time t for horizon h
    /// with smoothing window k.
    ///
    /// # Arguments
    ///
    /// * `horizon` - The prediction horizon to compute threshold for
    ///
    /// # Returns
    ///
    /// The TLOB dynamic threshold for this horizon, or None if insufficient data.
    ///
    /// # Research Reference
    ///
    /// TLOB repository: `utils/utils_data.py::labeling()`
    /// ```python
    /// alpha = np.abs(percentage_change).mean() / 2
    /// ```
    fn compute_tlob_dynamic_threshold_for_horizon(&self, horizon: usize) -> Option<f64> {
        if let ThresholdStrategy::TlobDynamic { divisor, .. } = &self.config.threshold_strategy {
            let k = self.config.smoothing_window;
            let total = self.prices.len();

            // Valid range: [k, total - horizon - k)
            let start = k;
            let end = total.saturating_sub(horizon + k);

            if start >= end {
                return None;
            }

            // Compute all smoothed percentage changes l(t,h,k) for this horizon
            let smoothed_changes: Vec<f64> = (start..end)
                .map(|t| {
                    let past_smooth = self.smoothed_past(t, k);
                    let future_smooth = self.smoothed_future(t, horizon, k);
                    (future_smooth - past_smooth) / past_smooth
                })
                .collect();

            if smoothed_changes.is_empty() {
                return None;
            }

            // Compute alpha = mean(|percentage_change|) / divisor
            let mean_abs_change: f64 = smoothed_changes.iter().map(|x| x.abs()).sum::<f64>()
                / smoothed_changes.len() as f64;
            let alpha = mean_abs_change / divisor;

            return Some(alpha);
        }
        None
    }

    /// Generate labels for all horizons.
    ///
    /// # Returns
    ///
    /// * `Ok(MultiHorizonLabels)` - Labels for all horizons
    /// * `Err` - If insufficient data
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::labeling::{MultiHorizonConfig, MultiHorizonLabelGenerator};
    ///
    /// let config = MultiHorizonConfig::new(vec![10, 20], 5, Default::default());
    /// let mut gen = MultiHorizonLabelGenerator::new(config);
    ///
    /// // Add enough prices (need 2*5 + 20 + 1 = 31 prices)
    /// let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.1).collect();
    /// gen.add_prices(&prices);
    ///
    /// let result = gen.generate_labels().unwrap();
    /// assert!(result.has_horizon(10));
    /// assert!(result.has_horizon(20));
    /// ```
    pub fn generate_labels(&self) -> Result<MultiHorizonLabels> {
        let min_required = self.config.min_prices_required();
        let total = self.prices.len();

        if total < min_required {
            return Err(TlobError::generic(format!(
                "Need at least {} prices for multi-horizon labeling (have {}). \
                 Required: 2*k + max_h + 1 = 2*{} + {} + 1 = {}",
                min_required,
                total,
                self.config.smoothing_window,
                self.config.max_horizon(),
                min_required
            )));
        }

        let k = self.config.smoothing_window;
        let mut result = MultiHorizonLabels::new(self.config.clone(), total);

        // Determine which threshold computation method to use
        let use_quantile_threshold = matches!(
            self.config.threshold_strategy,
            ThresholdStrategy::Quantile { .. }
        );
        let use_tlob_dynamic_threshold = matches!(
            self.config.threshold_strategy,
            ThresholdStrategy::TlobDynamic { .. }
        );

        // Generate labels for each horizon
        for &horizon in &self.config.horizons {
            // Compute horizon-specific threshold based on strategy:
            // - Quantile: Quantile-based threshold from l(t,h,k) distribution
            // - TlobDynamic: mean(|l(t,h,k)|) / divisor (TLOB paper formula)
            // - Fixed/RollingSpread: Same threshold for all horizons
            let threshold = if use_quantile_threshold {
                self.compute_quantile_threshold_for_horizon(horizon)
                    .unwrap_or_else(|| self.current_threshold())
            } else if use_tlob_dynamic_threshold {
                self.compute_tlob_dynamic_threshold_for_horizon(horizon)
                    .unwrap_or_else(|| self.current_threshold())
            } else {
                self.current_threshold()
            };

            let labels = self.generate_for_horizon(horizon, k, threshold)?;
            let stats = Self::compute_stats(&labels);
            result.labels.insert(horizon, labels);
            result.stats.insert(horizon, stats);
        }

        Ok(result)
    }

    /// Generate labels for a single horizon.
    fn generate_for_horizon(
        &self,
        horizon: usize,
        k: usize,
        threshold: f64,
    ) -> Result<Vec<(usize, TrendLabel, f64)>> {
        let total = self.prices.len();

        // Valid range: [k, total - horizon - k)
        // We need k past prices and k+horizon future prices
        let start = k;
        let end = total.saturating_sub(horizon + k);

        if start >= end {
            return Err(TlobError::generic(format!(
                "Not enough prices for horizon {}. Need {} prices, have {}",
                horizon,
                k + horizon + k + 1,
                total
            )));
        }

        let mut labels = Vec::with_capacity(end - start);

        for t in start..end {
            // Past smoothed: average of prices from t-k to t (k+1 terms)
            let past_smooth = self.smoothed_past(t, k);

            // Future smoothed: average of prices from t+h-k to t+h (k+1 terms)
            let future_smooth = self.smoothed_future(t, horizon, k);

            // Percentage change
            let pct_change = (future_smooth - past_smooth) / past_smooth;

            // Classify
            let label = Self::classify(pct_change, threshold);

            labels.push((t, label, pct_change));
        }

        Ok(labels)
    }

    /// Compute past smoothed mid-price: w-(t,h,k).
    #[inline]
    fn smoothed_past(&self, t: usize, k: usize) -> f64 {
        let start = t.saturating_sub(k);
        let end = t + 1;
        let sum: f64 = self.prices[start..end].iter().sum();
        sum / (k + 1) as f64
    }

    /// Compute future smoothed mid-price: w+(t,h,k).
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
    fn classify(pct_change: f64, threshold: f64) -> TrendLabel {
        if pct_change > threshold {
            TrendLabel::Up
        } else if pct_change < -threshold {
            TrendLabel::Down
        } else {
            TrendLabel::Stable
        }
    }

    /// Compute label statistics.
    fn compute_stats(labels: &[(usize, TrendLabel, f64)]) -> LabelStats {
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

    // ========================================================================
    // ThresholdStrategy Tests
    // ========================================================================

    #[test]
    fn test_threshold_strategy_fixed() {
        let strategy = ThresholdStrategy::Fixed(0.002);
        assert_eq!(strategy.get_threshold(None), 0.002);
        assert_eq!(strategy.get_threshold(Some(0.005)), 0.002); // Ignores input
    }

    #[test]
    fn test_threshold_strategy_rolling_spread() {
        let strategy = ThresholdStrategy::RollingSpread {
            window_size: 100,
            multiplier: 1.0,
            fallback: 0.002,
        };

        // Without rolling spread data, use fallback
        assert_eq!(strategy.get_threshold(None), 0.002);

        // With rolling spread data
        assert_eq!(strategy.get_threshold(Some(0.003)), 0.003);

        // With multiplier
        let strategy2 = ThresholdStrategy::RollingSpread {
            window_size: 100,
            multiplier: 2.0,
            fallback: 0.002,
        };
        assert_eq!(strategy2.get_threshold(Some(0.003)), 0.006);
    }

    #[test]
    fn test_threshold_strategy_default() {
        let strategy = ThresholdStrategy::default();
        assert!(matches!(strategy, ThresholdStrategy::Fixed(t) if (t - 0.002).abs() < 1e-10));
    }

    // ========================================================================
    // MultiHorizonConfig Tests
    // ========================================================================

    #[test]
    fn test_config_new() {
        let config = MultiHorizonConfig::new(vec![10, 20, 50], 5, ThresholdStrategy::Fixed(0.002));

        assert_eq!(config.horizons(), &[10, 20, 50]);
        assert_eq!(config.smoothing_window, 5);
        assert_eq!(config.max_horizon(), 50);
        assert_eq!(config.min_horizon(), 10);
    }

    #[test]
    fn test_config_sorts_and_deduplicates() {
        let config =
            MultiHorizonConfig::new(vec![50, 10, 20, 10, 50], 5, ThresholdStrategy::Fixed(0.002));

        assert_eq!(config.horizons(), &[10, 20, 50]);
    }

    #[test]
    fn test_config_removes_zero_horizons() {
        let config =
            MultiHorizonConfig::new(vec![0, 10, 0, 20], 5, ThresholdStrategy::Fixed(0.002));

        assert_eq!(config.horizons(), &[10, 20]);
    }

    #[test]
    fn test_config_fi2010() {
        let config = MultiHorizonConfig::fi2010();
        assert_eq!(config.horizons(), &[10, 20, 30, 50, 100]);
        assert_eq!(config.smoothing_window, 5);
    }

    #[test]
    fn test_config_deeplob() {
        let config = MultiHorizonConfig::deeplob();
        assert_eq!(config.horizons(), &[10, 20, 50, 100]);
    }

    #[test]
    fn test_config_tlob() {
        let config = MultiHorizonConfig::tlob();
        assert_eq!(config.horizons(), &[1, 3, 5, 10, 30, 50]);
    }

    #[test]
    fn test_config_hft() {
        let config = MultiHorizonConfig::hft();
        assert_eq!(config.horizons(), &[5, 10, 20]);
        assert_eq!(config.smoothing_window, 3);
    }

    #[test]
    fn test_config_min_prices_required() {
        // min_prices = 2*k + max_h + 1
        let config = MultiHorizonConfig::new(vec![10, 100], 5, ThresholdStrategy::Fixed(0.002));
        // 2*5 + 100 + 1 = 111
        assert_eq!(config.min_prices_required(), 111);
    }

    #[test]
    fn test_config_validate() {
        let valid = MultiHorizonConfig::fi2010();
        assert!(valid.validate().is_ok());

        // Invalid threshold
        let invalid = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(0.0));
        assert!(invalid.validate().is_err());

        let invalid2 = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(1.5));
        assert!(invalid2.validate().is_err());
    }

    #[test]
    fn test_config_to_label_config() {
        let config = MultiHorizonConfig::new(vec![10, 20], 5, ThresholdStrategy::Fixed(0.003));

        let label_config = config.to_label_config(20);
        assert_eq!(label_config.horizon, 20);
        assert_eq!(label_config.smoothing_window, 5);
        assert_eq!(label_config.threshold, 0.003);
    }

    #[test]
    #[should_panic(expected = "Horizons cannot be empty")]
    fn test_config_empty_horizons_panics() {
        MultiHorizonConfig::new(vec![], 5, ThresholdStrategy::Fixed(0.002));
    }

    #[test]
    #[should_panic(expected = "Smoothing window must be > 0")]
    fn test_config_zero_smoothing_panics() {
        MultiHorizonConfig::new(vec![10], 0, ThresholdStrategy::Fixed(0.002));
    }

    // ========================================================================
    // MultiHorizonLabelGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_creation() {
        let config = MultiHorizonConfig::fi2010();
        let gen = MultiHorizonLabelGenerator::new(config);
        assert!(gen.is_empty());
        assert!(!gen.can_generate());
    }

    #[test]
    fn test_generator_add_prices() {
        let config = MultiHorizonConfig::fi2010();
        let mut gen = MultiHorizonLabelGenerator::new(config);

        gen.add_price(100.0);
        gen.add_price(101.0);
        assert_eq!(gen.len(), 2);

        gen.add_prices(&[102.0, 103.0, 104.0]);
        assert_eq!(gen.len(), 5);
    }

    #[test]
    fn test_generator_clear() {
        let config = MultiHorizonConfig::fi2010();
        let mut gen = MultiHorizonLabelGenerator::new(config);

        gen.add_prices(&[100.0, 101.0, 102.0]);
        assert_eq!(gen.len(), 3);

        gen.clear();
        assert!(gen.is_empty());
    }

    #[test]
    fn test_generator_can_generate() {
        // Need 2*5 + 100 + 1 = 111 prices for FI-2010
        let config = MultiHorizonConfig::fi2010();
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..110).map(|i| 100.0 + i as f64).collect();
        gen.add_prices(&prices);
        assert!(!gen.can_generate()); // 110 < 111

        gen.add_price(200.0);
        assert!(gen.can_generate()); // 111 >= 111
    }

    #[test]
    fn test_generator_insufficient_data_error() {
        let config = MultiHorizonConfig::new(vec![10, 20], 5, ThresholdStrategy::Fixed(0.002));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Need 2*5 + 20 + 1 = 31 prices
        gen.add_prices(&[100.0; 20]); // Only 20 prices

        let result = gen.generate_labels();
        assert!(result.is_err());
    }

    #[test]
    fn test_generator_simple_upward_trend() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Strong upward trend
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();

        assert!(result.has_horizon(5));
        assert!(result.has_horizon(10));

        // Should detect upward trend
        for horizon in [5, 10] {
            let labels = result.labels_for_horizon(horizon).unwrap();
            assert!(!labels.is_empty());

            let up_count = labels
                .iter()
                .filter(|(_, l, _)| *l == TrendLabel::Up)
                .count();
            assert!(
                up_count > labels.len() / 2,
                "Horizon {} should have majority Up labels",
                horizon
            );
        }
    }

    #[test]
    fn test_generator_simple_downward_trend() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Downward trend
        let prices: Vec<f64> = (0..50).map(|i| 150.0 - i as f64 * 0.5).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();

        // Should detect downward trend
        for horizon in [5, 10] {
            let labels = result.labels_for_horizon(horizon).unwrap();
            let down_count = labels
                .iter()
                .filter(|(_, l, _)| *l == TrendLabel::Down)
                .count();
            assert!(
                down_count > labels.len() / 2,
                "Horizon {} should have majority Down labels",
                horizon
            );
        }
    }

    #[test]
    fn test_generator_stable_prices() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Stable prices (tiny fluctuations)
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.001).sin() * 0.001)
            .collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();

        // Should detect stable
        for horizon in [5, 10] {
            let labels = result.labels_for_horizon(horizon).unwrap();
            let stable_count = labels
                .iter()
                .filter(|(_, l, _)| *l == TrendLabel::Stable)
                .count();
            assert!(
                stable_count == labels.len(),
                "Horizon {} should have all Stable labels for flat prices",
                horizon
            );
        }
    }

    #[test]
    fn test_generator_deterministic() {
        let config = MultiHorizonConfig::fi2010();
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();

        let mut gen1 = MultiHorizonLabelGenerator::new(config.clone());
        gen1.add_prices(&prices);
        let result1 = gen1.generate_labels().unwrap();

        let mut gen2 = MultiHorizonLabelGenerator::new(config);
        gen2.add_prices(&prices);
        let result2 = gen2.generate_labels().unwrap();

        // Should produce identical results
        for horizon in result1.horizons() {
            let labels1 = result1.labels_for_horizon(*horizon).unwrap();
            let labels2 = result2.labels_for_horizon(*horizon).unwrap();

            assert_eq!(labels1.len(), labels2.len());
            for (l1, l2) in labels1.iter().zip(labels2.iter()) {
                assert_eq!(l1.0, l2.0); // Same index
                assert_eq!(l1.1, l2.1); // Same label
                assert!((l1.2 - l2.2).abs() < 1e-15); // Same change
            }
        }
    }

    #[test]
    fn test_generator_no_nan_or_inf() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Test with various price patterns
        let patterns = [
            (0..50).map(|i| 100.0 + i as f64).collect::<Vec<_>>(),
            (0..50).map(|i| 100.0 - i as f64 * 0.1).collect::<Vec<_>>(),
            vec![100.0; 50], // Flat
        ];

        for prices in patterns {
            gen.clear();
            gen.add_prices(&prices);

            let result = gen.generate_labels().unwrap();
            for horizon in result.horizons() {
                let labels = result.labels_for_horizon(*horizon).unwrap();
                for (_, _, change) in labels {
                    assert!(!change.is_nan(), "Found NaN");
                    assert!(!change.is_infinite(), "Found Inf");
                }
            }
        }
    }

    #[test]
    fn test_generator_label_indices_are_valid() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();

        for horizon in result.horizons() {
            let labels = result.labels_for_horizon(*horizon).unwrap();
            for (idx, _, _) in labels {
                assert!(*idx < gen.len(), "Index {} should be < {}", idx, gen.len());
            }
        }
    }

    // ========================================================================
    // MultiHorizonLabels Tests
    // ========================================================================

    #[test]
    fn test_labels_accessors() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.3).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();

        // Test horizons()
        let horizons = result.horizons();
        assert_eq!(horizons.len(), 2);

        // Test label_sequence()
        let seq = result.label_sequence(5).unwrap();
        assert!(!seq.is_empty());
        assert!(seq
            .iter()
            .all(|l| matches!(l, TrendLabel::Up | TrendLabel::Down | TrendLabel::Stable)));

        // Test class_indices()
        let indices = result.class_indices(5).unwrap();
        assert_eq!(indices.len(), seq.len());
        assert!(indices.iter().all(|&i| i <= 2));

        // Test stats_for_horizon()
        let stats = result.stats_for_horizon(5).unwrap();
        assert_eq!(stats.total, seq.len());
        assert_eq!(
            stats.up_count + stats.down_count + stats.stable_count,
            stats.total
        );
    }

    #[test]
    fn test_labels_summary() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.3).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        let summary = result.summary();

        assert_eq!(summary.num_horizons, 2);
        assert!(summary.total_labels > 0);
        assert_eq!(
            summary.total_up + summary.total_down + summary.total_stable,
            summary.total_labels
        );
        assert_eq!(summary.total_prices, 50);
    }

    #[test]
    fn test_labels_label_counts() {
        let config = MultiHorizonConfig::new(vec![5, 10], 2, ThresholdStrategy::Fixed(0.01));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        let counts = result.label_counts();

        assert!(counts.contains_key(&5));
        assert!(counts.contains_key(&10));
        assert!(counts[&5] > 0);
        assert!(counts[&10] > 0);

        // Shorter horizon should have more labels (more valid positions)
        assert!(counts[&5] >= counts[&10]);
    }

    // ========================================================================
    // Consistency Tests (vs TlobLabelGenerator)
    // ========================================================================

    #[test]
    fn test_consistency_with_tlob_generator() {
        use super::super::TlobLabelGenerator;

        let horizon = 10;
        let smoothing = 5;
        let threshold = 0.002;

        // Multi-horizon config with single horizon
        let multi_config = MultiHorizonConfig::new(
            vec![horizon],
            smoothing,
            ThresholdStrategy::Fixed(threshold),
        );

        // Single TLOB config
        let tlob_config = LabelConfig {
            horizon,
            smoothing_window: smoothing,
            threshold,
        };

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();

        // Generate with multi-horizon
        let mut multi_gen = MultiHorizonLabelGenerator::new(multi_config);
        multi_gen.add_prices(&prices);
        let multi_result = multi_gen.generate_labels().unwrap();
        let multi_labels = multi_result.labels_for_horizon(horizon).unwrap();

        // Generate with TLOB
        let mut tlob_gen = TlobLabelGenerator::new(tlob_config);
        tlob_gen.add_prices(&prices);
        let tlob_labels = tlob_gen.generate_labels().unwrap();

        // Should produce identical results
        assert_eq!(
            multi_labels.len(),
            tlob_labels.len(),
            "Multi-horizon and TLOB should produce same number of labels"
        );

        for (multi, tlob) in multi_labels.iter().zip(tlob_labels.iter()) {
            assert_eq!(multi.0, tlob.0, "Indices should match");
            assert_eq!(multi.1, tlob.1, "Labels should match at index {}", multi.0);
            assert!(
                (multi.2 - tlob.2).abs() < 1e-14,
                "Percentage changes should match at index {}",
                multi.0
            );
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_edge_case_minimum_prices() {
        // Test with exact minimum required prices
        let config = MultiHorizonConfig::new(vec![5], 2, ThresholdStrategy::Fixed(0.01));
        let min_required = config.min_prices_required(); // 2*2 + 5 + 1 = 10

        let mut gen = MultiHorizonLabelGenerator::new(config);
        let prices: Vec<f64> = (0..min_required).map(|i| 100.0 + i as f64).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels();
        assert!(result.is_ok(), "Should succeed with minimum prices");

        let labels = result.unwrap().labels_for_horizon(5).unwrap().clone();
        assert!(
            !labels.is_empty(),
            "Should generate at least one label with minimum prices"
        );
    }

    #[test]
    fn test_edge_case_single_horizon() {
        let config = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(0.002));
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        assert_eq!(result.num_horizons(), 1);
        assert!(result.has_horizon(10));
    }

    #[test]
    fn test_edge_case_many_horizons() {
        let config = MultiHorizonConfig::new(
            vec![1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
            3,
            ThresholdStrategy::Fixed(0.01),
        );
        let mut gen = MultiHorizonLabelGenerator::new(config);

        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        assert_eq!(result.num_horizons(), 10);

        // Verify all horizons have labels
        for h in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30] {
            assert!(result.has_horizon(h), "Should have horizon {}", h);
            let labels = result.labels_for_horizon(h).unwrap();
            assert!(!labels.is_empty(), "Horizon {} should have labels", h);
        }
    }

    #[test]
    fn test_edge_case_very_small_threshold() {
        let config = MultiHorizonConfig::new(vec![5], 2, ThresholdStrategy::Fixed(0.00001)); // 0.001%
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Even tiny price movements should be classified as Up/Down
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.001).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        let labels = result.labels_for_horizon(5).unwrap();

        // With tiny threshold and increasing prices, should see Up labels
        let up_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Up)
            .count();
        assert!(up_count > 0, "Should have Up labels with tiny threshold");
    }

    #[test]
    fn test_edge_case_very_large_threshold() {
        let config = MultiHorizonConfig::new(vec![5], 2, ThresholdStrategy::Fixed(0.5)); // 50%
        let mut gen = MultiHorizonLabelGenerator::new(config);

        // Normal price movements should all be Stable
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        let labels = result.labels_for_horizon(5).unwrap();

        // With huge threshold, should see mostly Stable
        let stable_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Stable)
            .count();
        assert!(
            stable_count > labels.len() / 2,
            "Should have mostly Stable labels with large threshold"
        );
    }

    // ========================================================================
    // TlobDynamic Threshold Strategy Tests
    // ========================================================================

    #[test]
    fn test_threshold_strategy_tlob_dynamic_creation() {
        let strategy = ThresholdStrategy::tlob_dynamic(0.0008, 2.0);
        assert!(matches!(
            strategy,
            ThresholdStrategy::TlobDynamic { fallback: f, divisor: d }
            if (f - 0.0008).abs() < 1e-10 && (d - 2.0).abs() < 1e-10
        ));

        // Test default divisor helper
        let strategy_default = ThresholdStrategy::tlob_dynamic_default(0.001);
        assert_eq!(strategy_default.tlob_divisor(), Some(2.0));
        assert_eq!(strategy_default.fallback(), Some(0.001));
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_needs_global() {
        let fixed = ThresholdStrategy::Fixed(0.002);
        let rolling = ThresholdStrategy::rolling_spread(100, 1.0, 0.002);
        let quantile = ThresholdStrategy::quantile(0.33, 5000, 0.002);
        let tlob_dynamic = ThresholdStrategy::tlob_dynamic_default(0.002);

        assert!(!fixed.needs_global_computation());
        assert!(!rolling.needs_global_computation());
        assert!(!quantile.needs_global_computation());
        assert!(
            tlob_dynamic.needs_global_computation(),
            "TlobDynamic should need global computation"
        );
    }

    #[test]
    fn test_tlob_dynamic_threshold_computation() {
        // Create a predictable price series
        // Linear upward trend: 100, 101, 102, ... 149 (50 prices)
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let config =
            MultiHorizonConfig::new(vec![5], 2, ThresholdStrategy::tlob_dynamic_default(0.0008));
        let mut gen = MultiHorizonLabelGenerator::new(config);
        gen.add_prices(&prices);

        // Compute threshold
        let threshold = gen.compute_tlob_dynamic_threshold_for_horizon(5);
        assert!(
            threshold.is_some(),
            "Should compute threshold with sufficient data"
        );

        let alpha = threshold.unwrap();
        assert!(alpha > 0.0, "Alpha should be positive for trending data");
        assert!(alpha.is_finite(), "Alpha should be finite");
        assert!(!alpha.is_nan(), "Alpha should not be NaN");

        // For a linear upward trend, all percentage changes should be similar
        // and alpha should be roughly mean(|change|) / 2
    }

    #[test]
    fn test_tlob_dynamic_produces_labels() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();

        let config = MultiHorizonConfig::new(
            vec![10, 20],
            5,
            ThresholdStrategy::tlob_dynamic_default(0.0008),
        );
        let mut gen = MultiHorizonLabelGenerator::new(config);
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();

        // Should generate labels for both horizons
        assert!(result.has_horizon(10));
        assert!(result.has_horizon(20));

        // Labels should not be empty
        let labels_10 = result.labels_for_horizon(10).unwrap();
        let labels_20 = result.labels_for_horizon(20).unwrap();
        assert!(!labels_10.is_empty(), "Horizon 10 should have labels");
        assert!(!labels_20.is_empty(), "Horizon 20 should have labels");

        // All labels should be valid
        for (idx, label, change) in labels_10.iter() {
            assert!(*idx < prices.len(), "Index should be within bounds");
            assert!(change.is_finite(), "Change should be finite");
            assert!(matches!(
                label,
                TrendLabel::Up | TrendLabel::Down | TrendLabel::Stable
            ));
        }
    }

    #[test]
    fn test_tlob_dynamic_deterministic() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();

        let config =
            MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::tlob_dynamic_default(0.0008));

        // Run twice with same data
        let mut gen1 = MultiHorizonLabelGenerator::new(config.clone());
        gen1.add_prices(&prices);
        let result1 = gen1.generate_labels().unwrap();

        let mut gen2 = MultiHorizonLabelGenerator::new(config);
        gen2.add_prices(&prices);
        let result2 = gen2.generate_labels().unwrap();

        // Results should be identical
        let labels1 = result1.labels_for_horizon(10).unwrap();
        let labels2 = result2.labels_for_horizon(10).unwrap();

        assert_eq!(
            labels1.len(),
            labels2.len(),
            "Should produce same number of labels"
        );
        for (l1, l2) in labels1.iter().zip(labels2.iter()) {
            assert_eq!(l1.0, l2.0, "Indices should match");
            assert_eq!(l1.1, l2.1, "Labels should match");
            assert!((l1.2 - l2.2).abs() < 1e-15, "Changes should match");
        }
    }

    #[test]
    fn test_tlob_dynamic_formula_verification() {
        // Verify the formula: alpha = mean(|percentage_change|) / divisor
        // Using a simple price series where we can manually compute expected alpha

        // Flat prices with one big move
        let mut prices = vec![100.0; 30];
        prices[15] = 102.0; // 2% spike

        let config = MultiHorizonConfig::new(
            vec![3],
            1, // Small smoothing for easier verification
            ThresholdStrategy::tlob_dynamic(0.0008, 2.0),
        );
        let mut gen = MultiHorizonLabelGenerator::new(config);
        gen.add_prices(&prices);

        let result = gen.generate_labels().unwrap();
        let labels = result.labels_for_horizon(3).unwrap();

        // Should have produced some non-stable labels around the spike
        let non_stable = labels
            .iter()
            .filter(|(_, l, _)| *l != TrendLabel::Stable)
            .count();
        // The spike should cause at least some Up labels (around t=12-14) and Down labels (around t=16-18)
        assert!(
            non_stable > 0,
            "Should detect the price spike, found {} non-stable labels",
            non_stable
        );
    }

    #[test]
    fn test_tlob_dynamic_different_divisors() {
        let prices: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.5).collect();

        // Same data, different divisors
        let config_div2 =
            MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::tlob_dynamic(0.0008, 2.0));
        let config_div4 = MultiHorizonConfig::new(
            vec![10],
            5,
            ThresholdStrategy::tlob_dynamic(0.0008, 4.0), // Larger divisor = smaller threshold
        );

        let mut gen_div2 = MultiHorizonLabelGenerator::new(config_div2);
        gen_div2.add_prices(&prices);
        let result_div2 = gen_div2.generate_labels().unwrap();

        let mut gen_div4 = MultiHorizonLabelGenerator::new(config_div4);
        gen_div4.add_prices(&prices);
        let result_div4 = gen_div4.generate_labels().unwrap();

        // Larger divisor = smaller threshold = more non-stable labels
        let stats_div2 = result_div2.stats_for_horizon(10).unwrap();
        let stats_div4 = result_div4.stats_for_horizon(10).unwrap();

        // With smaller threshold (larger divisor), we should have fewer stable labels
        assert!(
            stats_div4.stable_count <= stats_div2.stable_count,
            "Larger divisor should result in fewer stable labels. div2: {}, div4: {}",
            stats_div2.stable_count,
            stats_div4.stable_count
        );
    }

    #[test]
    fn test_tlob_dynamic_fallback_used() {
        // Test with insufficient data - should use fallback
        let _prices: Vec<f64> = vec![100.0, 101.0, 102.0]; // Too few prices (unused, just for documentation)

        let config = MultiHorizonConfig::new(
            vec![100], // Horizon too large for 3 prices
            5,
            ThresholdStrategy::tlob_dynamic(0.05, 2.0), // Large fallback
        );
        let gen = MultiHorizonLabelGenerator::new(config);

        let threshold = gen.compute_tlob_dynamic_threshold_for_horizon(100);
        assert!(
            threshold.is_none(),
            "Should return None for insufficient data"
        );

        // The current_threshold() should return the fallback
        let fallback_threshold = gen.current_threshold();
        assert!(
            (fallback_threshold - 0.05).abs() < 1e-10,
            "Should use fallback threshold"
        );
    }

    #[test]
    #[should_panic(expected = "divisor must be > 0.0")]
    fn test_tlob_dynamic_invalid_divisor() {
        ThresholdStrategy::tlob_dynamic(0.0008, 0.0);
    }

    #[test]
    fn test_tlob_dynamic_no_nan_or_inf() {
        // Test with various price patterns to ensure no NaN/Inf
        let patterns: Vec<Vec<f64>> = vec![
            (0..50).map(|i| 100.0 + i as f64).collect(), // Upward
            (0..50).map(|i| 100.0 - i as f64 * 0.1).collect(), // Downward
            vec![100.0; 50],                             // Flat
            (0..50)
                .map(|i| 100.0 + (i as f64 * 0.2).sin() * 2.0)
                .collect(), // Oscillating
        ];

        for prices in patterns {
            let config = MultiHorizonConfig::new(
                vec![5, 10],
                2,
                ThresholdStrategy::tlob_dynamic_default(0.0008),
            );
            let mut gen = MultiHorizonLabelGenerator::new(config);
            gen.add_prices(&prices);

            let result = gen.generate_labels().unwrap();

            for horizon in result.horizons() {
                let labels = result.labels_for_horizon(*horizon).unwrap();
                for (_, _, change) in labels {
                    assert!(!change.is_nan(), "Found NaN in percentage change");
                    assert!(!change.is_infinite(), "Found Inf in percentage change");
                }
            }
        }
    }
}
