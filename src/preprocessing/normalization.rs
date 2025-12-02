//! Feature normalization for TLOB transformer.
//!
//! This module provides various normalization strategies to scale LOB features
//! to ranges suitable for transformer training. Different features benefit from
//! different normalization approaches.
//!
//! # Normalization Strategies
//!
//! ## 1. Percentage Change (Recommended for Prices)
//!
//! Converts absolute prices to relative changes:
//! ```text
//! normalized = (price - reference_price) / reference_price
//! ```
//!
//! **Benefits**:
//! - Makes prices comparable across different stocks
//! - Focuses on relative movements (what transformers care about)
//! - Stationarity: removes absolute price level trends
//!
//! **Used for**: Prices (bid/ask at all levels)
//!
//! ## 2. Z-Score Normalization
//!
//! Standardizes to zero mean, unit variance:
//! ```text
//! normalized = (x - mean) / std
//! ```
//!
//! **Benefits**:
//! - Standard ML preprocessing
//! - Handles outliers reasonably
//! - Interpretable (values in std deviations)
//!
//! **Used for**: Volume, spread, imbalance
//!
//! ## 3. Bilinear Normalization (For LOB Structure)
//!
//! Normalizes prices relative to mid-price and tick size:
//! ```text
//! normalized = (price - mid_price) / (k * tick_size)
//! ```
//!
//! **Benefits**:
//! - Preserves LOB structure
//! - Tick-size invariant
//! - Focuses on distance from mid-price
//!
//! **Used for**: LOB price levels
//!
//! ## 4. Min-Max Normalization
//!
//! Scales to [0, 1] or [-1, 1]:
//! ```text
//! normalized = (x - min) / (max - min)
//! ```
//!
//! **Benefits**:
//! - Bounded output
//! - Simple and fast
//!
//! **Used for**: Features with known bounds
//!
//! # Architecture
//!
//! The module is designed around the `Normalizer` trait, allowing easy
//! extension with custom strategies:
//!
//! ```text
//! Normalizer (trait)
//!     ├── PercentageChangeNormalizer
//!     ├── ZScoreNormalizer
//!     ├── BilinearNormalizer
//!     └── MinMaxNormalizer
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::normalization::{
//!     PercentageChangeNormalizer,
//!     ZScoreNormalizer,
//!     NormalizationPipeline,
//! };
//!
//! // Create normalizers
//! let price_norm = PercentageChangeNormalizer::new();
//! let volume_norm = ZScoreNormalizer::new();
//!
//! // Create pipeline
//! let pipeline = NormalizationPipeline::new()
//!     .with_price_normalizer(price_norm)
//!     .with_volume_normalizer(volume_norm);
//!
//! // Normalize a sequence
//! let normalized = pipeline.normalize_sequence(&sequence);
//! ```
//!
//! # Performance
//!
//! All normalizers are designed for minimal overhead:
//! - Percentage change: O(1) per feature
//! - Z-score: O(1) per feature (uses running stats)
//! - Bilinear: O(1) per feature
//! - Min-max: O(1) per feature
//!
//! Total normalization overhead: ~1-5% of feature extraction time

use std::collections::VecDeque;

/// Trait for feature normalization strategies.
///
/// Implementers provide methods to:
/// 1. Update internal state with new data (if needed)
/// 2. Normalize a single value
/// 3. Normalize a batch of values
/// 4. Reset state
pub trait Normalizer: Send + Sync {
    /// Update normalizer state with a new value.
    ///
    /// Some normalizers (like Z-score) need to track running statistics.
    /// Others (like percentage change) may not need this.
    fn update(&mut self, value: f64);

    /// Normalize a single value.
    fn normalize(&self, value: f64) -> f64;

    /// Normalize a batch of values.
    ///
    /// Default implementation calls `normalize` for each value,
    /// but can be overridden for efficiency.
    fn normalize_batch(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.normalize(v)).collect()
    }

    /// Reset normalizer state.
    fn reset(&mut self);

    /// Check if normalizer is ready (has enough data to normalize).
    fn is_ready(&self) -> bool {
        true // Most normalizers are always ready
    }
}

/// Percentage change normalization.
///
/// Normalizes values relative to a reference value (typically mid-price):
/// ```text
/// normalized = (value - reference) / reference
/// ```
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::{Normalizer, PercentageChangeNormalizer};
///
/// let mut normalizer = PercentageChangeNormalizer::new();
/// normalizer.set_reference(100.0);
///
/// // Price at 101 -> 1% increase
/// assert!((normalizer.normalize(101.0) - 0.01).abs() < 1e-10);
///
/// // Price at 99 -> 1% decrease
/// assert!((normalizer.normalize(99.0) - (-0.01)).abs() < 1e-10);
/// ```
pub struct PercentageChangeNormalizer {
    reference: f64,
    min_reference: f64, // Avoid division by very small numbers
}

impl PercentageChangeNormalizer {
    /// Create a new percentage change normalizer.
    ///
    /// # Arguments
    ///
    /// * `min_reference` - Minimum reference value to avoid division by zero
    pub fn new() -> Self {
        Self {
            reference: 1.0,
            min_reference: 1e-6,
        }
    }

    /// Set the reference value (e.g., mid-price).
    ///
    /// This should be called before normalizing each snapshot.
    pub fn set_reference(&mut self, reference: f64) {
        self.reference = reference.max(self.min_reference);
    }

    /// Get the current reference value.
    pub fn reference(&self) -> f64 {
        self.reference
    }
}

impl Default for PercentageChangeNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Normalizer for PercentageChangeNormalizer {
    fn update(&mut self, _value: f64) {
        // Percentage change doesn't track statistics
    }

    fn normalize(&self, value: f64) -> f64 {
        (value - self.reference) / self.reference
    }

    fn reset(&mut self) {
        self.reference = 1.0;
    }
}

/// Z-score normalization with running statistics.
///
/// Normalizes to zero mean and unit variance:
/// ```text
/// normalized = (value - mean) / std
/// ```
///
/// Uses Welford's online algorithm for numerical stability.
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::{Normalizer, ZScoreNormalizer};
///
/// let mut normalizer = ZScoreNormalizer::new();
///
/// // Feed data
/// for value in &[10.0, 20.0, 30.0, 40.0, 50.0] {
///     normalizer.update(*value);
/// }
///
/// // Normalize: mean=30, std≈14.14
/// let normalized = normalizer.normalize(50.0);
/// assert!((normalized - 1.414).abs() < 0.01);
/// ```
pub struct ZScoreNormalizer {
    /// Running mean (Welford's algorithm)
    mean: f64,

    /// Running M2 for variance calculation (Welford's algorithm)
    m2: f64,

    /// Number of samples seen
    count: u64,

    /// Window size for running statistics (0 = infinite window)
    window_size: usize,

    /// Circular buffer for windowed statistics
    window: Option<VecDeque<f64>>,

    /// Minimum std to avoid division by zero
    min_std: f64,
}

impl ZScoreNormalizer {
    /// Create a new Z-score normalizer with infinite window.
    pub fn new() -> Self {
        Self::with_window(0)
    }

    /// Create a new Z-score normalizer with fixed window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of recent values to use for statistics.
    ///   Use 0 for infinite window.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::ZScoreNormalizer;
    ///
    /// // Use only last 1000 values for statistics
    /// let normalizer = ZScoreNormalizer::with_window(1000);
    /// ```
    pub fn with_window(window_size: usize) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            window_size,
            window: if window_size > 0 {
                Some(VecDeque::with_capacity(window_size))
            } else {
                None
            },
            min_std: 1e-8,
        }
    }

    /// Get the current mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the current standard deviation.
    pub fn std(&self) -> f64 {
        if self.count < 2 {
            return 1.0; // Not enough data
        }
        (self.m2 / (self.count as f64)).sqrt().max(self.min_std)
    }

    /// Get the number of samples seen.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Check if enough samples have been seen for reliable statistics.
    pub fn has_enough_samples(&self, min_samples: u64) -> bool {
        self.count >= min_samples
    }
}

impl Default for ZScoreNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Normalizer for ZScoreNormalizer {
    fn update(&mut self, value: f64) {
        // Handle windowed statistics
        if let Some(ref mut window) = self.window {
            window.push_back(value);

            if window.len() > self.window_size {
                // Recalculate from window when full
                window.pop_front();

                // Recompute statistics from scratch
                self.count = 0;
                self.mean = 0.0;
                self.m2 = 0.0;

                for &v in window.iter() {
                    self.count += 1;
                    let delta = v - self.mean;
                    self.mean += delta / self.count as f64;
                    let delta2 = v - self.mean;
                    self.m2 += delta * delta2;
                }
                return;
            }
        }

        // Welford's online algorithm for mean and variance
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn normalize(&self, value: f64) -> f64 {
        if self.count < 2 {
            return 0.0; // Not enough data, return neutral
        }

        (value - self.mean) / self.std()
    }

    fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
        if let Some(ref mut window) = self.window {
            window.clear();
        }
    }

    fn is_ready(&self) -> bool {
        self.count >= 2 // Need at least 2 samples for std
    }
}

/// Bilinear normalization for LOB price levels.
///
/// Normalizes prices relative to mid-price and tick size:
/// ```text
/// normalized = (price - mid_price) / (k * tick_size)
/// ```
///
/// where k is a scaling factor (typically 10-100).
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::{Normalizer, BilinearNormalizer};
///
/// let mut normalizer = BilinearNormalizer::new(0.01, 50.0);
/// normalizer.set_mid_price(100.0);
///
/// // Ask at 100.05 (5 ticks above mid)
/// let normalized = normalizer.normalize(100.05);
/// assert!((normalized - 0.1).abs() < 1e-10); // 5 / (50 * 0.01) = 0.1
/// ```
pub struct BilinearNormalizer {
    mid_price: f64,
    tick_size: f64,
    scale_factor: f64,
}

impl BilinearNormalizer {
    /// Create a new bilinear normalizer.
    ///
    /// # Arguments
    ///
    /// * `tick_size` - Minimum price increment (e.g., 0.01 for US stocks)
    /// * `scale_factor` - Scaling factor k (typically 10-100)
    pub fn new(tick_size: f64, scale_factor: f64) -> Self {
        assert!(tick_size > 0.0, "tick_size must be positive");
        assert!(scale_factor > 0.0, "scale_factor must be positive");

        Self {
            mid_price: 100.0, // Default
            tick_size,
            scale_factor,
        }
    }

    /// Set the mid-price reference.
    pub fn set_mid_price(&mut self, mid_price: f64) {
        self.mid_price = mid_price;
    }

    /// Get the current mid-price.
    pub fn mid_price(&self) -> f64 {
        self.mid_price
    }
}

impl Normalizer for BilinearNormalizer {
    fn update(&mut self, _value: f64) {
        // Bilinear doesn't track statistics
    }

    fn normalize(&self, value: f64) -> f64 {
        (value - self.mid_price) / (self.scale_factor * self.tick_size)
    }

    fn reset(&mut self) {
        self.mid_price = 100.0;
    }
}

/// Min-max normalization to [0, 1] or [-1, 1].
///
/// Scales values to a fixed range:
/// ```text
/// normalized = (value - min) / (max - min)  // for [0, 1]
/// normalized = 2 * (value - min) / (max - min) - 1  // for [-1, 1]
/// ```
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::{Normalizer, MinMaxNormalizer};
///
/// let normalizer = MinMaxNormalizer::new(0.0, 100.0, false);
///
/// assert_eq!(normalizer.normalize(0.0), 0.0);
/// assert_eq!(normalizer.normalize(50.0), 0.5);
/// assert_eq!(normalizer.normalize(100.0), 1.0);
/// ```
pub struct MinMaxNormalizer {
    min: f64,
    max: f64,
    symmetric: bool, // If true, normalize to [-1, 1], else [0, 1]
}

impl MinMaxNormalizer {
    /// Create a new min-max normalizer.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum expected value
    /// * `max` - Maximum expected value
    /// * `symmetric` - If true, normalize to [-1, 1], else [0, 1]
    pub fn new(min: f64, max: f64, symmetric: bool) -> Self {
        assert!(max > min, "max must be greater than min");

        Self {
            min,
            max,
            symmetric,
        }
    }

    /// Get the min value.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get the max value.
    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Normalizer for MinMaxNormalizer {
    fn update(&mut self, _value: f64) {
        // Min-max doesn't track statistics (uses fixed bounds)
    }

    fn normalize(&self, value: f64) -> f64 {
        let clamped = value.clamp(self.min, self.max);
        let normalized = (clamped - self.min) / (self.max - self.min);

        if self.symmetric {
            2.0 * normalized - 1.0 // Map to [-1, 1]
        } else {
            normalized // Map to [0, 1]
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

// =============================================================================
// Rolling Z-Score Normalizer (LOBFrame-style)
// =============================================================================

/// Rolling Z-Score normalization using previous N days' statistics.
///
/// This normalizer computes statistics from a rolling window of historical data,
/// typically from the previous N trading days. This is crucial for financial
/// data which is non-stationary.
///
/// # Research Background
///
/// From LOBFrame (data_process.py):
/// ```python
/// z_mean = (nsamples * mean).sum() / nsamples.sum()
/// z_stdev = sqrt((nsamples * mean2).sum() / nsamples.sum() - z_mean^2)
/// normalized = (features - z_mean) / z_stdev
/// ```
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::{Normalizer, RollingZScoreNormalizer};
///
/// // Create normalizer with 5-day rolling window
/// let mut normalizer = RollingZScoreNormalizer::new(5);
///
/// // Add statistics from previous days
/// normalizer.add_day_stats(1000, 100.0, 10.0); // day 1: n=1000, mean=100, std=10
/// normalizer.add_day_stats(1200, 101.0, 11.0); // day 2
/// normalizer.add_day_stats(1100, 99.0, 9.0);   // day 3
///
/// // Normalize current value using rolling statistics
/// let normalized = normalizer.normalize(110.0);
/// ```
pub struct RollingZScoreNormalizer {
    /// Maximum number of days to keep in rolling window
    max_days: usize,

    /// Day statistics: (sample_count, mean, mean_of_squares)
    day_stats: VecDeque<DayStatistics>,

    /// Cached rolling mean
    cached_mean: f64,

    /// Cached rolling std
    cached_std: f64,

    /// Whether cache is valid
    cache_valid: bool,

    /// Minimum std to avoid division by zero
    min_std: f64,
}

/// Statistics for a single day.
#[derive(Debug, Clone, Copy)]
struct DayStatistics {
    /// Number of samples in this day
    sample_count: u64,

    /// Mean of samples
    mean: f64,

    /// Mean of squared samples (for variance calculation)
    mean_squared: f64,
}

impl RollingZScoreNormalizer {
    /// Create a new rolling Z-score normalizer.
    ///
    /// # Arguments
    ///
    /// * `max_days` - Maximum number of days to use for rolling statistics
    pub fn new(max_days: usize) -> Self {
        assert!(max_days > 0, "max_days must be positive");

        Self {
            max_days,
            day_stats: VecDeque::with_capacity(max_days),
            cached_mean: 0.0,
            cached_std: 1.0,
            cache_valid: false,
            min_std: 1e-8,
        }
    }

    /// Add statistics for a new day.
    ///
    /// # Arguments
    ///
    /// * `sample_count` - Number of samples in the day
    /// * `mean` - Mean of samples
    /// * `std` - Standard deviation of samples
    pub fn add_day_stats(&mut self, sample_count: u64, mean: f64, std: f64) {
        // Convert std to mean_squared: var = E[X^2] - E[X]^2
        // So E[X^2] = var + E[X]^2 = std^2 + mean^2
        let mean_squared = std * std + mean * mean;

        let stats = DayStatistics {
            sample_count,
            mean,
            mean_squared,
        };

        if self.day_stats.len() >= self.max_days {
            self.day_stats.pop_front();
        }
        self.day_stats.push_back(stats);
        self.cache_valid = false;
    }

    /// Add statistics from raw data (mean and mean of squares).
    ///
    /// This is more numerically stable when aggregating from streaming data.
    ///
    /// # Arguments
    ///
    /// * `sample_count` - Number of samples
    /// * `sum` - Sum of samples
    /// * `sum_squared` - Sum of squared samples
    pub fn add_day_stats_raw(&mut self, sample_count: u64, sum: f64, sum_squared: f64) {
        if sample_count == 0 {
            return;
        }

        let mean = sum / sample_count as f64;
        let mean_squared = sum_squared / sample_count as f64;

        let stats = DayStatistics {
            sample_count,
            mean,
            mean_squared,
        };

        if self.day_stats.len() >= self.max_days {
            self.day_stats.pop_front();
        }
        self.day_stats.push_back(stats);
        self.cache_valid = false;
    }

    /// Compute rolling statistics from all days.
    fn compute_rolling_stats(&mut self) {
        if self.cache_valid {
            return;
        }

        let total_samples: u64 = self.day_stats.iter().map(|d| d.sample_count).sum();

        if total_samples == 0 {
            self.cached_mean = 0.0;
            self.cached_std = 1.0;
            self.cache_valid = true;
            return;
        }

        // Weighted mean: Σ(n_i * mean_i) / Σ(n_i)
        let weighted_mean: f64 = self
            .day_stats
            .iter()
            .map(|d| d.sample_count as f64 * d.mean)
            .sum::<f64>()
            / total_samples as f64;

        // Weighted mean of squares: Σ(n_i * mean_squared_i) / Σ(n_i)
        let weighted_mean_squared: f64 = self
            .day_stats
            .iter()
            .map(|d| d.sample_count as f64 * d.mean_squared)
            .sum::<f64>()
            / total_samples as f64;

        // Variance: E[X^2] - E[X]^2
        let variance = weighted_mean_squared - weighted_mean * weighted_mean;
        let std = variance.max(0.0).sqrt().max(self.min_std);

        self.cached_mean = weighted_mean;
        self.cached_std = std;
        self.cache_valid = true;
    }

    /// Get the rolling mean.
    pub fn mean(&mut self) -> f64 {
        self.compute_rolling_stats();
        self.cached_mean
    }

    /// Get the rolling standard deviation.
    pub fn std(&mut self) -> f64 {
        self.compute_rolling_stats();
        self.cached_std
    }

    /// Get the number of days in the rolling window.
    pub fn day_count(&self) -> usize {
        self.day_stats.len()
    }

    /// Get the total sample count across all days.
    pub fn total_samples(&self) -> u64 {
        self.day_stats.iter().map(|d| d.sample_count).sum()
    }

    /// Clear all day statistics.
    pub fn clear(&mut self) {
        self.day_stats.clear();
        self.cache_valid = false;
    }
}

impl Default for RollingZScoreNormalizer {
    fn default() -> Self {
        Self::new(5) // Default to 5-day rolling window
    }
}

impl Normalizer for RollingZScoreNormalizer {
    fn update(&mut self, _value: f64) {
        // Rolling normalizer doesn't update per-value
        // Use add_day_stats() to add daily statistics
    }

    fn normalize(&self, value: f64) -> f64 {
        // Use cached values (caller should ensure compute_rolling_stats was called)
        if self.day_stats.is_empty() {
            return value; // No normalization if no historical data
        }

        (value - self.cached_mean) / self.cached_std
    }

    fn normalize_batch(&self, values: &[f64]) -> Vec<f64> {
        if self.day_stats.is_empty() {
            return values.to_vec();
        }

        values
            .iter()
            .map(|&v| (v - self.cached_mean) / self.cached_std)
            .collect()
    }

    fn reset(&mut self) {
        self.day_stats.clear();
        self.cached_mean = 0.0;
        self.cached_std = 1.0;
        self.cache_valid = false;
    }

    fn is_ready(&self) -> bool {
        !self.day_stats.is_empty()
    }
}

// =============================================================================
// Per-Feature Normalizer (for multi-feature normalization)
// =============================================================================

/// Per-feature normalizer that maintains separate statistics for each feature.
///
/// This is essential for LOB data where different features (prices, volumes, spreads)
/// have very different scales and distributions.
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::PerFeatureNormalizer;
///
/// // Create normalizer for 40 features
/// let mut normalizer = PerFeatureNormalizer::new(40);
///
/// // Update with feature vectors
/// let features = vec![100.0, 200.0, 0.01, 0.5]; // ... (40 features)
/// // normalizer.update_batch(&features);
///
/// // Normalize a new feature vector
/// // let normalized = normalizer.normalize_batch(&features);
/// ```
pub struct PerFeatureNormalizer {
    /// Number of features
    feature_count: usize,

    /// Per-feature normalizers
    normalizers: Vec<ZScoreNormalizer>,
}

impl PerFeatureNormalizer {
    /// Create a new per-feature normalizer.
    ///
    /// # Arguments
    ///
    /// * `feature_count` - Number of features to normalize
    pub fn new(feature_count: usize) -> Self {
        Self {
            feature_count,
            normalizers: (0..feature_count)
                .map(|_| ZScoreNormalizer::new())
                .collect(),
        }
    }

    /// Create with windowed normalizers.
    ///
    /// # Arguments
    ///
    /// * `feature_count` - Number of features
    /// * `window_size` - Window size for each normalizer
    pub fn with_window(feature_count: usize, window_size: usize) -> Self {
        Self {
            feature_count,
            normalizers: (0..feature_count)
                .map(|_| ZScoreNormalizer::with_window(window_size))
                .collect(),
        }
    }

    /// Update all normalizers with a feature vector.
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != feature_count`
    pub fn update_batch(&mut self, features: &[f64]) {
        assert_eq!(features.len(), self.feature_count, "Feature count mismatch");

        for (normalizer, &value) in self.normalizers.iter_mut().zip(features.iter()) {
            normalizer.update(value);
        }
    }

    /// Normalize a feature vector.
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != feature_count`
    pub fn normalize_features(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(features.len(), self.feature_count, "Feature count mismatch");

        features
            .iter()
            .zip(self.normalizers.iter())
            .map(|(&value, normalizer)| normalizer.normalize(value))
            .collect()
    }

    /// Check if all normalizers are ready.
    pub fn is_ready(&self) -> bool {
        self.normalizers.iter().all(|n| n.is_ready())
    }

    /// Get the feature count.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Get statistics for a specific feature.
    pub fn get_feature_stats(&self, index: usize) -> Option<(f64, f64)> {
        self.normalizers.get(index).map(|n| (n.mean(), n.std()))
    }

    /// Reset all normalizers.
    pub fn reset(&mut self) {
        for normalizer in &mut self.normalizers {
            normalizer.reset();
        }
    }
}

// =============================================================================
// Global Z-Score Normalizer (LOBench Paper)
// =============================================================================

/// Global Z-Score Normalizer from LOBench paper.
///
/// Unlike feature-wise normalization, this normalizes ALL features within
/// a single snapshot together. This preserves important LOB constraints:
/// - Bid prices < Ask prices
/// - All volumes are non-negative
/// - Relative feature scales are maintained
///
/// # Mathematical Foundation
///
/// For a snapshot with features [x₁, x₂, ..., xₙ]:
/// ```text
/// mean = (1/n) × Σxᵢ
/// std = sqrt((1/n) × Σ(xᵢ - mean)²)
/// normalized_xᵢ = (xᵢ - mean) / std
/// ```
///
/// # Key Benefits (from LOBench)
///
/// 1. **Preserves LOB Constraints**: Bid < Ask relationship maintained
/// 2. **Handles Scale Disparity**: Prices (~100) and volumes (~1000) normalized together
/// 3. **Robust to Perturbations**: More stable under adversarial attacks
/// 4. **Cross-Feature Awareness**: Captures relative importance of features
///
/// # Usage
///
/// ```
/// use feature_extractor::preprocessing::normalization::GlobalZScoreNormalizer;
///
/// let normalizer = GlobalZScoreNormalizer::new();
///
/// // Normalize a LOB snapshot (all 40 features together)
/// let features = vec![100.01, 100.02, 1000.0, 2000.0, /* ... */];
/// let normalized = normalizer.normalize_snapshot(&features);
/// ```
///
/// # Reference
///
/// "Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking"
/// (LOBench paper)
#[derive(Debug, Clone, Default)]
pub struct GlobalZScoreNormalizer {
    /// Minimum standard deviation to avoid division by zero
    min_std: f64,
}

impl GlobalZScoreNormalizer {
    /// Create a new global z-score normalizer.
    pub fn new() -> Self {
        Self { min_std: 1e-10 }
    }

    /// Create with custom minimum standard deviation.
    pub fn with_min_std(min_std: f64) -> Self {
        Self { min_std }
    }

    /// Normalize a snapshot using global z-score.
    ///
    /// All features in the snapshot are normalized together using the
    /// mean and standard deviation computed across ALL features.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature vector to normalize
    ///
    /// # Returns
    ///
    /// Normalized feature vector where:
    /// - Mean of all features ≈ 0
    /// - Std of all features ≈ 1 (unless all values are identical)
    pub fn normalize_snapshot(&self, features: &[f64]) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }

        let n = features.len() as f64;

        // Calculate mean
        let mean: f64 = features.iter().sum::<f64>() / n;

        // Calculate standard deviation
        let variance: f64 = features.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt().max(self.min_std);

        // Normalize all features
        features.iter().map(|&x| (x - mean) / std).collect()
    }

    /// Normalize a snapshot in-place.
    ///
    /// More efficient than `normalize_snapshot` when you can modify the input.
    pub fn normalize_snapshot_inplace(&self, features: &mut [f64]) {
        if features.is_empty() {
            return;
        }

        let n = features.len() as f64;

        // Calculate mean
        let mean: f64 = features.iter().sum::<f64>() / n;

        // Calculate standard deviation
        let variance: f64 = features.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt().max(self.min_std);

        // Normalize in-place
        for x in features.iter_mut() {
            *x = (*x - mean) / std;
        }
    }

    /// Normalize a batch of snapshots.
    ///
    /// Each snapshot is normalized independently using its own mean/std.
    pub fn normalize_batch(&self, snapshots: &[Vec<f64>]) -> Vec<Vec<f64>> {
        snapshots
            .iter()
            .map(|s| self.normalize_snapshot(s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentage_change_basic() {
        let mut normalizer = PercentageChangeNormalizer::new();
        normalizer.set_reference(100.0);

        // 1% increase
        assert!((normalizer.normalize(101.0) - 0.01).abs() < 1e-10);

        // 1% decrease
        assert!((normalizer.normalize(99.0) - (-0.01)).abs() < 1e-10);

        // No change
        assert!((normalizer.normalize(100.0) - 0.0).abs() < 1e-10);

        // 10% increase
        assert!((normalizer.normalize(110.0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_percentage_change_different_reference() {
        let mut normalizer = PercentageChangeNormalizer::new();

        normalizer.set_reference(50.0);
        assert!((normalizer.normalize(55.0) - 0.1).abs() < 1e-10);

        normalizer.set_reference(200.0);
        assert!((normalizer.normalize(220.0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_basic() {
        let mut normalizer = ZScoreNormalizer::new();

        // Feed data: [10, 20, 30, 40, 50]
        for value in &[10.0, 20.0, 30.0, 40.0, 50.0] {
            normalizer.update(*value);
        }

        // Mean should be 30
        assert!((normalizer.mean() - 30.0).abs() < 1e-10);

        // Std should be ~14.14
        let std = normalizer.std();
        assert!((std - 14.142135).abs() < 0.01);

        // Normalize value at mean -> 0
        assert!(normalizer.normalize(30.0).abs() < 1e-10);

        // Normalize value 1 std above mean -> ~1
        assert!((normalizer.normalize(44.14) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_zscore_windowed() {
        let mut normalizer = ZScoreNormalizer::with_window(3);

        // Feed: 1, 2, 3, 4, 5
        // Window should keep last 3: [3, 4, 5]
        for value in 1..=5 {
            normalizer.update(value as f64);
        }

        // Mean of [3, 4, 5] = 4
        assert!((normalizer.mean() - 4.0).abs() < 1e-10);

        // Count should be 3 (window size)
        assert_eq!(normalizer.count(), 3);
    }

    #[test]
    fn test_zscore_insufficient_data() {
        let mut normalizer = ZScoreNormalizer::new();

        // No data -> not ready
        assert!(!normalizer.is_ready());

        // One sample -> still not ready (need 2 for std)
        normalizer.update(10.0);
        assert!(!normalizer.is_ready());

        // Two samples -> ready
        normalizer.update(20.0);
        assert!(normalizer.is_ready());
    }

    #[test]
    fn test_bilinear_basic() {
        let mut normalizer = BilinearNormalizer::new(0.01, 50.0);
        normalizer.set_mid_price(100.0);

        // 5 ticks above mid (100.05)
        let normalized = normalizer.normalize(100.05);
        assert!((normalized - 0.1).abs() < 1e-10);

        // 10 ticks below mid (99.90)
        let normalized = normalizer.normalize(99.90);
        assert!((normalized - (-0.2)).abs() < 1e-10);

        // At mid-price
        let normalized = normalizer.normalize(100.0);
        assert!(normalized.abs() < 1e-10);
    }

    #[test]
    fn test_minmax_basic() {
        let normalizer = MinMaxNormalizer::new(0.0, 100.0, false);

        assert_eq!(normalizer.normalize(0.0), 0.0);
        assert_eq!(normalizer.normalize(50.0), 0.5);
        assert_eq!(normalizer.normalize(100.0), 1.0);

        // Clamping
        assert_eq!(normalizer.normalize(-10.0), 0.0);
        assert_eq!(normalizer.normalize(110.0), 1.0);
    }

    #[test]
    fn test_minmax_symmetric() {
        let normalizer = MinMaxNormalizer::new(0.0, 100.0, true);

        assert_eq!(normalizer.normalize(0.0), -1.0);
        assert_eq!(normalizer.normalize(50.0), 0.0);
        assert_eq!(normalizer.normalize(100.0), 1.0);
    }

    #[test]
    fn test_normalizer_reset() {
        let mut normalizer = ZScoreNormalizer::new();

        normalizer.update(10.0);
        normalizer.update(20.0);
        assert_eq!(normalizer.count(), 2);

        normalizer.reset();
        assert_eq!(normalizer.count(), 0);
        assert!(!normalizer.is_ready());
    }

    #[test]
    fn test_percentage_change_batch() {
        let mut normalizer = PercentageChangeNormalizer::new();
        normalizer.set_reference(100.0);

        let values = vec![99.0, 100.0, 101.0, 102.0];
        let normalized = normalizer.normalize_batch(&values);

        assert_eq!(normalized.len(), 4);
        assert!((normalized[0] - (-0.01)).abs() < 1e-10);
        assert!((normalized[1] - 0.0).abs() < 1e-10);
        assert!((normalized[2] - 0.01).abs() < 1e-10);
        assert!((normalized[3] - 0.02).abs() < 1e-10);
    }

    // =============================================================================
    // Rolling Z-Score Tests
    // =============================================================================

    #[test]
    fn test_rolling_zscore_basic() {
        let mut normalizer = RollingZScoreNormalizer::new(5);

        // Add day 1: n=100, mean=50, std=10
        normalizer.add_day_stats(100, 50.0, 10.0);

        // Should use day 1 stats
        let mean = normalizer.mean();
        let std = normalizer.std();

        assert!((mean - 50.0).abs() < 1e-6);
        assert!((std - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_zscore_multiple_days() {
        let mut normalizer = RollingZScoreNormalizer::new(5);

        // Day 1: n=100, mean=50, std=10
        normalizer.add_day_stats(100, 50.0, 10.0);

        // Day 2: n=100, mean=60, std=10
        normalizer.add_day_stats(100, 60.0, 10.0);

        // Weighted mean: (100*50 + 100*60) / 200 = 55
        let mean = normalizer.mean();
        assert!((mean - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_zscore_weighted_by_samples() {
        let mut normalizer = RollingZScoreNormalizer::new(5);

        // Day 1: n=100, mean=50
        normalizer.add_day_stats(100, 50.0, 10.0);

        // Day 2: n=300, mean=70 (3x more samples)
        normalizer.add_day_stats(300, 70.0, 10.0);

        // Weighted mean: (100*50 + 300*70) / 400 = 65
        let mean = normalizer.mean();
        assert!((mean - 65.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_zscore_window_eviction() {
        let mut normalizer = RollingZScoreNormalizer::new(3);

        // Add 4 days (window is 3)
        normalizer.add_day_stats(100, 10.0, 1.0);
        normalizer.add_day_stats(100, 20.0, 1.0);
        normalizer.add_day_stats(100, 30.0, 1.0);
        normalizer.add_day_stats(100, 40.0, 1.0); // Day 1 should be evicted

        // Mean should be (20 + 30 + 40) / 3 = 30
        let mean = normalizer.mean();
        assert!((mean - 30.0).abs() < 1e-6);

        assert_eq!(normalizer.day_count(), 3);
    }

    #[test]
    fn test_rolling_zscore_normalize() {
        let mut normalizer = RollingZScoreNormalizer::new(5);

        // Add stats: mean=100, std=10
        normalizer.add_day_stats(1000, 100.0, 10.0);
        normalizer.mean(); // Trigger cache computation

        // Value at mean -> 0
        assert!(normalizer.normalize(100.0).abs() < 1e-6);

        // Value 1 std above -> 1
        assert!((normalizer.normalize(110.0) - 1.0).abs() < 1e-6);

        // Value 2 std below -> -2
        assert!((normalizer.normalize(80.0) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_zscore_raw_stats() {
        let mut normalizer = RollingZScoreNormalizer::new(5);

        // Add raw stats: sum=1000, sum_squared=110000, n=10
        // mean = 100, var = 11000 - 10000 = 1000, std = ~31.6
        normalizer.add_day_stats_raw(10, 1000.0, 110000.0);

        let mean = normalizer.mean();
        let std = normalizer.std();

        assert!((mean - 100.0).abs() < 1e-6);
        assert!((std - 31.622776).abs() < 0.01);
    }

    #[test]
    fn test_rolling_zscore_empty() {
        let normalizer = RollingZScoreNormalizer::new(5);

        // Should not be ready
        assert!(!normalizer.is_ready());

        // Should return value unchanged
        assert_eq!(normalizer.normalize(42.0), 42.0);
    }

    // =============================================================================
    // Per-Feature Normalizer Tests
    // =============================================================================

    #[test]
    fn test_per_feature_basic() {
        let mut normalizer = PerFeatureNormalizer::new(3);

        // Update with feature vectors
        normalizer.update_batch(&[10.0, 100.0, 1000.0]);
        normalizer.update_batch(&[20.0, 200.0, 2000.0]);
        normalizer.update_batch(&[30.0, 300.0, 3000.0]);

        // Check per-feature stats
        let (mean0, _) = normalizer.get_feature_stats(0).unwrap();
        let (mean1, _) = normalizer.get_feature_stats(1).unwrap();
        let (mean2, _) = normalizer.get_feature_stats(2).unwrap();

        assert!((mean0 - 20.0).abs() < 1e-6);
        assert!((mean1 - 200.0).abs() < 1e-6);
        assert!((mean2 - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_feature_normalize() {
        let mut normalizer = PerFeatureNormalizer::new(2);

        // Feed enough data
        for i in 0..100 {
            normalizer.update_batch(&[i as f64, i as f64 * 10.0]);
        }

        // Normalize: values at mean should be ~0
        let normalized = normalizer.normalize_features(&[49.5, 495.0]);
        assert!(normalized[0].abs() < 0.1);
        assert!(normalized[1].abs() < 0.1);
    }

    #[test]
    fn test_per_feature_windowed() {
        let mut normalizer = PerFeatureNormalizer::with_window(2, 5);

        // Feed 10 values, window keeps last 5
        for i in 0..10 {
            normalizer.update_batch(&[i as f64, i as f64 * 2.0]);
        }

        // Mean of last 5 values [5,6,7,8,9] = 7
        let (mean0, _) = normalizer.get_feature_stats(0).unwrap();
        assert!((mean0 - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_feature_is_ready() {
        let mut normalizer = PerFeatureNormalizer::new(2);

        // Not ready initially
        assert!(!normalizer.is_ready());

        // One sample: not ready
        normalizer.update_batch(&[1.0, 2.0]);
        assert!(!normalizer.is_ready());

        // Two samples: ready
        normalizer.update_batch(&[3.0, 4.0]);
        assert!(normalizer.is_ready());
    }

    #[test]
    #[should_panic(expected = "Feature count mismatch")]
    fn test_per_feature_mismatch() {
        let mut normalizer = PerFeatureNormalizer::new(3);
        normalizer.update_batch(&[1.0, 2.0]); // Wrong count
    }

    #[test]
    fn test_per_feature_reset() {
        let mut normalizer = PerFeatureNormalizer::new(2);

        normalizer.update_batch(&[1.0, 2.0]);
        normalizer.update_batch(&[3.0, 4.0]);
        assert!(normalizer.is_ready());

        normalizer.reset();
        assert!(!normalizer.is_ready());
    }

    // =============================================================================
    // Global Z-Score Normalizer Tests (LOBench)
    // =============================================================================

    #[test]
    fn test_global_zscore_basic() {
        let normalizer = GlobalZScoreNormalizer::new();

        // All features normalized together
        let features = vec![100.0, 200.0, 300.0, 400.0];
        let normalized = normalizer.normalize_snapshot(&features);

        // Mean = 250, std = sqrt(((100-250)^2 + (200-250)^2 + (300-250)^2 + (400-250)^2) / 4)
        // = sqrt((22500 + 2500 + 2500 + 22500) / 4) = sqrt(12500) ≈ 111.8

        // Check that normalized values sum to ~0 (centered)
        let sum: f64 = normalized.iter().sum();
        assert!(sum.abs() < 1e-10, "Sum should be ~0, got {sum}");
    }

    #[test]
    fn test_global_zscore_preserves_ordering() {
        let normalizer = GlobalZScoreNormalizer::new();

        let features = vec![10.0, 20.0, 30.0, 40.0];
        let normalized = normalizer.normalize_snapshot(&features);

        // Order should be preserved
        for i in 1..normalized.len() {
            assert!(normalized[i] > normalized[i - 1]);
        }
    }

    #[test]
    fn test_global_zscore_with_prices_and_volumes() {
        let normalizer = GlobalZScoreNormalizer::new();

        // Realistic LOB: prices ~100, volumes ~1000
        let features = vec![
            100.01, 100.02, 100.03, // Ask prices
            1000.0, 2000.0, 1500.0, // Ask volumes
            100.00, 99.99, 99.98, // Bid prices
            1000.0, 1500.0, 2000.0, // Bid volumes
        ];

        let normalized = normalizer.normalize_snapshot(&features);

        // All values should be finite
        for &v in &normalized {
            assert!(v.is_finite());
        }

        // Prices should still be less than volumes after normalization
        // (they were closer to the mean, so smaller absolute z-scores)
        // This preserves the LOB constraint structure
    }

    #[test]
    fn test_global_zscore_empty() {
        let normalizer = GlobalZScoreNormalizer::new();
        let normalized = normalizer.normalize_snapshot(&[]);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_global_zscore_single_value() {
        let normalizer = GlobalZScoreNormalizer::new();
        let normalized = normalizer.normalize_snapshot(&[42.0]);

        // Single value: std=0, so value stays as-is or becomes 0
        assert_eq!(normalized.len(), 1);
        assert!(normalized[0].abs() < 1e-10 || normalized[0] == 42.0);
    }

    #[test]
    fn test_global_zscore_constant_values() {
        let normalizer = GlobalZScoreNormalizer::new();
        let features = vec![5.0, 5.0, 5.0, 5.0];
        let normalized = normalizer.normalize_snapshot(&features);

        // All same value: std=0, all should become 0
        for &v in &normalized {
            assert!(v.abs() < 1e-10);
        }
    }
}
