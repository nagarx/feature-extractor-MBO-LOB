//! Magnitude-Based Return Generation for Regression Experiments
//!
//! This module provides utilities for generating continuous return values
//! instead of discrete classification labels. This enables:
//!
//! - **Regression models**: Predict exact return magnitude
//! - **Position sizing**: Scale position size by predicted return
//! - **Flexible thresholding**: Apply thresholds at inference time, not training
//!
//! # Overview
//!
//! Instead of classifying returns into Up/Down/Stable, this module exports
//! the actual return values, allowing downstream models to:
//!
//! 1. Learn to predict return magnitude directly
//! 2. Apply custom thresholds at inference time
//! 3. Weight predictions by confidence (predicted magnitude)
//!
//! # Return Types
//!
//! | Type | Description | Use Case |
//! |------|-------------|----------|
//! | `PointReturn` | Return at a specific horizon | Simple prediction |
//! | `PeakReturn` | Max and min returns in horizon | Opportunity detection |
//! | `PathStats` | Mean, std, skew of returns | Path analysis |
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{MagnitudeConfig, MagnitudeGenerator, ReturnType};
//!
//! // Configure for peak returns with 50-event horizon
//! let config = MagnitudeConfig::peak_returns(50);
//! let mut generator = MagnitudeGenerator::new(config);
//!
//! // Add prices
//! let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0).collect();
//! generator.add_prices(&prices);
//!
//! let returns = generator.generate_returns().unwrap();
//! for (idx, ret) in &returns {
//!     println!("t={}: return={:.4}%", idx, ret.point_return * 100.0);
//! }
//! ```

use mbo_lob_reconstructor::{Result, TlobError};
use serde::{Deserialize, Serialize};

// ============================================================================
// Return Output Types
// ============================================================================

/// Return data for a single entry point.
///
/// Contains various return metrics computed for a given horizon.
/// All returns are expressed as decimals (e.g., 0.01 = 1%).
#[derive(Debug, Clone, Copy)]
pub struct ReturnData {
    /// Simple point-to-point return at horizon.
    ///
    /// Computed as: (price[t+h] - price[t]) / price[t]
    pub point_return: f64,

    /// Maximum return achieved within the horizon.
    ///
    /// Computed as: max(price[t+1..t+h+1]) / price[t] - 1
    pub max_return: f64,

    /// Minimum return observed within the horizon.
    ///
    /// Computed as: min(price[t+1..t+h+1]) / price[t] - 1
    pub min_return: f64,

    /// Mean return across the horizon (averaged over all timesteps).
    ///
    /// Computed as: mean(price[t+1..t+h+1]) / price[t] - 1
    pub mean_return: f64,

    /// Standard deviation of returns within the horizon.
    ///
    /// Measures path volatility during the holding period.
    pub return_std: f64,

    /// Time to maximum return (offset from entry).
    ///
    /// Useful for understanding when the opportunity peaks.
    pub time_to_max: usize,

    /// Time to minimum return (offset from entry).
    ///
    /// Useful for understanding when the drawdown occurs.
    pub time_to_min: usize,
}

impl Default for ReturnData {
    fn default() -> Self {
        Self {
            point_return: 0.0,
            max_return: 0.0,
            min_return: 0.0,
            mean_return: 0.0,
            return_std: 0.0,
            time_to_max: 0,
            time_to_min: 0,
        }
    }
}

impl ReturnData {
    /// Get the larger magnitude return (max or min, whichever is bigger in absolute terms).
    pub fn dominant_return(&self) -> f64 {
        if self.max_return.abs() >= self.min_return.abs() {
            self.max_return
        } else {
            self.min_return
        }
    }

    /// Get the range of returns (max - min).
    pub fn return_range(&self) -> f64 {
        self.max_return - self.min_return
    }

    /// Check if this represents an upward opportunity (max > -min).
    pub fn is_bullish(&self) -> bool {
        self.max_return > -self.min_return
    }

    /// Check if this represents a downward opportunity (|min| > max).
    pub fn is_bearish(&self) -> bool {
        -self.min_return > self.max_return
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Type of return to compute as the primary output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ReturnType {
    /// Simple point return at horizon: (price[t+h] - price[t]) / price[t]
    #[default]
    PointReturn,

    /// Maximum return within horizon (for long opportunities)
    MaxReturn,

    /// Minimum return within horizon (for short opportunities)
    MinReturn,

    /// Dominant return (max or min, whichever is larger in magnitude)
    DominantReturn,

    /// Mean return over the horizon
    MeanReturn,
}

/// Configuration for magnitude/return generation.
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::{MagnitudeConfig, ReturnType};
///
/// // Simple point return at horizon 50
/// let config = MagnitudeConfig::point_return(50);
///
/// // Peak returns (max/min) for opportunity detection
/// let config = MagnitudeConfig::peak_returns(100);
///
/// // Multi-horizon returns
/// let config = MagnitudeConfig::multi_horizon(vec![10, 50, 100, 200]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagnitudeConfig {
    /// Prediction horizons.
    ///
    /// If single horizon, output is a single return value per sample.
    /// If multiple horizons, output is an array of returns per sample.
    pub horizons: Vec<usize>,

    /// Which return type to use as the primary output.
    #[serde(default)]
    pub return_type: ReturnType,

    /// Whether to compute all return statistics (not just primary).
    ///
    /// If true, generates full ReturnData for each sample.
    /// If false, only generates the primary return type.
    #[serde(default)]
    pub compute_all_stats: bool,

    /// Optional smoothing window for computing smoothed returns.
    ///
    /// If set, applies TLOB-style smoothing to the return calculation.
    /// If None, uses raw prices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smoothing_window: Option<usize>,
}

impl MagnitudeConfig {
    /// Create config for simple point return at a single horizon.
    pub fn point_return(horizon: usize) -> Self {
        Self {
            horizons: vec![horizon],
            return_type: ReturnType::PointReturn,
            compute_all_stats: false,
            smoothing_window: None,
        }
    }

    /// Create config for peak returns (max/min) at a single horizon.
    ///
    /// This is useful for opportunity detection - identifying the best
    /// possible entry points for long or short positions.
    pub fn peak_returns(horizon: usize) -> Self {
        Self {
            horizons: vec![horizon],
            return_type: ReturnType::DominantReturn,
            compute_all_stats: true,
            smoothing_window: None,
        }
    }

    /// Create config for multi-horizon returns.
    ///
    /// Generates returns at multiple horizons for each entry point,
    /// allowing analysis of signal decay over time.
    pub fn multi_horizon(horizons: Vec<usize>) -> Self {
        assert!(!horizons.is_empty(), "horizons cannot be empty");
        Self {
            horizons,
            return_type: ReturnType::PointReturn,
            compute_all_stats: false,
            smoothing_window: None,
        }
    }

    /// Enable computation of all return statistics.
    pub fn with_all_stats(mut self) -> Self {
        self.compute_all_stats = true;
        self
    }

    /// Set smoothing window for TLOB-style smoothed returns.
    pub fn with_smoothing(mut self, window: usize) -> Self {
        self.smoothing_window = Some(window);
        self
    }

    /// Get the maximum horizon (for determining minimum required prices).
    pub fn max_horizon(&self) -> usize {
        *self.horizons.iter().max().unwrap_or(&0)
    }

    /// Calculate minimum required prices.
    pub fn min_prices_required(&self) -> usize {
        let max_h = self.max_horizon();
        let smooth = self.smoothing_window.unwrap_or(0);
        // Need enough prices for smoothing + max horizon
        smooth + max_h + 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.horizons.is_empty() {
            return Err(TlobError::generic("horizons cannot be empty"));
        }
        for &h in &self.horizons {
            if h == 0 {
                return Err(TlobError::generic("horizon must be > 0"));
            }
        }
        if let Some(s) = self.smoothing_window {
            if s == 0 {
                return Err(TlobError::generic("smoothing_window must be > 0 if set"));
            }
        }
        Ok(())
    }
}

impl Default for MagnitudeConfig {
    fn default() -> Self {
        Self::point_return(50)
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for magnitude generation.
#[derive(Debug, Clone, Default)]
pub struct MagnitudeStats {
    /// Total number of samples generated
    pub total: usize,

    /// Average point return across all samples
    pub avg_point_return: f64,

    /// Standard deviation of point returns
    pub std_point_return: f64,

    /// Average maximum return
    pub avg_max_return: f64,

    /// Average minimum return
    pub avg_min_return: f64,

    /// Skewness of return distribution
    pub skewness: f64,

    /// Fraction of samples with positive point return
    pub positive_rate: f64,
}

// ============================================================================
// Generator
// ============================================================================

/// Magnitude/return generator.
///
/// Generates continuous return values for regression experiments.
///
/// # Design
///
/// - Computes returns for each entry point and horizon
/// - Supports multiple return types (point, peak, mean, etc.)
/// - Optionally computes full return statistics
///
/// # Thread Safety
///
/// Not thread-safe. Use separate instances for parallel processing.
pub struct MagnitudeGenerator {
    config: MagnitudeConfig,
    prices: Vec<f64>,
}

/// Output for single-horizon magnitude generation.
pub type MagnitudeOutput = (usize, ReturnData);

/// Output for multi-horizon magnitude generation.
pub type MultiHorizonMagnitudeOutput = (usize, Vec<ReturnData>);

impl MagnitudeGenerator {
    /// Create a new magnitude generator.
    pub fn new(config: MagnitudeConfig) -> Self {
        Self {
            config,
            prices: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(config: MagnitudeConfig, capacity: usize) -> Self {
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
    pub fn config(&self) -> &MagnitudeConfig {
        &self.config
    }

    /// Check if we have enough prices to generate returns.
    pub fn can_generate(&self) -> bool {
        self.prices.len() >= self.config.min_prices_required()
    }

    /// Generate returns for all valid entry points (single horizon).
    ///
    /// Only valid when config has a single horizon.
    pub fn generate_returns(&self) -> Result<Vec<MagnitudeOutput>> {
        if self.config.horizons.len() != 1 {
            return Err(TlobError::generic(
                "Use generate_multi_horizon_returns for multiple horizons",
            ));
        }

        let horizon = self.config.horizons[0];
        let total = self.prices.len();
        let min_required = self.config.min_prices_required();

        if total < min_required {
            return Err(TlobError::generic(format!(
                "Need at least {} prices for magnitude generation (have {})",
                min_required, total
            )));
        }

        let smooth = self.config.smoothing_window.unwrap_or(0);
        let mut results = Vec::with_capacity(total - horizon - smooth);

        // Generate returns for each valid entry point
        for t in smooth..(total - horizon) {
            let entry_price = self.get_entry_price(t);
            let return_data = self.compute_returns(t, horizon, entry_price);
            results.push((t, return_data));
        }

        Ok(results)
    }

    /// Generate returns for all horizons at each entry point.
    pub fn generate_multi_horizon_returns(&self) -> Result<Vec<MultiHorizonMagnitudeOutput>> {
        let max_h = self.config.max_horizon();
        let total = self.prices.len();
        let min_required = self.config.min_prices_required();

        if total < min_required {
            return Err(TlobError::generic(format!(
                "Need at least {} prices for magnitude generation (have {})",
                min_required, total
            )));
        }

        let smooth = self.config.smoothing_window.unwrap_or(0);
        let mut results = Vec::with_capacity(total - max_h - smooth);

        // Generate returns for each valid entry point
        for t in smooth..(total - max_h) {
            let entry_price = self.get_entry_price(t);

            let horizon_returns: Vec<ReturnData> = self
                .config
                .horizons
                .iter()
                .map(|&h| self.compute_returns(t, h, entry_price))
                .collect();

            results.push((t, horizon_returns));
        }

        Ok(results)
    }

    /// Get the primary return value based on configuration.
    pub fn generate_primary_returns(&self) -> Result<Vec<(usize, f64)>> {
        let returns = self.generate_returns()?;

        let primary: Vec<(usize, f64)> = returns
            .into_iter()
            .map(|(idx, data)| {
                let value = match self.config.return_type {
                    ReturnType::PointReturn => data.point_return,
                    ReturnType::MaxReturn => data.max_return,
                    ReturnType::MinReturn => data.min_return,
                    ReturnType::DominantReturn => data.dominant_return(),
                    ReturnType::MeanReturn => data.mean_return,
                };
                (idx, value)
            })
            .collect();

        Ok(primary)
    }

    /// Get entry price (with optional smoothing).
    fn get_entry_price(&self, t: usize) -> f64 {
        if let Some(window) = self.config.smoothing_window {
            // TLOB-style smoothed entry price
            let start = t.saturating_sub(window);
            let sum: f64 = self.prices[start..=t].iter().sum();
            sum / (window + 1) as f64
        } else {
            self.prices[t]
        }
    }

    /// Compute all return statistics for a given entry point and horizon.
    fn compute_returns(&self, t: usize, horizon: usize, entry_price: f64) -> ReturnData {
        let window = &self.prices[(t + 1)..=(t + horizon)];

        // Find max, min, and their positions
        let mut max_price = f64::NEG_INFINITY;
        let mut min_price = f64::INFINITY;
        let mut max_idx = 0;
        let mut min_idx = 0;
        let mut sum = 0.0;

        for (i, &price) in window.iter().enumerate() {
            if price > max_price {
                max_price = price;
                max_idx = i + 1; // Offset from entry
            }
            if price < min_price {
                min_price = price;
                min_idx = i + 1;
            }
            sum += price;
        }

        let mean_price = sum / horizon as f64;

        // Compute returns relative to entry price
        let point_return = (self.prices[t + horizon] - entry_price) / entry_price;
        let max_return = (max_price - entry_price) / entry_price;
        let min_return = (min_price - entry_price) / entry_price;
        let mean_return = (mean_price - entry_price) / entry_price;

        // Compute return standard deviation
        let return_std = if self.config.compute_all_stats {
            let returns: Vec<f64> = window
                .iter()
                .map(|&p| (p - entry_price) / entry_price)
                .collect();
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / returns.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        ReturnData {
            point_return,
            max_return,
            min_return,
            mean_return,
            return_std,
            time_to_max: max_idx,
            time_to_min: min_idx,
        }
    }

    /// Compute statistics for the generated returns.
    pub fn compute_stats(&self, returns: &[MagnitudeOutput]) -> MagnitudeStats {
        if returns.is_empty() {
            return MagnitudeStats::default();
        }

        let n = returns.len() as f64;
        let mut sum_point = 0.0;
        let mut sum_max = 0.0;
        let mut sum_min = 0.0;
        let mut positive_count = 0;

        for (_, data) in returns {
            sum_point += data.point_return;
            sum_max += data.max_return;
            sum_min += data.min_return;
            if data.point_return > 0.0 {
                positive_count += 1;
            }
        }

        let avg_point = sum_point / n;
        let avg_max = sum_max / n;
        let avg_min = sum_min / n;

        // Compute std dev
        let variance: f64 = returns
            .iter()
            .map(|(_, d)| (d.point_return - avg_point).powi(2))
            .sum::<f64>()
            / n;
        let std_point = variance.sqrt();

        // Compute skewness
        let skewness = if std_point > 0.0 {
            let m3: f64 = returns
                .iter()
                .map(|(_, d)| ((d.point_return - avg_point) / std_point).powi(3))
                .sum::<f64>()
                / n;
            m3
        } else {
            0.0
        };

        MagnitudeStats {
            total: returns.len(),
            avg_point_return: avg_point,
            std_point_return: std_point,
            avg_max_return: avg_max,
            avg_min_return: avg_min,
            skewness,
            positive_rate: positive_count as f64 / n,
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
    // ReturnData Tests
    // ========================================================================

    #[test]
    fn test_return_data_dominant() {
        let data = ReturnData {
            max_return: 0.05,
            min_return: -0.03,
            ..Default::default()
        };
        assert_eq!(data.dominant_return(), 0.05); // Max is larger

        let data = ReturnData {
            max_return: 0.03,
            min_return: -0.05,
            ..Default::default()
        };
        assert_eq!(data.dominant_return(), -0.05); // Min is larger in magnitude
    }

    #[test]
    fn test_return_data_bullish_bearish() {
        let bullish = ReturnData {
            max_return: 0.05,
            min_return: -0.02,
            ..Default::default()
        };
        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());

        let bearish = ReturnData {
            max_return: 0.02,
            min_return: -0.05,
            ..Default::default()
        };
        assert!(!bearish.is_bullish());
        assert!(bearish.is_bearish());
    }

    // ========================================================================
    // MagnitudeConfig Tests
    // ========================================================================

    #[test]
    fn test_config_point_return() {
        let config = MagnitudeConfig::point_return(50);
        assert_eq!(config.horizons, vec![50]);
        assert_eq!(config.return_type, ReturnType::PointReturn);
        assert!(!config.compute_all_stats);
    }

    #[test]
    fn test_config_peak_returns() {
        let config = MagnitudeConfig::peak_returns(100);
        assert_eq!(config.horizons, vec![100]);
        assert_eq!(config.return_type, ReturnType::DominantReturn);
        assert!(config.compute_all_stats);
    }

    #[test]
    fn test_config_multi_horizon() {
        let config = MagnitudeConfig::multi_horizon(vec![10, 50, 100]);
        assert_eq!(config.horizons, vec![10, 50, 100]);
        assert_eq!(config.max_horizon(), 100);
    }

    #[test]
    fn test_config_validation() {
        let valid = MagnitudeConfig::point_return(50);
        assert!(valid.validate().is_ok());

        // Empty horizons
        let invalid = MagnitudeConfig {
            horizons: vec![],
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Zero horizon
        let invalid = MagnitudeConfig {
            horizons: vec![0],
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Zero smoothing
        let invalid = MagnitudeConfig {
            horizons: vec![50],
            smoothing_window: Some(0),
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_config_min_prices() {
        let config = MagnitudeConfig::point_return(50);
        assert_eq!(config.min_prices_required(), 51);

        let config = MagnitudeConfig::multi_horizon(vec![10, 50, 100]);
        assert_eq!(config.min_prices_required(), 101);

        let config = MagnitudeConfig::point_return(50).with_smoothing(10);
        assert_eq!(config.min_prices_required(), 61); // 10 + 50 + 1
    }

    // ========================================================================
    // MagnitudeGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_creation() {
        let config = MagnitudeConfig::point_return(10);
        let gen = MagnitudeGenerator::new(config);
        assert!(gen.is_empty());
        assert!(!gen.can_generate());
    }

    #[test]
    fn test_generator_can_generate() {
        let config = MagnitudeConfig::point_return(10);
        let mut gen = MagnitudeGenerator::new(config);

        let prices: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        gen.add_prices(&prices);
        assert!(!gen.can_generate()); // 10 < 11

        gen.add_price(110.0);
        assert!(gen.can_generate()); // 11 >= 11
    }

    #[test]
    fn test_generator_point_return_calculation() {
        let config = MagnitudeConfig::point_return(5);
        let mut gen = MagnitudeGenerator::new(config);

        // Entry at 100, exit at 105 → 5% return
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        gen.add_prices(&prices);

        let returns = gen.generate_returns().unwrap();
        assert!(!returns.is_empty());

        let (idx, data) = &returns[0];
        assert_eq!(*idx, 0);
        assert!((data.point_return - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_generator_peak_returns() {
        let config = MagnitudeConfig::peak_returns(5);
        let mut gen = MagnitudeGenerator::new(config);

        // Prices: 100, 102, 98, 101, 99, 100
        // Max = 102 (2% up), Min = 98 (2% down)
        let prices = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0];
        gen.add_prices(&prices);

        let returns = gen.generate_returns().unwrap();
        let (_, data) = &returns[0];

        assert!((data.max_return - 0.02).abs() < 1e-10);
        assert!((data.min_return - (-0.02)).abs() < 1e-10);
        assert_eq!(data.time_to_max, 1); // Max at offset 1
        assert_eq!(data.time_to_min, 2); // Min at offset 2
    }

    #[test]
    fn test_generator_multi_horizon() {
        let config = MagnitudeConfig::multi_horizon(vec![2, 5]);
        let mut gen = MagnitudeGenerator::new(config);

        // Linear increase: 100, 101, 102, 103, 104, 105
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        gen.add_prices(&prices);

        let returns = gen.generate_multi_horizon_returns().unwrap();
        assert!(!returns.is_empty());

        let (idx, horizon_data) = &returns[0];
        assert_eq!(*idx, 0);
        assert_eq!(horizon_data.len(), 2);

        // h=2: return = (102 - 100) / 100 = 2%
        assert!((horizon_data[0].point_return - 0.02).abs() < 1e-10);

        // h=5: return = (105 - 100) / 100 = 5%
        assert!((horizon_data[1].point_return - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_generator_with_smoothing() {
        let config = MagnitudeConfig::point_return(3).with_smoothing(2);
        let mut gen = MagnitudeGenerator::new(config);

        // Prices: 100, 101, 102, 103, 104, 105, 106
        // With smoothing=2, entry at t=2 uses avg(100, 101, 102) = 101.0
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0];
        gen.add_prices(&prices);

        let returns = gen.generate_returns().unwrap();
        let (idx, data) = &returns[0];

        assert_eq!(*idx, 2); // First valid entry is at t=2 (after smoothing window)

        // Smoothed entry = (100 + 101 + 102) / 3 = 101.0
        // Exit at t=5 = 105.0
        // Return = (105 - 101) / 101 ≈ 3.96%
        assert!((data.point_return - 0.0396).abs() < 0.001);
    }

    #[test]
    fn test_generator_primary_returns() {
        let config = MagnitudeConfig {
            horizons: vec![5],
            return_type: ReturnType::MaxReturn,
            compute_all_stats: true,
            smoothing_window: None,
        };
        let mut gen = MagnitudeGenerator::new(config);

        // Prices with a spike
        let prices = vec![100.0, 101.0, 105.0, 103.0, 102.0, 101.0];
        gen.add_prices(&prices);

        let primary = gen.generate_primary_returns().unwrap();
        let (_, value) = &primary[0];

        // Max return is 5% (price hit 105)
        assert!((value - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_generator_deterministic() {
        let config = MagnitudeConfig::peak_returns(10);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 3.0)
            .collect();

        let mut gen1 = MagnitudeGenerator::new(config.clone());
        gen1.add_prices(&prices);
        let ret1 = gen1.generate_returns().unwrap();

        let mut gen2 = MagnitudeGenerator::new(config);
        gen2.add_prices(&prices);
        let ret2 = gen2.generate_returns().unwrap();

        assert_eq!(ret1.len(), ret2.len());
        for (r1, r2) in ret1.iter().zip(ret2.iter()) {
            assert_eq!(r1.0, r2.0);
            assert!((r1.1.point_return - r2.1.point_return).abs() < 1e-15);
            assert!((r1.1.max_return - r2.1.max_return).abs() < 1e-15);
            assert!((r1.1.min_return - r2.1.min_return).abs() < 1e-15);
        }
    }

    #[test]
    fn test_generator_no_nan_or_inf() {
        let config = MagnitudeConfig::peak_returns(10);
        let mut gen = MagnitudeGenerator::new(config);

        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();
        gen.add_prices(&prices);

        let returns = gen.generate_returns().unwrap();
        for (_, data) in &returns {
            assert!(!data.point_return.is_nan());
            assert!(!data.point_return.is_infinite());
            assert!(!data.max_return.is_nan());
            assert!(!data.min_return.is_nan());
        }
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_stats_computation() {
        let config = MagnitudeConfig::peak_returns(10);
        let mut gen = MagnitudeGenerator::new(config);

        // Upward trending prices → positive returns
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        gen.add_prices(&prices);

        let returns = gen.generate_returns().unwrap();
        let stats = gen.compute_stats(&returns);

        assert_eq!(stats.total, returns.len());
        assert!(stats.avg_point_return > 0.0); // Upward trend
        assert!(stats.positive_rate > 0.5); // More positive than negative
    }

    #[test]
    fn test_stats_empty() {
        let gen = MagnitudeGenerator::new(MagnitudeConfig::point_return(10));
        let stats = gen.compute_stats(&[]);
        assert_eq!(stats.total, 0);
        assert_eq!(stats.positive_rate, 0.0);
    }
}

