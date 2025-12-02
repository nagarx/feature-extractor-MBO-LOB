//! Feature Validation Module
//!
//! Provides validation utilities for LOB features to ensure data quality
//! and detect anomalies before they propagate through the ML pipeline.
//!
//! # Validation Categories
//!
//! 1. **LOB Consistency**: Crossed quotes, locked quotes, price ordering
//! 2. **Feature Ranges**: NaN/Inf detection, reasonable bounds
//! 3. **Volume Sanity**: Non-negative volumes, reasonable sizes
//! 4. **Timestamp Ordering**: Monotonic timestamps, gap detection
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::validation::{FeatureValidator, ValidationResult};
//!
//! let validator = FeatureValidator::default();
//! let result = validator.validate_lob(&lob_state);
//!
//! if !result.is_valid() {
//!     for warning in result.warnings() {
//!         println!("Warning: {}", warning);
//!     }
//! }
//! ```

use mbo_lob_reconstructor::LobState;
use std::fmt;

/// Validation result for a single check.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationLevel {
    /// Data is valid
    Valid,
    /// Data has minor issues (warnings)
    Warning(String),
    /// Data has serious issues (errors)
    Error(String),
}

impl ValidationLevel {
    /// Check if this result indicates valid data.
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationLevel::Valid)
    }

    /// Check if this result is a warning.
    pub fn is_warning(&self) -> bool {
        matches!(self, ValidationLevel::Warning(_))
    }

    /// Check if this result is an error.
    pub fn is_error(&self) -> bool {
        matches!(self, ValidationLevel::Error(_))
    }
}

impl fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationLevel::Valid => write!(f, "Valid"),
            ValidationLevel::Warning(msg) => write!(f, "Warning: {msg}"),
            ValidationLevel::Error(msg) => write!(f, "Error: {msg}"),
        }
    }
}

/// Aggregated validation result.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// All validation results
    results: Vec<(String, ValidationLevel)>,
}

impl ValidationResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a validation result.
    pub fn add(&mut self, check_name: &str, level: ValidationLevel) {
        self.results.push((check_name.to_string(), level));
    }

    /// Check if all validations passed (no errors or warnings).
    pub fn is_valid(&self) -> bool {
        self.results.iter().all(|(_, level)| level.is_valid())
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        self.results.iter().any(|(_, level)| level.is_error())
    }

    /// Check if there are any warnings.
    pub fn has_warnings(&self) -> bool {
        self.results.iter().any(|(_, level)| level.is_warning())
    }

    /// Get all warnings.
    pub fn warnings(&self) -> Vec<&str> {
        self.results
            .iter()
            .filter_map(|(name, level)| match level {
                ValidationLevel::Warning(msg) => Some(format!("{name}: {msg}").leak() as &str),
                _ => None,
            })
            .collect()
    }

    /// Get all errors.
    pub fn errors(&self) -> Vec<&str> {
        self.results
            .iter()
            .filter_map(|(name, level)| match level {
                ValidationLevel::Error(msg) => Some(format!("{name}: {msg}").leak() as &str),
                _ => None,
            })
            .collect()
    }

    /// Get all results.
    pub fn all_results(&self) -> &[(String, ValidationLevel)] {
        &self.results
    }

    /// Get the number of checks performed.
    pub fn check_count(&self) -> usize {
        self.results.len()
    }

    /// Get the number of passed checks.
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|(_, l)| l.is_valid()).count()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let passed = self.passed_count();
        let total = self.check_count();
        writeln!(f, "Validation: {passed}/{total} checks passed")?;

        for (name, level) in &self.results {
            if !level.is_valid() {
                writeln!(f, "  - {name}: {level}")?;
            }
        }

        Ok(())
    }
}

/// Configuration for feature validation.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum allowed spread in basis points
    pub max_spread_bps: f64,

    /// Maximum allowed price in dollars
    pub max_price: f64,

    /// Minimum allowed price in dollars
    pub min_price: f64,

    /// Maximum allowed volume per level
    pub max_volume: u64,

    /// Maximum allowed price gap between levels (in ticks)
    pub max_level_gap_ticks: f64,

    /// Tick size for gap calculation
    pub tick_size: f64,

    /// Check for crossed quotes (bid >= ask)
    pub check_crossed_quotes: bool,

    /// Check for locked quotes (bid == ask)
    pub check_locked_quotes: bool,

    /// Check for NaN/Inf values
    pub check_nan_inf: bool,

    /// Check for price ordering (prices should decrease for bids, increase for asks)
    pub check_price_ordering: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_spread_bps: 1000.0,      // 10% max spread
            max_price: 1_000_000.0,      // $1M max price
            min_price: 0.0001,           // $0.0001 min price
            max_volume: 1_000_000_000,   // 1B shares max
            max_level_gap_ticks: 1000.0, // 1000 ticks max gap
            tick_size: 0.01,
            check_crossed_quotes: true,
            check_locked_quotes: true,
            check_nan_inf: true,
            check_price_ordering: true,
        }
    }
}

/// Feature validator for LOB data.
#[derive(Debug, Clone, Default)]
pub struct FeatureValidator {
    config: ValidationConfig,
}

impl FeatureValidator {
    /// Create a new validator with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a validator with custom configuration.
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a LOB state.
    pub fn validate_lob(&self, lob: &LobState) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check for crossed/locked quotes
        if self.config.check_crossed_quotes || self.config.check_locked_quotes {
            self.validate_quote_consistency(lob, &mut result);
        }

        // Check price ordering
        if self.config.check_price_ordering {
            self.validate_price_ordering(lob, &mut result);
        }

        // Check price ranges
        self.validate_price_ranges(lob, &mut result);

        // Check volume ranges
        self.validate_volume_ranges(lob, &mut result);

        // Check spread
        self.validate_spread(lob, &mut result);

        result
    }

    /// Validate a feature vector.
    pub fn validate_features(&self, features: &[f64]) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check for NaN/Inf
        if self.config.check_nan_inf {
            for (i, &f) in features.iter().enumerate() {
                if f.is_nan() {
                    result.add(
                        &format!("feature_{i}"),
                        ValidationLevel::Error(format!("NaN value at index {i}")),
                    );
                } else if f.is_infinite() {
                    result.add(
                        &format!("feature_{i}"),
                        ValidationLevel::Error(format!("Infinite value at index {i}")),
                    );
                }
            }
        }

        if result.results.is_empty() || result.is_valid() {
            result.add("nan_inf_check", ValidationLevel::Valid);
        }

        result
    }

    /// Validate quote consistency (crossed/locked).
    fn validate_quote_consistency(&self, lob: &LobState, result: &mut ValidationResult) {
        match (lob.best_bid, lob.best_ask) {
            (Some(bid), Some(ask)) => {
                if bid > ask && self.config.check_crossed_quotes {
                    result.add(
                        "crossed_quotes",
                        ValidationLevel::Error(format!(
                            "Crossed quotes: bid {} > ask {}",
                            bid as f64 / 1e9,
                            ask as f64 / 1e9
                        )),
                    );
                } else if bid == ask && self.config.check_locked_quotes {
                    result.add(
                        "locked_quotes",
                        ValidationLevel::Warning(format!(
                            "Locked quotes: bid == ask = {}",
                            bid as f64 / 1e9
                        )),
                    );
                } else {
                    result.add("quote_consistency", ValidationLevel::Valid);
                }
            }
            (None, None) => {
                result.add(
                    "quote_consistency",
                    ValidationLevel::Warning("No quotes available".to_string()),
                );
            }
            (Some(_), None) => {
                result.add(
                    "quote_consistency",
                    ValidationLevel::Warning("No ask quote available".to_string()),
                );
            }
            (None, Some(_)) => {
                result.add(
                    "quote_consistency",
                    ValidationLevel::Warning("No bid quote available".to_string()),
                );
            }
        }
    }

    /// Validate price ordering (decreasing bids, increasing asks).
    fn validate_price_ordering(&self, lob: &LobState, result: &mut ValidationResult) {
        let levels = lob.levels;

        // Check bid prices (should be decreasing)
        let mut bid_ordered = true;
        for i in 1..levels {
            if lob.bid_prices[i] > 0
                && lob.bid_prices[i - 1] > 0
                && lob.bid_prices[i] > lob.bid_prices[i - 1]
            {
                bid_ordered = false;
                break;
            }
        }

        if !bid_ordered {
            result.add(
                "bid_price_ordering",
                ValidationLevel::Error("Bid prices not in decreasing order".to_string()),
            );
        } else {
            result.add("bid_price_ordering", ValidationLevel::Valid);
        }

        // Check ask prices (should be increasing)
        let mut ask_ordered = true;
        for i in 1..levels {
            if lob.ask_prices[i] > 0
                && lob.ask_prices[i - 1] > 0
                && lob.ask_prices[i] < lob.ask_prices[i - 1]
            {
                ask_ordered = false;
                break;
            }
        }

        if !ask_ordered {
            result.add(
                "ask_price_ordering",
                ValidationLevel::Error("Ask prices not in increasing order".to_string()),
            );
        } else {
            result.add("ask_price_ordering", ValidationLevel::Valid);
        }
    }

    /// Validate price ranges.
    fn validate_price_ranges(&self, lob: &LobState, result: &mut ValidationResult) {
        let levels = lob.levels;
        let mut all_valid = true;

        for i in 0..levels {
            let bid_price = lob.bid_prices[i] as f64 / 1e9;
            let ask_price = lob.ask_prices[i] as f64 / 1e9;

            if bid_price > 0.0 {
                if bid_price > self.config.max_price {
                    result.add(
                        &format!("bid_price_{i}"),
                        ValidationLevel::Warning(format!(
                            "Bid price {} exceeds max {}",
                            bid_price, self.config.max_price
                        )),
                    );
                    all_valid = false;
                }
                if bid_price < self.config.min_price {
                    result.add(
                        &format!("bid_price_{i}"),
                        ValidationLevel::Warning(format!(
                            "Bid price {} below min {}",
                            bid_price, self.config.min_price
                        )),
                    );
                    all_valid = false;
                }
            }

            if ask_price > 0.0 {
                if ask_price > self.config.max_price {
                    result.add(
                        &format!("ask_price_{i}"),
                        ValidationLevel::Warning(format!(
                            "Ask price {} exceeds max {}",
                            ask_price, self.config.max_price
                        )),
                    );
                    all_valid = false;
                }
                if ask_price < self.config.min_price {
                    result.add(
                        &format!("ask_price_{i}"),
                        ValidationLevel::Warning(format!(
                            "Ask price {} below min {}",
                            ask_price, self.config.min_price
                        )),
                    );
                    all_valid = false;
                }
            }
        }

        if all_valid {
            result.add("price_ranges", ValidationLevel::Valid);
        }
    }

    /// Validate volume ranges.
    fn validate_volume_ranges(&self, lob: &LobState, result: &mut ValidationResult) {
        let levels = lob.levels;
        let mut all_valid = true;

        for i in 0..levels {
            let bid_size = lob.bid_sizes[i] as u64;
            let ask_size = lob.ask_sizes[i] as u64;

            if bid_size > self.config.max_volume {
                result.add(
                    &format!("bid_size_{i}"),
                    ValidationLevel::Warning(format!(
                        "Bid size {} exceeds max {}",
                        bid_size, self.config.max_volume
                    )),
                );
                all_valid = false;
            }

            if ask_size > self.config.max_volume {
                result.add(
                    &format!("ask_size_{i}"),
                    ValidationLevel::Warning(format!(
                        "Ask size {} exceeds max {}",
                        ask_size, self.config.max_volume
                    )),
                );
                all_valid = false;
            }
        }

        if all_valid {
            result.add("volume_ranges", ValidationLevel::Valid);
        }
    }

    /// Validate spread.
    fn validate_spread(&self, lob: &LobState, result: &mut ValidationResult) {
        if let (Some(bid), Some(ask)) = (lob.best_bid, lob.best_ask) {
            let bid_f64 = bid as f64 / 1e9;
            let ask_f64 = ask as f64 / 1e9;
            let mid = (bid_f64 + ask_f64) / 2.0;

            if mid > 0.0 {
                let spread_bps = ((ask_f64 - bid_f64) / mid) * 10_000.0;

                if spread_bps > self.config.max_spread_bps {
                    result.add(
                        "spread",
                        ValidationLevel::Warning(format!(
                            "Spread {:.2} bps exceeds max {:.2} bps",
                            spread_bps, self.config.max_spread_bps
                        )),
                    );
                } else if spread_bps < 0.0 {
                    result.add(
                        "spread",
                        ValidationLevel::Error(format!("Negative spread: {spread_bps:.2} bps")),
                    );
                } else {
                    result.add("spread", ValidationLevel::Valid);
                }
            }
        }
    }
}

/// Validate a sequence of timestamps for monotonicity.
pub fn validate_timestamps(timestamps: &[u64]) -> ValidationResult {
    let mut result = ValidationResult::new();

    if timestamps.is_empty() {
        result.add(
            "timestamps",
            ValidationLevel::Warning("No timestamps to validate".to_string()),
        );
        return result;
    }

    let mut monotonic = true;
    let mut max_gap_ns = 0u64;

    for i in 1..timestamps.len() {
        if timestamps[i] < timestamps[i - 1] {
            monotonic = false;
            result.add(
                "timestamp_ordering",
                ValidationLevel::Error(format!(
                    "Non-monotonic timestamp at index {}: {} < {}",
                    i,
                    timestamps[i],
                    timestamps[i - 1]
                )),
            );
            break;
        }

        let gap = timestamps[i] - timestamps[i - 1];
        max_gap_ns = max_gap_ns.max(gap);
    }

    if monotonic {
        result.add("timestamp_ordering", ValidationLevel::Valid);
    }

    // Report max gap (in seconds)
    let max_gap_s = max_gap_ns as f64 / 1e9;
    if max_gap_s > 60.0 {
        result.add(
            "timestamp_gaps",
            ValidationLevel::Warning(format!("Max timestamp gap: {max_gap_s:.2} seconds")),
        );
    } else {
        result.add("timestamp_gaps", ValidationLevel::Valid);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_lob() -> LobState {
        let mut lob = LobState::new(10);

        // Valid bid prices (decreasing)
        lob.bid_prices[0] = 100_000_000_000; // $100.00
        lob.bid_prices[1] = 99_990_000_000; // $99.99
        lob.bid_prices[2] = 99_980_000_000; // $99.98
        lob.bid_sizes[0] = 100;
        lob.bid_sizes[1] = 200;
        lob.bid_sizes[2] = 150;

        // Valid ask prices (increasing)
        lob.ask_prices[0] = 100_010_000_000; // $100.01
        lob.ask_prices[1] = 100_020_000_000; // $100.02
        lob.ask_prices[2] = 100_030_000_000; // $100.03
        lob.ask_sizes[0] = 100;
        lob.ask_sizes[1] = 200;
        lob.ask_sizes[2] = 150;

        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);

        lob
    }

    #[test]
    fn test_valid_lob() {
        let validator = FeatureValidator::new();
        let lob = create_valid_lob();
        let result = validator.validate_lob(&lob);

        assert!(result.is_valid());
        assert!(!result.has_errors());
        assert!(!result.has_warnings());
    }

    #[test]
    fn test_crossed_quotes() {
        let validator = FeatureValidator::new();
        let mut lob = create_valid_lob();

        // Create crossed quotes
        lob.best_bid = Some(100_020_000_000); // $100.02
        lob.best_ask = Some(100_010_000_000); // $100.01

        let result = validator.validate_lob(&lob);

        assert!(result.has_errors());
        assert!(!result.is_valid());
    }

    #[test]
    fn test_locked_quotes() {
        let validator = FeatureValidator::new();
        let mut lob = create_valid_lob();

        // Create locked quotes
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_000_000_000);

        let result = validator.validate_lob(&lob);

        assert!(result.has_warnings());
    }

    #[test]
    fn test_validate_features_nan() {
        let validator = FeatureValidator::new();
        let features = vec![1.0, 2.0, f64::NAN, 4.0];

        let result = validator.validate_features(&features);

        assert!(result.has_errors());
    }

    #[test]
    fn test_validate_features_inf() {
        let validator = FeatureValidator::new();
        let features = vec![1.0, f64::INFINITY, 3.0];

        let result = validator.validate_features(&features);

        assert!(result.has_errors());
    }

    #[test]
    fn test_validate_features_valid() {
        let validator = FeatureValidator::new();
        let features = vec![1.0, 2.0, 3.0, 4.0];

        let result = validator.validate_features(&features);

        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_timestamps_monotonic() {
        let timestamps = vec![1_000_000, 2_000_000, 3_000_000, 4_000_000];
        let result = validate_timestamps(&timestamps);

        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_timestamps_non_monotonic() {
        let timestamps = vec![1_000_000, 3_000_000, 2_000_000, 4_000_000];
        let result = validate_timestamps(&timestamps);

        assert!(result.has_errors());
    }

    #[test]
    fn test_validation_result_display() {
        let mut result = ValidationResult::new();
        result.add("test1", ValidationLevel::Valid);
        result.add("test2", ValidationLevel::Warning("minor issue".to_string()));
        result.add("test3", ValidationLevel::Error("major issue".to_string()));

        let display = format!("{result}");
        assert!(display.contains("3"));
        assert!(display.contains("1/3"));
    }

    #[test]
    fn test_price_ordering_invalid_bids() {
        let validator = FeatureValidator::new();
        let mut lob = create_valid_lob();

        // Invalid bid ordering (increasing instead of decreasing)
        lob.bid_prices[1] = 100_010_000_000; // Higher than level 0

        let result = validator.validate_lob(&lob);

        assert!(result.has_errors());
    }

    #[test]
    fn test_price_ordering_invalid_asks() {
        let validator = FeatureValidator::new();
        let mut lob = create_valid_lob();

        // Invalid ask ordering (decreasing instead of increasing)
        lob.ask_prices[1] = 100_000_000_000; // Lower than level 0

        let result = validator.validate_lob(&lob);

        assert!(result.has_errors());
    }
}
