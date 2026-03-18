//! Experimental Feature Modules
//!
//! This module contains **experimental** features that are:
//! - Opt-in via `include_experimental` in `FeatureConfig`
//! - NOT part of the main schema (indices 0-97)
//! - Subject to change without schema version bumps
//! - Designed for analysis and experimentation before promotion
//!
//! # Architecture
//!
//! Experimental features are organized by purpose:
//! - `institutional_v2`: Enhanced institutional/whale detection
//! - `volatility`: Realized volatility and regime features
//! - `seasonality`: Enhanced time-of-day features
//!
//! # Usage
//!
//! Enable experimental features via TOML config:
//! ```toml
//! [features]
//! include_experimental = true
//! experimental_groups = ["institutional_v2", "volatility"]
//! ```
//!
//! Or programmatically:
//! ```ignore
//! let config = FeatureConfig::default()
//!     .with_experimental(true)
//!     .with_experimental_groups(vec!["institutional_v2", "volatility"]);
//! ```
//!
//! # Feature Index Layout (Experimental)
//!
//! When experimental features are enabled, they are appended AFTER
//! the standard 98 features:
//!
//! | Index Range | Group | Features |
//! |-------------|-------|----------|
//! | 98-105 | institutional_v2 | 8 features |
//! | 106-111 | volatility | 6 features |
//! | 112-115 | seasonality | 4 features |
//!
//! # Promotion Path
//!
//! Features that prove valuable in analysis can be promoted to the
//! main feature set. This requires:
//! 1. Validation via Python analyzer
//! 2. Schema version bump
//! 3. Documentation update
//! 4. Python contract update

pub mod institutional_v2;
pub mod kolm_of;
pub mod mlofi;
pub mod seasonality;
pub mod volatility;

use std::collections::HashSet;

/// Available experimental feature groups.
pub const AVAILABLE_GROUPS: &[&str] = &["institutional_v2", "volatility", "seasonality", "mlofi", "kolm_of"];

/// Configuration for experimental features.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ExperimentalConfig {
    /// Enable experimental features.
    pub enabled: bool,

    /// Which experimental groups to include.
    /// Empty means all groups if `enabled` is true.
    pub groups: Vec<String>,

    /// Window size for volatility computation (samples).
    /// Default: 50 (fast), 500 (slow) - configured per-window.
    pub volatility_fast_window: usize,
    pub volatility_slow_window: usize,

    /// Window size for institutional pattern detection.
    pub institutional_window: usize,

    /// Threshold for "large" orders (percentile).
    pub large_order_percentile: f64,

    /// Round lot size (default: 100 for US equities).
    pub round_lot_size: u32,
}

impl ExperimentalConfig {
    /// Default configuration with sensible values.
    pub fn new() -> Self {
        Self {
            enabled: false,
            groups: vec![],
            volatility_fast_window: 50,
            volatility_slow_window: 500,
            institutional_window: 200,
            large_order_percentile: 90.0,
            round_lot_size: 100,
        }
    }

    /// Enable experimental features with all groups.
    pub fn with_all_groups(mut self) -> Self {
        self.enabled = true;
        self.groups = AVAILABLE_GROUPS.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Enable specific groups.
    pub fn with_groups(mut self, groups: Vec<String>) -> Self {
        self.enabled = !groups.is_empty();
        self.groups = groups;
        self
    }

    /// Get the set of enabled group names.
    pub fn enabled_groups(&self) -> HashSet<&str> {
        if !self.enabled {
            return HashSet::new();
        }
        if self.groups.is_empty() {
            // All groups if enabled but none specified
            AVAILABLE_GROUPS.iter().copied().collect()
        } else {
            self.groups.iter().map(|s| s.as_str()).collect()
        }
    }

    /// Compute total experimental feature count.
    pub fn feature_count(&self) -> usize {
        if !self.enabled {
            return 0;
        }

        let groups = self.enabled_groups();
        let mut count = 0;

        if groups.contains("institutional_v2") {
            count += institutional_v2::FEATURE_COUNT;
        }
        if groups.contains("volatility") {
            count += volatility::FEATURE_COUNT;
        }
        if groups.contains("seasonality") {
            count += seasonality::FEATURE_COUNT;
        }
        if groups.contains("mlofi") {
            count += mlofi::FEATURE_COUNT;
        }
        if groups.contains("kolm_of") {
            count += kolm_of::FEATURE_COUNT;
        }

        count
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }

        // Validate group names
        for group in &self.groups {
            if !AVAILABLE_GROUPS.contains(&group.as_str()) {
                return Err(format!(
                    "Unknown experimental group: '{}'. Available: {:?}",
                    group, AVAILABLE_GROUPS
                ));
            }
        }

        // Validate window sizes
        if self.volatility_fast_window < 10 {
            return Err("volatility_fast_window must be >= 10".to_string());
        }
        if self.volatility_slow_window < self.volatility_fast_window {
            return Err("volatility_slow_window must be >= volatility_fast_window".to_string());
        }
        if self.institutional_window < 20 {
            return Err("institutional_window must be >= 20".to_string());
        }

        Ok(())
    }
}

/// Main experimental feature extractor.
///
/// Coordinates extraction across all enabled experimental groups.
pub struct ExperimentalExtractor {
    config: ExperimentalConfig,
    institutional: Option<institutional_v2::InstitutionalDetectorV2>,
    volatility: Option<volatility::VolatilityComputer>,
    seasonality: Option<seasonality::SeasonalityComputer>,
    mlofi: Option<mlofi::MlofiComputer>,
    kolm_of: Option<kolm_of::KolmOfComputer>,
}

impl ExperimentalExtractor {
    /// Create a new experimental extractor.
    pub fn new(config: ExperimentalConfig) -> Self {
        let groups = config.enabled_groups();

        let institutional = if groups.contains("institutional_v2") {
            Some(institutional_v2::InstitutionalDetectorV2::new(
                config.institutional_window,
                config.large_order_percentile,
                config.round_lot_size,
            ))
        } else {
            None
        };

        let volatility = if groups.contains("volatility") {
            Some(volatility::VolatilityComputer::new(
                config.volatility_fast_window,
                config.volatility_slow_window,
            ))
        } else {
            None
        };

        let seasonality = if groups.contains("seasonality") {
            Some(seasonality::SeasonalityComputer::new())
        } else {
            None
        };

        let mlofi = if groups.contains("mlofi") {
            Some(mlofi::MlofiComputer::new())
        } else {
            None
        };

        let kolm_of = if groups.contains("kolm_of") {
            Some(kolm_of::KolmOfComputer::new())
        } else {
            None
        };

        Self {
            config,
            institutional,
            volatility,
            seasonality,
            mlofi,
            kolm_of,
        }
    }

    /// Process an MBO event.
    #[inline]
    pub fn process_event(&mut self, event: &super::mbo_features::MboEvent) {
        if let Some(ref mut inst) = self.institutional {
            inst.process_event(event);
        }
    }

    /// Update with a new mid-price sample (for volatility).
    #[inline]
    pub fn update_price(&mut self, mid_price: f64, timestamp_ns: u64) {
        if let Some(ref mut vol) = self.volatility {
            vol.update(mid_price, timestamp_ns);
        }
    }

    /// Update with a new LOB state (for MLOFI and Kolm OF).
    ///
    /// Must be called on every LOB state transition, not just at sample time.
    #[inline]
    pub fn update_lob(&mut self, lob: &mbo_lob_reconstructor::LobState) {
        if let Some(ref mut m) = self.mlofi {
            m.update(lob);
        }
        if let Some(ref mut k) = self.kolm_of {
            k.update(lob);
        }
    }

    /// Extract all enabled experimental features.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    /// * `output` - Output buffer to append features to
    pub fn extract_into(&mut self, timestamp_ns: u64, output: &mut Vec<f64>) {
        if !self.config.enabled {
            return;
        }

        // Extract in deterministic order
        if let Some(ref mut inst) = self.institutional {
            inst.extract_into(output);
        }

        if let Some(ref vol) = self.volatility {
            vol.extract_into(output);
        }

        if let Some(ref seas) = self.seasonality {
            seas.extract_into(timestamp_ns, output);
        }

        if let Some(ref m) = self.mlofi {
            m.extract_into(output);
        }

        if let Some(ref k) = self.kolm_of {
            k.extract_into(output);
        }
    }

    /// Extract all enabled experimental features AND reset continuous accumulators.
    ///
    /// This is the production method called at each sample point. It extracts
    /// features in the same order as `extract_into()`, but additionally resets
    /// MLOFI and Kolm OF accumulators so each sample captures only the
    /// interval's order flow (not cumulative since day start).
    ///
    /// Groups that are stateless or use rolling windows (institutional, volatility,
    /// seasonality) are extracted normally — they don't need sample-and-reset.
    pub fn extract_and_reset_into(&mut self, timestamp_ns: u64, output: &mut Vec<f64>) {
        if !self.config.enabled {
            return;
        }

        // Stateless / rolling-window groups: extract normally
        if let Some(ref mut inst) = self.institutional {
            inst.extract_into(output);
        }

        if let Some(ref vol) = self.volatility {
            vol.extract_into(output);
        }

        if let Some(ref seas) = self.seasonality {
            seas.extract_into(timestamp_ns, output);
        }

        // Continuous accumulators: extract AND reset for interval-scoped values
        if let Some(ref mut m) = self.mlofi {
            m.sample_and_reset(output);
        }

        if let Some(ref mut k) = self.kolm_of {
            k.sample_and_reset(output);
        }
    }

    /// Get the total feature count.
    pub fn feature_count(&self) -> usize {
        self.config.feature_count()
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        if let Some(ref mut inst) = self.institutional {
            inst.reset();
        }
        if let Some(ref mut vol) = self.volatility {
            vol.reset();
        }
        if let Some(ref mut m) = self.mlofi {
            m.reset();
        }
        if let Some(ref mut k) = self.kolm_of {
            k.reset();
        }
    }

    /// Check if warmed up (enough data for valid features).
    pub fn is_warm(&self) -> bool {
        // All enabled components must be warm
        if let Some(ref inst) = self.institutional {
            if !inst.is_warm() {
                return false;
            }
        }
        if let Some(ref vol) = self.volatility {
            if !vol.is_warm() {
                return false;
            }
        }
        if let Some(ref m) = self.mlofi {
            if !m.is_warm() {
                return false;
            }
        }
        if let Some(ref k) = self.kolm_of {
            if !k.is_warm() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExperimentalConfig::new();
        assert!(!config.enabled);
        assert_eq!(config.feature_count(), 0);
    }

    #[test]
    fn test_all_groups() {
        let config = ExperimentalConfig::new().with_all_groups();
        assert!(config.enabled);

        let expected = institutional_v2::FEATURE_COUNT
            + volatility::FEATURE_COUNT
            + seasonality::FEATURE_COUNT
            + mlofi::FEATURE_COUNT
            + kolm_of::FEATURE_COUNT;
        assert_eq!(config.feature_count(), expected);
    }

    #[test]
    fn test_specific_groups() {
        let config = ExperimentalConfig::new().with_groups(vec!["volatility".to_string()]);

        assert!(config.enabled);
        assert_eq!(config.feature_count(), volatility::FEATURE_COUNT);
    }

    #[test]
    fn test_validation() {
        let mut config = ExperimentalConfig::new();
        config.enabled = true;
        config.groups = vec!["invalid_group".to_string()];

        assert!(config.validate().is_err());
    }
}
