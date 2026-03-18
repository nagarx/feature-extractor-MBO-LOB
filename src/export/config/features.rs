//! Feature set and experimental feature configuration.
//!
//! These types define WHICH features to extract and export.
//! They are model-agnostic and support flexible feature selection.

use crate::features::FeatureConfig;
use serde::{Deserialize, Serialize};

// ============================================================================
// Feature Set Configuration
// ============================================================================

/// Configuration for which features to extract.
///
/// This is the model-agnostic feature selection layer.
/// The exported features can be used by any downstream model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSetConfig {
    /// Number of LOB levels (typically 10)
    #[serde(default = "default_lob_levels")]
    pub lob_levels: usize,

    /// Include derived features (+8 features: mid-price, spread, etc.)
    #[serde(default)]
    pub include_derived: bool,

    /// Include MBO microstructure features (+36 features)
    #[serde(default)]
    pub include_mbo: bool,

    /// Include trading signals (+14 features: OFI, microprice, etc.)
    ///
    /// **Note**: Requires `include_derived` and `include_mbo` to be enabled.
    #[serde(default)]
    pub include_signals: bool,

    /// MBO window size (number of messages for rolling statistics)
    #[serde(default = "default_mbo_window_size")]
    pub mbo_window_size: usize,

    /// Enable queue position tracking for MBO features.
    ///
    /// When enabled, queue-related features (avg_queue_position, queue_size_ahead)
    /// will contain actual computed values. When disabled, they return 0.0.
    ///
    /// **Note**: Requires `include_mbo` to be enabled. Disabled by default.
    #[serde(default)]
    pub include_queue_tracking: bool,

    /// Experimental feature configuration.
    ///
    /// Experimental features are opt-in and appended after standard features (index 98+).
    /// Groups available: `institutional_v2`, `volatility`, `seasonality`.
    ///
    /// # Example
    ///
    /// ```toml
    /// [features.experimental]
    /// enabled = true
    /// groups = ["institutional_v2", "volatility"]
    /// ```
    #[serde(default)]
    pub experimental: ExperimentalFeatureConfig,
}

/// Configuration for experimental features in TOML.
///
/// Experimental features are designed for analysis and experimentation
/// before promotion to the main schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentalFeatureConfig {
    /// Enable experimental features.
    #[serde(default)]
    pub enabled: bool,

    /// Which experimental groups to include.
    /// Available: "institutional_v2", "volatility", "seasonality"
    /// Empty = all groups when enabled.
    #[serde(default)]
    pub groups: Vec<String>,

    /// Window size for volatility computation (samples).
    #[serde(default = "default_volatility_fast_window")]
    pub volatility_fast_window: usize,

    /// Window size for slow volatility computation.
    #[serde(default = "default_volatility_slow_window")]
    pub volatility_slow_window: usize,

    /// Window size for institutional pattern detection.
    #[serde(default = "default_institutional_window")]
    pub institutional_window: usize,

    /// Threshold percentile for "large" orders.
    #[serde(default = "default_large_order_percentile")]
    pub large_order_percentile: f64,

    /// Round lot size (default: 100 for US equities).
    #[serde(default = "default_round_lot_size")]
    pub round_lot_size: u32,
}

fn default_volatility_fast_window() -> usize {
    50
}

fn default_volatility_slow_window() -> usize {
    500
}

fn default_institutional_window() -> usize {
    200
}

fn default_large_order_percentile() -> f64 {
    90.0
}

fn default_round_lot_size() -> u32 {
    100
}

impl ExperimentalFeatureConfig {
    /// Convert to internal ExperimentalConfig.
    pub fn to_internal(&self) -> crate::features::experimental::ExperimentalConfig {
        crate::features::experimental::ExperimentalConfig {
            enabled: self.enabled,
            groups: self.groups.clone(),
            volatility_fast_window: self.volatility_fast_window,
            volatility_slow_window: self.volatility_slow_window,
            institutional_window: self.institutional_window,
            large_order_percentile: self.large_order_percentile,
            round_lot_size: self.round_lot_size,
        }
    }
}

fn default_lob_levels() -> usize {
    10
}

fn default_mbo_window_size() -> usize {
    1000
}

impl Default for FeatureSetConfig {
    fn default() -> Self {
        Self {
            lob_levels: 10,
            include_derived: false,
            include_mbo: false,
            include_signals: false,
            mbo_window_size: 1000,
            include_queue_tracking: false,
            experimental: ExperimentalFeatureConfig::default(),
        }
    }
}

impl FeatureSetConfig {
    /// Create configuration for full 98-feature mode.
    ///
    /// Enables all standard features: LOB + Derived + MBO + Signals.
    /// Experimental features are NOT enabled by default.
    pub fn full() -> Self {
        Self {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: true,
            mbo_window_size: 1000,
            include_queue_tracking: false, // Disabled by default for performance
            experimental: ExperimentalFeatureConfig::default(),
        }
    }

    /// Create configuration with all features including experimental.
    ///
    /// Total: 98 standard + 18 experimental = 116 features.
    pub fn full_with_experimental() -> Self {
        Self {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: true,
            mbo_window_size: 1000,
            include_queue_tracking: false,
            experimental: ExperimentalFeatureConfig {
                enabled: true,
                groups: vec![
                    "institutional_v2".to_string(),
                    "volatility".to_string(),
                    "seasonality".to_string(),
                ],
                ..Default::default()
            },
        }
    }

    /// Create configuration for 84-feature baseline.
    ///
    /// Enables: LOB + Derived + MBO (no signals).
    pub fn baseline() -> Self {
        Self {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: false,
            mbo_window_size: 1000,
            include_queue_tracking: false,
            experimental: ExperimentalFeatureConfig::default(),
        }
    }

    /// Create configuration for raw LOB only (40 features).
    pub fn raw_lob() -> Self {
        Self::default()
    }

    /// Compute the total number of features.
    ///
    /// Uses named constants from `FeatureConfig` (single source of truth)
    /// to prevent drift between production and export configurations.
    pub fn feature_count(&self) -> usize {
        use crate::features::FeatureConfig;

        let mut count = self.lob_levels * 4; // Raw LOB

        if self.include_derived {
            count += FeatureConfig::DERIVED_FEATURE_COUNT;
        }

        if self.include_mbo {
            count += FeatureConfig::MBO_FEATURE_COUNT;
        }

        if self.include_signals {
            count += FeatureConfig::SIGNAL_FEATURE_COUNT;
        }

        // Experimental features (indices 98+)
        if self.experimental.enabled {
            count += self.experimental.to_internal().feature_count();
        }

        count
    }

    /// Convert to internal FeatureConfig.
    ///
    /// # Arguments
    /// * `tick_size` - The tick size from SymbolConfig (must be propagated from outer config)
    pub fn to_feature_config(&self, tick_size: f64) -> FeatureConfig {
        FeatureConfig {
            lob_levels: self.lob_levels,
            tick_size,
            include_derived: self.include_derived,
            include_mbo: self.include_mbo,
            mbo_window_size: self.mbo_window_size,
            include_signals: self.include_signals,
            include_queue_tracking: self.include_queue_tracking,
            experimental: self.experimental.to_internal(),
        }
    }

    /// Validate the feature configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.lob_levels == 0 {
            return Err("lob_levels must be > 0".to_string());
        }
        if self.lob_levels > 50 {
            return Err("lob_levels must be <= 50 (practical limit)".to_string());
        }

        if self.include_signals && !self.include_derived {
            return Err("include_signals requires include_derived to be enabled".to_string());
        }

        if self.include_signals && !self.include_mbo {
            return Err("include_signals requires include_mbo to be enabled".to_string());
        }

        if self.include_mbo && self.mbo_window_size == 0 {
            return Err("mbo_window_size must be > 0 when MBO is enabled".to_string());
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_set_counts() {
        assert_eq!(FeatureSetConfig::raw_lob().feature_count(), 40);
        assert_eq!(FeatureSetConfig::baseline().feature_count(), 84);
        assert_eq!(FeatureSetConfig::full().feature_count(), 98);
    }
}
