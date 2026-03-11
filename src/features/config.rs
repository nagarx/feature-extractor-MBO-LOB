//! Feature extraction configuration.
//!
//! `FeatureConfig` controls which feature groups are enabled and their parameters.

use super::experimental;

/// Configuration for feature extraction.
///
/// # Feature Count Calculation
///
/// The total number of features is automatically computed based on configuration:
///
/// | Configuration | Feature Count | Breakdown |
/// |--------------|---------------|-----------|
/// | Default (LOB only) | 40 | 10 levels x 4 features |
/// | LOB + derived | 48 | 40 + 8 derived |
/// | LOB + MBO | 76 | 40 + 36 MBO |
/// | LOB + derived + MBO | 84 | 40 + 8 + 36 |
///
/// Use [`FeatureConfig::feature_count()`] to get the computed count.
///
/// # Example
///
/// ```
/// use feature_extractor::FeatureConfig;
///
/// // Default: 40 features (raw LOB only)
/// let config = FeatureConfig::default();
/// assert_eq!(config.feature_count(), 40);
///
/// // With derived features: 48 features
/// let config = FeatureConfig::default().with_derived(true);
/// assert_eq!(config.feature_count(), 48);
///
/// // With MBO features: 76 features
/// let config = FeatureConfig::default().with_mbo(true);
/// assert_eq!(config.feature_count(), 76);
///
/// // Full feature set: 84 features
/// let config = FeatureConfig::default().with_derived(true).with_mbo(true);
/// assert_eq!(config.feature_count(), 84);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureConfig {
    /// Number of LOB levels to extract (typically 10)
    pub lob_levels: usize,

    /// Minimum tick size for price normalization (e.g., 0.01 for US stocks)
    pub tick_size: f64,

    /// Whether to include derived features (8 additional features)
    pub include_derived: bool,

    /// Whether to include MBO features (36 additional features)
    pub include_mbo: bool,

    /// MBO window size (number of messages to track)
    pub mbo_window_size: usize,

    /// Whether to include trading signals (14 additional features, indices 84-97)
    ///
    /// Requires `include_derived` and `include_mbo` to be enabled.
    /// Adds signals: true_ofi, depth_norm_ofi, executed_pressure, etc.
    pub include_signals: bool,

    /// Whether to enable queue position tracking for MBO features.
    ///
    /// When enabled, queue-related features (avg_queue_position, queue_size_ahead)
    /// will contain actual computed values. When disabled, they return 0.0.
    ///
    /// Requires `include_mbo` to be enabled.
    ///
    /// # Performance Note
    ///
    /// Queue tracking adds memory overhead (~O(active_orders)) and
    /// per-event processing cost. Only enable when queue features are needed.
    ///
    /// Default: false (disabled for performance)
    pub include_queue_tracking: bool,

    /// Experimental features configuration.
    ///
    /// Experimental features are opt-in and NOT part of the main schema.
    /// They are designed for analysis and experimentation before promotion.
    ///
    /// Available groups:
    /// - `institutional_v2`: Enhanced institutional detection (8 features)
    /// - `volatility`: Realized volatility features (6 features)
    /// - `seasonality`: Enhanced time-of-day features (4 features)
    ///
    /// # Schema Note
    ///
    /// Experimental features are appended AFTER standard features (index 98+).
    /// Their indices may change without schema version bumps.
    #[serde(default)]
    pub experimental: experimental::ExperimentalConfig,
}

impl FeatureConfig {
    /// Number of derived features when enabled.
    pub const DERIVED_FEATURE_COUNT: usize = 8;

    /// Number of MBO features when enabled.
    pub const MBO_FEATURE_COUNT: usize = 36;

    /// Number of trading signal features when enabled (indices 84-97).
    pub const SIGNAL_FEATURE_COUNT: usize = 14;

    /// Create a new feature configuration with default values.
    pub fn new(lob_levels: usize) -> Self {
        Self {
            lob_levels,
            ..Default::default()
        }
    }

    /// Enable or disable derived features.
    ///
    /// Derived features include: mid-price, spread, spread_bps, total_bid_volume,
    /// total_ask_volume, volume_imbalance, weighted_mid_price, price_impact.
    pub fn with_derived(mut self, enabled: bool) -> Self {
        self.include_derived = enabled;
        self
    }

    /// Enable or disable MBO features.
    ///
    /// MBO features include order flow imbalance, trade intensity, order arrival
    /// rates, cancellation rates, and other microstructure metrics.
    pub fn with_mbo(mut self, enabled: bool) -> Self {
        self.include_mbo = enabled;
        self
    }

    /// Set the MBO window size.
    pub fn with_mbo_window(mut self, window_size: usize) -> Self {
        self.mbo_window_size = window_size;
        self
    }

    /// Set the tick size.
    pub fn with_tick_size(mut self, tick_size: f64) -> Self {
        self.tick_size = tick_size;
        self
    }

    /// Enable or disable trading signals (14 additional features).
    ///
    /// Trading signals include: true_ofi, depth_norm_ofi, executed_pressure,
    /// signed_mp_delta_bps, trade_asymmetry, cancel_asymmetry, fragility_score,
    /// depth_asymmetry, book_valid, time_regime, mbo_ready, dt_seconds,
    /// invalidity_delta, schema_version.
    ///
    /// **Note**: Signals require both `include_derived` and `include_mbo` to be enabled.
    pub fn with_signals(mut self, enabled: bool) -> Self {
        self.include_signals = enabled;
        self
    }

    /// Enable or disable queue position tracking for MBO features.
    ///
    /// When enabled, queue-related features (avg_queue_position, queue_size_ahead)
    /// will contain actual computed values. When disabled, they return 0.0.
    ///
    /// Requires `include_mbo` to be enabled.
    ///
    /// # Performance Note
    ///
    /// Queue tracking adds memory overhead (~O(active_orders)) and
    /// per-event processing cost. Only enable when queue features are needed.
    pub fn with_queue_tracking(mut self, enabled: bool) -> Self {
        self.include_queue_tracking = enabled;
        self
    }

    /// Enable or disable experimental features.
    ///
    /// Experimental features are appended after standard features (index 98+).
    /// Use this for analysis and experimentation before schema promotion.
    ///
    /// # Arguments
    ///
    /// * `config` - Experimental configuration specifying which groups to enable
    pub fn with_experimental(mut self, config: experimental::ExperimentalConfig) -> Self {
        self.experimental = config;
        self
    }

    /// Compute the total number of features based on configuration.
    ///
    /// This is the authoritative source for feature count calculation.
    /// Use this method instead of manually computing feature counts.
    ///
    /// # Returns
    ///
    /// Total number of features:
    /// - Base: `lob_levels * 4` (ask_price, ask_size, bid_price, bid_size per level)
    /// - If `include_derived`: + 8 derived features
    /// - If `include_mbo`: + 36 MBO features
    /// - If `include_signals`: + 14 signals
    /// - If `experimental.enabled`: + experimental features (varies by groups)
    #[inline]
    pub fn feature_count(&self) -> usize {
        let base = self.lob_levels * 4;
        let derived = if self.include_derived {
            Self::DERIVED_FEATURE_COUNT
        } else {
            0
        };
        let mbo = if self.include_mbo {
            Self::MBO_FEATURE_COUNT
        } else {
            0
        };
        let signals = if self.include_signals {
            Self::SIGNAL_FEATURE_COUNT
        } else {
            0
        };
        let experimental = self.experimental.feature_count();
        base + derived + mbo + signals + experimental
    }

    /// Get the number of raw LOB features.
    #[inline]
    pub fn lob_feature_count(&self) -> usize {
        self.lob_levels * 4
    }

    /// Get the base feature count (excluding signals).
    ///
    /// This is the number of features produced by point-in-time extraction
    /// (LOB + derived + MBO), without the streaming signals.
    #[inline]
    pub fn base_feature_count(&self) -> usize {
        let base = self.lob_levels * 4;
        let derived = if self.include_derived {
            Self::DERIVED_FEATURE_COUNT
        } else {
            0
        };
        let mbo = if self.include_mbo {
            Self::MBO_FEATURE_COUNT
        } else {
            0
        };
        base + derived + mbo
    }

    /// Get the signal feature count (0 if signals disabled).
    #[inline]
    pub fn signal_feature_count(&self) -> usize {
        if self.include_signals {
            Self::SIGNAL_FEATURE_COUNT
        } else {
            0
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.lob_levels == 0 {
            return Err("lob_levels must be > 0".to_string());
        }
        if self.lob_levels > 50 {
            return Err("lob_levels must be <= 50 (practical limit)".to_string());
        }
        if self.tick_size <= 0.0 {
            return Err("tick_size must be > 0".to_string());
        }
        if self.include_mbo && self.mbo_window_size == 0 {
            return Err("mbo_window_size must be > 0 when MBO is enabled".to_string());
        }
        if self.include_signals && !self.include_derived {
            return Err("include_signals requires include_derived to be enabled".to_string());
        }
        if self.include_signals && !self.include_mbo {
            return Err("include_signals requires include_mbo to be enabled".to_string());
        }
        if self.include_queue_tracking && !self.include_mbo {
            return Err("include_queue_tracking requires include_mbo to be enabled".to_string());
        }
        // Signal indices are hardcoded for 10 levels:
        // - derived_indices::MID_PRICE = 40 (assumes 10 x 4 = 40 raw features)
        // - mbo_indices::CANCEL_RATE_BID = 50 (assumes MBO starts at 48)
        if self.include_signals && self.lob_levels != 10 {
            return Err(format!(
                "include_signals requires exactly 10 lob_levels (got {}). \
                 Signal feature indices are hardcoded for the 10-level layout. \
                 See signals.rs for details.",
                self.lob_levels
            ));
        }
        self.experimental.validate()?;

        if self.experimental.enabled && !self.include_mbo {
            return Err("experimental features require include_mbo to be enabled".to_string());
        }
        Ok(())
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
            include_signals: false,
            include_queue_tracking: false,
            experimental: experimental::ExperimentalConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count_default() {
        let config = FeatureConfig::default();
        assert_eq!(config.feature_count(), 40);
    }

    #[test]
    fn test_feature_count_no_derived() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            ..Default::default()
        };
        assert_eq!(config.feature_count(), 40);
    }

    #[test]
    fn test_feature_count_5_levels() {
        let config = FeatureConfig::new(5);
        assert_eq!(config.feature_count(), 20);
    }

    #[test]
    fn test_feature_count_with_derived() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        assert_eq!(config.feature_count(), 48);
    }

    #[test]
    fn test_feature_count_with_mbo() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            ..Default::default()
        };
        assert_eq!(config.feature_count(), 84);
    }

    #[test]
    fn test_feature_count_mbo_only() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            ..Default::default()
        };
        assert_eq!(config.feature_count(), 76);
    }

    #[test]
    fn test_feature_count_with_signals() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: true,
            ..Default::default()
        };
        assert_eq!(config.feature_count(), 98);
    }

    #[test]
    fn test_base_feature_count() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: true,
            ..Default::default()
        };
        assert_eq!(config.base_feature_count(), 84);
        assert_eq!(config.signal_feature_count(), 14);
    }

    #[test]
    fn test_builder_pattern() {
        let config = FeatureConfig::new(10)
            .with_derived(true)
            .with_mbo(true)
            .with_signals(true)
            .with_tick_size(0.005)
            .with_mbo_window(500);

        assert!(config.include_derived);
        assert!(config.include_mbo);
        assert!(config.include_signals);
        assert_eq!(config.tick_size, 0.005);
        assert_eq!(config.mbo_window_size, 500);
    }

    #[test]
    fn test_validate_signals_requires_derived() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            include_signals: true,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_signals_requires_mbo() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            include_signals: true,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_signals_requires_10_levels() {
        let config = FeatureConfig {
            lob_levels: 5,
            include_derived: true,
            include_mbo: true,
            include_signals: true,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
