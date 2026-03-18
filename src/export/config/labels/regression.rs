use serde::{Deserialize, Serialize};

use super::{ExportLabelConfig, LabelingStrategy};

/// Return formula for regression labeling targets.
///
/// Determines how forward returns are computed for each horizon.
/// All formulas output values as fractional returns that are then
/// converted to basis points (* 10000) in the export pipeline.
///
/// Reference: Kolm et al. (2023), "Modern Perspectives on Reinforcement
/// Learning in Finance", §3.2.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionReturnType {
    /// TLOB-style smoothed average return (default).
    ///
    /// Formula: `avg(price[t+1:t+h+1]) / avg(price[t-k:t+1]) - 1`
    /// where k = smoothing_window.
    ///
    /// Matches the existing TLOB labeling formula but outputs the continuous
    /// value instead of discretizing. Good baseline, reduces noise via smoothing.
    #[default]
    SmoothedReturn,

    /// Simple point-to-point return at horizon.
    ///
    /// Formula: `(price[t+h] - price[t]) / price[t]`
    ///
    /// No smoothing. Direct measurement of forward return. Most interpretable.
    PointReturn,

    /// Maximum absolute return within horizon (peak opportunity).
    ///
    /// Formula: `max_by_abs(max_return, min_return)` where
    ///   `max_return = max(price[t+1:t+h+1]) / price[t] - 1`
    ///   `min_return = min(price[t+1:t+h+1]) / price[t] - 1`
    ///
    /// Captures the best opportunity in either direction within the horizon.
    PeakReturn,

    /// Mean return over the horizon window.
    ///
    /// Formula: `mean(price[t+1:t+h+1]) / price[t] - 1`
    ///
    /// Smoothed target without backward-looking window.
    MeanReturn,

    /// Dominant return: larger of max/min by magnitude.
    ///
    /// Same as PeakReturn (alias for clarity).
    DominantReturn,
}

/// Fully resolved configuration for regression label export.
///
/// Combines the horizon/smoothing config (from `MultiHorizonConfig`) with
/// the regression-specific return type selection.
#[derive(Debug, Clone)]
pub struct RegressionExportConfig {
    /// Multi-horizon configuration for return computation.
    pub multi_horizon_config: crate::labeling::MultiHorizonConfig,
    /// Which return formula to use for generating regression targets.
    pub return_type: RegressionReturnType,
}

impl ExportLabelConfig {
    /// Convert to regression export configuration.
    ///
    /// Returns `Some(RegressionExportConfig)` when `strategy = "regression"` and
    /// horizons are configured (multi-horizon or single-horizon).
    /// Returns `None` for non-regression strategies.
    pub fn to_regression_config(&self) -> Option<RegressionExportConfig> {
        if !matches!(self.strategy, LabelingStrategy::Regression) {
            return None;
        }
        let horizons = if self.is_multi_horizon() {
            self.horizons.clone()
        } else if self.horizon > 0 {
            vec![self.horizon]
        } else {
            return None;
        };
        let internal_strategy = self.effective_threshold_strategy().to_internal();
        let multi_config = crate::labeling::MultiHorizonConfig::new(
            horizons,
            self.smoothing_window,
            internal_strategy,
        );
        Some(RegressionExportConfig {
            multi_horizon_config: multi_config,
            return_type: self.return_type.unwrap_or_default(),
        })
    }

    /// Validate regression-specific configuration.
    pub(super) fn validate_regression(&self) -> Result<(), String> {
        if !self.is_multi_horizon() {
            return Err(
                "Regression labeling requires multi-horizon (set horizons = [...])".to_string(),
            );
        }
        if self.smoothing_window == 0 {
            return Err("smoothing_window must be > 0 for regression labeling".to_string());
        }
        Ok(())
    }
}
