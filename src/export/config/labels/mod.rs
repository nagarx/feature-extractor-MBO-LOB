//! Label generation configuration.
//!
//! Supports TLOB/DeepLOB trend labeling, opportunity detection, and Triple Barrier
//! trade outcome labeling with per-horizon barriers and volatility-adaptive scaling.

mod common;
mod opportunity;
mod regression;
mod strategy;
mod threshold;
mod tlob;
mod triple_barrier;

#[cfg(test)]
mod tests;

pub use common::{ExportConflictPriority, ExportTimeoutStrategy};
pub use regression::{RegressionExportConfig, RegressionReturnType};
pub use strategy::LabelingStrategy;
pub use threshold::ExportThresholdStrategy;

use serde::{Deserialize, Serialize};

fn default_horizon() -> usize {
    50
}

fn default_smoothing_window() -> usize {
    10
}

fn default_threshold() -> f64 {
    0.0008
}

fn default_volatility_floor() -> Option<f64> {
    None
}

fn default_volatility_cap() -> Option<f64> {
    None
}

/// Export-specific label configuration.
///
/// Supports both single-horizon and multi-horizon labeling with configurable
/// threshold strategies.
///
/// # Schema Version: 2.2
///
/// ## Single Horizon Mode (backward compatible)
///
/// ```toml
/// [labels]
/// horizon = 200
/// smoothing_window = 50
/// threshold = 0.0008
/// ```
///
/// ## Multi-Horizon Mode (FI-2010, DeepLOB benchmarks)
///
/// ```toml
/// [labels]
/// horizons = [10, 20, 50, 100, 200]
/// smoothing_window = 10
/// threshold = 0.0008
/// ```
///
/// ## With Explicit Threshold Strategy (Schema 2.2+)
///
/// ```toml
/// [labels]
/// horizons = [10, 20, 50, 100, 200]
/// smoothing_window = 10
///
/// [labels.threshold_strategy]
/// type = "quantile"
/// target_proportion = 0.33
/// window_size = 5000
/// fallback = 0.0008
/// ```
///
/// When `horizons` is non-empty, multi-horizon mode is used.
/// When `threshold_strategy` is provided, it takes precedence over `threshold`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportLabelConfig {
    /// Labeling strategy selection (Schema 2.3+).
    ///
    /// Determines HOW labels are computed from price data:
    /// - `tlob` (default): Smoothed average trend labels
    /// - `opportunity`: Peak return based big-move detection
    /// - `triple_barrier`: Trade outcome based on profit/stop barriers
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "opportunity"  # Use opportunity detection
    /// horizons = [50, 100, 200]
    /// threshold = 0.005
    /// ```
    #[serde(default)]
    pub strategy: LabelingStrategy,

    /// Conflict priority for opportunity labeling (Schema 2.3+).
    ///
    /// When both up and down thresholds are exceeded in the same horizon,
    /// this determines how to resolve the conflict.
    ///
    /// Only used when `strategy = "opportunity"`.
    /// Ignored for other strategies.
    ///
    /// Options: `larger_magnitude`, `up_priority`, `down_priority`, `ambiguous`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflict_priority: Option<ExportConflictPriority>,

    /// Single prediction horizon (number of samples ahead).
    ///
    /// Used when `horizons` is empty. For multi-horizon mode, use `horizons` instead.
    #[serde(default = "default_horizon")]
    pub horizon: usize,

    /// Multiple prediction horizons for multi-horizon labeling.
    ///
    /// When non-empty, enables multi-horizon mode (FI-2010, DeepLOB benchmarks).
    /// Each horizon specifies steps ahead to predict.
    ///
    /// Example: `[10, 20, 50, 100, 200]` generates labels for 5 horizons.
    /// The exported labels will have shape `(N, 5)` instead of `(N,)`.
    ///
    /// **TOML aliases**: Both `horizons` and `max_horizons` are accepted.
    /// - Use `horizons` for TLOB/DeepLOB (prediction horizons)
    /// - Use `max_horizons` for Triple Barrier (max holding periods)
    #[serde(default, skip_serializing_if = "Vec::is_empty", alias = "max_horizons")]
    pub horizons: Vec<usize>,

    /// Smoothing window for noise reduction (k in TLOB formula).
    ///
    /// Shared across all horizons in multi-horizon mode.
    /// Recommended: 5-10 for most applications.
    #[serde(default = "default_smoothing_window")]
    pub smoothing_window: usize,

    /// Classification threshold (θ) as proportion (backward compatible).
    ///
    /// **Deprecated in Schema 2.2+**: Use `threshold_strategy` instead.
    ///
    /// This field is used only when `threshold_strategy` is not provided.
    /// When both are present, `threshold_strategy` takes precedence.
    ///
    /// Common values:
    /// - 0.0002 (2 bps) for HFT
    /// - 0.0008 (8 bps) for short-term
    /// - 0.002 (20 bps) for standard (TLOB/DeepLOB papers)
    #[serde(default = "default_threshold")]
    pub threshold: f64,

    /// Threshold strategy configuration (Schema 2.2+).
    ///
    /// When provided, this takes precedence over the `threshold` field.
    /// Supports three strategies:
    ///
    /// - `fixed`: Constant threshold (equivalent to `threshold` field)
    /// - `rolling_spread`: Adaptive threshold based on bid-ask spread
    /// - `quantile`: Ensures balanced class distribution
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels.threshold_strategy]
    /// type = "quantile"
    /// target_proportion = 0.33
    /// window_size = 5000
    /// fallback = 0.0008
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_strategy: Option<ExportThresholdStrategy>,

    // ========================================================================
    // Triple Barrier Strategy Fields (Schema 2.4+)
    // ========================================================================
    //
    // These fields are ONLY used when `strategy = "triple_barrier"`.
    // They are ignored for other strategies.
    //
    // Research Reference:
    // López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.

    /// Triple Barrier: profit target percentage (upper barrier).
    ///
    /// The trade is labeled as ProfitTarget (class 2) if the price rises
    /// by this percentage before hitting the stop-loss or max horizon.
    ///
    /// **Only used when `strategy = "triple_barrier"`.**
    ///
    /// # Units
    ///
    /// Expressed as a decimal proportion, NOT percentage:
    /// - 0.005 = 0.5% = 50 basis points
    /// - 0.01 = 1.0% = 100 basis points
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "triple_barrier"
    /// profit_target_pct = 0.005  # 0.5% profit target
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profit_target_pct: Option<f64>,

    /// Triple Barrier: stop-loss percentage (lower barrier).
    ///
    /// The trade is labeled as StopLoss (class 0) if the price falls
    /// by this percentage before hitting the profit target or max horizon.
    ///
    /// **Only used when `strategy = "triple_barrier"`.**
    ///
    /// # Asymmetric Barriers
    ///
    /// When `stop_loss_pct != profit_target_pct`, you have asymmetric barriers.
    /// This affects the risk/reward ratio:
    /// - profit_target / stop_loss = risk/reward ratio
    /// - Example: 0.005 / 0.003 = 1.67:1 (need ~38% win rate to break even)
    ///
    /// # Units
    ///
    /// Expressed as a decimal proportion, NOT percentage:
    /// - 0.003 = 0.3% = 30 basis points
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "triple_barrier"
    /// profit_target_pct = 0.005  # 0.5% profit
    /// stop_loss_pct = 0.003      # 0.3% stop (asymmetric)
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_loss_pct: Option<f64>,

    /// Triple Barrier: timeout handling strategy.
    ///
    /// Determines how to label samples when neither the profit target
    /// nor stop-loss is hit within the maximum holding period.
    ///
    /// **Only used when `strategy = "triple_barrier"`.**
    ///
    /// Options:
    /// - `label_as_timeout` (default): Always class 1 (Timeout)
    /// - `use_return_sign`: Class based on return direction at timeout
    /// - `use_fractional_threshold`: Class based on partial barrier hit
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "triple_barrier"
    /// timeout_strategy = "label_as_timeout"
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_strategy: Option<ExportTimeoutStrategy>,

    /// Triple Barrier: minimum holding period before barriers apply.
    ///
    /// If set, barriers are not checked until this many steps have passed
    /// from entry. Useful for avoiding early exits due to market noise.
    ///
    /// **Only used when `strategy = "triple_barrier"`.**
    ///
    /// Must be less than the minimum horizon in `horizons`.
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "triple_barrier"
    /// horizons = [50, 100, 200]
    /// min_holding_period = 5  # Don't check barriers for first 5 steps
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_holding_period: Option<usize>,

    // ========================================================================
    // Per-Horizon Barrier Overrides (Schema 3.2+)
    // ========================================================================
    //
    // These allow setting different barriers for each horizon, which is critical
    // because price movement scales with time (volatility ∝ sqrt(time)).
    //
    // When provided, these arrays must have the same length as `horizons`.
    // If not provided, the global `profit_target_pct` / `stop_loss_pct` is used.

    /// Triple Barrier: per-horizon profit target percentages.
    ///
    /// Allows setting different profit targets for each horizon, which is
    /// important because price movements scale with time (volatility ∝ sqrt(time)).
    ///
    /// **Must have the same length as `horizons` if provided.**
    ///
    /// When not provided, `profit_target_pct` is used for all horizons.
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "triple_barrier"
    /// max_horizons = [50, 100, 200]
    /// # Per-horizon profit targets (calibrated to each horizon's volatility)
    /// profit_targets = [0.0028, 0.0039, 0.0059]  # 28, 39, 59 bps
    /// stop_losses = [0.0019, 0.0026, 0.0040]     # 19, 26, 40 bps
    /// ```
    ///
    /// # Calibration
    ///
    /// Use tools/calibrate_triple_barrier.py to compute data-driven thresholds:
    /// ```bash
    /// python tools/calibrate_triple_barrier.py --data-dir ../data/exports/my_export \
    ///     --horizons 50 100 200 --target-pt-rate 0.25 --target-sl-rate 0.25
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profit_targets: Option<Vec<f64>>,

    /// Triple Barrier: per-horizon stop-loss percentages.
    ///
    /// Allows setting different stop-losses for each horizon.
    ///
    /// **Must have the same length as `horizons` if provided.**
    ///
    /// When not provided, `stop_loss_pct` is used for all horizons.
    ///
    /// # Example
    ///
    /// See `profit_targets` for full example.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_losses: Option<Vec<f64>>,

    // ========================================================================
    // Volatility-Adaptive Barrier Scaling (Schema 3.3+)
    // ========================================================================
    //
    // When enabled, barriers are scaled per-day based on realized volatility:
    //   barrier_day = barrier_base * clamp(vol_day / vol_reference, floor, cap)
    //
    // This ensures balanced class distributions across market regimes.
    //
    // Reference:
    // López de Prado (2018), Ch. 3: Barriers should adapt to volatility.
    // Calibrate with: tools/calibrate_triple_barrier.py --per-day

    /// Enable volatility-adaptive barrier scaling (Schema 3.3+).
    ///
    /// When true, barriers are scaled per-day based on realized volatility:
    ///   `barrier_day = barrier_base * clamp(vol_day / vol_reference, floor, cap)`
    ///
    /// Requires `volatility_reference` to be set.
    /// `profit_targets` / `stop_losses` serve as the base (unscaled) barriers.
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "triple_barrier"
    /// max_horizons = [50, 100, 200]
    /// profit_targets = [0.0020, 0.0028, 0.0042]
    /// stop_losses = [0.0013, 0.0019, 0.0028]
    /// volatility_scaling = true
    /// volatility_reference = 0.00015
    /// volatility_floor = 0.3
    /// volatility_cap = 3.0
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volatility_scaling: Option<bool>,

    /// Reference volatility for barrier scaling.
    ///
    /// This is the realized volatility (std of log returns) that the base
    /// barriers were calibrated against. Typically the median daily volatility
    /// across training days.
    ///
    /// Computed by: `tools/calibrate_triple_barrier.py --per-day`
    ///
    /// Units: dimensionless (per-tick std of log returns).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volatility_reference: Option<f64>,

    /// Minimum scaling factor for volatility scaling.
    ///
    /// Prevents absurdly tight barriers on extremely low-volatility days.
    /// Default: 0.3 (barriers can shrink to 30% of base).
    #[serde(
        default = "default_volatility_floor",
        skip_serializing_if = "Option::is_none"
    )]
    pub volatility_floor: Option<f64>,

    /// Maximum scaling factor for volatility scaling.
    ///
    /// Prevents absurdly wide barriers on extremely high-volatility days
    /// (e.g., earnings, market crashes).
    /// Default: 3.0 (barriers can grow to 300% of base).
    #[serde(
        default = "default_volatility_cap",
        skip_serializing_if = "Option::is_none"
    )]
    pub volatility_cap: Option<f64>,

    // ========================================================================
    // Regression Strategy Fields
    // ========================================================================
    //
    // These fields are ONLY used when `strategy = "regression"`.
    // They are ignored for other strategies.

    /// Regression return type: which return formula to use for regression targets.
    ///
    /// **Only used when `strategy = "regression"`.**
    ///
    /// Options:
    /// - `smoothed_return` (default): TLOB-style smoothed average return
    /// - `point_return`: Simple point-to-point return at horizon
    /// - `peak_return`: Maximum absolute return within horizon (dominant direction)
    /// - `mean_return`: Mean return over the horizon window
    /// - `dominant_return`: Larger of max/min by magnitude
    ///
    /// All return types output values in basis points (bps).
    ///
    /// # Example
    ///
    /// ```toml
    /// [labels]
    /// strategy = "regression"
    /// horizons = [10, 60, 300]
    /// return_type = "point_return"
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_type: Option<RegressionReturnType>,
}

impl Default for ExportLabelConfig {
    fn default() -> Self {
        Self {
            strategy: LabelingStrategy::Tlob,
            conflict_priority: None,
            horizon: 50,
            horizons: Vec::new(),
            smoothing_window: 10,
            threshold: 0.0008,
            threshold_strategy: None, // Uses threshold field as Fixed fallback
            // Triple Barrier fields (None = not a Triple Barrier config)
            profit_target_pct: None,
            stop_loss_pct: None,
            timeout_strategy: None,
            min_holding_period: None,
            // Per-horizon overrides (Schema 3.2+)
            profit_targets: None,
            stop_losses: None,
            // Volatility-adaptive scaling (Schema 3.3+)
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
            // Regression fields
            return_type: None,
        }
    }
}

impl ExportLabelConfig {
    /// Check if multi-horizon mode is enabled.
    ///
    /// Returns `true` if `horizons` array is non-empty.
    #[inline]
    pub fn is_multi_horizon(&self) -> bool {
        !self.horizons.is_empty()
    }

    /// Get the effective horizons.
    ///
    /// Returns `horizons` if non-empty, otherwise `[horizon]` as single-element vec.
    pub fn effective_horizons(&self) -> Vec<usize> {
        if self.is_multi_horizon() {
            self.horizons.clone()
        } else {
            vec![self.horizon]
        }
    }

    /// Get the maximum horizon (for buffer sizing).
    pub fn max_horizon(&self) -> usize {
        if self.is_multi_horizon() {
            *self.horizons.iter().max().unwrap_or(&self.horizon)
        } else {
            self.horizon
        }
    }

    /// Validate the label configuration.
    ///
    /// Validates:
    /// - At least one horizon is configured
    /// - All horizons are > 0
    /// - Smoothing window is > 0 and <= minimum horizon (for TLOB/DeepLOB only)
    /// - Threshold/threshold_strategy is in valid range
    /// - Strategy-specific parameters are present
    pub fn validate(&self) -> Result<(), String> {
        self.validate_common()?;

        match &self.strategy {
            LabelingStrategy::Tlob => self.validate_tlob(),
            LabelingStrategy::Opportunity => self.validate_opportunity(),
            LabelingStrategy::TripleBarrier => self.validate_triple_barrier(),
            LabelingStrategy::Regression => self.validate_regression(),
        }
    }

    /// Common validation shared across all labeling strategies.
    fn validate_common(&self) -> Result<(), String> {
        if self.is_multi_horizon() {
            if self.horizons.contains(&0) {
                return Err("All horizons must be > 0".to_string());
            }
        } else if self.horizon == 0 {
            return Err("horizon must be > 0".to_string());
        }
        Ok(())
    }

    /// Get a human-readable description of the label configuration.
    pub fn description(&self) -> String {
        let horizons_str = if self.is_multi_horizon() {
            format!("horizons={:?}", self.horizons)
        } else {
            format!("horizon={}", self.horizon)
        };

        match &self.strategy {
            LabelingStrategy::Tlob => {
                let threshold_str = self.effective_threshold_strategy().description();
                format!(
                    "TLOB: {}, smoothing_window={}, {}",
                    horizons_str, self.smoothing_window, threshold_str
                )
            }
            LabelingStrategy::Opportunity => {
                let priority_str = self
                    .conflict_priority
                    .as_ref()
                    .map(|p| format!("{:?}", p))
                    .unwrap_or_else(|| "LargerMagnitude".to_string());
                format!(
                    "Opportunity: {}, threshold={:.4}%, conflict_priority={}",
                    horizons_str,
                    self.threshold * 100.0,
                    priority_str
                )
            }
            LabelingStrategy::TripleBarrier => {
                let profit_str = self
                    .profit_target_pct
                    .map(|p| format!("{:.4}%", p * 100.0))
                    .unwrap_or_else(|| "NOT SET".to_string());
                let stop_str = self
                    .stop_loss_pct
                    .map(|s| format!("{:.4}%", s * 100.0))
                    .unwrap_or_else(|| "NOT SET".to_string());
                let timeout_str = self
                    .timeout_strategy
                    .map(|t| format!("{:?}", t))
                    .unwrap_or_else(|| "LabelAsTimeout".to_string());
                let rr_ratio = match (self.profit_target_pct, self.stop_loss_pct) {
                    (Some(p), Some(s)) if s > 0.0 => format!("{:.2}:1", p / s),
                    _ => "N/A".to_string(),
                };
                format!(
                    "TripleBarrier: {}, profit_target={}, stop_loss={}, R:R={}, timeout={}",
                    horizons_str, profit_str, stop_str, rr_ratio, timeout_str
                )
            }
            LabelingStrategy::Regression => {
                format!(
                    "Regression: {}, smoothing_window={}, continuous bps returns (float64)",
                    horizons_str, self.smoothing_window
                )
            }
        }
    }
}
