//! Label generation configuration.
//!
//! Supports TLOB/DeepLOB trend labeling, opportunity detection, and Triple Barrier
//! trade outcome labeling with per-horizon barriers and volatility-adaptive scaling.

use serde::{Deserialize, Serialize};

use crate::labeling::LabelConfig;

// ============================================================================
// Labeling Strategy Selection
// ============================================================================

/// Labeling strategy selection for export configuration.
///
/// This enum determines HOW labels are computed from price data.
/// Different strategies are suited for different trading objectives.
///
/// # Strategies
///
/// | Strategy | Use Case | Label Meaning |
/// |----------|----------|---------------|
/// | `tlob` (default) | Trend prediction | Up/Down/Stable based on smoothed avg return |
/// | `opportunity` | Big move detection | BigUp/BigDown/NoOpportunity based on peak return |
/// | `triple_barrier` | Trade outcomes | ProfitTake/StopLoss/Timeout based on barriers |
///
/// # TOML Configuration Examples
///
/// ## TLOB/DeepLOB (default, backward compatible)
/// ```toml
/// [labels]
/// # No strategy field = defaults to TLOB
/// horizon = 50
/// smoothing_window = 10
/// threshold = 0.0008
/// ```
///
/// ## Opportunity Detection (big moves)
/// ```toml
/// [labels]
/// strategy = "opportunity"
/// horizons = [50, 100, 200]
/// threshold = 0.005  # 50 bps threshold
/// conflict_priority = "larger_magnitude"
/// ```
///
/// ## Triple Barrier (trade outcomes)
/// ```toml
/// [labels]
/// strategy = "triple_barrier"
/// horizon = 100
/// profit_target = 0.005   # 50 bps profit target
/// stop_loss = 0.003       # 30 bps stop loss
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelingStrategy {
    /// TLOB/DeepLOB smoothed average labeling (default).
    ///
    /// Computes label based on smoothed average mid-price change.
    /// Best for trend prediction with relatively balanced classes.
    #[default]
    Tlob,

    /// Opportunity detection based on peak returns.
    ///
    /// Labels based on max/min return within horizon.
    /// Best for detecting "big moves" with expected class imbalance.
    Opportunity,

    /// Triple barrier labeling for trade outcomes.
    ///
    /// Labels based on which barrier is hit first: profit, stop-loss, or timeout.
    /// Best for directly modeling trading profitability.
    TripleBarrier,
}

impl LabelingStrategy {
    /// Check if this strategy requires opportunity-specific config fields.
    pub fn is_opportunity(&self) -> bool {
        matches!(self, Self::Opportunity)
    }

    /// Check if this strategy requires triple-barrier-specific config fields.
    pub fn is_triple_barrier(&self) -> bool {
        matches!(self, Self::TripleBarrier)
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Tlob => "TLOB/DeepLOB (smoothed average trend)",
            Self::Opportunity => "Opportunity (peak return detection)",
            Self::TripleBarrier => "Triple Barrier (trade outcomes)",
        }
    }
}

// ============================================================================
// Conflict Priority for Opportunity Labeling
// ============================================================================

/// Priority strategy when both big up and big down occur in the same horizon.
///
/// During volatile periods, both max_return > threshold AND min_return < -threshold
/// may occur. This strategy determines how to resolve the conflict.
///
/// Used only when `strategy = "opportunity"`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportConflictPriority {
    /// Label based on which move has larger absolute magnitude (default).
    #[default]
    LargerMagnitude,

    /// Always prioritize BigUp when both trigger.
    UpPriority,

    /// Always prioritize BigDown when both trigger.
    DownPriority,

    /// Label as NoOpportunity when both trigger.
    Ambiguous,
}

impl ExportConflictPriority {
    /// Convert to internal labeling ConflictPriority.
    pub fn to_internal(&self) -> crate::labeling::ConflictPriority {
        match self {
            Self::LargerMagnitude => crate::labeling::ConflictPriority::LargerMagnitude,
            Self::UpPriority => crate::labeling::ConflictPriority::UpPriority,
            Self::DownPriority => crate::labeling::ConflictPriority::DownPriority,
            Self::Ambiguous => crate::labeling::ConflictPriority::Ambiguous,
        }
    }
}

// ============================================================================
// Timeout Strategy for Triple Barrier Labeling
// ============================================================================

/// Timeout strategy for Triple Barrier labeling (export-layer enum).
///
/// When the vertical barrier (max_horizon) is hit without touching either
/// the profit target or stop-loss barrier, this strategy determines how
/// to assign the final label.
///
/// Used only when `strategy = "triple_barrier"`.
///
/// # Research Reference
///
/// López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.
/// The Triple Barrier Method.
///
/// # TOML Example
///
/// ```toml
/// [labels]
/// strategy = "triple_barrier"
/// profit_target_pct = 0.005
/// stop_loss_pct = 0.003
/// horizons = [50, 100, 200]
/// timeout_strategy = "label_as_timeout"  # or "use_return_sign"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportTimeoutStrategy {
    /// Always label timeout as class 1 (Timeout / no clear outcome).
    ///
    /// This is the standard Triple Barrier approach from López de Prado.
    /// The timeout class represents "don't trade" signals - situations
    /// where neither barrier was hit within the holding period.
    #[default]
    LabelAsTimeout,

    /// Label based on the sign of the return at timeout.
    ///
    /// If return > 0 at timeout → ProfitTarget (class 2)
    /// If return < 0 at timeout → StopLoss (class 0)
    /// If return = 0 at timeout → Timeout (class 1)
    ///
    /// Use when you want no "inconclusive" labels - every sample
    /// gets a directional label based on where price ended up.
    UseReturnSign,

    /// Label based on whether return exceeds a fraction of the barriers.
    ///
    /// Uses 50% of the barrier thresholds to determine label at timeout:
    /// - return > 0.5 * profit_target → ProfitTarget
    /// - return < -0.5 * stop_loss → StopLoss
    /// - otherwise → Timeout
    ///
    /// A middle ground between strict timeout and return sign.
    UseFractionalThreshold,
}

impl ExportTimeoutStrategy {
    /// Convert to internal labeling TimeoutStrategy.
    pub fn to_internal(&self) -> crate::labeling::TimeoutStrategy {
        match self {
            Self::LabelAsTimeout => crate::labeling::TimeoutStrategy::LabelAsTimeout,
            Self::UseReturnSign => crate::labeling::TimeoutStrategy::UseReturnSign,
            Self::UseFractionalThreshold => {
                crate::labeling::TimeoutStrategy::UseFractionalThreshold
            }
        }
    }
}

// ============================================================================
// Threshold Strategy
// ============================================================================

/// Threshold strategy selection for TOML configuration.
///
/// This enum configures how classification thresholds are determined.
/// Each strategy has different tradeoffs:
///
/// - **Fixed**: Constant threshold, reproducible, good for benchmarks
/// - **RollingSpread**: Adapts to market conditions via bid-ask spread
/// - **Quantile**: Ensures balanced class distribution regardless of volatility
///
/// # Research Reference
///
/// From TLOB paper Section 4.1.3:
/// > "We argue that relating θ to transaction costs can better align trend
/// >  predictions with profitability."
///
/// # TOML Configuration Examples
///
/// ## Fixed Threshold (default, backward compatible)
/// ```toml
/// [labels]
/// threshold = 0.0008  # Simple fixed threshold (8 bps)
/// ```
///
/// ## Explicit Fixed Strategy
/// ```toml
/// [labels.threshold_strategy]
/// type = "fixed"
/// value = 0.0008
/// ```
///
/// ## Rolling Spread Strategy (adaptive to market conditions)
/// ```toml
/// [labels.threshold_strategy]
/// type = "rolling_spread"
/// window_size = 1000      # Rolling window for spread averaging
/// multiplier = 1.5        # threshold = multiplier × avg_spread
/// fallback = 0.0008       # Used when insufficient data
/// ```
///
/// ## Quantile Strategy (balanced classes, recommended for training)
/// ```toml
/// [labels.threshold_strategy]
/// type = "quantile"
/// target_proportion = 0.33  # ~33% Up, ~33% Down, ~34% Stable
/// window_size = 5000        # Window for quantile computation
/// fallback = 0.0008         # Used when insufficient data
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExportThresholdStrategy {
    /// Fixed percentage threshold.
    ///
    /// Simple and reproducible. Use for benchmark comparison.
    /// Value is the classification threshold as a proportion (e.g., 0.0008 = 8 bps).
    Fixed {
        /// Classification threshold as proportion (e.g., 0.0008 = 8 basis points)
        value: f64,
    },

    /// Rolling spread-based threshold.
    ///
    /// Threshold = multiplier × rolling_average_spread / mid_price
    /// Adapts to market conditions automatically.
    RollingSpread {
        /// Number of samples for rolling spread computation
        window_size: usize,
        /// Multiplier for the average spread (typically 0.5-2.0)
        multiplier: f64,
        /// Fallback threshold when insufficient data
        fallback: f64,
    },

    /// Quantile-based threshold for balanced class distribution.
    ///
    /// Computes threshold from rolling quantile of absolute price changes.
    /// Ensures roughly equal Up/Down proportions regardless of market volatility.
    ///
    /// With target_proportion = 0.33:
    /// - ~33% Up labels
    /// - ~33% Down labels  
    /// - ~34% Stable labels
    Quantile {
        /// Target proportion for up/down classes (0.0 to 0.5)
        /// Example: 0.33 means ~33% Up, ~33% Down, ~34% Stable
        target_proportion: f64,
        /// Number of samples for quantile computation
        window_size: usize,
        /// Fallback threshold when insufficient data
        fallback: f64,
    },

    /// TLOB paper dynamic threshold (global computation).
    ///
    /// Computes threshold from the **entire** price series as:
    /// `alpha = mean(|percentage_change|) / divisor`
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
    /// # TOML Configuration
    ///
    /// ```toml
    /// [labels.threshold_strategy]
    /// type = "tlob_dynamic"
    /// fallback = 0.0008         # Used when insufficient data
    /// divisor = 2.0             # Default per TLOB paper
    /// ```
    TlobDynamic {
        /// Fallback threshold when insufficient data
        fallback: f64,
        /// Divisor for mean absolute change (default: 2.0 per TLOB paper)
        /// alpha = mean(|percentage_change|) / divisor
        #[serde(default = "default_tlob_divisor")]
        divisor: f64,
    },
}

fn default_tlob_divisor() -> f64 {
    2.0
}

impl Default for ExportThresholdStrategy {
    fn default() -> Self {
        // Default: Fixed at 8 bps (common HFT threshold)
        Self::Fixed { value: 0.0008 }
    }
}

impl ExportThresholdStrategy {
    /// Create a fixed threshold strategy.
    pub fn fixed(value: f64) -> Self {
        Self::Fixed { value }
    }

    /// Create a rolling spread strategy.
    ///
    /// # Arguments
    /// * `window_size` - Rolling window size for spread averaging
    /// * `multiplier` - Multiplier for average spread (e.g., 1.5 = 1.5x spread)
    /// * `fallback` - Fallback threshold when insufficient data
    pub fn rolling_spread(window_size: usize, multiplier: f64, fallback: f64) -> Self {
        Self::RollingSpread {
            window_size,
            multiplier,
            fallback,
        }
    }

    /// Create a quantile-based strategy for balanced classes.
    ///
    /// # Arguments
    /// * `target_proportion` - Target proportion for Up/Down classes (0.0-0.5)
    /// * `window_size` - Window size for quantile computation
    /// * `fallback` - Fallback threshold when insufficient data
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportThresholdStrategy;
    ///
    /// // Target 33% Up, 33% Down, 34% Stable
    /// let strategy = ExportThresholdStrategy::quantile(0.33, 5000, 0.0008);
    /// ```
    pub fn quantile(target_proportion: f64, window_size: usize, fallback: f64) -> Self {
        Self::Quantile {
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
    /// * `fallback` - Fallback threshold when insufficient data
    /// * `divisor` - Divisor for mean (default: 2.0 per TLOB paper)
    ///
    /// # Research Reference
    ///
    /// TLOB repository: `utils/utils_data.py::labeling()`
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportThresholdStrategy;
    ///
    /// // Match official TLOB repository labeling
    /// let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
    /// ```
    pub fn tlob_dynamic(fallback: f64, divisor: f64) -> Self {
        Self::TlobDynamic { fallback, divisor }
    }

    /// Create a TLOB paper dynamic threshold with default divisor (2.0).
    ///
    /// # Arguments
    /// * `fallback` - Fallback threshold when insufficient data
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportThresholdStrategy;
    ///
    /// // Match official TLOB repository labeling with default divisor
    /// let strategy = ExportThresholdStrategy::tlob_dynamic_default(0.0008);
    /// ```
    pub fn tlob_dynamic_default(fallback: f64) -> Self {
        Self::tlob_dynamic(fallback, 2.0)
    }

    /// Convert to internal ThresholdStrategy for label generation.
    pub fn to_internal(&self) -> crate::labeling::ThresholdStrategy {
        match self {
            Self::Fixed { value } => crate::labeling::ThresholdStrategy::Fixed(*value),
            Self::RollingSpread {
                window_size,
                multiplier,
                fallback,
            } => crate::labeling::ThresholdStrategy::RollingSpread {
                window_size: *window_size,
                multiplier: *multiplier,
                fallback: *fallback,
            },
            Self::Quantile {
                target_proportion,
                window_size,
                fallback,
            } => crate::labeling::ThresholdStrategy::Quantile {
                target_proportion: *target_proportion,
                window_size: *window_size,
                fallback: *fallback,
            },
            Self::TlobDynamic { fallback, divisor } => {
                crate::labeling::ThresholdStrategy::TlobDynamic {
                    fallback: *fallback,
                    divisor: *divisor,
                }
            }
        }
    }

    /// Validate the strategy parameters.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Fixed { value } => {
                if *value <= 0.0 {
                    return Err("Fixed threshold value must be > 0".to_string());
                }
                if *value > 0.1 {
                    return Err("Fixed threshold value seems too large (> 10%)".to_string());
                }
            }
            Self::RollingSpread {
                window_size,
                multiplier,
                fallback,
            } => {
                if *window_size == 0 {
                    return Err("RollingSpread window_size must be > 0".to_string());
                }
                if *multiplier <= 0.0 {
                    return Err("RollingSpread multiplier must be > 0".to_string());
                }
                if *fallback <= 0.0 {
                    return Err("RollingSpread fallback must be > 0".to_string());
                }
            }
            Self::Quantile {
                target_proportion,
                window_size,
                fallback,
            } => {
                if *target_proportion <= 0.0 || *target_proportion > 0.5 {
                    return Err(
                        "Quantile target_proportion must be in range (0.0, 0.5]".to_string()
                    );
                }
                if *window_size == 0 {
                    return Err("Quantile window_size must be > 0".to_string());
                }
                if *fallback <= 0.0 {
                    return Err("Quantile fallback must be > 0".to_string());
                }
            }
            Self::TlobDynamic { fallback, divisor } => {
                if *fallback <= 0.0 {
                    return Err("TlobDynamic fallback must be > 0".to_string());
                }
                if *divisor <= 0.0 {
                    return Err("TlobDynamic divisor must be > 0".to_string());
                }
            }
        }
        Ok(())
    }

    /// Get a human-readable description of this strategy.
    pub fn description(&self) -> String {
        match self {
            Self::Fixed { value } => format!("Fixed threshold: {:.4}%", value * 100.0),
            Self::RollingSpread {
                window_size,
                multiplier,
                ..
            } => format!(
                "Rolling spread: {}x avg spread over {} samples",
                multiplier, window_size
            ),
            Self::Quantile {
                target_proportion,
                window_size,
                ..
            } => format!(
                "Quantile: {:.0}% Up/Down target over {} samples",
                target_proportion * 100.0,
                window_size
            ),
            Self::TlobDynamic { divisor, .. } => {
                format!("TLOB Dynamic: mean(|pct_change|) / {} (global)", divisor)
            }
        }
    }

    /// Check if this strategy requires global (full dataset) computation.
    ///
    /// Returns `true` for strategies that need all data before threshold can be computed.
    pub fn needs_global_computation(&self) -> bool {
        matches!(self, Self::TlobDynamic { .. })
    }
}

// ============================================================================
// Label Configuration (Export-specific wrapper)
// ============================================================================

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
}

fn default_volatility_floor() -> Option<f64> {
    None
}

fn default_volatility_cap() -> Option<f64> {
    None
}

fn default_horizon() -> usize {
    50
}

fn default_smoothing_window() -> usize {
    10
}

fn default_threshold() -> f64 {
    0.0008
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
        }
    }
}

impl ExportLabelConfig {
    /// Create single-horizon configuration with fixed threshold (TLOB strategy).
    ///
    /// # Arguments
    ///
    /// * `horizon` - Steps ahead to predict
    /// * `smoothing_window` - Smoothing window size
    /// * `threshold` - Classification threshold
    pub fn single(horizon: usize, smoothing_window: usize, threshold: f64) -> Self {
        Self {
            strategy: LabelingStrategy::Tlob,
            conflict_priority: None,
            horizon,
            horizons: Vec::new(),
            smoothing_window,
            threshold,
            threshold_strategy: None,
            profit_target_pct: None,
            stop_loss_pct: None,
            timeout_strategy: None,
            min_holding_period: None,
            profit_targets: None,
            stop_losses: None,
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    /// Create opportunity-based configuration for big-move detection.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Prediction horizons to look ahead
    /// * `threshold` - Minimum return to qualify as a "big move" (e.g., 0.005 = 50 bps)
    /// * `conflict_priority` - How to resolve when both up and down exceed threshold
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::{ExportLabelConfig, ExportConflictPriority};
    ///
    /// // Detect 0.5% moves with larger-magnitude priority
    /// let config = ExportLabelConfig::opportunity(
    ///     vec![50, 100, 200],
    ///     0.005,
    ///     ExportConflictPriority::LargerMagnitude,
    /// );
    /// assert!(config.strategy.is_opportunity());
    /// ```
    pub fn opportunity(
        horizons: Vec<usize>,
        threshold: f64,
        conflict_priority: ExportConflictPriority,
    ) -> Self {
        let max_horizon = *horizons.iter().max().unwrap_or(&50);
        Self {
            strategy: LabelingStrategy::Opportunity,
            conflict_priority: Some(conflict_priority),
            horizon: max_horizon,
            horizons,
            smoothing_window: 1, // Not used for opportunity labeling
            threshold,
            threshold_strategy: None,
            profit_target_pct: None,
            stop_loss_pct: None,
            timeout_strategy: None,
            min_holding_period: None,
            profit_targets: None,
            stop_losses: None,
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    // ========================================================================
    // Triple Barrier Factory Methods
    // ========================================================================

    /// Create Triple Barrier configuration for trade outcome labeling.
    ///
    /// Triple Barrier labeling determines which of three barriers is hit first:
    /// - **Upper barrier** (profit target): Price rises by `profit_target_pct`
    /// - **Lower barrier** (stop-loss): Price falls by `stop_loss_pct`
    /// - **Vertical barrier** (timeout): Maximum holding period reached
    ///
    /// # Arguments
    ///
    /// * `horizons` - Maximum holding periods (vertical barriers) for each horizon
    /// * `profit_target_pct` - Upper barrier as decimal (e.g., 0.005 = 0.5%)
    /// * `stop_loss_pct` - Lower barrier as decimal (e.g., 0.003 = 0.3%)
    ///
    /// # Label Encoding
    ///
    /// | Label | Class Index | Meaning |
    /// |-------|-------------|---------|
    /// | StopLoss | 0 | Hit lower barrier first |
    /// | Timeout | 1 | Neither barrier hit |
    /// | ProfitTarget | 2 | Hit upper barrier first |
    ///
    /// # Risk/Reward Ratio
    ///
    /// The ratio `profit_target_pct / stop_loss_pct` determines break-even win rate:
    /// - 1:1 ratio → need 50% win rate
    /// - 2:1 ratio → need 33% win rate
    /// - 1.67:1 ratio (0.5%/0.3%) → need ~38% win rate
    ///
    /// # Research Reference
    ///
    /// López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.
    /// The Triple Barrier Method.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// // Intraday config: 0.5% profit, 0.3% stop, multiple horizons
    /// let config = ExportLabelConfig::triple_barrier(
    ///     vec![50, 100, 200],
    ///     0.005,  // 0.5% profit target
    ///     0.003,  // 0.3% stop-loss
    /// );
    /// assert!(config.strategy.is_triple_barrier());
    /// ```
    pub fn triple_barrier(
        horizons: Vec<usize>,
        profit_target_pct: f64,
        stop_loss_pct: f64,
    ) -> Self {
        let max_horizon = *horizons.iter().max().unwrap_or(&100);
        Self {
            strategy: LabelingStrategy::TripleBarrier,
            conflict_priority: None, // Not used for Triple Barrier
            horizon: max_horizon,
            horizons,
            smoothing_window: 1,          // Not used for Triple Barrier
            threshold: profit_target_pct, // For backward compat in description()
            threshold_strategy: None,
            profit_target_pct: Some(profit_target_pct),
            stop_loss_pct: Some(stop_loss_pct),
            timeout_strategy: None, // Uses default: LabelAsTimeout
            min_holding_period: None,
            profit_targets: None, // Using global barriers
            stop_losses: None,
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    /// Create Triple Barrier configuration with symmetric barriers.
    ///
    /// Convenience method for when profit target equals stop-loss.
    /// Results in a 1:1 risk/reward ratio (need 50% win rate to break even).
    ///
    /// # Arguments
    ///
    /// * `horizons` - Maximum holding periods
    /// * `barrier_pct` - Both profit target and stop-loss percentage
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// // Symmetric 0.5% barriers
    /// let config = ExportLabelConfig::triple_barrier_symmetric(
    ///     vec![50, 100],
    ///     0.005,
    /// );
    /// ```
    pub fn triple_barrier_symmetric(horizons: Vec<usize>, barrier_pct: f64) -> Self {
        Self::triple_barrier(horizons, barrier_pct, barrier_pct)
    }

    /// Create fully-configured Triple Barrier configuration.
    ///
    /// Allows setting all Triple Barrier parameters including timeout strategy
    /// and minimum holding period.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Maximum holding periods
    /// * `profit_target_pct` - Upper barrier as decimal
    /// * `stop_loss_pct` - Lower barrier as decimal
    /// * `timeout_strategy` - How to label when timeout occurs
    /// * `min_holding_period` - Steps before barriers apply (optional)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::{ExportLabelConfig, ExportTimeoutStrategy};
    ///
    /// let config = ExportLabelConfig::triple_barrier_full(
    ///     vec![50, 100, 200],
    ///     0.005,
    ///     0.003,
    ///     ExportTimeoutStrategy::UseReturnSign,
    ///     Some(5),  // Don't check barriers for first 5 steps
    /// );
    /// ```
    pub fn triple_barrier_full(
        horizons: Vec<usize>,
        profit_target_pct: f64,
        stop_loss_pct: f64,
        timeout_strategy: ExportTimeoutStrategy,
        min_holding_period: Option<usize>,
    ) -> Self {
        let max_horizon = *horizons.iter().max().unwrap_or(&100);
        Self {
            strategy: LabelingStrategy::TripleBarrier,
            conflict_priority: None,
            horizon: max_horizon,
            horizons,
            smoothing_window: 1,
            threshold: profit_target_pct,
            threshold_strategy: None,
            profit_target_pct: Some(profit_target_pct),
            stop_loss_pct: Some(stop_loss_pct),
            timeout_strategy: Some(timeout_strategy),
            min_holding_period,
            profit_targets: None, // Using global barriers
            stop_losses: None,
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    /// Triple Barrier preset for scalping (tight barriers, short horizon).
    ///
    /// - Horizons: [10, 20, 50]
    /// - Profit target: 0.2% (20 bps)
    /// - Stop-loss: 0.15% (15 bps)
    /// - Risk/Reward: 1.33:1
    pub fn triple_barrier_scalping() -> Self {
        Self::triple_barrier(vec![10, 20, 50], 0.002, 0.0015)
    }

    /// Triple Barrier preset for intraday trading.
    ///
    /// - Horizons: [50, 100, 200]
    /// - Profit target: 0.5% (50 bps)
    /// - Stop-loss: 0.3% (30 bps)
    /// - Risk/Reward: 1.67:1 (need ~38% win rate)
    pub fn triple_barrier_intraday() -> Self {
        Self::triple_barrier(vec![50, 100, 200], 0.005, 0.003)
    }

    /// Triple Barrier preset for swing trading (wider barriers, longer horizon).
    ///
    /// - Horizons: [100, 200, 500]
    /// - Profit target: 1.0% (100 bps)
    /// - Stop-loss: 0.5% (50 bps)
    /// - Risk/Reward: 2:1 (need ~33% win rate)
    pub fn triple_barrier_swing() -> Self {
        Self::triple_barrier(vec![100, 200, 500], 0.01, 0.005)
    }

    /// Create Triple Barrier configuration with per-horizon barrier thresholds.
    ///
    /// This is the recommended approach for multi-horizon Triple Barrier labeling
    /// because price movements scale with time (volatility ∝ sqrt(time)).
    /// Using the same barriers for all horizons causes severe class imbalance.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Maximum holding periods (vertical barriers)
    /// * `profit_targets` - Profit target for each horizon (must match horizons length)
    /// * `stop_losses` - Stop loss for each horizon (must match horizons length)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// // Per-horizon barriers calibrated to NVDA volatility
    /// let config = ExportLabelConfig::triple_barrier_per_horizon(
    ///     vec![50, 100, 200],
    ///     vec![0.0028, 0.0039, 0.0059],  // 28, 39, 59 bps
    ///     vec![0.0019, 0.0026, 0.0040],  // 19, 26, 40 bps
    /// );
    /// ```
    ///
    /// # Calibration
    ///
    /// Use tools/calibrate_triple_barrier.py to compute data-driven thresholds:
    /// ```bash
    /// python tools/calibrate_triple_barrier.py --data-dir ../data/exports/my_export \
    ///     --horizons 50 100 200 --target-pt-rate 0.25 --target-sl-rate 0.25
    /// ```
    pub fn triple_barrier_per_horizon(
        horizons: Vec<usize>,
        profit_targets: Vec<f64>,
        stop_losses: Vec<f64>,
    ) -> Self {
        assert_eq!(
            horizons.len(),
            profit_targets.len(),
            "profit_targets must have same length as horizons"
        );
        assert_eq!(
            horizons.len(),
            stop_losses.len(),
            "stop_losses must have same length as horizons"
        );

        let max_horizon = *horizons.iter().max().unwrap_or(&100);
        // Use first horizon's values as global defaults (for backward compat)
        let default_pt = profit_targets.first().copied().unwrap_or(0.005);
        let default_sl = stop_losses.first().copied().unwrap_or(0.003);

        Self {
            strategy: LabelingStrategy::TripleBarrier,
            conflict_priority: None,
            horizon: max_horizon,
            horizons,
            smoothing_window: 1,
            threshold: default_pt,
            threshold_strategy: None,
            profit_target_pct: Some(default_pt),
            stop_loss_pct: Some(default_sl),
            timeout_strategy: None,
            min_holding_period: None,
            profit_targets: Some(profit_targets),
            stop_losses: Some(stop_losses),
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    /// Create multi-horizon configuration with fixed threshold (TLOB strategy).
    ///
    /// # Arguments
    ///
    /// * `horizons` - Array of prediction horizons
    /// * `smoothing_window` - Shared smoothing window size
    /// * `threshold` - Classification threshold
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// // FI-2010 benchmark horizons
    /// let config = ExportLabelConfig::multi(vec![10, 20, 30, 50, 100], 5, 0.002);
    /// assert!(config.is_multi_horizon());
    /// assert_eq!(config.horizons.len(), 5);
    /// ```
    pub fn multi(horizons: Vec<usize>, smoothing_window: usize, threshold: f64) -> Self {
        // Use max horizon as fallback single-horizon (for methods that need one)
        let max_horizon = *horizons.iter().max().unwrap_or(&50);
        Self {
            strategy: LabelingStrategy::Tlob,
            conflict_priority: None,
            horizon: max_horizon,
            horizons,
            smoothing_window,
            threshold,
            threshold_strategy: None,
            profit_target_pct: None,
            stop_loss_pct: None,
            timeout_strategy: None,
            min_holding_period: None,
            profit_targets: None,
            stop_losses: None,
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    /// Create multi-horizon configuration with explicit threshold strategy.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Array of prediction horizons
    /// * `smoothing_window` - Shared smoothing window size
    /// * `strategy` - Threshold strategy (Fixed, RollingSpread, or Quantile)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::{ExportLabelConfig, ExportThresholdStrategy};
    ///
    /// // Balanced class distribution for training
    /// let config = ExportLabelConfig::multi_with_strategy(
    ///     vec![10, 20, 50, 100, 200],
    ///     10,
    ///     ExportThresholdStrategy::quantile(0.33, 5000, 0.0008),
    /// );
    /// ```
    pub fn multi_with_strategy(
        horizons: Vec<usize>,
        smoothing_window: usize,
        thresh_strategy: ExportThresholdStrategy,
    ) -> Self {
        let max_horizon = *horizons.iter().max().unwrap_or(&50);
        let fallback_threshold = match &thresh_strategy {
            ExportThresholdStrategy::Fixed { value } => *value,
            ExportThresholdStrategy::RollingSpread { fallback, .. } => *fallback,
            ExportThresholdStrategy::Quantile { fallback, .. } => *fallback,
            ExportThresholdStrategy::TlobDynamic { fallback, .. } => *fallback,
        };
        Self {
            strategy: LabelingStrategy::Tlob,
            conflict_priority: None,
            horizon: max_horizon,
            horizons,
            smoothing_window,
            threshold: fallback_threshold,
            threshold_strategy: Some(thresh_strategy),
            profit_target_pct: None,
            stop_loss_pct: None,
            timeout_strategy: None,
            min_holding_period: None,
            profit_targets: None,
            stop_losses: None,
            volatility_scaling: None,
            volatility_reference: None,
            volatility_floor: None,
            volatility_cap: None,
        }
    }

    /// Create FI-2010 benchmark configuration.
    ///
    /// Horizons: [10, 20, 30, 50, 100]
    /// Smoothing: 5
    /// Threshold: 0.002 (20 bps)
    pub fn fi2010() -> Self {
        Self::multi(vec![10, 20, 30, 50, 100], 5, 0.002)
    }

    /// Create DeepLOB benchmark configuration.
    ///
    /// Horizons: [10, 20, 50, 100]
    /// Smoothing: 5
    /// Threshold: 0.002 (20 bps)
    pub fn deeplob() -> Self {
        Self::multi(vec![10, 20, 50, 100], 5, 0.002)
    }

    /// Create configuration optimized for balanced classes (recommended for training).
    ///
    /// Uses quantile-based thresholding to ensure roughly equal Up/Down/Stable
    /// proportions regardless of market volatility.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Array of prediction horizons
    /// * `smoothing_window` - Shared smoothing window size
    /// * `target_up_down_proportion` - Target proportion for Up+Down classes (e.g., 0.33 each)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// // ~33% Up, ~33% Down, ~34% Stable
    /// let config = ExportLabelConfig::balanced(
    ///     vec![10, 20, 50, 100, 200],
    ///     10,
    ///     0.33,
    /// );
    /// ```
    pub fn balanced(
        horizons: Vec<usize>,
        smoothing_window: usize,
        target_up_down_proportion: f64,
    ) -> Self {
        Self::multi_with_strategy(
            horizons,
            smoothing_window,
            ExportThresholdStrategy::quantile(target_up_down_proportion, 5000, 0.0008),
        )
    }

    /// Create configuration with spread-based adaptive threshold.
    ///
    /// Threshold adapts to market conditions based on bid-ask spread.
    /// Good for ensuring predictions are profitable after transaction costs.
    ///
    /// # Arguments
    ///
    /// * `horizons` - Array of prediction horizons
    /// * `smoothing_window` - Shared smoothing window size
    /// * `spread_multiplier` - Threshold = multiplier × avg_spread (e.g., 1.5)
    pub fn spread_adaptive(
        horizons: Vec<usize>,
        smoothing_window: usize,
        spread_multiplier: f64,
    ) -> Self {
        Self::multi_with_strategy(
            horizons,
            smoothing_window,
            ExportThresholdStrategy::rolling_spread(1000, spread_multiplier, 0.0008),
        )
    }

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

    /// Convert to internal LabelConfig (single-horizon).
    ///
    /// For multi-horizon mode, uses the first horizon.
    /// Prefer `to_multi_horizon_config()` for multi-horizon exports.
    pub fn to_label_config(&self) -> LabelConfig {
        let effective_horizon = if self.is_multi_horizon() {
            *self.horizons.first().unwrap_or(&self.horizon)
        } else {
            self.horizon
        };

        LabelConfig {
            horizon: effective_horizon,
            smoothing_window: self.smoothing_window,
            threshold: self.threshold,
        }
    }

    /// Get the effective threshold strategy.
    ///
    /// Returns the explicit `threshold_strategy` if provided,
    /// otherwise creates a `Fixed` strategy from the `threshold` field.
    pub fn effective_threshold_strategy(&self) -> ExportThresholdStrategy {
        self.threshold_strategy
            .clone()
            .unwrap_or_else(|| ExportThresholdStrategy::fixed(self.threshold))
    }

    /// Convert to MultiHorizonConfig for multi-horizon exports.
    ///
    /// Returns `Some(MultiHorizonConfig)` if multi-horizon mode is enabled,
    /// `None` otherwise (use `to_label_config()` for single-horizon mode).
    ///
    /// Uses the effective threshold strategy (explicit `threshold_strategy`
    /// if provided, otherwise `Fixed(threshold)`).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// let config = ExportLabelConfig::fi2010();
    /// let multi_config = config.to_multi_horizon_config().unwrap();
    /// assert_eq!(multi_config.horizons().len(), 5);
    /// ```
    pub fn to_multi_horizon_config(&self) -> Option<crate::labeling::MultiHorizonConfig> {
        if !self.is_multi_horizon() {
            return None;
        }

        let internal_strategy = self.effective_threshold_strategy().to_internal();

        Some(crate::labeling::MultiHorizonConfig::new(
            self.horizons.clone(),
            self.smoothing_window,
            internal_strategy,
        ))
    }

    /// Convert to OpportunityConfig for opportunity-based labeling.
    ///
    /// Returns `Some(OpportunityConfig)` if this is an opportunity labeling config,
    /// `None` otherwise.
    ///
    /// # Returns
    ///
    /// - `Some((Vec<OpportunityConfig>, horizons))` if `strategy == Opportunity`
    /// - `None` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::{ExportLabelConfig, ExportConflictPriority};
    ///
    /// let config = ExportLabelConfig::opportunity(vec![50, 100], 0.005, ExportConflictPriority::LargerMagnitude);
    /// let (opp_configs, horizons) = config.to_opportunity_configs().unwrap();
    /// assert_eq!(horizons.len(), 2);
    /// ```
    pub fn to_opportunity_configs(
        &self,
    ) -> Option<(Vec<crate::labeling::OpportunityConfig>, Vec<usize>)> {
        if !self.strategy.is_opportunity() {
            return None;
        }

        let horizons = self.effective_horizons();
        let conflict_priority = self.conflict_priority.unwrap_or_default().to_internal();

        let configs: Vec<crate::labeling::OpportunityConfig> = horizons
            .iter()
            .map(|&h| {
                crate::labeling::OpportunityConfig::new(h, self.threshold)
                    .with_conflict_priority(conflict_priority)
            })
            .collect();

        Some((configs, horizons))
    }

    /// Convert to TripleBarrierConfigs for Triple Barrier labeling.
    ///
    /// Returns `Some((Vec<TripleBarrierConfig>, horizons))` if `strategy == TripleBarrier`,
    /// `None` otherwise.
    ///
    /// Creates one `TripleBarrierConfig` per horizon, all sharing the same
    /// profit_target_pct and stop_loss_pct but with different max_horizon values.
    ///
    /// # Returns
    ///
    /// - `Some((configs, horizons))` where:
    ///   - `configs[i]` has `max_horizon = horizons[i]`
    ///   - All configs share the same barriers and timeout strategy
    /// - `None` if this is not a Triple Barrier config
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::export::ExportLabelConfig;
    ///
    /// let config = ExportLabelConfig::triple_barrier(vec![50, 100, 200], 0.005, 0.003);
    /// let (tb_configs, horizons) = config.to_triple_barrier_configs().unwrap();
    ///
    /// assert_eq!(horizons.len(), 3);
    /// assert_eq!(tb_configs[0].max_horizon, 50);
    /// assert_eq!(tb_configs[1].max_horizon, 100);
    /// assert_eq!(tb_configs[2].max_horizon, 200);
    /// ```
    ///
    /// # Research Reference
    ///
    /// López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.
    pub fn to_triple_barrier_configs(
        &self,
    ) -> Option<(Vec<crate::labeling::TripleBarrierConfig>, Vec<usize>)> {
        if !self.strategy.is_triple_barrier() {
            return None;
        }

        let horizons = self.effective_horizons();
        let timeout_strategy = self.timeout_strategy.unwrap_or_default().to_internal();
        let min_hold = self.min_holding_period;

        // Determine whether to use per-horizon or global barriers
        // Per-horizon overrides take precedence when provided
        let use_per_horizon = self.profit_targets.is_some() && self.stop_losses.is_some();

        if use_per_horizon {
            // Per-horizon barriers (Schema 3.2+)
            // Validates that arrays match horizon count
            let profit_targets = self.profit_targets.as_ref()?;
            let stop_losses = self.stop_losses.as_ref()?;

            if profit_targets.len() != horizons.len() || stop_losses.len() != horizons.len() {
                // This should be caught by validate(), but defensive check here
                eprintln!(
                    "WARNING: profit_targets/stop_losses length ({}/{}) != horizons length ({}). Using global barriers.",
                    profit_targets.len(),
                    stop_losses.len(),
                    horizons.len()
                );
                // Fall back to global barriers
                return self.to_triple_barrier_configs_global();
            }

            // Create one TripleBarrierConfig per horizon with its specific barriers
            let configs: Vec<crate::labeling::TripleBarrierConfig> = horizons
                .iter()
                .enumerate()
                .map(|(i, &max_horizon)| {
                    let pt = profit_targets[i];
                    let sl = stop_losses[i];
                    let mut config = crate::labeling::TripleBarrierConfig::new(pt, sl, max_horizon);
                    config = config.with_timeout_strategy(timeout_strategy);
                    if let Some(min_hold_period) = min_hold {
                        config = config.with_min_holding_period(min_hold_period);
                    }
                    config
                })
                .collect();

            Some((configs, horizons))
        } else {
            // Global barriers (backward compatible)
            self.to_triple_barrier_configs_global()
        }
    }

    /// Create Triple Barrier configs using global (single) barrier values.
    ///
    /// This is the backward-compatible path where all horizons share the same
    /// profit_target_pct and stop_loss_pct.
    fn to_triple_barrier_configs_global(
        &self,
    ) -> Option<(Vec<crate::labeling::TripleBarrierConfig>, Vec<usize>)> {
        let profit_target = self.profit_target_pct?;
        let stop_loss = self.stop_loss_pct?;

        let horizons = self.effective_horizons();
        let timeout_strategy = self.timeout_strategy.unwrap_or_default().to_internal();
        let min_hold = self.min_holding_period;

        let configs: Vec<crate::labeling::TripleBarrierConfig> = horizons
            .iter()
            .map(|&max_horizon| {
                let mut config = crate::labeling::TripleBarrierConfig::new(
                    profit_target,
                    stop_loss,
                    max_horizon,
                );
                config = config.with_timeout_strategy(timeout_strategy);
                if let Some(min_hold_period) = min_hold {
                    config = config.with_min_holding_period(min_hold_period);
                }
                config
            })
            .collect();

        Some((configs, horizons))
    }

    /// Check if volatility-adaptive barrier scaling is enabled (Schema 3.3+).
    pub fn is_volatility_scaling(&self) -> bool {
        self.volatility_scaling.unwrap_or(false)
    }

    /// Get volatility scaling parameters if enabled.
    ///
    /// Returns `Some((reference_vol, floor, cap))` if volatility scaling is enabled,
    /// `None` otherwise.
    pub fn volatility_scaling_params(&self) -> Option<(f64, f64, f64)> {
        if !self.is_volatility_scaling() {
            return None;
        }
        let reference = self.volatility_reference?;
        let floor = self.volatility_floor.unwrap_or(0.3);
        let cap = self.volatility_cap.unwrap_or(3.0);
        Some((reference, floor, cap))
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
        // Validate horizons (common to all strategies)
        if self.is_multi_horizon() {
            if self.horizons.contains(&0) {
                return Err("All horizons must be > 0".to_string());
            }
        } else if self.horizon == 0 {
            return Err("horizon must be > 0".to_string());
        }

        // Strategy-specific validation
        match &self.strategy {
            LabelingStrategy::Tlob => {
                // TLOB requires smoothing_window
                if self.smoothing_window == 0 {
                    return Err("smoothing_window must be > 0 for TLOB labeling".to_string());
                }
                let min_horizon = if self.is_multi_horizon() {
                    *self.horizons.iter().min().unwrap_or(&self.horizon)
                } else {
                    self.horizon
                };
                if self.smoothing_window > min_horizon {
                    return Err(format!(
                        "smoothing_window ({}) should be <= minimum horizon ({})",
                        self.smoothing_window, min_horizon
                    ));
                }

                // Validate threshold for TLOB (typically 2-100 bps)
                if let Some(ref ts) = self.threshold_strategy {
                    ts.validate()
                        .map_err(|e| format!("threshold_strategy: {}", e))?;
                } else {
                    if self.threshold <= 0.0 {
                        return Err("threshold must be > 0".to_string());
                    }
                    if self.threshold > 0.1 {
                        return Err("threshold seems too large (> 10%), check units".to_string());
                    }
                }
            }
            LabelingStrategy::Opportunity => {
                // Opportunity uses higher thresholds (typically 30-500 bps)
                if self.threshold <= 0.0 {
                    return Err("threshold must be > 0 for opportunity labeling".to_string());
                }
                if self.threshold > 0.5 {
                    return Err(
                        "threshold > 50% seems unreasonable for opportunity detection".to_string(),
                    );
                }
            }
            LabelingStrategy::TripleBarrier => {
                // Triple Barrier requires profit_target_pct and stop_loss_pct
                let profit_target = self.profit_target_pct.ok_or_else(|| {
                    "profit_target_pct is required for triple_barrier strategy".to_string()
                })?;
                let stop_loss = self.stop_loss_pct.ok_or_else(|| {
                    "stop_loss_pct is required for triple_barrier strategy".to_string()
                })?;

                // Validate profit_target_pct range: must be in (0, 1)
                if profit_target <= 0.0 {
                    return Err(format!(
                        "profit_target_pct must be > 0, got {}",
                        profit_target
                    ));
                }
                if profit_target >= 1.0 {
                    return Err(format!(
                        "profit_target_pct must be < 1.0 (100%), got {}",
                        profit_target
                    ));
                }
                // Sanity check: > 50% seems unreasonable
                if profit_target > 0.5 {
                    return Err(format!(
                        "profit_target_pct > 50% seems unreasonable, got {:.2}%",
                        profit_target * 100.0
                    ));
                }

                // Validate stop_loss_pct range: must be in (0, 1)
                if stop_loss <= 0.0 {
                    return Err(format!("stop_loss_pct must be > 0, got {}", stop_loss));
                }
                if stop_loss >= 1.0 {
                    return Err(format!(
                        "stop_loss_pct must be < 1.0 (100%), got {}",
                        stop_loss
                    ));
                }
                // Sanity check: > 50% seems unreasonable
                if stop_loss > 0.5 {
                    return Err(format!(
                        "stop_loss_pct > 50% seems unreasonable, got {:.2}%",
                        stop_loss * 100.0
                    ));
                }

                // Validate min_holding_period if present
                if let Some(min_hold) = self.min_holding_period {
                    let min_horizon = if self.is_multi_horizon() {
                        *self.horizons.iter().min().unwrap_or(&self.horizon)
                    } else {
                        self.horizon
                    };
                    if min_hold >= min_horizon {
                        return Err(format!(
                            "min_holding_period ({}) must be < minimum horizon ({})",
                            min_hold, min_horizon
                        ));
                    }
                }

                // At least one horizon required
                if self.horizons.is_empty() && self.horizon == 0 {
                    return Err(
                        "At least one horizon (max holding period) required for triple_barrier"
                            .to_string(),
                    );
                }

                // Validate per-horizon barrier arrays (Schema 3.2+)
                let effective_horizons = self.effective_horizons();
                if let Some(ref profit_targets) = self.profit_targets {
                    if profit_targets.len() != effective_horizons.len() {
                        return Err(format!(
                            "profit_targets length ({}) must match horizons length ({}). \
                             profit_targets should have one value per horizon: {:?}",
                            profit_targets.len(),
                            effective_horizons.len(),
                            effective_horizons
                        ));
                    }
                    // Validate each value
                    for (i, &pt) in profit_targets.iter().enumerate() {
                        if pt <= 0.0 || pt >= 1.0 {
                            return Err(format!(
                                "profit_targets[{}] must be in (0, 1), got {}",
                                i, pt
                            ));
                        }
                        if pt > 0.5 {
                            return Err(format!(
                                "profit_targets[{}] > 50% seems unreasonable, got {:.2}%",
                                i,
                                pt * 100.0
                            ));
                        }
                    }
                }
                if let Some(ref stop_losses) = self.stop_losses {
                    if stop_losses.len() != effective_horizons.len() {
                        return Err(format!(
                            "stop_losses length ({}) must match horizons length ({}). \
                             stop_losses should have one value per horizon: {:?}",
                            stop_losses.len(),
                            effective_horizons.len(),
                            effective_horizons
                        ));
                    }
                    // Validate each value
                    for (i, &sl) in stop_losses.iter().enumerate() {
                        if sl <= 0.0 || sl >= 1.0 {
                            return Err(format!(
                                "stop_losses[{}] must be in (0, 1), got {}",
                                i, sl
                            ));
                        }
                        if sl > 0.5 {
                            return Err(format!(
                                "stop_losses[{}] > 50% seems unreasonable, got {:.2}%",
                                i,
                                sl * 100.0
                            ));
                        }
                    }
                }

                // If one per-horizon array is provided, both must be
                if self.profit_targets.is_some() != self.stop_losses.is_some() {
                    return Err(
                        "If using per-horizon barriers, both profit_targets and stop_losses \
                         must be provided together"
                            .to_string(),
                    );
                }

                // Validate volatility scaling fields (Schema 3.3+)
                if self.volatility_scaling.unwrap_or(false) {
                    if self.volatility_reference.is_none() {
                        return Err(
                            "volatility_reference is required when volatility_scaling = true. \
                             Run tools/calibrate_triple_barrier.py --per-day to compute it."
                                .to_string(),
                        );
                    }
                    let vol_ref = self.volatility_reference.unwrap();
                    if vol_ref <= 0.0 {
                        return Err(format!("volatility_reference must be > 0, got {}", vol_ref));
                    }
                    if vol_ref > 0.1 {
                        return Err(format!(
                            "volatility_reference > 0.1 seems unreasonable \
                             (that's 1000 bps/tick), got {}",
                            vol_ref
                        ));
                    }

                    if self.profit_targets.is_none() || self.stop_losses.is_none() {
                        return Err("volatility_scaling requires per-horizon barriers \
                             (profit_targets and stop_losses) as base values"
                            .to_string());
                    }

                    if let Some(floor) = self.volatility_floor {
                        if floor <= 0.0 || floor > 1.0 {
                            return Err(format!(
                                "volatility_floor must be in (0, 1], got {}",
                                floor
                            ));
                        }
                    }
                    if let Some(cap) = self.volatility_cap {
                        if cap < 1.0 {
                            return Err(format!("volatility_cap must be >= 1.0, got {}", cap));
                        }
                        if cap > 100.0 {
                            return Err(format!(
                                "volatility_cap > 100 seems unreasonable, got {}",
                                cap
                            ));
                        }
                    }
                    if let (Some(floor), Some(cap)) = (self.volatility_floor, self.volatility_cap) {
                        if floor >= cap {
                            return Err(format!(
                                "volatility_floor ({}) must be < volatility_cap ({})",
                                floor, cap
                            ));
                        }
                    }
                }
            }
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
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ExportLabelConfig Tests
    // ========================================================================

    #[test]
    fn test_label_config_default_is_single_horizon() {
        let config = ExportLabelConfig::default();
        assert!(
            !config.is_multi_horizon(),
            "Default should be single-horizon mode"
        );
        assert_eq!(config.horizon, 50);
        assert!(config.horizons.is_empty());
        assert_eq!(config.smoothing_window, 10);
        assert!((config.threshold - 0.0008).abs() < 1e-10);
    }

    #[test]
    fn test_label_config_single_constructor() {
        let config = ExportLabelConfig::single(100, 20, 0.002);
        assert!(!config.is_multi_horizon());
        assert_eq!(config.horizon, 100);
        assert_eq!(config.smoothing_window, 20);
        assert!((config.threshold - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_label_config_multi_constructor() {
        let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.001);
        assert!(config.is_multi_horizon());
        assert_eq!(config.horizons, vec![10, 20, 50]);
        assert_eq!(config.smoothing_window, 5);
        assert!((config.threshold - 0.001).abs() < 1e-10);
        // max_horizon should be set as fallback single horizon
        assert_eq!(config.horizon, 50);
    }

    #[test]
    fn test_label_config_fi2010_preset() {
        let config = ExportLabelConfig::fi2010();
        assert!(config.is_multi_horizon());
        assert_eq!(config.horizons, vec![10, 20, 30, 50, 100]);
        assert_eq!(config.smoothing_window, 5);
        assert!((config.threshold - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_label_config_deeplob_preset() {
        let config = ExportLabelConfig::deeplob();
        assert!(config.is_multi_horizon());
        assert_eq!(config.horizons, vec![10, 20, 50, 100]);
        assert_eq!(config.smoothing_window, 5);
    }

    #[test]
    fn test_label_config_effective_horizons_single() {
        let config = ExportLabelConfig::single(200, 50, 0.0008);
        let effective = config.effective_horizons();
        assert_eq!(effective, vec![200]);
    }

    #[test]
    fn test_label_config_effective_horizons_multi() {
        let config = ExportLabelConfig::multi(vec![10, 20, 100], 5, 0.002);
        let effective = config.effective_horizons();
        assert_eq!(effective, vec![10, 20, 100]);
    }

    #[test]
    fn test_label_config_max_horizon_single() {
        let config = ExportLabelConfig::single(150, 30, 0.001);
        assert_eq!(config.max_horizon(), 150);
    }

    #[test]
    fn test_label_config_max_horizon_multi() {
        let config = ExportLabelConfig::multi(vec![10, 50, 200, 100], 5, 0.002);
        assert_eq!(config.max_horizon(), 200);
    }

    #[test]
    fn test_label_config_to_label_config_single() {
        let config = ExportLabelConfig::single(100, 20, 0.003);
        let label_config = config.to_label_config();
        assert_eq!(label_config.horizon, 100);
        assert_eq!(label_config.smoothing_window, 20);
        assert!((label_config.threshold - 0.003).abs() < 1e-10);
    }

    #[test]
    fn test_label_config_to_label_config_multi_uses_first_horizon() {
        // When converting multi-horizon to single LabelConfig, use first horizon
        let config = ExportLabelConfig::multi(vec![10, 50, 100], 5, 0.002);
        let label_config = config.to_label_config();
        assert_eq!(
            label_config.horizon, 10,
            "Should use first horizon for single-horizon conversion"
        );
        assert_eq!(label_config.smoothing_window, 5);
    }

    #[test]
    fn test_label_config_to_multi_horizon_config_single_returns_none() {
        let config = ExportLabelConfig::single(100, 20, 0.002);
        let multi_config = config.to_multi_horizon_config();
        assert!(
            multi_config.is_none(),
            "Single-horizon config should return None for multi-horizon conversion"
        );
    }

    #[test]
    fn test_label_config_to_multi_horizon_config_multi_returns_some() {
        let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.002);
        let multi_config = config.to_multi_horizon_config();
        assert!(multi_config.is_some());

        let mc = multi_config.unwrap();
        assert_eq!(mc.horizons(), &[10, 20, 50]);
        assert_eq!(mc.smoothing_window, 5);
    }

    #[test]
    fn test_label_config_validation_single_valid() {
        let config = ExportLabelConfig::single(100, 20, 0.002);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_label_config_validation_multi_valid() {
        let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.002);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_label_config_validation_single_zero_horizon() {
        let config = ExportLabelConfig::single(0, 5, 0.002);
        let result = config.validate();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("horizon must be > 0"),
            "Should reject zero horizon"
        );
    }

    #[test]
    fn test_label_config_validation_multi_zero_horizon_in_array() {
        let config = ExportLabelConfig::multi(vec![10, 0, 50], 5, 0.002);
        let result = config.validate();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("All horizons must be > 0"),
            "Should reject zero horizon in array"
        );
    }

    #[test]
    fn test_label_config_validation_zero_smoothing() {
        let config = ExportLabelConfig::single(100, 0, 0.002);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("smoothing_window must be > 0"));
    }

    #[test]
    fn test_label_config_validation_smoothing_greater_than_horizon() {
        let config = ExportLabelConfig::single(10, 20, 0.002);
        let result = config.validate();
        assert!(result.is_err());
        // Updated to match new error message format
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("smoothing_window")
                && err_msg.contains("should be <= minimum horizon")
        );
    }

    #[test]
    fn test_label_config_validation_multi_smoothing_greater_than_min_horizon() {
        // smoothing=20, but min horizon is 10
        let config = ExportLabelConfig::multi(vec![10, 50, 100], 20, 0.002);
        let result = config.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("smoothing_window (20) should be <= minimum horizon (10)"),
            "Should reject smoothing > min horizon"
        );
    }

    #[test]
    fn test_label_config_validation_zero_threshold() {
        let config = ExportLabelConfig::single(100, 10, 0.0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("threshold must be > 0"));
    }

    #[test]
    fn test_label_config_validation_negative_threshold() {
        let config = ExportLabelConfig::single(100, 10, -0.001);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("threshold must be > 0"));
    }

    #[test]
    fn test_label_config_validation_threshold_too_large() {
        let config = ExportLabelConfig::single(100, 10, 0.15);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("threshold seems too large"));
    }

    #[test]
    fn test_label_config_toml_serialization_single() {
        let config = ExportLabelConfig::single(100, 20, 0.0008);
        let toml_str = toml::to_string(&config).unwrap();

        // Should contain single horizon, not horizons array
        assert!(toml_str.contains("horizon = 100"));
        assert!(toml_str.contains("smoothing_window = 20"));
        assert!(
            !toml_str.contains("horizons"),
            "Empty horizons should be skipped"
        );
    }

    #[test]
    fn test_label_config_toml_serialization_multi() {
        let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.002);
        let toml_str = toml::to_string(&config).unwrap();

        // Should contain horizons array
        assert!(toml_str.contains("horizons = [10, 20, 50]"));
        assert!(toml_str.contains("smoothing_window = 5"));
    }

    #[test]
    fn test_label_config_toml_deserialization_single() {
        let toml_str = r#"
horizon = 200
smoothing_window = 50
threshold = 0.0008
"#;
        let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.is_multi_horizon());
        assert_eq!(config.horizon, 200);
        assert_eq!(config.smoothing_window, 50);
        assert!((config.threshold - 0.0008).abs() < 1e-10);
    }

    #[test]
    fn test_label_config_toml_deserialization_multi() {
        let toml_str = r#"
horizons = [10, 20, 50, 100, 200]
smoothing_window = 10
threshold = 0.0008
"#;
        let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
        assert!(config.is_multi_horizon());
        assert_eq!(config.horizons, vec![10, 20, 50, 100, 200]);
        assert_eq!(config.smoothing_window, 10);
    }

    #[test]
    fn test_label_config_toml_roundtrip_single() {
        let original = ExportLabelConfig::single(100, 20, 0.0015);
        let toml_str = toml::to_string(&original).unwrap();
        let loaded: ExportLabelConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(original.horizon, loaded.horizon);
        assert_eq!(original.smoothing_window, loaded.smoothing_window);
        assert!((original.threshold - loaded.threshold).abs() < 1e-10);
        assert_eq!(original.is_multi_horizon(), loaded.is_multi_horizon());
    }

    #[test]
    fn test_label_config_toml_roundtrip_multi() {
        let original = ExportLabelConfig::multi(vec![10, 20, 50, 100], 5, 0.002);
        let toml_str = toml::to_string(&original).unwrap();
        let loaded: ExportLabelConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(original.horizons, loaded.horizons);
        assert_eq!(original.smoothing_window, loaded.smoothing_window);
        assert!((original.threshold - loaded.threshold).abs() < 1e-10);
        assert_eq!(original.is_multi_horizon(), loaded.is_multi_horizon());
    }

    #[test]
    fn test_label_config_backward_compatibility() {
        // Old TOML format (single horizon only) should still work
        let old_format = r#"
horizon = 50
smoothing_window = 10
threshold = 0.0008
"#;
        let config: ExportLabelConfig = toml::from_str(old_format).unwrap();
        assert!(!config.is_multi_horizon());
        assert_eq!(config.horizon, 50);
        assert!(config.validate().is_ok());
    }

    // ========================================================================
    // TlobDynamic Threshold Strategy Tests
    // ========================================================================

    #[test]
    fn test_threshold_strategy_tlob_dynamic_creation() {
        let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
        assert!(matches!(
            strategy,
            ExportThresholdStrategy::TlobDynamic { fallback, divisor }
            if (fallback - 0.0008).abs() < 1e-10 && (divisor - 2.0).abs() < 1e-10
        ));

        // Test default divisor helper
        let strategy_default = ExportThresholdStrategy::tlob_dynamic_default(0.001);
        assert!(matches!(
            strategy_default,
            ExportThresholdStrategy::TlobDynamic { fallback, divisor }
            if (fallback - 0.001).abs() < 1e-10 && (divisor - 2.0).abs() < 1e-10
        ));
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_to_internal() {
        let export_strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.5);
        let internal = export_strategy.to_internal();

        assert!(matches!(
            internal,
            crate::labeling::ThresholdStrategy::TlobDynamic { fallback, divisor }
            if (fallback - 0.0008).abs() < 1e-10 && (divisor - 2.5).abs() < 1e-10
        ));
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_validation() {
        // Valid
        let valid = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
        assert!(valid.validate().is_ok());

        // Invalid fallback
        let invalid_fallback = ExportThresholdStrategy::tlob_dynamic(0.0, 2.0);
        assert!(invalid_fallback.validate().is_err());

        // Invalid divisor
        let invalid_divisor = ExportThresholdStrategy::tlob_dynamic(0.0008, 0.0);
        assert!(invalid_divisor.validate().is_err());
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_description() {
        let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
        let desc = strategy.description();
        assert!(desc.contains("TLOB Dynamic"));
        assert!(desc.contains("2"));
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_needs_global() {
        let fixed = ExportThresholdStrategy::fixed(0.002);
        let rolling = ExportThresholdStrategy::rolling_spread(100, 1.0, 0.002);
        let quantile = ExportThresholdStrategy::quantile(0.33, 5000, 0.002);
        let tlob_dynamic = ExportThresholdStrategy::tlob_dynamic_default(0.002);

        assert!(!fixed.needs_global_computation());
        assert!(!rolling.needs_global_computation());
        assert!(!quantile.needs_global_computation());
        assert!(
            tlob_dynamic.needs_global_computation(),
            "TlobDynamic should need global computation"
        );
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_toml_serialization() {
        let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
        let toml_str = toml::to_string(&strategy).expect("Should serialize TlobDynamic");

        // Verify it contains expected fields
        assert!(toml_str.contains("tlob_dynamic") || toml_str.contains("type"));
        assert!(toml_str.contains("fallback"));
        assert!(toml_str.contains("divisor"));
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_toml_deserialization() {
        let toml_str = r#"
            type = "tlob_dynamic"
            fallback = 0.0008
            divisor = 2.0
        "#;

        let strategy: ExportThresholdStrategy =
            toml::from_str(toml_str).expect("Should deserialize TlobDynamic");

        assert!(matches!(
            strategy,
            ExportThresholdStrategy::TlobDynamic { fallback, divisor }
            if (fallback - 0.0008).abs() < 1e-10 && (divisor - 2.0).abs() < 1e-10
        ));
    }

    #[test]
    fn test_threshold_strategy_tlob_dynamic_toml_roundtrip() {
        let original = ExportThresholdStrategy::tlob_dynamic(0.0005, 3.0);
        let toml_str = toml::to_string(&original).expect("Should serialize");
        let loaded: ExportThresholdStrategy =
            toml::from_str(&toml_str).expect("Should deserialize");

        assert_eq!(original, loaded);
    }

    #[test]
    fn test_label_config_with_tlob_dynamic_strategy() {
        let config = ExportLabelConfig::multi_with_strategy(
            vec![10, 20, 50],
            5,
            ExportThresholdStrategy::tlob_dynamic_default(0.0008),
        );

        assert!(config.is_multi_horizon());
        assert_eq!(config.horizons.len(), 3);

        // Check the threshold strategy was set correctly
        let thresh_strategy = config.effective_threshold_strategy();
        assert!(matches!(
            thresh_strategy,
            ExportThresholdStrategy::TlobDynamic { divisor, .. }
            if (divisor - 2.0).abs() < 1e-10
        ));
    }
}
