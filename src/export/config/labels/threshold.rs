use serde::{Deserialize, Serialize};

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
