use crate::labeling::LabelConfig;

use super::{ExportLabelConfig, ExportThresholdStrategy, LabelingStrategy};

impl ExportLabelConfig {
    // ========================================================================
    // TLOB/DeepLOB Factory Methods
    // ========================================================================

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
            return_type: None,
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
            return_type: None,
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
            return_type: None,
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

    // ========================================================================
    // TLOB/DeepLOB Conversion Methods
    // ========================================================================

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

    // ========================================================================
    // TLOB Validation
    // ========================================================================

    /// Validate TLOB-specific configuration.
    pub(super) fn validate_tlob(&self) -> Result<(), String> {
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

        Ok(())
    }
}
