use super::{ExportConflictPriority, ExportLabelConfig, LabelingStrategy};

impl ExportLabelConfig {
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
            return_type: None,
        }
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

    /// Validate opportunity-specific configuration.
    pub(super) fn validate_opportunity(&self) -> Result<(), String> {
        if self.threshold <= 0.0 {
            return Err("threshold must be > 0 for opportunity labeling".to_string());
        }
        if self.threshold > 0.5 {
            return Err(
                "threshold > 50% seems unreasonable for opportunity detection".to_string(),
            );
        }
        Ok(())
    }
}
