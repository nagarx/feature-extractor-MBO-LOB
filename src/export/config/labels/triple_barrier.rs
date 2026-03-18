use super::{ExportLabelConfig, ExportTimeoutStrategy, LabelingStrategy};

impl ExportLabelConfig {
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
            return_type: None,
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
            return_type: None,
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
            return_type: None,
        }
    }

    // ========================================================================
    // Triple Barrier Conversion Methods
    // ========================================================================

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

        // Per-horizon overrides take precedence when provided
        let use_per_horizon = self.profit_targets.is_some() && self.stop_losses.is_some();

        if use_per_horizon {
            // Per-horizon barriers (Schema 3.2+)
            let profit_targets = self.profit_targets.as_ref()?;
            let stop_losses = self.stop_losses.as_ref()?;

            if profit_targets.len() != horizons.len() || stop_losses.len() != horizons.len() {
                eprintln!(
                    "WARNING: profit_targets/stop_losses length ({}/{}) != horizons length ({}). Using global barriers.",
                    profit_targets.len(),
                    stop_losses.len(),
                    horizons.len()
                );
                return self.to_triple_barrier_configs_global();
            }

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

    // ========================================================================
    // Triple Barrier Validation
    // ========================================================================

    /// Validate Triple Barrier-specific configuration.
    pub(super) fn validate_triple_barrier(&self) -> Result<(), String> {
        let profit_target = self.profit_target_pct.ok_or_else(|| {
            "profit_target_pct is required for triple_barrier strategy".to_string()
        })?;
        let stop_loss = self.stop_loss_pct.ok_or_else(|| {
            "stop_loss_pct is required for triple_barrier strategy".to_string()
        })?;

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
        if profit_target > 0.5 {
            return Err(format!(
                "profit_target_pct > 50% seems unreasonable, got {:.2}%",
                profit_target * 100.0
            ));
        }

        if stop_loss <= 0.0 {
            return Err(format!("stop_loss_pct must be > 0, got {}", stop_loss));
        }
        if stop_loss >= 1.0 {
            return Err(format!(
                "stop_loss_pct must be < 1.0 (100%), got {}",
                stop_loss
            ));
        }
        if stop_loss > 0.5 {
            return Err(format!(
                "stop_loss_pct > 50% seems unreasonable, got {:.2}%",
                stop_loss * 100.0
            ));
        }

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

        Ok(())
    }
}
