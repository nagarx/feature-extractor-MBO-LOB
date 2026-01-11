//! Triple Barrier Labeling for Trading Strategy Development
//!
//! This module implements the Triple Barrier labeling method, a sophisticated
//! approach for generating trading labels that directly models entry/exit logic.
//!
//! # Research Reference
//!
//! López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.
//! The Triple Barrier Method.
//!
//! # Overview
//!
//! The Triple Barrier method places three barriers around each entry point:
//!
//! ```text
//!     Price                          Upper Barrier (Profit Target)
//!       ▲  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
//!       │                            ↑ profit_target_pct
//!       │  ●─────────●───────────────●
//!       │  Entry    │                │  ← Price path
//!       │           │                │
//!       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
//!                                    │     Lower Barrier (Stop-Loss)
//!                                    │
//!                   │◄───────────────►│
//!                     max_horizon (Vertical Barrier)
//! ```
//!
//! # Label Determination
//!
//! The label is determined by **which barrier is hit first**:
//!
//! | First Barrier Hit | Label | Meaning |
//! |-------------------|-------|---------|
//! | Upper (Profit Target) | +1 | Profitable trade |
//! | Lower (Stop-Loss) | -1 | Losing trade |
//! | Vertical (Time Limit) | 0 | No clear outcome |
//!
//! # Advantages Over Simple Labeling
//!
//! 1. **Directly models trading logic**: Entry → (Hold) → Exit
//! 2. **Incorporates risk/reward**: Asymmetric barriers possible
//! 3. **Handles time decay**: Vertical barrier captures "no opportunity" cases
//! 4. **Path-dependent**: Considers the entire price path, not just endpoints
//!
//! # Example
//!
//! ```
//! use feature_extractor::labeling::{TripleBarrierConfig, TripleBarrierLabeler, BarrierLabel};
//!
//! // Configure: 0.5% profit target, 0.3% stop-loss, 50 event max horizon
//! let config = TripleBarrierConfig::new(0.005, 0.003, 50);
//! let mut labeler = TripleBarrierLabeler::new(config);
//!
//! // Add prices
//! let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0).collect();
//! labeler.add_prices(&prices);
//!
//! let labels = labeler.generate_labels().unwrap();
//! for (idx, label, exit_time, exit_return) in &labels {
//!     println!("t={}: {:?}, exit_t={}, return={:.4}%",
//!              idx, label, exit_time, exit_return * 100.0);
//! }
//! ```

use mbo_lob_reconstructor::{Result, TlobError};
use serde::{Deserialize, Serialize};

// ============================================================================
// Barrier Label Enum
// ============================================================================

/// Triple Barrier label classification.
///
/// Represents the outcome of a trade entered at time t, based on which
/// barrier is hit first during the holding period.
///
/// # Research Reference
///
/// López de Prado (2018), §3.5: "The side of the barrier that is touched first
/// determines the label."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BarrierLabel {
    /// Stop-loss hit first: losing trade
    ///
    /// The price fell below the lower barrier before hitting the profit target
    /// or reaching the time limit.
    StopLoss = -1,

    /// Time limit reached: no clear outcome
    ///
    /// Neither the profit target nor stop-loss was hit within the maximum
    /// holding period. The trade is "inconclusive."
    ///
    /// Note: The final return at timeout may be positive or negative,
    /// but neither barrier was actually hit.
    Timeout = 0,

    /// Profit target hit first: profitable trade
    ///
    /// The price rose above the upper barrier before hitting the stop-loss
    /// or reaching the time limit.
    ProfitTarget = 1,
}

impl BarrierLabel {
    /// Convert to integer representation for ML models.
    ///
    /// Returns: -1 (StopLoss), 0 (Timeout), 1 (ProfitTarget)
    #[inline]
    pub fn as_int(&self) -> i8 {
        *self as i8
    }

    /// Convert to class index for softmax output (0-indexed).
    ///
    /// Returns: 0 (StopLoss), 1 (Timeout), 2 (ProfitTarget)
    ///
    /// This matches the convention used in TrendLabel and OpportunityLabel
    /// for consistency across the codebase.
    #[inline]
    pub fn as_class_index(&self) -> usize {
        match self {
            BarrierLabel::StopLoss => 0,
            BarrierLabel::Timeout => 1,
            BarrierLabel::ProfitTarget => 2,
        }
    }

    /// Create from integer representation.
    pub fn from_int(value: i8) -> Option<Self> {
        match value {
            -1 => Some(BarrierLabel::StopLoss),
            0 => Some(BarrierLabel::Timeout),
            1 => Some(BarrierLabel::ProfitTarget),
            _ => None,
        }
    }

    /// Create from class index (0-indexed).
    pub fn from_class_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(BarrierLabel::StopLoss),
            1 => Some(BarrierLabel::Timeout),
            2 => Some(BarrierLabel::ProfitTarget),
            _ => None,
        }
    }

    /// Get the string name of this label.
    pub fn name(&self) -> &'static str {
        match self {
            BarrierLabel::StopLoss => "StopLoss",
            BarrierLabel::Timeout => "Timeout",
            BarrierLabel::ProfitTarget => "ProfitTarget",
        }
    }

    /// Check if this label represents a decisive outcome (not timeout).
    #[inline]
    pub fn is_decisive(&self) -> bool {
        !matches!(self, BarrierLabel::Timeout)
    }

    /// Check if this is a winning trade.
    #[inline]
    pub fn is_win(&self) -> bool {
        matches!(self, BarrierLabel::ProfitTarget)
    }

    /// Check if this is a losing trade.
    #[inline]
    pub fn is_loss(&self) -> bool {
        matches!(self, BarrierLabel::StopLoss)
    }
}

impl std::fmt::Display for BarrierLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Triple Barrier labeling.
///
/// # Parameters
///
/// - `profit_target_pct`: Upper barrier as a percentage of entry price
/// - `stop_loss_pct`: Lower barrier as a percentage of entry price
/// - `max_horizon`: Maximum holding period in time steps (vertical barrier)
///
/// # Risk/Reward Ratio
///
/// The ratio `profit_target_pct / stop_loss_pct` determines the risk/reward:
///
/// | Ratio | Strategy Type | Expected Win Rate |
/// |-------|---------------|-------------------|
/// | 1:1 | Balanced | ~50% needed |
/// | 2:1 | High R:R | ~33% needed |
/// | 1:2 | High Win Rate | ~67% needed |
///
/// # Common Configurations
///
/// | Use Case | Profit Target | Stop-Loss | Max Horizon |
/// |----------|---------------|-----------|-------------|
/// | Scalping | 0.2% (20 bps) | 0.1% | 10-20 events |
/// | Intraday | 0.5% (50 bps) | 0.3% | 50-100 events |
/// | Swing | 1.0% (100 bps) | 0.5% | 200-500 events |
///
/// # Example
///
/// ```
/// use feature_extractor::labeling::TripleBarrierConfig;
///
/// // Balanced 0.5% barriers with 50-event horizon
/// let config = TripleBarrierConfig::symmetric(0.005, 50);
/// assert_eq!(config.profit_target_pct, 0.005);
/// assert_eq!(config.stop_loss_pct, 0.005);
///
/// // Asymmetric: 2:1 reward/risk ratio
/// let config = TripleBarrierConfig::new(0.01, 0.005, 100);
/// assert_eq!(config.risk_reward_ratio(), 2.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleBarrierConfig {
    /// Profit target percentage (upper barrier).
    ///
    /// The trade is labeled as ProfitTarget (+1) if the price rises by
    /// this percentage before hitting the stop-loss or max horizon.
    ///
    /// Example: 0.005 = 0.5% = 50 basis points
    pub profit_target_pct: f64,

    /// Stop-loss percentage (lower barrier).
    ///
    /// The trade is labeled as StopLoss (-1) if the price falls by
    /// this percentage before hitting the profit target or max horizon.
    ///
    /// Example: 0.003 = 0.3% = 30 basis points
    pub stop_loss_pct: f64,

    /// Maximum holding period in time steps (vertical barrier).
    ///
    /// If neither the profit target nor stop-loss is hit within this
    /// number of time steps, the trade is labeled as Timeout (0).
    pub max_horizon: usize,

    /// Optional: minimum holding period before barriers apply.
    ///
    /// If set, barriers are not checked until this many steps have passed.
    /// Useful for avoiding early exits due to noise.
    ///
    /// Default: None (barriers apply immediately)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_holding_period: Option<usize>,

    /// How to handle the label when timeout occurs.
    ///
    /// Some strategies use the sign of the return at timeout to assign
    /// a label, rather than always using Timeout (0).
    #[serde(default)]
    pub timeout_strategy: TimeoutStrategy,
}

/// Strategy for handling timeout cases.
///
/// When the vertical barrier (max_horizon) is hit, different strategies
/// can be used to assign the final label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TimeoutStrategy {
    /// Always label timeout as 0 (no clear outcome).
    ///
    /// This is the standard Triple Barrier approach.
    #[default]
    LabelAsTimeout,

    /// Label based on the sign of the return at timeout.
    ///
    /// If return > 0 at timeout, label as ProfitTarget.
    /// If return < 0 at timeout, label as StopLoss.
    /// If return = 0, label as Timeout.
    ///
    /// Use when you want no "inconclusive" labels.
    UseReturnSign,

    /// Label based on whether return exceeds a fraction of the barriers.
    ///
    /// Uses a configurable fraction of the target/stop to determine label.
    /// Example: If fraction = 0.5 and return > 0.5 * profit_target,
    /// label as ProfitTarget.
    UseFractionalThreshold,
}

impl TripleBarrierConfig {
    /// Create a new Triple Barrier configuration.
    ///
    /// # Arguments
    ///
    /// * `profit_target_pct` - Upper barrier as percentage (e.g., 0.005 = 0.5%)
    /// * `stop_loss_pct` - Lower barrier as percentage (e.g., 0.003 = 0.3%)
    /// * `max_horizon` - Maximum holding period in time steps
    ///
    /// # Panics
    ///
    /// Panics if any parameter is invalid (zero, negative, or too large).
    pub fn new(profit_target_pct: f64, stop_loss_pct: f64, max_horizon: usize) -> Self {
        assert!(
            profit_target_pct > 0.0 && profit_target_pct < 1.0,
            "profit_target_pct must be in range (0, 1), got {}",
            profit_target_pct
        );
        assert!(
            stop_loss_pct > 0.0 && stop_loss_pct < 1.0,
            "stop_loss_pct must be in range (0, 1), got {}",
            stop_loss_pct
        );
        assert!(max_horizon > 0, "max_horizon must be > 0");

        Self {
            profit_target_pct,
            stop_loss_pct,
            max_horizon,
            min_holding_period: None,
            timeout_strategy: TimeoutStrategy::default(),
        }
    }

    /// Create a symmetric configuration (profit target = stop-loss).
    ///
    /// # Arguments
    ///
    /// * `barrier_pct` - Both profit target and stop-loss percentage
    /// * `max_horizon` - Maximum holding period in time steps
    pub fn symmetric(barrier_pct: f64, max_horizon: usize) -> Self {
        Self::new(barrier_pct, barrier_pct, max_horizon)
    }

    /// Preset for scalping (tight barriers, short horizon).
    ///
    /// - Profit target: 0.2% (20 bps)
    /// - Stop-loss: 0.15% (15 bps)
    /// - Max horizon: 20 events
    pub fn scalping() -> Self {
        Self::new(0.002, 0.0015, 20)
    }

    /// Preset for intraday trading.
    ///
    /// - Profit target: 0.5% (50 bps)
    /// - Stop-loss: 0.3% (30 bps)
    /// - Max horizon: 100 events
    pub fn intraday() -> Self {
        Self::new(0.005, 0.003, 100)
    }

    /// Preset for swing trading (wider barriers, longer horizon).
    ///
    /// - Profit target: 1.0% (100 bps)
    /// - Stop-loss: 0.5% (50 bps)
    /// - Max horizon: 500 events
    pub fn swing() -> Self {
        Self::new(0.01, 0.005, 500)
    }

    /// Set minimum holding period before barriers apply.
    pub fn with_min_holding_period(mut self, min_period: usize) -> Self {
        self.min_holding_period = Some(min_period);
        self
    }

    /// Set timeout handling strategy.
    pub fn with_timeout_strategy(mut self, strategy: TimeoutStrategy) -> Self {
        self.timeout_strategy = strategy;
        self
    }

    /// Calculate the risk/reward ratio.
    ///
    /// Returns profit_target_pct / stop_loss_pct.
    ///
    /// Example: 0.01 / 0.005 = 2.0 (2:1 reward:risk)
    pub fn risk_reward_ratio(&self) -> f64 {
        self.profit_target_pct / self.stop_loss_pct
    }

    /// Calculate required win rate for break-even.
    ///
    /// For a risk/reward ratio of R:1, break-even win rate is 1/(1+R).
    ///
    /// Example:
    /// - R = 1.0 → need 50% win rate
    /// - R = 2.0 → need 33% win rate
    /// - R = 0.5 → need 67% win rate
    pub fn breakeven_win_rate(&self) -> f64 {
        let r = self.risk_reward_ratio();
        1.0 / (1.0 + r)
    }

    /// Calculate minimum required prices for label generation.
    ///
    /// We need max_horizon + 1 prices to generate one label.
    pub fn min_prices_required(&self) -> usize {
        self.max_horizon + 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.profit_target_pct <= 0.0 {
            return Err(TlobError::generic("profit_target_pct must be > 0"));
        }
        if self.profit_target_pct >= 1.0 {
            return Err(TlobError::generic("profit_target_pct must be < 1.0"));
        }
        if self.stop_loss_pct <= 0.0 {
            return Err(TlobError::generic("stop_loss_pct must be > 0"));
        }
        if self.stop_loss_pct >= 1.0 {
            return Err(TlobError::generic("stop_loss_pct must be < 1.0"));
        }
        if self.max_horizon == 0 {
            return Err(TlobError::generic("max_horizon must be > 0"));
        }
        if let Some(min_period) = self.min_holding_period {
            if min_period >= self.max_horizon {
                return Err(TlobError::generic(
                    "min_holding_period must be < max_horizon",
                ));
            }
        }
        Ok(())
    }
}

impl Default for TripleBarrierConfig {
    /// Default configuration: intraday preset.
    fn default() -> Self {
        Self::intraday()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for Triple Barrier label generation.
#[derive(Debug, Clone, Default)]
pub struct TripleBarrierStats {
    /// Total number of labels generated
    pub total: usize,

    /// Number of ProfitTarget (+1) labels
    pub profit_target_count: usize,

    /// Number of StopLoss (-1) labels
    pub stop_loss_count: usize,

    /// Number of Timeout (0) labels
    pub timeout_count: usize,

    /// Average time to barrier hit (excluding timeouts)
    pub avg_time_to_barrier: f64,

    /// Average return at exit (across all labels)
    pub avg_exit_return: f64,

    /// Average return for profit target exits
    pub avg_profit_return: f64,

    /// Average return for stop-loss exits
    pub avg_loss_return: f64,
}

impl TripleBarrierStats {
    /// Calculate win rate (profit targets / decisive outcomes).
    ///
    /// Excludes timeouts from the calculation.
    pub fn win_rate(&self) -> f64 {
        let decisive = self.profit_target_count + self.stop_loss_count;
        if decisive == 0 {
            return 0.0;
        }
        self.profit_target_count as f64 / decisive as f64
    }

    /// Calculate decisive rate (non-timeout / total).
    ///
    /// High decisive rate means barriers are being hit.
    /// Low decisive rate suggests barriers may be too wide.
    pub fn decisive_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.profit_target_count + self.stop_loss_count) as f64 / self.total as f64
    }

    /// Calculate class balance as (stop_loss_pct, timeout_pct, profit_target_pct).
    pub fn class_balance(&self) -> (f64, f64, f64) {
        if self.total == 0 {
            return (0.0, 0.0, 0.0);
        }
        let total = self.total as f64;
        (
            self.stop_loss_count as f64 / total,
            self.timeout_count as f64 / total,
            self.profit_target_count as f64 / total,
        )
    }

    /// Calculate expected return per trade.
    ///
    /// This is a simplified calculation assuming fixed position sizes.
    pub fn expected_return_per_trade(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.avg_exit_return
    }
}

// ============================================================================
// Labeler
// ============================================================================

/// Triple Barrier label generator.
///
/// Generates labels based on which barrier is hit first during the
/// simulated holding period for each entry point.
///
/// # Design
///
/// - Scans price path from entry to max_horizon
/// - Checks if profit target or stop-loss is hit at each step
/// - Labels based on first barrier touched
///
/// # Performance
///
/// - Time: O(T × H) where T = number of prices, H = max_horizon
/// - Space: O(T) for price storage
///
/// # Thread Safety
///
/// Not thread-safe. Use separate instances for parallel processing.
pub struct TripleBarrierLabeler {
    config: TripleBarrierConfig,
    prices: Vec<f64>,
}

/// Output tuple for Triple Barrier labels.
///
/// (entry_index, label, exit_time_offset, exit_return)
///
/// - `entry_index`: The time step at which the trade was entered
/// - `label`: The barrier label (ProfitTarget, StopLoss, Timeout)
/// - `exit_time_offset`: Number of steps from entry to exit (1 to max_horizon)
/// - `exit_return`: Return at exit point (may differ from barrier threshold)
pub type TripleBarrierOutput = (usize, BarrierLabel, usize, f64);

impl TripleBarrierLabeler {
    /// Create a new Triple Barrier labeler.
    pub fn new(config: TripleBarrierConfig) -> Self {
        Self {
            config,
            prices: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(config: TripleBarrierConfig, capacity: usize) -> Self {
        Self {
            config,
            prices: Vec::with_capacity(capacity),
        }
    }

    /// Add a mid-price to the buffer.
    #[inline]
    pub fn add_price(&mut self, mid_price: f64) {
        self.prices.push(mid_price);
    }

    /// Add multiple prices at once.
    pub fn add_prices(&mut self, prices: &[f64]) {
        self.prices.extend_from_slice(prices);
    }

    /// Get the number of prices in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Clear the price buffer.
    pub fn clear(&mut self) {
        self.prices.clear();
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TripleBarrierConfig {
        &self.config
    }

    /// Check if we have enough prices to generate at least one label.
    pub fn can_generate(&self) -> bool {
        self.prices.len() >= self.config.min_prices_required()
    }

    /// Generate labels for all valid entry points.
    ///
    /// Returns a vector of (entry_index, label, exit_time, exit_return) tuples.
    ///
    /// # Valid Range
    ///
    /// Labels are generated for entry points t in [0, T - max_horizon).
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TripleBarrierOutput>)` - Labels for each entry point
    /// * `Err` - If insufficient data
    pub fn generate_labels(&self) -> Result<Vec<TripleBarrierOutput>> {
        let max_h = self.config.max_horizon;
        let total = self.prices.len();
        let min_required = self.config.min_prices_required();

        if total < min_required {
            return Err(TlobError::generic(format!(
                "Need at least {} prices for Triple Barrier labeling with max_horizon {} (have {})",
                min_required, max_h, total
            )));
        }

        let mut labels = Vec::with_capacity(total - max_h);

        let profit_target = self.config.profit_target_pct;
        let stop_loss = self.config.stop_loss_pct;
        let min_hold = self.config.min_holding_period.unwrap_or(0);

        // Generate labels for each valid entry point
        for entry_t in 0..(total - max_h) {
            let entry_price = self.prices[entry_t];

            // Calculate barrier prices
            let upper_barrier = entry_price * (1.0 + profit_target);
            let lower_barrier = entry_price * (1.0 - stop_loss);

            // Scan price path to find first barrier hit
            let (label, exit_offset, exit_return) =
                self.find_first_barrier_hit(entry_t, entry_price, upper_barrier, lower_barrier, min_hold);

            labels.push((entry_t, label, exit_offset, exit_return));
        }

        Ok(labels)
    }

    /// Find which barrier is hit first from an entry point.
    ///
    /// Returns (label, exit_time_offset, exit_return).
    fn find_first_barrier_hit(
        &self,
        entry_t: usize,
        entry_price: f64,
        upper_barrier: f64,
        lower_barrier: f64,
        min_hold: usize,
    ) -> (BarrierLabel, usize, f64) {
        let max_h = self.config.max_horizon;

        // Scan each time step in the holding period
        for offset in 1..=max_h {
            // Skip if within minimum holding period
            if offset <= min_hold {
                continue;
            }

            let current_price = self.prices[entry_t + offset];
            let current_return = (current_price - entry_price) / entry_price;

            // Check upper barrier (profit target)
            if current_price >= upper_barrier {
                return (BarrierLabel::ProfitTarget, offset, current_return);
            }

            // Check lower barrier (stop-loss)
            if current_price <= lower_barrier {
                return (BarrierLabel::StopLoss, offset, current_return);
            }
        }

        // Vertical barrier (timeout) reached
        let final_price = self.prices[entry_t + max_h];
        let final_return = (final_price - entry_price) / entry_price;

        // Apply timeout strategy
        let label = match self.config.timeout_strategy {
            TimeoutStrategy::LabelAsTimeout => BarrierLabel::Timeout,
            TimeoutStrategy::UseReturnSign => {
                if final_return > 0.0 {
                    BarrierLabel::ProfitTarget
                } else if final_return < 0.0 {
                    BarrierLabel::StopLoss
                } else {
                    BarrierLabel::Timeout
                }
            }
            TimeoutStrategy::UseFractionalThreshold => {
                let profit_threshold = self.config.profit_target_pct * 0.5;
                let loss_threshold = self.config.stop_loss_pct * 0.5;

                if final_return > profit_threshold {
                    BarrierLabel::ProfitTarget
                } else if final_return < -loss_threshold {
                    BarrierLabel::StopLoss
                } else {
                    BarrierLabel::Timeout
                }
            }
        };

        (label, max_h, final_return)
    }

    /// Generate labels and return only the labels (without exit info).
    pub fn generate_label_sequence(&self) -> Result<Vec<BarrierLabel>> {
        let labels = self.generate_labels()?;
        Ok(labels.into_iter().map(|(_, label, _, _)| label).collect())
    }

    /// Generate labels and return as class indices.
    pub fn generate_class_indices(&self) -> Result<Vec<usize>> {
        let labels = self.generate_labels()?;
        Ok(labels
            .into_iter()
            .map(|(_, label, _, _)| label.as_class_index())
            .collect())
    }

    /// Compute statistics for the generated labels.
    pub fn compute_stats(&self, labels: &[TripleBarrierOutput]) -> TripleBarrierStats {
        if labels.is_empty() {
            return TripleBarrierStats::default();
        }

        let mut stats = TripleBarrierStats::default();
        stats.total = labels.len();

        let mut sum_exit_return = 0.0;
        let mut sum_profit_return = 0.0;
        let mut sum_loss_return = 0.0;
        let mut sum_time_to_barrier = 0usize;
        let mut barrier_hit_count = 0usize;

        for (_, label, exit_time, exit_return) in labels {
            match label {
                BarrierLabel::ProfitTarget => {
                    stats.profit_target_count += 1;
                    sum_profit_return += exit_return;
                    sum_time_to_barrier += exit_time;
                    barrier_hit_count += 1;
                }
                BarrierLabel::StopLoss => {
                    stats.stop_loss_count += 1;
                    sum_loss_return += exit_return;
                    sum_time_to_barrier += exit_time;
                    barrier_hit_count += 1;
                }
                BarrierLabel::Timeout => {
                    stats.timeout_count += 1;
                }
            }

            sum_exit_return += exit_return;
        }

        stats.avg_exit_return = sum_exit_return / stats.total as f64;

        if barrier_hit_count > 0 {
            stats.avg_time_to_barrier = sum_time_to_barrier as f64 / barrier_hit_count as f64;
        }

        if stats.profit_target_count > 0 {
            stats.avg_profit_return = sum_profit_return / stats.profit_target_count as f64;
        }

        if stats.stop_loss_count > 0 {
            stats.avg_loss_return = sum_loss_return / stats.stop_loss_count as f64;
        }

        stats
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BarrierLabel Tests
    // ========================================================================

    #[test]
    fn test_barrier_label_conversion() {
        // Test int conversion
        assert_eq!(BarrierLabel::ProfitTarget.as_int(), 1);
        assert_eq!(BarrierLabel::Timeout.as_int(), 0);
        assert_eq!(BarrierLabel::StopLoss.as_int(), -1);

        // Test roundtrip
        for label in [
            BarrierLabel::StopLoss,
            BarrierLabel::Timeout,
            BarrierLabel::ProfitTarget,
        ] {
            let as_int = label.as_int();
            let recovered = BarrierLabel::from_int(as_int).unwrap();
            assert_eq!(label, recovered);
        }
    }

    #[test]
    fn test_barrier_label_class_index() {
        assert_eq!(BarrierLabel::StopLoss.as_class_index(), 0);
        assert_eq!(BarrierLabel::Timeout.as_class_index(), 1);
        assert_eq!(BarrierLabel::ProfitTarget.as_class_index(), 2);

        // Test roundtrip
        for label in [
            BarrierLabel::StopLoss,
            BarrierLabel::Timeout,
            BarrierLabel::ProfitTarget,
        ] {
            let as_idx = label.as_class_index();
            let recovered = BarrierLabel::from_class_index(as_idx).unwrap();
            assert_eq!(label, recovered);
        }
    }

    #[test]
    fn test_barrier_label_predicates() {
        assert!(BarrierLabel::ProfitTarget.is_decisive());
        assert!(BarrierLabel::StopLoss.is_decisive());
        assert!(!BarrierLabel::Timeout.is_decisive());

        assert!(BarrierLabel::ProfitTarget.is_win());
        assert!(!BarrierLabel::StopLoss.is_win());
        assert!(!BarrierLabel::Timeout.is_win());

        assert!(BarrierLabel::StopLoss.is_loss());
        assert!(!BarrierLabel::ProfitTarget.is_loss());
        assert!(!BarrierLabel::Timeout.is_loss());
    }

    // ========================================================================
    // TripleBarrierConfig Tests
    // ========================================================================

    #[test]
    fn test_config_creation() {
        let config = TripleBarrierConfig::new(0.005, 0.003, 50);
        assert_eq!(config.profit_target_pct, 0.005);
        assert_eq!(config.stop_loss_pct, 0.003);
        assert_eq!(config.max_horizon, 50);
        assert!(config.min_holding_period.is_none());
    }

    #[test]
    fn test_config_symmetric() {
        let config = TripleBarrierConfig::symmetric(0.01, 100);
        assert_eq!(config.profit_target_pct, 0.01);
        assert_eq!(config.stop_loss_pct, 0.01);
        assert_eq!(config.risk_reward_ratio(), 1.0);
    }

    #[test]
    fn test_config_presets() {
        let scalping = TripleBarrierConfig::scalping();
        assert!(scalping.validate().is_ok());
        assert_eq!(scalping.max_horizon, 20);

        let intraday = TripleBarrierConfig::intraday();
        assert!(intraday.validate().is_ok());
        assert_eq!(intraday.max_horizon, 100);

        let swing = TripleBarrierConfig::swing();
        assert!(swing.validate().is_ok());
        assert_eq!(swing.max_horizon, 500);
    }

    #[test]
    fn test_config_risk_reward() {
        // 2:1 reward/risk
        let config = TripleBarrierConfig::new(0.01, 0.005, 100);
        assert!((config.risk_reward_ratio() - 2.0).abs() < 1e-10);
        assert!((config.breakeven_win_rate() - 0.333333).abs() < 0.001);

        // 1:2 reward/risk
        let config = TripleBarrierConfig::new(0.005, 0.01, 100);
        assert!((config.risk_reward_ratio() - 0.5).abs() < 1e-10);
        assert!((config.breakeven_win_rate() - 0.666666).abs() < 0.001);
    }

    #[test]
    fn test_config_validation() {
        let valid = TripleBarrierConfig::new(0.005, 0.003, 50);
        assert!(valid.validate().is_ok());

        // Invalid profit target
        let config = TripleBarrierConfig {
            profit_target_pct: 0.0,
            stop_loss_pct: 0.003,
            max_horizon: 50,
            min_holding_period: None,
            timeout_strategy: TimeoutStrategy::default(),
        };
        assert!(config.validate().is_err());

        // Invalid stop loss
        let config = TripleBarrierConfig {
            profit_target_pct: 0.005,
            stop_loss_pct: 1.5,
            max_horizon: 50,
            min_holding_period: None,
            timeout_strategy: TimeoutStrategy::default(),
        };
        assert!(config.validate().is_err());

        // Invalid min_holding_period
        let config = TripleBarrierConfig {
            profit_target_pct: 0.005,
            stop_loss_pct: 0.003,
            max_horizon: 50,
            min_holding_period: Some(60), // > max_horizon
            timeout_strategy: TimeoutStrategy::default(),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_min_prices_required() {
        let config = TripleBarrierConfig::new(0.005, 0.003, 50);
        assert_eq!(config.min_prices_required(), 51); // max_horizon + 1
    }

    #[test]
    #[should_panic(expected = "profit_target_pct must be in range (0, 1)")]
    fn test_config_invalid_profit_target_panics() {
        TripleBarrierConfig::new(0.0, 0.003, 50);
    }

    #[test]
    #[should_panic(expected = "max_horizon must be > 0")]
    fn test_config_zero_horizon_panics() {
        TripleBarrierConfig::new(0.005, 0.003, 0);
    }

    // ========================================================================
    // TripleBarrierLabeler Tests
    // ========================================================================

    #[test]
    fn test_labeler_creation() {
        let config = TripleBarrierConfig::new(0.01, 0.01, 10);
        let labeler = TripleBarrierLabeler::new(config);
        assert!(labeler.is_empty());
        assert!(!labeler.can_generate());
    }

    #[test]
    fn test_labeler_add_prices() {
        let config = TripleBarrierConfig::new(0.01, 0.01, 10);
        let mut labeler = TripleBarrierLabeler::new(config);

        labeler.add_price(100.0);
        labeler.add_price(101.0);
        assert_eq!(labeler.len(), 2);

        labeler.add_prices(&[102.0, 103.0, 104.0]);
        assert_eq!(labeler.len(), 5);
    }

    #[test]
    fn test_labeler_can_generate() {
        // Need max_horizon + 1 = 11 prices
        let config = TripleBarrierConfig::new(0.01, 0.01, 10);
        let mut labeler = TripleBarrierLabeler::new(config);

        let prices: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        labeler.add_prices(&prices);
        assert!(!labeler.can_generate()); // 10 < 11

        labeler.add_price(110.0);
        assert!(labeler.can_generate()); // 11 >= 11
    }

    #[test]
    fn test_labeler_profit_target_hit() {
        // Create scenario where profit target is hit
        let config = TripleBarrierConfig::new(0.02, 0.01, 10); // 2% profit, 1% stop
        let mut labeler = TripleBarrierLabeler::new(config);

        // Prices: 100, then steadily rising to 103 (3% up)
        // Profit target (2%) should be hit
        let prices = vec![
            100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0,
        ];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        assert!(!labels.is_empty());

        let (idx, label, exit_time, exit_return) = &labels[0];
        assert_eq!(*idx, 0);
        assert_eq!(*label, BarrierLabel::ProfitTarget);
        assert!(*exit_return >= 0.02); // Should be at or above profit target
        assert!(*exit_time < 10); // Should hit before max horizon
    }

    #[test]
    fn test_labeler_stop_loss_hit() {
        // Create scenario where stop-loss is hit
        let config = TripleBarrierConfig::new(0.02, 0.01, 10); // 2% profit, 1% stop
        let mut labeler = TripleBarrierLabeler::new(config);

        // Prices: 100, then steadily falling to 98 (2% down)
        // Stop-loss (1%) should be hit
        let prices = vec![
            100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 97.0, 97.0, 97.0, 97.0,
        ];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        assert!(!labels.is_empty());

        let (idx, label, exit_time, exit_return) = &labels[0];
        assert_eq!(*idx, 0);
        assert_eq!(*label, BarrierLabel::StopLoss);
        assert!(*exit_return <= -0.01); // Should be at or below stop-loss
        assert!(*exit_time < 10); // Should hit before max horizon
    }

    #[test]
    fn test_labeler_timeout() {
        // Create scenario where neither barrier is hit
        let config = TripleBarrierConfig::new(0.05, 0.05, 5); // Wide 5% barriers
        let mut labeler = TripleBarrierLabeler::new(config);

        // Prices: stable around 100 (small movements, never hitting 5% barriers)
        let prices = vec![100.0, 100.1, 99.9, 100.2, 99.8, 100.0];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        assert!(!labels.is_empty());

        let (_, label, exit_time, _) = &labels[0];
        assert_eq!(*label, BarrierLabel::Timeout);
        assert_eq!(*exit_time, 5); // Should exit at max horizon
    }

    #[test]
    fn test_labeler_timeout_with_return_sign_strategy() {
        let config = TripleBarrierConfig::new(0.05, 0.05, 5)
            .with_timeout_strategy(TimeoutStrategy::UseReturnSign);
        let mut labeler = TripleBarrierLabeler::new(config);

        // Prices: end higher than start (positive return at timeout)
        let prices = vec![100.0, 100.1, 100.2, 100.3, 100.4, 101.0];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        let (_, label, _, exit_return) = &labels[0];

        // With UseReturnSign, positive return at timeout → ProfitTarget
        assert!(*exit_return > 0.0);
        assert_eq!(*label, BarrierLabel::ProfitTarget);
    }

    #[test]
    fn test_labeler_min_holding_period() {
        // Test that barriers are not checked during min holding period
        let config = TripleBarrierConfig::new(0.01, 0.01, 10).with_min_holding_period(3);
        let mut labeler = TripleBarrierLabeler::new(config);

        // Price spikes up 2% at t=1, but min_hold=3 means it should be ignored
        // Then returns to normal
        let prices = vec![
            100.0, 102.0, // Spike at t=1 (within min_hold, should be ignored)
            100.0, 100.0, // Returns to normal
            100.5, 100.5, 101.0, 101.0, 101.5, 102.0, 102.0,
        ];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        let (_, label, exit_time, _) = &labels[0];

        // The spike at t=1 should be ignored due to min_holding_period
        // Label should be from later in the path
        assert!(*exit_time >= 3, "Exit should be after min_holding_period");

        // This tests the min_holding_period is working, the exact label
        // depends on whether the later price path hits a barrier
        println!(
            "Label: {:?}, exit_time: {}, (expected >= 3)",
            label, exit_time
        );
    }

    #[test]
    fn test_labeler_deterministic() {
        let config = TripleBarrierConfig::new(0.01, 0.01, 10);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let mut lab1 = TripleBarrierLabeler::new(config.clone());
        lab1.add_prices(&prices);
        let labels1 = lab1.generate_labels().unwrap();

        let mut lab2 = TripleBarrierLabeler::new(config);
        lab2.add_prices(&prices);
        let labels2 = lab2.generate_labels().unwrap();

        assert_eq!(labels1.len(), labels2.len());
        for (l1, l2) in labels1.iter().zip(labels2.iter()) {
            assert_eq!(l1.0, l2.0); // Same index
            assert_eq!(l1.1, l2.1); // Same label
            assert_eq!(l1.2, l2.2); // Same exit time
            assert!((l1.3 - l2.3).abs() < 1e-15); // Same exit return
        }
    }

    #[test]
    fn test_labeler_no_nan_or_inf() {
        let config = TripleBarrierConfig::new(0.01, 0.01, 10);
        let mut labeler = TripleBarrierLabeler::new(config);

        // Various price patterns
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();

        for (_, _, _, exit_return) in &labels {
            assert!(!exit_return.is_nan(), "Found NaN in exit_return");
            assert!(!exit_return.is_infinite(), "Found Inf in exit_return");
        }
    }

    #[test]
    fn test_labeler_clear_and_reuse() {
        let config = TripleBarrierConfig::new(0.02, 0.01, 5);
        let mut labeler = TripleBarrierLabeler::new(config);

        // First use - upward movement
        labeler.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
        let labels1 = labeler.generate_labels().unwrap();
        assert!(!labels1.is_empty());

        // Clear and reuse - downward movement
        labeler.clear();
        assert!(labeler.is_empty());

        labeler.add_prices(&[100.0, 99.0, 98.0, 97.0, 96.0, 95.0]);
        let labels2 = labeler.generate_labels().unwrap();
        assert!(!labels2.is_empty());

        // First should be profit target, second should be stop-loss
        assert_eq!(labels1[0].1, BarrierLabel::ProfitTarget);
        assert_eq!(labels2[0].1, BarrierLabel::StopLoss);
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_stats_computation() {
        let config = TripleBarrierConfig::new(0.02, 0.01, 10);
        let mut labeler = TripleBarrierLabeler::new(config);

        // Create mixed scenario
        let mut prices = Vec::new();

        // Upward spike (profit target)
        prices.extend(vec![100.0, 101.0, 102.0, 103.0, 102.5, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0]);

        // Stable period (extend for another entry point)
        prices.extend(vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);

        // Downward spike (stop-loss)
        prices.extend(vec![100.0, 99.0, 98.0, 97.0, 97.5, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0]);

        labeler.add_prices(&prices);
        let labels = labeler.generate_labels().unwrap();
        let stats = labeler.compute_stats(&labels);

        // Verify counts sum to total
        assert_eq!(
            stats.profit_target_count + stats.stop_loss_count + stats.timeout_count,
            stats.total
        );

        // Verify we have at least some decisive outcomes
        assert!(
            stats.profit_target_count + stats.stop_loss_count > 0,
            "Should have some decisive outcomes"
        );

        // Verify class balance sums to 1
        let (sl, to, pt) = stats.class_balance();
        let sum = sl + to + pt;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Class balance should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_stats_win_rate() {
        let stats = TripleBarrierStats {
            total: 100,
            profit_target_count: 40,
            stop_loss_count: 30,
            timeout_count: 30,
            avg_time_to_barrier: 5.0,
            avg_exit_return: 0.001,
            avg_profit_return: 0.02,
            avg_loss_return: -0.01,
        };

        // Win rate = 40 / (40 + 30) = 40/70 ≈ 0.571
        let win_rate = stats.win_rate();
        assert!((win_rate - 0.5714).abs() < 0.001);

        // Decisive rate = 70 / 100 = 0.70
        let decisive_rate = stats.decisive_rate();
        assert!((decisive_rate - 0.70).abs() < 0.001);
    }

    #[test]
    fn test_stats_empty() {
        let stats = TripleBarrierStats::default();
        assert_eq!(stats.win_rate(), 0.0);
        assert_eq!(stats.decisive_rate(), 0.0);
        assert_eq!(stats.class_balance(), (0.0, 0.0, 0.0));
    }

    // ========================================================================
    // Formula Correctness Tests
    // ========================================================================

    #[test]
    fn test_formula_barrier_prices() {
        // Verify barrier price calculations
        let config = TripleBarrierConfig::new(0.02, 0.01, 10); // 2% profit, 1% stop
        let mut labeler = TripleBarrierLabeler::new(config);

        // Entry at 100.0
        // Upper barrier should be at 102.0 (100 * 1.02)
        // Lower barrier should be at 99.0 (100 * 0.99)
        let prices = vec![
            100.0, 100.5, 101.0, 101.5, 102.0, // Exactly at upper barrier
            102.5, 103.0, 103.0, 103.0, 103.0, 103.0,
        ];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        let (_, label, exit_time, exit_return) = &labels[0];

        assert_eq!(*label, BarrierLabel::ProfitTarget);
        // Price hit 102.0 at t=4 (offset 4 from entry at t=0)
        assert_eq!(*exit_time, 4);
        // Return should be exactly 2%
        assert!((*exit_return - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_exact_barrier_touch() {
        // Test behavior when price exactly touches barrier
        let config = TripleBarrierConfig::new(0.03, 0.02, 5); // 3% profit, 2% stop
        let mut labeler = TripleBarrierLabeler::new(config);

        // Entry at 100, upper barrier at 103, lower at 98
        // Price exactly touches upper barrier
        let prices = vec![100.0, 101.0, 102.0, 103.0, 102.0, 101.0];
        labeler.add_prices(&prices);

        let labels = labeler.generate_labels().unwrap();
        let (_, label, _, _) = &labels[0];

        // Should trigger at exact touch (>= for upper, <= for lower)
        assert_eq!(*label, BarrierLabel::ProfitTarget);
    }

    #[test]
    fn test_insufficient_data_error() {
        let config = TripleBarrierConfig::new(0.01, 0.01, 50);
        let mut labeler = TripleBarrierLabeler::new(config);

        labeler.add_prices(&[100.0; 20]); // Only 20 prices

        let result = labeler.generate_labels();
        assert!(result.is_err());
    }
}

