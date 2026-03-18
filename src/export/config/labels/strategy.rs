use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Regression: continuous forward returns in bps.
    ///
    /// Outputs the raw smoothed return value (same formula as TLOB) as
    /// float64 instead of discretizing into Up/Down/Stable. Enables
    /// regression models to predict return magnitude directly.
    ///
    /// Export format: `{day}_regression_labels.npy` as float64 [N, H].
    Regression,
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
            Self::Regression => "Regression (continuous bps returns)",
        }
    }
}
