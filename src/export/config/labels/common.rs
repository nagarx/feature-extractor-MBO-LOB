use serde::{Deserialize, Serialize};

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
