//! Type definitions for the aligned export module.
//!
//! Contains all public types used across the export pipeline:
//! - `LabelEncoding`: Label encoding contract per labeling strategy
//! - `NormalizationStrategy` / `NormalizationParams`: Normalization metadata
//! - `AlignedDayExport`: Result of a single-day export

use mbo_lob_reconstructor::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

// ============================================================================
// Label Encoding Contract
// ============================================================================

/// Defines the valid label encoding contract for each labeling strategy.
///
/// This is the single source of truth for label encoding across the export
/// pipeline. Every labeling strategy must define its encoding here, and all
/// validation uses this contract.
///
/// Supports both classification (discrete i8 labels) and regression
/// (continuous f64 labels) strategies.
///
/// # Label Encoding Summary
///
/// | Strategy        | Type           | Values           | Description                       |
/// |-----------------|----------------|------------------|-----------------------------------|
/// | TLOB            | Classification | {-1, 0, 1} i8   | Down, Stable, Up                  |
/// | Opportunity     | Classification | {-1, 0, 1} i8   | BigDown, NoOpportunity, BigUp     |
/// | Triple Barrier  | Classification | {0, 1, 2} i8    | StopLoss, Timeout, ProfitTarget   |
/// | ContinuousBps   | Regression     | f64              | Forward returns in basis points   |
#[derive(Debug, Clone, PartialEq)]
pub enum LabelEncoding {
    /// TLOB / Multi-horizon: labels as signed integers {-1, 0, 1}
    SignedTrend,
    /// Opportunity: labels as signed integers {-1, 0, 1}
    SignedOpportunity,
    /// Triple Barrier: labels as class indices {0, 1, 2}
    TripleBarrierClassIndex,
    /// Regression: continuous forward returns in basis points (f64).
    /// Primary output is `regression_label_matrix`; classification labels are optional.
    ContinuousBps,
}

impl LabelEncoding {
    /// Whether this encoding represents a regression (continuous) target.
    pub fn is_regression(&self) -> bool {
        matches!(self, LabelEncoding::ContinuousBps)
    }

    /// Whether this encoding represents a classification (discrete) target.
    pub fn is_classification(&self) -> bool {
        !self.is_regression()
    }

    /// The numpy dtype string for the primary label output.
    pub fn label_dtype(&self) -> &'static str {
        match self {
            LabelEncoding::ContinuousBps => "float64",
            _ => "int8",
        }
    }

    /// Valid label range (inclusive) for classification encodings.
    ///
    /// Returns `None` for regression encodings (continuous values have no fixed range).
    pub fn valid_range(&self) -> Option<(i8, i8)> {
        match self {
            LabelEncoding::SignedTrend | LabelEncoding::SignedOpportunity => Some((-1, 1)),
            LabelEncoding::TripleBarrierClassIndex => Some((0, 2)),
            LabelEncoding::ContinuousBps => None,
        }
    }

    /// Number of distinct classes for classification encodings.
    ///
    /// Returns `None` for regression.
    pub fn num_classes(&self) -> Option<usize> {
        match self {
            LabelEncoding::ContinuousBps => None,
            _ => Some(3),
        }
    }

    /// Human-readable class names for classification encodings.
    ///
    /// Returns `None` for regression.
    pub fn class_names(&self) -> Option<Vec<&'static str>> {
        match self {
            LabelEncoding::SignedTrend => Some(vec!["Down", "Stable", "Up"]),
            LabelEncoding::SignedOpportunity => Some(vec!["BigDown", "NoOpportunity", "BigUp"]),
            LabelEncoding::TripleBarrierClassIndex => {
                Some(vec!["StopLoss", "Timeout", "ProfitTarget"])
            }
            LabelEncoding::ContinuousBps => None,
        }
    }

    /// Strategy name for error messages and logging.
    pub fn strategy_name(&self) -> &'static str {
        match self {
            LabelEncoding::SignedTrend => "TLOB",
            LabelEncoding::SignedOpportunity => "Opportunity",
            LabelEncoding::TripleBarrierClassIndex => "Triple Barrier",
            LabelEncoding::ContinuousBps => "Regression",
        }
    }

    /// Human-readable description of the expected range for error messages.
    pub fn expected_range_description(&self) -> String {
        match self {
            LabelEncoding::ContinuousBps => "continuous f64 (basis points)".to_string(),
            _ => {
                let (min, _max) = self.valid_range().unwrap();
                let names = self.class_names().unwrap();
                let mapping: Vec<String> = names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| format!("{}={}", min + i as i8, name))
                    .collect();
                format!("{{{}}}", mapping.join(", "))
            }
        }
    }
}

// ============================================================================
// Normalization Types
// ============================================================================

/// Normalization strategy used for feature export.
///
/// This is included in metadata so consumers know how data was normalized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationStrategy {
    /// No normalization applied (raw values)
    None,
    /// Per-feature z-score: (x - mean) / std
    PerFeatureZScore,
    /// Market-structure preserving z-score:
    /// - Prices: shared mean/std per level (ask_L and bid_L)
    /// - Sizes: independent mean/std
    #[default]
    MarketStructureZScore,
    /// Global z-score: all features share same mean/std
    GlobalZScore,
    /// Bilinear: (price - mid_price) / (k * tick_size)
    Bilinear,
}

impl std::fmt::Display for NormalizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::PerFeatureZScore => write!(f, "per_feature_zscore"),
            Self::MarketStructureZScore => write!(f, "market_structure_zscore"),
            Self::GlobalZScore => write!(f, "global_zscore"),
            Self::Bilinear => write!(f, "bilinear"),
        }
    }
}

/// Normalization parameters for a single day export.
///
/// Contains mean/std values for each feature group, enabling:
/// - Python-side validation of normalization
/// - Denormalization for interpretability
/// - Transfer to inference pipeline
///
/// # Feature Layout
///
/// Standard 40-feature LOB layout:
/// - `price_params[0..10]`: Ask prices (levels 1-10)
/// - `size_params[0..10]`: Ask sizes (levels 1-10)
/// - `price_params[0..10]`: Also used for Bid prices (shared stats)
/// - `size_params[10..20]`: Bid sizes (levels 1-10)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Normalization strategy applied
    pub strategy: NormalizationStrategy,

    /// Whether normalization was actually applied to the exported data.
    /// When false, the stats below are placeholders (means=0, stds=1)
    /// and Python-side normalization is expected.
    /// Part of the normalization boundary contract (pipeline_contract.toml).
    #[serde(default)]
    pub normalization_applied: bool,

    /// Mean values for price features (per-level, shared ask/bid)
    /// Length: 10 (one per LOB level)
    pub price_means: Vec<f64>,

    /// Std values for price features (per-level, shared ask/bid)
    /// Length: 10 (one per LOB level)
    pub price_stds: Vec<f64>,

    /// Mean values for size features (independent per feature)
    /// Length: 20 (10 ask sizes + 10 bid sizes)
    pub size_means: Vec<f64>,

    /// Std values for size features (independent per feature)
    /// Length: 20 (10 ask sizes + 10 bid sizes)
    pub size_stds: Vec<f64>,

    /// Number of samples used to compute statistics
    pub sample_count: usize,

    /// Feature layout description for validation
    pub feature_layout: String,

    /// Number of LOB levels
    pub levels: usize,
}

impl NormalizationParams {
    /// Create new normalization params with market-structure preserving stats
    pub fn new(
        price_means: Vec<f64>,
        price_stds: Vec<f64>,
        size_means: Vec<f64>,
        size_stds: Vec<f64>,
        sample_count: usize,
        levels: usize,
    ) -> Self {
        let strategy = NormalizationStrategy::MarketStructureZScore;
        let normalization_applied = strategy != NormalizationStrategy::None;
        Self {
            strategy,
            normalization_applied,
            price_means,
            price_stds,
            size_means,
            size_stds,
            sample_count,
            feature_layout: format!(
                "ask_prices_{}_ask_sizes_{}_bid_prices_{}_bid_sizes_{}",
                levels, levels, levels, levels
            ),
            levels,
        }
    }

    /// Save normalization params to JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        serde_json::to_writer_pretty(file, self).map_err(|e| {
            std::io::Error::other(format!("Failed to write normalization params: {e}"))
        })?;
        Ok(())
    }

    /// Load normalization params from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let params: Self = serde_json::from_reader(file).map_err(|e| {
            std::io::Error::other(format!("Failed to read normalization params: {e}"))
        })?;
        Ok(params)
    }
}

// ============================================================================
// LabelingResult - Common intermediary for all labeling strategies
// ============================================================================

/// Unified output from any labeling strategy, consumed by the common export pipeline.
///
/// This struct eliminates ~1000 lines of duplicated code by decoupling label
/// generation (strategy-specific) from the export pipeline (shared).
///
/// At least one of `classification_labels` or `regression_labels` must be `Some`.
pub(crate) struct LabelingResult {
    /// Label indices used for alignment (snapshot indices with valid labels).
    /// Must have the same length as whichever label matrix is present.
    pub label_indices: Vec<usize>,
    /// Classification label matrix: each row has discrete labels for all horizons.
    /// For single-horizon: each row is `vec![single_label]`.
    /// `None` for pure regression strategies.
    pub classification_labels: Option<Vec<Vec<i8>>>,
    /// Regression label matrix: continuous forward returns in bps (float64).
    /// Each row has returns for all horizons: `[ret_h1, ret_h2, ...]`.
    /// `None` for classification-only strategies.
    pub regression_labels: Option<Vec<Vec<f64>>>,
    /// Label encoding contract for validation
    pub encoding: LabelEncoding,
    /// Label distribution for metadata/reporting
    pub distribution: HashMap<String, usize>,
    /// Strategy name for metadata JSON ("tlob", "opportunity", "triple_barrier", "regression")
    pub strategy_name: &'static str,
    /// Strategy-specific metadata block for the "labeling" field in metadata JSON
    pub strategy_metadata: serde_json::Value,
    /// Whether to export labels as 2D (multi-horizon) vs 1D (single-horizon)
    pub is_multi_horizon: bool,
    /// Content for `{day}_horizons.json` (None for single-horizon TLOB)
    pub horizons_config: Option<serde_json::Value>,
}

// ============================================================================
// Export Types
// ============================================================================

/// Result of exporting a single day with aligned sequences and labels
#[derive(Debug)]
pub struct AlignedDayExport {
    /// Day identifier
    pub day: String,

    /// Number of sequences exported
    pub n_sequences: usize,

    /// Sequence dimensions: (window_size, n_features)
    pub seq_shape: (usize, usize),

    /// Label distribution
    pub label_distribution: HashMap<String, usize>,

    /// Messages processed
    pub messages_processed: usize,

    /// Export path
    pub export_path: PathBuf,

    /// VERIFICATION: Total features extracted (snapshots generated)
    pub features_extracted: usize,

    /// VERIFICATION: Buffer size used
    pub buffer_size: usize,

    /// VERIFICATION: Total sequences generated before alignment
    pub sequences_generated: usize,

    /// VERIFICATION: Sequences dropped during alignment
    pub sequences_dropped: usize,
}
