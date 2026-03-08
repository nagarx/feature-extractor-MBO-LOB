//! Normalization configuration for per-feature-group strategies.
//!
//! Different research papers require different normalization approaches.
//! This module provides presets for TLOB, DeepLOB, LOBench, and FI-2010,
//! plus a builder pattern for custom configurations.

use serde::{Deserialize, Serialize};

/// Normalization strategy for a feature group.
///
/// Different research papers require different normalization approaches.
/// This enum provides flexibility to configure the exact preprocessing
/// needed for each model architecture.
///
/// # Research Paper Requirements
///
/// | Paper | LOB Prices | LOB Sizes | Derived | MBO | Signals |
/// |-------|------------|-----------|---------|-----|---------|
/// | **TLOB** | None (BiN handles) | None | None | None | None |
/// | **DeepLOB** | GlobalZScore | GlobalZScore | N/A | N/A | N/A |
/// | **LOBench** | GlobalZScore | GlobalZScore | GlobalZScore | GlobalZScore | None |
/// | **FI-2010** | ZScore | ZScore | ZScore | N/A | N/A |
///
/// # TOML Configuration Examples
///
/// ## Raw export for TLOB (model uses BiN internally)
/// ```toml
/// [normalization]
/// lob_prices = "none"
/// lob_sizes = "none"
/// derived = "none"
/// mbo = "none"
/// signals = "none"
/// ```
///
/// ## Standard ML preprocessing
/// ```toml
/// [normalization]
/// lob_prices = "zscore"
/// lob_sizes = "zscore"
/// derived = "zscore"
/// mbo = "zscore"
/// signals = "none"  # Categoricals should never be normalized
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FeatureNormStrategy {
    /// No normalization - export raw values.
    ///
    /// Use for models with internal normalization (e.g., TLOB with BiN).
    /// Also required for categorical features that must not be normalized.
    None,

    /// Per-feature Z-score normalization: `(x - mean) / std`
    ///
    /// Computed per feature across all samples in the export.
    /// Standard ML preprocessing for general-purpose models.
    ZScore,

    /// Global Z-score normalization (LOBench paper).
    ///
    /// All features in a snapshot share the same mean/std.
    /// Preserves relative LOB structure (ask > bid ordering).
    GlobalZScore,

    /// Market-structure preserving Z-score (current default).
    ///
    /// For LOB prices: ask_L and bid_L at each level share statistics.
    /// For LOB sizes: independent statistics per feature.
    /// Guarantees: `ask_price > bid_price` ordering preserved after normalization.
    #[default]
    MarketStructure,

    /// Percentage change relative to reference (typically mid-price).
    ///
    /// Formula: `(value - reference) / reference`
    /// Good for cross-instrument training and removing absolute price levels.
    PercentageChange,

    /// Min-max normalization to [0, 1] range.
    ///
    /// Formula: `(value - min) / (max - min)`
    /// Use when bounded output is required.
    ///
    /// **NOTE: NOT YET IMPLEMENTED** - Currently falls back to Z-score.
    /// Full implementation requires tracking min/max per feature.
    /// Use ZScore or GlobalZScore instead until this is implemented.
    MinMax,

    /// Bilinear normalization (TLOB paper style).
    ///
    /// Formula: `(price - mid_price) / (k * tick_size)`
    /// Focuses on distance from mid-price in tick units.
    ///
    /// **NOTE: NOT YET IMPLEMENTED** - Currently uses simplified approximation.
    /// Full implementation requires per-timestep mid_price access.
    /// For TLOB training, use `NormalizationConfig::raw()` to let the model's
    /// BiN layer handle normalization internally.
    Bilinear,
}

impl FeatureNormStrategy {
    /// Check if this strategy requires no transformation.
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if this strategy requires statistical computation (mean/std).
    #[inline]
    pub fn requires_statistics(&self) -> bool {
        matches!(
            self,
            Self::ZScore | Self::GlobalZScore | Self::MarketStructure
        )
    }

    /// Get human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "Raw values (no normalization)",
            Self::ZScore => "Per-feature Z-score: (x - mean) / std",
            Self::GlobalZScore => "Global Z-score: shared mean/std across all features",
            Self::MarketStructure => "Market-structure preserving (ask/bid level sharing)",
            Self::PercentageChange => "Percentage change: (x - ref) / ref",
            Self::MinMax => "Min-max scaling to [0, 1] (NOT FULLY IMPLEMENTED - uses Z-score fallback)",
            Self::Bilinear => "Bilinear: (price - mid) / (k * tick) (NOT FULLY IMPLEMENTED - uses approximation)",
        }
    }

    /// Check if this strategy is fully implemented.
    ///
    /// MinMax and Bilinear are declared but use fallback implementations.
    /// Users should prefer fully implemented strategies for production use.
    #[inline]
    pub fn is_fully_implemented(&self) -> bool {
        !matches!(self, Self::MinMax | Self::Bilinear)
    }
}

/// Per-feature-group normalization configuration.
///
/// This configuration controls how each feature group is normalized during export.
/// Different models require different preprocessing:
///
/// - **TLOB**: No pre-normalization (uses BiN as first layer)
/// - **DeepLOB**: Global Z-score for all features
/// - **LOBench**: Global Z-score (preserves LOB constraints)
/// - **Standard ML**: Per-feature Z-score
///
/// # Feature Index Layout (98-feature mode)
///
/// | Group | Indices | Count | Description |
/// |-------|---------|-------|-------------|
/// | LOB Prices | 0-9, 20-29 | 20 | Ask/Bid prices at 10 levels |
/// | LOB Sizes | 10-19, 30-39 | 20 | Ask/Bid sizes at 10 levels |
/// | Derived | 40-47 | 8 | Mid-price, spread, imbalance, etc. |
/// | MBO | 48-83 | 36 | Order flow microstructure features |
/// | Signals | 84-97 | 14 | Trading signals (includes categoricals) |
///
/// # Example
///
/// ```ignore
/// use feature_extractor::export::{NormalizationConfig, FeatureNormStrategy};
///
/// // For TLOB (raw export - model handles normalization)
/// let config = NormalizationConfig::raw();
///
/// // For DeepLOB (global z-score)
/// let config = NormalizationConfig::deeplob();
///
/// // Custom configuration
/// let config = NormalizationConfig::default()
///     .with_lob_prices(FeatureNormStrategy::GlobalZScore)
///     .with_signals(FeatureNormStrategy::None);  // Never normalize categoricals
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Strategy for LOB prices (indices 0-9, 20-29).
    ///
    /// Default: `None` (raw values for TLOB paper compatibility).
    #[serde(default = "default_norm_none")]
    pub lob_prices: FeatureNormStrategy,

    /// Strategy for LOB sizes (indices 10-19, 30-39).
    ///
    /// Default: `None` (raw values for TLOB paper compatibility).
    #[serde(default = "default_norm_none")]
    pub lob_sizes: FeatureNormStrategy,

    /// Strategy for derived features (indices 40-47).
    ///
    /// Default: `None` (features like spread, imbalance are already in reasonable ranges).
    #[serde(default = "default_norm_none")]
    pub derived: FeatureNormStrategy,

    /// Strategy for MBO features (indices 48-83).
    ///
    /// Default: `None` (many are ratios or pre-normalized).
    #[serde(default = "default_norm_none")]
    pub mbo: FeatureNormStrategy,

    /// Strategy for signal features (indices 84-97).
    ///
    /// **IMPORTANT**: Always `None` for signals containing categoricals
    /// (book_valid, time_regime, mbo_ready, schema_version).
    /// Normalizing categorical features destroys their semantics.
    ///
    /// Default: `None`
    #[serde(default = "default_norm_none")]
    pub signals: FeatureNormStrategy,

    /// Reference price strategy for percentage change normalization.
    ///
    /// When using `PercentageChange` strategy, this determines the reference:
    /// - `"mid_price"`: Use mid-price from derived features
    /// - `"first_ask"`: Use best ask price
    /// - `"first_bid"`: Use best bid price
    ///
    /// Default: `"mid_price"`
    #[serde(default = "default_reference_price")]
    pub reference_price: String,

    /// Scale factor for bilinear normalization.
    ///
    /// Used when `lob_prices = "bilinear"`.
    /// Formula: `(price - mid_price) / (scale_factor * tick_size)`
    ///
    /// Default: 50.0
    #[serde(default = "default_bilinear_scale")]
    pub bilinear_scale_factor: f64,
}

/// Default normalization strategy: None (raw values).
///
/// This is the serde default for all feature groups, ensuring that:
/// 1. Omitted TOML fields default to raw (no normalization)
/// 2. This matches TLOB paper requirements (model handles normalization via BiN)
/// 3. Users must explicitly configure normalization if needed
fn default_norm_none() -> FeatureNormStrategy {
    FeatureNormStrategy::None
}

fn default_reference_price() -> String {
    "mid_price".to_string()
}

fn default_bilinear_scale() -> f64 {
    50.0
}

impl Default for NormalizationConfig {
    /// Default: raw values (no normalization) for all feature groups.
    ///
    /// This matches TLOB paper requirements where raw LOB data is fed to the model,
    /// and the BiN layer handles normalization internally. Users who need preprocessing
    /// should explicitly use presets like `tlob_repo()`, `deeplob()`, or `lobench()`.
    fn default() -> Self {
        Self {
            lob_prices: FeatureNormStrategy::None,
            lob_sizes: FeatureNormStrategy::None,
            derived: FeatureNormStrategy::None,
            mbo: FeatureNormStrategy::None,
            signals: FeatureNormStrategy::None,
            reference_price: "mid_price".to_string(),
            bilinear_scale_factor: 50.0,
        }
    }
}

impl NormalizationConfig {
    /// Create configuration with NO normalization (raw export).
    ///
    /// Use for models with internal normalization like TLOB (BiN layer).
    /// This matches the official TLOB paper preprocessing requirements.
    ///
    /// # Research Reference
    ///
    /// TLOB paper Section 3.2:
    /// > "We apply Bilinear Normalization (BiN) as the first layer..."
    ///
    /// The model expects raw LOB data and handles normalization internally.
    pub fn raw() -> Self {
        Self {
            lob_prices: FeatureNormStrategy::None,
            lob_sizes: FeatureNormStrategy::None,
            derived: FeatureNormStrategy::None,
            mbo: FeatureNormStrategy::None,
            signals: FeatureNormStrategy::None,
            reference_price: "mid_price".to_string(),
            bilinear_scale_factor: 50.0,
        }
    }

    /// Create configuration for TLOB paper (alias for `raw()`).
    ///
    /// The TLOB paper specifies that raw LOB data should be fed to the model,
    /// which applies Bilinear Normalization (BiN) as its first layer.
    ///
    /// # Research Reference
    ///
    /// TLOB paper Section 3.2:
    /// > "We apply Bilinear Normalization (BiN) as the first layer..."
    ///
    /// # See Also
    ///
    /// - `raw()`: Identical functionality
    /// - `tlob_repo()`: Matches actual TLOB repository (pre-applies global Z-score)
    pub fn tlob_paper() -> Self {
        Self::raw()
    }

    /// Create configuration matching official TLOB repository preprocessing.
    ///
    /// The official TLOB implementation applies global Z-score to prices
    /// and sizes BEFORE the BiN layer. This differs from the paper description
    /// but matches the actual codebase.
    ///
    /// # Repository Reference
    ///
    /// `TLOB/utils/utils_data.py::z_score_orderbook()`:
    /// - All price columns normalized together (shared mean/std)
    /// - All size columns normalized together (shared mean/std)
    ///
    /// # Note
    ///
    /// Use `raw()` for pure paper implementation, or this preset
    /// for exact repository reproduction.
    pub fn tlob_repo() -> Self {
        Self {
            lob_prices: FeatureNormStrategy::GlobalZScore,
            lob_sizes: FeatureNormStrategy::GlobalZScore,
            derived: FeatureNormStrategy::None,
            mbo: FeatureNormStrategy::None,
            signals: FeatureNormStrategy::None,
            reference_price: "mid_price".to_string(),
            bilinear_scale_factor: 50.0,
        }
    }

    /// Create configuration for DeepLOB paper.
    ///
    /// DeepLOB uses per-feature Z-score normalization.
    ///
    /// # Research Reference
    ///
    /// "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
    /// Section 4.1: Data preprocessing
    pub fn deeplob() -> Self {
        Self {
            lob_prices: FeatureNormStrategy::ZScore,
            lob_sizes: FeatureNormStrategy::ZScore,
            derived: FeatureNormStrategy::None, // DeepLOB uses only 40 features
            mbo: FeatureNormStrategy::None,
            signals: FeatureNormStrategy::None,
            reference_price: "mid_price".to_string(),
            bilinear_scale_factor: 50.0,
        }
    }

    /// Create configuration for LOBench paper.
    ///
    /// LOBench uses global Z-score for all features together,
    /// which preserves LOB ordering constraints.
    ///
    /// # Research Reference
    ///
    /// "Representation Learning of Limit Order Book: A Comprehensive Study"
    pub fn lobench() -> Self {
        Self {
            lob_prices: FeatureNormStrategy::GlobalZScore,
            lob_sizes: FeatureNormStrategy::GlobalZScore,
            derived: FeatureNormStrategy::GlobalZScore,
            mbo: FeatureNormStrategy::GlobalZScore,
            signals: FeatureNormStrategy::None, // Categoricals never normalized
            reference_price: "mid_price".to_string(),
            bilinear_scale_factor: 50.0,
        }
    }

    /// Create configuration for FI-2010 benchmark.
    ///
    /// FI-2010 uses per-feature Z-score normalization.
    pub fn fi2010() -> Self {
        Self {
            lob_prices: FeatureNormStrategy::ZScore,
            lob_sizes: FeatureNormStrategy::ZScore,
            derived: FeatureNormStrategy::ZScore,
            mbo: FeatureNormStrategy::None, // FI-2010 doesn't use MBO
            signals: FeatureNormStrategy::None,
            reference_price: "mid_price".to_string(),
            bilinear_scale_factor: 50.0,
        }
    }

    // Builder pattern methods

    /// Set LOB prices normalization strategy.
    pub fn with_lob_prices(mut self, strategy: FeatureNormStrategy) -> Self {
        self.lob_prices = strategy;
        self
    }

    /// Set LOB sizes normalization strategy.
    pub fn with_lob_sizes(mut self, strategy: FeatureNormStrategy) -> Self {
        self.lob_sizes = strategy;
        self
    }

    /// Set derived features normalization strategy.
    pub fn with_derived(mut self, strategy: FeatureNormStrategy) -> Self {
        self.derived = strategy;
        self
    }

    /// Set MBO features normalization strategy.
    pub fn with_mbo(mut self, strategy: FeatureNormStrategy) -> Self {
        self.mbo = strategy;
        self
    }

    /// Set signal features normalization strategy.
    ///
    /// **Warning**: Setting this to anything other than `None` will
    /// normalize categorical features which destroys their semantics.
    pub fn with_signals(mut self, strategy: FeatureNormStrategy) -> Self {
        self.signals = strategy;
        self
    }

    /// Set bilinear scale factor.
    pub fn with_bilinear_scale(mut self, scale: f64) -> Self {
        self.bilinear_scale_factor = scale;
        self
    }

    /// Check if any normalization will be applied.
    pub fn any_normalization(&self) -> bool {
        !self.lob_prices.is_none()
            || !self.lob_sizes.is_none()
            || !self.derived.is_none()
            || !self.mbo.is_none()
            || !self.signals.is_none()
    }

    /// Check if all feature groups use the same strategy.
    pub fn is_uniform(&self) -> bool {
        self.lob_prices == self.lob_sizes
            && self.lob_sizes == self.derived
            && self.derived == self.mbo
            && self.mbo == self.signals
    }

    /// Validate the normalization configuration.
    pub fn validate(&self) -> Result<(), String> {
        // Warning: normalizing signals is usually wrong
        if !self.signals.is_none() {
            eprintln!(
                "⚠️  WARNING: Normalizing signals is not recommended - \
                categorical features (book_valid, time_regime, mbo_ready, schema_version) \
                will lose their semantics"
            );
        }

        // Validate bilinear scale factor
        if self.bilinear_scale_factor <= 0.0 {
            return Err("bilinear_scale_factor must be > 0".to_string());
        }

        // Validate reference price
        let valid_refs = ["mid_price", "first_ask", "first_bid"];
        if !valid_refs.contains(&self.reference_price.as_str()) {
            return Err(format!(
                "reference_price must be one of {:?}, got '{}'",
                valid_refs, self.reference_price
            ));
        }

        Ok(())
    }

    /// Get human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "LOB prices: {}, LOB sizes: {}, Derived: {}, MBO: {}, Signals: {}",
            self.lob_prices.description(),
            self.lob_sizes.description(),
            self.derived.description(),
            self.mbo.description(),
            self.signals.description()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_config_default() {
        let config = NormalizationConfig::default();
        // Default is raw (no normalization) for TLOB paper compatibility
        assert_eq!(config.lob_prices, FeatureNormStrategy::None);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::None);
        assert_eq!(config.derived, FeatureNormStrategy::None);
        assert_eq!(config.mbo, FeatureNormStrategy::None);
        assert_eq!(config.signals, FeatureNormStrategy::None);
        assert_eq!(config.reference_price, "mid_price");
        assert!((config.bilinear_scale_factor - 50.0).abs() < 1e-10);
        // Default matches raw() preset
        assert!(!config.any_normalization());
    }

    #[test]
    fn test_normalization_config_raw_preset() {
        let config = NormalizationConfig::raw();
        assert_eq!(config.lob_prices, FeatureNormStrategy::None);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::None);
        assert_eq!(config.derived, FeatureNormStrategy::None);
        assert_eq!(config.mbo, FeatureNormStrategy::None);
        assert_eq!(config.signals, FeatureNormStrategy::None);
        assert!(!config.any_normalization(), "raw() should not apply any normalization");
    }

    #[test]
    fn test_normalization_config_tlob_paper_preset() {
        let config = NormalizationConfig::tlob_paper();
        // TLOB paper: raw data fed to model, BiN handles normalization
        assert_eq!(config.lob_prices, FeatureNormStrategy::None);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::None);
        assert!(!config.any_normalization());
    }

    #[test]
    fn test_normalization_config_tlob_repo_preset() {
        let config = NormalizationConfig::tlob_repo();
        // TLOB repo: global z-score BEFORE BiN
        assert_eq!(config.lob_prices, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.derived, FeatureNormStrategy::None);
        assert_eq!(config.mbo, FeatureNormStrategy::None);
        assert_eq!(config.signals, FeatureNormStrategy::None);
        assert!(config.any_normalization());
    }

    #[test]
    fn test_normalization_config_deeplob_preset() {
        let config = NormalizationConfig::deeplob();
        assert_eq!(config.lob_prices, FeatureNormStrategy::ZScore);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::ZScore);
        assert_eq!(config.derived, FeatureNormStrategy::None);
        assert!(config.any_normalization());
    }

    #[test]
    fn test_normalization_config_lobench_preset() {
        let config = NormalizationConfig::lobench();
        assert_eq!(config.lob_prices, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.derived, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.mbo, FeatureNormStrategy::GlobalZScore);
        // Signals should NEVER be normalized (categoricals)
        assert_eq!(config.signals, FeatureNormStrategy::None);
    }

    #[test]
    fn test_normalization_config_fi2010_preset() {
        let config = NormalizationConfig::fi2010();
        assert_eq!(config.lob_prices, FeatureNormStrategy::ZScore);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::ZScore);
        assert_eq!(config.derived, FeatureNormStrategy::ZScore);
        // FI-2010 doesn't use MBO features
        assert_eq!(config.mbo, FeatureNormStrategy::None);
    }

    #[test]
    fn test_normalization_config_builder_pattern() {
        let config = NormalizationConfig::default()
            .with_lob_prices(FeatureNormStrategy::GlobalZScore)
            .with_lob_sizes(FeatureNormStrategy::PercentageChange)
            .with_derived(FeatureNormStrategy::ZScore)
            .with_mbo(FeatureNormStrategy::MinMax)
            .with_signals(FeatureNormStrategy::None)
            .with_bilinear_scale(100.0);

        assert_eq!(config.lob_prices, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::PercentageChange);
        assert_eq!(config.derived, FeatureNormStrategy::ZScore);
        assert_eq!(config.mbo, FeatureNormStrategy::MinMax);
        assert_eq!(config.signals, FeatureNormStrategy::None);
        assert!((config.bilinear_scale_factor - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_config_any_normalization_true() {
        let config = NormalizationConfig::raw()
            .with_derived(FeatureNormStrategy::ZScore);
        assert!(config.any_normalization());
    }

    #[test]
    fn test_normalization_config_any_normalization_false() {
        let config = NormalizationConfig::raw();
        assert!(!config.any_normalization());
    }

    #[test]
    fn test_normalization_config_is_uniform_true() {
        let config = NormalizationConfig::raw(); // All None
        assert!(config.is_uniform());
    }

    #[test]
    fn test_normalization_config_is_uniform_false() {
        let config = NormalizationConfig::deeplob(); // prices/sizes are ZScore, others are None
        assert!(!config.is_uniform());
    }

    #[test]
    fn test_normalization_config_validation_valid() {
        let config = NormalizationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_normalization_config_validation_invalid_bilinear_scale() {
        let mut config = NormalizationConfig::default();
        config.bilinear_scale_factor = 0.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("bilinear_scale_factor must be > 0"));
    }

    #[test]
    fn test_normalization_config_validation_invalid_reference_price() {
        let mut config = NormalizationConfig::default();
        config.reference_price = "invalid_reference".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("reference_price must be one of"));
    }

    #[test]
    fn test_normalization_config_validation_valid_reference_prices() {
        for ref_price in &["mid_price", "first_ask", "first_bid"] {
            let mut config = NormalizationConfig::default();
            config.reference_price = ref_price.to_string();
            assert!(
                config.validate().is_ok(),
                "reference_price '{}' should be valid",
                ref_price
            );
        }
    }

    #[test]
    fn test_normalization_config_summary() {
        let config = NormalizationConfig::tlob_repo();
        let summary = config.summary();
        assert!(summary.contains("Global Z-score"));
        assert!(summary.contains("Raw values"));
    }

    #[test]
    fn test_feature_norm_strategy_is_none() {
        assert!(FeatureNormStrategy::None.is_none());
        assert!(!FeatureNormStrategy::ZScore.is_none());
        assert!(!FeatureNormStrategy::GlobalZScore.is_none());
    }

    #[test]
    fn test_feature_norm_strategy_requires_statistics() {
        assert!(FeatureNormStrategy::ZScore.requires_statistics());
        assert!(FeatureNormStrategy::GlobalZScore.requires_statistics());
        assert!(FeatureNormStrategy::MarketStructure.requires_statistics());
        assert!(!FeatureNormStrategy::None.requires_statistics());
        assert!(!FeatureNormStrategy::PercentageChange.requires_statistics());
        assert!(!FeatureNormStrategy::MinMax.requires_statistics());
        assert!(!FeatureNormStrategy::Bilinear.requires_statistics());
    }

    #[test]
    fn test_feature_norm_strategy_is_fully_implemented() {
        // Fully implemented strategies
        assert!(FeatureNormStrategy::None.is_fully_implemented());
        assert!(FeatureNormStrategy::ZScore.is_fully_implemented());
        assert!(FeatureNormStrategy::GlobalZScore.is_fully_implemented());
        assert!(FeatureNormStrategy::MarketStructure.is_fully_implemented());
        assert!(FeatureNormStrategy::PercentageChange.is_fully_implemented());

        // Strategies with fallback implementations (not fully implemented)
        assert!(!FeatureNormStrategy::MinMax.is_fully_implemented());
        assert!(!FeatureNormStrategy::Bilinear.is_fully_implemented());
    }

    #[test]
    fn test_feature_norm_strategy_description() {
        let strat = FeatureNormStrategy::ZScore;
        assert!(strat.description().contains("Z-score"));

        let strat = FeatureNormStrategy::None;
        assert!(strat.description().contains("Raw"));
    }

    #[test]
    fn test_normalization_config_toml_serialization() {
        let config = NormalizationConfig::tlob_repo();
        let toml_str = toml::to_string(&config).unwrap();

        // Verify key fields are present
        assert!(toml_str.contains("lob_prices"));
        assert!(toml_str.contains("lob_sizes"));
        assert!(toml_str.contains("global_z_score"));
    }

    #[test]
    fn test_normalization_config_toml_deserialization() {
        let toml_str = r#"
lob_prices = "z_score"
lob_sizes = "global_z_score"
derived = "none"
mbo = "percentage_change"
signals = "none"
reference_price = "first_ask"
bilinear_scale_factor = 75.0
"#;
        let config: NormalizationConfig = toml::from_str(toml_str).unwrap();

        assert_eq!(config.lob_prices, FeatureNormStrategy::ZScore);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::GlobalZScore);
        assert_eq!(config.derived, FeatureNormStrategy::None);
        assert_eq!(config.mbo, FeatureNormStrategy::PercentageChange);
        assert_eq!(config.signals, FeatureNormStrategy::None);
        assert_eq!(config.reference_price, "first_ask");
        assert!((config.bilinear_scale_factor - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_config_toml_roundtrip() {
        let original = NormalizationConfig::default()
            .with_lob_prices(FeatureNormStrategy::Bilinear)
            .with_derived(FeatureNormStrategy::MinMax)
            .with_bilinear_scale(42.0);

        let toml_str = toml::to_string(&original).unwrap();
        let loaded: NormalizationConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(original.lob_prices, loaded.lob_prices);
        assert_eq!(original.lob_sizes, loaded.lob_sizes);
        assert_eq!(original.derived, loaded.derived);
        assert_eq!(original.mbo, loaded.mbo);
        assert_eq!(original.signals, loaded.signals);
        assert!((original.bilinear_scale_factor - loaded.bilinear_scale_factor).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_config_serde_field_defaults() {
        // When TOML fields are omitted, serde uses custom default_norm_none() = None.
        // This ensures omitted config = raw (no normalization) = TLOB paper compatible.
        let minimal_toml = ""; // Empty config should use field defaults
        let config: NormalizationConfig = toml::from_str(minimal_toml).unwrap();

        // All fields default to None (raw values)
        assert_eq!(config.lob_prices, FeatureNormStrategy::None);
        assert_eq!(config.lob_sizes, FeatureNormStrategy::None);
        assert_eq!(config.derived, FeatureNormStrategy::None);
        assert_eq!(config.mbo, FeatureNormStrategy::None);
        assert_eq!(config.signals, FeatureNormStrategy::None);

        // Serde defaults match NormalizationConfig::default() exactly
        let default_config = NormalizationConfig::default();
        assert_eq!(config.lob_prices, default_config.lob_prices);
        assert_eq!(config.lob_sizes, default_config.lob_sizes);
        assert_eq!(config.derived, default_config.derived);
        assert_eq!(config.mbo, default_config.mbo);
        assert_eq!(config.signals, default_config.signals);
    }

    #[test]
    fn test_normalization_config_all_strategies_serializable() {
        // Ensure all FeatureNormStrategy variants can be serialized
        let strategies = vec![
            FeatureNormStrategy::None,
            FeatureNormStrategy::ZScore,
            FeatureNormStrategy::GlobalZScore,
            FeatureNormStrategy::MarketStructure,
            FeatureNormStrategy::PercentageChange,
            FeatureNormStrategy::MinMax,
            FeatureNormStrategy::Bilinear,
        ];

        for strat in strategies {
            let config = NormalizationConfig::raw().with_lob_prices(strat.clone());
            let toml_str = toml::to_string(&config)
                .expect(&format!("Should serialize strategy {:?}", strat));
            let _loaded: NormalizationConfig = toml::from_str(&toml_str)
                .expect(&format!("Should deserialize strategy {:?}", strat));
        }
    }
}
