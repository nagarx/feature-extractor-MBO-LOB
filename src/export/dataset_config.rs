//! Dataset Export Configuration
//!
//! Configuration-driven, symbol-agnostic export system for feature datasets.
//! This module provides the infrastructure to export feature datasets with
//! any combination of features, for any symbol, without hard-coding paths or parameters.
//!
//! # Design Philosophy
//!
//! - **Model-agnostic**: Exports features, not trading decisions
//! - **Symbol-agnostic**: Works for any instrument (NVDA, AAPL, etc.)
//! - **Configuration-driven**: All parameters via config, no hard-coding
//! - **Flexible feature sets**: Support 40, 48, 76, 84, or 98 features
//! - **Serializable**: TOML/JSON for experiment reproducibility
//! - **Validated**: Catches configuration errors early
//!
//! # Feature Set Reference
//!
//! | Configuration | Feature Count | Description |
//! |--------------|---------------|-------------|
//! | Raw LOB | 40 | 10 levels × 4 (ask_price, ask_size, bid_price, bid_size) |
//! | + Derived | 48 | + 8 derived features |
//! | + MBO | 76 | + 36 MBO microstructure features |
//! | + Derived + MBO | 84 | Full baseline |
//! | + Signals | 98 | + 14 trading signals (Cont et al. 2014) |
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::export::DatasetConfig;
//!
//! // Load configuration from TOML
//! let config = DatasetConfig::load_toml("configs/nvda_98feat.toml")?;
//!
//! // Validate before use
//! config.validate()?;
//!
//! // Get the pipeline configuration
//! let pipeline_config = config.to_pipeline_config();
//! ```

use crate::config::{PipelineConfig, SamplingConfig, SamplingStrategy};
use crate::features::FeatureConfig;
use crate::labeling::LabelConfig;
use crate::sequence_builder::SequenceConfig;
use chrono::Datelike;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

// ============================================================================
// Symbol Configuration
// ============================================================================

/// Configuration for a trading symbol/instrument.
///
/// Provides symbol-specific parameters like exchange, naming conventions,
/// and file path patterns. This decouples the pipeline from any specific symbol.
///
/// # Example
///
/// ```ignore
/// let nvda = SymbolConfig {
///     name: "NVDA".to_string(),
///     exchange: "XNAS".to_string(),
///     filename_pattern: "xnas-itch-{date}.mbo.dbn.zst".to_string(),
///     tick_size: 0.01,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolConfig {
    /// Symbol name (e.g., "NVDA", "AAPL", "GOOGL")
    pub name: String,

    /// Exchange code (e.g., "XNAS" for NASDAQ)
    pub exchange: String,

    /// Filename pattern for data files.
    ///
    /// Use `{date}` as placeholder for YYYYMMDD date format.
    /// Examples:
    /// - `"xnas-itch-{date}.mbo.dbn.zst"` → `xnas-itch-20250203.mbo.dbn.zst`
    /// - `"{symbol}_{date}.dbn.zst"` → `NVDA_20250203.dbn.zst`
    pub filename_pattern: String,

    /// Minimum tick size for price normalization (e.g., 0.01 for US stocks)
    #[serde(default = "default_tick_size")]
    pub tick_size: f64,
}

fn default_tick_size() -> f64 {
    0.01
}

impl SymbolConfig {
    /// Create a new symbol configuration.
    pub fn new(name: &str, exchange: &str, filename_pattern: &str) -> Self {
        Self {
            name: name.to_string(),
            exchange: exchange.to_string(),
            filename_pattern: filename_pattern.to_string(),
            tick_size: 0.01,
        }
    }

    /// Create NASDAQ symbol configuration with standard naming.
    ///
    /// Uses pattern: `xnas-itch-{date}.mbo.dbn.zst`
    pub fn nasdaq(name: &str) -> Self {
        Self {
            name: name.to_string(),
            exchange: "XNAS".to_string(),
            filename_pattern: "xnas-itch-{date}.mbo.dbn.zst".to_string(),
            tick_size: 0.01,
        }
    }

    /// Generate filename for a specific date.
    ///
    /// # Arguments
    ///
    /// * `date` - Date in YYYY-MM-DD format (e.g., "2025-02-03")
    ///
    /// # Returns
    ///
    /// Filename with date substituted (e.g., "xnas-itch-20250203.mbo.dbn.zst")
    pub fn filename_for_date(&self, date: &str) -> String {
        // Convert YYYY-MM-DD to YYYYMMDD
        let date_compact = date.replace('-', "");
        self.filename_pattern
            .replace("{date}", &date_compact)
            .replace("{symbol}", &self.name)
    }

    /// Validate the symbol configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Symbol name cannot be empty".to_string());
        }
        if self.exchange.is_empty() {
            return Err("Exchange cannot be empty".to_string());
        }
        if !self.filename_pattern.contains("{date}") {
            return Err("Filename pattern must contain {date} placeholder".to_string());
        }
        if self.tick_size <= 0.0 {
            return Err("Tick size must be positive".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// Data Path Configuration
// ============================================================================

/// Configuration for input and output data paths.
///
/// Supports both compressed (.dbn.zst) and decompressed (.dbn) files,
/// with optional hot store for faster processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPathConfig {
    /// Directory containing raw MBO data files.
    pub input_dir: PathBuf,

    /// Optional hot store directory with decompressed files.
    ///
    /// When set, the processor prefers decompressed files for ~30% faster processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hot_store_dir: Option<PathBuf>,

    /// Output directory for exported datasets.
    pub output_dir: PathBuf,
}

impl DataPathConfig {
    /// Create a new data path configuration.
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(input_dir: P1, output_dir: P2) -> Self {
        Self {
            input_dir: input_dir.as_ref().to_path_buf(),
            hot_store_dir: None,
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }

    /// Set hot store directory for faster processing.
    pub fn with_hot_store<P: AsRef<Path>>(mut self, hot_store_dir: P) -> Self {
        self.hot_store_dir = Some(hot_store_dir.as_ref().to_path_buf());
        self
    }

    /// Validate paths exist (or can be created for output).
    pub fn validate(&self) -> Result<(), String> {
        if !self.input_dir.exists() {
            return Err(format!(
                "Input directory does not exist: {}",
                self.input_dir.display()
            ));
        }

        if let Some(ref hot_store) = self.hot_store_dir {
            if !hot_store.exists() {
                return Err(format!(
                    "Hot store directory does not exist: {}",
                    hot_store.display()
                ));
            }
        }

        // Output directory will be created if needed
        Ok(())
    }

    /// Validate paths exist (lenient - allows creation of output dir).
    pub fn validate_lenient(&self) -> Result<(), String> {
        if !self.input_dir.exists() {
            return Err(format!(
                "Input directory does not exist: {}",
                self.input_dir.display()
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Date Range Configuration
// ============================================================================

/// Configuration for date range selection.
///
/// Supports both explicit date lists and date ranges.
/// For ranges, weekends are automatically excluded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRangeConfig {
    /// Start date (inclusive), format: YYYY-MM-DD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_date: Option<String>,

    /// End date (inclusive), format: YYYY-MM-DD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_date: Option<String>,

    /// Explicit list of dates (overrides start/end if provided).
    ///
    /// Format: YYYY-MM-DD for each date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_dates: Option<Vec<String>>,

    /// Dates to exclude (e.g., holidays).
    ///
    /// Format: YYYY-MM-DD for each date.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude_dates: Vec<String>,
}

impl Default for DateRangeConfig {
    fn default() -> Self {
        Self {
            start_date: None,
            end_date: None,
            explicit_dates: None,
            exclude_dates: Vec::new(),
        }
    }
}

impl DateRangeConfig {
    /// Create from explicit list of dates.
    pub fn from_dates(dates: Vec<String>) -> Self {
        Self {
            start_date: None,
            end_date: None,
            explicit_dates: Some(dates),
            exclude_dates: Vec::new(),
        }
    }

    /// Create from date range.
    pub fn from_range(start: &str, end: &str) -> Self {
        Self {
            start_date: Some(start.to_string()),
            end_date: Some(end.to_string()),
            explicit_dates: None,
            exclude_dates: Vec::new(),
        }
    }

    /// Add dates to exclude.
    pub fn with_exclusions(mut self, dates: Vec<String>) -> Self {
        self.exclude_dates = dates;
        self
    }

    /// Get the list of dates to process.
    ///
    /// If `explicit_dates` is provided, returns those.
    /// Otherwise, generates dates from `start_date` to `end_date`,
    /// excluding weekends and `exclude_dates`.
    pub fn get_dates(&self) -> Result<Vec<String>, String> {
        if let Some(ref explicit) = self.explicit_dates {
            // Filter out excluded dates
            let dates: Vec<String> = explicit
                .iter()
                .filter(|d| !self.exclude_dates.contains(d))
                .cloned()
                .collect();
            return Ok(dates);
        }

        // Generate from range
        let start = self
            .start_date
            .as_ref()
            .ok_or("Either explicit_dates or start_date must be provided")?;
        let end = self
            .end_date
            .as_ref()
            .ok_or("Either explicit_dates or end_date must be provided")?;

        // Parse dates
        let start_date = chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
            .map_err(|e| format!("Invalid start_date '{}': {}", start, e))?;
        let end_date = chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
            .map_err(|e| format!("Invalid end_date '{}': {}", end, e))?;

        if start_date > end_date {
            return Err(format!(
                "start_date ({}) must be before end_date ({})",
                start, end
            ));
        }

        // Generate all weekdays in range
        let mut dates = Vec::new();
        let mut current = start_date;

        while current <= end_date {
            // Skip weekends (Saturday = 6, Sunday = 0 in chrono)
            let weekday = current.weekday();
            if weekday != chrono::Weekday::Sat && weekday != chrono::Weekday::Sun {
                let date_str = current.format("%Y-%m-%d").to_string();
                if !self.exclude_dates.contains(&date_str) {
                    dates.push(date_str);
                }
            }
            current = current.succ_opt().unwrap_or(current);
        }

        Ok(dates)
    }

    /// Validate the date configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.explicit_dates.is_none()
            && (self.start_date.is_none() || self.end_date.is_none())
        {
            return Err(
                "Either explicit_dates or both start_date and end_date must be provided"
                    .to_string(),
            );
        }

        // Validate date formats
        if let Some(ref start) = self.start_date {
            chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
                .map_err(|e| format!("Invalid start_date '{}': {}", start, e))?;
        }

        if let Some(ref end) = self.end_date {
            chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
                .map_err(|e| format!("Invalid end_date '{}': {}", end, e))?;
        }

        for date in &self.exclude_dates {
            chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
                .map_err(|e| format!("Invalid exclude_date '{}': {}", date, e))?;
        }

        if let Some(ref explicit) = self.explicit_dates {
            for date in explicit {
                chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
                    .map_err(|e| format!("Invalid explicit date '{}': {}", date, e))?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Feature Set Configuration
// ============================================================================

/// Configuration for which features to extract.
///
/// This is the model-agnostic feature selection layer.
/// The exported features can be used by any downstream model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSetConfig {
    /// Number of LOB levels (typically 10)
    #[serde(default = "default_lob_levels")]
    pub lob_levels: usize,

    /// Include derived features (+8 features: mid-price, spread, etc.)
    #[serde(default)]
    pub include_derived: bool,

    /// Include MBO microstructure features (+36 features)
    #[serde(default)]
    pub include_mbo: bool,

    /// Include trading signals (+14 features: OFI, microprice, etc.)
    ///
    /// **Note**: Requires `include_derived` and `include_mbo` to be enabled.
    #[serde(default)]
    pub include_signals: bool,

    /// MBO window size (number of messages for rolling statistics)
    #[serde(default = "default_mbo_window_size")]
    pub mbo_window_size: usize,

    /// Enable queue position tracking for MBO features.
    ///
    /// When enabled, queue-related features (avg_queue_position, queue_size_ahead)
    /// will contain actual computed values. When disabled, they return 0.0.
    ///
    /// **Note**: Requires `include_mbo` to be enabled. Disabled by default.
    #[serde(default)]
    pub include_queue_tracking: bool,
}

fn default_lob_levels() -> usize {
    10
}

fn default_mbo_window_size() -> usize {
    1000
}

impl Default for FeatureSetConfig {
    fn default() -> Self {
        Self {
            lob_levels: 10,
            include_derived: false,
            include_mbo: false,
            include_signals: false,
            mbo_window_size: 1000,
            include_queue_tracking: false,
        }
    }
}

impl FeatureSetConfig {
    /// Create configuration for full 98-feature mode.
    ///
    /// Enables all features: LOB + Derived + MBO + Signals.
    pub fn full() -> Self {
        Self {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: true,
            mbo_window_size: 1000,
            include_queue_tracking: false, // Disabled by default for performance
        }
    }

    /// Create configuration for 84-feature baseline.
    ///
    /// Enables: LOB + Derived + MBO (no signals).
    pub fn baseline() -> Self {
        Self {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            include_signals: false,
            mbo_window_size: 1000,
            include_queue_tracking: false,
        }
    }

    /// Create configuration for raw LOB only (40 features).
    pub fn raw_lob() -> Self {
        Self::default()
    }

    /// Compute the total number of features.
    pub fn feature_count(&self) -> usize {
        let mut count = self.lob_levels * 4; // Raw LOB

        if self.include_derived {
            count += 8; // Derived features
        }

        if self.include_mbo {
            count += 36; // MBO features
        }

        if self.include_signals {
            count += 14; // Trading signals
        }

        count
    }

    /// Convert to internal FeatureConfig.
    ///
    /// # Arguments
    /// * `tick_size` - The tick size from SymbolConfig (must be propagated from outer config)
    pub fn to_feature_config(&self, tick_size: f64) -> FeatureConfig {
        FeatureConfig {
            lob_levels: self.lob_levels,
            tick_size,
            include_derived: self.include_derived,
            include_mbo: self.include_mbo,
            mbo_window_size: self.mbo_window_size,
            include_signals: self.include_signals,
            include_queue_tracking: self.include_queue_tracking,
        }
    }

    /// Validate the feature configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.lob_levels == 0 {
            return Err("lob_levels must be > 0".to_string());
        }
        if self.lob_levels > 50 {
            return Err("lob_levels must be <= 50 (practical limit)".to_string());
        }

        if self.include_signals && !self.include_derived {
            return Err(
                "include_signals requires include_derived to be enabled".to_string()
            );
        }

        if self.include_signals && !self.include_mbo {
            return Err("include_signals requires include_mbo to be enabled".to_string());
        }

        if self.include_mbo && self.mbo_window_size == 0 {
            return Err("mbo_window_size must be > 0 when MBO is enabled".to_string());
        }

        Ok(())
    }
}

// ============================================================================
// Normalization Configuration (Per-Feature-Group Strategies)
// ============================================================================

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

// ============================================================================
// Sampling Configuration (Export-specific wrapper)
// ============================================================================

/// Sampling strategy selection for export configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategyConfig {
    /// Sample every N shares traded
    VolumeBased,
    /// Sample every N events
    EventBased,
}

impl Default for SamplingStrategyConfig {
    fn default() -> Self {
        Self::EventBased
    }
}

/// Export-specific sampling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSamplingConfig {
    /// Sampling strategy
    #[serde(default)]
    pub strategy: SamplingStrategyConfig,

    /// Event count for event-based sampling
    #[serde(default = "default_event_count")]
    pub event_count: usize,

    /// Volume threshold for volume-based sampling
    #[serde(default = "default_volume_threshold")]
    pub volume_threshold: u64,
}

fn default_event_count() -> usize {
    1000
}

fn default_volume_threshold() -> u64 {
    1000
}

impl Default for ExportSamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategyConfig::EventBased,
            event_count: 1000,
            volume_threshold: 1000,
        }
    }
}

impl ExportSamplingConfig {
    /// Convert to internal SamplingConfig.
    pub fn to_sampling_config(&self) -> SamplingConfig {
        match self.strategy {
            SamplingStrategyConfig::EventBased => SamplingConfig {
                strategy: SamplingStrategy::EventBased,
                event_count: Some(self.event_count),
                volume_threshold: None,
                min_time_interval_ns: Some(1_000_000),
                adaptive: None,
                multiscale: None,
            },
            SamplingStrategyConfig::VolumeBased => SamplingConfig {
                strategy: SamplingStrategy::VolumeBased,
                event_count: None,
                volume_threshold: Some(self.volume_threshold),
                min_time_interval_ns: Some(1_000_000),
                adaptive: None,
                multiscale: None,
            },
        }
    }
}

// ============================================================================
// Sequence Configuration (Export-specific wrapper)
// ============================================================================

/// Export-specific sequence configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSequenceConfig {
    /// Window size (number of snapshots per sequence)
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// Stride (overlap between consecutive sequences)
    #[serde(default = "default_stride")]
    pub stride: usize,

    /// Maximum buffer size for sequence building
    #[serde(default = "default_buffer_size")]
    pub max_buffer_size: usize,
}

fn default_window_size() -> usize {
    100
}

fn default_stride() -> usize {
    10
}

fn default_buffer_size() -> usize {
    50000
}

impl Default for ExportSequenceConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            stride: 10,
            max_buffer_size: 50000,
        }
    }
}

impl ExportSequenceConfig {
    /// Convert to internal SequenceConfig.
    pub fn to_sequence_config(&self, feature_count: usize) -> SequenceConfig {
        SequenceConfig {
            window_size: self.window_size,
            stride: self.stride,
            feature_count,
            max_buffer_size: self.max_buffer_size,
        }
    }

    /// Validate the sequence configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("window_size must be > 0".to_string());
        }
        if self.stride == 0 {
            return Err("stride must be > 0".to_string());
        }
        if self.stride > self.window_size {
            return Err("stride should typically be <= window_size".to_string());
        }
        if self.max_buffer_size < self.window_size {
            return Err("max_buffer_size must be >= window_size".to_string());
        }
        Ok(())
    }
}

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
// Label Configuration (Export-specific wrapper)
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
                        "Quantile target_proportion must be in range (0.0, 0.5]".to_string(),
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
            Self::TlobDynamic { divisor, .. } => format!(
                "TLOB Dynamic: mean(|pct_change|) / {} (global)",
                divisor
            ),
        }
    }

    /// Check if this strategy requires global (full dataset) computation.
    ///
    /// Returns `true` for strategies that need all data before threshold can be computed.
    pub fn needs_global_computation(&self) -> bool {
        matches!(self, Self::TlobDynamic { .. })
    }
}

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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    pub fn to_multi_horizon_config(
        &self,
    ) -> Option<crate::labeling::MultiHorizonConfig> {
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
        let conflict_priority = self
            .conflict_priority
            .unwrap_or_default()
            .to_internal();

        let configs: Vec<crate::labeling::OpportunityConfig> = horizons
            .iter()
            .map(|&h| {
                crate::labeling::OpportunityConfig::new(h, self.threshold)
                    .with_conflict_priority(conflict_priority)
            })
            .collect();

        Some((configs, horizons))
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
            if self.horizons.iter().any(|&h| h == 0) {
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
                    ts.validate().map_err(|e| format!("threshold_strategy: {}", e))?;
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
                    return Err("threshold > 50% seems unreasonable for opportunity detection".to_string());
                }
            }
            LabelingStrategy::TripleBarrier => {
                // Triple barrier validation (to be implemented when used)
                if self.threshold <= 0.0 {
                    return Err("threshold must be > 0 for triple barrier labeling".to_string());
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
                    horizons_str, self.threshold * 100.0, priority_str
                )
            }
            LabelingStrategy::TripleBarrier => {
                format!(
                    "TripleBarrier: {}, profit_target={:.4}%",
                    horizons_str, self.threshold * 100.0
                )
            }
        }
    }
}

// ============================================================================
// Processing Configuration
// ============================================================================

/// Configuration for parallel processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Number of threads for parallel processing.
    ///
    /// Defaults to number of CPU cores - 2 if not specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,

    /// Error handling mode.
    ///
    /// - `"fail_fast"`: Stop on first error
    /// - `"collect_errors"`: Continue processing, collect all errors
    #[serde(default = "default_error_mode")]
    pub error_mode: String,

    /// Enable verbose progress reporting.
    #[serde(default)]
    pub verbose: bool,
}

fn default_error_mode() -> String {
    "collect_errors".to_string()
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            threads: None,
            error_mode: "collect_errors".to_string(),
            verbose: false,
        }
    }
}

impl ProcessingConfig {
    /// Validate the processing configuration.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(threads) = self.threads {
            if threads == 0 {
                return Err("threads must be > 0".to_string());
            }
        }

        if self.error_mode != "fail_fast" && self.error_mode != "collect_errors" {
            return Err(format!(
                "error_mode must be 'fail_fast' or 'collect_errors', got '{}'",
                self.error_mode
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Split Configuration
// ============================================================================

/// Configuration for train/validation/test splits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    /// Proportion of days for training (0.0 to 1.0)
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,

    /// Proportion of days for validation (0.0 to 1.0)
    #[serde(default = "default_val_ratio")]
    pub val_ratio: f64,

    /// Proportion of days for testing (0.0 to 1.0)
    #[serde(default = "default_test_ratio")]
    pub test_ratio: f64,
}

fn default_train_ratio() -> f64 {
    0.7
}

fn default_val_ratio() -> f64 {
    0.15
}

fn default_test_ratio() -> f64 {
    0.15
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            val_ratio: 0.15,
            test_ratio: 0.15,
        }
    }
}

impl SplitConfig {
    /// Validate the split configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.train_ratio < 0.0 || self.train_ratio > 1.0 {
            return Err("train_ratio must be between 0 and 1".to_string());
        }
        if self.val_ratio < 0.0 || self.val_ratio > 1.0 {
            return Err("val_ratio must be between 0 and 1".to_string());
        }
        if self.test_ratio < 0.0 || self.test_ratio > 1.0 {
            return Err("test_ratio must be between 0 and 1".to_string());
        }

        let total = self.train_ratio + self.val_ratio + self.test_ratio;
        if (total - 1.0).abs() > 0.001 {
            return Err(format!(
                "Split ratios must sum to 1.0, got {} + {} + {} = {}",
                self.train_ratio, self.val_ratio, self.test_ratio, total
            ));
        }

        Ok(())
    }

    /// Split a list of days according to the ratios.
    ///
    /// Days are kept in order (chronological split).
    pub fn split_days<'a>(&self, days: &'a [String]) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
        let total = days.len();
        let train_end = (total as f64 * self.train_ratio).round() as usize;
        let val_end = train_end + (total as f64 * self.val_ratio).round() as usize;

        let train: Vec<&str> = days[..train_end].iter().map(|s| s.as_str()).collect();
        let val: Vec<&str> = days[train_end..val_end]
            .iter()
            .map(|s| s.as_str())
            .collect();
        let test: Vec<&str> = days[val_end..].iter().map(|s| s.as_str()).collect();

        (train, val, test)
    }
}

// ============================================================================
// Experiment Metadata
// ============================================================================

/// Metadata for experiment tracking and reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentInfo {
    /// Experiment name
    pub name: String,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Version
    #[serde(default = "default_version")]
    pub version: String,

    /// Tags for categorization
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

impl Default for ExperimentInfo {
    fn default() -> Self {
        Self {
            name: "Unnamed Experiment".to_string(),
            description: None,
            version: "1.0.0".to_string(),
            tags: Vec::new(),
        }
    }
}

// ============================================================================
// Main Dataset Configuration
// ============================================================================

/// Complete dataset export configuration.
///
/// This is the top-level configuration that combines all sub-configurations.
/// It can be loaded from TOML or JSON files for experiment reproducibility.
///
/// # Example TOML
///
/// ```toml
/// [experiment]
/// name = "NVDA 98-Feature Dataset"
/// version = "1.0.0"
///
/// [symbol]
/// name = "NVDA"
/// exchange = "XNAS"
/// filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
///
/// [data]
/// input_dir = "data/NVDA_2025-02-01_to_2025-09-30"
/// output_dir = "data/exports/nvda_98feat"
///
/// [dates]
/// start_date = "2025-02-03"
/// end_date = "2025-09-29"
///
/// [features]
/// lob_levels = 10
/// include_derived = true
/// include_mbo = true
/// include_signals = true
///
/// [sampling]
/// strategy = "event_based"
/// event_count = 1000
///
/// [sequence]
/// window_size = 100
/// stride = 10
///
/// [labels]
/// horizon = 50
/// threshold = 0.0008
///
/// # Optional: Normalization configuration (defaults to raw/no normalization)
/// [normalization]
/// preset = "tlob_paper"  # Or configure per-group:
/// # lob_prices = "raw"
/// # lob_sizes = "raw"  
/// # derived = "raw"
/// # mbo = "raw"
/// # signals = "raw"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Experiment metadata
    #[serde(default)]
    pub experiment: ExperimentInfo,

    /// Symbol configuration
    pub symbol: SymbolConfig,

    /// Data paths
    pub data: DataPathConfig,

    /// Date range
    pub dates: DateRangeConfig,

    /// Feature selection
    #[serde(default)]
    pub features: FeatureSetConfig,

    /// Sampling configuration
    #[serde(default)]
    pub sampling: ExportSamplingConfig,

    /// Sequence building
    #[serde(default)]
    pub sequence: ExportSequenceConfig,

    /// Label generation
    #[serde(default)]
    pub labels: ExportLabelConfig,

    /// Train/val/test split
    #[serde(default)]
    pub split: SplitConfig,

    /// Processing configuration
    #[serde(default)]
    pub processing: ProcessingConfig,

    /// Normalization configuration (per-feature-group strategies)
    /// Defaults to raw (no normalization) for maximum flexibility
    #[serde(default)]
    pub normalization: NormalizationConfig,
}

impl DatasetConfig {
    /// Create a new dataset configuration with minimal required fields.
    pub fn new(symbol: SymbolConfig, data: DataPathConfig, dates: DateRangeConfig) -> Self {
        Self {
            experiment: ExperimentInfo::default(),
            symbol,
            data,
            dates,
            features: FeatureSetConfig::default(),
            sampling: ExportSamplingConfig::default(),
            sequence: ExportSequenceConfig::default(),
            labels: ExportLabelConfig::default(),
            split: SplitConfig::default(),
            processing: ProcessingConfig::default(),
            normalization: NormalizationConfig::default(),
        }
    }

    /// Set experiment metadata.
    pub fn with_experiment(mut self, experiment: ExperimentInfo) -> Self {
        self.experiment = experiment;
        self
    }

    /// Set feature configuration.
    pub fn with_features(mut self, features: FeatureSetConfig) -> Self {
        self.features = features;
        self
    }

    /// Use full 98-feature mode.
    pub fn with_full_features(mut self) -> Self {
        self.features = FeatureSetConfig::full();
        self
    }

    /// Set sampling configuration.
    pub fn with_sampling(mut self, sampling: ExportSamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set sequence configuration.
    pub fn with_sequence(mut self, sequence: ExportSequenceConfig) -> Self {
        self.sequence = sequence;
        self
    }

    /// Set label configuration.
    pub fn with_labels(mut self, labels: ExportLabelConfig) -> Self {
        self.labels = labels;
        self
    }

    /// Set split configuration.
    pub fn with_split(mut self, split: SplitConfig) -> Self {
        self.split = split;
        self
    }

    /// Set processing configuration.
    pub fn with_processing(mut self, processing: ProcessingConfig) -> Self {
        self.processing = processing;
        self
    }

    /// Set normalization configuration.
    pub fn with_normalization(mut self, normalization: NormalizationConfig) -> Self {
        self.normalization = normalization;
        self
    }

    /// Use TLOB paper normalization preset (raw data for model-side BiN).
    pub fn with_tlob_normalization(mut self) -> Self {
        self.normalization = NormalizationConfig::tlob_paper();
        self
    }

    /// Use LOBench benchmark normalization preset.
    pub fn with_lobench_normalization(mut self) -> Self {
        self.normalization = NormalizationConfig::lobench();
        self
    }

    /// Validate the complete configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.symbol.validate().map_err(|e| format!("symbol: {}", e))?;
        self.data
            .validate_lenient()
            .map_err(|e| format!("data: {}", e))?;
        self.dates.validate().map_err(|e| format!("dates: {}", e))?;
        self.features
            .validate()
            .map_err(|e| format!("features: {}", e))?;
        self.sequence
            .validate()
            .map_err(|e| format!("sequence: {}", e))?;
        self.labels.validate().map_err(|e| format!("labels: {}", e))?;
        self.split.validate().map_err(|e| format!("split: {}", e))?;
        self.processing
            .validate()
            .map_err(|e| format!("processing: {}", e))?;
        self.normalization
            .validate()
            .map_err(|e| format!("normalization: {}", e))?;
        Ok(())
    }

    /// Convert to PipelineConfig for processing.
    ///
    /// Propagates `tick_size` from `SymbolConfig` to `FeatureConfig`.
    pub fn to_pipeline_config(&self) -> PipelineConfig {
        let feature_config = self.features.to_feature_config(self.symbol.tick_size);
        let feature_count = feature_config.feature_count();

        PipelineConfig {
            features: feature_config,
            sequence: self.sequence.to_sequence_config(feature_count),
            sampling: Some(self.sampling.to_sampling_config()),
            metadata: None,
        }
    }

    /// Get the expected feature count.
    pub fn feature_count(&self) -> usize {
        self.features.feature_count()
    }

    /// Load configuration from TOML file.
    pub fn load_toml<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: DatasetConfig = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to TOML file.
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let toml_string = toml::to_string_pretty(self)?;
        fs::write(path, toml_string)?;
        Ok(())
    }

    /// Load configuration from JSON file.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: DatasetConfig = serde_json::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to JSON file.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json_string = serde_json::to_string_pretty(self)?;
        fs::write(path, json_string)?;
        Ok(())
    }

    /// Generate file path for a specific date.
    pub fn file_path_for_date(&self, date: &str) -> PathBuf {
        let filename = self.symbol.filename_for_date(date);
        self.data.input_dir.join(filename)
    }

    /// Get all file paths for configured dates.
    pub fn all_file_paths(&self) -> Result<Vec<PathBuf>, String> {
        let dates = self.dates.get_dates()?;
        Ok(dates.iter().map(|d| self.file_path_for_date(d)).collect())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_symbol_config_filename_generation() {
        let symbol = SymbolConfig::nasdaq("NVDA");
        assert_eq!(
            symbol.filename_for_date("2025-02-03"),
            "xnas-itch-20250203.mbo.dbn.zst"
        );
    }

    #[test]
    fn test_feature_set_counts() {
        assert_eq!(FeatureSetConfig::raw_lob().feature_count(), 40);
        assert_eq!(FeatureSetConfig::baseline().feature_count(), 84);
        assert_eq!(FeatureSetConfig::full().feature_count(), 98);
    }

    #[test]
    fn test_date_range_generation() {
        let dates = DateRangeConfig::from_range("2025-02-03", "2025-02-07");
        let result = dates.get_dates().unwrap();
        // Feb 3-7, 2025: Mon, Tue, Wed, Thu, Fri (all weekdays)
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], "2025-02-03");
        assert_eq!(result[4], "2025-02-07");
    }

    #[test]
    fn test_date_range_skips_weekends() {
        let dates = DateRangeConfig::from_range("2025-02-07", "2025-02-10");
        let result = dates.get_dates().unwrap();
        // Feb 7 (Fri) and Feb 10 (Mon) - skips Sat/Sun
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "2025-02-07");
        assert_eq!(result[1], "2025-02-10");
    }

    #[test]
    fn test_split_config_validation() {
        let valid = SplitConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = SplitConfig {
            train_ratio: 0.5,
            val_ratio: 0.5,
            test_ratio: 0.5, // Sum > 1
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_split_days() {
        let split = SplitConfig {
            train_ratio: 0.6,
            val_ratio: 0.2,
            test_ratio: 0.2,
        };

        let days: Vec<String> = (1..=10).map(|i| format!("2025-02-{:02}", i)).collect();
        let (train, val, test) = split.split_days(&days);

        assert_eq!(train.len(), 6);
        assert_eq!(val.len(), 2);
        assert_eq!(test.len(), 2);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = DatasetConfig::new(
            SymbolConfig::nasdaq("NVDA"),
            DataPathConfig::new("/tmp/input", "/tmp/output"),
            DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
        )
        .with_full_features();

        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("test.toml");

        // Save and reload
        config.save_toml(&toml_path).unwrap();

        // Manually patch input dir to exist for validation
        let loaded = {
            let contents = fs::read_to_string(&toml_path).unwrap();
            let mut config: DatasetConfig = toml::from_str(&contents).unwrap();
            config.data.input_dir = temp_dir.path().to_path_buf(); // Use temp dir that exists
            config
        };

        // Now validate should pass
        assert!(loaded.validate().is_ok());
        assert_eq!(loaded.features.feature_count(), 98);
    }

    #[test]
    fn test_to_pipeline_config() {
        let config = DatasetConfig::new(
            SymbolConfig::nasdaq("NVDA"),
            DataPathConfig::new("/tmp/input", "/tmp/output"),
            DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
        )
        .with_full_features();

        let pipeline_config = config.to_pipeline_config();
        assert_eq!(pipeline_config.features.feature_count(), 98);
        assert!(pipeline_config.features.include_signals);
    }

    // ========================================================================
    // ExportLabelConfig Tests
    // ========================================================================

    #[test]
    fn test_label_config_default_is_single_horizon() {
        let config = ExportLabelConfig::default();
        assert!(!config.is_multi_horizon(), "Default should be single-horizon mode");
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
        assert!(err_msg.contains("smoothing_window") && err_msg.contains("should be <= minimum horizon"));
    }

    #[test]
    fn test_label_config_validation_multi_smoothing_greater_than_min_horizon() {
        // smoothing=20, but min horizon is 10
        let config = ExportLabelConfig::multi(vec![10, 50, 100], 20, 0.002);
        let result = config.validate();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("smoothing_window (20) should be <= minimum horizon (10)"),
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
        assert!(!toml_str.contains("horizons"), "Empty horizons should be skipped");
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
    // NormalizationConfig Tests
    // ========================================================================

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
    fn test_dataset_config_with_normalization() {
        let config = DatasetConfig::new(
            SymbolConfig::nasdaq("NVDA"),
            DataPathConfig::new("/tmp/input", "/tmp/output"),
            DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
        )
        .with_tlob_normalization();
        
        assert_eq!(config.normalization.lob_prices, FeatureNormStrategy::None);
        assert!(!config.normalization.any_normalization());
    }

    #[test]
    fn test_dataset_config_with_lobench_normalization() {
        let config = DatasetConfig::new(
            SymbolConfig::nasdaq("NVDA"),
            DataPathConfig::new("/tmp/input", "/tmp/output"),
            DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
        )
        .with_lobench_normalization();
        
        assert_eq!(config.normalization.lob_prices, FeatureNormStrategy::GlobalZScore);
        assert!(config.normalization.any_normalization());
    }

    #[test]
    fn test_dataset_config_normalization_in_toml_roundtrip() {
        let config = DatasetConfig::new(
            SymbolConfig::nasdaq("NVDA"),
            DataPathConfig::new("/tmp/input", "/tmp/output"),
            DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
        )
        .with_normalization(NormalizationConfig::deeplob());
        
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("test_norm.toml");
        
        config.save_toml(&toml_path).unwrap();
        
        let loaded = {
            let contents = fs::read_to_string(&toml_path).unwrap();
            let mut config: DatasetConfig = toml::from_str(&contents).unwrap();
            config.data.input_dir = temp_dir.path().to_path_buf();
            config
        };
        
        assert_eq!(loaded.normalization.lob_prices, FeatureNormStrategy::ZScore);
        assert_eq!(loaded.normalization.lob_sizes, FeatureNormStrategy::ZScore);
        assert_eq!(loaded.normalization.derived, FeatureNormStrategy::None);
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
        assert!(tlob_dynamic.needs_global_computation(), "TlobDynamic should need global computation");
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
        
        let strategy: ExportThresholdStrategy = toml::from_str(toml_str)
            .expect("Should deserialize TlobDynamic");
        
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
        let loaded: ExportThresholdStrategy = toml::from_str(&toml_str).expect("Should deserialize");
        
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

