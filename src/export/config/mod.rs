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

pub mod features;
pub mod labels;
pub mod normalization;
pub mod processing;
pub mod sampling;
pub mod sequence;
pub mod symbol;

// Re-export all public types for ergonomic access
pub use features::{ExperimentalFeatureConfig, FeatureSetConfig};
pub use labels::{
    ExportConflictPriority, ExportLabelConfig, ExportThresholdStrategy, ExportTimeoutStrategy,
    LabelingStrategy,
};
pub use normalization::{FeatureNormStrategy, NormalizationConfig};
pub use processing::{ExperimentInfo, ProcessingConfig, SplitConfig};
pub use sampling::{ExportSamplingConfig, SamplingStrategyConfig};
pub use sequence::ExportSequenceConfig;
pub use symbol::{DataPathConfig, DateRangeConfig, SymbolConfig};

use crate::config::PipelineConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

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
        self.symbol
            .validate()
            .map_err(|e| format!("symbol: {}", e))?;
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
        self.labels
            .validate()
            .map_err(|e| format!("labels: {}", e))?;
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
            let contents = std::fs::read_to_string(&toml_path).unwrap();
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

        assert_eq!(
            config.normalization.lob_prices,
            FeatureNormStrategy::GlobalZScore
        );
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
            let contents = std::fs::read_to_string(&toml_path).unwrap();
            let mut config: DatasetConfig = toml::from_str(&contents).unwrap();
            config.data.input_dir = temp_dir.path().to_path_buf();
            config
        };

        assert_eq!(loaded.normalization.lob_prices, FeatureNormStrategy::ZScore);
        assert_eq!(loaded.normalization.lob_sizes, FeatureNormStrategy::ZScore);
        assert_eq!(loaded.normalization.derived, FeatureNormStrategy::None);
    }
}
