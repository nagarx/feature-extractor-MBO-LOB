//! Pipeline configuration management.
//!
//! This module provides unified configuration for the entire feature extraction
//! and preprocessing pipeline, with serialization support for experiment
//! reproducibility.
//!
//! # Features
//!
//! - **Unified Configuration**: Single struct combining all pipeline stages
//! - **Serialization**: Save/load configurations to TOML or JSON
//! - **Validation**: Ensure configurations are valid before use
//! - **Reproducibility**: Version control friendly configuration files
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::config::PipelineConfig;
//!
//! // Create configuration
//! let config = PipelineConfig::default();
//!
//! // Save to file
//! config.save_toml("experiment_config.toml")?;
//!
//! // Load from file
//! let loaded = PipelineConfig::load_toml("experiment_config.toml")?;
//!
//! // Use with pipeline
//! let pipeline = Pipeline::from_config(loaded)?;
//! ```

use crate::{FeatureConfig, SequenceConfig};
use std::fs;
use std::path::Path;

/// Unified pipeline configuration.
///
/// Contains all configuration parameters for the complete feature extraction
/// and preprocessing pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineConfig {
    /// Feature extraction configuration
    pub features: FeatureConfig,

    /// Sequence building configuration
    pub sequence: SequenceConfig,

    /// Sampling configuration (optional - can be set programmatically)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingConfig>,

    /// Experiment metadata (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExperimentMetadata>,
}

/// Sampling configuration.
///
/// Determines which LOB snapshots to use for feature extraction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingConfig {
    /// Sampling strategy type
    pub strategy: SamplingStrategy,

    /// Volume threshold (for volume-based sampling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume_threshold: Option<u64>,

    /// Minimum time interval in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_time_interval_ns: Option<u64>,

    /// Event count threshold (for event-based sampling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_count: Option<usize>,

    /// Phase 1: Adaptive sampling configuration (opt-in)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptive: Option<AdaptiveSamplingConfig>,

    /// Phase 1: Multi-scale window configuration (opt-in)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiscale: Option<MultiScaleConfig>,
}

/// Phase 1: Adaptive Sampling Configuration
///
/// Enables market-regime-aware sampling that dynamically adjusts
/// thresholds based on realized volatility.
///
/// # When to Use
///
/// - Variable market conditions (quiet → volatile transitions)
/// - Need to balance data volume with signal quality
/// - Want automatic threshold adaptation
///
/// # Example
///
/// ```ignore
/// let adaptive = AdaptiveSamplingConfig {
///     enabled: true,
///     volatility_window: 1000,
///     calibration_size: 100,
///     base_threshold: 1000,
///     min_multiplier: 0.5,
///     max_multiplier: 2.0,
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveSamplingConfig {
    /// Enable adaptive threshold adjustment
    pub enabled: bool,

    /// Volatility estimation window (number of mid-price updates)
    pub volatility_window: usize,

    /// Calibration period (number of volatility samples to collect)
    pub calibration_size: usize,

    /// Base volume threshold (adjusted by volatility multiplier)
    pub base_threshold: u64,

    /// Minimum threshold multiplier (e.g., 0.5 = 50% of base)
    pub min_multiplier: f64,

    /// Maximum threshold multiplier (e.g., 2.0 = 200% of base)
    pub max_multiplier: f64,
}

/// Phase 1: Multi-Scale Window Configuration
///
/// Enables three-scale temporal analysis with intelligent decimation.
///
/// # Scale Definitions
///
/// - **Fast**: Full resolution (decimation=1), captures microstructure
/// - **Medium**: 2× decimation, captures short-term trends
/// - **Slow**: 4× decimation, captures long-term context
///
/// # Example
///
/// ```ignore
/// let multiscale = MultiScaleConfig {
///     enabled: true,
///     fast_window: 100,
///     medium_window: 500,
///     medium_decimation: 2,
///     slow_window: 1000,
///     slow_decimation: 4,
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiScaleConfig {
    /// Enable multi-scale windowing
    pub enabled: bool,

    /// Fast scale window size (full resolution)
    pub fast_window: usize,

    /// Medium scale window size
    pub medium_window: usize,

    /// Medium scale decimation factor
    pub medium_decimation: usize,

    /// Slow scale window size
    pub slow_window: usize,

    /// Slow scale decimation factor
    pub slow_decimation: usize,
}

/// Sampling strategy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SamplingStrategy {
    /// Sample every N shares traded
    VolumeBased,

    /// Sample every N events
    EventBased,

    /// Sample at fixed time intervals
    TimeBased,

    /// Multi-scale sampling
    MultiScale,
}

/// Experiment metadata for tracking and reproducibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExperimentMetadata {
    /// Experiment name
    pub name: String,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Creation timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// Version or git commit
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Custom tags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            features: FeatureConfig::default(),
            sequence: SequenceConfig::default(),
            sampling: Some(SamplingConfig::default()),
            metadata: None,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: Some(1000),          // 1000 shares
            min_time_interval_ns: Some(1_000_000), // 1ms
            event_count: None,
            adaptive: None,   // Phase 1: opt-in
            multiscale: None, // Phase 1: opt-in
        }
    }
}

impl Default for AdaptiveSamplingConfig {
    /// Default configuration based on TLOB research and Phase 1 design.
    fn default() -> Self {
        Self {
            enabled: true,
            volatility_window: 1000, // 1000 mid-price updates
            calibration_size: 100,   // 100 volatility samples
            base_threshold: 1000,    // 1000 shares (matches default)
            min_multiplier: 0.5,     // 50% of base in quiet markets
            max_multiplier: 2.0,     // 200% of base in volatile markets
        }
    }
}

impl Default for MultiScaleConfig {
    /// Default configuration based on TLOB paper recommendations.
    fn default() -> Self {
        Self {
            enabled: true,
            fast_window: 100,     // Full resolution, microstructure
            medium_window: 500,   // Short-term trends
            medium_decimation: 2, // Every 2nd event
            slow_window: 1000,    // Long-term context
            slow_decimation: 4,   // Every 4th event
        }
    }
}

impl PipelineConfig {
    /// Create a new pipeline configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set experiment metadata.
    pub fn with_metadata(mut self, metadata: ExperimentMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set feature configuration.
    pub fn with_features(mut self, config: FeatureConfig) -> Self {
        self.features = config;
        self
    }

    /// Set sequence configuration.
    pub fn with_sequence(mut self, config: SequenceConfig) -> Self {
        self.sequence = config;
        self
    }

    /// Set sampling configuration.
    pub fn with_sampling(mut self, config: SamplingConfig) -> Self {
        self.sampling = Some(config);
        self
    }

    /// Enable Phase 1 adaptive sampling with default parameters.
    ///
    /// This configures market-regime-aware sampling that dynamically
    /// adjusts volume thresholds based on realized volatility.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = PipelineConfig::default()
    ///     .enable_adaptive_sampling();
    /// ```
    pub fn enable_adaptive_sampling(mut self) -> Self {
        if let Some(ref mut sampling) = self.sampling {
            sampling.adaptive = Some(AdaptiveSamplingConfig::default());
        } else {
            let sampling = SamplingConfig {
                adaptive: Some(AdaptiveSamplingConfig::default()),
                ..Default::default()
            };
            self.sampling = Some(sampling);
        }
        self
    }

    /// Enable Phase 1 multi-scale windowing with default parameters.
    ///
    /// This configures three-scale temporal analysis (fast/medium/slow)
    /// with intelligent decimation for memory efficiency.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = PipelineConfig::default()
    ///     .enable_multiscale_windowing();
    /// ```
    pub fn enable_multiscale_windowing(mut self) -> Self {
        if let Some(ref mut sampling) = self.sampling {
            sampling.multiscale = Some(MultiScaleConfig::default());
        } else {
            let sampling = SamplingConfig {
                multiscale: Some(MultiScaleConfig::default()),
                ..Default::default()
            };
            self.sampling = Some(sampling);
        }
        self
    }

    /// Enable both Phase 1 features (adaptive sampling + multi-scale windowing).
    ///
    /// This is a convenience method that enables all Phase 1 optimizations
    /// with recommended default parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = PipelineConfig::default()
    ///     .enable_phase1();
    /// ```
    pub fn enable_phase1(self) -> Self {
        self.enable_adaptive_sampling()
            .enable_multiscale_windowing()
    }

    /// Validate the configuration.
    ///
    /// Returns Ok(()) if valid, Err(msg) otherwise.
    pub fn validate(&self) -> Result<(), String> {
        // Validate sequence config
        self.sequence.validate()?;

        // Validate sampling config if present
        if let Some(sampling) = &self.sampling {
            sampling.validate()?;
        }

        // ✅ FIX: Validate feature count consistency with all possible combinations
        let base_lob = 40; // Raw LOB features (10 levels × 4 features/level)
        let derived_count = if self.features.include_derived { 8 } else { 0 };
        let mbo_count = if self.features.include_mbo { 36 } else { 0 };
        let expected_features = base_lob + derived_count + mbo_count;

        if self.sequence.feature_count != expected_features {
            let config_desc = match (self.features.include_derived, self.features.include_mbo) {
                (false, false) => "raw LOB only",
                (true, false) => "LOB + derived",
                (false, true) => "LOB + MBO",
                (true, true) => "LOB + derived + MBO",
            };
            return Err(format!(
                "Feature count mismatch: sequence expects {}, but {} suggests {} (40 base + {} derived + {} MBO)",
                self.sequence.feature_count, config_desc, expected_features, derived_count, mbo_count
            ));
        }

        Ok(())
    }

    /// Save configuration to TOML file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// config.save_toml("configs/experiment1.toml")?;
    /// ```
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let toml_string = toml::to_string_pretty(self)?;
        fs::write(path, toml_string)?;
        Ok(())
    }

    /// Load configuration from TOML file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = PipelineConfig::load_toml("configs/experiment1.toml")?;
    /// ```
    pub fn load_toml<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: PipelineConfig = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to JSON file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// config.save_json("configs/experiment1.json")?;
    /// ```
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json_string = serde_json::to_string_pretty(self)?;
        fs::write(path, json_string)?;
        Ok(())
    }

    /// Load configuration from JSON file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = PipelineConfig::load_json("configs/experiment1.json")?;
    /// ```
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: PipelineConfig = serde_json::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }
}

impl SamplingConfig {
    /// Validate sampling configuration.
    pub fn validate(&self) -> Result<(), String> {
        match self.strategy {
            SamplingStrategy::VolumeBased => {
                if self.volume_threshold.is_none() {
                    return Err("Volume-based sampling requires volume_threshold".to_string());
                }
                if self.volume_threshold == Some(0) {
                    return Err("volume_threshold must be > 0".to_string());
                }
            }
            SamplingStrategy::EventBased => {
                if self.event_count.is_none() {
                    return Err("Event-based sampling requires event_count".to_string());
                }
                if self.event_count == Some(0) {
                    return Err("event_count must be > 0".to_string());
                }
            }
            SamplingStrategy::TimeBased => {
                if self.min_time_interval_ns.is_none() {
                    return Err("Time-based sampling requires min_time_interval_ns".to_string());
                }
            }
            SamplingStrategy::MultiScale => {
                // Multi-scale validation would need more complex logic
            }
        }

        // Validate Phase 1 adaptive sampling config if present
        if let Some(ref adaptive) = self.adaptive {
            adaptive.validate()?;
        }

        // Validate Phase 1 multi-scale config if present
        if let Some(ref multiscale) = self.multiscale {
            multiscale.validate()?;
        }

        Ok(())
    }
}

impl AdaptiveSamplingConfig {
    /// Validate adaptive sampling configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.volatility_window == 0 {
            return Err("volatility_window must be > 0".to_string());
        }

        if self.calibration_size == 0 {
            return Err("calibration_size must be > 0".to_string());
        }

        if self.calibration_size < 100 {
            return Err("calibration_size must be >= 100 for statistical reliability".to_string());
        }

        if self.base_threshold == 0 {
            return Err("base_threshold must be > 0".to_string());
        }

        if self.min_multiplier <= 0.0 {
            return Err("min_multiplier must be > 0".to_string());
        }

        if self.max_multiplier <= 0.0 {
            return Err("max_multiplier must be > 0".to_string());
        }

        if self.min_multiplier >= self.max_multiplier {
            return Err("min_multiplier must be < max_multiplier".to_string());
        }

        Ok(())
    }
}

impl MultiScaleConfig {
    /// Validate multi-scale configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.fast_window == 0 {
            return Err("fast_window must be > 0".to_string());
        }

        if self.medium_window == 0 {
            return Err("medium_window must be > 0".to_string());
        }

        if self.slow_window == 0 {
            return Err("slow_window must be > 0".to_string());
        }

        if self.medium_decimation == 0 {
            return Err("medium_decimation must be > 0".to_string());
        }

        if self.slow_decimation == 0 {
            return Err("slow_decimation must be > 0".to_string());
        }

        // Ensure logical ordering: fast < medium < slow
        if self.fast_window >= self.medium_window {
            return Err("fast_window must be < medium_window".to_string());
        }

        if self.medium_window >= self.slow_window {
            return Err("medium_window must be < slow_window".to_string());
        }

        // Ensure decimation increases with scale
        if self.medium_decimation >= self.slow_decimation {
            return Err("medium_decimation must be < slow_decimation".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_save_load_toml() {
        let config = PipelineConfig::default().with_metadata(ExperimentMetadata {
            name: "test_experiment".to_string(),
            description: Some("Test configuration".to_string()),
            created_at: None,
            version: Some("0.1.0".to_string()),
            tags: Some(vec!["test".to_string()]),
        });

        let path = "test_config.toml";

        // Save
        config.save_toml(path).unwrap();

        // Load
        let loaded = PipelineConfig::load_toml(path).unwrap();

        // Verify
        assert_eq!(loaded.features.lob_levels, config.features.lob_levels);
        assert_eq!(loaded.sequence.window_size, config.sequence.window_size);
        assert!(loaded.metadata.is_some());

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_json() {
        let config = PipelineConfig::default();
        let path = "test_config.json";

        // Save
        config.save_json(path).unwrap();

        // Load
        let loaded = PipelineConfig::load_json(path).unwrap();

        // Verify
        assert_eq!(loaded.features.lob_levels, config.features.lob_levels);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_sampling_config_validation() {
        // Valid volume-based
        let config = SamplingConfig {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: Some(1000),
            min_time_interval_ns: Some(1_000_000),
            event_count: None,
            adaptive: None,
            multiscale: None,
        };
        assert!(config.validate().is_ok());

        // Invalid - no volume threshold
        let config = SamplingConfig {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: None,
            min_time_interval_ns: Some(1_000_000),
            event_count: None,
            adaptive: None,
            multiscale: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feature_count_validation() {
        let mut config = PipelineConfig::default();

        // Valid raw LOB-only (default: include_derived = false)
        config.features.include_mbo = false;
        config.features.include_derived = false;
        config.sequence.feature_count = 40; // 10 levels × 4 = 40 raw LOB features
        assert!(config.validate().is_ok());

        // Invalid - wrong feature count for raw LOB
        config.sequence.feature_count = 84;
        assert!(config.validate().is_err());

        // Valid LOB + derived
        config.features.include_derived = true;
        config.sequence.feature_count = 48; // 40 raw + 8 derived
        assert!(config.validate().is_ok());

        // Valid LOB + MBO (no derived)
        config.features.include_mbo = true;
        config.features.include_derived = false;
        config.sequence.feature_count = 76; // 40 raw + 36 MBO
        assert!(config.validate().is_ok());

        // Valid LOB + derived + MBO
        config.features.include_derived = true;
        config.sequence.feature_count = 84; // 40 raw + 8 derived + 36 MBO
        assert!(config.validate().is_ok());
    }

    // Phase 1 Configuration Tests

    #[test]
    fn test_adaptive_sampling_config_default() {
        let config = AdaptiveSamplingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.volatility_window, 1000);
        assert_eq!(config.calibration_size, 100);
        assert_eq!(config.base_threshold, 1000);
        assert_eq!(config.min_multiplier, 0.5);
        assert_eq!(config.max_multiplier, 2.0);
    }

    #[test]
    fn test_adaptive_sampling_validation() {
        // Valid config
        let mut config = AdaptiveSamplingConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: calibration_size < 100
        config.calibration_size = 50;
        assert!(config.validate().is_err());
        config.calibration_size = 100; // Fix

        // Invalid: min_multiplier >= max_multiplier
        config.min_multiplier = 2.0;
        config.max_multiplier = 1.0;
        assert!(config.validate().is_err());
        config.min_multiplier = 0.5; // Fix
        config.max_multiplier = 2.0;

        // Invalid: base_threshold = 0
        config.base_threshold = 0;
        assert!(config.validate().is_err());
        config.base_threshold = 1000; // Fix

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multiscale_config_default() {
        let config = MultiScaleConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.fast_window, 100);
        assert_eq!(config.medium_window, 500);
        assert_eq!(config.medium_decimation, 2);
        assert_eq!(config.slow_window, 1000);
        assert_eq!(config.slow_decimation, 4);
    }

    #[test]
    fn test_multiscale_validation() {
        // Valid config
        let mut config = MultiScaleConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: fast >= medium
        config.fast_window = 500;
        config.medium_window = 500;
        assert!(config.validate().is_err());
        config.fast_window = 100; // Fix

        // Invalid: medium >= slow
        config.medium_window = 1000;
        config.slow_window = 1000;
        assert!(config.validate().is_err());
        config.medium_window = 500; // Fix

        // Invalid: medium_decimation >= slow_decimation
        config.medium_decimation = 4;
        config.slow_decimation = 4;
        assert!(config.validate().is_err());
        config.medium_decimation = 2; // Fix

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_enable_phase1_features() {
        // Start with default config
        let config = PipelineConfig::default();
        assert!(config.sampling.as_ref().unwrap().adaptive.is_none());
        assert!(config.sampling.as_ref().unwrap().multiscale.is_none());

        // Enable adaptive sampling
        let config = config.enable_adaptive_sampling();
        assert!(config.sampling.as_ref().unwrap().adaptive.is_some());
        assert!(config.sampling.as_ref().unwrap().multiscale.is_none());

        // Enable multiscale windowing
        let config = PipelineConfig::default().enable_multiscale_windowing();
        assert!(config.sampling.as_ref().unwrap().adaptive.is_none());
        assert!(config.sampling.as_ref().unwrap().multiscale.is_some());

        // Enable both (Phase 1)
        let config = PipelineConfig::default().enable_phase1();
        assert!(config.sampling.as_ref().unwrap().adaptive.is_some());
        assert!(config.sampling.as_ref().unwrap().multiscale.is_some());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_phase1_config_serialization() {
        // Create config with Phase 1 features
        let config = PipelineConfig::default().enable_phase1();

        // Save to TOML
        let path = "test_phase1_config.toml";
        config.save_toml(path).unwrap();

        // Load and verify
        let loaded = PipelineConfig::load_toml(path).unwrap();
        assert!(loaded.sampling.as_ref().unwrap().adaptive.is_some());
        assert!(loaded.sampling.as_ref().unwrap().multiscale.is_some());
        assert!(loaded.validate().is_ok());

        // Verify values
        let adaptive = loaded.sampling.as_ref().unwrap().adaptive.as_ref().unwrap();
        assert_eq!(adaptive.volatility_window, 1000);
        assert_eq!(adaptive.calibration_size, 100);

        let multiscale = loaded
            .sampling
            .as_ref()
            .unwrap()
            .multiscale
            .as_ref()
            .unwrap();
        assert_eq!(multiscale.fast_window, 100);
        assert_eq!(multiscale.medium_window, 500);

        // Cleanup
        fs::remove_file(path).ok();
    }
}
