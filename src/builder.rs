//! Fluent builder for pipeline configuration.
//!
//! This module provides a builder pattern for constructing pipeline configurations
//! in a clean, readable, and type-safe manner.
//!
//! # Quick Start
//!
//! ```ignore
//! use feature_extractor::PipelineBuilder;
//!
//! // Simple usage with defaults
//! let mut pipeline = PipelineBuilder::new()
//!     .build()?;
//!
//! // Process data
//! let output = pipeline.process("data/NVDA.mbo.dbn.zst")?;
//! ```
//!
//! # Feature Count Reference
//!
//! The feature count is automatically computed based on configuration:
//!
//! | Configuration | Formula | Count |
//! |--------------|---------|-------|
//! | Raw LOB only | levels × 4 | 40 (10 levels) |
//! | + Derived | + 8 | 48 |
//! | + MBO | + 36 | 76 |
//! | + Both | + 8 + 36 | 84 |
//!
//! Raw LOB features per level: ask_price, ask_size, bid_price, bid_size
//!
//! # Presets Reference
//!
//! | Preset | Features | Window | Stride | Use Case |
//! |--------|----------|--------|--------|----------|
//! | `DeepLOB` | 40 | 100 | 1 | CNN-LSTM models |
//! | `TLOB` | 40 | 100 | 1 | Transformer models |
//! | `FI2010` | 48 | 100 | 1 | Benchmark comparison |
//! | `TransLOB` | 40 | 100 | 1 | Multi-horizon |
//! | `LiT` | 80 | 100 | 1 | 20-level LOB |
//! | `Minimal` | 40 | 1 | 1 | Testing/debugging |
//! | `Full` | 84 | 100 | 1 | Maximum features |
//!
//! # Common Configurations
//!
//! ## DeepLOB Style (40 features)
//!
//! ```ignore
//! let pipeline = PipelineBuilder::new()
//!     .lob_levels(10)
//!     .window(100, 1)
//!     .volume_sampling(1000)
//!     .build()?;
//! ```
//!
//! ## TLOB Style (40 features)
//!
//! ```ignore
//! let pipeline = PipelineBuilder::from_preset(Preset::TLOB)
//!     .volume_sampling(1000)
//!     .build()?;
//! ```
//!
//! ## Full Feature Set (84 features)
//!
//! ```ignore
//! let pipeline = PipelineBuilder::new()
//!     .lob_levels(10)
//!     .with_derived_features()   // +8 features
//!     .with_mbo_features()       // +36 features
//!     .window(100, 10)
//!     .event_sampling(1000)
//!     .build()?;
//! ```

use crate::config::{
    AdaptiveSamplingConfig, ExperimentMetadata, MultiScaleConfig, PipelineConfig, SamplingConfig,
    SamplingStrategy,
};
use crate::features::FeatureConfig;
use crate::pipeline::Pipeline;
use crate::schema::Preset;
use crate::sequence_builder::{HorizonAwareConfig, SequenceConfig};
use mbo_lob_reconstructor::Result;

/// Fluent builder for creating pipeline configurations.
///
/// The builder provides a clean, readable API for constructing complex
/// pipeline configurations while ensuring consistency between components.
///
/// # Features
///
/// - **Auto-sync**: Feature count is automatically synchronized
/// - **Validation**: Configuration is validated before building
/// - **Presets**: Quick setup with research paper presets
/// - **Defaults**: Sensible defaults for all parameters
///
/// # Example
///
/// ```ignore
/// use feature_extractor::PipelineBuilder;
///
/// let pipeline = PipelineBuilder::new()
///     .lob_levels(10)
///     .with_derived_features()
///     .window(100, 10)
///     .volume_sampling(1000)
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct PipelineBuilder {
    features: FeatureConfig,
    window_size: usize,
    stride: usize,
    max_buffer_size: Option<usize>,
    sampling: SamplingConfig,
    metadata: Option<ExperimentMetadata>,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    /// Create a new pipeline builder with default settings.
    ///
    /// Default configuration:
    /// - 10 LOB levels (40 raw features)
    /// - Window size: 100, stride: 1
    /// - Volume-based sampling (1000 shares threshold)
    pub fn new() -> Self {
        Self {
            features: FeatureConfig::default(),
            window_size: 100,
            stride: 1,
            max_buffer_size: None,
            sampling: SamplingConfig::default(),
            metadata: None,
        }
    }

    /// Create a builder from a research paper preset.
    ///
    /// Available presets:
    /// - `Preset::DeepLOB`: 40 features, 100 window, stride 1
    /// - `Preset::TLOB`: 40 features, 100 window, stride 1
    /// - `Preset::FI2010`: 144 features (40 raw + 104 handcrafted)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = PipelineBuilder::from_preset(Preset::TLOB)
    ///     .volume_sampling(500)  // Override sampling
    ///     .build()?;
    /// ```
    pub fn from_preset(preset: Preset) -> Self {
        let mut builder = Self::new();

        match preset {
            Preset::DeepLOB => {
                builder.features = FeatureConfig::new(10);
                builder.window_size = 100;
                builder.stride = 1;
            }
            Preset::TLOB => {
                builder.features = FeatureConfig::new(10);
                builder.window_size = 100;
                builder.stride = 1;
            }
            Preset::FI2010 => {
                builder.features = FeatureConfig::new(10).with_derived(true);
                builder.window_size = 100;
                builder.stride = 1;
            }
            Preset::TransLOB => {
                builder.features = FeatureConfig::new(10);
                builder.window_size = 100;
                builder.stride = 1;
            }
            Preset::LiT => {
                // LiT uses 20 levels with patching
                builder.features = FeatureConfig::new(20);
                builder.window_size = 100;
                builder.stride = 1;
            }
            Preset::Minimal => {
                // Minimal: 5 levels, basic features
                builder.features = FeatureConfig::new(5);
                builder.window_size = 50;
                builder.stride = 1;
            }
            Preset::Full => {
                // Full: All features enabled
                builder.features = FeatureConfig::new(10).with_derived(true).with_mbo(true);
                builder.window_size = 100;
                builder.stride = 1;
            }
        }

        builder
    }

    // =========================================================================
    // Feature Configuration
    // =========================================================================

    /// Set the number of LOB levels to extract.
    ///
    /// Each level contributes 4 features: ask_price, ask_size, bid_price, bid_size.
    /// Default: 10 levels (40 features).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = PipelineBuilder::new()
    ///     .lob_levels(5);  // 20 raw features
    /// ```
    pub fn lob_levels(mut self, levels: usize) -> Self {
        self.features.lob_levels = levels;
        self
    }

    /// Enable derived features (8 additional features).
    ///
    /// Derived features include:
    /// - Mid-price
    /// - Spread (absolute and basis points)
    /// - Total bid/ask volume
    /// - Volume imbalance
    /// - Weighted mid-price
    /// - Price impact estimate
    pub fn with_derived_features(mut self) -> Self {
        self.features.include_derived = true;
        self
    }

    /// Enable MBO (Market-By-Order) features (36 additional features).
    ///
    /// MBO features capture order flow dynamics:
    /// - Order arrival rates
    /// - Trade intensity
    /// - Order flow imbalance
    /// - Cancellation rates
    /// - Queue position metrics
    pub fn with_mbo_features(mut self) -> Self {
        self.features.include_mbo = true;
        self
    }

    /// Set the MBO aggregation window size.
    ///
    /// Default: 1000 messages.
    pub fn mbo_window(mut self, size: usize) -> Self {
        self.features.mbo_window_size = size;
        self
    }

    /// Set the tick size for price calculations.
    ///
    /// Default: 0.01 (US stocks).
    pub fn tick_size(mut self, tick: f64) -> Self {
        self.features.tick_size = tick;
        self
    }

    // =========================================================================
    // Sequence Configuration
    // =========================================================================

    /// Set the sequence window size and stride.
    ///
    /// - `window_size`: Number of snapshots per sequence (e.g., 100)
    /// - `stride`: Number of snapshots to skip between sequences (e.g., 1 or 10)
    ///
    /// # Guidelines
    ///
    /// - For training: stride 10-20 for good coverage without too much overlap
    /// - For inference: stride 1 for maximum responsiveness
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = PipelineBuilder::new()
    ///     .window(100, 10);  // 100 snapshots, stride 10
    /// ```
    pub fn window(mut self, window_size: usize, stride: usize) -> Self {
        self.window_size = window_size;
        self.stride = stride;
        self
    }

    /// Set the maximum buffer size for sequence building.
    ///
    /// Default: max(window_size, 1000).
    pub fn max_buffer(mut self, size: usize) -> Self {
        self.max_buffer_size = Some(size);
        self
    }

    /// Configure window and stride automatically based on prediction horizon.
    ///
    /// This uses research-backed scaling formulas from TLOB/PatchTST papers
    /// to set appropriate lookback windows based on prediction horizon.
    ///
    /// # Formula
    ///
    /// - `lookback_window = max(base_window, horizon × scaling_factor)`
    /// - `stride = lookback_window / target_sequence_length`
    ///
    /// # Default Parameters
    ///
    /// - Base window: 100 (minimum context)
    /// - Scaling factor: 10.0 (10× ratio from TLOB research)
    /// - Target sequence length: 100 (controls output samples)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::PipelineBuilder;
    ///
    /// // Short-term prediction (10 events ahead)
    /// let pipeline = PipelineBuilder::new()
    ///     .with_horizon_aware(10)  // window=100, stride=1
    ///     .build()?;
    ///
    /// // Medium-term prediction (50 events ahead)
    /// let pipeline = PipelineBuilder::new()
    ///     .with_horizon_aware(50)  // window=500, stride=5
    ///     .build()?;
    ///
    /// // Long-term prediction (100 events ahead)
    /// let pipeline = PipelineBuilder::new()
    ///     .with_horizon_aware(100)  // window=1000, stride=10
    ///     .build()?;
    /// ```
    ///
    /// # Research Reference
    ///
    /// - TLOB paper: "10× lookback-to-horizon ratio captures sufficient context"
    /// - PatchTST: "Longer patches for longer horizons improve forecasting"
    pub fn with_horizon_aware(mut self, horizon: usize) -> Self {
        let config = HorizonAwareConfig::new(horizon);
        self.window_size = config.lookback_window();
        self.stride = config.optimal_stride();
        self
    }

    /// Configure window and stride with custom horizon-aware settings.
    ///
    /// Use this when you need fine-grained control over the scaling parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::prelude::*;
    ///
    /// // Conservative scaling (5× ratio instead of 10×)
    /// let config = HorizonAwareConfig::new(50)
    ///     .with_scaling(5.0)
    ///     .with_bounds(100, 2000);
    ///
    /// let pipeline = PipelineBuilder::new()
    ///     .with_horizon_aware_config(config)
    ///     .build()?;
    /// ```
    pub fn with_horizon_aware_config(mut self, config: HorizonAwareConfig) -> Self {
        self.window_size = config.lookback_window();
        self.stride = config.optimal_stride();
        self
    }

    // =========================================================================
    // Sampling Configuration
    // =========================================================================

    /// Use volume-based sampling.
    ///
    /// Samples are taken after a specified volume of shares has traded.
    /// This creates more samples during active periods.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Volume threshold in shares (e.g., 1000)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = PipelineBuilder::new()
    ///     .volume_sampling(1000);  // Sample every 1000 shares
    /// ```
    pub fn volume_sampling(mut self, threshold: u64) -> Self {
        self.sampling.strategy = SamplingStrategy::VolumeBased;
        self.sampling.volume_threshold = Some(threshold);
        self.sampling.event_count = None;
        self
    }

    /// Use event-based sampling.
    ///
    /// Samples are taken after a specified number of MBO events.
    /// This creates uniform sampling in event time.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of events between samples (e.g., 100)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = PipelineBuilder::new()
    ///     .event_sampling(100);  // Sample every 100 events
    /// ```
    pub fn event_sampling(mut self, count: usize) -> Self {
        self.sampling.strategy = SamplingStrategy::EventBased;
        self.sampling.event_count = Some(count);
        self.sampling.volume_threshold = None;
        self
    }

    /// Set minimum time interval between samples (nanoseconds).
    ///
    /// This prevents over-sampling during very active periods.
    /// Default: 1,000,000 ns (1 millisecond).
    pub fn min_sample_interval_ns(mut self, interval: u64) -> Self {
        self.sampling.min_time_interval_ns = Some(interval);
        self
    }

    /// Enable adaptive sampling based on market volatility.
    ///
    /// Adaptive sampling adjusts the volume threshold based on
    /// realized volatility, sampling more during volatile periods.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = PipelineBuilder::new()
    ///     .volume_sampling(1000)
    ///     .with_adaptive_sampling();  // Enable adaptive adjustment
    /// ```
    pub fn with_adaptive_sampling(mut self) -> Self {
        self.sampling.adaptive = Some(AdaptiveSamplingConfig::default());
        self
    }

    /// Configure adaptive sampling with custom parameters.
    pub fn with_adaptive_sampling_config(mut self, config: AdaptiveSamplingConfig) -> Self {
        self.sampling.adaptive = Some(config);
        self
    }

    /// Enable multi-scale windowing.
    ///
    /// Multi-scale windowing maintains three temporal scales:
    /// - Fast: Full resolution (microstructure)
    /// - Medium: 2x decimation (short-term trends)
    /// - Slow: 4x decimation (long-term context)
    pub fn with_multiscale(mut self) -> Self {
        self.sampling.multiscale = Some(MultiScaleConfig::default());
        self
    }

    /// Configure multi-scale windowing with custom parameters.
    pub fn with_multiscale_config(mut self, config: MultiScaleConfig) -> Self {
        self.sampling.multiscale = Some(config);
        self
    }

    // =========================================================================
    // Metadata
    // =========================================================================

    /// Set experiment metadata for tracking and reproducibility.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = PipelineBuilder::new()
    ///     .experiment("nvda_baseline_v1", "Initial NVDA experiment");
    /// ```
    pub fn experiment(mut self, name: &str, description: &str) -> Self {
        self.metadata = Some(ExperimentMetadata {
            name: name.to_string(),
            description: Some(description.to_string()),
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            version: None,
            tags: None,
        });
        self
    }

    /// Set experiment metadata with full control.
    pub fn with_metadata(mut self, metadata: ExperimentMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the pipeline configuration.
    ///
    /// This validates the configuration and returns a `PipelineConfig`
    /// that can be used to create a `Pipeline`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn build_config(self) -> std::result::Result<PipelineConfig, String> {
        // Build sequence config with auto-computed feature count
        let sequence = SequenceConfig {
            window_size: self.window_size,
            stride: self.stride,
            max_buffer_size: self.max_buffer_size.unwrap_or(self.window_size.max(1000)),
            feature_count: self.features.feature_count(),
        };

        let config = PipelineConfig {
            features: self.features,
            sequence,
            sampling: Some(self.sampling),
            metadata: self.metadata,
        };

        // Validate before returning
        config.validate()?;

        Ok(config)
    }

    /// Build and return a ready-to-use Pipeline.
    ///
    /// This is the most common entry point - it builds the configuration
    /// and creates a Pipeline in one step.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut pipeline = PipelineBuilder::new()
    ///     .lob_levels(10)
    ///     .window(100, 10)
    ///     .volume_sampling(1000)
    ///     .build()?;
    ///
    /// let output = pipeline.process("data.dbn.zst")?;
    /// ```
    pub fn build(self) -> Result<Pipeline> {
        let config = self
            .build_config()
            .map_err(mbo_lob_reconstructor::TlobError::generic)?;
        Pipeline::from_config(config)
    }

    /// Get the computed feature count based on current configuration.
    ///
    /// This is useful for understanding the output dimensionality
    /// before building the pipeline.
    pub fn feature_count(&self) -> usize {
        self.features.feature_count()
    }

    /// Get a summary of the current configuration.
    pub fn summary(&self) -> String {
        let sampling_desc = match self.sampling.strategy {
            SamplingStrategy::VolumeBased => {
                format!(
                    "Volume-based ({})",
                    self.sampling.volume_threshold.unwrap_or(0)
                )
            }
            SamplingStrategy::EventBased => {
                format!("Event-based ({})", self.sampling.event_count.unwrap_or(0))
            }
            SamplingStrategy::TimeBased => "Time-based".to_string(),
            SamplingStrategy::MultiScale => "Multi-scale".to_string(),
        };

        let features_desc = match (self.features.include_derived, self.features.include_mbo) {
            (false, false) => "Raw LOB only",
            (true, false) => "LOB + Derived",
            (false, true) => "LOB + MBO",
            (true, true) => "LOB + Derived + MBO",
        };

        format!(
            "PipelineBuilder Summary:\n\
             - LOB levels: {}\n\
             - Features: {} ({} total)\n\
             - Window: {} snapshots, stride {}\n\
             - Sampling: {}\n\
             - Adaptive: {}\n\
             - Multi-scale: {}",
            self.features.lob_levels,
            features_desc,
            self.feature_count(),
            self.window_size,
            self.stride,
            sampling_desc,
            self.sampling.adaptive.is_some(),
            self.sampling.multiscale.is_some(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let builder = PipelineBuilder::new();
        assert_eq!(builder.features.lob_levels, 10);
        assert_eq!(builder.window_size, 100);
        assert_eq!(builder.stride, 1);
        assert_eq!(builder.feature_count(), 40);
    }

    #[test]
    fn test_builder_with_derived() {
        let builder = PipelineBuilder::new().with_derived_features();
        assert!(builder.features.include_derived);
        assert_eq!(builder.feature_count(), 48);
    }

    #[test]
    fn test_builder_with_mbo() {
        let builder = PipelineBuilder::new().with_mbo_features();
        assert!(builder.features.include_mbo);
        assert_eq!(builder.feature_count(), 76);
    }

    #[test]
    fn test_builder_full_features() {
        let builder = PipelineBuilder::new()
            .with_derived_features()
            .with_mbo_features();
        assert_eq!(builder.feature_count(), 84);
    }

    #[test]
    fn test_builder_lob_levels() {
        let builder = PipelineBuilder::new().lob_levels(5);
        assert_eq!(builder.features.lob_levels, 5);
        assert_eq!(builder.feature_count(), 20);
    }

    #[test]
    fn test_builder_window() {
        let builder = PipelineBuilder::new().window(200, 20);
        assert_eq!(builder.window_size, 200);
        assert_eq!(builder.stride, 20);
    }

    #[test]
    fn test_builder_volume_sampling() {
        let builder = PipelineBuilder::new().volume_sampling(500);
        assert_eq!(builder.sampling.strategy, SamplingStrategy::VolumeBased);
        assert_eq!(builder.sampling.volume_threshold, Some(500));
    }

    #[test]
    fn test_builder_event_sampling() {
        let builder = PipelineBuilder::new().event_sampling(200);
        assert_eq!(builder.sampling.strategy, SamplingStrategy::EventBased);
        assert_eq!(builder.sampling.event_count, Some(200));
    }

    #[test]
    fn test_builder_adaptive_sampling() {
        let builder = PipelineBuilder::new().with_adaptive_sampling();
        assert!(builder.sampling.adaptive.is_some());
    }

    #[test]
    fn test_builder_multiscale() {
        let builder = PipelineBuilder::new().with_multiscale();
        assert!(builder.sampling.multiscale.is_some());
    }

    #[test]
    fn test_builder_experiment() {
        let builder = PipelineBuilder::new().experiment("test", "Test experiment");
        assert!(builder.metadata.is_some());
        assert_eq!(builder.metadata.as_ref().unwrap().name, "test");
    }

    #[test]
    fn test_builder_build_config() {
        let config = PipelineBuilder::new()
            .lob_levels(10)
            .with_derived_features()
            .window(100, 10)
            .volume_sampling(1000)
            .build_config()
            .expect("Should build valid config");

        assert_eq!(config.features.lob_levels, 10);
        assert!(config.features.include_derived);
        assert_eq!(config.sequence.window_size, 100);
        assert_eq!(config.sequence.stride, 10);
        assert_eq!(config.sequence.feature_count, 48);
    }

    #[test]
    fn test_builder_feature_count_auto_sync() {
        // This is the key test - feature count should be auto-computed
        let config = PipelineBuilder::new()
            .lob_levels(10)
            .with_derived_features()
            .with_mbo_features()
            .build_config()
            .expect("Should build valid config");

        // Feature count should be auto-computed: 40 + 8 + 36 = 84
        assert_eq!(config.sequence.feature_count, 84);
        assert_eq!(config.features.feature_count(), 84);
    }

    #[test]
    fn test_builder_from_preset_deeplob() {
        let builder = PipelineBuilder::from_preset(Preset::DeepLOB);
        assert_eq!(builder.features.lob_levels, 10);
        assert!(!builder.features.include_derived);
        assert!(!builder.features.include_mbo);
        assert_eq!(builder.feature_count(), 40);
    }

    #[test]
    fn test_builder_from_preset_tlob() {
        let builder = PipelineBuilder::from_preset(Preset::TLOB);
        assert_eq!(builder.features.lob_levels, 10);
        assert_eq!(builder.window_size, 100);
        assert_eq!(builder.stride, 1);
    }

    #[test]
    fn test_builder_summary() {
        let builder = PipelineBuilder::new()
            .lob_levels(10)
            .with_derived_features()
            .window(100, 10)
            .volume_sampling(1000);

        let summary = builder.summary();
        assert!(summary.contains("LOB levels: 10"));
        assert!(summary.contains("LOB + Derived"));
        assert!(summary.contains("48 total"));
        assert!(summary.contains("100 snapshots"));
        assert!(summary.contains("stride 10"));
    }

    #[test]
    fn test_builder_chaining() {
        // Test that all builder methods return Self for chaining
        let _builder = PipelineBuilder::new()
            .lob_levels(10)
            .with_derived_features()
            .with_mbo_features()
            .mbo_window(500)
            .tick_size(0.01)
            .window(100, 10)
            .max_buffer(2000)
            .volume_sampling(1000)
            .min_sample_interval_ns(1_000_000)
            .with_adaptive_sampling()
            .with_multiscale()
            .experiment("test", "Test");
    }

    #[test]
    fn test_builder_invalid_config() {
        // Zero LOB levels should fail validation
        let result = PipelineBuilder::new().lob_levels(0).build_config();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_horizon_aware_short() {
        // Short horizon: should use base window
        let builder = PipelineBuilder::new().with_horizon_aware(10);
        assert_eq!(builder.window_size, 100); // Base window
        assert_eq!(builder.stride, 1); // 100 / 100 = 1
    }

    #[test]
    fn test_builder_with_horizon_aware_medium() {
        // Medium horizon: should scale
        let builder = PipelineBuilder::new().with_horizon_aware(50);
        assert_eq!(builder.window_size, 500); // 50 × 10
        assert_eq!(builder.stride, 5); // 500 / 100 = 5
    }

    #[test]
    fn test_builder_with_horizon_aware_long() {
        // Long horizon: should scale
        let builder = PipelineBuilder::new().with_horizon_aware(100);
        assert_eq!(builder.window_size, 1000); // 100 × 10
        assert_eq!(builder.stride, 10); // 1000 / 100 = 10
    }

    #[test]
    fn test_builder_with_horizon_aware_config() {
        // Custom config with different scaling
        let config = HorizonAwareConfig::new(50).with_scaling(5.0);
        let builder = PipelineBuilder::new().with_horizon_aware_config(config);
        assert_eq!(builder.window_size, 250); // 50 × 5
        assert_eq!(builder.stride, 2); // 250 / 100 = 2
    }

    #[test]
    fn test_builder_with_horizon_aware_fi2010() {
        // FI-2010 uses horizon 100
        let builder = PipelineBuilder::from_preset(Preset::FI2010).with_horizon_aware(100);

        assert_eq!(builder.window_size, 1000);
        assert_eq!(builder.stride, 10);
        assert!(builder.features.include_derived);
    }

    #[test]
    fn test_builder_with_horizon_aware_bounds() {
        // Very long horizon should be bounded
        let config = HorizonAwareConfig::new(1000).with_bounds(100, 2000);
        let builder = PipelineBuilder::new().with_horizon_aware_config(config);
        assert_eq!(builder.window_size, 2000); // Clamped to max
    }

    #[test]
    fn test_builder_horizon_aware_chaining() {
        // Should chain with other methods
        let builder = PipelineBuilder::new()
            .lob_levels(10)
            .with_derived_features()
            .with_horizon_aware(50)
            .volume_sampling(1000);

        assert_eq!(builder.window_size, 500);
        assert_eq!(builder.stride, 5);
        assert!(builder.features.include_derived);
        assert_eq!(builder.feature_count(), 48);
    }
}
