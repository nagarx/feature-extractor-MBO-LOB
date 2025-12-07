//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types and traits for
//! ergonomic usage of the feature extraction library.
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::prelude::*;
//!
//! // Now you have access to all common types
//! let config = PipelineConfig::default();
//! let mut pipeline = Pipeline::from_config(config)?;
//! let output = pipeline.process("data.dbn.zst")?;
//! ```
//!
//! # What's Included
//!
//! ## Core Pipeline
//! - [`Pipeline`] - Main processing pipeline
//! - [`PipelineConfig`] - Pipeline configuration
//! - [`PipelineOutput`] - Pipeline output container
//!
//! ## Feature Extraction
//! - [`FeatureExtractor`] - Feature extraction engine
//! - [`FeatureConfig`] - Feature extraction configuration
//!
//! ## Sequence Building
//! - [`SequenceBuilder`] - Sequence generation for transformers
//! - [`SequenceConfig`] - Sequence building configuration
//! - [`Sequence`] - Output sequence container
//!
//! ## Labeling
//! - [`LabelConfig`] - Label generation configuration
//! - [`LabelGenerator`] - Generic label generator
//! - [`TlobLabelGenerator`] - TLOB-style label generation
//! - [`TrendLabel`] - Label enum (Up, Down, Stable)
//!
//! ## Normalization
//! - [`Normalizer`] - Normalization trait
//! - [`ZScoreNormalizer`] - Z-score normalization
//! - [`GlobalZScoreNormalizer`] - Global z-score normalization
//! - [`BilinearNormalizer`] - Bilinear normalization (TLOB)
//!
//! ## Sampling
//! - [`SamplingConfig`] - Sampling configuration
//! - [`SamplingStrategy`] - Sampling strategy enum
//! - [`VolumeBasedSampler`] - Volume-based sampling
//! - [`EventBasedSampler`] - Event-based sampling
//!
//! ## Export
//! - [`NumpyExporter`] - Export to NumPy format
//! - [`BatchExporter`] - Batch export with labeling
//!
//! ## Validation
//! - [`FeatureValidator`] - Feature validation
//! - [`ValidationResult`] - Validation result container
//!
//! ## From MBO-LOB-Reconstructor
//! - [`LobState`] - LOB state snapshot
//! - [`DbnLoader`] - Databento DBN file loader
//! - [`LobReconstructor`] - MBO to LOB reconstruction

// ============================================================================
// Core Pipeline
// ============================================================================

pub use crate::builder::PipelineBuilder;
pub use crate::config::{
    AdaptiveSamplingConfig, ExperimentMetadata, MultiScaleConfig, PipelineConfig, SamplingConfig,
    SamplingStrategy,
};
pub use crate::pipeline::{Pipeline, PipelineOutput};

// ============================================================================
// Feature Extraction
// ============================================================================

pub use crate::features::fi2010::{FI2010Config, FI2010Extractor};
pub use crate::features::mbo_features::{MboAggregator, MboEvent};
pub use crate::features::order_flow::{OrderFlowFeatures, OrderFlowTracker};
pub use crate::features::{FeatureConfig, FeatureExtractor};

// ============================================================================
// Sequence Building
// ============================================================================

pub use crate::sequence_builder::{
    HorizonAwareConfig, MultiScaleConfig as MultiScaleSequenceConfig, MultiScaleSequence,
    MultiScaleWindow, Sequence, SequenceBuilder, SequenceConfig, SequenceError,
};

// ============================================================================
// Labeling
// ============================================================================

pub use crate::labeling::{
    DeepLobLabelGenerator, DeepLobMethod, LabelConfig, LabelGenerator, LabelStats,
    TlobLabelGenerator, TrendLabel,
};

// ============================================================================
// Preprocessing (Normalization & Sampling)
// ============================================================================

pub use crate::preprocessing::{
    AdaptiveVolumeThreshold,
    BilinearNormalizer,
    EventBasedSampler,
    GlobalZScoreNormalizer,
    MinMaxNormalizer,
    // Normalization trait and implementations
    Normalizer,
    PerFeatureNormalizer,

    PercentageChangeNormalizer,
    RollingZScoreNormalizer,
    VolatilityEstimator,
    // Sampling
    VolumeBasedSampler,
    ZScoreNormalizer,
};

// ============================================================================
// Schema (Feature Definitions & Presets)
// ============================================================================

pub use crate::schema::{FeatureCategory, FeatureDef, FeatureSchema, Preset, PresetConfig};

// ============================================================================
// Export
// ============================================================================

pub use crate::export::{
    export_to_numpy, BatchExportResult, BatchExporter, DayExportResult, ExportMetadata,
    NumpyExporter, SplitConfig,
};

pub use crate::export_aligned::{AlignedBatchExporter, AlignedDayExport};

// ============================================================================
// Validation
// ============================================================================

pub use crate::validation::{
    validate_timestamps, FeatureValidator, ValidationConfig, ValidationLevel, ValidationResult,
};

// ============================================================================
// From MBO-LOB-Reconstructor (re-exported)
// ============================================================================

pub use mbo_lob_reconstructor::{
    Action,
    CrossedQuotePolicy,

    // Statistics
    DayStats,
    // Analytics
    DepthStats,
    LiquidityMetrics,
    LobConfig,
    // LOB reconstruction
    LobReconstructor,
    // Core types
    LobState,
    MarketImpact,

    MboMessage,
    NormalizationParams,

    // Error handling
    Result,
    Side,

    TlobError,
};

// Conditionally re-export Databento support
#[cfg(feature = "databento")]
pub use mbo_lob_reconstructor::{DbnBridge, DbnLoader, LoaderStats};

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

/// Feature vector type (single snapshot)
pub type FeatureVector = Vec<f64>;

/// Feature matrix type (multiple snapshots)
pub type FeatureMatrix = Vec<Vec<f64>>;

/// Label with index and price change
pub type LabeledSample = (usize, TrendLabel, f64);

// ============================================================================
// Batch Processing (parallel feature)
// ============================================================================

#[cfg(feature = "parallel")]
pub use crate::batch::{
    process_files_parallel, process_files_with_threads, BatchConfig, BatchOutput, BatchProcessor,
    CancellationToken, ConsoleProgress, DayResult, ErrorMode, FileError, ProgressCallback,
    ProgressInfo,
};
