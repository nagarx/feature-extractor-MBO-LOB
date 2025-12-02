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
pub use crate::pipeline::{Pipeline, PipelineOutput};
pub use crate::config::{
    PipelineConfig, 
    SamplingConfig, 
    SamplingStrategy,
    ExperimentMetadata,
    AdaptiveSamplingConfig,
    MultiScaleConfig,
};

// ============================================================================
// Feature Extraction
// ============================================================================

pub use crate::features::{FeatureConfig, FeatureExtractor};
pub use crate::features::mbo_features::{MboAggregator, MboEvent};
pub use crate::features::order_flow::{OrderFlowFeatures, OrderFlowTracker};
pub use crate::features::fi2010::{FI2010Config, FI2010Extractor};

// ============================================================================
// Sequence Building
// ============================================================================

pub use crate::sequence_builder::{
    Sequence, 
    SequenceBuilder, 
    SequenceConfig,
    SequenceError,
    HorizonAwareConfig,
    MultiScaleConfig as MultiScaleSequenceConfig,
    MultiScaleSequence,
    MultiScaleWindow,
};

// ============================================================================
// Labeling
// ============================================================================

pub use crate::labeling::{
    LabelConfig, 
    LabelGenerator, 
    LabelStats,
    TlobLabelGenerator,
    DeepLobLabelGenerator,
    DeepLobMethod,
    TrendLabel,
};

// ============================================================================
// Preprocessing (Normalization & Sampling)
// ============================================================================

pub use crate::preprocessing::{
    // Normalization trait and implementations
    Normalizer,
    ZScoreNormalizer,
    RollingZScoreNormalizer,
    GlobalZScoreNormalizer,
    BilinearNormalizer,
    MinMaxNormalizer,
    PercentageChangeNormalizer,
    PerFeatureNormalizer,
    
    // Sampling
    VolumeBasedSampler,
    EventBasedSampler,
    AdaptiveVolumeThreshold,
    VolatilityEstimator,
};

// ============================================================================
// Schema (Feature Definitions & Presets)
// ============================================================================

pub use crate::schema::{
    Preset,
    PresetConfig,
    FeatureSchema,
    FeatureDef,
    FeatureCategory,
};

// ============================================================================
// Export
// ============================================================================

pub use crate::export::{
    NumpyExporter,
    BatchExporter,
    ExportMetadata,
    DayExportResult,
    BatchExportResult,
    SplitConfig,
    export_to_numpy,
};

pub use crate::export_aligned::{
    AlignedBatchExporter,
    AlignedDayExport,
};

// ============================================================================
// Validation
// ============================================================================

pub use crate::validation::{
    FeatureValidator,
    ValidationResult,
    ValidationConfig,
    ValidationLevel,
    validate_timestamps,
};

// ============================================================================
// From MBO-LOB-Reconstructor (re-exported)
// ============================================================================

pub use mbo_lob_reconstructor::{
    // Core types
    LobState,
    MboMessage,
    Action,
    Side,
    
    // LOB reconstruction
    LobReconstructor,
    LobConfig,
    CrossedQuotePolicy,
    
    // Statistics
    DayStats,
    NormalizationParams,
    
    // Analytics
    DepthStats,
    LiquidityMetrics,
    MarketImpact,
    
    // Error handling
    Result,
    TlobError,
};

// Conditionally re-export Databento support
#[cfg(feature = "databento")]
pub use mbo_lob_reconstructor::{DbnLoader, LoaderStats, DbnBridge};

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

/// Feature vector type (single snapshot)
pub type FeatureVector = Vec<f64>;

/// Feature matrix type (multiple snapshots)
pub type FeatureMatrix = Vec<Vec<f64>>;

/// Label with index and price change
pub type LabeledSample = (usize, TrendLabel, f64);

