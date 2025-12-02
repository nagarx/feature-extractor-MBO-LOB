//! Feature Extractor
//!
//! High-performance LOB and MBO feature extraction for deep learning models.
//!
//! # Overview
//!
//! This library provides a modular, research-aligned feature extraction pipeline
//! for limit order book (LOB) data. It supports multiple paper configurations:
//!
//! - **DeepLOB**: 40 raw LOB features with Z-score normalization
//! - **TLOB**: 40 raw LOB features with bilinear normalization
//! - **FI-2010**: 144 features (40 raw + 104 handcrafted)
//! - **TransLOB**: Multi-horizon feature support
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Feature Extractor                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  schema/        - Feature definitions and paper presets        │
//! │  features/      - Raw feature extraction (no normalization)    │
//! │  preprocessing/ - Normalization and sampling                   │
//! │  sequence/      - Sequence building for transformers           │
//! │  export/        - NumPy export for Python/PyTorch              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::{FeatureExtractor, schema::{Preset, FeatureSchema}};
//!
//! // Use a paper-aligned preset
//! let schema = FeatureSchema::from_preset(Preset::DeepLOB);
//! let extractor = FeatureExtractor::new(schema.levels);
//!
//! // Extract features from LOB state
//! let features = extractor.extract_lob_features(&lob_state)?;
//! ```

pub mod config;
pub mod export;
pub mod export_aligned;
pub mod features;
pub mod labeling;
pub mod pipeline;
pub mod preprocessing;
pub mod schema;
pub mod sequence_builder;
pub mod validation;

// Re-exports - Schema
pub use schema::{FeatureCategory, FeatureDef, FeatureSchema, Preset, PresetConfig};

// Re-exports - Config
pub use config::{ExperimentMetadata, PipelineConfig, SamplingConfig, SamplingStrategy};

// Re-exports - Features
pub use features::fi2010::{FI2010Config, FI2010Extractor};
pub use features::market_impact::{
    estimate_buy_impact, estimate_sell_impact, MarketImpactFeatures,
};
pub use features::mbo_features::{MboAggregator, MboEvent};
pub use features::order_flow::{MultiLevelOfiTracker, OrderFlowFeatures, OrderFlowTracker};
pub use features::{FeatureConfig, FeatureExtractor};

// Re-exports - Preprocessing
pub use preprocessing::{
    AdaptiveVolumeThreshold, BilinearNormalizer, EventBasedSampler, GlobalZScoreNormalizer,
    MinMaxNormalizer, Normalizer, PerFeatureNormalizer, PercentageChangeNormalizer,
    RollingZScoreNormalizer, VolatilityEstimator, VolumeBasedSampler, ZScoreNormalizer,
};

// Re-exports - Sequence Building
pub use sequence_builder::{Sequence, SequenceBuilder, SequenceConfig};

// Re-exports - Export
pub use export::{
    export_to_numpy, BatchExportResult, BatchExporter, DayExportResult, ExportMetadata,
    NumpyExporter, SplitConfig,
};
pub use export_aligned::{AlignedBatchExporter, AlignedDayExport};

// Re-exports - Validation
pub use validation::{
    validate_timestamps, FeatureValidator, ValidationConfig, ValidationLevel, ValidationResult,
};

// Re-exports - Labeling
pub use labeling::{
    DeepLobLabelGenerator, DeepLobMethod, LabelConfig, LabelGenerator, LabelStats,
    TlobLabelGenerator, TrendLabel,
};

// Re-exports - Pipeline
pub use mbo_lob_reconstructor::{LobState, Result};
pub use pipeline::{Pipeline, PipelineOutput};
