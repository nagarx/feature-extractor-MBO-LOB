//! Data Export Module
//!
//! Export processed features and labels to various formats for ML training.
//!
//! # Modules
//!
//! - **config**: Configuration-driven, symbol-agnostic export system
//! - **dataset_config**: Dataset configuration types
//! - **tensor_format**: Model-specific tensor reshaping (DeepLOB, HLOB, etc.)
//!
//! # Current Export Path
//!
//! All new exports should use `AlignedBatchExporter` from the `export_aligned` module.

pub mod config;
pub mod dataset_config;
pub mod tensor_format;

pub use config::{
    DataPathConfig, DatasetConfig, DateRangeConfig, ExperimentInfo, ExportConflictPriority,
    ExportLabelConfig, ExportSamplingConfig, ExportSequenceConfig, ExportThresholdStrategy,
    ExportTimeoutStrategy, FeatureNormStrategy, FeatureSetConfig, LabelingStrategy,
    NormalizationConfig, ProcessingConfig, SamplingStrategyConfig, SplitConfig, SymbolConfig,
};

pub use tensor_format::{FeatureMapping, TensorFormat, TensorFormatter, TensorOutput};
