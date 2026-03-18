//! Export infrastructure: configuration types + formatting utilities.
//!
//! # Module Structure
//!
//! - **config/** — TOML/JSON-serializable configuration types for experiment reproducibility.
//!   Shared by `export_aligned/`, the CLI tool (`export_dataset`), and tests.
//!   Includes: `DatasetConfig`, `NormalizationConfig`, `ExportLabelConfig`, etc.
//!
//! - **tensor_format** — Tensor reshaping for model-specific formats (DeepLOB, HLOB, Image, Flat).
//!   Used by `export_aligned/npy_export.rs` for optional format transformations.
//!
//! # Architecture Note
//!
//! The actual export ENGINE is in `export_aligned/`. This module provides the
//! configuration and utility types that `export_aligned` consumes. This is
//! intentional separation of concerns (config vs implementation), NOT a
//! half-finished migration. If a new exporter is added (e.g., Parquet, streaming),
//! it would also consume these shared config types.

pub mod config;
pub mod tensor_format;

pub use config::{
    DataPathConfig, DatasetConfig, DateRangeConfig, ExperimentInfo, ExportConflictPriority,
    ExportLabelConfig, ExportSamplingConfig, ExportSequenceConfig, ExportThresholdStrategy,
    ExportTimeoutStrategy, FeatureNormStrategy, FeatureSetConfig, LabelingStrategy,
    NormalizationConfig, ProcessingConfig, SamplingStrategyConfig, SplitConfig, SymbolConfig,
};

pub use tensor_format::{FeatureMapping, TensorFormat, TensorFormatter, TensorOutput};
