//! Sequence generation for transformer models.
//!
//! This module provides efficient sequence generation from feature snapshots.
//! The TLOB model requires sequences of LOB snapshots (typically 100 snapshots)
//! rather than individual snapshots.
//!
//! # Architecture
//!
//! - **SequenceBuilder**: Main builder with circular buffer
//! - **SequenceConfig**: Configuration for window size, stride, etc.
//! - **Sequence**: Output structure containing features and metadata
//! - **HorizonAwareConfig**: Automatic lookback window scaling by horizon
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::sequence_builder::{SequenceBuilder, SequenceConfig};
//!
//! let config = SequenceConfig::new(100, 1); // 100 snapshots, stride 1
//! let mut builder = SequenceBuilder::with_config(config);
//!
//! // Add features as they're sampled
//! for (timestamp, features) in sampled_features {
//!     builder.push(timestamp, features);
//!     
//!     // Try to get a complete sequence
//!     if let Some(seq) = builder.try_build_sequence() {
//!         // Feed to transformer
//!         model.predict(seq.features);
//!     }
//! }
//! ```

mod builder;
pub mod horizon_aware;
pub mod multiscale;

// Re-export all public types
pub use builder::{FeatureVec, Sequence, SequenceBuilder, SequenceConfig, SequenceError};
pub use horizon_aware::HorizonAwareConfig;
pub use multiscale::{MultiScaleConfig, MultiScaleSequence, MultiScaleWindow, ScaleConfig};
