//! Feature preprocessing for TLOB transformer.
//!
//! This module contains all preprocessing operations applied to features
//! before sequence generation:
//!
//! - **Sampling**: Determine which LOB snapshots to use
//!   - Volume-based sampling (TLOB paper recommendation)
//!   - Event-based sampling (legacy)
//!   - Multi-timescale sampling (future)
//!
//! - **Normalization**: Scale features appropriately
//!   - Percentage change (for prices)
//!   - Z-score (for volumes, spreads)
//!   - Bilinear (for LOB structure)
//!   - Min-max (for bounded features)
//!
//! - **Volatility Estimation**: Adaptive sampling support
//!   - Rolling realized volatility (Welford's algorithm)
//!   - Numerically stable variance calculation
//!
//! - **Adaptive Sampling**: Market-regime-aware thresholds
//!   - Dynamic volume threshold adjustment
//!   - Auto-calibration from initial samples
//!   - Bounded adaptation (min/max multipliers)
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::preprocessing::{
//!     VolumeBasedSampler,
//!     PercentageChangeNormalizer,
//!     Normalizer,
//! };
//!
//! // Create sampler
//! let mut sampler = VolumeBasedSampler::new(1000, 1_000_000);
//!
//! // Create normalizer
//! let mut normalizer = PercentageChangeNormalizer::new();
//! normalizer.set_reference(100.0);
//!
//! // Use in pipeline
//! if sampler.should_sample(volume, timestamp) {
//!     let normalized = normalizer.normalize(price);
//! }
//! ```

pub mod adaptive_sampling;
pub mod normalization;
pub mod sampling;
pub mod volatility;

// Re-export commonly used types for convenience
pub use adaptive_sampling::AdaptiveVolumeThreshold;
pub use normalization::{
    BilinearNormalizer, GlobalZScoreNormalizer, MinMaxNormalizer, Normalizer, PerFeatureNormalizer,
    PercentageChangeNormalizer, RollingZScoreNormalizer, ZScoreNormalizer,
};
pub use sampling::{EventBasedSampler, VolumeBasedSampler};
pub use volatility::VolatilityEstimator;
