//! Feature extraction for LOB-based deep learning models.
//!
//! This module provides efficient feature extraction from MBO data and LOB snapshots.
//! Features are designed based on research papers:
//! - DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
//! - TLOB: A Novel Transformer Model with Dual Attention
//! - FI-2010: Benchmark Dataset for Mid-Price Forecasting
//! - The Price Impact of Order Book Events (Cont et al.)
//!
//! # Architecture
//!
//! The feature extraction is organized into:
//! - `config`: Feature extraction configuration (`FeatureConfig`)
//! - `extractor`: Main orchestrator (`FeatureExtractor`)
//! - `lob_features`: Extract RAW features from LOB snapshots (40 features by default)
//! - `order_flow`: Order Flow Imbalance (OFI), queue imbalance (8 features)
//! - `derived_features`: Compute derived metrics (8 features, opt-in)
//! - `mbo_features`: Aggregate MBO data into features (36 features, opt-in)
//! - `signals`: Trading signals (14 features, opt-in)
//! - `fi2010`: FI-2010 benchmark handcrafted features (80 features)
//! - `market_impact`: Market impact estimation features
//! - `experimental`: Opt-in experimental feature groups
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::features::{FeatureExtractor, FeatureConfig};
//!
//! let config = FeatureConfig::new(10).with_derived(true).with_mbo(true);
//! let mut extractor = FeatureExtractor::with_config(config);
//! let features = extractor.extract_all_features(&lob_state)?;
//! ```

mod config;
mod extractor;

pub mod derived_features;
pub mod experimental;
pub mod fi2010;
pub mod lob_features;
pub mod market_impact;
pub mod mbo_features;
pub mod order_flow;
pub mod signals;

// Re-export primary API types
pub use config::FeatureConfig;
pub use extractor::FeatureExtractor;
pub use signals::SignalContext;
