//! Unified error type for the feature-extractor crate.
//!
//! Provides [`FeatureExtractorError`] as the single error enum for all
//! failure modes across the pipeline: LOB reconstruction, feature extraction,
//! sequence building, export, configuration, and numerical computation.
//!
//! Automatic conversion from dependency error types via `#[from]`:
//! - [`mbo_lob_reconstructor::TlobError`] → `FeatureExtractorError::Lob`
//! - [`std::io::Error`] → `FeatureExtractorError::Io`
//! - [`SequenceError`] → `FeatureExtractorError::Sequence`
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::error::{FeatureExtractorError, Result};
//!
//! fn process_data() -> Result<()> {
//!     // TlobError automatically converts via ?
//!     let output = pipeline.process("data.dbn.zst")?;
//!
//!     // I/O errors automatically convert via ?
//!     let file = File::create("output.npy")?;
//!
//!     // Domain-specific errors use explicit construction
//!     if features.len() != expected {
//!         return Err(FeatureExtractorError::Contract {
//!             msg: format!("Expected {} features, got {}", expected, features.len()),
//!         });
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Design Decisions
//!
//! - **`thiserror`** for derive macros (consistent with mbo-lob-reconstructor)
//! - **Structured variants** for domain errors (Feature, Config, Contract, Numerical)
//!   instead of a single `String` variant — enables pattern matching on error types
//! - **`Other` escape hatch** for edge cases — use sparingly, prefer specific variants
//! - **`#[from]` conversions** for dependency errors — enables `?` propagation

use thiserror::Error;

use crate::sequence_builder::SequenceError;

/// Unified result type for the feature-extractor crate.
///
/// Alias for `std::result::Result<T, FeatureExtractorError>`.
pub type Result<T> = std::result::Result<T, FeatureExtractorError>;

/// Unified error type for the feature-extractor crate.
///
/// Covers all failure modes across the pipeline:
/// - LOB reconstruction errors (from mbo-lob-reconstructor)
/// - I/O errors (file operations, NPY export)
/// - Sequence building errors (feature count mismatch, buffer overflow)
/// - Feature extraction errors (NaN, invalid state)
/// - Configuration errors (invalid parameters)
/// - Contract violations (schema mismatch, feature count)
/// - Numerical errors (division by zero, overflow)
#[derive(Error, Debug)]
pub enum FeatureExtractorError {
    /// Error from LOB reconstruction (MBO-LOB-reconstructor crate).
    ///
    /// Includes: invalid orders, crossed quotes, locked quotes, etc.
    #[error("LOB reconstruction: {0}")]
    Lob(#[from] mbo_lob_reconstructor::TlobError),

    /// I/O error (file operations, NPY export, directory creation).
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Sequence building error (feature count mismatch, buffer issues).
    #[error("Sequence: {0}")]
    Sequence(#[from] SequenceError),

    /// Feature extraction error (NaN in features, signal computation failure).
    ///
    /// Used when the extraction pipeline encounters invalid data or state.
    #[error("Feature extraction: {msg}")]
    Feature {
        /// Describes what went wrong during feature extraction.
        msg: String,
    },

    /// Configuration error (invalid parameters, incompatible settings).
    ///
    /// Used when config validation fails at initialization time.
    #[error("Configuration: {msg}")]
    Config {
        /// Describes the configuration issue.
        msg: String,
    },

    /// Contract violation (schema mismatch, feature count, label encoding).
    ///
    /// Used when exported data doesn't match the pipeline contract.
    #[error("Contract violation: {msg}")]
    Contract {
        /// Describes the contract violation.
        msg: String,
    },

    /// Numerical error (division by zero, overflow, NaN propagation).
    ///
    /// Used for computational failures in normalization, labeling, etc.
    #[error("Numerical: {msg}")]
    Numerical {
        /// Describes the numerical issue.
        msg: String,
    },

    /// Generic error (escape hatch — prefer specific variants).
    #[error("{0}")]
    Other(String),
}

// Convenience constructors for common patterns
impl FeatureExtractorError {
    /// Create a feature extraction error.
    #[inline]
    pub fn feature(msg: impl Into<String>) -> Self {
        Self::Feature { msg: msg.into() }
    }

    /// Create a configuration error.
    #[inline]
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config { msg: msg.into() }
    }

    /// Create a contract violation error.
    #[inline]
    pub fn contract(msg: impl Into<String>) -> Self {
        Self::Contract { msg: msg.into() }
    }

    /// Create a numerical error.
    #[inline]
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical { msg: msg.into() }
    }
}

// Allow converting from format strings that produce io::Error
impl From<String> for FeatureExtractorError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for FeatureExtractorError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}
