//! Trading Signal Layer
//!
//! Computes derived trading signals from raw LOB and MBO features.
//! These signals form indices 84-97 in the feature vector.
//!
//! # Architecture
//!
//! The signal layer is designed as a thin computation layer that:
//! 1. Takes existing features (raw LOB, derived, MBO) as input
//! 2. Computes higher-level trading signals
//! 3. Outputs a fixed-size signal vector
//!
//! # Signal Categories
//!
//! - **Safety Gates**: `book_valid`, `mbo_ready` - must pass before any trading
//! - **Direction Signals**: `true_ofi`, `depth_norm_ofi`, `executed_pressure`
//! - **Confirmation Signals**: `trade_asymmetry`, `cancel_asymmetry`
//! - **Impact Signals**: `fragility_score`, `depth_asymmetry`
//! - **Timing Signals**: `signed_mp_delta_bps`, `time_regime`
//! - **Meta Signals**: `dt_seconds`, `invalidity_delta`, `schema_version`
//!
//! # Research Foundation
//!
//! - OFI: Cont, Kukanov & Stoikov (2014) "The Price Impact of Order Book Events"
//! - Microprice: Stoikov (2018) "The Micro-Price"
//! - Time regimes: Cont et al. §3.3 (intraday price impact patterns)
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::features::signals::{OfiComputer, TimeRegime};
//!
//! let mut computer = OfiComputer::new();
//!
//! // Update on every LOB state transition
//! computer.update(&lob_state);
//!
//! // At sampling points, sample OFI and compute signals
//! let ofi_sample = computer.sample_and_reset(timestamp_ns);
//! let signals = compute_signals(&features_84, &ofi_sample, timestamp_ns, invalidity_delta);
//! ```

mod book_valid;
mod compute;
pub mod indices;
mod ofi;
mod time_regime;

pub use book_valid::{is_book_valid, is_book_valid_from_lob};
pub use compute::{compute_signals, compute_signals_with_book_valid, SignalVector};
pub use ofi::{OfiComputer, OfiSample, MIN_WARMUP_STATE_CHANGES};
pub use time_regime::{compute_time_regime, compute_time_regime_auto, N_REGIMES, REGIME_LABELS};

// ============================================================================
// Signal Context
// ============================================================================

/// Context required for signal computation.
///
/// This struct encapsulates the additional context needed for computing
/// trading signals beyond the LOB state and base features.
///
/// # Fields
///
/// * `timestamp_ns` - Current timestamp in nanoseconds (UTC since epoch)
/// * `invalidity_delta` - Count of crossed/locked book events since last sample
#[derive(Debug, Clone, Copy, Default)]
pub struct SignalContext {
    /// Current timestamp in nanoseconds (UTC since epoch).
    ///
    /// Used for:
    /// - Computing `time_regime` (market session detection)
    /// - Computing `dt_seconds` (time since last sample)
    pub timestamp_ns: u64,

    /// Count of crossed/locked book events since last sample.
    ///
    /// Crossed quotes occur when best_bid >= best_ask, indicating
    /// feed problems or latency issues. Higher values suggest
    /// unreliable LOB state.
    pub invalidity_delta: u64,
}

impl SignalContext {
    /// Create a new signal context.
    pub fn new(timestamp_ns: u64, invalidity_delta: u64) -> Self {
        Self {
            timestamp_ns,
            invalidity_delta,
        }
    }
}
