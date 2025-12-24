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
//! use feature_extractor::features::signals::{SignalComputer, TimeRegime};
//!
//! let mut computer = SignalComputer::new();
//!
//! // Update on every LOB state transition
//! computer.update_ofi(&lob_state);
//!
//! // At sampling points, compute all signals
//! let signals = computer.compute_signals(&features_84, timestamp_ns, invalidity_delta);
//! ```

use mbo_lob_reconstructor::LobState;

// ============================================================================
// Constants
// ============================================================================

/// Epsilon for avoiding division by zero
const EPSILON: f64 = 1e-10;

/// Schema version for forward compatibility.
///
/// Version history:
/// - 2.0: Initial signal layer implementation
/// - 2.1: Fixed sign convention for net_trade_flow (index 56) and net_cancel_flow (index 55)
///        to follow RULE.md §9: > 0 = BULLISH, < 0 = BEARISH
pub const SCHEMA_VERSION: f64 = 2.1;

/// Number of signals in the signal layer (indices 84-97)
pub const SIGNAL_COUNT: usize = 14;

/// Minimum effective state changes before OFI is considered "warm"
/// This counts ACTUAL LOB state transitions, not raw messages
pub const MIN_WARMUP_STATE_CHANGES: u64 = 100;

/// Nanoseconds per second
const NS_PER_SECOND: i64 = 1_000_000_000;

/// Nanoseconds per hour
const NS_PER_HOUR: i64 = 3_600_000_000_000;

/// US Eastern Time offset from UTC (standard time: -5 hours)
const ET_OFFSET_STANDARD_NS: i64 = -5 * NS_PER_HOUR;

/// US Eastern Time offset from UTC (daylight saving: -4 hours)
const ET_OFFSET_DST_NS: i64 = -4 * NS_PER_HOUR;

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

// ============================================================================
// Time Regime
// ============================================================================

/// Market session time regime.
///
/// Based on Cont et al. (2014) §3.3: "Price impact is five times higher
/// at the market open compared to the market close."
///
/// Different regimes require different thresholds to prevent false signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TimeRegime {
    /// Market open (9:30-9:45 ET): Highest volatility, widest spreads
    Open = 0,
    /// Early session (9:45-10:30 ET): Settling period
    Early = 1,
    /// Midday (10:30-15:30 ET): Most stable trading period
    Midday = 2,
    /// Market close (15:30-16:00 ET): Position squaring, increased activity
    Close = 3,
    /// Outside market hours
    Closed = 4,
}

impl TimeRegime {
    /// Convert to f64 for feature vector.
    #[inline]
    pub fn as_f64(self) -> f64 {
        self as u8 as f64
    }
}

/// Compute time regime from UTC nanosecond timestamp.
///
/// Handles UTC to Eastern Time conversion with DST approximation.
///
/// # Arguments
///
/// * `timestamp_ns` - Unix timestamp in nanoseconds (UTC)
///
/// # Returns
///
/// The `TimeRegime` for the given timestamp.
///
/// # DST Handling
///
/// Uses a calendar-based approximation for DST:
/// - DST begins: Second Sunday of March (approx day 70-80)
/// - DST ends: First Sunday of November (approx day 305-315)
///
/// This is an approximation. For production trading, consider using
/// a proper timezone library like `chrono-tz`.
#[inline]
pub fn compute_time_regime(timestamp_ns: i64) -> TimeRegime {
    // Convert UTC timestamp to ET
    let et_offset = estimate_et_offset(timestamp_ns);
    let et_timestamp_ns = timestamp_ns + et_offset;

    // Extract hour and minute in ET
    // Seconds since midnight ET
    let seconds_since_midnight = ((et_timestamp_ns / NS_PER_SECOND) % 86400 + 86400) % 86400;
    let hour = (seconds_since_midnight / 3600) as u8;
    let minute = ((seconds_since_midnight % 3600) / 60) as u8;

    compute_time_regime_from_et(hour, minute)
}

/// Compute time regime from ET hour and minute.
///
/// Separated for testability - allows direct testing without timestamp conversion.
#[inline]
pub fn compute_time_regime_from_et(hour: u8, minute: u8) -> TimeRegime {
    match (hour, minute) {
        // Before market open
        (0..=8, _) => TimeRegime::Closed,
        (9, 0..=29) => TimeRegime::Closed,

        // OPEN: 9:30-9:45 ET - Highest volatility
        (9, 30..=44) => TimeRegime::Open,

        // EARLY: 9:45-10:30 ET - Settling period
        (9, 45..=59) => TimeRegime::Early,
        (10, 0..=29) => TimeRegime::Early,

        // MIDDAY: 10:30-15:30 ET - Most stable
        (10, 30..=59) => TimeRegime::Midday,
        (11..=14, _) => TimeRegime::Midday,
        (15, 0..=29) => TimeRegime::Midday,

        // CLOSE: 15:30-16:00 ET - Position squaring
        (15, 30..=59) => TimeRegime::Close,

        // After market close
        _ => TimeRegime::Closed,
    }
}

/// Estimate ET offset from UTC based on approximate DST rules.
///
/// Returns nanoseconds to add to UTC to get ET.
#[inline]
fn estimate_et_offset(timestamp_ns: i64) -> i64 {
    // Approximate day of year (0-365)
    // Unix epoch (1970-01-01) was a Thursday
    // We only need approximate day of year for DST detection
    let seconds = timestamp_ns / NS_PER_SECOND;
    let days_since_epoch = seconds / 86400;

    // Approximate day of year (very rough, but sufficient for DST detection)
    // This ignores leap years but that's fine for regime classification
    let day_of_year = (days_since_epoch % 365) as i32;

    // DST in US: roughly March 10 (day ~70) to November 3 (day ~307)
    // Using conservative bounds to avoid edge cases
    if day_of_year >= 70 && day_of_year < 307 {
        ET_OFFSET_DST_NS
    } else {
        ET_OFFSET_STANDARD_NS
    }
}

// ============================================================================
// Book Validity
// ============================================================================

/// Check if the order book is valid for trading.
///
/// A valid book has:
/// 1. Both bid and ask prices present
/// 2. Bid < Ask (not crossed)
/// 3. Positive spread
///
/// # Arguments
///
/// * `best_bid` - Best bid price (None if empty)
/// * `best_ask` - Best ask price (None if empty)
///
/// # Returns
///
/// `true` if book is valid, `false` otherwise.
#[inline]
pub fn is_book_valid(best_bid: Option<i64>, best_ask: Option<i64>) -> bool {
    match (best_bid, best_ask) {
        (Some(bid), Some(ask)) => bid < ask,
        _ => false,
    }
}

/// Check if book is valid from LobState.
#[inline]
pub fn is_book_valid_from_lob(lob: &LobState) -> bool {
    is_book_valid(lob.best_bid, lob.best_ask)
}

// ============================================================================
// OFI Computer
// ============================================================================

/// Sample returned at sampling points.
///
/// Contains OFI and metadata accumulated since last sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct OfiSample {
    /// Accumulated OFI since last sample (positive = buy pressure)
    pub ofi: f64,

    /// OFI from bid side only
    pub ofi_bid: f64,

    /// OFI from ask side only
    pub ofi_ask: f64,

    /// Average depth (bid + ask L0 size) over sampling interval
    pub avg_depth: f64,

    /// Number of state transitions in this interval
    pub event_count: u64,

    /// Whether MBO/OFI has sufficient warmup
    pub is_warm: bool,

    /// Seconds since last sample
    pub dt_seconds: f64,
}

impl OfiSample {
    /// Compute depth-normalized OFI.
    ///
    /// Per Cont et al. (2014): β ∝ 1/AD, so we normalize by average depth.
    #[inline]
    pub fn depth_norm_ofi(&self) -> f64 {
        if self.avg_depth > EPSILON {
            self.ofi / self.avg_depth
        } else {
            0.0
        }
    }
}

/// Streaming OFI computer per Cont et al. (2014).
///
/// Accumulates OFI on every LOB state transition and provides
/// sample-and-reset functionality for discrete sampling.
///
/// # Key Features
///
/// - **Streaming accumulation**: Updates on every state change
/// - **Warmup tracking**: Counts effective state changes for `mbo_ready`
/// - **Sample duration**: Tracks `dt_seconds` between samples
/// - **Clear handling**: Resets on `Action::Clear` events
///
/// # OFI Formula (Cont et al. 2014, §2.1)
///
/// For each state transition n:
/// ```text
/// e_n = bid_contribution - ask_contribution
///
/// where:
///   bid_contribution:
///     if curr_bid > prev_bid: +curr_bid_size  (price improved)
///     if curr_bid < prev_bid: -prev_bid_size  (price dropped)
///     else: curr_bid_size - prev_bid_size     (size change only)
///
///   ask_contribution (note: inverted sign):
///     if curr_ask < prev_ask: +curr_ask_size  (price improved)
///     if curr_ask > prev_ask: -prev_ask_size  (price dropped)
///     else: curr_ask_size - prev_ask_size     (size change only)
///
/// OFI = Σ e_n over sampling interval
/// ```
#[derive(Debug, Clone)]
pub struct OfiComputer {
    // Previous state for delta computation
    prev_bid_price: Option<i64>,
    prev_bid_size: u32,
    prev_ask_price: Option<i64>,
    prev_ask_size: u32,

    // Accumulators (reset at each sample)
    cumulative_ofi_bid: f64,
    cumulative_ofi_ask: f64,
    depth_sum: f64,
    event_count: u64,

    // Warmup tracking (persists across samples, reset on Clear)
    state_changes_since_reset: u64,

    // Sample duration tracking
    last_sample_timestamp_ns: Option<i64>,
}

impl Default for OfiComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl OfiComputer {
    /// Create a new OFI computer.
    pub fn new() -> Self {
        Self {
            prev_bid_price: None,
            prev_bid_size: 0,
            prev_ask_price: None,
            prev_ask_size: 0,
            cumulative_ofi_bid: 0.0,
            cumulative_ofi_ask: 0.0,
            depth_sum: 0.0,
            event_count: 0,
            state_changes_since_reset: 0,
            last_sample_timestamp_ns: None,
        }
    }

    /// Update OFI with a new LOB state.
    ///
    /// **CRITICAL**: Call this on EVERY LOB state transition, not just at sampling points.
    ///
    /// Only counts as a "state change" for warmup if L0 prices/sizes actually changed.
    #[inline]
    pub fn update(&mut self, lob: &LobState) {
        let curr_bid_price = lob.best_bid;
        let curr_ask_price = lob.best_ask;
        let curr_bid_size = lob.bid_sizes[0];
        let curr_ask_size = lob.ask_sizes[0];

        // Accumulate depth for averaging
        self.depth_sum += (curr_bid_size + curr_ask_size) as f64;
        self.event_count += 1;

        // Skip if no previous state (first call after reset)
        let (prev_bid, prev_ask) = match (self.prev_bid_price, self.prev_ask_price) {
            (Some(b), Some(a)) => (b, a),
            _ => {
                // Initialize previous state and return
                self.prev_bid_price = curr_bid_price;
                self.prev_bid_size = curr_bid_size;
                self.prev_ask_price = curr_ask_price;
                self.prev_ask_size = curr_ask_size;
                return;
            }
        };

        // Handle missing current prices (empty book)
        let (curr_bid, curr_ask) = match (curr_bid_price, curr_ask_price) {
            (Some(b), Some(a)) => (b, a),
            _ => {
                // Book became empty, just update prev state
                self.prev_bid_price = curr_bid_price;
                self.prev_bid_size = curr_bid_size;
                self.prev_ask_price = curr_ask_price;
                self.prev_ask_size = curr_ask_size;
                return;
            }
        };

        // Check if this is an ACTUAL state change (not a no-op)
        let bid_changed = curr_bid != prev_bid || curr_bid_size != self.prev_bid_size;
        let ask_changed = curr_ask != prev_ask || curr_ask_size != self.prev_ask_size;

        if bid_changed || ask_changed {
            self.state_changes_since_reset += 1;
        }

        // === BID SIDE CONTRIBUTION (demand) ===
        let ofi_bid = if curr_bid > prev_bid {
            // Price improved: new demand at better price
            curr_bid_size as f64
        } else if curr_bid < prev_bid {
            // Price dropped: demand removed
            -(self.prev_bid_size as f64)
        } else {
            // Same price: net size change
            (curr_bid_size as i64 - self.prev_bid_size as i64) as f64
        };

        // === ASK SIDE CONTRIBUTION (supply) ===
        // Note: signs are inverted because ask improvement means price DOWN
        let ofi_ask = if curr_ask < prev_ask {
            // Price improved (lower ask): supply at better price
            curr_ask_size as f64
        } else if curr_ask > prev_ask {
            // Price worsened (higher ask): supply removed
            -(self.prev_ask_size as f64)
        } else {
            // Same price: net size change
            (curr_ask_size as i64 - self.prev_ask_size as i64) as f64
        };

        // Accumulate
        self.cumulative_ofi_bid += ofi_bid;
        self.cumulative_ofi_ask -= ofi_ask; // Subtract ask contribution

        // Update previous state
        self.prev_bid_price = curr_bid_price;
        self.prev_bid_size = curr_bid_size;
        self.prev_ask_price = curr_ask_price;
        self.prev_ask_size = curr_ask_size;
    }

    /// Sample accumulated OFI and reset accumulators.
    ///
    /// Call this at each sampling point to get the OFI for the interval.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    ///
    /// # Returns
    ///
    /// An `OfiSample` containing OFI and metadata for the interval.
    #[inline]
    pub fn sample_and_reset(&mut self, timestamp_ns: i64) -> OfiSample {
        // Compute dt_seconds
        let dt_seconds = match self.last_sample_timestamp_ns {
            Some(prev_ts) => (timestamp_ns - prev_ts) as f64 / NS_PER_SECOND as f64,
            None => 0.0,
        };

        // Compute average depth
        let avg_depth = if self.event_count > 0 {
            self.depth_sum / self.event_count as f64
        } else {
            0.0
        };

        let sample = OfiSample {
            ofi: self.cumulative_ofi_bid + self.cumulative_ofi_ask,
            ofi_bid: self.cumulative_ofi_bid,
            ofi_ask: self.cumulative_ofi_ask,
            avg_depth,
            event_count: self.event_count,
            is_warm: self.is_warm(),
            dt_seconds,
        };

        // Reset accumulators (but NOT warmup counter)
        self.cumulative_ofi_bid = 0.0;
        self.cumulative_ofi_ask = 0.0;
        self.depth_sum = 0.0;
        self.event_count = 0;
        self.last_sample_timestamp_ns = Some(timestamp_ns);

        sample
    }

    /// Reset on Clear action (day boundary, session reset).
    ///
    /// Resets ALL state including warmup counter.
    pub fn reset_on_clear(&mut self) {
        self.prev_bid_price = None;
        self.prev_bid_size = 0;
        self.prev_ask_price = None;
        self.prev_ask_size = 0;
        self.cumulative_ofi_bid = 0.0;
        self.cumulative_ofi_ask = 0.0;
        self.depth_sum = 0.0;
        self.event_count = 0;
        self.state_changes_since_reset = 0;
        self.last_sample_timestamp_ns = None;
    }

    /// Check if OFI has sufficient warmup.
    ///
    /// Returns `true` if at least `MIN_WARMUP_STATE_CHANGES` effective
    /// state changes have occurred since the last reset.
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.state_changes_since_reset >= MIN_WARMUP_STATE_CHANGES
    }

    /// Get count of state changes since last reset.
    #[inline]
    pub fn state_changes_since_reset(&self) -> u64 {
        self.state_changes_since_reset
    }
}

// ============================================================================
// Signal Computer (Main API)
// ============================================================================

/// Complete signal vector computed at each sampling point.
///
/// Contains all 14 signals (indices 84-97).
#[derive(Debug, Clone, Default)]
pub struct SignalVector {
    /// Signals array [14 elements]
    pub signals: [f64; SIGNAL_COUNT],
}

impl SignalVector {
    /// Create a new signal vector.
    pub fn new() -> Self {
        Self {
            signals: [0.0; SIGNAL_COUNT],
        }
    }

    /// Convert to Vec for appending to feature vector.
    #[inline]
    pub fn to_vec(&self) -> Vec<f64> {
        self.signals.to_vec()
    }

    /// Get signal by absolute index (84-97).
    ///
    /// Returns `None` if index is out of range.
    #[inline]
    pub fn get(&self, absolute_index: usize) -> Option<f64> {
        if absolute_index >= indices::TRUE_OFI && absolute_index <= indices::SCHEMA_VERSION {
            Some(self.signals[absolute_index - indices::TRUE_OFI])
        } else {
            None
        }
    }
}

/// Compute all 14 derived signals from base features and OFI sample.
///
/// This is the main API for signal computation at each sampling point.
///
/// # Arguments
///
/// * `base_features` - The 84 base features (raw LOB + derived + MBO)
/// * `ofi_sample` - OFI sample from `OfiComputer::sample_and_reset()`
/// * `timestamp_ns` - Current timestamp in nanoseconds (UTC)
/// * `invalidity_delta` - Count of crossed/locked events since last sample
///
/// # Returns
///
/// A `SignalVector` containing all 14 signals.
///
/// # Feature Requirements
///
/// Requires these indices in `base_features`:
/// - 40: mid_price
/// - 43: total_bid_volume
/// - 44: total_ask_volume
/// - 46: weighted_mid_price
/// - 50-51: cancel_rate_bid/ask
/// - 52-53: trade_rate_bid/ask
/// - 71: level_concentration
/// - 72-73: depth_ticks_bid/ask
pub fn compute_signals(
    base_features: &[f64],
    ofi_sample: &OfiSample,
    timestamp_ns: i64,
    invalidity_delta: u32,
) -> SignalVector {
    let mut signals = SignalVector::new();

    // Validate input length
    if base_features.len() < 84 {
        // Return default signals if input is too short
        signals.signals[indices::SCHEMA_VERSION - indices::TRUE_OFI] = SCHEMA_VERSION;
        return signals;
    }

    // Extract needed features with safe indexing
    let mid_price = base_features[derived_indices::MID_PRICE];
    let total_bid_vol = base_features[derived_indices::TOTAL_BID_VOLUME];
    let total_ask_vol = base_features[derived_indices::TOTAL_ASK_VOLUME];
    let weighted_mid = base_features[derived_indices::WEIGHTED_MID_PRICE];

    let cancel_rate_bid = base_features[mbo_indices::CANCEL_RATE_BID];
    let cancel_rate_ask = base_features[mbo_indices::CANCEL_RATE_ASK];
    let trade_rate_bid = base_features[mbo_indices::TRADE_RATE_BID];
    let trade_rate_ask = base_features[mbo_indices::TRADE_RATE_ASK];
    let level_conc = base_features[mbo_indices::LEVEL_CONCENTRATION];
    let depth_bid = base_features[mbo_indices::DEPTH_TICKS_BID];
    let depth_ask = base_features[mbo_indices::DEPTH_TICKS_ASK];

    // === Signal 84: true_ofi ===
    signals.signals[0] = ofi_sample.ofi;

    // === Signal 85: depth_norm_ofi ===
    signals.signals[1] = ofi_sample.depth_norm_ofi();

    // === Signal 86: executed_pressure ===
    // trade_ask = BUY-initiated, trade_bid = SELL-initiated
    // executed_pressure = buys - sells (> 0 = buy pressure)
    signals.signals[2] = trade_rate_ask - trade_rate_bid;

    // === Signal 87: signed_mp_delta_bps ===
    // Microprice delta in basis points
    // > 0 = microprice above mid = buy pressure
    if mid_price.abs() > EPSILON {
        signals.signals[3] = (weighted_mid - mid_price) / mid_price * 10_000.0;
    } else {
        signals.signals[3] = 0.0;
    }

    // === Signal 88: trade_asymmetry ===
    // Normalized trade pressure [-1, 1]
    let total_trades = trade_rate_bid + trade_rate_ask;
    if total_trades > EPSILON {
        signals.signals[4] = (trade_rate_ask - trade_rate_bid) / total_trades;
    } else {
        signals.signals[4] = 0.0;
    }

    // === Signal 89: cancel_asymmetry ===
    // > 0 = more ask cancels = sellers pulling quotes = bullish
    let total_cancels = cancel_rate_bid + cancel_rate_ask;
    if total_cancels > EPSILON {
        signals.signals[5] = (cancel_rate_ask - cancel_rate_bid) / total_cancels;
    } else {
        signals.signals[5] = 0.0;
    }

    // === Signal 90: fragility_score ===
    // High concentration + low depth = fragile book
    let avg_depth = (total_bid_vol + total_ask_vol) / 2.0;
    if avg_depth > 1.0 {
        signals.signals[6] = level_conc / avg_depth.ln();
    } else {
        signals.signals[6] = level_conc;
    }

    // === Signal 91: depth_asymmetry ===
    // > 0 = more bid depth = stronger support = bullish
    let total_depth = depth_bid + depth_ask;
    if total_depth > EPSILON {
        signals.signals[7] = (depth_bid - depth_ask) / total_depth;
    } else {
        signals.signals[7] = 0.0;
    }

    // === Signal 92: book_valid ===
    // This should be computed from LOB state, but we approximate from features
    // If mid_price is valid and spread > 0, book is valid
    // Note: Caller should set this directly from LOB state for accuracy
    let best_bid = base_features.get(20).copied().unwrap_or(0.0); // bid_price_0
    let best_ask = base_features.first().copied().unwrap_or(0.0); // ask_price_0
    if best_bid > EPSILON && best_ask > EPSILON && best_bid < best_ask {
        signals.signals[8] = 1.0;
    } else {
        signals.signals[8] = 0.0;
    }

    // === Signal 93: time_regime ===
    signals.signals[9] = compute_time_regime(timestamp_ns).as_f64();

    // === Signal 94: mbo_ready ===
    signals.signals[10] = if ofi_sample.is_warm { 1.0 } else { 0.0 };

    // === Signal 95: dt_seconds ===
    signals.signals[11] = ofi_sample.dt_seconds;

    // === Signal 96: invalidity_delta ===
    signals.signals[12] = invalidity_delta as f64;

    // === Signal 97: schema_version ===
    signals.signals[13] = SCHEMA_VERSION;

    signals
}

/// Compute signals with explicit book validity (preferred method).
///
/// Use this when you have direct access to LobState for accurate book_valid.
#[inline]
pub fn compute_signals_with_book_valid(
    base_features: &[f64],
    ofi_sample: &OfiSample,
    timestamp_ns: i64,
    invalidity_delta: u32,
    book_valid: bool,
) -> SignalVector {
    let mut signals = compute_signals(base_features, ofi_sample, timestamp_ns, invalidity_delta);
    signals.signals[8] = if book_valid { 1.0 } else { 0.0 };
    signals
}

// ============================================================================
// Signal Indices (for documentation and validation)
// ============================================================================

/// Signal indices in the 98-feature output vector.
///
/// These are the indices of signals computed by the signal layer.
pub mod indices {
    /// Streaming OFI per Cont et al. (unbounded, >0 = buy pressure)
    pub const TRUE_OFI: usize = 84;

    /// OFI normalized by average depth
    pub const DEPTH_NORM_OFI: usize = 85;

    /// Trade execution pressure (trade_ask - trade_bid)
    pub const EXECUTED_PRESSURE: usize = 86;

    /// Microprice delta in basis points
    pub const SIGNED_MP_DELTA_BPS: usize = 87;

    /// Normalized trade asymmetry [-1, 1]
    pub const TRADE_ASYMMETRY: usize = 88;

    /// Normalized cancel asymmetry [-1, 1]
    pub const CANCEL_ASYMMETRY: usize = 89;

    /// Book fragility score
    pub const FRAGILITY_SCORE: usize = 90;

    /// Depth asymmetry [-1, 1]
    pub const DEPTH_ASYMMETRY: usize = 91;

    /// Book validity flag (1 = valid, 0 = invalid)
    pub const BOOK_VALID: usize = 92;

    /// Time regime (0-4)
    pub const TIME_REGIME: usize = 93;

    /// MBO warmup flag (1 = ready, 0 = warming up)
    pub const MBO_READY: usize = 94;

    /// Sample duration in seconds
    pub const DT_SECONDS: usize = 95;

    /// Invalidity events since last sample
    pub const INVALIDITY_DELTA: usize = 96;

    /// Schema version (2.0)
    pub const SCHEMA_VERSION: usize = 97;
}

// ============================================================================
// MBO Feature Indices (for signal computation)
// ============================================================================

/// Indices of MBO features used for signal computation.
/// These are in the 84-feature base vector.
mod mbo_indices {
    /// Cancel rate on bid side (index in full feature vector)
    pub const CANCEL_RATE_BID: usize = 50;
    /// Cancel rate on ask side
    pub const CANCEL_RATE_ASK: usize = 51;
    /// Trade rate on bid side (SELL-initiated)
    pub const TRADE_RATE_BID: usize = 52;
    /// Trade rate on ask side (BUY-initiated)
    pub const TRADE_RATE_ASK: usize = 53;
    /// Level concentration (HHI)
    pub const LEVEL_CONCENTRATION: usize = 71;
    /// Depth ticks on bid side
    pub const DEPTH_TICKS_BID: usize = 72;
    /// Depth ticks on ask side
    pub const DEPTH_TICKS_ASK: usize = 73;
}

/// Indices of derived features used for signal computation.
mod derived_indices {
    /// Mid price
    pub const MID_PRICE: usize = 40;
    /// Total bid volume
    pub const TOTAL_BID_VOLUME: usize = 43;
    /// Total ask volume
    pub const TOTAL_ASK_VOLUME: usize = 44;
    /// Weighted mid price (microprice)
    pub const WEIGHTED_MID_PRICE: usize = 46;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Time Regime Tests
    // ========================================================================

    #[test]
    fn test_time_regime_from_et_open() {
        // 9:30 ET - Market open
        assert_eq!(compute_time_regime_from_et(9, 30), TimeRegime::Open);
        assert_eq!(compute_time_regime_from_et(9, 35), TimeRegime::Open);
        assert_eq!(compute_time_regime_from_et(9, 44), TimeRegime::Open);
    }

    #[test]
    fn test_time_regime_from_et_early() {
        // 9:45 ET - Early session starts
        assert_eq!(compute_time_regime_from_et(9, 45), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(9, 59), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(10, 0), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(10, 29), TimeRegime::Early);
    }

    #[test]
    fn test_time_regime_from_et_midday() {
        // 10:30 ET - Midday starts
        assert_eq!(compute_time_regime_from_et(10, 30), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(12, 0), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(14, 30), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(15, 0), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(15, 29), TimeRegime::Midday);
    }

    #[test]
    fn test_time_regime_from_et_close() {
        // 15:30 ET - Close session
        assert_eq!(compute_time_regime_from_et(15, 30), TimeRegime::Close);
        assert_eq!(compute_time_regime_from_et(15, 45), TimeRegime::Close);
        assert_eq!(compute_time_regime_from_et(15, 59), TimeRegime::Close);
    }

    #[test]
    fn test_time_regime_from_et_closed() {
        // Before market open
        assert_eq!(compute_time_regime_from_et(0, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(8, 59), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(9, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(9, 29), TimeRegime::Closed);

        // After market close
        assert_eq!(compute_time_regime_from_et(16, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(17, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(23, 59), TimeRegime::Closed);
    }

    #[test]
    fn test_time_regime_boundaries() {
        // Test exact boundaries
        // 9:29 -> Closed, 9:30 -> Open
        assert_eq!(compute_time_regime_from_et(9, 29), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(9, 30), TimeRegime::Open);

        // 9:44 -> Open, 9:45 -> Early
        assert_eq!(compute_time_regime_from_et(9, 44), TimeRegime::Open);
        assert_eq!(compute_time_regime_from_et(9, 45), TimeRegime::Early);

        // 10:29 -> Early, 10:30 -> Midday
        assert_eq!(compute_time_regime_from_et(10, 29), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(10, 30), TimeRegime::Midday);

        // 15:29 -> Midday, 15:30 -> Close
        assert_eq!(compute_time_regime_from_et(15, 29), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(15, 30), TimeRegime::Close);

        // 15:59 -> Close, 16:00 -> Closed
        assert_eq!(compute_time_regime_from_et(15, 59), TimeRegime::Close);
        assert_eq!(compute_time_regime_from_et(16, 0), TimeRegime::Closed);
    }

    #[test]
    fn test_time_regime_as_f64() {
        assert_eq!(TimeRegime::Open.as_f64(), 0.0);
        assert_eq!(TimeRegime::Early.as_f64(), 1.0);
        assert_eq!(TimeRegime::Midday.as_f64(), 2.0);
        assert_eq!(TimeRegime::Close.as_f64(), 3.0);
        assert_eq!(TimeRegime::Closed.as_f64(), 4.0);
    }

    #[test]
    fn test_compute_time_regime_known_timestamp() {
        // February 3, 2025 at 14:30:00 UTC
        // Calculation:
        //   Days from epoch to Feb 3, 2025 = 20123 days
        //   Seconds at start of day = 20123 * 86400 = 1738627200
        //   Plus 14:30:00 = 52200 seconds
        //   Total = 1738679400
        // This is during standard time (not DST)
        // UTC 14:30 - 5 hours = ET 09:30 (market open)
        let feb_3_2025_1430_utc_ns: i64 = 1738679400 * NS_PER_SECOND;
        assert_eq!(compute_time_regime(feb_3_2025_1430_utc_ns), TimeRegime::Open);

        // Add 15 minutes -> ET 09:45 (Early)
        let feb_3_2025_1445_utc_ns = feb_3_2025_1430_utc_ns + 15 * 60 * NS_PER_SECOND;
        assert_eq!(compute_time_regime(feb_3_2025_1445_utc_ns), TimeRegime::Early);

        // Add 1 hour -> ET 10:30 (Midday)
        let feb_3_2025_1530_utc_ns = feb_3_2025_1430_utc_ns + 60 * 60 * NS_PER_SECOND;
        assert_eq!(compute_time_regime(feb_3_2025_1530_utc_ns), TimeRegime::Midday);

        // Add 6 hours -> ET 15:30 (Close)
        let feb_3_2025_2030_utc_ns = feb_3_2025_1430_utc_ns + 6 * 60 * 60 * NS_PER_SECOND;
        assert_eq!(compute_time_regime(feb_3_2025_2030_utc_ns), TimeRegime::Close);
    }

    #[test]
    fn test_compute_time_regime_dst_summer() {
        // July 1, 2025 at 13:30:00 UTC
        // Calculation:
        //   Days from epoch to July 1, 2025 = 20270 days
        //   Seconds at start of day = 20270 * 86400 = 1751328000
        //   Plus 13:30:00 = 48600 seconds
        //   Total = 1751376600
        // This is during DST
        // UTC 13:30 - 4 hours = ET 09:30 (market open)
        let july_1_2025_1330_utc_ns: i64 = 1751376600 * NS_PER_SECOND;
        assert_eq!(compute_time_regime(july_1_2025_1330_utc_ns), TimeRegime::Open);

        // Add 15 minutes -> ET 09:45 (Early)
        let july_1_2025_1345_utc_ns = july_1_2025_1330_utc_ns + 15 * 60 * NS_PER_SECOND;
        assert_eq!(compute_time_regime(july_1_2025_1345_utc_ns), TimeRegime::Early);
    }

    // ========================================================================
    // Book Validity Tests
    // ========================================================================

    #[test]
    fn test_is_book_valid_normal_spread() {
        // Normal case: bid < ask
        assert!(is_book_valid(Some(100_000_000), Some(100_010_000)));
    }

    #[test]
    fn test_is_book_valid_crossed() {
        // Crossed: bid > ask
        assert!(!is_book_valid(Some(100_010_000), Some(100_000_000)));
    }

    #[test]
    fn test_is_book_valid_locked() {
        // Locked: bid == ask
        assert!(!is_book_valid(Some(100_000_000), Some(100_000_000)));
    }

    #[test]
    fn test_is_book_valid_empty_bid() {
        assert!(!is_book_valid(None, Some(100_000_000)));
    }

    #[test]
    fn test_is_book_valid_empty_ask() {
        assert!(!is_book_valid(Some(100_000_000), None));
    }

    #[test]
    fn test_is_book_valid_both_empty() {
        assert!(!is_book_valid(None, None));
    }

    #[test]
    fn test_is_book_valid_wide_spread() {
        // Wide spread is still valid
        assert!(is_book_valid(Some(100_000_000), Some(110_000_000)));
    }

    #[test]
    fn test_is_book_valid_one_tick_spread() {
        // Minimum valid spread (1 tick = $0.01 = 10,000 in fixed-point)
        assert!(is_book_valid(Some(100_000_000), Some(100_010_000)));
    }

    // ========================================================================
    // Signal Index Tests
    // ========================================================================

    #[test]
    fn test_signal_indices_are_contiguous() {
        // Verify indices are contiguous from 84 to 97
        assert_eq!(indices::TRUE_OFI, 84);
        assert_eq!(indices::DEPTH_NORM_OFI, 85);
        assert_eq!(indices::EXECUTED_PRESSURE, 86);
        assert_eq!(indices::SIGNED_MP_DELTA_BPS, 87);
        assert_eq!(indices::TRADE_ASYMMETRY, 88);
        assert_eq!(indices::CANCEL_ASYMMETRY, 89);
        assert_eq!(indices::FRAGILITY_SCORE, 90);
        assert_eq!(indices::DEPTH_ASYMMETRY, 91);
        assert_eq!(indices::BOOK_VALID, 92);
        assert_eq!(indices::TIME_REGIME, 93);
        assert_eq!(indices::MBO_READY, 94);
        assert_eq!(indices::DT_SECONDS, 95);
        assert_eq!(indices::INVALIDITY_DELTA, 96);
        assert_eq!(indices::SCHEMA_VERSION, 97);
    }

    #[test]
    fn test_signal_count() {
        assert_eq!(SIGNAL_COUNT, 14);
        // Verify: 97 - 84 + 1 = 14
        assert_eq!(indices::SCHEMA_VERSION - indices::TRUE_OFI + 1, SIGNAL_COUNT);
    }

    // ========================================================================
    // ET Offset Tests
    // ========================================================================

    #[test]
    fn test_et_offset_standard_time() {
        // January 15, 2025 - Standard time (-5 hours)
        // Days from epoch = 20102
        // Seconds = 20102 * 86400 = 1736812800
        let jan_15_2025_ns: i64 = 1736812800 * NS_PER_SECOND;
        let offset = estimate_et_offset(jan_15_2025_ns);
        assert_eq!(offset, ET_OFFSET_STANDARD_NS);
    }

    #[test]
    fn test_et_offset_dst() {
        // July 15, 2025 - DST (-4 hours)
        // Days from epoch = 20284
        // Seconds = 20284 * 86400 = 1752537600
        let july_15_2025_ns: i64 = 1752537600 * NS_PER_SECOND;
        let offset = estimate_et_offset(july_15_2025_ns);
        assert_eq!(offset, ET_OFFSET_DST_NS);
    }

    // ========================================================================
    // OFI Computer Tests
    // ========================================================================

    /// Create a test LOB state with specific bid/ask prices and sizes.
    fn make_lob(bid_price: i64, bid_size: u32, ask_price: i64, ask_size: u32) -> LobState {
        let mut lob = LobState::new(10);
        lob.best_bid = Some(bid_price);
        lob.best_ask = Some(ask_price);
        lob.bid_sizes[0] = bid_size;
        lob.ask_sizes[0] = ask_size;
        lob
    }

    #[test]
    fn test_ofi_computer_new() {
        let computer = OfiComputer::new();
        assert!(!computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), 0);
    }

    #[test]
    fn test_ofi_computer_bid_size_increase() {
        // Scenario: Bid size increases (demand increases)
        // Expected OFI: positive
        let mut computer = OfiComputer::new();

        // Initial state: bid=$100, size=100
        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Size increases to 150 at same price
        let lob2 = make_lob(100_000_000, 150, 100_010_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI = bid_contribution = 150 - 100 = 50
        assert_eq!(sample.ofi_bid, 50.0);
        assert_eq!(sample.ofi, 50.0);
        assert!(sample.ofi > 0.0, "Bid size increase should be positive OFI");
    }

    #[test]
    fn test_ofi_computer_bid_size_decrease() {
        // Scenario: Bid size decreases (demand decreases)
        // Expected OFI: negative
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Size decreases to 50 at same price
        let lob2 = make_lob(100_000_000, 50, 100_010_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI = bid_contribution = 50 - 100 = -50
        assert_eq!(sample.ofi_bid, -50.0);
        assert!(sample.ofi < 0.0, "Bid size decrease should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_bid_price_improvement() {
        // Scenario: Bid price improves (higher bid)
        // Expected OFI: positive (new demand at better price)
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_020_000, 100);
        computer.update(&lob1);

        // Bid improves to $100.01 with 80 shares
        let lob2 = make_lob(100_010_000, 80, 100_020_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI bid = +curr_size (price improved) = 80
        assert_eq!(sample.ofi_bid, 80.0);
        assert!(sample.ofi > 0.0, "Bid price improvement should be positive OFI");
    }

    #[test]
    fn test_ofi_computer_bid_price_drop() {
        // Scenario: Bid price drops (lower bid)
        // Expected OFI: negative (demand removed)
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_010_000, 100, 100_020_000, 100);
        computer.update(&lob1);

        // Bid drops to $100.00 with 120 shares
        let lob2 = make_lob(100_000_000, 120, 100_020_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI bid = -prev_size (price dropped) = -100
        assert_eq!(sample.ofi_bid, -100.0);
        assert!(sample.ofi < 0.0, "Bid price drop should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_ask_size_increase() {
        // Scenario: Ask size increases (supply increases)
        // Expected OFI: negative (more supply = downward pressure)
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Ask size increases to 150
        let lob2 = make_lob(100_000_000, 100, 100_010_000, 150);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI ask contribution = -(150 - 100) = -50
        assert_eq!(sample.ofi_ask, -50.0);
        assert!(sample.ofi < 0.0, "Ask size increase should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_ask_price_improvement() {
        // Scenario: Ask price improves (lower ask)
        // Expected OFI: negative (supply at better price)
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_020_000, 100);
        computer.update(&lob1);

        // Ask improves to $100.01 with 80 shares
        let lob2 = make_lob(100_000_000, 100, 100_010_000, 80);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI ask = -curr_size (price improved) = -80
        assert_eq!(sample.ofi_ask, -80.0);
        assert!(sample.ofi < 0.0, "Ask price improvement should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_warmup_tracking() {
        let mut computer = OfiComputer::new();

        // Initially not warm
        assert!(!computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), 0);

        // First update initializes state (doesn't count as state change)
        let lob_init = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob_init);
        assert_eq!(computer.state_changes_since_reset(), 0);

        // Add MIN_WARMUP_STATE_CHANGES actual state changes
        // Each update changes the bid size, so each counts as a state change
        for i in 1..=MIN_WARMUP_STATE_CHANGES {
            let size = (100 + i) as u32; // Incrementing size ensures change
            let lob = make_lob(100_000_000, size, 100_010_000, 100);
            computer.update(&lob);
        }

        // Should be warm now
        assert!(computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), MIN_WARMUP_STATE_CHANGES);
    }

    #[test]
    fn test_ofi_computer_warmup_persists_across_samples() {
        let mut computer = OfiComputer::new();

        // Initialize state
        let lob_init = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob_init);

        // Warm up the computer
        for i in 1..=MIN_WARMUP_STATE_CHANGES + 10 {
            let size = (100 + i) as u32; // Incrementing size
            let lob = make_lob(100_000_000, size, 100_010_000, 100);
            computer.update(&lob);
        }

        assert!(computer.is_warm());
        let changes_before = computer.state_changes_since_reset();

        // Sample and reset
        let _ = computer.sample_and_reset(1_000_000_000);

        // Warmup should persist
        assert!(computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), changes_before);
    }

    #[test]
    fn test_ofi_computer_reset_on_clear() {
        let mut computer = OfiComputer::new();

        // Initialize state
        let lob_init = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob_init);

        // Warm up
        for i in 1..=MIN_WARMUP_STATE_CHANGES + 10 {
            let size = (100 + i) as u32; // Incrementing size
            let lob = make_lob(100_000_000, size, 100_010_000, 100);
            computer.update(&lob);
        }

        assert!(computer.is_warm());

        // Reset on clear
        computer.reset_on_clear();

        // Should be cold again
        assert!(!computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), 0);
    }

    #[test]
    fn test_ofi_computer_dt_seconds() {
        let mut computer = OfiComputer::new();

        let lob = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob);

        // First sample: dt_seconds should be 0 (no previous timestamp)
        let sample1 = computer.sample_and_reset(1_000_000_000);
        assert_eq!(sample1.dt_seconds, 0.0);

        // Add more events
        computer.update(&lob);

        // Second sample: dt_seconds should be the difference
        let sample2 = computer.sample_and_reset(2_000_000_000);
        assert_eq!(sample2.dt_seconds, 1.0); // 1 second

        // Third sample after 0.5 seconds
        computer.update(&lob);
        let sample3 = computer.sample_and_reset(2_500_000_000);
        assert_eq!(sample3.dt_seconds, 0.5);
    }

    #[test]
    fn test_ofi_computer_avg_depth() {
        let mut computer = OfiComputer::new();

        // Event 1: total depth = 100 + 100 = 200
        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Event 2: total depth = 200 + 300 = 500
        let lob2 = make_lob(100_000_000, 200, 100_010_000, 300);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // Average = (200 + 500) / 2 = 350
        assert_eq!(sample.avg_depth, 350.0);
    }

    #[test]
    fn test_ofi_computer_depth_norm_ofi() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Bid increases by 100 (OFI = 100)
        let lob2 = make_lob(100_000_000, 200, 100_010_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // OFI = 100
        // Avg depth = (200 + 300) / 2 = 250
        // Depth norm OFI = 100 / 250 = 0.4
        let depth_norm = sample.depth_norm_ofi();
        assert!((depth_norm - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_ofi_computer_no_change_no_warmup_increment() {
        let mut computer = OfiComputer::new();

        // Same state multiple times
        let lob = make_lob(100_000_000, 100, 100_010_000, 100);

        computer.update(&lob);
        let changes1 = computer.state_changes_since_reset();

        // Same exact state - should NOT increment warmup
        computer.update(&lob);
        let changes2 = computer.state_changes_since_reset();

        // First update initializes, second is no-change
        assert_eq!(changes2, changes1);
    }

    #[test]
    fn test_ofi_computer_balanced_flow() {
        // Scenario: Equal buy and sell pressure
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Bid +50, Ask +50 -> OFI should be near 0
        let lob2 = make_lob(100_000_000, 150, 100_010_000, 150);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // Bid contribution: +50
        // Ask contribution: -(150 - 100) = -50
        // Net OFI = 50 + (-50) = 0
        assert_eq!(sample.ofi_bid, 50.0);
        assert_eq!(sample.ofi_ask, -50.0);
        assert_eq!(sample.ofi, 0.0);
    }

    #[test]
    fn test_ofi_computer_accumulation_multiple_events() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        // Event 1: Bid +50
        let lob2 = make_lob(100_000_000, 150, 100_010_000, 100);
        computer.update(&lob2);

        // Event 2: Bid +30 more
        let lob3 = make_lob(100_000_000, 180, 100_010_000, 100);
        computer.update(&lob3);

        // Event 3: Ask +20 (negative contribution)
        let lob4 = make_lob(100_000_000, 180, 100_010_000, 120);
        computer.update(&lob4);

        let sample = computer.sample_and_reset(1_000_000_000);

        // Bid: 50 + 30 = 80
        // Ask: -20
        // Total: 60
        assert_eq!(sample.ofi_bid, 80.0);
        assert_eq!(sample.ofi_ask, -20.0);
        assert_eq!(sample.ofi, 60.0);
        assert_eq!(sample.event_count, 4);
    }

    // ========================================================================
    // Signal Computation Tests
    // ========================================================================

    /// Create a base feature vector for testing.
    fn make_base_features() -> Vec<f64> {
        let mut features = vec![0.0; 84];

        // Ask prices (0-9)
        features[0] = 100.01; // ask_price_0

        // Bid prices (20-29)
        features[20] = 100.00; // bid_price_0

        // Derived features (40-47)
        features[derived_indices::MID_PRICE] = 100.005; // mid_price
        features[derived_indices::TOTAL_BID_VOLUME] = 1000.0;
        features[derived_indices::TOTAL_ASK_VOLUME] = 1000.0;
        features[derived_indices::WEIGHTED_MID_PRICE] = 100.007; // microprice slightly above mid

        // MBO features (48-83)
        features[mbo_indices::CANCEL_RATE_BID] = 10.0;
        features[mbo_indices::CANCEL_RATE_ASK] = 15.0; // More ask cancels
        features[mbo_indices::TRADE_RATE_BID] = 5.0;
        features[mbo_indices::TRADE_RATE_ASK] = 8.0; // More buy trades
        features[mbo_indices::LEVEL_CONCENTRATION] = 0.3;
        features[mbo_indices::DEPTH_TICKS_BID] = 50.0;
        features[mbo_indices::DEPTH_TICKS_ASK] = 40.0; // More bid depth

        features
    }

    #[test]
    fn test_signal_vector_new() {
        let signals = SignalVector::new();
        assert_eq!(signals.signals.len(), SIGNAL_COUNT);
        for &s in &signals.signals {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_signal_vector_get() {
        let mut signals = SignalVector::new();
        signals.signals[0] = 42.0;

        // Valid index
        assert_eq!(signals.get(indices::TRUE_OFI), Some(42.0));

        // Out of range
        assert_eq!(signals.get(83), None);
        assert_eq!(signals.get(98), None);
    }

    #[test]
    fn test_compute_signals_basic() {
        let features = make_base_features();
        let ofi_sample = OfiSample {
            ofi: 100.0,
            ofi_bid: 80.0,
            ofi_ask: 20.0,
            avg_depth: 200.0,
            event_count: 50,
            is_warm: true,
            dt_seconds: 1.5,
        };

        // February 3, 2025 15:30 UTC = 10:30 ET (Midday)
        let timestamp_ns: i64 = 1738683000 * NS_PER_SECOND;

        let signals = compute_signals(&features, &ofi_sample, timestamp_ns, 0);

        // Check all signals are set
        assert_eq!(signals.signals.len(), SIGNAL_COUNT);

        // Signal 84: true_ofi
        assert_eq!(signals.signals[0], 100.0);

        // Signal 85: depth_norm_ofi = 100 / 200 = 0.5
        assert_eq!(signals.signals[1], 0.5);

        // Signal 86: executed_pressure = 8 - 5 = 3
        assert_eq!(signals.signals[2], 3.0);

        // Signal 88: trade_asymmetry = (8-5)/(8+5) = 3/13
        let expected_trade_asym = 3.0 / 13.0;
        assert!((signals.signals[4] - expected_trade_asym).abs() < 0.001);

        // Signal 89: cancel_asymmetry = (15-10)/(15+10) = 5/25 = 0.2
        assert_eq!(signals.signals[5], 0.2);

        // Signal 91: depth_asymmetry = (50-40)/(50+40) = 10/90
        let expected_depth_asym = 10.0 / 90.0;
        assert!((signals.signals[7] - expected_depth_asym).abs() < 0.001);

        // Signal 92: book_valid = 1 (bid < ask)
        assert_eq!(signals.signals[8], 1.0);

        // Signal 94: mbo_ready = 1 (is_warm = true)
        assert_eq!(signals.signals[10], 1.0);

        // Signal 95: dt_seconds = 1.5
        assert_eq!(signals.signals[11], 1.5);

        // Signal 96: invalidity_delta = 0
        assert_eq!(signals.signals[12], 0.0);

        // Signal 97: schema_version = 2.1
        assert_eq!(signals.signals[13], SCHEMA_VERSION);
    }

    #[test]
    fn test_compute_signals_executed_pressure_sign() {
        let mut features = make_base_features();

        // More sell trades than buy trades
        features[mbo_indices::TRADE_RATE_BID] = 10.0; // Sells
        features[mbo_indices::TRADE_RATE_ASK] = 5.0; // Buys

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // executed_pressure = buys - sells = 5 - 10 = -5
        assert_eq!(signals.signals[2], -5.0);
        assert!(signals.signals[2] < 0.0, "More sells should give negative pressure");
    }

    #[test]
    fn test_compute_signals_cancel_asymmetry_bullish() {
        let mut features = make_base_features();

        // More ask cancels = sellers pulling quotes = bullish
        features[mbo_indices::CANCEL_RATE_BID] = 5.0;
        features[mbo_indices::CANCEL_RATE_ASK] = 15.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // cancel_asymmetry = (15 - 5) / (15 + 5) = 0.5
        assert_eq!(signals.signals[5], 0.5);
        assert!(signals.signals[5] > 0.0, "More ask cancels should be bullish");
    }

    #[test]
    fn test_compute_signals_cancel_asymmetry_bearish() {
        let mut features = make_base_features();

        // More bid cancels = buyers pulling quotes = bearish
        features[mbo_indices::CANCEL_RATE_BID] = 20.0;
        features[mbo_indices::CANCEL_RATE_ASK] = 5.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // cancel_asymmetry = (5 - 20) / (5 + 20) = -0.6
        assert_eq!(signals.signals[5], -0.6);
        assert!(signals.signals[5] < 0.0, "More bid cancels should be bearish");
    }

    #[test]
    fn test_compute_signals_depth_asymmetry_bullish() {
        let mut features = make_base_features();

        // More bid depth = stronger support = bullish
        features[mbo_indices::DEPTH_TICKS_BID] = 80.0;
        features[mbo_indices::DEPTH_TICKS_ASK] = 20.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // depth_asymmetry = (80 - 20) / (80 + 20) = 0.6
        assert_eq!(signals.signals[7], 0.6);
        assert!(signals.signals[7] > 0.0, "More bid depth should be bullish");
    }

    #[test]
    fn test_compute_signals_microprice_delta() {
        let mut features = make_base_features();

        // Microprice above mid = buy pressure
        features[derived_indices::MID_PRICE] = 100.0;
        features[derived_indices::WEIGHTED_MID_PRICE] = 100.01; // 1 cent above

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // signed_mp_delta_bps = (100.01 - 100) / 100 * 10000 = 1 bp
        assert!((signals.signals[3] - 1.0).abs() < 0.01);
        assert!(signals.signals[3] > 0.0, "Microprice above mid should be positive");
    }

    #[test]
    fn test_compute_signals_fragility_high() {
        let mut features = make_base_features();

        // High concentration + low depth = fragile
        features[mbo_indices::LEVEL_CONCENTRATION] = 0.8; // High concentration
        features[derived_indices::TOTAL_BID_VOLUME] = 100.0; // Low depth
        features[derived_indices::TOTAL_ASK_VOLUME] = 100.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // fragility = 0.8 / ln(100) = 0.8 / 4.605 ≈ 0.174
        assert!(signals.signals[6] > 0.15 && signals.signals[6] < 0.2);
    }

    #[test]
    fn test_compute_signals_mbo_not_ready() {
        let features = make_base_features();
        let ofi_sample = OfiSample {
            is_warm: false,
            ..Default::default()
        };

        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // mbo_ready = 0 when not warm
        assert_eq!(signals.signals[10], 0.0);
    }

    #[test]
    fn test_compute_signals_invalidity_delta() {
        let features = make_base_features();
        let ofi_sample = OfiSample::default();

        // With invalidity events
        let signals = compute_signals(&features, &ofi_sample, 0, 5);

        assert_eq!(signals.signals[12], 5.0);
    }

    #[test]
    fn test_compute_signals_with_book_valid() {
        let features = make_base_features();
        let ofi_sample = OfiSample::default();

        // Force book_valid to false
        let signals = compute_signals_with_book_valid(&features, &ofi_sample, 0, 0, false);
        assert_eq!(signals.signals[8], 0.0);

        // Force book_valid to true
        let signals = compute_signals_with_book_valid(&features, &ofi_sample, 0, 0, true);
        assert_eq!(signals.signals[8], 1.0);
    }

    #[test]
    fn test_compute_signals_zero_division_safety() {
        let features = vec![0.0; 84];

        // All zeros - should not panic
        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // Check no NaN/Inf
        for (i, &s) in signals.signals.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Signal {} should be finite, got {}",
                i + indices::TRUE_OFI,
                s
            );
        }
    }

    #[test]
    fn test_signal_vector_to_vec() {
        let mut signals = SignalVector::new();
        signals.signals[0] = 1.0;
        signals.signals[13] = 2.0;

        let vec = signals.to_vec();
        assert_eq!(vec.len(), SIGNAL_COUNT);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[13], 2.0);
    }
}

