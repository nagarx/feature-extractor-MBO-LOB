//! Signal computation -- the main API for producing the 14-signal vector.

use crate::contract::{FLOAT_CMP_EPS, SCHEMA_VERSION, SIGNAL_COUNT};
use super::indices;
use super::ofi::OfiSample;
use super::time_regime::compute_time_regime;

/// Indices of MBO features used for signal computation.
/// These are in the 84-feature base vector.
mod mbo_indices {
    pub const CANCEL_RATE_BID: usize = 50;
    pub const CANCEL_RATE_ASK: usize = 51;
    pub const TRADE_RATE_BID: usize = 52;
    pub const TRADE_RATE_ASK: usize = 53;
    pub const LEVEL_CONCENTRATION: usize = 71;
    pub const DEPTH_TICKS_BID: usize = 72;
    pub const DEPTH_TICKS_ASK: usize = 73;
}

/// Indices of derived features used for signal computation.
mod derived_indices {
    pub const MID_PRICE: usize = 40;
    pub const TOTAL_BID_VOLUME: usize = 43;
    pub const TOTAL_ASK_VOLUME: usize = 44;
    pub const WEIGHTED_MID_PRICE: usize = 46;
}

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

    if base_features.len() < 84 {
        signals.signals[indices::SCHEMA_VERSION - indices::TRUE_OFI] = SCHEMA_VERSION;
        return signals;
    }

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
    if mid_price.abs() > FLOAT_CMP_EPS {
        signals.signals[3] = (weighted_mid - mid_price) / mid_price * 10_000.0;
    } else {
        signals.signals[3] = 0.0;
    }

    // === Signal 88: trade_asymmetry ===
    // Normalized trade pressure [-1, 1]
    let total_trades = trade_rate_bid + trade_rate_ask;
    if total_trades > FLOAT_CMP_EPS {
        signals.signals[4] = (trade_rate_ask - trade_rate_bid) / total_trades;
    } else {
        signals.signals[4] = 0.0;
    }

    // === Signal 89: cancel_asymmetry ===
    // > 0 = more ask cancels = sellers pulling quotes = bullish
    let total_cancels = cancel_rate_bid + cancel_rate_ask;
    if total_cancels > FLOAT_CMP_EPS {
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
    if total_depth > FLOAT_CMP_EPS {
        signals.signals[7] = (depth_bid - depth_ask) / total_depth;
    } else {
        signals.signals[7] = 0.0;
    }

    // === Signal 92: book_valid ===
    // Approximated from features; caller should use compute_signals_with_book_valid for accuracy
    let best_bid = base_features.get(20).copied().unwrap_or(0.0); // bid_price_0
    let best_ask = base_features.first().copied().unwrap_or(0.0); // ask_price_0
    if best_bid > FLOAT_CMP_EPS && best_ask > FLOAT_CMP_EPS && best_bid < best_ask {
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

#[cfg(test)]
mod tests {
    use super::*;

    const NS_PER_SECOND: i64 = 1_000_000_000;

    fn make_base_features() -> Vec<f64> {
        let mut features = vec![0.0; 84];

        features[0] = 100.01; // ask_price_0
        features[20] = 100.00; // bid_price_0

        features[derived_indices::MID_PRICE] = 100.005;
        features[derived_indices::TOTAL_BID_VOLUME] = 1000.0;
        features[derived_indices::TOTAL_ASK_VOLUME] = 1000.0;
        features[derived_indices::WEIGHTED_MID_PRICE] = 100.007;

        features[mbo_indices::CANCEL_RATE_BID] = 10.0;
        features[mbo_indices::CANCEL_RATE_ASK] = 15.0;
        features[mbo_indices::TRADE_RATE_BID] = 5.0;
        features[mbo_indices::TRADE_RATE_ASK] = 8.0;
        features[mbo_indices::LEVEL_CONCENTRATION] = 0.3;
        features[mbo_indices::DEPTH_TICKS_BID] = 50.0;
        features[mbo_indices::DEPTH_TICKS_ASK] = 40.0;

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

        assert_eq!(signals.get(indices::TRUE_OFI), Some(42.0));

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

        // Signal 97: schema_version = 2.2
        assert_eq!(signals.signals[13], SCHEMA_VERSION);
    }

    #[test]
    fn test_compute_signals_executed_pressure_sign() {
        let mut features = make_base_features();

        features[mbo_indices::TRADE_RATE_BID] = 10.0;
        features[mbo_indices::TRADE_RATE_ASK] = 5.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        assert_eq!(signals.signals[2], -5.0);
        assert!(signals.signals[2] < 0.0, "More sells should give negative pressure");
    }

    #[test]
    fn test_compute_signals_cancel_asymmetry_bullish() {
        let mut features = make_base_features();

        features[mbo_indices::CANCEL_RATE_BID] = 5.0;
        features[mbo_indices::CANCEL_RATE_ASK] = 15.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        assert_eq!(signals.signals[5], 0.5);
        assert!(signals.signals[5] > 0.0, "More ask cancels should be bullish");
    }

    #[test]
    fn test_compute_signals_cancel_asymmetry_bearish() {
        let mut features = make_base_features();

        features[mbo_indices::CANCEL_RATE_BID] = 20.0;
        features[mbo_indices::CANCEL_RATE_ASK] = 5.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        assert_eq!(signals.signals[5], -0.6);
        assert!(signals.signals[5] < 0.0, "More bid cancels should be bearish");
    }

    #[test]
    fn test_compute_signals_depth_asymmetry_bullish() {
        let mut features = make_base_features();

        features[mbo_indices::DEPTH_TICKS_BID] = 80.0;
        features[mbo_indices::DEPTH_TICKS_ASK] = 20.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        assert_eq!(signals.signals[7], 0.6);
        assert!(signals.signals[7] > 0.0, "More bid depth should be bullish");
    }

    #[test]
    fn test_compute_signals_microprice_delta() {
        let mut features = make_base_features();

        features[derived_indices::MID_PRICE] = 100.0;
        features[derived_indices::WEIGHTED_MID_PRICE] = 100.01;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // signed_mp_delta_bps = (100.01 - 100) / 100 * 10000 = 1 bp
        assert!((signals.signals[3] - 1.0).abs() < 0.01);
        assert!(signals.signals[3] > 0.0, "Microprice above mid should be positive");
    }

    #[test]
    fn test_compute_signals_fragility_high() {
        let mut features = make_base_features();

        features[mbo_indices::LEVEL_CONCENTRATION] = 0.8;
        features[derived_indices::TOTAL_BID_VOLUME] = 100.0;
        features[derived_indices::TOTAL_ASK_VOLUME] = 100.0;

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

        // fragility = 0.8 / ln(100) = 0.8 / 4.605 ~ 0.174
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

        assert_eq!(signals.signals[10], 0.0);
    }

    #[test]
    fn test_compute_signals_invalidity_delta() {
        let features = make_base_features();
        let ofi_sample = OfiSample::default();

        let signals = compute_signals(&features, &ofi_sample, 0, 5);

        assert_eq!(signals.signals[12], 5.0);
    }

    #[test]
    fn test_compute_signals_with_book_valid_fn() {
        let features = make_base_features();
        let ofi_sample = OfiSample::default();

        let signals = compute_signals_with_book_valid(&features, &ofi_sample, 0, 0, false);
        assert_eq!(signals.signals[8], 0.0);

        let signals = compute_signals_with_book_valid(&features, &ofi_sample, 0, 0, true);
        assert_eq!(signals.signals[8], 1.0);
    }

    #[test]
    fn test_compute_signals_zero_division_safety() {
        let features = vec![0.0; 84];

        let ofi_sample = OfiSample::default();
        let signals = compute_signals(&features, &ofi_sample, 0, 0);

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
