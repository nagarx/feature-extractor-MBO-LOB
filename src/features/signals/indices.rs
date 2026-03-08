//! Signal indices in the 98-feature output vector.
//!
//! These are the absolute indices of signals computed by the signal layer.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::SIGNAL_COUNT;

    #[test]
    fn test_signal_indices_are_contiguous() {
        assert_eq!(TRUE_OFI, 84);
        assert_eq!(DEPTH_NORM_OFI, 85);
        assert_eq!(EXECUTED_PRESSURE, 86);
        assert_eq!(SIGNED_MP_DELTA_BPS, 87);
        assert_eq!(TRADE_ASYMMETRY, 88);
        assert_eq!(CANCEL_ASYMMETRY, 89);
        assert_eq!(FRAGILITY_SCORE, 90);
        assert_eq!(DEPTH_ASYMMETRY, 91);
        assert_eq!(BOOK_VALID, 92);
        assert_eq!(TIME_REGIME, 93);
        assert_eq!(MBO_READY, 94);
        assert_eq!(DT_SECONDS, 95);
        assert_eq!(INVALIDITY_DELTA, 96);
        assert_eq!(SCHEMA_VERSION, 97);
    }

    #[test]
    fn test_signal_count() {
        assert_eq!(SIGNAL_COUNT, 14);
        assert_eq!(SCHEMA_VERSION - TRUE_OFI + 1, SIGNAL_COUNT);
    }
}
