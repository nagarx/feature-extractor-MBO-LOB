//! Market session time regime classification.
//!
//! Delegates to `hft_statistics::time::regime` for the single-source-of-truth
//! 7-regime classification with exact DST handling.
//!
//! | Value | Label | Time (ET) |
//! |-------|-------|-----------|
//! | 0 | pre-market | Before 9:30 |
//! | 1 | open-auction | 9:30 - 9:35 |
//! | 2 | morning | 9:35 - 10:30 |
//! | 3 | midday | 10:30 - 15:00 |
//! | 4 | afternoon | 15:00 - 15:55 |
//! | 5 | close-auction | 15:55 - 16:00 |
//! | 6 | post-market | After 16:00 |
//!
//! Previously this module had its own 5-regime system with approximate DST.
//! Now unified on the hft-statistics 7-regime system with exact DST.
//!
//! Reference: Cont et al. (2014) §3.3, hft-statistics/src/time/regime.rs

pub use hft_statistics::time::regime::{time_regime, N_REGIMES, REGIME_LABELS};

#[allow(unused_imports)]
pub use hft_statistics::time::regime::{utc_offset_for_date, day_epoch_ns};

/// Compute time regime from UTC nanosecond timestamp and pre-computed UTC offset.
///
/// This is the main entry point used by the feature extractor. The offset
/// should be computed once per day via `utc_offset_for_date()`.
#[inline]
pub fn compute_time_regime(timestamp_ns: i64, utc_offset_hours: i32) -> f64 {
    time_regime(timestamp_ns, utc_offset_hours) as f64
}

/// Compute time regime from UTC nanosecond timestamp with auto-detected offset.
///
/// Convenience function that infers the UTC offset from the timestamp.
/// Slightly more expensive than the 2-arg version (date extraction per call).
/// Use the 2-arg version in hot paths where the offset is known.
#[inline]
pub fn compute_time_regime_auto(timestamp_ns: i64) -> f64 {
    let offset = hft_statistics::time::regime::infer_utc_offset(&[timestamp_ns]);
    time_regime(timestamp_ns, offset) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS_PER_SECOND: i64 = 1_000_000_000;
    const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
    const NS_PER_HOUR: i64 = 3600 * NS_PER_SECOND;

    #[test]
    fn test_7_regimes_exist() {
        assert_eq!(N_REGIMES, 7);
        assert_eq!(REGIME_LABELS.len(), 7);
    }

    #[test]
    fn test_regime_boundaries_est() {
        let utc_offset = -5;
        let base_930_et = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        assert_eq!(time_regime(base_930_et - 1, utc_offset), 0, "9:29:59 → pre-market");
        assert_eq!(time_regime(base_930_et, utc_offset), 1, "9:30:00 → open-auction");
        assert_eq!(time_regime(base_930_et + 5 * NS_PER_MINUTE, utc_offset), 2, "9:35 → morning");
        assert_eq!(time_regime(base_930_et + 60 * NS_PER_MINUTE, utc_offset), 3, "10:30 → midday");

        let base_1500_et = 20 * NS_PER_HOUR;
        assert_eq!(time_regime(base_1500_et, utc_offset), 4, "15:00 → afternoon");
        assert_eq!(time_regime(base_1500_et + 55 * NS_PER_MINUTE, utc_offset), 5, "15:55 → close-auction");
        assert_eq!(time_regime(base_1500_et + 60 * NS_PER_MINUTE, utc_offset), 6, "16:00 → post-market");
    }

    #[test]
    fn test_dst_exact() {
        assert_eq!(utc_offset_for_date(2025, 1, 15), -5, "January = EST");
        assert_eq!(utc_offset_for_date(2025, 7, 15), -4, "July = EDT");
        assert_eq!(utc_offset_for_date(2025, 3, 8), -5, "March 8 = EST");
        assert_eq!(utc_offset_for_date(2025, 3, 9), -4, "March 9 = EDT (DST starts)");
    }

    #[test]
    fn test_compute_time_regime_convenience() {
        let utc_offset = -5;
        let noon_et = 17 * NS_PER_HOUR; // 12:00 ET = 17:00 UTC
        let result = compute_time_regime(noon_et, utc_offset);
        assert_eq!(result, 3.0, "12:00 ET should be midday (regime 3)");
    }
}
