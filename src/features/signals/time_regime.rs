//! Market session time regime classification.
//!
//! Based on Cont et al. (2014) §3.3: intraday price impact patterns
//! vary significantly across market sessions.

/// Nanoseconds per second
const NS_PER_SECOND: i64 = 1_000_000_000;

/// Nanoseconds per hour
const NS_PER_HOUR: i64 = 3_600_000_000_000;

/// US Eastern Time offset from UTC (standard time: -5 hours)
const ET_OFFSET_STANDARD_NS: i64 = -5 * NS_PER_HOUR;

/// US Eastern Time offset from UTC (daylight saving: -4 hours)
const ET_OFFSET_DST_NS: i64 = -4 * NS_PER_HOUR;

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
    let et_offset = estimate_et_offset(timestamp_ns);
    let et_timestamp_ns = timestamp_ns + et_offset;

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
    let seconds = timestamp_ns / NS_PER_SECOND;
    let days_since_epoch = seconds / 86400;

    // Approximate day of year (ignores leap years -- sufficient for regime classification)
    let day_of_year = (days_since_epoch % 365) as i32;

    // DST in US: roughly March 10 (day ~70) to November 3 (day ~307)
    if (70..307).contains(&day_of_year) {
        ET_OFFSET_DST_NS
    } else {
        ET_OFFSET_STANDARD_NS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_regime_from_et_open() {
        assert_eq!(compute_time_regime_from_et(9, 30), TimeRegime::Open);
        assert_eq!(compute_time_regime_from_et(9, 35), TimeRegime::Open);
        assert_eq!(compute_time_regime_from_et(9, 44), TimeRegime::Open);
    }

    #[test]
    fn test_time_regime_from_et_early() {
        assert_eq!(compute_time_regime_from_et(9, 45), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(9, 59), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(10, 0), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(10, 29), TimeRegime::Early);
    }

    #[test]
    fn test_time_regime_from_et_midday() {
        assert_eq!(compute_time_regime_from_et(10, 30), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(12, 0), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(14, 30), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(15, 0), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(15, 29), TimeRegime::Midday);
    }

    #[test]
    fn test_time_regime_from_et_close() {
        assert_eq!(compute_time_regime_from_et(15, 30), TimeRegime::Close);
        assert_eq!(compute_time_regime_from_et(15, 45), TimeRegime::Close);
        assert_eq!(compute_time_regime_from_et(15, 59), TimeRegime::Close);
    }

    #[test]
    fn test_time_regime_from_et_closed() {
        assert_eq!(compute_time_regime_from_et(0, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(8, 59), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(9, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(9, 29), TimeRegime::Closed);

        assert_eq!(compute_time_regime_from_et(16, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(17, 0), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(23, 59), TimeRegime::Closed);
    }

    #[test]
    fn test_time_regime_boundaries() {
        assert_eq!(compute_time_regime_from_et(9, 29), TimeRegime::Closed);
        assert_eq!(compute_time_regime_from_et(9, 30), TimeRegime::Open);

        assert_eq!(compute_time_regime_from_et(9, 44), TimeRegime::Open);
        assert_eq!(compute_time_regime_from_et(9, 45), TimeRegime::Early);

        assert_eq!(compute_time_regime_from_et(10, 29), TimeRegime::Early);
        assert_eq!(compute_time_regime_from_et(10, 30), TimeRegime::Midday);

        assert_eq!(compute_time_regime_from_et(15, 29), TimeRegime::Midday);
        assert_eq!(compute_time_regime_from_et(15, 30), TimeRegime::Close);

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
        // Standard time: UTC 14:30 - 5 hours = ET 09:30 (market open)
        let feb_3_2025_1430_utc_ns: i64 = 1738679400 * NS_PER_SECOND;
        assert_eq!(
            compute_time_regime(feb_3_2025_1430_utc_ns),
            TimeRegime::Open
        );

        // +15 min -> ET 09:45 (Early)
        let feb_3_2025_1445_utc_ns = feb_3_2025_1430_utc_ns + 15 * 60 * NS_PER_SECOND;
        assert_eq!(
            compute_time_regime(feb_3_2025_1445_utc_ns),
            TimeRegime::Early
        );

        // +1 hour -> ET 10:30 (Midday)
        let feb_3_2025_1530_utc_ns = feb_3_2025_1430_utc_ns + 60 * 60 * NS_PER_SECOND;
        assert_eq!(
            compute_time_regime(feb_3_2025_1530_utc_ns),
            TimeRegime::Midday
        );

        // +6 hours -> ET 15:30 (Close)
        let feb_3_2025_2030_utc_ns = feb_3_2025_1430_utc_ns + 6 * 60 * 60 * NS_PER_SECOND;
        assert_eq!(
            compute_time_regime(feb_3_2025_2030_utc_ns),
            TimeRegime::Close
        );
    }

    #[test]
    fn test_compute_time_regime_dst_summer() {
        // July 1, 2025 at 13:30:00 UTC (DST: UTC 13:30 - 4 hours = ET 09:30)
        let july_1_2025_1330_utc_ns: i64 = 1751376600 * NS_PER_SECOND;
        assert_eq!(
            compute_time_regime(july_1_2025_1330_utc_ns),
            TimeRegime::Open
        );

        // +15 min -> ET 09:45 (Early)
        let july_1_2025_1345_utc_ns = july_1_2025_1330_utc_ns + 15 * 60 * NS_PER_SECOND;
        assert_eq!(
            compute_time_regime(july_1_2025_1345_utc_ns),
            TimeRegime::Early
        );
    }

    #[test]
    fn test_et_offset_standard_time() {
        // January 15, 2025 - Standard time (-5 hours)
        let jan_15_2025_ns: i64 = 1736812800 * NS_PER_SECOND;
        let offset = estimate_et_offset(jan_15_2025_ns);
        assert_eq!(offset, ET_OFFSET_STANDARD_NS);
    }

    #[test]
    fn test_et_offset_dst() {
        // July 15, 2025 - DST (-4 hours)
        let july_15_2025_ns: i64 = 1752537600 * NS_PER_SECOND;
        let offset = estimate_et_offset(july_15_2025_ns);
        assert_eq!(offset, ET_OFFSET_DST_NS);
    }
}
