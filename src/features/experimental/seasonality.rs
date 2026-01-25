//! Enhanced Seasonality Features for Intraday Trading
//!
//! This module provides 4 features related to time-of-day patterns
//! that are more granular than the base TIME_REGIME feature.
//!
//! # Feature Index Layout (when experimental enabled)
//!
//! | Offset | Name | Description | Range |
//! |--------|------|-------------|-------|
//! | 0 | minutes_since_open | Minutes since market open (9:30 ET) | [0, 390] |
//! | 1 | minutes_until_close | Minutes until market close (4:00 PM ET) | [0, 390] |
//! | 2 | session_progress | Normalized session progress | [0.0, 1.0] |
//! | 3 | time_bucket | 15-minute bucket index | [0, 26] |
//!
//! # Usage in Intraday Trading
//!
//! 1. **Session Progress**: Use for time-decay strategies
//!    - Progress = 0.0 → Start of day, high uncertainty
//!    - Progress = 1.0 → End of day, closing imbalances
//!
//! 2. **Time Buckets**: Categorical feature for regime-specific models
//!    - Bucket 0-1 (9:30-10:00) → Opening volatility
//!    - Bucket 2-4 (10:00-11:15) → Early session
//!    - Bucket 5-23 (11:15-15:15) → Midday
//!    - Bucket 24-26 (15:15-16:00) → Closing period
//!
//! # Time Conversion
//!
//! Timestamps are expected in UTC nanoseconds. The module converts to
//! US Eastern Time for market session calculations.
//!
//! # Note
//!
//! This module is stateless - it computes features purely from the
//! provided timestamp.

/// Number of features in this group.
pub const FEATURE_COUNT: usize = 4;

/// Feature indices (relative to group start).
pub mod indices {
    pub const MINUTES_SINCE_OPEN: usize = 0;
    pub const MINUTES_UNTIL_CLOSE: usize = 1;
    pub const SESSION_PROGRESS: usize = 2;
    pub const TIME_BUCKET: usize = 3;
}

/// Market open time in nanoseconds since midnight UTC.
/// 9:30 AM ET = 14:30 UTC (during EST) or 13:30 UTC (during EDT)
/// We use 14:30 UTC as default (EST)
const MARKET_OPEN_NS: u64 = 14 * 3600 * 1_000_000_000 + 30 * 60 * 1_000_000_000;

/// Market close time in nanoseconds since midnight UTC.
/// 4:00 PM ET = 21:00 UTC (during EST) or 20:00 UTC (during EDT)
/// We use 21:00 UTC as default (EST)
const MARKET_CLOSE_NS: u64 = 21 * 3600 * 1_000_000_000;

/// Trading session duration in minutes (6.5 hours = 390 minutes).
const SESSION_DURATION_MINUTES: f64 = 390.0;

/// Time bucket size in minutes (15 minutes).
const BUCKET_SIZE_MINUTES: u64 = 15;

/// Number of time buckets per session (390 / 15 = 26).
const NUM_BUCKETS: usize = 26;

/// Seasonality feature computer.
///
/// This is a stateless computer - features are computed purely
/// from the provided timestamp.
pub struct SeasonalityComputer {
    /// Whether to use DST-aware computation.
    /// For now, we use a simple heuristic based on month.
    _use_dst_aware: bool,
}

impl SeasonalityComputer {
    /// Create a new seasonality computer.
    pub fn new() -> Self {
        Self {
            _use_dst_aware: true,
        }
    }

    /// Extract all 4 features into the output buffer.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ns` - Current timestamp in nanoseconds since epoch
    /// * `output` - Output buffer to append features to
    pub fn extract_into(&self, timestamp_ns: u64, output: &mut Vec<f64>) {
        // Convert timestamp to time of day (nanoseconds since midnight UTC)
        let ns_per_day: u64 = 24 * 3600 * 1_000_000_000;
        let time_of_day_ns = timestamp_ns % ns_per_day;

        // Estimate DST offset based on rough date (not perfect but reasonable)
        // This is a simplification - production code would use a proper timezone library
        let is_dst = self.estimate_is_dst(timestamp_ns);
        let market_open = if is_dst {
            MARKET_OPEN_NS - 3600 * 1_000_000_000 // 1 hour earlier in UTC during DST
        } else {
            MARKET_OPEN_NS
        };
        let market_close = if is_dst {
            MARKET_CLOSE_NS - 3600 * 1_000_000_000
        } else {
            MARKET_CLOSE_NS
        };

        // Compute minutes since open
        let minutes_since_open = if time_of_day_ns >= market_open {
            ((time_of_day_ns - market_open) as f64 / 60_000_000_000.0).min(SESSION_DURATION_MINUTES)
        } else {
            0.0
        };
        output.push(minutes_since_open);

        // Compute minutes until close
        let minutes_until_close = if time_of_day_ns < market_close {
            ((market_close - time_of_day_ns) as f64 / 60_000_000_000.0)
                .max(0.0)
                .min(SESSION_DURATION_MINUTES)
        } else {
            0.0
        };
        output.push(minutes_until_close);

        // Compute session progress (0.0 to 1.0)
        let session_progress = (minutes_since_open / SESSION_DURATION_MINUTES).clamp(0.0, 1.0);
        output.push(session_progress);

        // Compute time bucket (0 to 25)
        let bucket = (minutes_since_open as u64 / BUCKET_SIZE_MINUTES).min(NUM_BUCKETS as u64 - 1);
        output.push(bucket as f64);
    }

    /// Rough DST estimation based on timestamp.
    ///
    /// US DST: Second Sunday in March to First Sunday in November
    /// This is approximate - for production, use a timezone library.
    fn estimate_is_dst(&self, timestamp_ns: u64) -> bool {
        // Convert to approximate day of year
        let seconds_since_epoch = timestamp_ns / 1_000_000_000;
        let days_since_epoch = seconds_since_epoch / 86400;
        
        // January 1, 1970 was a Thursday
        // Rough estimate of day of year (ignoring leap years for simplicity)
        let day_of_year = (days_since_epoch % 365) as u32;

        // DST roughly: March 10 (day ~70) to November 3 (day ~307)
        // This is approximate but sufficient for features
        day_of_year >= 70 && day_of_year <= 307
    }
}

impl Default for SeasonalityComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute session progress for a given minute.
///
/// # Arguments
///
/// * `minutes_since_open` - Minutes since market open
///
/// # Returns
///
/// Session progress in range [0.0, 1.0]
#[inline]
pub fn session_progress(minutes_since_open: f64) -> f64 {
    (minutes_since_open / SESSION_DURATION_MINUTES).clamp(0.0, 1.0)
}

/// Compute time bucket for a given minute.
///
/// # Arguments
///
/// * `minutes_since_open` - Minutes since market open
///
/// # Returns
///
/// Time bucket index (0 to 25)
#[inline]
pub fn time_bucket(minutes_since_open: f64) -> usize {
    ((minutes_since_open as u64) / BUCKET_SIZE_MINUTES).min(NUM_BUCKETS as u64 - 1) as usize
}

/// Get human-readable label for a time bucket.
///
/// # Arguments
///
/// * `bucket` - Time bucket index (0-25)
///
/// # Returns
///
/// Human-readable label (e.g., "09:30-09:45")
pub fn bucket_label(bucket: usize) -> String {
    let start_min = bucket * BUCKET_SIZE_MINUTES as usize;
    let end_min = start_min + BUCKET_SIZE_MINUTES as usize;

    let start_hour = 9 + start_min / 60;
    let start_minute = 30 + start_min % 60;
    let adjusted_start_hour = start_hour + start_minute / 60;
    let adjusted_start_minute = start_minute % 60;

    let end_hour = 9 + end_min / 60;
    let end_minute = 30 + end_min % 60;
    let adjusted_end_hour = end_hour + end_minute / 60;
    let adjusted_end_minute = end_minute % 60;

    format!(
        "{:02}:{:02}-{:02}:{:02}",
        adjusted_start_hour, adjusted_start_minute, adjusted_end_hour, adjusted_end_minute
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        assert_eq!(FEATURE_COUNT, 4);
    }

    #[test]
    fn test_session_progress() {
        assert_eq!(session_progress(0.0), 0.0);
        assert!((session_progress(195.0) - 0.5).abs() < 0.01); // Midday
        assert_eq!(session_progress(390.0), 1.0);
        assert_eq!(session_progress(500.0), 1.0); // Clamped
    }

    #[test]
    fn test_time_bucket() {
        assert_eq!(time_bucket(0.0), 0); // 9:30-9:45
        assert_eq!(time_bucket(14.9), 0); // Still 9:30-9:45
        assert_eq!(time_bucket(15.0), 1); // 9:45-10:00
        assert_eq!(time_bucket(195.0), 13); // Midday
        assert_eq!(time_bucket(389.0), 25); // End of day
    }

    #[test]
    fn test_bucket_labels() {
        assert_eq!(bucket_label(0), "09:30-09:45");
        assert_eq!(bucket_label(1), "09:45-10:00");
        assert_eq!(bucket_label(2), "10:00-10:15");
    }

    #[test]
    fn test_extract_midday() {
        let computer = SeasonalityComputer::new();

        // Midday timestamp (roughly 12:30 PM ET = 17:30 UTC during EST)
        // Nanoseconds for 17:30 UTC
        let timestamp_ns = 17 * 3600 * 1_000_000_000 + 30 * 60 * 1_000_000_000;

        let mut output = Vec::new();
        computer.extract_into(timestamp_ns, &mut output);

        assert_eq!(output.len(), FEATURE_COUNT);

        // Minutes since open should be around 180 (3 hours after 9:30)
        let minutes_since_open = output[indices::MINUTES_SINCE_OPEN];
        assert!(
            minutes_since_open > 150.0 && minutes_since_open < 210.0,
            "Expected ~180 minutes, got {}",
            minutes_since_open
        );

        // Session progress should be around 0.46 (180/390)
        let progress = output[indices::SESSION_PROGRESS];
        assert!(
            progress > 0.3 && progress < 0.6,
            "Expected ~0.46, got {}",
            progress
        );
    }

    #[test]
    fn test_extract_all_finite() {
        let computer = SeasonalityComputer::new();

        // Test various timestamps
        let timestamps = [
            0u64,
            1_000_000_000_000_000_000,
            1_700_000_000_000_000_000,
        ];

        for ts in timestamps {
            let mut output = Vec::new();
            computer.extract_into(ts, &mut output);

            assert_eq!(output.len(), FEATURE_COUNT);
            for (i, &val) in output.iter().enumerate() {
                assert!(val.is_finite(), "Feature {} is not finite: {}", i, val);
            }
        }
    }
}
