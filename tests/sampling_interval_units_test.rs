//! Test to expose and validate the sampling interval units issue.
//!
//! ISSUE: Pipeline stores `min_time_interval_ns` in nanoseconds (1_000_000 = 1ms),
//! but `VolumeBasedSampler::new(_, min_time_interval_ms)` expects milliseconds
//! and multiplies by 1_000_000. This results in 1ms becoming 1000 seconds!
//!
//! This test exposes the bug and validates the fix.

use feature_extractor::preprocessing::VolumeBasedSampler;

/// Test that demonstrates the units mismatch bug.
///
/// The bug: Pipeline passes nanoseconds to a constructor expecting milliseconds.
///
/// Expected behavior (1ms minimum interval):
/// - Sample 1 at t=0
/// - Attempt sample at t=0.5ms should FAIL (too soon)
/// - Sample 2 at t=1.5ms should SUCCEED (1.5ms > 1ms minimum)
///
/// Buggy behavior (1000s minimum interval due to units mismatch):
/// - Sample 1 at t=0
/// - Sample 2 at t=1.5ms should FAIL (1.5ms < 1000s)
/// - Would need to wait 1000 seconds for next sample!
#[test]
fn test_sampling_interval_units_from_pipeline_perspective() {
    // This is what the pipeline does:
    // let min_interval = sampling_config.min_time_interval_ns.unwrap_or(1_000_000);
    // VolumeBasedSampler::new(threshold, min_interval)
    //
    // The config value is in NANOSECONDS (1_000_000 ns = 1ms)
    // But the constructor expects MILLISECONDS and multiplies by 1_000_000

    let config_value_ns: u64 = 1_000_000; // 1ms in nanoseconds (from config)
    let volume_threshold = 100;

    // Create sampler the way pipeline does (BEFORE FIX)
    // This passes nanoseconds to a milliseconds parameter
    let mut sampler = VolumeBasedSampler::new(volume_threshold, config_value_ns);

    // Base timestamp: 1 second in nanoseconds
    let t0: u64 = 1_000_000_000;

    // First sample should succeed (volume met, first sample)
    assert!(
        sampler.should_sample(volume_threshold as u32, t0),
        "First sample should succeed"
    );
    assert_eq!(sampler.sample_count(), 1);

    // Try to sample 1.5ms later
    // With correct 1ms minimum: should SUCCEED
    // With buggy 1000s minimum: would FAIL
    let t1 = t0 + 1_500_000; // 1.5ms later
    let can_sample_at_1_5ms = sampler.should_sample(volume_threshold as u32, t1);

    // ASSERTION: This should be TRUE with correct implementation
    // The test will FAIL if the bug is present (1000s interval)
    assert!(
        can_sample_at_1_5ms,
        "BUG DETECTED: Cannot sample at 1.5ms! This indicates the min_time_interval \
         is ~1000 seconds instead of 1ms. The units mismatch bug is present. \
         Config passes nanoseconds (1_000_000) but constructor expects milliseconds \
         and multiplies by 1_000_000, resulting in 1_000_000_000_000 ns = ~1000 seconds."
    );

    assert_eq!(sampler.sample_count(), 2, "Should have 2 samples after fix");
}

/// Test that the internal min_time_interval_ns is set correctly.
///
/// With the fix, passing nanoseconds should store nanoseconds (no multiplication).
#[test]
fn test_min_interval_stored_correctly() {
    let min_interval_ns: u64 = 1_000_000; // 1ms in nanoseconds

    // Use non-zero base time to avoid edge case where last_sample_time == 0
    // is used to detect "first sample"
    let t0: u64 = 1_000_000_000; // 1 second (arbitrary non-zero)
    let mut sampler = VolumeBasedSampler::new(100, min_interval_ns);

    // First sample
    assert!(sampler.should_sample(100, t0));

    // At exactly min_interval_ns later, should be able to sample
    let t1 = t0 + min_interval_ns; // Exactly 1ms later
    assert!(
        sampler.should_sample(100, t1),
        "Should sample at exactly min_interval after last sample. \
         If this fails, the interval is not 1ms as expected."
    );

    // Test that we can't sample too soon
    let mut sampler2 = VolumeBasedSampler::new(100, min_interval_ns);
    assert!(sampler2.should_sample(100, t0)); // First sample

    let t_too_soon = t0 + min_interval_ns - 1; // 1 nanosecond too soon
    // Note: accumulated_volume is 0 after first sample, adding 100 meets threshold
    // But time check should fail
    assert!(
        !sampler2.should_sample(100, t_too_soon),
        "Should NOT sample when time is insufficient (1 ns too soon)"
    );
}

/// Test realistic pipeline scenario with multiple samples.
#[test]
fn test_realistic_pipeline_sampling() {
    // Simulate what pipeline does with a realistic 1ms minimum interval
    let min_interval_ns: u64 = 1_000_000; // 1ms
    let volume_threshold: u64 = 1000;
    let mut sampler = VolumeBasedSampler::new(volume_threshold, min_interval_ns);

    // Simulate 10ms of trading with samples every ~1ms when volume allows
    let mut sample_times = Vec::new();
    let start_time: u64 = 1_000_000_000; // 1 second

    // Add 10 events, each 1.5ms apart with volume threshold met
    for i in 0..10 {
        let timestamp = start_time + (i * 1_500_000); // 1.5ms apart
        if sampler.should_sample(volume_threshold as u32, timestamp) {
            sample_times.push(timestamp);
        }
    }

    // With 1ms minimum and 1.5ms between events, all 10 should sample
    assert_eq!(
        sample_times.len(),
        10,
        "Expected 10 samples with 1.5ms intervals and 1ms minimum. \
         Got {} samples. If 0 or 1, the units bug is present (1000s interval).",
        sample_times.len()
    );

    // Verify intervals between samples
    for i in 1..sample_times.len() {
        let interval_ns = sample_times[i] - sample_times[i - 1];
        let interval_ms = interval_ns as f64 / 1_000_000.0;
        assert!(
            interval_ms >= 1.0,
            "Interval {}ms should be >= 1ms minimum",
            interval_ms
        );
    }
}

/// Document the expected API after fix.
///
/// After fix, the constructor parameter should be renamed to `_ns`:
/// `VolumeBasedSampler::new(target_volume: u64, min_time_interval_ns: u64)`
///
/// And no multiplication should occur internally.
#[test]
fn test_api_consistency_with_config() {
    // Config stores nanoseconds with field name `min_time_interval_ns`
    // After fix, constructor should accept nanoseconds with param name `_ns`
    // This test documents the expected consistent API

    let ten_milliseconds_ns: u64 = 10_000_000;

    // Use non-zero base time to avoid edge case where last_sample_time == 0
    let t0: u64 = 1_000_000_000; // 1 second
    
    // Create sampler with 10ms minimum interval
    let mut sampler = VolumeBasedSampler::new(100, ten_milliseconds_ns);
    assert!(sampler.should_sample(100, t0)); // First sample

    // At 5ms: should NOT sample (less than 10ms)
    let t_5ms = t0 + 5_000_000;
    assert!(
        !sampler.should_sample(100, t_5ms),
        "Should not sample at 5ms when minimum is 10ms"
    );

    // At 10ms: SHOULD sample
    let t_10ms = t0 + ten_milliseconds_ns;
    assert!(
        sampler.should_sample(100, t_10ms),
        "Should sample at 10ms when minimum is 10ms"
    );
}

