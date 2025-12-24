//! Test to expose and validate the multi-scale reset leak bug.
//!
//! This test demonstrates that calling reset() on MultiScaleWindow without
//! calling try_build_all() first will cause accumulated sequences from
//! the previous day to leak into the next day's output.
//!
//! This is a data integrity bug that violates RULE.md §6 (Determinism & Reproducibility)
//! and §7 (Data Integrity).

use feature_extractor::sequence_builder::{MultiScaleConfig, MultiScaleWindow, ScaleConfig};
use std::sync::Arc;

/// Create a feature vector with a marker value to identify which "day" it came from.
fn make_features(day_marker: f64, event_id: usize, feature_count: usize) -> Arc<Vec<f64>> {
    let mut features = vec![day_marker; feature_count];
    features[0] = day_marker;
    features[1] = event_id as f64;
    Arc::new(features)
}

#[test]
fn test_reset_clears_accumulated_sequences() {
    // Configuration: small windows to quickly generate sequences
    let config = MultiScaleConfig::new(
        ScaleConfig::new(5, 1, 1),  // Fast: window 5, decimation 1, stride 1
        ScaleConfig::new(10, 2, 1), // Medium: window 10, decimation 2, stride 1
        ScaleConfig::new(20, 4, 1), // Slow: window 20, decimation 4, stride 1
    );
    let feature_count = 4;
    let mut window = MultiScaleWindow::new(config, feature_count);

    // === DAY 1: Push 50 events (marker = 1.0) ===
    for i in 0..50 {
        window.push_arc(i as u64 * 1_000_000, make_features(1.0, i, feature_count));
    }

    // Verify Day 1 produced sequences
    let (fast_count, medium_count, slow_count) = window.buffer_counts();
    assert!(
        fast_count > 0 || medium_count > 0 || slow_count > 0,
        "Day 1 should produce some sequences in buffers"
    );

    // Check accumulated sequences exist
    let acc_counts = window.accumulated_counts();
    println!(
        "Day 1 accumulated: fast={}, medium={}, slow={}",
        acc_counts.0, acc_counts.1, acc_counts.2
    );

    // === RESET WITHOUT CALLING try_build_all() ===
    // This is where the bug manifests: accumulated sequences should be cleared
    window.reset();

    // === DAY 2: Push 50 events (marker = 2.0) ===
    for i in 0..50 {
        window.push_arc(
            (i as u64 + 100) * 1_000_000, // Different timestamp range
            make_features(2.0, i, feature_count),
        );
    }

    // === BUILD ALL: Should only contain Day 2 sequences ===
    let multiscale = window.try_build_all();
    assert!(multiscale.is_some(), "Should have sequences after Day 2");

    let ms = multiscale.unwrap();
    let (fast_seqs, medium_seqs, slow_seqs) = (ms.fast(), ms.medium(), ms.slow());

    // Check that ALL sequences have Day 2 marker (2.0)
    let mut day1_leak_count = 0;
    let mut total_sequences = 0;

    for seq in fast_seqs.iter().chain(medium_seqs.iter()).chain(slow_seqs.iter()) {
        total_sequences += 1;
        // Check the first feature of the first snapshot in the sequence
        if !seq.features.is_empty() {
            let first_snapshot = &seq.features[0];
            let marker = first_snapshot[0];
            if (marker - 1.0).abs() < 0.001 {
                day1_leak_count += 1;
            }
        }
    }

    assert_eq!(
        day1_leak_count, 0,
        "DATA INTEGRITY BUG: {} of {} sequences have Day 1 marker after reset(). \
         This indicates accumulated sequences were not cleared on reset(), causing \
         cross-day state leakage. This violates RULE.md §6 (Determinism) and §7 (Data Integrity).",
        day1_leak_count, total_sequences
    );

    println!(
        "✓ All {} sequences have Day 2 marker (no leakage)",
        total_sequences
    );
}

#[test]
fn test_reset_clears_all_state() {
    let config = MultiScaleConfig::new(
        ScaleConfig::new(5, 1, 1),
        ScaleConfig::new(10, 2, 1),
        ScaleConfig::new(20, 4, 1),
    );
    let feature_count = 4;
    let mut window = MultiScaleWindow::new(config.clone(), feature_count);

    // Push some events
    for i in 0..100 {
        window.push_arc(i as u64 * 1_000_000, make_features(1.0, i, feature_count));
    }

    // Verify state was built up
    assert!(window.total_events() > 0, "Should have events tracked");
    let acc = window.accumulated_counts();
    assert!(
        acc.0 > 0 || acc.1 > 0 || acc.2 > 0,
        "Should have accumulated sequences"
    );

    // Reset
    window.reset();

    // Verify ALL state is cleared
    assert_eq!(window.total_events(), 0, "total_events should be 0 after reset");
    
    let counters = window.decimation_counters();
    assert_eq!(counters, (0, 0, 0), "decimation counters should be 0 after reset");
    
    let acc_after = window.accumulated_counts();
    assert_eq!(
        acc_after, (0, 0, 0),
        "ACCUMULATED SEQUENCES NOT CLEARED: got {:?}. \
         This is the bug - reset() must clear accumulated_fast/medium/slow vectors.",
        acc_after
    );
    
    let buffers = window.buffer_counts();
    assert_eq!(buffers, (0, 0, 0), "builder buffers should be empty after reset");
}

#[test]
fn test_reset_produces_identical_state_to_new() {
    let config = MultiScaleConfig::new(
        ScaleConfig::new(5, 1, 1),
        ScaleConfig::new(10, 2, 1),
        ScaleConfig::new(20, 4, 1),
    );
    let feature_count = 4;

    // Create window, use it, reset it
    let mut window1 = MultiScaleWindow::new(config.clone(), feature_count);
    for i in 0..100 {
        window1.push_arc(i as u64 * 1_000_000, make_features(1.0, i, feature_count));
    }
    window1.reset();

    // Create fresh window
    let window2 = MultiScaleWindow::new(config, feature_count);

    // Compare observable state
    assert_eq!(
        window1.total_events(),
        window2.total_events(),
        "total_events should match fresh window"
    );
    assert_eq!(
        window1.decimation_counters(),
        window2.decimation_counters(),
        "decimation_counters should match fresh window"
    );
    assert_eq!(
        window1.accumulated_counts(),
        window2.accumulated_counts(),
        "accumulated_counts should match fresh window (both should be (0,0,0))"
    );
    assert_eq!(
        window1.buffer_counts(),
        window2.buffer_counts(),
        "buffer_counts should match fresh window"
    );
}

#[test]
fn test_try_build_all_then_reset_is_clean() {
    // This tests the "happy path" where try_build_all() is called before reset()
    let config = MultiScaleConfig::new(
        ScaleConfig::new(5, 1, 1),
        ScaleConfig::new(10, 2, 1),
        ScaleConfig::new(20, 4, 1),
    );
    let feature_count = 4;
    let mut window = MultiScaleWindow::new(config, feature_count);

    // Day 1
    for i in 0..50 {
        window.push_arc(i as u64 * 1_000_000, make_features(1.0, i, feature_count));
    }

    // Consume Day 1 sequences
    let day1_result = window.try_build_all();
    assert!(day1_result.is_some(), "Day 1 should produce sequences");

    // Reset
    window.reset();

    // Day 2
    for i in 0..50 {
        window.push_arc(
            (i as u64 + 100) * 1_000_000,
            make_features(2.0, i, feature_count),
        );
    }

    // Day 2 sequences
    let day2_result = window.try_build_all();
    assert!(day2_result.is_some(), "Day 2 should produce sequences");

    let ms = day2_result.unwrap();
    
    // All should be Day 2 only
    for seq in ms.fast().iter().chain(ms.medium().iter()).chain(ms.slow().iter()) {
        if !seq.features.is_empty() {
            let marker = seq.features[0][0];
            assert!(
                (marker - 2.0).abs() < 0.001,
                "All Day 2 sequences should have marker 2.0, got {}",
                marker
            );
        }
    }
}

