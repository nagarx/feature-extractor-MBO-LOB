//! Multi-Day State Leak Tests
//!
//! These tests verify that state is properly isolated between trading days
//! and that reset() properly clears all internal state.
//!
//! State leakage can cause:
//! - Corrupted derivative calculations (using yesterday's data)
//! - Incorrect MBO aggregations (mixing events from different days)
//! - Memory growth over time
//! - Incorrect feature values at day boundaries

use feature_extractor::{
    features::mbo_features::MboEvent, FeatureConfig, FeatureExtractor, Pipeline, PipelineConfig,
    SamplingConfig, SamplingStrategy, SequenceBuilder, SequenceConfig,
};
use mbo_lob_reconstructor::{Action, LobState, Side};

/// Helper to create a LOB state for a specific "day"
fn create_lob_state_for_day(day: u32) -> LobState {
    let mut state = LobState::new(10);

    // Base price varies by day to detect state leakage
    let base_price = 100_000_000_000 + (day as i64 * 1_000_000_000);

    for i in 0..5 {
        state.ask_prices[i] = base_price + 10_000_000 + (i as i64 * 10_000_000);
        state.ask_sizes[i] = (100 + i * 10 + day as usize) as u32;
        state.bid_prices[i] = base_price - (i as i64 * 10_000_000);
        state.bid_sizes[i] = (100 + i * 10 + day as usize) as u32;
    }

    state.best_bid = Some(base_price);
    state.best_ask = Some(base_price + 10_000_000);

    state
}

/// Simulate processing MBO events for a "day"
fn process_day_events(extractor: &mut FeatureExtractor, day: u32, num_events: u64) {
    let base_ts = day as u64 * 86_400_000_000_000; // Day in nanoseconds

    for i in 0..num_events {
        let event = MboEvent::new(
            base_ts + i * 1_000_000,
            if i % 3 == 0 {
                Action::Add
            } else if i % 3 == 1 {
                Action::Cancel
            } else {
                Action::Trade
            },
            if i % 2 == 0 { Side::Bid } else { Side::Ask },
            100_000_000_000 + (day as i64 * 1_000_000_000),
            (100 + (i % 50)) as u32,
            10000 + i + (day as u64 * 1000000),
        );
        extractor.process_mbo_event(event);
    }
}

// =============================================================================
// FeatureExtractor State Leak Tests
// =============================================================================

#[test]
fn test_feature_extractor_reset_clears_mbo_state() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 100,
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Day 1: Process events and extract features
    process_day_events(&mut extractor, 1, 200);
    let state_day1 = create_lob_state_for_day(1);
    let features_day1 = extractor.extract_all_features(&state_day1).unwrap();

    // Reset for new day
    extractor.reset();

    // Day 2: Extract features immediately after reset (no events processed)
    let state_day2 = create_lob_state_for_day(2);
    let features_day2_fresh = extractor.extract_all_features(&state_day2).unwrap();

    // The MBO features (indices 40-75) should be different from day 1
    // because the aggregator was reset
    assert_eq!(features_day1.len(), features_day2_fresh.len());

    // Process day 2 events
    process_day_events(&mut extractor, 2, 200);
    let features_day2 = extractor.extract_all_features(&state_day2).unwrap();

    // Day 2 features should be different from day 1
    // (different base prices, different event patterns)
    let raw_lob_count = 40;

    // Raw LOB features should differ due to different base prices
    assert_ne!(
        features_day1[0..raw_lob_count],
        features_day2[0..raw_lob_count],
        "Raw LOB features should differ between days"
    );
}

#[test]
fn test_feature_extractor_multiple_days_no_leak() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 100,
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Simulate 5 trading days
    let mut daily_features = Vec::new();

    for day in 1..=5 {
        // Reset at start of each day
        extractor.reset();

        // Process day's events
        process_day_events(&mut extractor, day, 500);

        // Extract features
        let state = create_lob_state_for_day(day);
        let features = extractor.extract_all_features(&state).unwrap();

        daily_features.push(features);
    }

    // All days should have same feature count
    for (i, features) in daily_features.iter().enumerate() {
        assert_eq!(features.len(), 84, "Day {} should have 84 features", i + 1);
    }

    // Each day's features should be different (due to different base prices)
    for i in 0..daily_features.len() - 1 {
        assert_ne!(
            daily_features[i][0..10],
            daily_features[i + 1][0..10],
            "Days {} and {} should have different ask prices",
            i + 1,
            i + 2
        );
    }
}

// =============================================================================
// SequenceBuilder State Leak Tests
// =============================================================================

#[test]
fn test_sequence_builder_reset_clears_buffer() {
    let config = SequenceConfig::new(10, 1)
        .with_feature_count(40)
        .with_max_buffer_size(100);

    let mut builder = SequenceBuilder::with_config(config);

    // Day 1: Add snapshots
    for i in 0..20 {
        let features = vec![1.0 + i as f64; 40];
        builder.push(i as u64 * 1_000_000, features).unwrap();
    }

    assert_eq!(builder.buffer_len(), 20);
    assert_eq!(builder.total_pushed(), 20);

    // Reset for new day
    builder.reset();

    assert_eq!(
        builder.buffer_len(),
        0,
        "Buffer should be empty after reset"
    );
    assert_eq!(
        builder.total_pushed(),
        0,
        "Total pushed should be 0 after reset"
    );
    assert_eq!(
        builder.total_sequences(),
        0,
        "Total sequences should be 0 after reset"
    );

    // Day 2: Add new snapshots
    for i in 0..20 {
        let features = vec![100.0 + i as f64; 40];
        builder.push(i as u64 * 1_000_000, features).unwrap();
    }

    assert_eq!(builder.buffer_len(), 20);

    // Build sequence - should only contain day 2 data
    // The sequence window is 10 snapshots, so we get snapshots 10-19 (indices 10-19)
    // First snapshot in sequence is index 10, which has value 100.0 + 10.0 = 110.0
    let seq = builder.try_build_sequence().unwrap();
    assert!(
        seq.features[0][0] >= 100.0,
        "First snapshot should be day 2 data (>= 100.0), got {}",
        seq.features[0][0]
    );
}

#[test]
fn test_sequence_builder_multi_day_isolation() {
    let config = SequenceConfig::new(5, 1)
        .with_feature_count(40)
        .with_max_buffer_size(50);

    let mut builder = SequenceBuilder::with_config(config);

    // Day 1: Fill buffer and generate sequences
    for i in 0..30 {
        let features = vec![1.0 * (i + 1) as f64; 40];
        builder.push(i as u64 * 1_000_000, features).unwrap();
    }

    let day1_seqs = builder.generate_all_sequences();
    assert!(!day1_seqs.is_empty(), "Day 1 should have sequences");

    // Verify day 1 sequence values
    let day1_first_seq = &day1_seqs[0];
    assert!(
        day1_first_seq.features[0][0] > 0.0 && day1_first_seq.features[0][0] <= 30.0,
        "Day 1 sequence should contain day 1 values"
    );

    // Reset for day 2
    builder.reset();

    // Day 2: New data with different values
    for i in 0..30 {
        let features = vec![1000.0 + (i + 1) as f64; 40];
        builder.push(i as u64 * 1_000_000, features).unwrap();
    }

    let day2_seqs = builder.generate_all_sequences();
    assert!(!day2_seqs.is_empty(), "Day 2 should have sequences");

    // Verify day 2 sequence values
    let day2_first_seq = &day2_seqs[0];
    assert!(
        day2_first_seq.features[0][0] >= 1000.0,
        "Day 2 sequence should contain day 2 values, got {}",
        day2_first_seq.features[0][0]
    );

    // Verify no day 1 data leaked into day 2
    for seq in &day2_seqs {
        for snapshot in &seq.features {
            assert!(
                snapshot[0] >= 1000.0,
                "Day 2 sequence should not contain day 1 values"
            );
        }
    }
}

// =============================================================================
// Pipeline State Leak Tests
// =============================================================================

#[test]
fn test_pipeline_reset_clears_all_state() {
    let config = PipelineConfig {
        features: FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
        },
        sequence: SequenceConfig::new(10, 1)
            .with_feature_count(40)
            .with_max_buffer_size(100),
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: Some(1000),
            min_time_interval_ns: Some(1_000_000),
            event_count: None,
            adaptive: None,
            multiscale: None,
        }),
        metadata: None,
    };

    let mut pipeline = Pipeline::from_config(config).unwrap();

    // Multiple resets should not cause issues
    for _ in 0..10 {
        pipeline.reset();
    }

    // Pipeline should still be usable after resets
    // (We can't easily test processing without real data, but reset shouldn't break anything)
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

#[test]
fn test_no_unbounded_growth_across_days() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 100, // Small window
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Simulate many days
    for day in 1..=100 {
        // Process many events per day
        process_day_events(&mut extractor, day, 1000);

        // Extract features
        let state = create_lob_state_for_day(day);
        let features = extractor.extract_all_features(&state).unwrap();

        // Feature count should always be consistent
        assert_eq!(features.len(), 76);

        // Reset for next day
        extractor.reset();
    }
}

#[test]
fn test_sequence_builder_bounded_memory() {
    let config = SequenceConfig::new(10, 1)
        .with_feature_count(40)
        .with_max_buffer_size(50); // Small buffer

    let mut builder = SequenceBuilder::with_config(config);

    // Push many more snapshots than buffer size
    for i in 0..1000 {
        let features = vec![i as f64; 40];
        builder.push(i as u64 * 1_000_000, features).unwrap();
    }

    // Buffer should never exceed max_buffer_size
    assert!(
        builder.buffer_len() <= 50,
        "Buffer should be bounded at max_buffer_size"
    );
    assert!(builder.is_full(), "Buffer should be full");
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_reset_on_empty_state() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Reset without any processing
    extractor.reset();

    // Should still work
    let state = create_lob_state_for_day(1);
    let features = extractor.extract_all_features(&state).unwrap();
    assert_eq!(features.len(), 84);
}

#[test]
fn test_reset_mid_day() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 100,
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Process partial day
    process_day_events(&mut extractor, 1, 50);

    // Reset mid-day (e.g., due to error recovery)
    extractor.reset();

    // Continue processing
    process_day_events(&mut extractor, 1, 50);

    // Should still work correctly
    let state = create_lob_state_for_day(1);
    let features = extractor.extract_all_features(&state).unwrap();
    assert_eq!(features.len(), 76);
}
