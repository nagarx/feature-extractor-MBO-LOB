//! Feature Count Validation Tests
//!
//! These tests verify that feature extraction produces the exact number of features
//! expected for each configuration. This is critical for preventing runtime panics
//! in the sequence builder and ensuring model compatibility.

use feature_extractor::{features::mbo_features::MboEvent, FeatureConfig, FeatureExtractor};
use mbo_lob_reconstructor::{Action, LobState, Side};

/// Helper to create a realistic LOB state for testing
fn create_test_lob_state() -> LobState {
    let mut state = LobState::new(10);

    // Populate 5 levels on each side with realistic data
    for i in 0..5 {
        state.ask_prices[i] = 100_010_000_000 + (i as i64 * 10_000_000);
        state.ask_sizes[i] = (100 + i * 10) as u32;
        state.bid_prices[i] = 100_000_000_000 - (i as i64 * 10_000_000);
        state.bid_sizes[i] = (100 + i * 10) as u32;
    }

    state.best_bid = Some(100_000_000_000);
    state.best_ask = Some(100_010_000_000);

    state
}

// =============================================================================
// Feature Count Tests for All Configurations
// =============================================================================

#[test]
fn test_raw_lob_only_10_levels() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: false,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    let features = extractor.extract_lob_features(&state).unwrap();

    // Raw LOB: 10 levels × 4 features (ask_price, ask_size, bid_price, bid_size) = 40
    assert_eq!(
        features.len(),
        40,
        "Raw LOB (10 levels) should produce 40 features"
    );
    assert_eq!(extractor.feature_count(), 40);
}

#[test]
fn test_raw_lob_only_5_levels() {
    let config = FeatureConfig {
        lob_levels: 5,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: false,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    let features = extractor.extract_lob_features(&state).unwrap();

    // Raw LOB: 5 levels × 4 features = 20
    assert_eq!(
        features.len(),
        20,
        "Raw LOB (5 levels) should produce 20 features"
    );
    assert_eq!(extractor.feature_count(), 20);
}

#[test]
fn test_lob_with_derived_10_levels() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: false,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    let features = extractor.extract_lob_features(&state).unwrap();

    // Raw LOB (40) + Derived (8) = 48
    assert_eq!(
        features.len(),
        48,
        "LOB + Derived (10 levels) should produce 48 features"
    );
    assert_eq!(extractor.feature_count(), 48);
}

#[test]
fn test_lob_with_mbo_10_levels() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let mut extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    // Process some MBO events to populate the aggregator
    for i in 0..100 {
        let event = MboEvent::new(
            1_000_000_000 + i * 1_000_000,
            Action::Add,
            if i % 2 == 0 { Side::Bid } else { Side::Ask },
            100_000_000_000,
            100,
            10000 + i,
        );
        extractor.process_mbo_event(event);
    }

    let features = extractor.extract_all_features(&state).unwrap();

    // Raw LOB (40) + MBO (36) = 76
    assert_eq!(
        features.len(),
        76,
        "LOB + MBO (10 levels) should produce 76 features"
    );
    assert_eq!(extractor.feature_count(), 76);
}

#[test]
fn test_full_features_10_levels() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let mut extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    // Process some MBO events
    for i in 0..100 {
        let event = MboEvent::new(
            1_000_000_000 + i * 1_000_000,
            Action::Add,
            if i % 2 == 0 { Side::Bid } else { Side::Ask },
            100_000_000_000,
            100,
            10000 + i,
        );
        extractor.process_mbo_event(event);
    }

    let features = extractor.extract_all_features(&state).unwrap();

    // Raw LOB (40) + Derived (8) + MBO (36) = 84
    assert_eq!(
        features.len(),
        84,
        "Full features (10 levels) should produce 84 features"
    );
    assert_eq!(extractor.feature_count(), 84);
}

// =============================================================================
// Feature Count Consistency Tests
// =============================================================================

#[test]
fn test_feature_count_matches_extraction() {
    // Test that feature_count() always matches the actual extracted features
    let configs = vec![
        (false, false, 40), // Raw LOB only
        (true, false, 48),  // LOB + Derived
        (false, true, 76),  // LOB + MBO
        (true, true, 84),   // Full
    ];

    for (include_derived, include_mbo, expected) in configs {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived,
            include_mbo,
            mbo_window_size: 1000,
            include_signals: false,
        };

        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        // Process MBO events if MBO is enabled
        if include_mbo {
            for i in 0..100 {
                let event = MboEvent::new(
                    1_000_000_000 + i * 1_000_000,
                    Action::Add,
                    if i % 2 == 0 { Side::Bid } else { Side::Ask },
                    100_000_000_000,
                    100,
                    10000 + i,
                );
                extractor.process_mbo_event(event);
            }
        }

        let features = extractor.extract_all_features(&state).unwrap();

        assert_eq!(
            extractor.feature_count(),
            expected,
            "feature_count() should be {} for derived={}, mbo={}",
            expected,
            include_derived,
            include_mbo
        );

        assert_eq!(
            features.len(),
            extractor.feature_count(),
            "Extracted features should match feature_count() for derived={}, mbo={}",
            include_derived,
            include_mbo
        );
    }
}

#[test]
fn test_feature_count_scales_with_levels() {
    // Test that feature count scales correctly with LOB levels
    for levels in [5, 10, 15, 20] {
        let config = FeatureConfig {
            lob_levels: levels,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
            include_signals: false,
        };

        let extractor = FeatureExtractor::with_config(config);

        // Raw LOB features = levels × 4
        let expected = levels * 4;
        assert_eq!(
            extractor.feature_count(),
            expected,
            "Raw LOB with {} levels should produce {} features",
            levels,
            expected
        );
    }
}

// =============================================================================
// Reset and State Isolation Tests
// =============================================================================

#[test]
fn test_reset_clears_state() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 100,
        include_signals: false,
    };

    let mut extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    // Process events to build up state
    for i in 0..200 {
        let event = MboEvent::new(
            1_000_000_000 + i * 1_000_000,
            Action::Add,
            Side::Bid,
            100_000_000_000,
            100,
            10000 + i,
        );
        extractor.process_mbo_event(event);
    }

    // Extract features before reset
    let features_before = extractor.extract_all_features(&state).unwrap();

    // Reset
    extractor.reset();

    // Extract features after reset
    let features_after = extractor.extract_all_features(&state).unwrap();

    // Both should have same length
    assert_eq!(features_before.len(), features_after.len());

    // Feature count should still be correct
    assert_eq!(extractor.feature_count(), 76);
}

#[test]
fn test_multiple_resets() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Multiple resets should not cause issues
    for _ in 0..10 {
        extractor.reset();
    }

    // Should still work correctly
    let state = create_test_lob_state();
    let features = extractor.extract_all_features(&state).unwrap();
    assert_eq!(features.len(), 84);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_empty_lob_state() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::with_config(config);

    // Empty LOB state (all zeros)
    let state = LobState::new(10);

    let features = extractor.extract_lob_features(&state).unwrap();

    // Should still produce correct number of features
    assert_eq!(features.len(), 40);

    // Features should be zeros or defaults
    for f in &features {
        assert!(f.is_finite(), "All features should be finite");
    }
}

#[test]
fn test_single_level_lob() {
    let config = FeatureConfig {
        lob_levels: 1,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: false,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    let features = extractor.extract_lob_features(&state).unwrap();

    // 1 level × 4 features = 4
    assert_eq!(features.len(), 4);
}

// =============================================================================
// Derived Feature Tests
// =============================================================================

#[test]
fn test_derived_features_count() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: false,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    let features = extractor.extract_lob_features(&state).unwrap();

    // Derived features should add exactly 8 features:
    // - Mid-price
    // - Spread
    // - Spread (bps)
    // - Total bid volume
    // - Total ask volume
    // - Volume imbalance
    // - Weighted mid-price
    // - Price impact
    let raw_count = 10 * 4; // 40
    let derived_count = 8;
    assert_eq!(features.len(), raw_count + derived_count);
}

// =============================================================================
// MBO Feature Tests
// =============================================================================

#[test]
fn test_mbo_features_count() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: false,
    };

    let mut extractor = FeatureExtractor::with_config(config);
    let state = create_test_lob_state();

    // MBO features should add exactly 36 features
    let features = extractor.extract_all_features(&state).unwrap();

    let raw_count = 10 * 4; // 40
    let mbo_count = 36;
    assert_eq!(features.len(), raw_count + mbo_count);
}

#[test]
fn test_mbo_event_processing() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: false,
        include_mbo: true,
        mbo_window_size: 100,
        include_signals: false,
    };

    let mut extractor = FeatureExtractor::with_config(config);

    // Test all action types
    let actions = [Action::Add, Action::Cancel, Action::Trade, Action::Modify];

    let sides = [Side::Bid, Side::Ask];

    for (i, action) in actions.iter().enumerate() {
        for (j, side) in sides.iter().enumerate() {
            let event = MboEvent::new(
                1_000_000_000 + (i * 10 + j) as u64 * 1_000_000,
                *action,
                *side,
                100_000_000_000,
                100,
                10000 + (i * 10 + j) as u64,
            );
            extractor.process_mbo_event(event);
        }
    }

    let state = create_test_lob_state();
    let features = extractor.extract_all_features(&state).unwrap();

    // Should still produce correct count
    assert_eq!(features.len(), 76);
}
