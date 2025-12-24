//! Test to expose and validate tick_size propagation into MBO depth_ticks computation.
//!
//! This test demonstrates that the depth_ticks_bid/ask functions use a hardcoded
//! tick size of $0.01 instead of the configured tick_size, causing incorrect
//! depth calculations for non-US instruments.
//!
//! Per RULE.md §1 (Numerical Precision):
//! "Every quantity must have explicit units at boundaries"
//! "Zero tolerance for precision errors — they compound into losses"

use feature_extractor::features::mbo_features::{MboAggregator, MboEvent};
use mbo_lob_reconstructor::{Action, LobState, Side};

/// Helper to create a LOB state with prices at specific tick distances
fn create_lob_with_depth(best_bid: i64, best_ask: i64, tick_size_nano: i64) -> LobState {
    let mut lob = LobState::new(10);
    
    // Best bid and ask
    lob.best_bid = Some(best_bid);
    lob.best_ask = Some(best_ask);
    
    // Bid side: levels at 0, 1, 2, 3 ticks away from best
    // With 100 shares at each level
    for i in 0..4 {
        lob.bid_prices[i] = best_bid - (i as i64) * tick_size_nano;
        lob.bid_sizes[i] = 100;
    }
    
    // Ask side: levels at 0, 1, 2, 3 ticks away from best
    // With 100 shares at each level
    for i in 0..4 {
        lob.ask_prices[i] = best_ask + (i as i64) * tick_size_nano;
        lob.ask_sizes[i] = 100;
    }
    
    lob
}

#[test]
fn test_depth_ticks_with_standard_tick_size() {
    // Standard US equity: $0.01 tick = 10_000_000 nanodollars
    let tick_nano = 10_000_000i64; // $0.01
    let best_bid = 100_000_000_000i64; // $100.00
    let best_ask = 100_010_000_000i64; // $100.01
    
    let lob = create_lob_with_depth(best_bid, best_ask, tick_nano);
    
    // Create aggregator with default tick size ($0.01)
    let mut aggregator = MboAggregator::new();
    
    // Extract features
    let features = aggregator.extract_features(&lob);
    
    // Find depth_ticks_bid and depth_ticks_ask in features
    // MBO feature layout:
    // [0-11]: Order flow (12 features)
    // [12-19]: Size distribution (8 features)
    // [20-25]: Queue & depth (6 features)
    //   - [20] average_queue_position
    //   - [21] queue_size_ahead
    //   - [22] orders_per_level
    //   - [23] level_concentration
    //   - [24] depth_ticks_bid
    //   - [25] depth_ticks_ask
    let depth_ticks_bid = features[24];
    let depth_ticks_ask = features[25];
    
    // With 4 levels at 0, 1, 2, 3 ticks away, each with 100 shares:
    // Weighted avg = (100*0 + 100*1 + 100*2 + 100*3) / 400 = 600/400 = 1.5 ticks
    let expected = 1.5;
    
    assert!(
        (depth_ticks_bid - expected).abs() < 0.01,
        "Standard tick: depth_ticks_bid should be {}, got {}",
        expected, depth_ticks_bid
    );
    assert!(
        (depth_ticks_ask - expected).abs() < 0.01,
        "Standard tick: depth_ticks_ask should be {}, got {}",
        expected, depth_ticks_ask
    );
}

#[test]
fn test_depth_ticks_requires_correct_tick_size_configuration() {
    // This test demonstrates WHY tick_size MUST be configured correctly.
    // Using default $0.01 tick for crypto ($0.001 tick) gives WRONG results.
    
    // Crypto: $0.001 tick = 1_000_000 nanodollars (10x smaller than US equity)
    let crypto_tick_nano = 1_000_000i64; // $0.001
    let best_bid = 50_000_000_000_000i64; // $50,000.00 (Bitcoin-like)
    let best_ask = 50_000_001_000_000i64; // $50,000.001
    
    let lob = create_lob_with_depth(best_bid, best_ask, crypto_tick_nano);
    
    // Create aggregator with WRONG tick size (default $0.01 instead of $0.001)
    let mut aggregator = MboAggregator::new(); // Uses default $0.01
    let features = aggregator.extract_features(&lob);
    
    let depth_ticks_bid = features[24];
    let _depth_ticks_ask = features[25];
    
    // With WRONG tick size ($0.01 instead of $0.001):
    // Distance in $0.01 ticks: 0, 0.1, 0.2, 0.3 → integer division = 0, 0, 0, 0
    // Weighted avg = 0.0 (completely wrong!)
    //
    // This is the EXPECTED behavior when tick_size is misconfigured.
    // Users MUST call .with_tick_size(0.001) for crypto instruments.
    
    assert!(
        depth_ticks_bid < 0.01,
        "With WRONG tick_size ($0.01 default for $0.001 crypto), \
         depth_ticks_bid should round to 0.0. Got {}. \
         This demonstrates the importance of configuring tick_size correctly.",
        depth_ticks_bid
    );
    
    // Now verify CORRECT configuration gives right answer
    let mut aggregator_correct = MboAggregator::new()
        .with_tick_size(0.001); // Correct $0.001 tick for crypto
    let features_correct = aggregator_correct.extract_features(&lob);
    let depth_ticks_bid_correct = features_correct[24];
    
    let expected = 1.5;
    assert!(
        (depth_ticks_bid_correct - expected).abs() < 0.01,
        "With CORRECT tick_size ($0.001), depth_ticks_bid should be {}, got {}",
        expected, depth_ticks_bid_correct
    );
}

// ============================================================================
// TESTS REQUIRING THE FIX (uncomment after implementing with_tick_size)
// ============================================================================

#[test]
fn test_depth_ticks_with_configured_tick_size() {
    // This test validates the FIX: MboAggregator should accept tick_size parameter
    let crypto_tick_nano = 1_000_000i64; // $0.001
    let crypto_tick_dollars = 0.001f64;
    let best_bid = 50_000_000_000_000i64; // $50,000.00
    let best_ask = 50_000_001_000_000i64; // $50,000.001
    
    let lob = create_lob_with_depth(best_bid, best_ask, crypto_tick_nano);
    
    // Create aggregator with custom tick size
    let mut aggregator = MboAggregator::new()
        .with_tick_size(crypto_tick_dollars);
    
    let features = aggregator.extract_features(&lob);
    let depth_ticks_bid = features[24];
    let depth_ticks_ask = features[25];
    
    let expected = 1.5;
    
    assert!(
        (depth_ticks_bid - expected).abs() < 0.01,
        "With configured $0.001 tick, depth_ticks_bid should be {}, got {}",
        expected, depth_ticks_bid
    );
    assert!(
        (depth_ticks_ask - expected).abs() < 0.01,
        "With configured $0.001 tick, depth_ticks_ask should be {}, got {}",
        expected, depth_ticks_ask
    );
}

#[test]
fn test_feature_extractor_uses_config_tick_size() {
    use feature_extractor::features::FeatureExtractor;
    use feature_extractor::FeatureConfig;
    
    let crypto_tick_nano = 1_000_000i64; // $0.001
    let crypto_tick_dollars = 0.001f64;
    let best_bid = 50_000_000_000_000i64; // $50,000.00
    let best_ask = 50_000_001_000_000i64; // $50,000.001
    
    let lob = create_lob_with_depth(best_bid, best_ask, crypto_tick_nano);
    
    // Create FeatureExtractor with custom tick_size
    let config = FeatureConfig::new(10)
        .with_tick_size(crypto_tick_dollars)
        .with_mbo(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    
    // Process some events to warm up MBO (required for valid features)
    for i in 0..100 {
        let event = MboEvent::new(
            i * 1_000_000,
            Action::Add,
            Side::Bid,
            best_bid,
            100,
            i,
        );
        extractor.process_mbo_event(event);
    }
    
    let features = extractor.extract_all_features(&lob).unwrap();
    
    // MBO features start at index 40 (after 40 raw, derived disabled)
    // depth_ticks_bid is at MBO index 24, so full index = 40 + 24 = 64
    // depth_ticks_ask is at MBO index 25, so full index = 40 + 25 = 65
    
    let depth_ticks_bid = features[64];
    let _depth_ticks_ask = features[65];
    
    let expected = 1.5;
    
    assert!(
        (depth_ticks_bid - expected).abs() < 0.5, // Allow some tolerance due to MBO warmup
        "FeatureExtractor with $0.001 tick: depth_ticks_bid should be ~{}, got {}. \
         This indicates tick_size is not being propagated from FeatureConfig to MboAggregator.",
        expected, depth_ticks_bid
    );
}

#[test]
fn test_different_tick_sizes() {
    // Test various tick sizes to ensure correct scaling
    let test_cases = vec![
        (0.01, 10_000_000i64, "US equity ($0.01)"),
        (0.001, 1_000_000i64, "Crypto ($0.001)"),
        (0.0001, 100_000i64, "Forex pip ($0.0001)"),
        (1.0, 1_000_000_000i64, "Futures ($1.00)"),
        (0.05, 50_000_000i64, "Some ETFs ($0.05)"),
    ];
    
    for (tick_dollars, tick_nano, description) in test_cases {
        let best_bid = 100_000_000_000i64; // $100.00
        let best_ask = best_bid + tick_nano; // $100.00 + 1 tick
        
        let lob = create_lob_with_depth(best_bid, best_ask, tick_nano);
        
        let mut aggregator = MboAggregator::new()
            .with_tick_size(tick_dollars);
        
        let features = aggregator.extract_features(&lob);
        let depth_ticks_bid = features[24];
        let depth_ticks_ask = features[25];
        
        // Expected: 1.5 ticks for both sides
        let expected = 1.5;
        
        assert!(
            (depth_ticks_bid - expected).abs() < 0.01,
            "{}: depth_ticks_bid should be {}, got {}",
            description, expected, depth_ticks_bid
        );
        assert!(
            (depth_ticks_ask - expected).abs() < 0.01,
            "{}: depth_ticks_ask should be {}, got {}",
            description, expected, depth_ticks_ask
        );
    }
}

#[test]
fn test_default_tick_size_is_001() {
    // Verify that the default tick size is $0.01
    let aggregator = MboAggregator::new();
    
    // The default tick size should be $0.01 = 10_000_000 nanodollars
    assert_eq!(
        aggregator.tick_size_nanodollars(),
        10_000_000,
        "Default tick size should be 10_000_000 nanodollars ($0.01)"
    );
}

#[test]
fn test_with_tick_size_builder_pattern() {
    let aggregator = MboAggregator::new()
        .with_tick_size(0.001) // $0.001 = 1_000_000 nanodollars
        .with_queue_tracking();
    
    assert_eq!(
        aggregator.tick_size_nanodollars(),
        1_000_000,
        "with_tick_size(0.001) should set tick_size to 1_000_000 nanodollars"
    );
}
