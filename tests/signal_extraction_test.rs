//! Tests for signal extraction in FeatureExtractor.
//!
//! This test module exposes the contract violation where FeatureExtractor's
//! `feature_count()` claims 98 features when `include_signals=true`, but
//! `extract_into()` only produces 84 features.
//!
//! Per RULE.md §8 (Data Contracts):
//! "Treat any exported artifact or feature vector layout as a public contract"

use feature_extractor::features::mbo_features::MboEvent;
use feature_extractor::features::{FeatureConfig, FeatureExtractor, SignalContext};
use mbo_lob_reconstructor::{Action, LobState, Side};

/// Helper to create a basic LOB state
fn create_test_lob() -> LobState {
    let mut lob = LobState::new(10);
    lob.best_bid = Some(100_000_000_000); // $100.00
    lob.best_ask = Some(100_010_000_000); // $100.01
    for i in 0..10 {
        lob.bid_prices[i] = 100_000_000_000 - (i as i64) * 10_000_000;
        lob.ask_prices[i] = 100_010_000_000 + (i as i64) * 10_000_000;
        lob.bid_sizes[i] = 100;
        lob.ask_sizes[i] = 100;
    }
    lob
}

// ============================================================================
// Contract Verification Tests
// ============================================================================

#[test]
fn test_extract_into_matches_base_feature_count() {
    // Without signals, extract_into should produce exactly base_feature_count() features
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    let lob = create_test_lob();
    
    // Warm up MBO
    for i in 0..100 {
        extractor.process_mbo_event(MboEvent::new(
            i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i
        ));
    }
    
    let mut output = Vec::new();
    extractor.extract_into(&lob, &mut output).expect("extraction should succeed");
    
    // This should pass: base_feature_count() == output.len()
    assert_eq!(
        output.len(),
        extractor.base_feature_count(),
        "extract_into() should produce exactly base_feature_count() features"
    );
}

#[test]
fn test_base_feature_count_excludes_signals() {
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    
    let extractor = FeatureExtractor::with_config(config);
    
    // base_feature_count should be 84 (40 raw + 8 derived + 36 MBO)
    // total_feature_count should be 98 (84 + 14 signals)
    assert_eq!(
        extractor.base_feature_count(), 84,
        "base_feature_count should be 84 (without signals)"
    );
    assert_eq!(
        extractor.feature_count(), 98,
        "feature_count should be 98 (with signals)"
    );
}

#[test]
fn test_extract_into_produces_base_features_only() {
    // With signals enabled, extract_into should still produce only base features
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    let lob = create_test_lob();
    
    // Warm up MBO
    for i in 0..100 {
        extractor.process_mbo_event(MboEvent::new(
            i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i
        ));
    }
    
    let mut output = Vec::new();
    extractor.extract_into(&lob, &mut output).expect("extraction should succeed");
    
    // extract_into produces base features only (84)
    assert_eq!(
        output.len(),
        extractor.base_feature_count(),
        "extract_into() produces base features only, not signals"
    );
    assert_eq!(
        output.len(), 84,
        "Base features should be 84 (40 + 8 + 36)"
    );
}

// ============================================================================
// Signal Extraction Tests (New API)
// ============================================================================

#[test]
fn test_extract_with_signals_produces_full_feature_vector() {
    
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    let lob = create_test_lob();
    
    // Warm up MBO
    for i in 0..100 {
        extractor.process_mbo_event(MboEvent::new(
            i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i
        ));
    }
    
    // Update OFI state (simulate LOB transitions)
    for i in 0..150 {
        let mut lob_variant = lob.clone();
        lob_variant.bid_sizes[0] = 100 + (i % 50);
        extractor.update_ofi(&lob_variant);
    }
    
    let ctx = SignalContext {
        timestamp_ns: 1_000_000_000_000,
        invalidity_delta: 0,
    };
    
    let mut output = Vec::new();
    extractor.extract_with_signals(&lob, &ctx, &mut output)
        .expect("extraction with signals should succeed");
    
    // Should produce full 98 features
    assert_eq!(
        output.len(),
        extractor.feature_count(),
        "extract_with_signals() should produce feature_count() features"
    );
    assert_eq!(
        output.len(), 98,
        "Full feature vector should be 98 (84 base + 14 signals)"
    );
}

#[test]
fn test_update_ofi_tracks_state_changes() {
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    
    // Initially no state changes
    assert!(!extractor.is_signals_warm(), "Should not be warm initially");
    
    let lob = create_test_lob();
    
    // Update OFI 100+ times (MIN_WARMUP_STATE_CHANGES)
    for i in 0..150 {
        let mut lob_variant = lob.clone();
        lob_variant.bid_sizes[0] = 100 + (i % 50);
        extractor.update_ofi(&lob_variant);
    }
    
    assert!(extractor.is_signals_warm(), "Should be warm after 100+ updates");
}

#[test]
fn test_reset_clears_ofi_state() {
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    let lob = create_test_lob();
    
    // Update OFI to warm state
    for i in 0..150 {
        let mut lob_variant = lob.clone();
        lob_variant.bid_sizes[0] = 100 + (i % 50);
        extractor.update_ofi(&lob_variant);
    }
    
    assert!(extractor.is_signals_warm());
    
    // Reset should clear OFI state
    extractor.reset();
    
    assert!(!extractor.is_signals_warm(), "reset() should clear OFI warmup state");
}

#[test]
fn test_extract_with_signals_disabled_returns_error() {
    
    // Signals NOT enabled
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    let lob = create_test_lob();
    
    let ctx = SignalContext {
        timestamp_ns: 1_000_000_000_000,
        invalidity_delta: 0,
    };
    
    let mut output = Vec::new();
    let result = extractor.extract_with_signals(&lob, &ctx, &mut output);
    
    assert!(
        result.is_err(),
        "extract_with_signals() should fail when signals are disabled"
    );
}

// ============================================================================
// Signal Value Correctness Tests
// ============================================================================

#[test]
fn test_signals_have_correct_indices() {
    use feature_extractor::features::signals::indices;
    
    let config = FeatureConfig::new(10)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    
    let mut extractor = FeatureExtractor::with_config(config);
    let lob = create_test_lob();
    
    // Warm up MBO
    for i in 0..100 {
        extractor.process_mbo_event(MboEvent::new(
            i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i
        ));
    }
    
    // Update OFI
    for i in 0..150 {
        let mut lob_variant = lob.clone();
        lob_variant.bid_sizes[0] = 100 + (i % 50);
        extractor.update_ofi(&lob_variant);
    }
    
    let ctx = SignalContext {
        timestamp_ns: 1_000_000_000_000,
        invalidity_delta: 0,
    };
    
    let mut output = Vec::new();
    extractor.extract_with_signals(&lob, &ctx, &mut output).unwrap();
    
    // Verify schema_version at correct index
    assert!(
        (output[indices::SCHEMA_VERSION] - 2.1).abs() < 0.01,
        "schema_version should be 2.1 at index {}, got {}",
        indices::SCHEMA_VERSION,
        output[indices::SCHEMA_VERSION]
    );
    
    // Verify book_valid is binary (0 or 1)
    let book_valid = output[indices::BOOK_VALID];
    assert!(
        book_valid == 0.0 || book_valid == 1.0,
        "book_valid should be 0 or 1, got {}",
        book_valid
    );
    
    // Verify time_regime is in valid range [0, 4]
    let time_regime = output[indices::TIME_REGIME];
    assert!(
        (0.0..=4.0).contains(&time_regime),
        "time_regime should be 0-4, got {}",
        time_regime
    );
}

// ============================================================================
// Modularity Tests
// ============================================================================

#[test]
fn test_feature_extraction_not_tied_to_specific_config() {
    // Test that we can create different feature set configurations
    let configs = vec![
        ("raw_only", FeatureConfig::new(10)),
        ("with_derived", FeatureConfig::new(10).with_derived(true)),
        ("with_mbo", FeatureConfig::new(10).with_derived(true).with_mbo(true)),
        ("full", FeatureConfig::new(10).with_derived(true).with_mbo(true).with_signals(true)),
    ];
    
    let lob = create_test_lob();
    
    for (name, config) in configs {
        let expected_base = config.base_feature_count();
        let expected_total = config.feature_count();
        
        let mut extractor = FeatureExtractor::with_config(config.clone());
        
        // Warm up MBO if enabled
        if config.include_mbo {
            for i in 0..100 {
                extractor.process_mbo_event(MboEvent::new(
                    i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i
                ));
            }
        }
        
        let mut output = Vec::new();
        extractor.extract_into(&lob, &mut output).expect("extraction should succeed");
        
        assert_eq!(
            output.len(), expected_base,
            "Config '{}': extract_into should produce {} features, got {}",
            name, expected_base, output.len()
        );
        
        println!("{}: base={}, total={}", name, expected_base, expected_total);
    }
}

