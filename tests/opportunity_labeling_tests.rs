//! Integration Tests for Opportunity-Based Labeling
//!
//! These tests validate the opportunity labeling module's integration with
//! the feature extraction pipeline and verify correctness across different
//! configurations.

use feature_extractor::contract;
use feature_extractor::{
    ConflictPriority, OpportunityConfig, OpportunityLabel, OpportunityLabelGenerator,
};

// ============================================================================
// OpportunityLabel Integration Tests
// ============================================================================

#[test]
fn test_opportunity_label_roundtrip() {
    // Test int roundtrip
    for original in [
        OpportunityLabel::BigDown,
        OpportunityLabel::NoOpportunity,
        OpportunityLabel::BigUp,
    ] {
        let as_int = original.as_int();
        let recovered = OpportunityLabel::from_int(as_int).unwrap();
        assert_eq!(original, recovered);
    }

    // Test class index roundtrip
    for original in [
        OpportunityLabel::BigDown,
        OpportunityLabel::NoOpportunity,
        OpportunityLabel::BigUp,
    ] {
        let as_idx = original.as_class_index();
        let recovered = OpportunityLabel::from_class_index(as_idx).unwrap();
        assert_eq!(original, recovered);
    }
}

#[test]
fn test_opportunity_label_ordering() {
    // Class indices should be 0, 1, 2 (matching TrendLabel convention)
    assert_eq!(OpportunityLabel::BigDown.as_class_index(), 0);
    assert_eq!(OpportunityLabel::NoOpportunity.as_class_index(), 1);
    assert_eq!(OpportunityLabel::BigUp.as_class_index(), 2);

    // Int values should be -1, 0, 1
    assert_eq!(OpportunityLabel::BigDown.as_int(), -1);
    assert_eq!(OpportunityLabel::NoOpportunity.as_int(), 0);
    assert_eq!(OpportunityLabel::BigUp.as_int(), 1);
}

// ============================================================================
// OpportunityConfig Integration Tests
// ============================================================================

#[test]
fn test_opportunity_config_presets() {
    // Test all presets create valid configurations
    let presets = [
        OpportunityConfig::medium_moves(),
        OpportunityConfig::whale_detection(),
        OpportunityConfig::extreme_moves(),
        OpportunityConfig::default(),
    ];

    for config in presets {
        assert!(config.validate().is_ok());
        assert!(config.horizon > 0);
        assert!(config.threshold > 0.0);
        assert!(config.threshold < 1.0);
    }
}

#[test]
fn test_opportunity_config_asymmetric() {
    // Test asymmetric thresholds
    let config = OpportunityConfig::with_asymmetric_thresholds(100, 0.005, 0.003);

    assert_eq!(config.up_threshold(), 0.005);
    assert_eq!(config.down_threshold(), 0.003);
    assert!(config.validate().is_ok());
}

// ============================================================================
// OpportunityLabelGenerator Integration Tests
// ============================================================================

#[test]
fn test_opportunity_generator_realistic_scenario() {
    // Simulate a realistic scenario with price spikes
    let config = OpportunityConfig::new(10, 0.02); // 2% threshold, 10 step horizon
    let mut generator = OpportunityLabelGenerator::new(config);

    // Create a realistic price series:
    // - Start stable at 100
    // - Spike up to 103 (3% up)
    // - Return to 100
    // - Spike down to 97 (3% down)
    // - Return to 100
    let mut prices = Vec::new();

    // Stable period
    for _ in 0..10 {
        prices.push(100.0);
    }

    // Spike up
    prices.push(101.0);
    prices.push(102.0);
    prices.push(103.0);
    prices.push(102.0);
    prices.push(101.0);

    // Stable
    for _ in 0..5 {
        prices.push(100.0);
    }

    // Spike down
    prices.push(99.0);
    prices.push(98.0);
    prices.push(97.0);
    prices.push(98.0);
    prices.push(99.0);

    // Stable end
    for _ in 0..10 {
        prices.push(100.0);
    }

    generator.add_prices(&prices);
    let labels = generator.generate_labels().unwrap();
    let stats = generator.compute_stats(&labels);

    // Should have detected both up and down opportunities
    assert!(
        stats.big_up_count > 0,
        "Should detect upward spike opportunities"
    );
    assert!(
        stats.big_down_count > 0,
        "Should detect downward spike opportunities"
    );

    // Should have some no-opportunity labels
    assert!(
        stats.no_opportunity_count > 0,
        "Should have stable periods marked as no opportunity"
    );

    // Opportunity rate should be reasonable (not 100%)
    let opportunity_rate = stats.opportunity_rate();
    assert!(
        opportunity_rate < 1.0,
        "Not all samples should be opportunities"
    );
    assert!(
        opportunity_rate > 0.0,
        "Should have some opportunities detected"
    );
}

#[test]
fn test_opportunity_generator_volatile_market() {
    // Simulate a volatile market where both up and down moves occur
    let config = OpportunityConfig::new(5, 0.03); // 3% threshold
    let mut generator = OpportunityLabelGenerator::new(config);

    // Highly volatile prices with both directions in same window
    let prices = vec![
        100.0, 97.0, 104.0, 99.0, 101.0, 96.0, 103.0, 100.0, 98.0, 102.0, 100.0, 100.0, 100.0,
        100.0, 100.0,
    ];

    generator.add_prices(&prices);
    let labels = generator.generate_labels().unwrap();
    let stats = generator.compute_stats(&labels);

    // Should have detected conflicts (both up and down triggered)
    // The default LargerMagnitude strategy should resolve them
    assert!(
        stats.conflict_count > 0,
        "Should have some conflict cases in volatile data"
    );

    // All labels should be valid
    for (_, label, max_ret, min_ret) in &labels {
        assert!(!max_ret.is_nan());
        assert!(!min_ret.is_nan());
        assert!(matches!(
            label,
            OpportunityLabel::BigUp | OpportunityLabel::BigDown | OpportunityLabel::NoOpportunity
        ));
    }
}

#[test]
fn test_opportunity_generator_conflict_strategies() {
    // Create a scenario guaranteed to trigger conflicts
    // At t=0, the window [1..5] will have max=110 (10% up) and min=90 (10% down)
    let prices = vec![
        100.0, 90.0, 110.0, 90.0, 110.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];

    // Test LargerMagnitude strategy
    let config =
        OpportunityConfig::new(5, 0.05).with_conflict_priority(ConflictPriority::LargerMagnitude);
    let mut gen = OpportunityLabelGenerator::new(config);
    gen.add_prices(&prices);
    let labels = gen.generate_labels().unwrap();
    let (_, label, _, _) = &labels[0];
    // Both have same magnitude (10%), so should pick one (implementation-dependent)
    assert!(
        label.is_opportunity(),
        "Should resolve conflict to an opportunity"
    );

    // Test UpPriority strategy
    let config =
        OpportunityConfig::new(5, 0.05).with_conflict_priority(ConflictPriority::UpPriority);
    let mut gen = OpportunityLabelGenerator::new(config);
    gen.add_prices(&prices);
    let labels = gen.generate_labels().unwrap();
    let (_, label, _, _) = &labels[0];
    assert_eq!(
        *label,
        OpportunityLabel::BigUp,
        "UpPriority should resolve to BigUp"
    );

    // Test DownPriority strategy
    let config =
        OpportunityConfig::new(5, 0.05).with_conflict_priority(ConflictPriority::DownPriority);
    let mut gen = OpportunityLabelGenerator::new(config);
    gen.add_prices(&prices);
    let labels = gen.generate_labels().unwrap();
    let (_, label, _, _) = &labels[0];
    assert_eq!(
        *label,
        OpportunityLabel::BigDown,
        "DownPriority should resolve to BigDown"
    );

    // Test Ambiguous strategy
    let config =
        OpportunityConfig::new(5, 0.05).with_conflict_priority(ConflictPriority::Ambiguous);
    let mut gen = OpportunityLabelGenerator::new(config);
    gen.add_prices(&prices);
    let labels = gen.generate_labels().unwrap();
    let (_, label, _, _) = &labels[0];
    assert_eq!(
        *label,
        OpportunityLabel::NoOpportunity,
        "Ambiguous should resolve to NoOpportunity"
    );
}

#[test]
fn test_opportunity_stats_analysis() {
    let config = OpportunityConfig::new(10, 0.02);
    let mut generator = OpportunityLabelGenerator::new(config);

    // Create balanced-ish dataset
    let mut prices = Vec::new();

    // Up moves
    for _ in 0..5 {
        prices.extend(vec![100.0, 100.0, 100.0, 103.0, 103.0, 103.0]);
    }

    // Stable periods
    for _ in 0..10 {
        prices.extend(vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
    }

    // Down moves
    for _ in 0..5 {
        prices.extend(vec![100.0, 100.0, 100.0, 97.0, 97.0, 97.0]);
    }

    // End with stable
    for _ in 0..10 {
        prices.push(100.0);
    }

    generator.add_prices(&prices);
    let labels = generator.generate_labels().unwrap();
    let stats = generator.compute_stats(&labels);

    // Verify stats consistency
    assert_eq!(
        stats.big_up_count + stats.big_down_count + stats.no_opportunity_count,
        stats.total
    );

    // Verify class balance sums to 1
    let (up, no, down) = stats.class_balance();
    let sum = up + no + down;
    assert!(
        (sum - 1.0).abs() < contract::FLOAT_CMP_EPS,
        "Class balance should sum to 1.0, got {}",
        sum
    );

    // Verify peak returns are finite and sensible
    assert!(stats.peak_max_return.is_finite());
    assert!(stats.peak_min_return.is_finite());
    assert!(
        stats.peak_max_return >= 0.0,
        "Peak max return should be >= 0"
    );
    assert!(
        stats.peak_min_return <= 0.0,
        "Peak min return should be <= 0"
    );
}

#[test]
fn test_opportunity_generator_determinism() {
    // Verify that the same input always produces the same output
    let config = OpportunityConfig::new(10, 0.02);
    let prices: Vec<f64> = (0..100)
        .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
        .collect();

    let mut gen1 = OpportunityLabelGenerator::new(config.clone());
    gen1.add_prices(&prices);
    let labels1 = gen1.generate_labels().unwrap();

    let mut gen2 = OpportunityLabelGenerator::new(config);
    gen2.add_prices(&prices);
    let labels2 = gen2.generate_labels().unwrap();

    assert_eq!(labels1.len(), labels2.len());
    for (l1, l2) in labels1.iter().zip(labels2.iter()) {
        assert_eq!(l1.0, l2.0, "Indices should match");
        assert_eq!(l1.1, l2.1, "Labels should match");
        assert!(
            (l1.2 - l2.2).abs() < 1e-15,
            "Max returns should match exactly"
        );
        assert!(
            (l1.3 - l2.3).abs() < 1e-15,
            "Min returns should match exactly"
        );
    }
}

#[test]
fn test_opportunity_vs_trend_labeling_difference() {
    // Demonstrate the key difference between opportunity and trend labeling
    // Opportunity labeling looks for PEAK moves, not average trends

    use feature_extractor::{LabelConfig, TlobLabelGenerator, TrendLabel};

    // Create a price series where the average change is small but peak is large
    // Prices: 100, 100, 100, 105, 95, 100, 100, 100, 100, 100
    // Average future (smoothed) ≈ 100 (small change from 100)
    // But peak in window: max=105 (5% up), min=95 (5% down)
    let prices = vec![
        100.0, 100.0, 100.0, 105.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        100.0, 100.0,
    ];

    // TLOB labeling (trend-based, smoothed average)
    let tlob_config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.02, // 2% threshold
    };
    let mut tlob_gen = TlobLabelGenerator::new(tlob_config);
    tlob_gen.add_prices(&prices);
    let tlob_labels = tlob_gen.generate_labels().unwrap();

    // Opportunity labeling (peak-based)
    let opp_config = OpportunityConfig::new(5, 0.02);
    let mut opp_gen = OpportunityLabelGenerator::new(opp_config);
    opp_gen.add_prices(&prices);
    let opp_labels = opp_gen.generate_labels().unwrap();

    // The key assertion: opportunity labeling should detect the spike
    // while TLOB may not (due to smoothing averaging it out)
    let tlob_up = tlob_labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();
    let opp_up = opp_labels
        .iter()
        .filter(|(_, l, _, _)| *l == OpportunityLabel::BigUp)
        .count();

    // Opportunity labeling should detect more "up" signals in this scenario
    // because it sees the peak rather than the smoothed average
    println!(
        "TLOB Up count: {}, Opportunity BigUp count: {}",
        tlob_up, opp_up
    );

    // At minimum, opportunity should detect the spike that TLOB might miss
    // (This is the whole point of the different labeling strategy)
    assert!(
        opp_up > 0,
        "Opportunity labeling should detect the upward spike"
    );
}

#[test]
fn test_edge_case_exact_threshold() {
    // Test behavior at exact threshold boundary
    let config = OpportunityConfig::new(3, 0.03); // 3% threshold
    let mut generator = OpportunityLabelGenerator::new(config);

    // Create return exactly at 3%: (103 - 100) / 100 = 0.03
    generator.add_prices(&[100.0, 101.0, 102.0, 103.0]);

    let labels = generator.generate_labels().unwrap();
    let (_, label, max_return, _) = &labels[0];

    // max_return = 3% exactly, should NOT trigger (strict >)
    assert!(
        (*max_return - 0.03).abs() < contract::FLOAT_CMP_EPS,
        "max_return should be exactly 3%"
    );
    assert_eq!(
        *label,
        OpportunityLabel::NoOpportunity,
        "Exactly at threshold should be NoOpportunity (strict > comparison)"
    );
}

#[test]
fn test_edge_case_minimum_prices() {
    // Test with exactly the minimum required number of prices
    let config = OpportunityConfig::new(5, 0.01);
    let min_required = config.min_prices_required(); // Should be 6

    let mut generator = OpportunityLabelGenerator::new(config);
    let prices: Vec<f64> = (0..min_required).map(|i| 100.0 + i as f64).collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    // Should generate exactly 1 label (for t=0)
    assert_eq!(
        labels.len(),
        1,
        "Should generate exactly one label with minimum prices"
    );
}

#[test]
fn test_numerical_stability() {
    // Test with very small and very large prices
    let config = OpportunityConfig::new(5, 0.01);

    // Small prices
    let small_prices: Vec<f64> = (0..20).map(|i| 0.001 + i as f64 * 0.0001).collect();
    let mut gen = OpportunityLabelGenerator::new(config.clone());
    gen.add_prices(&small_prices);
    let labels = gen.generate_labels().unwrap();
    for (_, _, max_ret, min_ret) in &labels {
        assert!(!max_ret.is_nan(), "NaN in max_return with small prices");
        assert!(!min_ret.is_nan(), "NaN in min_return with small prices");
        assert!(max_ret.is_finite(), "Infinite max_return with small prices");
        assert!(min_ret.is_finite(), "Infinite min_return with small prices");
    }

    // Large prices
    let large_prices: Vec<f64> = (0..20).map(|i| 100000.0 + i as f64 * 100.0).collect();
    let mut gen = OpportunityLabelGenerator::new(config);
    gen.add_prices(&large_prices);
    let labels = gen.generate_labels().unwrap();
    for (_, _, max_ret, min_ret) in &labels {
        assert!(!max_ret.is_nan(), "NaN in max_return with large prices");
        assert!(!min_ret.is_nan(), "NaN in min_return with large prices");
        assert!(max_ret.is_finite(), "Infinite max_return with large prices");
        assert!(min_ret.is_finite(), "Infinite min_return with large prices");
    }
}
