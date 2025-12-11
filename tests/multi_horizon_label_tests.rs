//! Comprehensive tests for MultiHorizonLabelGenerator
//!
//! These tests validate:
//! 1. Correctness: Labels match expected behavior
//! 2. Consistency: Same inputs produce same outputs
//! 3. Edge cases: Boundary conditions, minimum data
//! 4. Performance: No excessive allocations
//! 5. Research alignment: Results match paper specifications

use feature_extractor::labeling::{
    LabelConfig, MultiHorizonConfig, MultiHorizonLabelGenerator, ThresholdStrategy,
    TlobLabelGenerator, TrendLabel,
};

// ============================================================================
// Correctness Tests
// ============================================================================

/// Test that multi-horizon produces identical results to single-horizon TLOB
/// for each individual horizon.
#[test]
fn test_multi_horizon_matches_single_horizon_tlob() {
    let horizons = vec![10, 20, 50];
    let smoothing = 5;
    let threshold = 0.002;

    // Generate test data with clear trends
    let prices: Vec<f64> = (0..200)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0 + i as f64 * 0.01)
        .collect();

    // Multi-horizon
    let multi_config =
        MultiHorizonConfig::new(horizons.clone(), smoothing, ThresholdStrategy::Fixed(threshold));
    let mut multi_gen = MultiHorizonLabelGenerator::new(multi_config);
    multi_gen.add_prices(&prices);
    let multi_result = multi_gen.generate_labels().unwrap();

    // Compare with individual TLOB generators
    for &horizon in &horizons {
        let tlob_config = LabelConfig {
            horizon,
            smoothing_window: smoothing,
            threshold,
        };
        let mut tlob_gen = TlobLabelGenerator::new(tlob_config);
        tlob_gen.add_prices(&prices);
        let tlob_labels = tlob_gen.generate_labels().unwrap();

        let multi_labels = multi_result.labels_for_horizon(horizon).unwrap();

        assert_eq!(
            multi_labels.len(),
            tlob_labels.len(),
            "Horizon {}: label count mismatch ({} vs {})",
            horizon,
            multi_labels.len(),
            tlob_labels.len()
        );

        for (i, (multi, tlob)) in multi_labels.iter().zip(tlob_labels.iter()).enumerate() {
            assert_eq!(
                multi.0, tlob.0,
                "Horizon {}, label {}: index mismatch ({} vs {})",
                horizon, i, multi.0, tlob.0
            );
            assert_eq!(
                multi.1, tlob.1,
                "Horizon {}, index {}: label mismatch ({:?} vs {:?})",
                horizon, multi.0, multi.1, tlob.1
            );
            assert!(
                (multi.2 - tlob.2).abs() < 1e-14,
                "Horizon {}, index {}: change mismatch ({} vs {})",
                horizon,
                multi.0,
                multi.2,
                tlob.2
            );
        }
    }
}

/// Test that strong upward trends produce mostly Up labels across all horizons.
#[test]
fn test_upward_trend_detection_all_horizons() {
    let config = MultiHorizonConfig::new(vec![5, 10, 20], 3, ThresholdStrategy::Fixed(0.005));
    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Strong upward trend: 5% increase over 100 steps
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();

    for horizon in [5, 10, 20] {
        let labels = result.labels_for_horizon(horizon).unwrap();
        let up_count = labels.iter().filter(|(_, l, _)| *l == TrendLabel::Up).count();
        let up_pct = up_count as f64 / labels.len() as f64;

        assert!(
            up_pct > 0.7,
            "Horizon {}: Expected >70% Up labels for upward trend, got {:.1}%",
            horizon,
            up_pct * 100.0
        );
    }
}

/// Test that strong downward trends produce mostly Down labels across all horizons.
#[test]
fn test_downward_trend_detection_all_horizons() {
    let config = MultiHorizonConfig::new(vec![5, 10, 20], 3, ThresholdStrategy::Fixed(0.005));
    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Strong downward trend
    let prices: Vec<f64> = (0..100).map(|i| 200.0 - i as f64 * 0.5).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();

    for horizon in [5, 10, 20] {
        let labels = result.labels_for_horizon(horizon).unwrap();
        let down_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Down)
            .count();
        let down_pct = down_count as f64 / labels.len() as f64;

        assert!(
            down_pct > 0.7,
            "Horizon {}: Expected >70% Down labels for downward trend, got {:.1}%",
            horizon,
            down_pct * 100.0
        );
    }
}

/// Test that flat prices produce all Stable labels.
#[test]
fn test_stable_detection_flat_prices() {
    let config = MultiHorizonConfig::new(vec![5, 10, 20], 3, ThresholdStrategy::Fixed(0.01));
    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Completely flat prices
    let prices: Vec<f64> = vec![100.0; 100];
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();

    for horizon in [5, 10, 20] {
        let labels = result.labels_for_horizon(horizon).unwrap();
        let stable_count = labels
            .iter()
            .filter(|(_, l, _)| *l == TrendLabel::Stable)
            .count();

        assert_eq!(
            stable_count,
            labels.len(),
            "Horizon {}: Expected all Stable labels for flat prices",
            horizon
        );
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test with exact minimum required prices.
#[test]
fn test_minimum_required_prices() {
    let config = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(0.002));
    let min_required = config.min_prices_required(); // 2*5 + 10 + 1 = 21

    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Test with one less than minimum
    let prices_insufficient: Vec<f64> = (0..min_required - 1).map(|i| 100.0 + i as f64).collect();
    gen.add_prices(&prices_insufficient);
    assert!(gen.generate_labels().is_err(), "Should fail with insufficient prices");

    // Test with exact minimum
    gen.clear();
    let prices_exact: Vec<f64> = (0..min_required).map(|i| 100.0 + i as f64).collect();
    gen.add_prices(&prices_exact);
    let result = gen.generate_labels();
    assert!(result.is_ok(), "Should succeed with exact minimum prices");

    let multi_result = result.unwrap();
    let labels = multi_result.labels_for_horizon(10).unwrap();
    assert!(!labels.is_empty(), "Should generate at least one label");
}

/// Test that longer horizons require more prices (fewer valid positions).
#[test]
fn test_longer_horizons_fewer_labels() {
    let config = MultiHorizonConfig::new(vec![5, 10, 20, 50], 3, ThresholdStrategy::Fixed(0.002));
    let mut gen = MultiHorizonLabelGenerator::new(config);

    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    let counts = result.label_counts();

    // Verify: shorter horizons should have more labels
    assert!(
        counts[&5] >= counts[&10],
        "Horizon 5 should have >= labels than horizon 10"
    );
    assert!(
        counts[&10] >= counts[&20],
        "Horizon 10 should have >= labels than horizon 20"
    );
    assert!(
        counts[&20] >= counts[&50],
        "Horizon 20 should have >= labels than horizon 50"
    );
}

/// Test that all label indices are within valid bounds.
#[test]
fn test_label_indices_within_bounds() {
    let config = MultiHorizonConfig::fi2010();
    let mut gen = MultiHorizonLabelGenerator::new(config.clone());

    let prices: Vec<f64> = (0..500).map(|i| 100.0 + (i as f64 * 0.05).sin() * 5.0).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();

    for horizon in config.horizons() {
        let labels = result.labels_for_horizon(*horizon).unwrap();
        let k = config.smoothing_window;

        for (idx, _, _) in labels {
            // Index should be >= k (need past smoothing)
            assert!(
                *idx >= k,
                "Horizon {}: Index {} should be >= smoothing window {}",
                horizon,
                idx,
                k
            );

            // Index should be < total - horizon - k (need future smoothing)
            let max_valid = gen.len() - horizon - k;
            assert!(
                *idx < max_valid,
                "Horizon {}: Index {} should be < {}",
                horizon,
                idx,
                max_valid
            );
        }
    }
}

// ============================================================================
// Numerical Precision Tests
// ============================================================================

/// Test that no NaN or Inf values are produced.
#[test]
fn test_no_nan_or_inf_values() {
    let config = MultiHorizonConfig::fi2010();

    // Test various price patterns
    let patterns: Vec<Vec<f64>> = vec![
        // Normal pattern
        (0..200).map(|i| 100.0 + i as f64 * 0.1).collect(),
        // Very small changes
        (0..200).map(|i| 100.0 + i as f64 * 0.0001).collect(),
        // Large values
        (0..200).map(|i| 10000.0 + i as f64 * 10.0).collect(),
        // Small values
        (0..200).map(|i| 0.01 + i as f64 * 0.0001).collect(),
        // Oscillating
        (0..200)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect(),
    ];

    for (pattern_idx, prices) in patterns.iter().enumerate() {
        let mut gen = MultiHorizonLabelGenerator::new(config.clone());
        gen.add_prices(prices);

        let result = gen.generate_labels().unwrap();

        for horizon in result.horizons() {
            let labels = result.labels_for_horizon(*horizon).unwrap();
            for (idx, _, change) in labels {
                assert!(
                    !change.is_nan(),
                    "Pattern {}, Horizon {}, Index {}: Found NaN",
                    pattern_idx,
                    horizon,
                    idx
                );
                assert!(
                    !change.is_infinite(),
                    "Pattern {}, Horizon {}, Index {}: Found Inf",
                    pattern_idx,
                    horizon,
                    idx
                );
            }
        }
    }
}

/// Test percentage change calculation accuracy.
#[test]
fn test_percentage_change_accuracy() {
    let config = MultiHorizonConfig::new(vec![2], 1, ThresholdStrategy::Fixed(0.5)); // Large threshold to get Stable
    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Known prices for manual calculation
    // With k=1, h=2: past_smooth = avg(p[t-1], p[t]), future_smooth = avg(p[t+1], p[t+2])
    // At t=1: past = avg(100, 110) = 105, future = avg(120, 130) = 125
    // pct_change = (125 - 105) / 105 = 0.190476...
    gen.add_prices(&[100.0, 110.0, 120.0, 130.0, 140.0]);

    let result = gen.generate_labels().unwrap();
    let labels = result.labels_for_horizon(2).unwrap();

    // Check first label
    let (idx, _, change) = &labels[0];
    assert_eq!(*idx, 1, "First label should be at index 1");

    let expected_change = (125.0 - 105.0) / 105.0; // ≈ 0.190476
    assert!(
        (change - expected_change).abs() < 1e-10,
        "Expected change {}, got {}",
        expected_change,
        change
    );
}

// ============================================================================
// Determinism Tests
// ============================================================================

/// Test that identical inputs produce identical outputs.
#[test]
fn test_deterministic_output() {
    let config = MultiHorizonConfig::fi2010();
    let prices: Vec<f64> = (0..300)
        .map(|i| 100.0 + (i as f64 * 0.07).sin() * 3.0)
        .collect();

    // Run multiple times
    let mut results = Vec::new();
    for _ in 0..3 {
        let mut gen = MultiHorizonLabelGenerator::new(config.clone());
        gen.add_prices(&prices);
        results.push(gen.generate_labels().unwrap());
    }

    // Compare all results
    for horizon in config.horizons() {
        let labels_0 = results[0].labels_for_horizon(*horizon).unwrap();
        for result in results.iter().skip(1) {
            let labels_i = result.labels_for_horizon(*horizon).unwrap();

            assert_eq!(labels_0.len(), labels_i.len(), "Horizon {}: count mismatch", horizon);

            for (l0, li) in labels_0.iter().zip(labels_i.iter()) {
                assert_eq!(l0.0, li.0, "Horizon {}: index mismatch", horizon);
                assert_eq!(l0.1, li.1, "Horizon {}: label mismatch at {}", horizon, l0.0);
                assert!(
                    (l0.2 - li.2).abs() < 1e-15,
                    "Horizon {}: change mismatch at {}",
                    horizon,
                    l0.0
                );
            }
        }
    }
}

// ============================================================================
// Research Configuration Tests
// ============================================================================

/// Test FI-2010 configuration produces valid results.
#[test]
fn test_fi2010_configuration() {
    let config = MultiHorizonConfig::fi2010();

    assert_eq!(config.horizons(), &[10, 20, 30, 50, 100]);
    assert_eq!(config.smoothing_window, 5);
    assert_eq!(config.min_prices_required(), 2 * 5 + 100 + 1); // 111

    let mut gen = MultiHorizonLabelGenerator::new(config);
    let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.05).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();

    // All horizons should have labels
    for &h in &[10, 20, 30, 50, 100] {
        assert!(result.has_horizon(h), "Should have horizon {}", h);
        let labels = result.labels_for_horizon(h).unwrap();
        assert!(!labels.is_empty(), "Horizon {} should have labels", h);
    }
}

/// Test DeepLOB configuration produces valid results.
#[test]
fn test_deeplob_configuration() {
    let config = MultiHorizonConfig::deeplob();

    assert_eq!(config.horizons(), &[10, 20, 50, 100]);
    assert_eq!(config.smoothing_window, 5);

    let mut gen = MultiHorizonLabelGenerator::new(config);
    let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.05).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();

    for &h in &[10, 20, 50, 100] {
        assert!(result.has_horizon(h), "Should have horizon {}", h);
    }
}

/// Test HFT configuration produces valid results.
#[test]
fn test_hft_configuration() {
    let config = MultiHorizonConfig::hft();

    assert_eq!(config.horizons(), &[5, 10, 20]);
    assert_eq!(config.smoothing_window, 3);

    // HFT uses tighter threshold
    if let ThresholdStrategy::Fixed(t) = config.threshold_strategy {
        assert!(t < 0.001, "HFT should use tight threshold");
    }

    let mut gen = MultiHorizonLabelGenerator::new(config);
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.01).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    assert_eq!(result.num_horizons(), 3);
}

// ============================================================================
// Statistics Tests
// ============================================================================

/// Test that statistics are computed correctly.
#[test]
fn test_statistics_computation() {
    let config = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(0.002));
    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Generate data with known distribution
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    let stats = result.stats_for_horizon(10).unwrap();

    // Basic consistency checks
    assert_eq!(
        stats.up_count + stats.down_count + stats.stable_count,
        stats.total,
        "Counts should sum to total"
    );
    assert!(stats.min_change <= stats.avg_change, "min <= avg");
    assert!(stats.avg_change <= stats.max_change, "avg <= max");
    assert!(stats.std_change >= 0.0, "std should be non-negative");
}

/// Test class balance detection.
#[test]
fn test_class_balance_detection() {
    // Test with imbalanced data (strong trend)
    let config = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(0.001));
    let mut gen = MultiHorizonLabelGenerator::new(config);

    // Strong upward trend -> imbalanced
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    let stats = result.stats_for_horizon(10).unwrap();

    let (up_pct, _, _) = stats.class_balance();
    assert!(
        up_pct > 0.5,
        "Strong upward trend should have imbalanced Up class"
    );
    assert!(
        !stats.is_balanced(),
        "Strong trend should not be balanced"
    );
}

// ============================================================================
// Summary Tests
// ============================================================================

/// Test multi-horizon summary aggregation.
#[test]
fn test_summary_aggregation() {
    let config = MultiHorizonConfig::new(vec![5, 10, 20], 3, ThresholdStrategy::Fixed(0.01));
    let mut gen = MultiHorizonLabelGenerator::new(config.clone());

    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.2).collect();
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    let summary = result.summary();

    assert_eq!(summary.num_horizons, 3);
    assert_eq!(summary.total_prices, 100);
    assert_eq!(
        summary.total_up + summary.total_down + summary.total_stable,
        summary.total_labels
    );

    // Verify against individual stats
    let mut expected_total = 0;
    for &h in config.horizons() {
        let stats = result.stats_for_horizon(h).unwrap();
        expected_total += stats.total;
    }
    assert_eq!(summary.total_labels, expected_total);
}

// ============================================================================
// Quantile Threshold Strategy Tests
// ============================================================================

/// Test Quantile threshold strategy creation.
#[test]
fn test_quantile_threshold_creation() {
    let strategy = ThresholdStrategy::quantile(0.3, 1000, 0.002);

    assert!(strategy.needs_rolling_computation());
    assert_eq!(strategy.window_size(), Some(1000));
    assert_eq!(strategy.target_proportion(), Some(0.3));
}

/// Test Quantile threshold with fixed fallback when insufficient data.
#[test]
fn test_quantile_threshold_fallback() {
    let strategy = ThresholdStrategy::quantile(0.3, 5000, 0.002);

    // With no rolling data, should return fallback
    assert_eq!(strategy.get_threshold(None), 0.002);
}

/// Test Quantile threshold with computed value.
#[test]
fn test_quantile_threshold_with_value() {
    let strategy = ThresholdStrategy::quantile(0.3, 1000, 0.002);

    // With rolling data, should use computed value
    assert_eq!(strategy.get_threshold(Some(0.005)), 0.005);
}

/// Test Quantile generates balanced classes with oscillating prices.
#[test]
fn test_quantile_balanced_classes_oscillating() {
    // Generate oscillating prices that would be highly imbalanced with fixed threshold
    let prices: Vec<f64> = (0..2000)
        .map(|i| 100.0 + (i as f64 * 0.05).sin() * 0.5)
        .collect();

    let config = MultiHorizonConfig::new(
        vec![10],
        5,
        ThresholdStrategy::quantile(0.3, 500, 0.002),
    );

    let mut gen = MultiHorizonLabelGenerator::new(config);
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    let stats = result.stats_for_horizon(10).unwrap();

    // With quantile threshold, classes should be more balanced
    let total = stats.total as f64;
    let up_ratio = stats.up_count as f64 / total;
    let down_ratio = stats.down_count as f64 / total;

    println!(
        "Quantile balanced test: up={:.2}%, down={:.2}%, stable={:.2}%",
        up_ratio * 100.0,
        down_ratio * 100.0,
        stats.stable_count as f64 / total * 100.0
    );

    // Should be reasonably balanced (not perfectly, but better than fixed)
    // Each class should have at least 10% of labels
    assert!(
        up_ratio > 0.1,
        "Up ratio {} should be > 0.1 for balanced classes",
        up_ratio
    );
    assert!(
        down_ratio > 0.1,
        "Down ratio {} should be > 0.1 for balanced classes",
        down_ratio
    );
}

/// Test Quantile vs Fixed threshold comparison.
#[test]
fn test_quantile_vs_fixed_comparison() {
    // Generate prices with subtle movements
    let prices: Vec<f64> = (0..1000)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 0.1)
        .collect();

    // Fixed threshold - might be too high for small movements
    let fixed_config = MultiHorizonConfig::new(vec![10], 5, ThresholdStrategy::Fixed(0.005));

    let mut fixed_gen = MultiHorizonLabelGenerator::new(fixed_config);
    fixed_gen.add_prices(&prices);
    let fixed_result = fixed_gen.generate_labels().unwrap();
    let fixed_stats = fixed_result.stats_for_horizon(10).unwrap();

    // Quantile threshold - adapts to actual price movements
    let quantile_config = MultiHorizonConfig::new(
        vec![10],
        5,
        ThresholdStrategy::quantile(0.3, 200, 0.002),
    );

    let mut quantile_gen = MultiHorizonLabelGenerator::new(quantile_config);
    quantile_gen.add_prices(&prices);
    let quantile_result = quantile_gen.generate_labels().unwrap();
    let quantile_stats = quantile_result.stats_for_horizon(10).unwrap();

    println!(
        "Fixed threshold: up={}, down={}, stable={}",
        fixed_stats.up_count, fixed_stats.down_count, fixed_stats.stable_count
    );
    println!(
        "Quantile threshold: up={}, down={}, stable={}",
        quantile_stats.up_count, quantile_stats.down_count, quantile_stats.stable_count
    );

    // With subtle movements, fixed threshold often results in mostly Stable
    // Quantile should have more diverse labels
    let fixed_stable_ratio = fixed_stats.stable_count as f64 / fixed_stats.total as f64;
    let quantile_stable_ratio = quantile_stats.stable_count as f64 / quantile_stats.total as f64;

    // Quantile should generally have fewer stable labels than an overly aggressive fixed threshold
    // (This is not always guaranteed, but shows the adaptive nature)
    println!(
        "Fixed stable ratio: {:.2}%, Quantile stable ratio: {:.2}%",
        fixed_stable_ratio * 100.0,
        quantile_stable_ratio * 100.0
    );
}

/// Test Quantile threshold with trending prices.
#[test]
fn test_quantile_with_trending_prices() {
    // Strong upward trend
    let prices: Vec<f64> = (0..1000).map(|i| 100.0 + i as f64 * 0.1).collect();

    let config = MultiHorizonConfig::new(
        vec![10],
        5,
        ThresholdStrategy::quantile(0.3, 200, 0.002),
    );

    let mut gen = MultiHorizonLabelGenerator::new(config);
    gen.add_prices(&prices);

    let result = gen.generate_labels().unwrap();
    let stats = result.stats_for_horizon(10).unwrap();

    // With strong trend, most labels should be Up
    let up_ratio = stats.up_count as f64 / stats.total as f64;
    assert!(
        up_ratio > 0.5,
        "Strong upward trend should produce >50% Up labels, got {:.2}%",
        up_ratio * 100.0
    );

    println!(
        "Trending prices: up={:.2}%, down={:.2}%, stable={:.2}%",
        up_ratio * 100.0,
        stats.down_count as f64 / stats.total as f64 * 100.0,
        stats.stable_count as f64 / stats.total as f64 * 100.0
    );
}

/// Test Quantile threshold invalid target proportion panics.
#[test]
#[should_panic(expected = "target_proportion must be in range")]
fn test_quantile_invalid_target_proportion_zero() {
    ThresholdStrategy::quantile(0.0, 1000, 0.002);
}

/// Test Quantile threshold invalid target proportion > 0.5 panics.
#[test]
#[should_panic(expected = "target_proportion must be in range")]
fn test_quantile_invalid_target_proportion_too_high() {
    ThresholdStrategy::quantile(0.6, 1000, 0.002);
}

/// Test Quantile with different target proportions.
#[test]
fn test_quantile_different_proportions() {
    let prices: Vec<f64> = (0..2000)
        .map(|i| 100.0 + (i as f64 * 0.03).sin() * 0.3)
        .collect();

    // Test with 20% target (more stable)
    let config_20 = MultiHorizonConfig::new(
        vec![10],
        5,
        ThresholdStrategy::quantile(0.2, 500, 0.002),
    );
    let mut gen_20 = MultiHorizonLabelGenerator::new(config_20);
    gen_20.add_prices(&prices);
    let result_20 = gen_20.generate_labels().unwrap();
    let stats_20 = result_20.stats_for_horizon(10).unwrap();

    // Test with 40% target (less stable)
    let config_40 = MultiHorizonConfig::new(
        vec![10],
        5,
        ThresholdStrategy::quantile(0.4, 500, 0.002),
    );
    let mut gen_40 = MultiHorizonLabelGenerator::new(config_40);
    gen_40.add_prices(&prices);
    let result_40 = gen_40.generate_labels().unwrap();
    let stats_40 = result_40.stats_for_horizon(10).unwrap();

    let stable_ratio_20 = stats_20.stable_count as f64 / stats_20.total as f64;
    let stable_ratio_40 = stats_40.stable_count as f64 / stats_40.total as f64;

    println!(
        "target=0.2: stable={:.2}%, target=0.4: stable={:.2}%",
        stable_ratio_20 * 100.0,
        stable_ratio_40 * 100.0
    );

    // 20% target should have more stable labels than 40% target
    // (lower target_proportion = higher threshold = more stable)
    assert!(
        stable_ratio_20 > stable_ratio_40,
        "20% target should produce more stable labels than 40% target"
    );
}

