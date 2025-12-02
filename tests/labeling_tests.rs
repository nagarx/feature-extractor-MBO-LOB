//! Labeling Integration Tests
//!
//! These tests validate the labeling module's integration with the
//! feature extraction pipeline and verify correctness across different
//! configurations.

use feature_extractor::{
    DeepLobLabelGenerator, DeepLobMethod, LabelConfig, LabelStats, TlobLabelGenerator, TrendLabel,
};

// ============================================================================
// TrendLabel Tests
// ============================================================================

#[test]
fn test_trend_label_roundtrip() {
    // Test int roundtrip
    for original in [TrendLabel::Down, TrendLabel::Stable, TrendLabel::Up] {
        let as_int = original.as_int();
        let recovered = TrendLabel::from_int(as_int).unwrap();
        assert_eq!(original, recovered);
    }

    // Test class index roundtrip
    for original in [TrendLabel::Down, TrendLabel::Stable, TrendLabel::Up] {
        let as_idx = original.as_class_index();
        let recovered = TrendLabel::from_class_index(as_idx).unwrap();
        assert_eq!(original, recovered);
    }
}

#[test]
fn test_trend_label_ordering() {
    // Class indices should be 0, 1, 2
    assert_eq!(TrendLabel::Down.as_class_index(), 0);
    assert_eq!(TrendLabel::Stable.as_class_index(), 1);
    assert_eq!(TrendLabel::Up.as_class_index(), 2);

    // Int values should be -1, 0, 1
    assert_eq!(TrendLabel::Down.as_int(), -1);
    assert_eq!(TrendLabel::Stable.as_int(), 0);
    assert_eq!(TrendLabel::Up.as_int(), 1);
}

// ============================================================================
// LabelConfig Tests
// ============================================================================

#[test]
fn test_label_config_presets() {
    // Test all presets create valid configurations
    let presets = [
        LabelConfig::default(),
        LabelConfig::hft(),
        LabelConfig::short_term(),
        LabelConfig::medium_term(),
        LabelConfig::fi2010(10),
        LabelConfig::fi2010(50),
        LabelConfig::fi2010(100),
    ];

    for config in presets {
        assert!(config.validate().is_ok());
        assert!(config.horizon > 0);
        assert!(config.smoothing_window > 0);
        assert!(config.threshold > 0.0);
        assert!(config.threshold < 1.0);
    }
}

#[test]
fn test_label_config_fi2010_horizon_equals_smoothing() {
    // FI-2010 method uses k = h
    for horizon in [10, 20, 50, 100] {
        let config = LabelConfig::fi2010(horizon);
        assert_eq!(config.horizon, config.smoothing_window);
    }
}

// ============================================================================
// TlobLabelGenerator Tests
// ============================================================================

#[test]
fn test_tlob_upward_trend() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.005,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Strong upward trend
    let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    assert!(!labels.is_empty());

    // Count labels
    let up_count = labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();
    let total = labels.len();

    // Should have majority Up labels
    assert!(
        up_count as f64 / total as f64 > 0.5,
        "Expected majority Up labels for upward trend"
    );
}

#[test]
fn test_tlob_downward_trend() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.005,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Strong downward trend
    let prices: Vec<f64> = (0..30).map(|i| 115.0 - i as f64 * 0.5).collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    assert!(!labels.is_empty());

    // Count labels
    let down_count = labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Down)
        .count();
    let total = labels.len();

    // Should have majority Down labels
    assert!(
        down_count as f64 / total as f64 > 0.5,
        "Expected majority Down labels for downward trend"
    );
}

#[test]
fn test_tlob_stable_prices() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.01, // 1% threshold
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Stable prices with tiny fluctuations (< 1%)
    let prices: Vec<f64> = (0..30)
        .map(|i| 100.0 + (i as f64 * 0.001).sin() * 0.1)
        .collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    assert!(!labels.is_empty());

    // All labels should be Stable
    for (_, label, _) in &labels {
        assert_eq!(*label, TrendLabel::Stable);
    }
}

#[test]
fn test_tlob_percentage_change_correctness() {
    let config = LabelConfig {
        horizon: 2,
        smoothing_window: 1,
        threshold: 0.01,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Known prices for manual calculation
    // p = [100, 101, 102, 103, 104, 105]
    generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

    let labels = generator.generate_labels().unwrap();

    // For t=1, k=1, h=2:
    // past_smooth = avg(p[0], p[1]) = avg(100, 101) = 100.5
    // future_smooth = avg(p[2], p[3]) = avg(102, 103) = 102.5
    // pct_change = (102.5 - 100.5) / 100.5 = 0.0199...

    let (_, _, pct_change) = &labels[0];
    let expected = (102.5 - 100.5) / 100.5;
    assert!(
        (pct_change - expected).abs() < 1e-10,
        "Expected {}, got {}",
        expected,
        pct_change
    );
}

// ============================================================================
// DeepLobLabelGenerator Tests
// ============================================================================

#[test]
fn test_deeplob_method1_upward_trend() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 5,
        threshold: 0.005,
    };

    let mut generator = DeepLobLabelGenerator::new(config);

    // Strong upward trend
    let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    assert!(!labels.is_empty());

    // Should detect upward trend
    let up_count = labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();
    assert!(up_count > 0, "DeepLOB Method 1 should detect upward trend");
}

#[test]
fn test_deeplob_method2_upward_trend() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 5,
        threshold: 0.005,
    };

    let mut generator = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);

    // Strong upward trend
    let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    assert!(!labels.is_empty());

    // Should detect upward trend
    let up_count = labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();
    assert!(up_count > 0, "DeepLOB Method 2 should detect upward trend");
}

#[test]
fn test_deeplob_vs_tlob_similar_results() {
    // When using similar parameters, both should detect the same trend direction
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 5,
        threshold: 0.005,
    };

    let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();

    // TLOB
    let mut tlob = TlobLabelGenerator::new(config.clone());
    tlob.add_prices(&prices);
    let tlob_labels = tlob.generate_labels().unwrap();

    // DeepLOB Method 2 (most similar to TLOB)
    let mut deeplob = DeepLobLabelGenerator::with_method(config, DeepLobMethod::VsPastAverage);
    deeplob.add_prices(&prices);
    let deeplob_labels = deeplob.generate_labels().unwrap();

    // Both should detect upward trend
    let tlob_up = tlob_labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();
    let deeplob_up = deeplob_labels
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();

    assert!(tlob_up > 0, "TLOB should detect upward trend");
    assert!(deeplob_up > 0, "DeepLOB should detect upward trend");
}

// ============================================================================
// LabelStats Tests
// ============================================================================

#[test]
fn test_label_stats_computation() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.005,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Mixed trend
    let prices: Vec<f64> = (0..50)
        .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0)
        .collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();
    let stats = generator.compute_stats(&labels);

    // Verify counts sum to total
    assert_eq!(
        stats.up_count + stats.down_count + stats.stable_count,
        stats.total
    );

    // Verify class balance sums to ~1.0
    let (up_pct, stable_pct, down_pct) = stats.class_balance();
    let sum = up_pct + stable_pct + down_pct;
    assert!((sum - 1.0).abs() < 1e-10, "Class balance should sum to 1.0");

    // Verify change statistics are finite
    assert!(stats.avg_change.is_finite());
    assert!(stats.std_change.is_finite());
    assert!(stats.min_change.is_finite());
    assert!(stats.max_change.is_finite());
}

#[test]
fn test_label_stats_imbalance_detection() {
    // Create imbalanced labels manually
    let stats = LabelStats {
        total: 100,
        up_count: 85,
        down_count: 5, // < 10% minority
        stable_count: 10,
        avg_change: 0.01,
        std_change: 0.005,
        min_change: -0.02,
        max_change: 0.05,
    };

    // Should detect imbalance
    assert!(!stats.is_balanced());
    assert!(stats.has_minority_class());
    assert_eq!(stats.majority_class(), TrendLabel::Up);

    // Imbalance ratio should be 17.0 (85/5)
    assert!((stats.imbalance_ratio() - 17.0).abs() < 0.01);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_insufficient_data_error() {
    let config = LabelConfig {
        horizon: 100,
        smoothing_window: 50,
        threshold: 0.002,
    };

    let mut generator = TlobLabelGenerator::new(config);
    generator.add_prices(&[100.0, 101.0, 102.0]); // Not enough

    let result = generator.generate_labels();
    assert!(result.is_err());
}

#[test]
fn test_empty_generator() {
    let config = LabelConfig::default();
    let generator = TlobLabelGenerator::new(config);

    assert!(generator.is_empty());
    assert_eq!(generator.len(), 0);
    assert!(!generator.can_generate());
}

#[test]
fn test_clear_and_reuse() {
    let config = LabelConfig {
        horizon: 2,
        smoothing_window: 1,
        threshold: 0.01,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // First use
    generator.add_prices(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
    let labels1 = generator.generate_labels().unwrap();
    assert!(!labels1.is_empty());

    // Clear and reuse with different data
    generator.clear();
    assert!(generator.is_empty());

    generator.add_prices(&[200.0, 199.0, 198.0, 197.0, 196.0, 195.0]);
    let labels2 = generator.generate_labels().unwrap();
    assert!(!labels2.is_empty());

    // Should be different labels (different trend direction)
    let up1 = labels1
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Up)
        .count();
    let down2 = labels2
        .iter()
        .filter(|(_, l, _)| *l == TrendLabel::Down)
        .count();

    // First had upward trend, second has downward
    assert!(up1 > 0);
    assert!(down2 > 0);
}

#[test]
fn test_no_nan_or_inf_in_output() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.005,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Various price patterns
    let prices: Vec<f64> = (0..100)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
        .collect();
    generator.add_prices(&prices);

    let labels = generator.generate_labels().unwrap();

    for (_, _, change) in &labels {
        assert!(!change.is_nan(), "Found NaN in percentage change");
        assert!(!change.is_infinite(), "Found Inf in percentage change");
    }
}

// ============================================================================
// Determinism Tests
// ============================================================================

#[test]
fn test_deterministic_output() {
    let config = LabelConfig {
        horizon: 5,
        smoothing_window: 2,
        threshold: 0.005,
    };

    let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.3).collect();

    // Run twice
    let mut gen1 = TlobLabelGenerator::new(config.clone());
    gen1.add_prices(&prices);
    let labels1 = gen1.generate_labels().unwrap();

    let mut gen2 = TlobLabelGenerator::new(config);
    gen2.add_prices(&prices);
    let labels2 = gen2.generate_labels().unwrap();

    // Should be identical
    assert_eq!(labels1.len(), labels2.len());
    for (l1, l2) in labels1.iter().zip(labels2.iter()) {
        assert_eq!(l1.0, l2.0); // Same index
        assert_eq!(l1.1, l2.1); // Same label
        assert!((l1.2 - l2.2).abs() < 1e-15); // Same change
    }
}
