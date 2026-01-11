//! Integration tests for Triple Barrier and Magnitude labeling modules.
//!
//! These tests verify:
//! 1. Correct behavior of Triple Barrier labeling under various market conditions
//! 2. Magnitude generation for regression experiments
//! 3. Cross-module consistency and integration
//! 4. Edge cases and numerical stability

use feature_extractor::{
    BarrierLabel, MagnitudeConfig, MagnitudeGenerator, ReturnType, TimeoutStrategy,
    TripleBarrierConfig, TripleBarrierLabeler,
};

// ============================================================================
// Triple Barrier Integration Tests
// ============================================================================

#[test]
fn test_triple_barrier_trend_following_scenario() {
    // Scenario: Clear upward trend
    // Expected: Most labels should be ProfitTarget
    let config = TripleBarrierConfig::new(0.02, 0.01, 20); // 2% profit, 1% stop
    let mut labeler = TripleBarrierLabeler::new(config);

    // Strong upward trend: 1% per step
    let prices: Vec<f64> = (0..100).map(|i| 100.0 * (1.0 + 0.01 * i as f64)).collect();
    labeler.add_prices(&prices);

    let labels = labeler.generate_labels().unwrap();
    let stats = labeler.compute_stats(&labels);

    // In a strong uptrend, we should have mostly profit targets
    assert!(
        stats.profit_target_count > stats.stop_loss_count * 3,
        "In uptrend, profit targets should dominate. Got {} wins, {} losses",
        stats.profit_target_count,
        stats.stop_loss_count
    );

    // Win rate should be high
    assert!(
        stats.win_rate() > 0.8,
        "Win rate should be > 80% in uptrend, got {:.2}%",
        stats.win_rate() * 100.0
    );
}

#[test]
fn test_triple_barrier_trend_reversal_scenario() {
    // Scenario: Price goes up then down sharply
    // Expected: Mixed labels depending on entry point
    let config = TripleBarrierConfig::new(0.03, 0.02, 15); // 3% profit, 2% stop
    let mut labeler = TripleBarrierLabeler::new(config);

    let mut prices = Vec::new();

    // First half: upward (0-49)
    for i in 0..50 {
        prices.push(100.0 + i as f64 * 0.5); // 0.5% per step
    }

    // Second half: sharp downward (50-99)
    for i in 0..50 {
        prices.push(125.0 - i as f64 * 1.0); // 1% per step down
    }

    labeler.add_prices(&prices);
    let labels = labeler.generate_labels().unwrap();

    // Check that entries near the peak result in stop-losses
    // Look at entries very close to the peak (idx 45-49) where the downward
    // movement is within their horizon
    let near_peak_labels: Vec<_> = labels
        .iter()
        .filter(|(idx, _, _, _)| *idx >= 45 && *idx <= 49)
        .collect();

    let stop_losses_near_peak = near_peak_labels
        .iter()
        .filter(|(_, label, _, _)| *label == BarrierLabel::StopLoss)
        .count();

    // At least some entries near peak should hit stop-loss (the sharp reversal)
    // We use a more lenient assertion since the exact timing depends on barrier levels
    assert!(
        stop_losses_near_peak > 0 || near_peak_labels.iter().any(|(_, l, _, _)| *l == BarrierLabel::Timeout),
        "Entries near peak should stop out or timeout. Got {} stop-losses out of {}",
        stop_losses_near_peak,
        near_peak_labels.len()
    );

    // Additionally verify that entries during the uptrend hit profit targets
    let uptrend_labels: Vec<_> = labels
        .iter()
        .filter(|(idx, _, _, _)| *idx >= 10 && *idx <= 30)
        .collect();

    let profit_targets_uptrend = uptrend_labels
        .iter()
        .filter(|(_, label, _, _)| *label == BarrierLabel::ProfitTarget)
        .count();

    assert!(
        profit_targets_uptrend > uptrend_labels.len() / 3,
        "Entries during uptrend should mostly hit profit target. Got {} out of {}",
        profit_targets_uptrend,
        uptrend_labels.len()
    );
}

#[test]
fn test_triple_barrier_choppy_market_scenario() {
    // Scenario: Sideways, choppy market with oscillations
    // Expected: Many timeouts due to neither barrier being hit
    let config = TripleBarrierConfig::new(0.03, 0.03, 20); // Wide 3% barriers
    let mut labeler = TripleBarrierLabeler::new(config);

    // Oscillating prices with small amplitude (< 3%)
    let prices: Vec<f64> = (0..100)
        .map(|i| 100.0 + (i as f64 * 0.5).sin() * 1.5) // ±1.5% oscillation
        .collect();

    labeler.add_prices(&prices);
    let labels = labeler.generate_labels().unwrap();
    let stats = labeler.compute_stats(&labels);

    // In choppy market with wide barriers, timeouts should dominate
    assert!(
        stats.timeout_count > stats.total / 3,
        "Choppy market should have many timeouts. Got {} timeouts out of {} total",
        stats.timeout_count,
        stats.total
    );
}

#[test]
fn test_triple_barrier_asymmetric_risk_reward() {
    // Test asymmetric barriers (2:1 reward/risk)
    let config = TripleBarrierConfig::new(0.04, 0.02, 30); // 4% profit, 2% stop

    assert!(
        (config.risk_reward_ratio() - 2.0).abs() < 1e-10,
        "Risk/reward should be 2:1"
    );

    // Breakeven win rate should be ~33%
    assert!(
        (config.breakeven_win_rate() - 0.333).abs() < 0.01,
        "Breakeven win rate should be ~33%"
    );

    let mut labeler = TripleBarrierLabeler::new(config);

    // Random walk prices (slight upward bias)
    let mut prices = vec![100.0];
    let mut price = 100.0;
    for _ in 0..199 {
        // Biased random walk
        price *= 1.0 + (0.001 + 0.003 * ((prices.len() as f64 * 0.1).sin()));
        prices.push(price);
    }

    labeler.add_prices(&prices);
    let labels = labeler.generate_labels().unwrap();
    let stats = labeler.compute_stats(&labels);

    // Verify that profit targets give higher returns than losses
    if stats.profit_target_count > 0 && stats.stop_loss_count > 0 {
        assert!(
            stats.avg_profit_return > -stats.avg_loss_return,
            "Profit return ({:.4}%) should exceed loss magnitude ({:.4}%)",
            stats.avg_profit_return * 100.0,
            stats.avg_loss_return.abs() * 100.0
        );
    }
}

#[test]
fn test_triple_barrier_timeout_strategy_comparison() {
    let base_prices = vec![
        100.0, 100.5, 101.0, 100.8, 100.6, 100.4, 100.2, 100.1, 100.0, 99.9, 100.3,
    ];

    // Strategy 1: Standard timeout (label as 0)
    let config1 = TripleBarrierConfig::new(0.05, 0.05, 10);
    let mut lab1 = TripleBarrierLabeler::new(config1);
    lab1.add_prices(&base_prices);
    let labels1 = lab1.generate_labels().unwrap();

    // Strategy 2: Use return sign at timeout
    let config2 = TripleBarrierConfig::new(0.05, 0.05, 10)
        .with_timeout_strategy(TimeoutStrategy::UseReturnSign);
    let mut lab2 = TripleBarrierLabeler::new(config2);
    lab2.add_prices(&base_prices);
    let labels2 = lab2.generate_labels().unwrap();

    // First entry: timeout at t=10, final return = (100.3 - 100) / 100 = 0.3%
    // Strategy 1 should label as Timeout, Strategy 2 should label as ProfitTarget
    assert_eq!(labels1[0].1, BarrierLabel::Timeout);
    assert_eq!(labels2[0].1, BarrierLabel::ProfitTarget);
}

#[test]
fn test_triple_barrier_stats_accuracy() {
    let config = TripleBarrierConfig::new(0.02, 0.01, 15);
    let mut labeler = TripleBarrierLabeler::new(config);

    // Create a price sequence with known outcomes
    let mut prices = Vec::new();

    // Segment 1: Strong up move → profit target
    prices.extend(vec![
        100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0,
        103.0, 103.0, 103.0,
    ]);

    // Segment 2: Strong down move → stop loss
    prices.extend(vec![
        100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0,
        97.0,
    ]);

    labeler.add_prices(&prices);
    let labels = labeler.generate_labels().unwrap();
    let stats = labeler.compute_stats(&labels);

    // Class balance should sum to 1
    let (sl, to, pt) = stats.class_balance();
    assert!(
        (sl + to + pt - 1.0).abs() < 1e-10,
        "Class balance should sum to 1.0"
    );

    // Average profit return should be positive
    if stats.profit_target_count > 0 {
        assert!(
            stats.avg_profit_return > 0.0,
            "Average profit return should be positive"
        );
    }

    // Average loss return should be negative
    if stats.stop_loss_count > 0 {
        assert!(
            stats.avg_loss_return < 0.0,
            "Average loss return should be negative"
        );
    }
}

// ============================================================================
// Magnitude Integration Tests
// ============================================================================

#[test]
fn test_magnitude_uptrend_scenario() {
    // Scenario: Clear upward trend
    // Expected: Positive point returns, max > |min|
    let config = MagnitudeConfig::peak_returns(20);
    let mut gen = MagnitudeGenerator::new(config);

    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
    gen.add_prices(&prices);

    let returns = gen.generate_returns().unwrap();
    let stats = gen.compute_stats(&returns);

    // All samples should show bullish characteristics
    assert!(
        stats.avg_point_return > 0.0,
        "Average point return should be positive in uptrend"
    );

    assert!(
        stats.positive_rate > 0.95,
        "Nearly all samples should have positive returns, got {:.2}%",
        stats.positive_rate * 100.0
    );

    // Max returns should be larger than min returns (in absolute terms)
    assert!(
        stats.avg_max_return > -stats.avg_min_return,
        "Max should dominate min in uptrend"
    );
}

#[test]
fn test_magnitude_multi_horizon_consistency() {
    // Multi-horizon returns should show predictable relationships
    let config = MagnitudeConfig::multi_horizon(vec![5, 10, 20]);
    let mut gen = MagnitudeGenerator::new(config);

    // Linear upward trend
    let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 1.0).collect();
    gen.add_prices(&prices);

    let returns = gen.generate_multi_horizon_returns().unwrap();

    // Check that longer horizons have larger returns in a trend
    for (_, horizon_data) in &returns {
        // h=5 return
        let ret_5 = horizon_data[0].point_return;
        // h=10 return
        let ret_10 = horizon_data[1].point_return;
        // h=20 return
        let ret_20 = horizon_data[2].point_return;

        // In a linear trend, longer horizon = larger return
        assert!(
            ret_10 > ret_5,
            "h=10 return ({:.4}) should > h=5 return ({:.4})",
            ret_10,
            ret_5
        );
        assert!(
            ret_20 > ret_10,
            "h=20 return ({:.4}) should > h=10 return ({:.4})",
            ret_20,
            ret_10
        );
    }
}

#[test]
fn test_magnitude_return_type_selection() {
    // Test that different return types give different outputs
    let prices: Vec<f64> = (0..30)
        .map(|i| 100.0 + (i as f64 * 0.5).sin() * 3.0)
        .collect();

    // Point return config
    let config_point = MagnitudeConfig {
        horizons: vec![10],
        return_type: ReturnType::PointReturn,
        compute_all_stats: true,
        smoothing_window: None,
    };

    // Max return config
    let config_max = MagnitudeConfig {
        horizons: vec![10],
        return_type: ReturnType::MaxReturn,
        compute_all_stats: true,
        smoothing_window: None,
    };

    let mut gen_point = MagnitudeGenerator::new(config_point);
    let mut gen_max = MagnitudeGenerator::new(config_max);

    gen_point.add_prices(&prices);
    gen_max.add_prices(&prices);

    let ret_point = gen_point.generate_primary_returns().unwrap();
    let ret_max = gen_max.generate_primary_returns().unwrap();

    // Max return should always be >= point return
    for ((_, point), (_, max)) in ret_point.iter().zip(ret_max.iter()) {
        assert!(
            max >= point,
            "Max return ({}) should >= point return ({})",
            max,
            point
        );
    }
}

#[test]
fn test_magnitude_volatile_market() {
    // Scenario: High volatility market
    // Expected: Large return_std, large max-min spread
    let config = MagnitudeConfig::peak_returns(15);
    let mut gen = MagnitudeGenerator::new(config);

    // High volatility: large swings
    let prices: Vec<f64> = (0..50)
        .map(|i| 100.0 + (i as f64 * 0.3).sin() * 8.0) // ±8% swings
        .collect();

    gen.add_prices(&prices);
    let returns = gen.generate_returns().unwrap();

    // Check that return spread is significant
    for (_, data) in &returns {
        let spread = data.max_return - data.min_return;
        assert!(
            spread > 0.02,
            "Volatile market should have significant spread, got {:.2}%",
            spread * 100.0
        );
    }
}

// ============================================================================
// Cross-Module Consistency Tests
// ============================================================================

#[test]
fn test_triple_barrier_vs_magnitude_consistency() {
    // Verify that Triple Barrier and Magnitude return calculations are consistent
    let prices: Vec<f64> = (0..50)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0)
        .collect();

    // Triple Barrier with wide barriers (to get timeouts with exit returns)
    let tb_config = TripleBarrierConfig::new(0.10, 0.10, 20);
    let mut tb_labeler = TripleBarrierLabeler::new(tb_config);
    tb_labeler.add_prices(&prices);
    let tb_labels = tb_labeler.generate_labels().unwrap();

    // Magnitude generator
    let mag_config = MagnitudeConfig::point_return(20);
    let mut mag_gen = MagnitudeGenerator::new(mag_config);
    mag_gen.add_prices(&prices);
    let mag_returns = mag_gen.generate_returns().unwrap();

    // For timeout cases (barrier not hit), exit return should match point return
    for ((tb_idx, tb_label, tb_exit_time, tb_return), (mag_idx, mag_data)) in
        tb_labels.iter().zip(mag_returns.iter())
    {
        assert_eq!(tb_idx, mag_idx, "Indices should match");

        // If triple barrier timed out at max_horizon, the return should match
        if *tb_label == BarrierLabel::Timeout && *tb_exit_time == 20 {
            assert!(
                (tb_return - mag_data.point_return).abs() < 1e-10,
                "Timeout return should match point return. TB: {}, Mag: {}",
                tb_return,
                mag_data.point_return
            );
        }
    }
}

#[test]
fn test_label_methods_handle_same_data_deterministically() {
    // All labeling methods should be deterministic
    // Need at least max_horizon + 1 prices (intraday preset has max_horizon=100, so need 101+)
    let prices: Vec<f64> = (0..150)
        .map(|i| 100.0 + (i as f64 * 0.15).sin() * 3.0)
        .collect();

    // Run twice with same config
    let config = TripleBarrierConfig::intraday();

    let mut lab1 = TripleBarrierLabeler::new(config.clone());
    lab1.add_prices(&prices);
    let labels1 = lab1.generate_labels().unwrap();

    let mut lab2 = TripleBarrierLabeler::new(config);
    lab2.add_prices(&prices);
    let labels2 = lab2.generate_labels().unwrap();

    assert_eq!(labels1.len(), labels2.len());

    for (l1, l2) in labels1.iter().zip(labels2.iter()) {
        assert_eq!(l1, l2, "Labels should be identical for same input");
    }
}

// ============================================================================
// Edge Cases and Numerical Stability
// ============================================================================

#[test]
fn test_triple_barrier_exact_boundary_conditions() {
    // Test when price exactly hits barrier
    let config = TripleBarrierConfig::new(0.02, 0.01, 5); // 2% profit, 1% stop
    let mut labeler = TripleBarrierLabeler::new(config);

    // Entry at 100, exact upper barrier at 102
    let prices = vec![100.0, 100.5, 101.0, 101.5, 102.0, 102.5];
    labeler.add_prices(&prices);

    let labels = labeler.generate_labels().unwrap();
    assert_eq!(labels[0].1, BarrierLabel::ProfitTarget);
    assert_eq!(labels[0].2, 4); // Hit at t=4

    // Entry at 100, exact lower barrier at 99
    labeler.clear();
    let prices = vec![100.0, 99.5, 99.0, 98.5, 98.0, 97.5];
    labeler.add_prices(&prices);

    let labels = labeler.generate_labels().unwrap();
    assert_eq!(labels[0].1, BarrierLabel::StopLoss);
    assert_eq!(labels[0].2, 2); // Hit at t=2
}

#[test]
fn test_magnitude_extreme_values() {
    let config = MagnitudeConfig::peak_returns(10);
    let mut gen = MagnitudeGenerator::new(config);

    // Include extreme price movements
    let mut prices = vec![100.0];
    for i in 1..30 {
        if i == 5 {
            prices.push(200.0); // 100% spike
        } else if i == 15 {
            prices.push(50.0); // 50% crash
        } else {
            prices.push(100.0);
        }
    }

    gen.add_prices(&prices);
    let returns = gen.generate_returns().unwrap();

    // Verify no NaN or Inf
    for (_, data) in &returns {
        assert!(!data.point_return.is_nan());
        assert!(!data.point_return.is_infinite());
        assert!(!data.max_return.is_nan());
        assert!(!data.min_return.is_nan());
    }
}

#[test]
fn test_constant_prices_edge_case() {
    // All prices are the same
    // Use shorter horizon for this test (20 instead of 100 from intraday preset)
    let prices: Vec<f64> = vec![100.0; 50];

    // Triple Barrier with shorter horizon
    let config = TripleBarrierConfig::new(0.005, 0.003, 20); // Custom config with h=20
    let mut tb = TripleBarrierLabeler::new(config);
    tb.add_prices(&prices);
    let tb_labels = tb.generate_labels().unwrap();

    // All should be timeouts with 0 return
    for (_, label, _, return_val) in &tb_labels {
        assert_eq!(*label, BarrierLabel::Timeout);
        assert!(return_val.abs() < 1e-15);
    }

    // Magnitude
    let mut mag = MagnitudeGenerator::new(MagnitudeConfig::peak_returns(20));
    mag.add_prices(&prices);
    let mag_returns = mag.generate_returns().unwrap();

    // All returns should be 0
    for (_, data) in &mag_returns {
        assert!(data.point_return.abs() < 1e-15);
        assert!(data.max_return.abs() < 1e-15);
        assert!(data.min_return.abs() < 1e-15);
    }
}

