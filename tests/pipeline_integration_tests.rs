//! Pipeline Integration Tests
//!
//! Tests for TensorFormatter and MultiHorizonLabelGenerator integration
//! with the PipelineOutput API.

use feature_extractor::prelude::*;
use std::sync::Arc;

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Create a test PipelineOutput with realistic LOB data.
///
/// # Arguments
/// * `n_sequences` - Number of sequences to create
/// * `window_size` - Timesteps per sequence
/// * `lob_levels` - Number of LOB levels (features = 4 * levels)
///
/// # Returns
/// PipelineOutput with:
/// - Sequences with feature vectors [ask_p, ask_v, bid_p, bid_v] × levels
/// - Mid-prices for each timestep
fn create_test_output(n_sequences: usize, window_size: usize, lob_levels: usize) -> PipelineOutput {
    let n_features = lob_levels * 4;
    let mut sequences = Vec::with_capacity(n_sequences);
    let mut mid_prices = Vec::new();

    let base_price = 100.0;
    let spread = 0.01;
    let tick_size = 0.01;

    for seq_idx in 0..n_sequences {
        let mut features: Vec<FeatureVec> = Vec::with_capacity(window_size);

        for t in 0..window_size {
            // Simulate price drift
            let price_drift = (seq_idx * window_size + t) as f64 * 0.0001;
            let mid_price = base_price + price_drift;
            let best_ask = mid_price + spread / 2.0;
            let best_bid = mid_price - spread / 2.0;

            let mut feature_vec = vec![0.0; n_features];

            // Fill LOB features: [ask_prices..., ask_sizes..., bid_prices..., bid_sizes...]
            for level in 0..lob_levels {
                // Ask prices (indices 0 to levels-1)
                feature_vec[level] = best_ask + (level as f64) * tick_size;
                // Ask sizes (indices levels to 2*levels-1)
                feature_vec[lob_levels + level] = (1000 - level * 50) as f64;
                // Bid prices (indices 2*levels to 3*levels-1)
                feature_vec[2 * lob_levels + level] = best_bid - (level as f64) * tick_size;
                // Bid sizes (indices 3*levels to 4*levels-1)
                feature_vec[3 * lob_levels + level] = (1000 - level * 50) as f64;
            }

            features.push(Arc::new(feature_vec));
            mid_prices.push(mid_price);
        }

        let start_ts = (seq_idx * window_size * 1_000_000) as u64;
        let end_ts = start_ts + (window_size * 1_000_000) as u64;

        sequences.push(Sequence {
            features,
            start_timestamp: start_ts,
            end_timestamp: end_ts,
            duration_ns: end_ts - start_ts,
            length: window_size,
        });
    }

    PipelineOutput {
        sequences,
        mid_prices,
        messages_processed: n_sequences * window_size * 10,
        features_extracted: n_sequences * window_size,
        sequences_generated: n_sequences,
        stride: 1,
        window_size,
        multiscale_sequences: None,
        adaptive_stats: None,
    }
}

/// Create output with a specific price trend for label testing.
fn create_trending_output(
    n_prices: usize,
    trend: &str,
    magnitude: f64,
) -> PipelineOutput {
    let mut mid_prices = Vec::with_capacity(n_prices);
    let base_price = 100.0;

    for i in 0..n_prices {
        let price = match trend {
            "up" => base_price + (i as f64) * magnitude,
            "down" => base_price - (i as f64) * magnitude,
            "stable" => base_price + ((i % 10) as f64 - 5.0) * 0.0001, // Small oscillation
            _ => base_price,
        };
        mid_prices.push(price);
    }

    // Create minimal sequences (not testing sequences here, just labels)
    let features: Vec<FeatureVec> = (0..10)
        .map(|_| Arc::new(vec![0.0; 40]))
        .collect();

    let sequences = vec![Sequence {
        features,
        start_timestamp: 0,
        end_timestamp: 10_000_000,
        duration_ns: 10_000_000,
        length: 10,
    }];

    PipelineOutput {
        sequences,
        mid_prices,
        messages_processed: n_prices * 10,
        features_extracted: n_prices,
        sequences_generated: 1,
        stride: 1,
        window_size: 10,
        multiscale_sequences: None,
        adaptive_stats: None,
    }
}

// ============================================================================
// TensorFormatter Integration Tests
// ============================================================================

#[test]
fn test_format_sequences_flat() {
    let output = create_test_output(5, 100, 10);

    // Format as Flat
    let formatter = TensorFormatter::flat();
    let tensor = output.format_sequences(&formatter).unwrap();

    // Flat format: batch stacks to (N, T, F) -> Array3
    assert_eq!(tensor.shape(), vec![5, 100, 40]);
}

#[test]
fn test_format_sequences_deeplob() {
    let output = create_test_output(5, 100, 10);

    // Format as DeepLOB (N, T, 4, L)
    let formatter = TensorFormatter::deeplob(10);
    let tensor = output.format_sequences(&formatter).unwrap();

    // DeepLOB format: (N, T, 4, L)
    assert_eq!(tensor.shape(), vec![5, 100, 4, 10]);
}

#[test]
fn test_format_sequences_hlob() {
    let output = create_test_output(5, 100, 10);

    // Format as HLOB (N, T, L, 4)
    let formatter = TensorFormatter::hlob(10);
    let tensor = output.format_sequences(&formatter).unwrap();

    // HLOB format: (N, T, L, 4)
    assert_eq!(tensor.shape(), vec![5, 100, 10, 4]);
}

#[test]
fn test_format_as_convenience() {
    let output = create_test_output(3, 50, 10);

    // Use convenience method
    let tensor = output.format_as(TensorFormat::deeplob()).unwrap();

    assert_eq!(tensor.shape(), vec![3, 50, 4, 10]);
}

#[test]
fn test_format_sequences_empty_error() {
    let empty_output = PipelineOutput {
        sequences: Vec::new(),
        mid_prices: Vec::new(),
        messages_processed: 0,
        features_extracted: 0,
        sequences_generated: 0,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let formatter = TensorFormatter::flat();
    let result = empty_output.format_sequences(&formatter);

    assert!(result.is_err(), "Should error on empty sequences");
}

#[test]
fn test_format_sequences_preserves_values() {
    let output = create_test_output(1, 10, 10);

    // Get flat features for comparison
    let flat = output.to_flat_features();
    let first_seq_first_step = &flat[0];

    // Format as flat tensor
    let tensor = output.format_as(TensorFormat::Flat).unwrap();
    let arr = tensor.into_array3();

    // Check that values match
    for (i, &expected) in first_seq_first_step.iter().enumerate() {
        let actual = arr[[0, 0, i]];
        assert!(
            (expected - actual).abs() < f64::EPSILON,
            "Value mismatch at feature {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_format_deeplob_channel_ordering() {
    // Create output with known values
    let n_levels = 10;
    let mut feature_vec = vec![0.0; n_levels * 4];

    // Set known values:
    // ask_prices: 1.0, 2.0, ..., 10.0
    // ask_sizes:  11.0, 12.0, ..., 20.0
    // bid_prices: 21.0, 22.0, ..., 30.0
    // bid_sizes:  31.0, 32.0, ..., 40.0
    for i in 0..n_levels {
        feature_vec[i] = (i + 1) as f64;              // ask_price
        feature_vec[n_levels + i] = (i + 11) as f64;   // ask_size
        feature_vec[2 * n_levels + i] = (i + 21) as f64; // bid_price
        feature_vec[3 * n_levels + i] = (i + 31) as f64; // bid_size
    }

    let sequences = vec![Sequence {
        features: vec![Arc::new(feature_vec)],
        start_timestamp: 0,
        end_timestamp: 1000,
        duration_ns: 1000,
        length: 1,
    }];

    let output = PipelineOutput {
        sequences,
        mid_prices: vec![100.0],
        messages_processed: 1,
        features_extracted: 1,
        sequences_generated: 1,
        stride: 1,
        window_size: 1,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    // Format as DeepLOB
    let formatter = TensorFormatter::deeplob(10);
    let tensor = output.format_sequences(&formatter).unwrap();
    let arr = tensor.into_array4();

    // Channel 0 should be ask_prices: [1, 2, ..., 10]
    assert_eq!(arr[[0, 0, 0, 0]], 1.0);
    assert_eq!(arr[[0, 0, 0, 9]], 10.0);

    // Channel 1 should be ask_sizes: [11, 12, ..., 20]
    assert_eq!(arr[[0, 0, 1, 0]], 11.0);
    assert_eq!(arr[[0, 0, 1, 9]], 20.0);

    // Channel 2 should be bid_prices: [21, 22, ..., 30]
    assert_eq!(arr[[0, 0, 2, 0]], 21.0);
    assert_eq!(arr[[0, 0, 2, 9]], 30.0);

    // Channel 3 should be bid_sizes: [31, 32, ..., 40]
    assert_eq!(arr[[0, 0, 3, 0]], 31.0);
    assert_eq!(arr[[0, 0, 3, 9]], 40.0);
}

#[test]
fn test_format_no_nan_or_inf() {
    let output = create_test_output(10, 100, 10);

    let tensor = output.format_as(TensorFormat::deeplob()).unwrap();
    let arr = tensor.into_array4();

    for &val in arr.iter() {
        assert!(!val.is_nan(), "Found NaN in formatted tensor");
        assert!(val.is_finite(), "Found Inf in formatted tensor");
    }
}

// ============================================================================
// Multi-Horizon Label Generation Tests
// ============================================================================

#[test]
fn test_generate_multi_horizon_labels_fi2010() {
    // FI-2010 requires horizons [10, 20, 30, 50, 100]
    // Need enough prices: max_horizon + smoothing_window + 1 = 100 + 5 + 1 = 106 minimum
    let output = create_trending_output(500, "up", 0.001);

    let config = MultiHorizonConfig::fi2010();
    let labels = output.generate_multi_horizon_labels(config).unwrap();

    // Should have labels for all FI-2010 horizons
    assert!(labels.labels_for_horizon(10).is_some());
    assert!(labels.labels_for_horizon(20).is_some());
    assert!(labels.labels_for_horizon(30).is_some());
    assert!(labels.labels_for_horizon(50).is_some());
    assert!(labels.labels_for_horizon(100).is_some());
}

#[test]
fn test_generate_multi_horizon_labels_deeplob() {
    let output = create_trending_output(500, "up", 0.001);

    let config = MultiHorizonConfig::deeplob();
    let labels = output.generate_multi_horizon_labels(config).unwrap();

    // DeepLOB horizons: [10, 20, 50, 100]
    assert!(labels.labels_for_horizon(10).is_some());
    assert!(labels.labels_for_horizon(20).is_some());
    assert!(labels.labels_for_horizon(50).is_some());
    assert!(labels.labels_for_horizon(100).is_some());
}

#[test]
fn test_generate_multi_horizon_labels_upward_trend() {
    let output = create_trending_output(500, "up", 0.01); // Strong upward trend

    let config = MultiHorizonConfig::new(vec![10, 50], 5, ThresholdStrategy::Fixed(0.0001)); // Low threshold
    let labels = output.generate_multi_horizon_labels(config).unwrap();

    // With strong upward trend, most labels should be Up
    // labels_for_horizon returns Option<&[(usize, TrendLabel, f64)]>
    let h10_labels = labels.labels_for_horizon(10).unwrap();
    let up_count = h10_labels.iter().filter(|(_, label, _)| *label == TrendLabel::Up).count();
    let total = h10_labels.len();

    assert!(
        up_count as f64 / total as f64 > 0.8,
        "Expected >80% Up labels for strong upward trend, got {}%",
        (up_count as f64 / total as f64) * 100.0
    );
}

#[test]
fn test_generate_multi_horizon_labels_downward_trend() {
    let output = create_trending_output(500, "down", 0.01); // Strong downward trend

    let config = MultiHorizonConfig::new(vec![10, 50], 5, ThresholdStrategy::Fixed(0.0001));
    let labels = output.generate_multi_horizon_labels(config).unwrap();

    // With strong downward trend, most labels should be Down
    let h10_labels = labels.labels_for_horizon(10).unwrap();
    let down_count = h10_labels.iter().filter(|(_, label, _)| *label == TrendLabel::Down).count();
    let total = h10_labels.len();

    assert!(
        down_count as f64 / total as f64 > 0.8,
        "Expected >80% Down labels for strong downward trend, got {}%",
        (down_count as f64 / total as f64) * 100.0
    );
}

#[test]
fn test_generate_multi_horizon_labels_stable() {
    let output = create_trending_output(500, "stable", 0.0); // Flat prices

    let config = MultiHorizonConfig::new(vec![10, 50], 5, ThresholdStrategy::Fixed(0.001)); // Higher threshold
    let labels = output.generate_multi_horizon_labels(config).unwrap();

    // With stable prices, most labels should be Stable
    let h10_labels = labels.labels_for_horizon(10).unwrap();
    let stable_count = h10_labels.iter().filter(|(_, label, _)| *label == TrendLabel::Stable).count();
    let total = h10_labels.len();

    assert!(
        stable_count as f64 / total as f64 > 0.7,
        "Expected >70% Stable labels for flat prices, got {}%",
        (stable_count as f64 / total as f64) * 100.0
    );
}

#[test]
fn test_generate_multi_horizon_labels_insufficient_data() {
    // Only 50 prices - not enough for h=100
    let output = create_trending_output(50, "up", 0.001);

    let config = MultiHorizonConfig::fi2010(); // Requires h=100
    let result = output.generate_multi_horizon_labels(config);

    assert!(result.is_err(), "Should error with insufficient data for large horizon");
}

#[test]
fn test_generate_multi_horizon_labels_empty_prices() {
    let empty_output = PipelineOutput {
        sequences: Vec::new(),
        mid_prices: Vec::new(),
        messages_processed: 0,
        features_extracted: 0,
        sequences_generated: 0,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let config = MultiHorizonConfig::hft();
    let result = empty_output.generate_multi_horizon_labels(config);

    assert!(result.is_err(), "Should error with empty mid-prices");
}

#[test]
fn test_multi_horizon_labels_summary() {
    let output = create_trending_output(500, "up", 0.005);

    let config = MultiHorizonConfig::fi2010();
    let labels = output.generate_multi_horizon_labels(config).unwrap();

    // Get summary
    let summary = labels.summary();

    assert!(summary.total_labels > 0);
    assert_eq!(summary.num_horizons, 5); // FI-2010 has 5 horizons
}

// ============================================================================
// Helper Method Tests
// ============================================================================

#[test]
fn test_feature_count() {
    let output = create_test_output(5, 100, 10);
    assert_eq!(output.feature_count(), Some(40)); // 10 levels × 4
}

#[test]
fn test_feature_count_empty() {
    let empty_output = PipelineOutput {
        sequences: Vec::new(),
        mid_prices: Vec::new(),
        messages_processed: 0,
        features_extracted: 0,
        sequences_generated: 0,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    assert_eq!(empty_output.feature_count(), None);
}

#[test]
fn test_lob_levels() {
    let output = create_test_output(5, 100, 10);
    assert_eq!(output.lob_levels(), Some(10));

    let output_5 = create_test_output(5, 100, 5);
    assert_eq!(output_5.lob_levels(), Some(5));
}

// ============================================================================
// Determinism Tests
// ============================================================================

#[test]
fn test_format_deterministic() {
    let output = create_test_output(5, 100, 10);

    let formatter = TensorFormatter::deeplob(10);
    let tensor1 = output.format_sequences(&formatter).unwrap();
    let tensor2 = output.format_sequences(&formatter).unwrap();

    assert_eq!(tensor1.shape(), tensor2.shape());

    let arr1 = tensor1.into_array4();
    let arr2 = tensor2.into_array4();

    // All values should be identical
    for (v1, v2) in arr1.iter().zip(arr2.iter()) {
        assert_eq!(v1, v2, "Tensor formatting should be deterministic");
    }
}

#[test]
fn test_label_generation_deterministic() {
    let output = create_trending_output(500, "up", 0.001);

    let config1 = MultiHorizonConfig::fi2010();
    let config2 = MultiHorizonConfig::fi2010();

    let labels1 = output.generate_multi_horizon_labels(config1).unwrap();
    let labels2 = output.generate_multi_horizon_labels(config2).unwrap();

    // Labels should be identical
    for horizon in [10, 20, 30, 50, 100] {
        let l1 = labels1.labels_for_horizon(horizon).unwrap();
        let l2 = labels2.labels_for_horizon(horizon).unwrap();

        assert_eq!(l1.len(), l2.len());
        for (a, b) in l1.iter().zip(l2.iter()) {
            assert_eq!(a, b, "Label generation should be deterministic");
        }
    }
}

