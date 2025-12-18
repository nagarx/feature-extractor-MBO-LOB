//! Tests for AlignedBatchExporter with tensor format and multi-horizon support.

use feature_extractor::prelude::*;

// ============================================================================
// Builder Pattern Tests
// ============================================================================

#[test]
fn test_aligned_exporter_default() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10);

    assert!(!exporter.is_tensor_formatted());
    assert!(!exporter.is_multi_horizon());
    assert!(exporter.tensor_format().is_none());
    assert!(exporter.multi_horizon_config().is_none());
}

#[test]
fn test_aligned_exporter_with_tensor_format() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::DeepLOB { levels: 10 });

    assert!(exporter.is_tensor_formatted());
    assert!(!exporter.is_multi_horizon());

    match exporter.tensor_format() {
        Some(TensorFormat::DeepLOB { levels: 10 }) => {}
        _ => panic!("Expected DeepLOB format with 10 levels"),
    }
}

#[test]
fn test_aligned_exporter_with_multi_horizon() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_multi_horizon_labels(MultiHorizonConfig::fi2010());

    assert!(!exporter.is_tensor_formatted());
    assert!(exporter.is_multi_horizon());

    let multi_config = exporter.multi_horizon_config().expect("Should have config");
    assert_eq!(multi_config.horizons(), &[10, 20, 30, 50, 100]);
}

#[test]
fn test_aligned_exporter_full_config() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::HLOB { levels: 10 })
        .with_multi_horizon_labels(MultiHorizonConfig::deeplob());

    assert!(exporter.is_tensor_formatted());
    assert!(exporter.is_multi_horizon());

    match exporter.tensor_format() {
        Some(TensorFormat::HLOB { levels: 10 }) => {}
        _ => panic!("Expected HLOB format with 10 levels"),
    }

    let multi_config = exporter.multi_horizon_config().expect("Should have config");
    assert_eq!(multi_config.horizons(), &[10, 20, 50, 100]);
}

#[test]
fn test_aligned_exporter_with_feature_mapping() {
    let config = LabelConfig::new(50, 5, 0.002);
    let mapping = FeatureMapping::standard_lob(10);

    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::DeepLOB { levels: 10 })
        .with_feature_mapping(mapping);

    assert!(exporter.is_tensor_formatted());
}

// ============================================================================
// Tensor Format Variant Tests
// ============================================================================

#[test]
fn test_tensor_format_flat() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::Flat);

    match exporter.tensor_format() {
        Some(TensorFormat::Flat) => {}
        _ => panic!("Expected Flat format"),
    }
}

#[test]
fn test_tensor_format_deeplob() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::DeepLOB { levels: 5 });

    match exporter.tensor_format() {
        Some(TensorFormat::DeepLOB { levels: 5 }) => {}
        _ => panic!("Expected DeepLOB format with 5 levels"),
    }
}

#[test]
fn test_tensor_format_hlob() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::HLOB { levels: 20 });

    match exporter.tensor_format() {
        Some(TensorFormat::HLOB { levels: 20 }) => {}
        _ => panic!("Expected HLOB format with 20 levels"),
    }
}

#[test]
fn test_tensor_format_image() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10).with_tensor_format(
        TensorFormat::Image {
            channels: 1,
            height: 10,
            width: 4,
        },
    );

    match exporter.tensor_format() {
        Some(TensorFormat::Image {
            channels: 1,
            height: 10,
            width: 4,
        }) => {}
        _ => panic!("Expected Image format"),
    }
}

// ============================================================================
// Multi-Horizon Config Tests
// ============================================================================

#[test]
fn test_multi_horizon_fi2010_preset() {
    let config = MultiHorizonConfig::fi2010();
    assert_eq!(config.horizons(), &[10, 20, 30, 50, 100]);
    assert_eq!(config.smoothing_window, 5);
}

#[test]
fn test_multi_horizon_deeplob_preset() {
    let config = MultiHorizonConfig::deeplob();
    assert_eq!(config.horizons(), &[10, 20, 50, 100]);
    assert_eq!(config.smoothing_window, 5);
}

#[test]
fn test_multi_horizon_tlob_preset() {
    let config = MultiHorizonConfig::tlob();
    // TLOB uses different horizons
    let horizons = config.horizons();
    assert!(!horizons.is_empty());
}

#[test]
fn test_multi_horizon_custom() {
    let config = MultiHorizonConfig::new(vec![5, 10, 15, 25], 3, ThresholdStrategy::Fixed(0.001));

    assert_eq!(config.horizons(), &[5, 10, 15, 25]);
    assert_eq!(config.smoothing_window, 3);
}

// ============================================================================
// Integration Tests (with synthetic data)
// ============================================================================

/// Create synthetic pipeline output for testing
fn create_synthetic_pipeline_output(n_sequences: usize, window_size: usize) -> PipelineOutput {
    use std::sync::Arc;

    let n_features = 40; // Standard LOB features
    let stride = 10;

    // Generate synthetic sequences
    let mut sequences = Vec::with_capacity(n_sequences);
    for seq_idx in 0..n_sequences {
        let features: Vec<Arc<Vec<f64>>> = (0..window_size)
            .map(|t| {
                // Create realistic LOB features
                // Layout: [ask_prices(10), ask_sizes(10), bid_prices(10), bid_sizes(10)]
                let mut f = Vec::with_capacity(n_features);

                // Ask prices (increasing from best ask)
                for l in 0..10 {
                    f.push(100.0 + 0.01 * (l as f64) + 0.001 * (t as f64));
                }
                // Ask sizes
                for l in 0..10 {
                    f.push(100.0 + 10.0 * (l as f64));
                }
                // Bid prices (decreasing from best bid, below ask)
                for l in 0..10 {
                    f.push(99.99 - 0.01 * (l as f64) + 0.001 * (t as f64));
                }
                // Bid sizes
                for l in 0..10 {
                    f.push(100.0 + 10.0 * (l as f64));
                }

                Arc::new(f)
            })
            .collect();

        let start_ts = (seq_idx * stride * 1_000_000) as u64; // Nanoseconds
        let end_ts = start_ts + (window_size * 1_000_000) as u64;

        sequences.push(Sequence {
            features,
            start_timestamp: start_ts,
            end_timestamp: end_ts,
            duration_ns: end_ts - start_ts,
            length: window_size,
        });
    }

    // Generate mid-prices for labeling (enough for the sequences)
    let n_prices = n_sequences * stride + window_size + 200; // Extra for label generation buffer
    let mid_prices: Vec<f64> = (0..n_prices)
        .map(|i| 100.0 + 0.001 * (i as f64).sin() * 10.0) // Oscillating prices
        .collect();

    PipelineOutput {
        sequences,
        mid_prices,
        messages_processed: n_prices * 10,
        features_extracted: n_prices,
        sequences_generated: n_sequences,
        stride,
        window_size,
        multiscale_sequences: None,
        adaptive_stats: None,
    }
}

#[test]
fn test_synthetic_output_creation() {
    let output = create_synthetic_pipeline_output(100, 50);

    assert_eq!(output.sequences.len(), 100);
    assert_eq!(output.sequences[0].features.len(), 50);
    assert_eq!(output.sequences[0].features[0].len(), 40);
    assert!(!output.mid_prices.is_empty());
}

#[test]
fn test_synthetic_output_spread_validity() {
    let output = create_synthetic_pipeline_output(10, 50);

    // Verify spreads are positive (ask > bid)
    for seq in &output.sequences {
        for timestep in &seq.features {
            let ask_price = timestep[0]; // Best ask
            let bid_price = timestep[20]; // Best bid
            assert!(
                ask_price > bid_price,
                "Spread should be positive: ask={} bid={}",
                ask_price,
                bid_price
            );
        }
    }
}

// ============================================================================
// Export Path Tests (file system interactions)
// ============================================================================

#[test]
fn test_exporter_creates_output_directory() {
    use std::path::Path;

    let test_dir = "/tmp/feature_extractor_test_output";
    let config = LabelConfig::new(50, 5, 0.002);

    // Clean up any previous test run
    let _ = std::fs::remove_dir_all(test_dir);

    let exporter = AlignedBatchExporter::new(test_dir, config, 100, 10);

    // Directory not created until export
    assert!(!Path::new(test_dir).exists());

    // After creating exporter, we could call export_day but need valid data
    // This just tests the builder doesn't create the dir prematurely
    drop(exporter);

    // Clean up
    let _ = std::fs::remove_dir_all(test_dir);
}

// ============================================================================
// Normalization Types Tests
// ============================================================================

#[test]
fn test_normalization_strategy_default() {
    let strategy = NormalizationStrategy::default();
    assert_eq!(strategy, NormalizationStrategy::MarketStructureZScore);
}

#[test]
fn test_normalization_strategy_display() {
    // Note: Display impl uses custom formatting, different from serde
    assert_eq!(NormalizationStrategy::None.to_string(), "none");
    assert_eq!(
        NormalizationStrategy::PerFeatureZScore.to_string(),
        "per_feature_zscore"
    );
    assert_eq!(
        NormalizationStrategy::MarketStructureZScore.to_string(),
        "market_structure_zscore"
    );
    assert_eq!(
        NormalizationStrategy::GlobalZScore.to_string(),
        "global_zscore"
    );
    assert_eq!(NormalizationStrategy::Bilinear.to_string(), "bilinear");

    // Serde uses rename_all = "snake_case" which produces slightly different output
    // e.g., MarketStructureZScore -> "market_structure_z_score" in JSON
}

#[test]
fn test_normalization_params_new() {
    let price_means = vec![100.0; 10];
    let price_stds = vec![0.1; 10];
    let size_means = vec![500.0; 20];
    let size_stds = vec![100.0; 20];

    let params = NormalizationParams::new(
        price_means.clone(),
        price_stds.clone(),
        size_means.clone(),
        size_stds.clone(),
        10000,
        10,
    );

    assert_eq!(
        params.strategy,
        NormalizationStrategy::MarketStructureZScore
    );
    assert_eq!(params.price_means.len(), 10);
    assert_eq!(params.price_stds.len(), 10);
    assert_eq!(params.size_means.len(), 20);
    assert_eq!(params.size_stds.len(), 20);
    assert_eq!(params.sample_count, 10000);
    assert_eq!(params.levels, 10);
    assert_eq!(
        params.feature_layout,
        "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10"
    );
}

#[test]
fn test_normalization_params_serialization() {
    let params = NormalizationParams::new(
        vec![
            100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07, 100.08, 100.09,
        ],
        vec![0.1; 10],
        vec![500.0; 20],
        vec![100.0; 20],
        5000,
        10,
    );

    // Serialize to JSON string
    let json = serde_json::to_string_pretty(&params).expect("Failed to serialize");

    // Verify JSON contains expected fields (strategy is serialized with serde rename_all)
    // Note: serde rename_all = "snake_case" converts MarketStructureZScore to market_structure_z_score
    assert!(
        json.contains("market_structure_z_score"),
        "JSON should contain strategy: {}",
        json
    );
    assert!(
        json.contains("sample_count"),
        "JSON should contain sample_count"
    );
    assert!(
        json.contains("5000"),
        "JSON should contain sample count value"
    );
    assert!(json.contains("levels"), "JSON should contain levels");
    assert!(
        json.contains("price_means"),
        "JSON should contain price_means"
    );
    assert!(
        json.contains("price_stds"),
        "JSON should contain price_stds"
    );
    assert!(
        json.contains("size_means"),
        "JSON should contain size_means"
    );
    assert!(json.contains("size_stds"), "JSON should contain size_stds");

    // Deserialize back
    let deserialized: NormalizationParams =
        serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.strategy, params.strategy);
    assert_eq!(deserialized.sample_count, params.sample_count);
    assert_eq!(deserialized.levels, params.levels);
    assert_eq!(deserialized.price_means.len(), 10);
}

#[test]
fn test_normalization_params_save_load() {
    let temp_dir = std::env::temp_dir().join("normalization_test");
    std::fs::create_dir_all(&temp_dir).unwrap();
    let path = temp_dir.join("test_norm_params.json");

    let params = NormalizationParams::new(
        vec![100.0; 10],
        vec![0.1; 10],
        vec![500.0; 20],
        vec![100.0; 20],
        8000,
        10,
    );

    // Save to file
    params.save_json(&path).expect("Failed to save");

    // Verify file exists
    assert!(path.exists());

    // Load from file
    let loaded = NormalizationParams::load_json(&path).expect("Failed to load");

    assert_eq!(loaded.strategy, params.strategy);
    assert_eq!(loaded.sample_count, params.sample_count);
    assert_eq!(loaded.levels, params.levels);
    assert_eq!(loaded.price_means, params.price_means);
    assert_eq!(loaded.price_stds, params.price_stds);
    assert_eq!(loaded.size_means, params.size_means);
    assert_eq!(loaded.size_stds, params.size_stds);

    // Cleanup
    std::fs::remove_dir_all(temp_dir).unwrap();
}
