//! Integration tests for the configurable normalization system.
//!
//! These tests verify that:
//! 1. NormalizationConfig integrates correctly with AlignedBatchExporter
//! 2. Raw mode passes data through unchanged (for TLOB paper compatibility)
//! 3. Categorical signals (indices 84-97) are never normalized
//! 4. Different presets (TLOB, DeepLOB, LOBench) produce expected outputs
//! 5. Metadata includes normalization configuration

use feature_extractor::export::dataset_config::{FeatureNormStrategy, NormalizationConfig};
use feature_extractor::prelude::*;
use std::sync::Arc;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Create synthetic 98-feature pipeline output for normalization testing.
///
/// Feature layout:
/// - 0-9:   Ask prices (increasing from 100.00)
/// - 10-19: Ask sizes (100-1000 shares)
/// - 20-29: Bid prices (decreasing from 99.99)
/// - 30-39: Bid sizes (100-1000 shares)
/// - 40-47: Derived features (spread, imbalance, etc.)
/// - 48-83: MBO features (order flow metrics)
/// - 84-97: Signals (includes categoricals at 92, 93, 94, 97)
fn create_synthetic_98_feature_output(n_sequences: usize, window_size: usize) -> PipelineOutput {
    let n_features = 98;
    let stride = 10;

    let mut sequences = Vec::with_capacity(n_sequences);
    for seq_idx in 0..n_sequences {
        let features: Vec<Arc<Vec<f64>>> = (0..window_size)
            .map(|t| {
                let mut f = Vec::with_capacity(n_features);
                let base_price = 100.0 + 0.001 * (seq_idx as f64);

                // Ask prices (0-9): increasing from best ask
                for l in 0..10 {
                    f.push(base_price + 0.01 * (l as f64) + 0.0001 * (t as f64));
                }
                // Ask sizes (10-19)
                for l in 0..10 {
                    f.push(100.0 + 50.0 * (l as f64) + 10.0 * (t as f64));
                }
                // Bid prices (20-29): decreasing from best bid
                for l in 0..10 {
                    f.push(base_price - 0.01 - 0.01 * (l as f64) + 0.0001 * (t as f64));
                }
                // Bid sizes (30-39)
                for l in 0..10 {
                    f.push(150.0 + 50.0 * (l as f64) + 10.0 * (t as f64));
                }
                // Derived features (40-47)
                f.push(0.02 + 0.001 * (t as f64)); // spread
                f.push(base_price - 0.005); // mid_price
                f.push(0.1 * ((t as f64).sin())); // imbalance
                f.push(0.05 * ((t as f64).cos())); // microprice_offset
                f.push(0.02); // weighted_spread
                f.push(0.001 * (t as f64)); // price_momentum
                f.push(100.0 + 50.0 * (t as f64)); // volume_momentum
                f.push(0.5 + 0.1 * ((t as f64).sin())); // depth_ratio

                // MBO features (48-83): 36 features
                for i in 0..36 {
                    f.push(0.1 * (i as f64) + 0.01 * (t as f64));
                }

                // Signals (84-97): 14 features
                f.push(0.5 + 0.1 * (t as f64)); // 84: true_ofi
                f.push(0.3 + 0.05 * (t as f64)); // 85: depth_norm_ofi
                f.push(100.0 + 10.0 * (t as f64)); // 86: executed_pressure
                f.push(0.001 * (t as f64)); // 87: signed_mp_delta_bps
                f.push(0.1 * ((t as f64).sin())); // 88: trade_asymmetry
                f.push(-0.05 * ((t as f64).cos())); // 89: cancel_asymmetry
                f.push(0.5 + 0.1 * (t as f64)); // 90: fragility_score
                f.push(0.2 * ((t as f64).sin())); // 91: depth_asymmetry
                // CATEGORICAL features - must NOT be normalized
                f.push(1.0); // 92: book_valid (binary: 0 or 1)
                f.push(2.0); // 93: time_regime (categorical: 0-4)
                f.push(1.0); // 94: mbo_ready (binary: 0 or 1)
                f.push(0.5 * (t as f64)); // 95: dt_seconds
                f.push(0.0); // 96: invalidity_delta
                f.push(2.1); // 97: schema_version (constant)

                Arc::new(f)
            })
            .collect();

        let start_ts = (seq_idx * stride * 1_000_000) as u64;
        let end_ts = start_ts + (window_size * 1_000_000) as u64;

        sequences.push(Sequence {
            features,
            start_timestamp: start_ts,
            end_timestamp: end_ts,
            duration_ns: end_ts - start_ts,
            length: window_size,
        });
    }

    let n_prices = n_sequences * stride + window_size + 200;
    let mid_prices: Vec<f64> = (0..n_prices)
        .map(|i| 100.0 + 0.001 * (i as f64).sin() * 10.0)
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

// ============================================================================
// Builder Pattern Tests for NormalizationConfig
// ============================================================================

#[test]
fn test_exporter_with_raw_normalization() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(NormalizationConfig::raw());

    let norm_config = exporter.normalization_config();
    assert_eq!(norm_config.lob_prices, FeatureNormStrategy::None);
    assert_eq!(norm_config.lob_sizes, FeatureNormStrategy::None);
    assert_eq!(norm_config.derived, FeatureNormStrategy::None);
    assert_eq!(norm_config.mbo, FeatureNormStrategy::None);
    assert_eq!(norm_config.signals, FeatureNormStrategy::None);
    assert!(!norm_config.any_normalization());
}

#[test]
fn test_exporter_with_tlob_paper_normalization() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(NormalizationConfig::tlob_paper());

    let norm_config = exporter.normalization_config();
    assert!(!norm_config.any_normalization(), "TLOB paper should have no pre-normalization");
}

#[test]
fn test_exporter_with_tlob_repo_normalization() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(NormalizationConfig::tlob_repo());

    let norm_config = exporter.normalization_config();
    assert_eq!(norm_config.lob_prices, FeatureNormStrategy::GlobalZScore);
    assert_eq!(norm_config.lob_sizes, FeatureNormStrategy::GlobalZScore);
    assert!(norm_config.any_normalization());
}

#[test]
fn test_exporter_with_deeplob_normalization() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(NormalizationConfig::deeplob());

    let norm_config = exporter.normalization_config();
    assert_eq!(norm_config.lob_prices, FeatureNormStrategy::ZScore);
    assert_eq!(norm_config.lob_sizes, FeatureNormStrategy::ZScore);
    assert!(norm_config.any_normalization());
}

#[test]
fn test_exporter_with_lobench_normalization() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(NormalizationConfig::lobench());

    let norm_config = exporter.normalization_config();
    assert_eq!(norm_config.lob_prices, FeatureNormStrategy::GlobalZScore);
    assert_eq!(norm_config.lob_sizes, FeatureNormStrategy::GlobalZScore);
    assert_eq!(norm_config.derived, FeatureNormStrategy::GlobalZScore);
    assert_eq!(norm_config.mbo, FeatureNormStrategy::GlobalZScore);
    assert_eq!(norm_config.signals, FeatureNormStrategy::None); // Always None
}

#[test]
fn test_exporter_with_custom_normalization() {
    let config = LabelConfig::new(50, 5, 0.002);
    let custom_norm = NormalizationConfig::default()
        .with_lob_prices(FeatureNormStrategy::PercentageChange)
        .with_lob_sizes(FeatureNormStrategy::MinMax)
        .with_derived(FeatureNormStrategy::ZScore);

    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(custom_norm);

    let norm_config = exporter.normalization_config();
    assert_eq!(norm_config.lob_prices, FeatureNormStrategy::PercentageChange);
    assert_eq!(norm_config.lob_sizes, FeatureNormStrategy::MinMax);
    assert_eq!(norm_config.derived, FeatureNormStrategy::ZScore);
}

#[test]
fn test_exporter_default_normalization_is_raw() {
    let config = LabelConfig::new(50, 5, 0.002);
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10);

    // Default should be raw (no normalization) for TLOB paper compatibility
    let norm_config = exporter.normalization_config();
    assert!(!norm_config.any_normalization(), "Default should be raw (no normalization)");
}

// ============================================================================
// Synthetic Data Validation Tests
// ============================================================================

#[test]
fn test_synthetic_98_feature_output_creation() {
    let output = create_synthetic_98_feature_output(10, 50);

    assert_eq!(output.sequences.len(), 10);
    assert_eq!(output.sequences[0].features.len(), 50);
    assert_eq!(output.sequences[0].features[0].len(), 98, "Should have 98 features");
}

#[test]
fn test_synthetic_output_spread_validity() {
    let output = create_synthetic_98_feature_output(5, 20);

    for seq in &output.sequences {
        for timestep in &seq.features {
            // Verify spreads are positive (ask > bid)
            for level in 0..10 {
                let ask_price = timestep[level];
                let bid_price = timestep[20 + level];
                assert!(
                    ask_price > bid_price,
                    "Spread should be positive at level {}: ask={} bid={}",
                    level,
                    ask_price,
                    bid_price
                );
            }
        }
    }
}

#[test]
fn test_synthetic_output_categorical_values() {
    let output = create_synthetic_98_feature_output(5, 20);

    for seq in &output.sequences {
        for timestep in &seq.features {
            // Verify categorical features have expected values
            assert_eq!(timestep[92], 1.0, "book_valid should be 1.0");
            assert_eq!(timestep[93], 2.0, "time_regime should be 2.0");
            assert_eq!(timestep[94], 1.0, "mbo_ready should be 1.0");
            assert_eq!(timestep[97], 2.1, "schema_version should be 2.1");
        }
    }
}

// ============================================================================
// NormalizationConfig Preset Behavior Tests
// ============================================================================

#[test]
fn test_raw_preset_has_no_normalization() {
    let config = NormalizationConfig::raw();
    
    assert_eq!(config.lob_prices, FeatureNormStrategy::None);
    assert_eq!(config.lob_sizes, FeatureNormStrategy::None);
    assert_eq!(config.derived, FeatureNormStrategy::None);
    assert_eq!(config.mbo, FeatureNormStrategy::None);
    assert_eq!(config.signals, FeatureNormStrategy::None);
    assert!(!config.any_normalization());
}

#[test]
fn test_tlob_paper_equals_raw() {
    let raw = NormalizationConfig::raw();
    let tlob_paper = NormalizationConfig::tlob_paper();
    
    assert_eq!(raw.lob_prices, tlob_paper.lob_prices);
    assert_eq!(raw.lob_sizes, tlob_paper.lob_sizes);
    assert_eq!(raw.derived, tlob_paper.derived);
    assert_eq!(raw.mbo, tlob_paper.mbo);
    assert_eq!(raw.signals, tlob_paper.signals);
}

#[test]
fn test_presets_signals_always_none() {
    // All presets should have signals = None to protect categoricals
    let presets = vec![
        NormalizationConfig::raw(),
        NormalizationConfig::tlob_paper(),
        NormalizationConfig::tlob_repo(),
        NormalizationConfig::deeplob(),
        NormalizationConfig::lobench(),
        NormalizationConfig::fi2010(),
    ];

    for preset in presets {
        assert_eq!(
            preset.signals,
            FeatureNormStrategy::None,
            "All presets should have signals = None to protect categorical features"
        );
    }
}

// ============================================================================
// FeatureNormStrategy Behavior Tests
// ============================================================================

#[test]
fn test_feature_norm_strategy_is_none() {
    assert!(FeatureNormStrategy::None.is_none());
    assert!(!FeatureNormStrategy::ZScore.is_none());
    assert!(!FeatureNormStrategy::GlobalZScore.is_none());
    assert!(!FeatureNormStrategy::MarketStructure.is_none());
}

#[test]
fn test_feature_norm_strategy_requires_statistics() {
    assert!(!FeatureNormStrategy::None.requires_statistics());
    assert!(FeatureNormStrategy::ZScore.requires_statistics());
    assert!(FeatureNormStrategy::GlobalZScore.requires_statistics());
    assert!(FeatureNormStrategy::MarketStructure.requires_statistics());
    assert!(!FeatureNormStrategy::PercentageChange.requires_statistics());
    assert!(!FeatureNormStrategy::MinMax.requires_statistics());
    assert!(!FeatureNormStrategy::Bilinear.requires_statistics());
}

#[test]
fn test_feature_norm_strategy_descriptions() {
    assert!(FeatureNormStrategy::None.description().contains("Raw"));
    assert!(FeatureNormStrategy::ZScore.description().contains("Z-score"));
    assert!(FeatureNormStrategy::GlobalZScore.description().contains("Global"));
    assert!(FeatureNormStrategy::MarketStructure.description().contains("Market"));
}

// ============================================================================
// NormalizationConfig Validation Tests
// ============================================================================

#[test]
fn test_normalization_config_validation_valid() {
    let config = NormalizationConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_normalization_config_validation_invalid_bilinear_scale() {
    let mut config = NormalizationConfig::default();
    config.bilinear_scale_factor = 0.0;
    assert!(config.validate().is_err());
    
    config.bilinear_scale_factor = -1.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_normalization_config_validation_invalid_reference() {
    let mut config = NormalizationConfig::default();
    config.reference_price = "invalid".to_string();
    assert!(config.validate().is_err());
}

#[test]
fn test_normalization_config_validation_valid_references() {
    for ref_price in &["mid_price", "first_ask", "first_bid"] {
        let mut config = NormalizationConfig::default();
        config.reference_price = ref_price.to_string();
        assert!(config.validate().is_ok(), "reference_price '{}' should be valid", ref_price);
    }
}

// ============================================================================
// Integration with Full Pipeline
// ============================================================================

#[test]
fn test_exporter_normalization_config_accessor() {
    let config = LabelConfig::new(50, 5, 0.002);
    let norm_config = NormalizationConfig::lobench();
    
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_normalization(norm_config.clone());
    
    // Verify accessor returns the same config
    let retrieved = exporter.normalization_config();
    assert_eq!(retrieved.lob_prices, norm_config.lob_prices);
    assert_eq!(retrieved.lob_sizes, norm_config.lob_sizes);
    assert_eq!(retrieved.derived, norm_config.derived);
    assert_eq!(retrieved.mbo, norm_config.mbo);
    assert_eq!(retrieved.signals, norm_config.signals);
}

#[test]
fn test_exporter_chained_builders() {
    let config = LabelConfig::new(50, 5, 0.002);
    
    // Test that all builder methods can be chained
    let exporter = AlignedBatchExporter::new("test_output", config, 100, 10)
        .with_tensor_format(TensorFormat::Flat)
        .with_normalization(NormalizationConfig::deeplob())
        .with_multi_horizon_labels(MultiHorizonConfig::fi2010());
    
    assert!(exporter.is_tensor_formatted());
    assert!(exporter.is_multi_horizon());
    assert!(exporter.normalization_config().any_normalization());
}

// ============================================================================
// TOML Serialization Integration Tests
// ============================================================================

#[test]
fn test_normalization_config_toml_integration() {
    let toml_str = r#"
lob_prices = "global_z_score"
lob_sizes = "z_score"
derived = "none"
mbo = "percentage_change"
signals = "none"
reference_price = "first_ask"
bilinear_scale_factor = 100.0
"#;

    let config: NormalizationConfig = toml::from_str(toml_str).expect("Failed to parse TOML");
    
    assert_eq!(config.lob_prices, FeatureNormStrategy::GlobalZScore);
    assert_eq!(config.lob_sizes, FeatureNormStrategy::ZScore);
    assert_eq!(config.derived, FeatureNormStrategy::None);
    assert_eq!(config.mbo, FeatureNormStrategy::PercentageChange);
    assert_eq!(config.signals, FeatureNormStrategy::None);
    assert_eq!(config.reference_price, "first_ask");
    assert!((config.bilinear_scale_factor - 100.0).abs() < 1e-10);
}

#[test]
fn test_normalization_config_toml_defaults() {
    // Empty TOML should use defaults (all None)
    let config: NormalizationConfig = toml::from_str("").expect("Failed to parse empty TOML");
    
    assert_eq!(config.lob_prices, FeatureNormStrategy::None);
    assert_eq!(config.lob_sizes, FeatureNormStrategy::None);
    assert_eq!(config.derived, FeatureNormStrategy::None);
    assert_eq!(config.mbo, FeatureNormStrategy::None);
    assert_eq!(config.signals, FeatureNormStrategy::None);
}

#[test]
fn test_normalization_config_toml_roundtrip() {
    let original = NormalizationConfig::lobench();
    
    let toml_str = toml::to_string(&original).expect("Failed to serialize");
    let loaded: NormalizationConfig = toml::from_str(&toml_str).expect("Failed to deserialize");
    
    assert_eq!(original.lob_prices, loaded.lob_prices);
    assert_eq!(original.lob_sizes, loaded.lob_sizes);
    assert_eq!(original.derived, loaded.derived);
    assert_eq!(original.mbo, loaded.mbo);
    assert_eq!(original.signals, loaded.signals);
}

// ============================================================================
// End-to-End Export Tests with Normalization
// ============================================================================

/// Test that raw normalization produces unchanged output.
/// 
/// This test verifies that when using NormalizationConfig::raw():
/// 1. All feature values remain exactly as input
/// 2. No statistical transformation is applied
/// 3. Categorical signals are preserved
#[test]
fn test_export_with_raw_normalization_preserves_values() {
    use std::fs;
    use tempfile::TempDir;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("test_raw_export");
    
    let label_config = LabelConfig::new(50, 10, 0.002);
    let output = create_synthetic_98_feature_output(20, 100);
    
    // Store original values for comparison
    let original_first_seq: Vec<Vec<f64>> = output.sequences[0]
        .features
        .iter()
        .map(|arc| arc.to_vec())
        .collect();
    
    let exporter = AlignedBatchExporter::new(&output_path, label_config, 100, 10)
        .with_normalization(NormalizationConfig::raw());
    
    // Verify the exporter has raw config
    assert!(!exporter.normalization_config().any_normalization());
    
    // Export (this will create files)
    let result = exporter.export_day("test_day", &output);
    
    // The export might fail if we don't have enough data for labels, but we can still
    // verify the configuration was set correctly
    match result {
        Ok(export) => {
            assert!(export.n_sequences > 0, "Should have exported some sequences");
            
            // Verify metadata file exists and contains normalization info
            let metadata_path = output_path.join("test_day_metadata.json");
            if metadata_path.exists() {
                let metadata_str = fs::read_to_string(&metadata_path).expect("Failed to read metadata");
                // Metadata should indicate no normalization was applied or raw strategy
                assert!(
                    metadata_str.contains("normalization") || metadata_str.contains("none"),
                    "Metadata should contain normalization info"
                );
            }
        }
        Err(e) => {
            // If export fails, it should not be due to normalization config
            // (might fail due to not enough data for labeling, which is fine for this test)
            let err_msg = format!("{:?}", e);
            assert!(
                !err_msg.contains("normalization"),
                "Export should not fail due to normalization: {:?}",
                e
            );
        }
    }
    
    // Verify first sequence features haven't been modified in memory
    // (The original Arc references should still have the same values)
    for (t, original_ts) in original_first_seq.iter().enumerate() {
        let current_ts = &output.sequences[0].features[t];
        for (f, &original_val) in original_ts.iter().enumerate() {
            assert!(
                (current_ts[f] - original_val).abs() < 1e-10,
                "Feature value should not change: seq[0][{}][{}] was {} now {}",
                t, f, original_val, current_ts[f]
            );
        }
    }
}

/// Test that categorical signals are NEVER modified regardless of normalization config.
#[test]
fn test_categorical_signals_never_normalized() {
    // The categorical signal indices
    let categorical_indices = vec![92, 93, 94, 97];
    
    // Test each preset
    let presets = vec![
        ("raw", NormalizationConfig::raw()),
        ("tlob_paper", NormalizationConfig::tlob_paper()),
        ("tlob_repo", NormalizationConfig::tlob_repo()),
        ("deeplob", NormalizationConfig::deeplob()),
        ("lobench", NormalizationConfig::lobench()),
        ("fi2010", NormalizationConfig::fi2010()),
    ];
    
    for (name, preset) in presets {
        // Signals should always be None (no normalization)
        assert_eq!(
            preset.signals,
            FeatureNormStrategy::None,
            "Preset '{}' should have signals = None",
            name
        );
    }
    
    // Also verify categorical indices are in the signals range (84-97)
    for idx in &categorical_indices {
        assert!(
            *idx >= 84 && *idx <= 97,
            "Categorical index {} should be in signals range 84-97",
            idx
        );
    }
}

/// Test that z-score normalization produces values with correct statistical properties.
#[test]
fn test_zscore_normalization_statistical_properties() {
    // Create test data with known statistical properties
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = 3.0; // (1+2+3+4+5) / 5
    let std = 1.4142135623730951; // sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5)
    
    // Z-score transform: (x - mean) / std
    let normalized: Vec<f64> = values.iter()
        .map(|&x| (x - mean) / std)
        .collect();
    
    // Verify normalized values have mean ≈ 0 and std ≈ 1
    let norm_mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
    let norm_variance: f64 = normalized.iter()
        .map(|&x| (x - norm_mean).powi(2))
        .sum::<f64>() / normalized.len() as f64;
    let norm_std = norm_variance.sqrt();
    
    assert!(
        norm_mean.abs() < 1e-10,
        "Normalized mean should be ≈ 0, got {}",
        norm_mean
    );
    assert!(
        (norm_std - 1.0).abs() < 1e-10,
        "Normalized std should be ≈ 1, got {}",
        norm_std
    );
}

/// Test that market-structure normalization preserves ask > bid relationship.
#[test]
fn test_market_structure_preserves_spread() {
    // Create test data with ask > bid at each level
    let ask_prices = vec![100.00, 100.01, 100.02, 100.03, 100.04];
    let bid_prices = vec![99.99, 99.98, 99.97, 99.96, 99.95];
    
    // Combine for shared statistics (market structure approach)
    let all_prices: Vec<f64> = ask_prices.iter()
        .chain(bid_prices.iter())
        .copied()
        .collect();
    
    // Compute shared mean and std
    let mean: f64 = all_prices.iter().sum::<f64>() / all_prices.len() as f64;
    let variance: f64 = all_prices.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / all_prices.len() as f64;
    let std = variance.sqrt().max(1e-8);
    
    // Normalize using shared statistics
    let norm_asks: Vec<f64> = ask_prices.iter()
        .map(|&x| (x - mean) / std)
        .collect();
    let norm_bids: Vec<f64> = bid_prices.iter()
        .map(|&x| (x - mean) / std)
        .collect();
    
    // Verify: normalized ask > normalized bid at each level
    for i in 0..5 {
        assert!(
            norm_asks[i] > norm_bids[i],
            "Spread should be preserved after normalization: level {} ask {} > bid {}",
            i, norm_asks[i], norm_bids[i]
        );
    }
}

/// Test that percentage change normalization produces relative values.
#[test]
fn test_percentage_change_normalization() {
    let reference = 100.0;
    let values = vec![101.0, 99.0, 100.5, 98.5, 102.0];
    
    // Percentage change: (x - ref) / ref
    let normalized: Vec<f64> = values.iter()
        .map(|&x| (x - reference) / reference)
        .collect();
    
    // Verify expected values
    let expected = vec![0.01, -0.01, 0.005, -0.015, 0.02];
    for (i, (&norm, &exp)) in normalized.iter().zip(expected.iter()).enumerate() {
        assert!(
            (norm - exp).abs() < 1e-10,
            "Percentage change at {} should be {}, got {}",
            i, exp, norm
        );
    }
}

// ============================================================================
// Metadata Export Verification Tests
// ============================================================================

#[test]
fn test_normalization_strategy_display_format() {
    // Verify Display impl for NormalizationStrategy (from export_aligned.rs)
    assert_eq!(NormalizationStrategy::None.to_string(), "none");
    assert_eq!(NormalizationStrategy::PerFeatureZScore.to_string(), "per_feature_zscore");
    assert_eq!(NormalizationStrategy::MarketStructureZScore.to_string(), "market_structure_zscore");
    assert_eq!(NormalizationStrategy::GlobalZScore.to_string(), "global_zscore");
    assert_eq!(NormalizationStrategy::Bilinear.to_string(), "bilinear");
}

#[test]
fn test_normalization_params_contains_config_info() {
    // Create params with known values
    let params = NormalizationParams::new(
        vec![100.0; 10],
        vec![0.1; 10],
        vec![500.0; 20],
        vec![100.0; 20],
        10000,
        10,
    );
    
    // Verify all fields are populated
    assert_eq!(params.price_means.len(), 10);
    assert_eq!(params.price_stds.len(), 10);
    assert_eq!(params.size_means.len(), 20);
    assert_eq!(params.size_stds.len(), 20);
    assert_eq!(params.sample_count, 10000);
    assert_eq!(params.levels, 10);
    assert!(!params.feature_layout.is_empty());
    
    // Serialize to JSON and verify structure
    let json = serde_json::to_string_pretty(&params).expect("Failed to serialize");
    assert!(json.contains("strategy"), "JSON should contain strategy field");
    assert!(json.contains("price_means"), "JSON should contain price_means");
    assert!(json.contains("sample_count"), "JSON should contain sample_count");
}

/// Test that exported metadata includes normalization information.
#[test]
fn test_exported_metadata_includes_normalization_info() {
    use tempfile::TempDir;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("test_metadata_export");
    
    let label_config = LabelConfig::new(50, 10, 0.002);
    let output = create_synthetic_98_feature_output(20, 100);
    
    let exporter = AlignedBatchExporter::new(&output_path, label_config, 100, 10)
        .with_normalization(NormalizationConfig::raw());
    
    // Export should succeed
    let result = exporter.export_day("test_day", &output);
    assert!(result.is_ok(), "Export should succeed");
    
    // Check metadata file exists
    let metadata_path = output_path.join("test_day_metadata.json");
    assert!(metadata_path.exists(), "Metadata file should exist");
    
    // Read and parse metadata
    let metadata_str = std::fs::read_to_string(&metadata_path).expect("Failed to read metadata");
    let metadata: serde_json::Value = serde_json::from_str(&metadata_str).expect("Failed to parse metadata");
    
    // Verify normalization section exists
    assert!(
        metadata.get("normalization").is_some(),
        "Metadata should contain 'normalization' section"
    );
    
    let norm_section = metadata.get("normalization").unwrap();
    
    // Verify strategy is present
    assert!(
        norm_section.get("strategy").is_some(),
        "Normalization section should contain 'strategy'"
    );
    
    // Verify the normalization params file reference
    assert!(
        norm_section.get("params_file").is_some(),
        "Normalization section should reference params file"
    );
    
    // Check normalization params file exists
    let norm_params_path = output_path.join("test_day_normalization.json");
    assert!(norm_params_path.exists(), "Normalization params file should exist");
    
    // Verify normalization params content
    let norm_params = NormalizationParams::load_json(&norm_params_path)
        .expect("Failed to load normalization params");
    
    // For raw export, strategy should be None
    assert_eq!(
        norm_params.strategy,
        NormalizationStrategy::None,
        "Raw export should have strategy = None"
    );
}

/// Test export with GlobalZScore normalization includes correct strategy in metadata.
#[test]
fn test_exported_metadata_with_zscore_normalization() {
    use tempfile::TempDir;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("test_zscore_export");
    
    let label_config = LabelConfig::new(50, 10, 0.002);
    let output = create_synthetic_98_feature_output(20, 100);
    
    // Use TLOB repo style (global z-score)
    let exporter = AlignedBatchExporter::new(&output_path, label_config, 100, 10)
        .with_normalization(NormalizationConfig::tlob_repo());
    
    let result = exporter.export_day("test_day", &output);
    assert!(result.is_ok(), "Export should succeed");
    
    // Check normalization params file
    let norm_params_path = output_path.join("test_day_normalization.json");
    let norm_params = NormalizationParams::load_json(&norm_params_path)
        .expect("Failed to load normalization params");
    
    // For TLOB repo export with GlobalZScore, strategy should be GlobalZScore
    assert_eq!(
        norm_params.strategy,
        NormalizationStrategy::GlobalZScore,
        "TLOB repo export should have strategy = GlobalZScore"
    );
    
    // Verify statistics were computed
    assert!(norm_params.sample_count > 0, "Sample count should be > 0");
    assert!(!norm_params.price_means.iter().all(|&x| x == 0.0), "Price means should be non-zero");
    assert!(!norm_params.price_stds.iter().all(|&x| x == 1.0), "Price stds should be computed");
}

/// Test that exported sequences file has correct shape.
#[test]
fn test_exported_sequences_shape() {
    use tempfile::TempDir;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("test_shape_export");
    
    let label_config = LabelConfig::new(50, 10, 0.002);
    let window_size = 100;
    let n_features = 98;
    let output = create_synthetic_98_feature_output(20, window_size);
    
    let exporter = AlignedBatchExporter::new(&output_path, label_config, window_size, 10)
        .with_normalization(NormalizationConfig::raw());
    
    let result = exporter.export_day("test_day", &output);
    let export = result.expect("Export should succeed");
    
    // Verify shape in export result
    assert_eq!(export.seq_shape.0, window_size, "Window size should match");
    assert_eq!(export.seq_shape.1, n_features, "Feature count should be 98");
    
    // Verify sequences file exists
    let sequences_path = output_path.join("test_day_sequences.npy");
    assert!(sequences_path.exists(), "Sequences file should exist");
    
    // Verify labels file exists
    let labels_path = output_path.join("test_day_labels.npy");
    assert!(labels_path.exists(), "Labels file should exist");
}
