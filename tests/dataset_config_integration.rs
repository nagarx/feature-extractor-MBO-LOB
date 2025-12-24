//! Integration tests for DatasetConfig with real data
//!
//! Validates that the configuration system properly:
//! 1. Loads and validates configurations
//! 2. Converts to PipelineConfig correctly
//! 3. Processes real data with expected feature counts

#![cfg(feature = "parallel")]

use feature_extractor::batch::{BatchConfig, BatchProcessor};
use feature_extractor::export::{
    DataPathConfig, DatasetConfig, DateRangeConfig, FeatureSetConfig, SymbolConfig,
};
use std::path::Path;
use std::sync::Arc;

/// Test directory path for hot store data
const HOT_STORE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/hot_store"
);

fn hot_store_exists() -> bool {
    Path::new(HOT_STORE_PATH).exists()
}

#[test]
fn test_dataset_config_feature_counts() {
    // Test that feature counts are computed correctly
    let config_40 = FeatureSetConfig::raw_lob();
    assert_eq!(config_40.feature_count(), 40, "Raw LOB should have 40 features");

    let config_84 = FeatureSetConfig::baseline();
    assert_eq!(
        config_84.feature_count(),
        84,
        "Baseline should have 84 features (40 + 8 + 36)"
    );

    let config_98 = FeatureSetConfig::full();
    assert_eq!(
        config_98.feature_count(),
        98,
        "Full should have 98 features (40 + 8 + 36 + 14)"
    );
}

#[test]
fn test_dataset_config_validation() {
    // Test configuration validation
    let valid_config = DatasetConfig::new(
        SymbolConfig::nasdaq("NVDA"),
        DataPathConfig::new("/tmp/input", "/tmp/output"),
        DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
    );

    // Dates validation should work
    assert!(valid_config.dates.validate().is_ok());

    // Features validation should work
    let mut features = FeatureSetConfig::full();
    assert!(features.validate().is_ok());

    // Invalid: signals without derived
    features.include_signals = true;
    features.include_derived = false;
    assert!(features.validate().is_err());
}

#[test]
fn test_dataset_config_to_pipeline_config() {
    // Test conversion to PipelineConfig
    let config = DatasetConfig::new(
        SymbolConfig::nasdaq("NVDA"),
        DataPathConfig::new("/tmp/input", "/tmp/output"),
        DateRangeConfig::from_range("2025-02-03", "2025-02-07"),
    )
    .with_full_features();

    let pipeline_config = config.to_pipeline_config();

    // Verify feature config
    assert!(pipeline_config.features.include_derived);
    assert!(pipeline_config.features.include_mbo);
    assert!(pipeline_config.features.include_signals);
    assert_eq!(pipeline_config.features.feature_count(), 98);

    // Verify sequence config
    assert_eq!(pipeline_config.sequence.window_size, 100);
    assert_eq!(pipeline_config.sequence.stride, 10);
    assert_eq!(pipeline_config.sequence.feature_count, 98);
}

#[test]
fn test_date_range_generation() {
    // Test date range with weekend exclusion
    let dates = DateRangeConfig::from_range("2025-02-03", "2025-02-10");
    let result = dates.get_dates().unwrap();

    // Feb 3-10, 2025: Mon, Tue, Wed, Thu, Fri, [Sat, Sun], Mon
    // Should be 6 days (excluding Sat=8, Sun=9)
    assert_eq!(result.len(), 6);
    assert!(result.contains(&"2025-02-03".to_string())); // Monday
    assert!(result.contains(&"2025-02-07".to_string())); // Friday
    assert!(!result.contains(&"2025-02-08".to_string())); // Saturday excluded
    assert!(!result.contains(&"2025-02-09".to_string())); // Sunday excluded
    assert!(result.contains(&"2025-02-10".to_string())); // Monday
}

#[test]
fn test_symbol_config_filename_patterns() {
    // NASDAQ standard pattern
    let nvda = SymbolConfig::nasdaq("NVDA");
    assert_eq!(
        nvda.filename_for_date("2025-02-03"),
        "xnas-itch-20250203.mbo.dbn.zst"
    );

    // Custom pattern with symbol substitution
    let custom = SymbolConfig::new("AAPL", "XNAS", "{symbol}_{date}.dbn.zst");
    assert_eq!(
        custom.filename_for_date("2025-02-03"),
        "AAPL_20250203.dbn.zst"
    );
}

#[test]
#[ignore = "Requires real data files"]
fn test_98_feature_export_with_real_data() {
    if !hot_store_exists() {
        eprintln!("Skipping: Hot store not found at {}", HOT_STORE_PATH);
        return;
    }

    // Create configuration pointing to hot store
    let config = DatasetConfig::new(
        SymbolConfig {
            name: "NVDA".to_string(),
            exchange: "XNAS".to_string(),
            // Hot store files are decompressed (no .zst)
            filename_pattern: "xnas-itch-{date}.mbo.dbn".to_string(),
            tick_size: 0.01,
        },
        DataPathConfig::new(HOT_STORE_PATH, "/tmp/test_export"),
        DateRangeConfig::from_dates(vec!["2025-02-03".to_string()]),
    )
    .with_full_features();

    // Get pipeline config
    let pipeline_config = Arc::new(config.to_pipeline_config());
    assert_eq!(pipeline_config.features.feature_count(), 98);

    // Create batch processor
    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new((*pipeline_config).clone(), batch_config);

    // Process single file
    let file_path = config.file_path_for_date("2025-02-03");
    if !file_path.exists() {
        eprintln!("Skipping: File not found at {:?}", file_path);
        return;
    }

    let output = processor.process_files(&[file_path.as_path()]).unwrap();

    // Verify 98 features
    for day_result in &output.results {
        let features = day_result.output.to_flat_features();
        if !features.is_empty() {
            assert_eq!(
                features[0].len(),
                98,
                "Each sample should have 98 features"
            );
        }
    }

    println!("✅ Successfully validated 98-feature extraction");
}

#[test]
fn test_split_config() {
    let days: Vec<String> = (1..=20).map(|i| format!("2025-02-{:02}", i)).collect();

    let split = feature_extractor::export::dataset_config::SplitConfig {
        train_ratio: 0.7,
        val_ratio: 0.15,
        test_ratio: 0.15,
    };

    let (train, val, test) = split.split_days(&days);

    // 20 days * 0.7 = 14 train
    // 20 days * 0.15 = 3 val
    // 20 days * 0.15 = 3 test
    assert_eq!(train.len(), 14);
    assert_eq!(val.len(), 3);
    assert_eq!(test.len(), 3);

    // Verify chronological order
    assert_eq!(train[0], "2025-02-01");
    assert_eq!(val[0], "2025-02-15");
    assert_eq!(test[0], "2025-02-18");
}

// ============================================================================
// Threshold Strategy Configuration Tests (Schema 2.2+)
// ============================================================================

use feature_extractor::export::{ExportLabelConfig, ExportThresholdStrategy};
use feature_extractor::labeling::ThresholdStrategy;

/// Test backward compatibility: configs without threshold_strategy should work
#[test]
fn test_threshold_strategy_backward_compatibility() {
    // Legacy TOML format (Schema 2.1)
    let toml_str = r#"
        horizons = [10, 20, 50, 100, 200]
        smoothing_window = 10
        threshold = 0.0008
    "#;

    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();

    // Should use Fixed strategy by default
    assert!(config.threshold_strategy.is_none());
    assert!(config.validate().is_ok());

    // effective_threshold_strategy should return Fixed
    let strategy = config.effective_threshold_strategy();
    match strategy {
        ExportThresholdStrategy::Fixed { value } => {
            assert!((value - 0.0008).abs() < 1e-10);
        }
        _ => panic!("Expected Fixed strategy for legacy config"),
    }
}

/// Test explicit Fixed threshold strategy
#[test]
fn test_threshold_strategy_fixed_from_toml() {
    let toml_str = r#"
        horizons = [10, 20, 50, 100, 200]
        smoothing_window = 10
        threshold = 0.001

        [threshold_strategy]
        type = "fixed"
        value = 0.002
    "#;

    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();

    // Explicit strategy should take precedence
    assert!(config.threshold_strategy.is_some());
    assert!(config.validate().is_ok());

    let strategy = config.effective_threshold_strategy();
    match strategy {
        ExportThresholdStrategy::Fixed { value } => {
            // Should use threshold_strategy.value, not threshold field
            assert!((value - 0.002).abs() < 1e-10);
        }
        _ => panic!("Expected Fixed strategy"),
    }

    // Verify conversion to internal type
    let internal = strategy.to_internal();
    match internal {
        ThresholdStrategy::Fixed(v) => {
            assert!((v - 0.002).abs() < 1e-10);
        }
        _ => panic!("Expected internal Fixed strategy"),
    }
}

/// Test RollingSpread threshold strategy from TOML
#[test]
fn test_threshold_strategy_rolling_spread_from_toml() {
    let toml_str = r#"
        horizons = [10, 20, 50, 100, 200]
        smoothing_window = 10
        threshold = 0.001

        [threshold_strategy]
        type = "rolling_spread"
        window_size = 1000
        multiplier = 1.5
        fallback = 0.0008
    "#;

    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
    assert!(config.validate().is_ok());

    let strategy = config.effective_threshold_strategy();
    match strategy {
        ExportThresholdStrategy::RollingSpread {
            window_size,
            multiplier,
            fallback,
        } => {
            assert_eq!(window_size, 1000);
            assert!((multiplier - 1.5).abs() < 1e-10);
            assert!((fallback - 0.0008).abs() < 1e-10);
        }
        _ => panic!("Expected RollingSpread strategy"),
    }

    // Verify conversion to internal type
    let internal = strategy.to_internal();
    match internal {
        ThresholdStrategy::RollingSpread {
            window_size,
            multiplier,
            fallback,
        } => {
            assert_eq!(window_size, 1000);
            assert!((multiplier - 1.5).abs() < 1e-10);
            assert!((fallback - 0.0008).abs() < 1e-10);
        }
        _ => panic!("Expected internal RollingSpread strategy"),
    }
}

/// Test Quantile threshold strategy from TOML (recommended for balanced classes)
#[test]
fn test_threshold_strategy_quantile_from_toml() {
    let toml_str = r#"
        horizons = [10, 20, 50, 100, 200]
        smoothing_window = 10
        threshold = 0.001

        [threshold_strategy]
        type = "quantile"
        target_proportion = 0.33
        window_size = 5000
        fallback = 0.0008
    "#;

    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
    assert!(config.validate().is_ok());

    let strategy = config.effective_threshold_strategy();
    match strategy {
        ExportThresholdStrategy::Quantile {
            target_proportion,
            window_size,
            fallback,
        } => {
            assert!((target_proportion - 0.33).abs() < 1e-10);
            assert_eq!(window_size, 5000);
            assert!((fallback - 0.0008).abs() < 1e-10);
        }
        _ => panic!("Expected Quantile strategy"),
    }

    // Verify conversion to internal type
    let internal = strategy.to_internal();
    match internal {
        ThresholdStrategy::Quantile {
            target_proportion,
            window_size,
            fallback,
        } => {
            assert!((target_proportion - 0.33).abs() < 1e-10);
            assert_eq!(window_size, 5000);
            assert!((fallback - 0.0008).abs() < 1e-10);
        }
        _ => panic!("Expected internal Quantile strategy"),
    }
}

/// Test validation errors for invalid strategy parameters
#[test]
fn test_threshold_strategy_validation_errors() {
    // Invalid: zero value for Fixed
    let strategy = ExportThresholdStrategy::fixed(0.0);
    assert!(strategy.validate().is_err());

    // Invalid: negative value for Fixed
    let strategy = ExportThresholdStrategy::fixed(-0.001);
    assert!(strategy.validate().is_err());

    // Invalid: too large value for Fixed
    let strategy = ExportThresholdStrategy::fixed(0.15);
    assert!(strategy.validate().is_err());

    // Invalid: zero window_size for RollingSpread
    let strategy = ExportThresholdStrategy::rolling_spread(0, 1.5, 0.001);
    assert!(strategy.validate().is_err());

    // Invalid: negative multiplier for RollingSpread
    let strategy = ExportThresholdStrategy::rolling_spread(100, -1.0, 0.001);
    assert!(strategy.validate().is_err());

    // Invalid: target_proportion > 0.5 for Quantile
    let strategy = ExportThresholdStrategy::quantile(0.6, 1000, 0.001);
    assert!(strategy.validate().is_err());

    // Invalid: target_proportion <= 0 for Quantile
    let strategy = ExportThresholdStrategy::quantile(0.0, 1000, 0.001);
    assert!(strategy.validate().is_err());

    // Invalid: zero window_size for Quantile
    let strategy = ExportThresholdStrategy::quantile(0.33, 0, 0.001);
    assert!(strategy.validate().is_err());
}

/// Test ExportLabelConfig builder methods
#[test]
fn test_export_label_config_builders() {
    // Test balanced() builder
    let config = ExportLabelConfig::balanced(vec![10, 20, 50], 5, 0.33);
    assert!(config.is_multi_horizon());
    assert!(config.threshold_strategy.is_some());

    match &config.threshold_strategy {
        Some(ExportThresholdStrategy::Quantile { target_proportion, .. }) => {
            assert!((target_proportion - 0.33).abs() < 1e-10);
        }
        _ => panic!("Expected Quantile strategy from balanced()"),
    }

    // Test spread_adaptive() builder
    let config = ExportLabelConfig::spread_adaptive(vec![10, 20, 50], 5, 1.5);
    assert!(config.is_multi_horizon());
    assert!(config.threshold_strategy.is_some());

    match &config.threshold_strategy {
        Some(ExportThresholdStrategy::RollingSpread { multiplier, .. }) => {
            assert!((multiplier - 1.5).abs() < 1e-10);
        }
        _ => panic!("Expected RollingSpread strategy from spread_adaptive()"),
    }

    // Test multi_with_strategy() builder
    let config = ExportLabelConfig::multi_with_strategy(
        vec![10, 20, 50],
        5,
        ExportThresholdStrategy::fixed(0.002),
    );
    assert!(config.is_multi_horizon());
    assert!(config.threshold_strategy.is_some());
}

/// Test to_multi_horizon_config() correctly passes through threshold strategy
#[test]
fn test_multi_horizon_config_conversion_with_strategy() {
    // Create config with Quantile strategy
    let config = ExportLabelConfig::balanced(vec![10, 20, 50, 100], 5, 0.33);

    let multi_config = config.to_multi_horizon_config().unwrap();

    // Verify horizons
    assert_eq!(multi_config.horizons(), &[10, 20, 50, 100]);
    assert_eq!(multi_config.smoothing_window, 5);

    // Verify threshold strategy was correctly converted
    match &multi_config.threshold_strategy {
        ThresholdStrategy::Quantile { target_proportion, .. } => {
            assert!((target_proportion - 0.33).abs() < 1e-10);
        }
        _ => panic!("Expected Quantile strategy in MultiHorizonConfig"),
    }
}

/// Test description() method for human-readable output
#[test]
fn test_threshold_strategy_description() {
    let fixed = ExportThresholdStrategy::fixed(0.0008);
    assert!(fixed.description().contains("Fixed"));
    assert!(fixed.description().contains("0.08"));

    let spread = ExportThresholdStrategy::rolling_spread(1000, 1.5, 0.001);
    assert!(spread.description().contains("spread"));
    assert!(spread.description().contains("1.5"));

    let quantile = ExportThresholdStrategy::quantile(0.33, 5000, 0.001);
    assert!(quantile.description().contains("33"));
    assert!(quantile.description().contains("Quantile"));
}

/// Test full TOML config round-trip with threshold strategy
#[test]
fn test_full_config_toml_roundtrip() {
    // This simulates loading nvda_balanced.toml
    let toml_str = r#"
        [experiment]
        name = "NVDA Balanced Classes"
        version = "2.2.0"

        [symbol]
        name = "NVDA"
        exchange = "XNAS"
        filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
        tick_size = 0.01

        [data]
        input_dir = "../data/NVDA_2025-02-01_to_2025-09-30"
        output_dir = "../data/exports/nvda_balanced"
        hot_store_dir = "../data/hot_store"

        [dates]
        start_date = "2025-02-03"
        end_date = "2025-09-29"

        [features]
        lob_levels = 10
        include_derived = true
        include_mbo = true
        include_signals = true

        [sampling]
        strategy = "event_based"
        event_count = 1000

        [sequence]
        window_size = 100
        stride = 10

        [labels]
        horizons = [10, 20, 50, 100, 200]
        smoothing_window = 10
        threshold = 0.0008

        [labels.threshold_strategy]
        type = "quantile"
        target_proportion = 0.33
        window_size = 5000
        fallback = 0.0008

        [split]
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
    "#;

    let config: DatasetConfig = toml::from_str(toml_str).unwrap();

    // Validate entire config
    assert!(config.dates.validate().is_ok());
    assert!(config.features.validate().is_ok());
    assert!(config.labels.validate().is_ok());

    // Verify threshold strategy
    assert!(config.labels.is_multi_horizon());
    let strategy = config.labels.effective_threshold_strategy();
    match strategy {
        ExportThresholdStrategy::Quantile {
            target_proportion,
            window_size,
            ..
        } => {
            assert!((target_proportion - 0.33).abs() < 1e-10);
            assert_eq!(window_size, 5000);
        }
        _ => panic!("Expected Quantile strategy from full config"),
    }

    // Verify description
    let desc = config.labels.description();
    assert!(desc.contains("horizons"));
    assert!(desc.contains("Quantile"));

    println!("✅ Full config with threshold strategy: {}", desc);
}

