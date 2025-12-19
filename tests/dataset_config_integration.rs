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

