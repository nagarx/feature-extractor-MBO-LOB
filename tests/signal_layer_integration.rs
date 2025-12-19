//! Integration test for the signal layer with real NVIDIA data.
//!
//! This test validates the complete signal layer (14 signals, indices 84-97)
//! using real market data through the batch processing pipeline.
//!
//! Run with: cargo test --features "parallel,databento" --test signal_layer_integration --release -- --ignored

#![cfg(feature = "parallel")]

use feature_extractor::batch::{BatchConfig, BatchProcessor};
use feature_extractor::builder::PipelineBuilder;
use feature_extractor::features::signals;
use std::path::Path;

// Expected data directory (hot store - decompressed DBN files)
const HOT_STORE_DIR: &str = "../data/hot_store";

/// Check if test data is available.
fn test_data_available() -> bool {
    Path::new(HOT_STORE_DIR).exists()
}

/// Get test files (first 1 day for quick testing).
fn get_test_files() -> Vec<String> {
    let path = Path::new(HOT_STORE_DIR);
    if !path.exists() {
        return vec![];
    }

    let mut files: Vec<_> = std::fs::read_dir(path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "zst" || ext == "dbn")
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    files.sort();
    files.truncate(1); // Just first day for quick test
    files
}

#[test]
#[ignore] // Run with: cargo test --test signal_layer_integration -- --ignored
fn test_signal_layer_feature_count() {
    if !test_data_available() {
        eprintln!("Skipping test: Hot store data not available at {}", HOT_STORE_DIR);
        return;
    }

    let files = get_test_files();
    if files.is_empty() {
        eprintln!("Skipping test: No data files found");
        return;
    }

    println!("Testing with file: {}", files[0]);

    // Build pipeline with trading signals enabled
    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 1)
        .with_trading_signals() // <-- This enables 98 features
        .build_config()
        .expect("Failed to create pipeline config");

    // Verify feature count
    assert_eq!(
        pipeline_config.features.feature_count(),
        98,
        "Expected 98 features (40 raw + 8 derived + 36 MBO + 14 signals)"
    );

    // Process with batch processor
    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

    let output = processor
        .process_files(&files)
        .expect("Failed to process files");

    println!(
        "Processed {} messages, {} features, {} sequences",
        output.total_messages(),
        output.total_features(),
        output.total_sequences()
    );

    // Verify sequences have 98 features
    for result in output.results.iter() {
        for seq in result.output.sequences.iter() {
            for (feat_idx, feat_vec) in seq.features.iter().enumerate() {
                assert_eq!(
                    feat_vec.len(),
                    98,
                    "Day {} feature {} should have 98 elements, got {}",
                    result.day,
                    feat_idx,
                    feat_vec.len()
                );
            }
        }
    }

    println!("✅ All sequences have 98 features");
}

#[test]
#[ignore] // Run with: cargo test --test signal_layer_integration -- --ignored
fn test_signal_values_in_expected_ranges() {
    if !test_data_available() {
        eprintln!("Skipping test: Hot store data not available at {}", HOT_STORE_DIR);
        return;
    }

    let files = get_test_files();
    if files.is_empty() {
        eprintln!("Skipping test: No data files found");
        return;
    }

    println!("Testing signal value ranges with file: {}", files[0]);

    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 1)
        .with_trading_signals()
        .build_config()
        .expect("Failed to create pipeline config");

    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

    let output = processor
        .process_files(&files)
        .expect("Failed to process files");

    // Track signal statistics
    let mut signal_stats = vec![(f64::MAX, f64::MIN, 0.0, 0u64); signals::SIGNAL_COUNT];

    // Collect statistics from all feature vectors
    for result in output.results.iter() {
        for seq in result.output.sequences.iter() {
            for feat_vec in seq.features.iter() {
                if feat_vec.len() >= 98 {
                    // Check signals (indices 84-97)
                    for (i, &val) in feat_vec[84..98].iter().enumerate() {
                        let (min, max, sum, count) = &mut signal_stats[i];
                        if val.is_finite() {
                            if val < *min {
                                *min = val;
                            }
                            if val > *max {
                                *max = val;
                            }
                            *sum += val;
                            *count += 1;
                        }
                    }
                }
            }
        }
    }

    // Validate signal ranges
    println!("\nSignal Statistics:");
    println!("{:<25} {:>12} {:>12} {:>12}", "Signal", "Min", "Max", "Mean");
    println!("{}", "-".repeat(65));

    let signal_names = [
        "true_ofi",
        "depth_norm_ofi",
        "executed_pressure",
        "signed_mp_delta_bps",
        "trade_asymmetry",
        "cancel_asymmetry",
        "fragility_score",
        "depth_asymmetry",
        "book_valid",
        "time_regime",
        "mbo_ready",
        "dt_seconds",
        "invalidity_delta",
        "schema_version",
    ];

    for (i, name) in signal_names.iter().enumerate() {
        let (min, max, sum, count) = signal_stats[i];
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        println!("{:<25} {:>12.4} {:>12.4} {:>12.4}", name, min, max, mean);

        // Validate specific signal ranges
        match i {
            4 => {
                // trade_asymmetry should be in [-1, 1]
                assert!(
                    min >= -1.0 - 0.001 && max <= 1.0 + 0.001,
                    "trade_asymmetry out of range: [{}, {}]",
                    min,
                    max
                );
            }
            5 => {
                // cancel_asymmetry should be in [-1, 1]
                assert!(
                    min >= -1.0 - 0.001 && max <= 1.0 + 0.001,
                    "cancel_asymmetry out of range: [{}, {}]",
                    min,
                    max
                );
            }
            7 => {
                // depth_asymmetry should be in [-1, 1]
                assert!(
                    min >= -1.0 - 0.001 && max <= 1.0 + 0.001,
                    "depth_asymmetry out of range: [{}, {}]",
                    min,
                    max
                );
            }
            8 => {
                // book_valid should be 0 or 1
                assert!(
                    min >= 0.0 && max <= 1.0,
                    "book_valid out of range: [{}, {}]",
                    min,
                    max
                );
            }
            9 => {
                // time_regime should be 0-4
                assert!(
                    min >= 0.0 && max <= 4.0,
                    "time_regime out of range: [{}, {}]",
                    min,
                    max
                );
            }
            10 => {
                // mbo_ready should be 0 or 1
                assert!(
                    min >= 0.0 && max <= 1.0,
                    "mbo_ready out of range: [{}, {}]",
                    min,
                    max
                );
            }
            11 => {
                // dt_seconds should be non-negative
                assert!(min >= 0.0, "dt_seconds negative: {}", min);
            }
            12 => {
                // invalidity_delta should be non-negative
                assert!(min >= 0.0, "invalidity_delta negative: {}", min);
            }
            13 => {
                // schema_version should be 2.0
                assert!(
                    (min - 2.0).abs() < 0.001 && (max - 2.0).abs() < 0.001,
                    "schema_version should be 2.0: [{}, {}]",
                    min,
                    max
                );
            }
            _ => {}
        }
    }

    println!("\n✅ All signal values within expected ranges");
}

#[test]
#[ignore] // Run with: cargo test --test signal_layer_integration -- --ignored
fn test_ofi_warmup_behavior() {
    if !test_data_available() {
        eprintln!("Skipping test: Hot store data not available at {}", HOT_STORE_DIR);
        return;
    }

    let files = get_test_files();
    if files.is_empty() {
        eprintln!("Skipping test: No data files found");
        return;
    }

    println!("Testing OFI warmup behavior with file: {}", files[0]);

    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(100) // Sample more frequently to see warmup
        .window(50, 1)
        .with_trading_signals()
        .build_config()
        .expect("Failed to create pipeline config");

    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

    let output = processor
        .process_files(&files)
        .expect("Failed to process files");

    // Track mbo_ready transitions
    let mut cold_samples = 0u64;
    let mut warm_samples = 0u64;

    for result in output.results.iter() {
        for seq in result.output.sequences.iter() {
            for feat_vec in seq.features.iter() {
                if feat_vec.len() >= 98 {
                    let mbo_ready = feat_vec[94]; // Index 94 = mbo_ready
                    if mbo_ready < 0.5 {
                        cold_samples += 1;
                    } else {
                        warm_samples += 1;
                    }
                }
            }
        }
    }

    println!("Cold samples (mbo_ready=0): {}", cold_samples);
    println!("Warm samples (mbo_ready=1): {}", warm_samples);

    // In a typical day, most samples should be warm
    // The first ~100 state changes are cold
    assert!(
        warm_samples > cold_samples,
        "Expected more warm samples than cold: {} warm, {} cold",
        warm_samples,
        cold_samples
    );

    println!("✅ OFI warmup behavior verified");
}

#[test]
#[ignore] // Run with: cargo test --test signal_layer_integration -- --ignored
fn test_signals_are_finite() {
    if !test_data_available() {
        eprintln!("Skipping test: Hot store data not available at {}", HOT_STORE_DIR);
        return;
    }

    let files = get_test_files();
    if files.is_empty() {
        eprintln!("Skipping test: No data files found");
        return;
    }

    println!("Testing signal finiteness with file: {}", files[0]);

    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 1)
        .with_trading_signals()
        .build_config()
        .expect("Failed to create pipeline config");

    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

    let output = processor
        .process_files(&files)
        .expect("Failed to process files");

    let mut total_signals = 0u64;
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;

    for result in output.results.iter() {
        for seq in result.output.sequences.iter() {
            for feat_vec in seq.features.iter() {
                if feat_vec.len() >= 98 {
                    for (i, &val) in feat_vec[84..98].iter().enumerate() {
                        total_signals += 1;
                        if val.is_nan() {
                            nan_count += 1;
                            eprintln!("NaN found in signal {} (index {})", i, 84 + i);
                        }
                        if val.is_infinite() {
                            inf_count += 1;
                            eprintln!("Inf found in signal {} (index {})", i, 84 + i);
                        }
                    }
                }
            }
        }
    }

    println!("Total signal values checked: {}", total_signals);
    println!("NaN count: {}", nan_count);
    println!("Inf count: {}", inf_count);

    assert_eq!(nan_count, 0, "Found {} NaN values in signals", nan_count);
    assert_eq!(inf_count, 0, "Found {} Inf values in signals", inf_count);

    println!("✅ All signal values are finite");
}
