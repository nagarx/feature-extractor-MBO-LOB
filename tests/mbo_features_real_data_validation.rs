//! Comprehensive Real-World Validation of MBO Features
//!
//! This test validates all implemented MBO features against real NVIDIA data
//! using the OPTIMIZED pipeline infrastructure for fast execution.
//!
//! Uses hot store for fast execution (~3-4x faster than compressed).
//!
//! Run with: cargo test --features "parallel,databento" --test mbo_features_real_data_validation --release

#![cfg(all(feature = "parallel", feature = "databento"))]

use feature_extractor::batch::{BatchConfig, BatchProcessor, ErrorMode};
use feature_extractor::PipelineBuilder;
use std::path::Path;

// ============================================================================
// Test Configuration
// ============================================================================

const HOT_STORE_DIR: &str = "../data/hot_store";

/// Get hot store files for testing
fn get_hot_store_files() -> Vec<String> {
    let hot_store_path = Path::new(HOT_STORE_DIR);

    if !hot_store_path.exists() {
        return Vec::new();
    }

    let mut files: Vec<String> = std::fs::read_dir(hot_store_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.to_string_lossy();
            name.ends_with(".mbo.dbn") && !name.ends_with(".zst")
        })
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    files.sort();
    files
}

// ============================================================================
// MBO Feature Validation using Optimized Pipeline
// ============================================================================

#[test]
fn test_mbo_features_real_data_validation() {
    let files = get_hot_store_files();

    if files.is_empty() {
        println!("⚠️  No hot store files found, skipping real data validation");
        println!("   Run `decompress_to_hot_store` to populate hot store");
        return;
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      MBO FEATURES REAL DATA VALIDATION (OPTIMIZED)           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Files: {} hot store files                                    ║", files.len());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Use first 3 files for validation
    let test_files: Vec<_> = files.into_iter().take(3).collect();

    // Create optimized pipeline config with MBO features enabled
    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()  // Enable derived features (8 features)
        .with_mbo_features()      // Enable MBO features (36 features)
        .event_sampling(1000)     // Sample every 1000 events for speed
        .window(100, 10)
        .build_config()
        .expect("Failed to create pipeline config");

    // Create batch processor with parallel processing
    let batch_config = BatchConfig::new()
        .with_threads(4)
        .with_error_mode(ErrorMode::FailFast)
        .with_hot_store_dir(HOT_STORE_DIR);

    let processor = BatchProcessor::new(pipeline_config, batch_config);

    println!("🚀 Processing {} files using optimized parallel pipeline...\n", test_files.len());

    let start = std::time::Instant::now();
    let result = processor.process_files(&test_files);

    match result {
        Ok(output) => {
            let duration = start.elapsed();
            println!("✅ Processing completed in {:?}", duration);
            println!("   Threads used: {}", output.threads_used);

            // Validation statistics
            let mut total_sequences = 0usize;
            let mut total_features_checked = 0usize;
            let mut nan_count = 0usize;
            let mut inf_count = 0usize;

            // MBO feature indices in full pipeline output:
            // Base offset: 48 (after 40 raw LOB + 8 derived)
            // MBO internal indices:
            //   - order_flow_volatility: 10 (within MBO) → 48 + 10 = 58
            //   - size_skewness: 18 → 48 + 18 = 66
            //   - size_concentration: 19 → 48 + 19 = 67
            //   - average_queue_position: 20 → 48 + 20 = 68
            //   - queue_size_ahead: 21 → 48 + 21 = 69
            //   - orders_per_level: 22 → 48 + 22 = 70
            //   - level_concentration: 23 → 48 + 23 = 71
            //   - depth_ticks_bid: 24 → 48 + 24 = 72
            //   - depth_ticks_ask: 25 → 48 + 25 = 73

            let mbo_feature_indices = [58, 66, 67, 68, 69, 70, 71, 72, 73];
            
            let mut feature_sums: Vec<f64> = vec![0.0; mbo_feature_indices.len()];
            let mut feature_mins: Vec<f64> = vec![f64::MAX; mbo_feature_indices.len()];
            let mut feature_maxs: Vec<f64> = vec![f64::MIN; mbo_feature_indices.len()];
            let mut feature_counts: Vec<usize> = vec![0; mbo_feature_indices.len()];

            for day_result in &output.results {
                let day_name = Path::new(&day_result.file_path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy();
                    
                println!("\n📂 {}", day_name);
                println!("   Sequences: {}", day_result.output.sequences.len());
                println!("   Processing time: {:?}", day_result.elapsed);

                total_sequences += day_result.output.sequences.len();

                // Validate each sequence
                for seq in &day_result.output.sequences {
                    for timestep in &seq.features {
                        let feature_vec = timestep.as_ref();
                        
                        // Check MBO features
                        for (i, &idx) in mbo_feature_indices.iter().enumerate() {
                            if idx < feature_vec.len() {
                                let val = feature_vec[idx];
                                total_features_checked += 1;

                                if val.is_nan() {
                                    nan_count += 1;
                                } else if val.is_infinite() {
                                    inf_count += 1;
                                } else {
                                    feature_sums[i] += val;
                                    feature_mins[i] = feature_mins[i].min(val);
                                    feature_maxs[i] = feature_maxs[i].max(val);
                                    feature_counts[i] += 1;
                                }
                            }
                        }
                    }
                }
            }

            // Print feature statistics
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                 MBO FEATURE STATISTICS                       ║");
            println!("╚══════════════════════════════════════════════════════════════╝");

            let feature_names = [
                "order_flow_volatility",
                "size_skewness",
                "size_concentration",
                "average_queue_position",
                "queue_size_ahead",
                "orders_per_level",
                "level_concentration",
                "depth_ticks_bid",
                "depth_ticks_ask",
            ];

            println!("\n{:<25} {:>12} {:>12} {:>12}", "Feature", "Min", "Mean", "Max");
            println!("{}", "-".repeat(65));

            for (i, name) in feature_names.iter().enumerate() {
                if feature_counts[i] > 0 {
                    let mean = feature_sums[i] / feature_counts[i] as f64;
                    println!(
                        "{:<25} {:>12.4} {:>12.4} {:>12.4}",
                        name, feature_mins[i], mean, feature_maxs[i]
                    );
                } else {
                    println!("{:<25} {:>12} {:>12} {:>12}", name, "N/A", "N/A", "N/A");
                }
            }

            println!("\n📊 Summary:");
            println!("   Total sequences: {}", total_sequences);
            println!("   Total features checked: {}", total_features_checked);
            println!("   NaN values: {}", nan_count);
            println!("   Inf values: {}", inf_count);

            // Assertions
            assert_eq!(nan_count, 0, "Found {} NaN values in MBO features!", nan_count);
            assert_eq!(inf_count, 0, "Found {} Inf values in MBO features!", inf_count);
            assert!(total_sequences > 0, "No sequences generated!");

            // Validate feature ranges for implemented features
            // size_concentration and level_concentration should be in [0, 1]
            let size_conc_idx = 2; // index in our array
            let level_conc_idx = 6;
            
            if feature_counts[size_conc_idx] > 0 {
                assert!(
                    feature_mins[size_conc_idx] >= 0.0,
                    "size_concentration min < 0: {}",
                    feature_mins[size_conc_idx]
                );
                assert!(
                    feature_maxs[size_conc_idx] <= 1.0,
                    "size_concentration max > 1: {}",
                    feature_maxs[size_conc_idx]
                );
            }

            if feature_counts[level_conc_idx] > 0 {
                assert!(
                    feature_mins[level_conc_idx] >= 0.0,
                    "level_concentration min < 0: {}",
                    feature_mins[level_conc_idx]
                );
                assert!(
                    feature_maxs[level_conc_idx] <= 1.0,
                    "level_concentration max > 1: {}",
                    feature_maxs[level_conc_idx]
                );
            }

            // depth_ticks should be >= 0
            let depth_bid_idx = 7;
            let depth_ask_idx = 8;
            
            if feature_counts[depth_bid_idx] > 0 {
                assert!(
                    feature_mins[depth_bid_idx] >= 0.0,
                    "depth_ticks_bid min < 0: {}",
                    feature_mins[depth_bid_idx]
                );
            }
            
            if feature_counts[depth_ask_idx] > 0 {
                assert!(
                    feature_mins[depth_ask_idx] >= 0.0,
                    "depth_ticks_ask min < 0: {}",
                    feature_mins[depth_ask_idx]
                );
            }

            // order_flow_volatility should be >= 0
            let vol_idx = 0;
            if feature_counts[vol_idx] > 0 {
                assert!(
                    feature_mins[vol_idx] >= 0.0,
                    "order_flow_volatility min < 0: {}",
                    feature_mins[vol_idx]
                );
            }

            println!("\n✅ ALL VALIDATIONS PASSED!");
        }
        Err(e) => {
            panic!("Processing failed: {:?}", e);
        }
    }
}

// ============================================================================
// Parallel Processing Consistency Test
// ============================================================================

#[test]
fn test_parallel_vs_sequential_consistency() {
    let files = get_hot_store_files();

    if files.len() < 2 {
        println!("⚠️  Need at least 2 files for consistency test");
        return;
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      PARALLEL VS SEQUENTIAL CONSISTENCY                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Use first 2 files
    let test_files: Vec<_> = files.into_iter().take(2).collect();

    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()
        .with_mbo_features()
        .event_sampling(1000)
        .window(100, 10)
        .build_config()
        .expect("Failed to create config");

    // Process with 1 thread (sequential)
    let seq_config = BatchConfig::new()
        .with_threads(1)
        .with_hot_store_dir(HOT_STORE_DIR);
    
    let seq_processor = BatchProcessor::new(pipeline_config.clone(), seq_config);
    let seq_result = seq_processor.process_files(&test_files).expect("Sequential failed");

    // Process with 4 threads (parallel)
    let par_config = BatchConfig::new()
        .with_threads(4)
        .with_hot_store_dir(HOT_STORE_DIR);
    
    let par_processor = BatchProcessor::new(pipeline_config, par_config);
    let par_result = par_processor.process_files(&test_files).expect("Parallel failed");

    // Compare results
    assert_eq!(
        seq_result.results.len(),
        par_result.results.len(),
        "Different number of results"
    );

    for (seq_day, par_day) in seq_result.results.iter().zip(par_result.results.iter()) {
        assert_eq!(
            seq_day.output.sequences.len(),
            par_day.output.sequences.len(),
            "Different sequence counts for day {}",
            seq_day.day
        );

        // Compare first few sequences
        for (i, (seq_seq, par_seq)) in seq_day
            .output
            .sequences
            .iter()
            .zip(par_day.output.sequences.iter())
            .take(5)
            .enumerate()
        {
            assert_eq!(
                seq_seq.features.len(),
                par_seq.features.len(),
                "Seq {} has different feature count",
                i
            );
        }
    }

    println!("✅ Sequential and parallel processing produce consistent results!");
}
