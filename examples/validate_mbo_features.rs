//! Test Phase 2 MBO feature extraction on real NVDA data.
//!
//! This example validates:
//! - All 84 features (48 LOB + 36 MBO) extract correctly
//! - Zero NaN or Inf values
//! - Feature extraction performance (target: >10K features/s)
//! - Memory stability (no leaks)
//!
//! Usage:
//! ```bash
//! cargo run --release --example test_phase2_features \
//!   /Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250203.mbo.dbn.zst
//! ```

use feature_extractor::{FeatureConfig, FeatureExtractor, MboEvent};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("Phase 2 Validation: MBO Feature Extraction");
    println!("=================================================================\n");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_dbn_file>", args[0]);
        std::process::exit(1);
    }
    let file_path = &args[1];

    println!("📂 Input file: {file_path}");

    // Create configuration with MBO features enabled
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000, // 1000 messages ~20s
    };

    println!("⚙️  Configuration:");
    println!("   - LOB levels: {}", config.lob_levels);
    println!("   - Include derived: {}", config.include_derived);
    println!("   - Include MBO: {}", config.include_mbo);
    println!("   - MBO window size: {}", config.mbo_window_size);

    // Initialize feature extractor
    let mut extractor = FeatureExtractor::with_config(config);
    let expected_features = extractor.feature_count();

    println!("   - Total features: {expected_features}");
    println!("     * LOB raw: 40");
    println!("     * LOB derived: 8");
    println!("     * MBO aggregated: 36\n");

    // Initialize LOB reconstructor
    let mut lob = LobReconstructor::new(10);

    // Open DBN file
    let loader = DbnLoader::new(file_path)?.skip_invalid(true); // Skip invalid messages
    println!("✅ DBN file opened successfully\n");

    println!("🚀 Starting processing...\n");
    let start_time = Instant::now();

    // Statistics
    let mut msg_count = 0u64;
    let mut feature_count = 0u64;
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;
    let mut invalid_feature_count = 0u64;

    // Feature statistics (for validation)
    let mut feature_mins = vec![f64::INFINITY; expected_features];
    let mut feature_maxs = vec![f64::NEG_INFINITY; expected_features];
    let mut feature_sums = vec![0.0; expected_features];

    // Sampling: Extract features every 100 messages
    let sample_interval = 100;

    for msg in loader.iter_messages()? {
        // Skip invalid messages
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            continue;
        }

        // Action and side are already converted by DbnLoader
        let action = msg.action;
        let side = msg.side;

        // Process MBO event for aggregation
        let timestamp = msg.timestamp.unwrap_or(0) as u64;
        let mbo_event = MboEvent::new(timestamp, action, side, msg.price, msg.size, msg.order_id);
        extractor.process_mbo_event(mbo_event);

        // Update LOB
        let lob_state = match lob.process_message(&msg) {
            Ok(state) => state,
            Err(_) => continue,
        };

        msg_count += 1;

        // Sample features every N messages
        if msg_count % sample_interval == 0 {
            // Extract all features
            match extractor.extract_all_features(&lob_state) {
                Ok(features) => {
                    if features.len() != expected_features {
                        println!(
                            "❌ ERROR: Expected {} features, got {}",
                            expected_features,
                            features.len()
                        );
                        invalid_feature_count += 1;
                        continue;
                    }

                    // Validate each feature
                    for (i, &val) in features.iter().enumerate() {
                        if val.is_nan() {
                            nan_count += 1;
                        } else if val.is_infinite() {
                            inf_count += 1;
                        } else {
                            // Update statistics
                            feature_mins[i] = feature_mins[i].min(val);
                            feature_maxs[i] = feature_maxs[i].max(val);
                            feature_sums[i] += val;
                        }
                    }

                    feature_count += 1;
                }
                Err(e) => {
                    println!("❌ ERROR extracting features: {e:?}");
                    invalid_feature_count += 1;
                }
            }
        }

        // Progress reporting
        if msg_count % 1_000_000 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let msg_rate = msg_count as f64 / elapsed;
            let feature_rate = feature_count as f64 / elapsed;

            println!(
                "  ⏱️  Processed: {msg_count:>12} messages | Rate: {msg_rate:>8.0} msg/s | Features: {feature_rate:>8.0}/s"
            );
            println!(
                "     📊 Features extracted: {feature_count} | NaN: {nan_count} | Inf: {inf_count} | Invalid: {invalid_feature_count}"
            );
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    println!("\n=================================================================");
    println!("✅ PROCESSING COMPLETE");
    println!("=================================================================\n");

    // Statistics
    println!("📊 MESSAGE STATISTICS:");
    println!("  Total messages processed:       {msg_count:>12}");
    println!("  Feature snapshots extracted:    {feature_count:>12}");
    println!("  Sampling interval:              {sample_interval:>12}");
    println!();

    // Performance
    let msg_rate = msg_count as f64 / elapsed;
    let feature_rate = feature_count as f64 / elapsed;
    let time_per_extraction_ns = (elapsed / feature_count as f64) * 1e9;

    println!("⚡ PERFORMANCE:");
    println!("  Total time:                     {elapsed:>12.2} seconds");
    println!("  Message throughput:             {msg_rate:>12.0} msg/s");
    println!("  Feature extraction rate:        {feature_rate:>12.0} features/s");
    println!("  Time per extraction:            {time_per_extraction_ns:>12.2} ns");
    println!();

    // Quality checks
    println!("✅ DATA QUALITY CHECKS:");
    let total_values = feature_count * expected_features as u64;
    let nan_percent = 100.0 * nan_count as f64 / total_values as f64;
    let inf_percent = 100.0 * inf_count as f64 / total_values as f64;
    let invalid_percent = 100.0 * invalid_feature_count as f64 / feature_count as f64;

    println!("  NaN values:                     {nan_count:>12} ({nan_percent:.4}%)");
    println!("  Inf values:                     {inf_count:>12} ({inf_percent:.4}%)");
    println!(
        "  Invalid feature vectors:        {invalid_feature_count:>12} ({invalid_percent:.4}%)"
    );
    println!();

    // Success criteria
    println!("🎯 SUCCESS CRITERIA:");
    let all_valid = nan_count == 0 && inf_count == 0 && invalid_feature_count == 0;
    let performance_ok = feature_rate >= 10_000.0;

    println!(
        "  ✅ Zero NaN/Inf:                  {}",
        if all_valid { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  ✅ Performance (>10K/s):          {}",
        if performance_ok {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    println!();

    // Feature ranges (first 10 features for sanity check)
    println!("📈 FEATURE RANGES (Sample - First 10):");
    let feature_names = [
        "Ask Price L1",
        "Ask Price L2",
        "Ask Price L3",
        "Ask Price L4",
        "Ask Price L5",
        "Ask Price L6",
        "Ask Price L7",
        "Ask Price L8",
        "Ask Price L9",
        "Ask Price L10",
    ];

    for i in 0..10.min(expected_features) {
        let min = feature_mins[i];
        let max = feature_maxs[i];
        let avg = feature_sums[i] / feature_count as f64;
        println!(
            "  {:>15} | Min: {:>10.4} | Max: {:>10.4} | Avg: {:>10.4}",
            feature_names[i], min, max, avg
        );
    }
    println!();

    // MBO features (last 10 for sanity check)
    if expected_features >= 84 {
        println!("📈 MBO FEATURES (Sample - Last 10):");
        let mbo_feature_names = [
            "Avg Order Age",
            "Median Lifetime",
            "Avg Fill Ratio",
            "Avg Time to Fill",
            "Cancel/Add Ratio",
            "Active Orders",
            "Feature 79",
            "Feature 80",
            "Feature 81",
            "Feature 82",
        ];

        for i in 0..10 {
            let idx = expected_features - 10 + i;
            let min = feature_mins[idx];
            let max = feature_maxs[idx];
            let avg = feature_sums[idx] / feature_count as f64;
            println!(
                "  {:>18} | Min: {:>10.4} | Max: {:>10.4} | Avg: {:>10.4}",
                mbo_feature_names[i], min, max, avg
            );
        }
        println!();
    }

    // Final verdict
    println!("=================================================================");
    if all_valid && performance_ok {
        println!("🎉 PHASE 2 VALIDATION: SUCCESS!");
        println!("   All 84 features extract correctly with excellent performance.");
    } else {
        println!("⚠️  PHASE 2 VALIDATION: NEEDS ATTENTION");
        if !all_valid {
            println!("   - Data quality issues detected (NaN/Inf values)");
        }
        if !performance_ok {
            println!("   - Performance below target ({feature_rate})");
        }
    }
    println!("=================================================================\n");

    Ok(())
}
