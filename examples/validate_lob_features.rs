//! Test Phase 1: LOB Feature Extraction on Real NVDA Data
//!
//! Usage: cargo run --release --example test_phase1_features <path_to_dbn_file>

use feature_extractor::FeatureExtractor;
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=================================================================");
    println!("Phase 1 Test: LOB Feature Extraction");
    println!("=================================================================\n");

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_dbn_file.dbn.zst>", args[0]);
        std::process::exit(1);
    }

    let dbn_path = &args[1];
    println!("📂 Input: {dbn_path}");

    // Initialize components
    let loader = DbnLoader::new(dbn_path)?.skip_invalid(true);
    let mut lob = LobReconstructor::new(10);
    let extractor = FeatureExtractor::new(10);

    println!(
        "✅ Feature extractor: {} features per snapshot\n",
        extractor.feature_count()
    );

    // Statistics
    let mut message_count = 0u64;
    let mut feature_extractions = 0u64;
    let mut skipped_invalid = 0u64;
    let mut lob_errors = 0u64;

    // Feature validation
    let mut min_features = vec![f64::INFINITY; 48];
    let mut max_features = vec![f64::NEG_INFINITY; 48];
    let mut sum_features = vec![0.0; 48];
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;

    // Sample features for inspection
    let sample_points = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000];

    let start_time = Instant::now();
    println!("🚀 Processing...\n");

    for mbo_msg in loader.iter_messages()? {
        // Skip invalid
        if mbo_msg.order_id == 0 || mbo_msg.size == 0 || mbo_msg.price <= 0 {
            skipped_invalid += 1;
            continue;
        }

        // Update LOB
        match lob.process_message(&mbo_msg) {
            Ok(_) => {}
            Err(e) => {
                lob_errors += 1;
                if lob_errors <= 10 {
                    log::warn!("LOB error at msg {message_count}: {e}");
                }
                continue;
            }
        }

        message_count += 1;

        // Extract features periodically (every 100 messages)
        if message_count % 100 == 0 {
            let state = lob.get_lob_state();

            // Only extract if LOB has valid state
            if state.is_valid() {
                match extractor.extract_lob_features(&state) {
                    Ok(features) => {
                        feature_extractions += 1;

                        // Validate and track statistics
                        for (i, &f) in features.iter().enumerate() {
                            if f.is_nan() {
                                nan_count += 1;
                            } else if f.is_infinite() {
                                inf_count += 1;
                            } else {
                                min_features[i] = min_features[i].min(f);
                                max_features[i] = max_features[i].max(f);
                                sum_features[i] += f;
                            }
                        }

                        // Print sample features
                        if sample_points.contains(&message_count) {
                            println!("📊 Sample at message {message_count}:");
                            println!("   Mid-price:   ${:.4}", features[40]);
                            println!("   Spread:      ${:.4}", features[41]);
                            println!("   Spread(bps): {:.2}", features[42]);
                            println!("   Bid volume:  {:.0}", features[43]);
                            println!("   Ask volume:  {:.0}", features[44]);
                            println!("   Imbalance:   {:.4}", features[45]);
                            println!();
                        }
                    }
                    Err(e) => {
                        log::error!("Feature extraction error: {e}");
                    }
                }
            }
        }

        // Progress every 1M messages
        if message_count % 1_000_000 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = message_count as f64 / elapsed;
            println!(
                "⏱️  {message_count} msgs | {rate:.0} msg/s | {feature_extractions} features extracted"
            );
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();

    println!("\n=================================================================");
    println!("✅ TEST COMPLETE");
    println!("=================================================================\n");

    // Statistics
    println!("📊 MESSAGE STATISTICS:");
    println!("  Valid messages:        {message_count:>15}");
    println!("  Skipped (invalid):     {skipped_invalid:>15}");
    println!("  LOB errors:            {lob_errors:>15}");
    println!();

    println!("🔍 FEATURE EXTRACTION:");
    println!("  Feature snapshots:     {feature_extractions:>15}");
    println!("  Features per snapshot: {:>15}", extractor.feature_count());
    println!("  NaN values detected:   {nan_count:>15}");
    println!("  Inf values detected:   {inf_count:>15}");
    println!();

    // Calculate averages
    let avg_features: Vec<f64> = sum_features
        .iter()
        .map(|&s| s / feature_extractions as f64)
        .collect();

    println!("📈 FEATURE RANGES (first 10 features):");
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
    for i in 0..10 {
        if max_features[i] > min_features[i] {
            println!(
                "  {:<12}: [{:>8.2}, {:>8.2}] avg={:>8.2}",
                feature_names[i], min_features[i], max_features[i], avg_features[i]
            );
        }
    }
    println!();

    println!("📈 DERIVED FEATURES:");
    let derived_names = [
        "Mid-price",
        "Spread",
        "Spread(bps)",
        "Bid Volume",
        "Ask Volume",
        "Imbalance",
        "Weighted Mid",
        "Price Impact",
    ];
    for i in 0..8 {
        let idx = 40 + i;
        if max_features[idx] > min_features[idx] {
            println!(
                "  {:<13}: [{:>10.4}, {:>10.4}] avg={:>10.4}",
                derived_names[i], min_features[idx], max_features[idx], avg_features[idx]
            );
        }
    }
    println!();

    println!("⚡ PERFORMANCE:");
    println!("  Total time:            {total_time:>12.2} seconds");
    println!(
        "  Message rate:          {:>12.0} msg/s",
        message_count as f64 / total_time
    );
    println!(
        "  Feature rate:          {:>12.0} features/s",
        feature_extractions as f64 / total_time
    );
    println!(
        "  Time per extraction:   {:>12.2} μs",
        (total_time * 1_000_000.0) / feature_extractions as f64
    );
    println!();

    // Validation checks
    println!("✅ VALIDATION CHECKS:");
    let mut all_checks_passed = true;

    // Check 1: No NaN or Inf values
    if nan_count > 0 {
        println!("  ❌ FAIL: {nan_count} NaN values detected");
        all_checks_passed = false;
    } else {
        println!("  ✅ PASS: No NaN values");
    }

    if inf_count > 0 {
        println!("  ❌ FAIL: {inf_count} Inf values detected");
        all_checks_passed = false;
    } else {
        println!("  ✅ PASS: No Inf values");
    }

    // Check 2: Reasonable feature ranges
    let mid_price_ok = min_features[40] > 50.0 && max_features[40] < 200.0;
    if mid_price_ok {
        println!("  ✅ PASS: Mid-price in reasonable range");
    } else {
        println!(
            "  ❌ FAIL: Mid-price out of range [{:.2}, {:.2}]",
            min_features[40], max_features[40]
        );
        all_checks_passed = false;
    }

    let spread_ok = min_features[41] > 0.0 && max_features[41] < 5.0;
    if spread_ok {
        println!("  ✅ PASS: Spread in reasonable range");
    } else {
        println!(
            "  ❌ FAIL: Spread out of range [{:.4}, {:.4}]",
            min_features[41], max_features[41]
        );
        all_checks_passed = false;
    }

    let imbalance_ok = min_features[45] >= -1.0 && max_features[45] <= 1.0;
    if imbalance_ok {
        println!("  ✅ PASS: Volume imbalance in [-1, 1]");
    } else {
        println!(
            "  ❌ FAIL: Imbalance out of range [{:.4}, {:.4}]",
            min_features[45], max_features[45]
        );
        all_checks_passed = false;
    }

    // Check 3: Feature extraction rate
    let extraction_rate = feature_extractions as f64 / total_time;
    if extraction_rate > 1000.0 {
        println!("  ✅ PASS: High extraction rate ({extraction_rate:.0} features/s)");
    } else {
        println!("  ⚠️  WARN: Low extraction rate ({extraction_rate:.0} features/s)");
    }

    println!();

    if all_checks_passed {
        println!("🎉 ALL VALIDATION CHECKS PASSED!");
        println!("✅ Phase 1 is working correctly and ready for Phase 2");
    } else {
        println!("⚠️  SOME CHECKS FAILED - Review output above");
    }

    println!("\n=================================================================\n");

    Ok(())
}
