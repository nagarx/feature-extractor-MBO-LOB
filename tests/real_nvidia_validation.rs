//! Comprehensive Real-World NVIDIA Data Validation
//!
//! This test suite validates the optimized pipeline against real NVIDIA MBO data.
//! It exposes subtle issues that unit tests might miss:
//! - Numerical stability across millions of messages
//! - State isolation between days (no leakage)
//! - Feature value validity and ranges
//! - Sequence generation correctness
//! - Memory stability during long runs

use feature_extractor::{Pipeline, PipelineConfig, SamplingConfig, SamplingStrategy};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor, LobState};
use std::path::Path;

const DATA_DIR: &str = "../data/NVDA_2025-02-01_to_2025-09-30";

fn get_test_files() -> Vec<String> {
    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        return Vec::new();
    }

    let mut files: Vec<String> = std::fs::read_dir(data_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "zst"))
        .filter(|p| {
            p.to_string_lossy()
                .contains(".mbo.dbn.zst")
        })
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    files.sort();
    files
}

fn data_available() -> bool {
    !get_test_files().is_empty()
}

// ============================================================================
// Test 1: Multi-Day Processing with State Isolation
// ============================================================================

#[test]
fn test_multi_day_state_isolation() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_files: Vec<_> = files.iter().take(3).collect(); // Test first 3 days

    println!("\n📊 Multi-Day State Isolation Test");
    println!("   Testing {} days for state leakage...\n", test_files.len());

    // Create pipeline with MBO features for more sensitive state testing
    let mut config = PipelineConfig::default();
    config.features.include_derived = true;
    config.features.include_mbo = true;
    config.sequence.feature_count = config.features.feature_count();
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        event_count: Some(500), // Sample every 500 events
        ..Default::default()
    });

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");

    let mut day_results: Vec<(String, f64, f64, usize)> = Vec::new(); // (filename, first_mid, last_mid, count)

    for (i, file) in test_files.iter().enumerate() {
        // Critical: Reset between days
        pipeline.reset();

        let filename = Path::new(file)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        println!("   Day {}: {}", i + 1, filename);

        let output = match pipeline.process(file) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("   ⚠️  Error processing {}: {}", filename, e);
                continue;
            }
        };

        // Collect statistics
        let first_mid = output.mid_prices.first().copied().unwrap_or(0.0);
        let last_mid = output.mid_prices.last().copied().unwrap_or(0.0);

        println!(
            "      Messages: {}, Features: {}, Sequences: {}",
            output.messages_processed, output.features_extracted, output.sequences_generated
        );
        println!(
            "      First mid: ${:.4}, Last mid: ${:.4}",
            first_mid, last_mid
        );

        day_results.push((filename, first_mid, last_mid, output.features_extracted));
    }

    // Validate state isolation: Each day should start fresh
    // NVIDIA price should be in reasonable range ($100-$200 for 2025)
    println!("\n   📈 Validation:");
    for (filename, first_mid, last_mid, _) in &day_results {
        assert!(
            *first_mid > 50.0 && *first_mid < 300.0,
            "Day {} first mid-price ${:.2} out of expected range - possible state leak",
            filename,
            first_mid
        );
        assert!(
            *last_mid > 50.0 && *last_mid < 300.0,
            "Day {} last mid-price ${:.2} out of expected range - possible state leak",
            filename,
            last_mid
        );
        println!(
            "   ✓ {} prices in valid range [${:.2} - ${:.2}]",
            filename, first_mid, last_mid
        );
    }

    println!("\n   ✅ Multi-day state isolation PASSED");
}

// ============================================================================
// Test 2: Feature Value Validation (No NaN/Inf, Valid Ranges)
// ============================================================================

#[test]
fn test_feature_value_validity() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_file = &files[0];

    println!("\n📊 Feature Value Validity Test");
    println!("   File: {}\n", test_file);

    let mut config = PipelineConfig::default();
    config.features.include_derived = true;
    config.features.include_mbo = false; // Test LOB + derived first
    config.sequence.feature_count = config.features.feature_count();
    config.sequence.window_size = 50;
    config.sequence.stride = 10;
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        event_count: Some(200),
        ..Default::default()
    });

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");
    let output = pipeline.process(test_file).expect("Failed to process");

    println!(
        "   Processed {} messages, extracted {} features",
        output.messages_processed, output.features_extracted
    );
    println!(
        "   Generated {} sequences (window={}, stride={})",
        output.sequences_generated, output.window_size, output.stride
    );

    // Validate all feature values
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut negative_price_count = 0;
    let mut total_values = 0;

    for seq in &output.sequences {
        for snapshot in &seq.features {
            for (i, &value) in snapshot.iter().enumerate() {
                total_values += 1;

                if value.is_nan() {
                    nan_count += 1;
                    println!("   ⚠️  NaN found at feature index {}", i);
                }
                if value.is_infinite() {
                    inf_count += 1;
                    println!("   ⚠️  Inf found at feature index {}", i);
                }
                // Price features (first 20 for 10 levels bid+ask) should be positive
                if i < 20 && value < 0.0 && value != 0.0 {
                    negative_price_count += 1;
                }
            }
        }
    }

    println!("\n   📈 Feature Value Statistics:");
    println!("      Total values checked: {}", total_values);
    println!("      NaN values: {}", nan_count);
    println!("      Inf values: {}", inf_count);
    println!("      Negative prices: {}", negative_price_count);

    assert_eq!(nan_count, 0, "Found {} NaN values in features", nan_count);
    assert_eq!(inf_count, 0, "Found {} Inf values in features", inf_count);
    assert_eq!(
        negative_price_count, 0,
        "Found {} negative price values",
        negative_price_count
    );

    println!("\n   ✅ Feature value validity PASSED");
}

// ============================================================================
// Test 3: LOB Reconstruction Numerical Precision
// ============================================================================

#[test]
fn test_lob_numerical_precision() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_file = &files[0];

    println!("\n📊 LOB Numerical Precision Test");
    println!("   File: {}\n", test_file);

    let loader = DbnLoader::new(test_file).expect("Failed to create loader");
    let mut lob = LobReconstructor::new(10);
    let mut lob_state = LobState::new(10);

    let mut msg_count = 0;
    let mut valid_states = 0;
    let mut crossed_count = 0;
    let mut spread_sum = 0.0;
    let mut min_spread = f64::MAX;
    let mut max_spread = 0.0f64;

    // Sample prices for precision validation
    let mut sample_mids: Vec<f64> = Vec::new();

    for msg in loader.iter_messages().expect("Failed to iterate") {
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            continue;
        }

        // Use the zero-allocation API we optimized
        if let Err(e) = lob.process_message_into(&msg, &mut lob_state) {
            eprintln!("   ⚠️  Error at msg {}: {}", msg_count, e);
            continue;
        }

        msg_count += 1;

        if let (Some(bid), Some(ask)) = (lob_state.best_bid, lob_state.best_ask) {
            if bid > ask {
                crossed_count += 1;
            } else {
                valid_states += 1;
                let spread = (ask - bid) as f64 / 1e9;
                spread_sum += spread;
                min_spread = min_spread.min(spread);
                max_spread = max_spread.max(spread);

                // Sample mid-prices for precision check
                if msg_count % 10000 == 0 {
                    if let Some(mid) = lob_state.mid_price() {
                        sample_mids.push(mid);
                    }
                }
            }
        }

        // Limit for test speed
        if msg_count >= 1_000_000 {
            break;
        }
    }

    let avg_spread = if valid_states > 0 {
        spread_sum / valid_states as f64
    } else {
        0.0
    };

    println!("   📈 LOB Statistics:");
    println!("      Messages processed: {}", msg_count);
    println!("      Valid book states: {}", valid_states);
    println!("      Crossed quotes: {}", crossed_count);
    println!("      Avg spread: ${:.6}", avg_spread);
    println!("      Min spread: ${:.6}", min_spread);
    println!("      Max spread: ${:.6}", max_spread);

    // Validate precision: spreads should be reasonable for NVDA
    assert!(
        avg_spread > 0.001 && avg_spread < 1.0,
        "Average spread ${:.6} out of expected range for NVDA",
        avg_spread
    );
    assert!(
        min_spread >= 0.0,
        "Min spread ${:.6} should not be negative",
        min_spread
    );
    let crossed_ratio = crossed_count as f64 / msg_count as f64;
    assert!(
        crossed_ratio < 0.01,
        "Too many crossed quotes: {} / {} = {:.4}",
        crossed_count,
        msg_count,
        crossed_ratio
    );

    // Validate mid-price precision: check for floating point drift
    if sample_mids.len() >= 2 {
        let price_range = sample_mids.iter().cloned().fold(f64::MAX, f64::min)
            ..=sample_mids.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "      Mid-price range: ${:.4} - ${:.4}",
            price_range.start(),
            price_range.end()
        );

        // NVDA should be in $100-$200 range for 2025
        assert!(
            *price_range.start() > 50.0 && *price_range.end() < 300.0,
            "Mid-prices out of expected range"
        );
    }

    println!("\n   ✅ LOB numerical precision PASSED");
}

// ============================================================================
// Test 4: Sequence Timestamp Ordering
// ============================================================================

#[test]
fn test_sequence_timestamp_ordering() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_file = &files[0];

    println!("\n📊 Sequence Timestamp Ordering Test");
    println!("   File: {}\n", test_file);

    let mut config = PipelineConfig::default();
    config.features.include_derived = false;
    config.features.include_mbo = false;
    config.sequence.feature_count = 40;
    config.sequence.window_size = 100;
    config.sequence.stride = 10;
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        event_count: Some(100),
        ..Default::default()
    });

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");
    let output = pipeline.process(test_file).expect("Failed to process");

    println!(
        "   Generated {} sequences",
        output.sequences_generated
    );

    let mut violations = 0;
    let mut prev_end_ts = 0u64;

    for (i, seq) in output.sequences.iter().enumerate() {
        // Within-sequence: timestamps should be monotonic
        if seq.start_timestamp > seq.end_timestamp {
            violations += 1;
            println!(
                "   ⚠️  Seq {}: start {} > end {}",
                i, seq.start_timestamp, seq.end_timestamp
            );
        }

        // Between sequences: should progress forward (with possible overlap)
        if i > 0 && seq.start_timestamp < prev_end_ts.saturating_sub(seq.duration_ns) {
            // Allow some overlap due to sliding window, but catch major regressions
            if seq.end_timestamp < prev_end_ts.saturating_sub(seq.duration_ns * 2) {
                violations += 1;
                println!(
                    "   ⚠️  Seq {}: end {} << prev end {} (significant regression)",
                    i, seq.end_timestamp, prev_end_ts
                );
            }
        }

        prev_end_ts = seq.end_timestamp;
    }

    println!("   Timestamp violations: {}", violations);
    assert_eq!(
        violations, 0,
        "Found {} timestamp ordering violations",
        violations
    );

    println!("\n   ✅ Sequence timestamp ordering PASSED");
}

// ============================================================================
// Test 5: Feature Extractor Determinism
// ============================================================================

#[test]
fn test_feature_extraction_determinism() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_file = &files[0];

    println!("\n📊 Feature Extraction Determinism Test");
    println!("   File: {}\n", test_file);

    // Run extraction twice with identical config
    let mut config = PipelineConfig::default();
    config.features.include_derived = true;
    config.features.include_mbo = false;
    config.sequence.feature_count = config.features.feature_count();
    config.sequence.window_size = 50;
    config.sequence.stride = 5;
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        event_count: Some(500),
        ..Default::default()
    });

    println!("   Run 1...");
    let mut pipeline1 = Pipeline::from_config(config.clone()).expect("Failed to create pipeline");
    let output1 = pipeline1.process(test_file).expect("Failed to process");

    println!("   Run 2...");
    let mut pipeline2 = Pipeline::from_config(config).expect("Failed to create pipeline");
    let output2 = pipeline2.process(test_file).expect("Failed to process");

    // Compare outputs
    assert_eq!(
        output1.messages_processed, output2.messages_processed,
        "Message counts differ: {} vs {}",
        output1.messages_processed, output2.messages_processed
    );
    assert_eq!(
        output1.features_extracted, output2.features_extracted,
        "Feature counts differ: {} vs {}",
        output1.features_extracted, output2.features_extracted
    );
    assert_eq!(
        output1.sequences_generated, output2.sequences_generated,
        "Sequence counts differ: {} vs {}",
        output1.sequences_generated, output2.sequences_generated
    );

    // Compare actual feature values (must be bit-identical)
    let mut value_mismatches = 0;
    let min_seqs = output1.sequences.len().min(output2.sequences.len());

    for i in 0..min_seqs {
        let s1 = &output1.sequences[i];
        let s2 = &output2.sequences[i];

        for (j, (f1, f2)) in s1.features.iter().zip(s2.features.iter()).enumerate() {
            for (k, (v1, v2)) in f1.iter().zip(f2.iter()).enumerate() {
                if (v1 - v2).abs() > 1e-15 {
                    value_mismatches += 1;
                    if value_mismatches <= 5 {
                        println!(
                            "   ⚠️  Mismatch seq[{}].features[{}][{}]: {} vs {}",
                            i, j, k, v1, v2
                        );
                    }
                }
            }
        }
    }

    println!("\n   📈 Determinism Results:");
    println!("      Sequences compared: {}", min_seqs);
    println!("      Value mismatches: {}", value_mismatches);

    assert_eq!(
        value_mismatches, 0,
        "Feature extraction is non-deterministic: {} mismatches",
        value_mismatches
    );

    println!("\n   ✅ Feature extraction determinism PASSED");
}

// ============================================================================
// Test 6: Memory Stability During Extended Processing
// ============================================================================

#[test]
fn test_memory_stability_extended() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_files: Vec<_> = files.iter().take(5).collect();

    println!("\n📊 Memory Stability Test (Extended)");
    println!("   Processing {} days...\n", test_files.len());

    let mut config = PipelineConfig::default();
    config.features.include_derived = true;
    config.features.include_mbo = true;
    config.sequence.feature_count = config.features.feature_count();
    config.sequence.window_size = 100;
    config.sequence.max_buffer_size = 1000; // Bounded buffer
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        event_count: Some(200),
        ..Default::default()
    });

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");

    let mut total_messages = 0u64;
    let mut total_sequences = 0u64;

    for (i, file) in test_files.iter().enumerate() {
        pipeline.reset(); // Critical: reset between days

        let filename = Path::new(file)
            .file_name()
            .unwrap()
            .to_string_lossy();

        let output = match pipeline.process(file) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("   Day {}: Error - {}", i + 1, e);
                continue;
            }
        };

        total_messages += output.messages_processed as u64;
        total_sequences += output.sequences_generated as u64;

        println!(
            "   Day {}: {} - {} msgs, {} seqs",
            i + 1,
            filename,
            output.messages_processed,
            output.sequences_generated
        );
    }

    println!("\n   📈 Summary:");
    println!("      Total messages: {}", total_messages);
    println!("      Total sequences: {}", total_sequences);

    // If we got here without panic/crash, memory is stable
    assert!(
        total_messages > 0,
        "Should have processed some messages"
    );
    assert!(
        total_sequences > 0,
        "Should have generated some sequences"
    );

    println!("\n   ✅ Memory stability PASSED (no crashes/panics during extended run)");
}

// ============================================================================
// Test 7: Zero-Allocation API Correctness
// ============================================================================

#[test]
fn test_zero_allocation_api_correctness() {
    if !data_available() {
        eprintln!("⚠️  Skipping: NVIDIA data not available");
        return;
    }

    let files = get_test_files();
    let test_file = &files[0];

    println!("\n📊 Zero-Allocation API Correctness Test");
    println!("   File: {}\n", test_file);

    let loader = DbnLoader::new(test_file).expect("Failed to create loader");
    
    // Create two LOBs: one using old API, one using new zero-allocation API
    let mut lob_old = LobReconstructor::new(10);
    let mut lob_new = LobReconstructor::new(10);
    let mut reusable_state = LobState::new(10);

    let mut msg_count = 0;
    let mut mismatches = 0;

    for msg in loader.iter_messages().expect("Failed to iterate") {
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            continue;
        }

        // Old API: returns new LobState each call
        let state_old = lob_old.process_message(&msg);
        
        // New API: fills pre-allocated state
        let result_new = lob_new.process_message_into(&msg, &mut reusable_state);

        // Both should succeed or fail together
        match (&state_old, &result_new) {
            (Ok(s1), Ok(())) => {
                // Compare states - they should be identical
                if s1.best_bid != reusable_state.best_bid {
                    mismatches += 1;
                    if mismatches <= 3 {
                        println!(
                            "   ⚠️  best_bid mismatch at msg {}: {:?} vs {:?}",
                            msg_count, s1.best_bid, reusable_state.best_bid
                        );
                    }
                }
                if s1.best_ask != reusable_state.best_ask {
                    mismatches += 1;
                    if mismatches <= 3 {
                        println!(
                            "   ⚠️  best_ask mismatch at msg {}: {:?} vs {:?}",
                            msg_count, s1.best_ask, reusable_state.best_ask
                        );
                    }
                }
                // Compare analytics
                let mid1 = s1.mid_price();
                let mid2 = reusable_state.mid_price();
                if mid1 != mid2 {
                    mismatches += 1;
                    if mismatches <= 3 {
                        println!(
                            "   ⚠️  mid_price mismatch at msg {}: {:?} vs {:?}",
                            msg_count, mid1, mid2
                        );
                    }
                }
            }
            (Err(_), Err(_)) => {
                // Both failed - expected
            }
            _ => {
                mismatches += 1;
                if mismatches <= 3 {
                    println!(
                        "   ⚠️  API result mismatch at msg {}: {:?} vs {:?}",
                        msg_count,
                        state_old.is_ok(),
                        result_new.is_ok()
                    );
                }
            }
        }

        msg_count += 1;

        // Test first 500K messages for speed
        if msg_count >= 500_000 {
            break;
        }
    }

    println!("   📈 Results:");
    println!("      Messages compared: {}", msg_count);
    println!("      Mismatches found: {}", mismatches);

    assert_eq!(
        mismatches, 0,
        "Zero-allocation API produces different results: {} mismatches",
        mismatches
    );

    println!("\n   ✅ Zero-allocation API correctness PASSED");
}

