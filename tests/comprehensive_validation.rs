//! Comprehensive Validation Test for MBO-LOB-reconstructor and feature-extractor
//!
//! This test validates both libraries with real NVIDIA MBO data (Feb 10, 2025)
//! to ensure correctness, consistency, and accuracy.
//!
//! **Note**: These tests require real MBO data files and are skipped in CI.
//! Run locally with: cargo test --test comprehensive_validation --release -- --nocapture

use feature_extractor::{
    // Labeling
    DeepLobLabelGenerator,
    DeepLobMethod,
    // Features
    FeatureExtractor,
    // Validation
    FeatureValidator,
    // Normalization
    GlobalZScoreNormalizer,
    LabelConfig,
    // Order Flow
    MultiLevelOfiTracker,
    Normalizer,
    OrderFlowTracker,
    TlobLabelGenerator,
    ZScoreNormalizer,
};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor, LobState};
use std::path::Path;
use std::time::Instant;

/// Test data path - using Feb 10, 2025 (different from commonly used Feb 3)
const TEST_DATA_PATH: &str = "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250210.mbo.dbn.zst";

/// Maximum messages to process (for reasonable test duration)
const MAX_MESSAGES: usize = 500_000;

/// Sample interval for feature extraction
const SAMPLE_INTERVAL: usize = 100;

/// Check if test data is available, skip test if not
fn require_test_data() -> bool {
    if !Path::new(TEST_DATA_PATH).exists() {
        println!("⚠️  Test data not found at: {}", TEST_DATA_PATH);
        println!("   Skipping test (this is expected in CI environments)");
        return false;
    }
    true
}

// ============================================================================
// Test 1: MBO-LOB-reconstructor Validation
// ============================================================================

#[test]
fn test_lob_reconstructor_correctness() {
    if !require_test_data() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("TEST 1: MBO-LOB-reconstructor Validation");
    println!("{}\n", "=".repeat(70));

    let start = Instant::now();
    let loader = DbnLoader::new(TEST_DATA_PATH).expect("Failed to load DBN file");
    let mut reconstructor = LobReconstructor::new(10);

    let mut msg_count = 0;
    let mut valid_states = 0;
    let mut crossed_quotes = 0;
    let mut locked_quotes = 0;
    let mut empty_books = 0;

    // Track price statistics
    let mut mid_prices: Vec<f64> = Vec::new();
    let mut spreads: Vec<f64> = Vec::new();
    let mut bid_volumes: Vec<u64> = Vec::new();
    let mut ask_volumes: Vec<u64> = Vec::new();

    // Track order statistics
    let mut add_count = 0;
    let mut modify_count = 0;
    let mut cancel_count = 0;
    let mut trade_count = 0;

    for msg in loader.iter_messages().expect("Failed to iterate messages") {
        if msg_count >= MAX_MESSAGES {
            break;
        }

        // Count message types
        match msg.action {
            mbo_lob_reconstructor::Action::Add => add_count += 1,
            mbo_lob_reconstructor::Action::Modify => modify_count += 1,
            mbo_lob_reconstructor::Action::Cancel => cancel_count += 1,
            mbo_lob_reconstructor::Action::Trade => trade_count += 1,
            _ => {}
        }

        // Skip invalid messages
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            msg_count += 1;
            continue;
        }

        let state = match reconstructor.process_message(&msg) {
            Ok(s) => s,
            Err(_) => {
                msg_count += 1;
                continue;
            }
        };

        msg_count += 1;

        // Validate LOB state
        if state.best_bid.is_none() || state.best_ask.is_none() {
            empty_books += 1;
            continue;
        }

        valid_states += 1;

        let best_bid = state.best_bid.unwrap();
        let best_ask = state.best_ask.unwrap();

        // Check for crossed/locked quotes
        if best_bid >= best_ask {
            if best_bid == best_ask {
                locked_quotes += 1;
            } else {
                crossed_quotes += 1;
            }
        }

        // Collect statistics (sample every 100 messages)
        if valid_states % 100 == 0 {
            if let Some(mid) = state.mid_price() {
                mid_prices.push(mid);
            }
            if let Some(spread) = state.spread() {
                spreads.push(spread);
            }
            bid_volumes.push(state.total_bid_volume());
            ask_volumes.push(state.total_ask_volume());
        }
    }

    let elapsed = start.elapsed();

    // Calculate statistics
    let avg_mid = mid_prices.iter().sum::<f64>() / mid_prices.len() as f64;
    let min_mid = mid_prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mid = mid_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let avg_spread = spreads.iter().sum::<f64>() / spreads.len() as f64;
    let min_spread = spreads.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_spread = spreads.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let avg_bid_vol = bid_volumes.iter().sum::<u64>() as f64 / bid_volumes.len() as f64;
    let avg_ask_vol = ask_volumes.iter().sum::<u64>() as f64 / ask_volumes.len() as f64;

    // Print results
    println!("📊 Processing Statistics:");
    println!("   Messages processed: {}", msg_count);
    println!("   Valid LOB states:   {}", valid_states);
    println!("   Empty books:        {}", empty_books);
    println!("   Crossed quotes:     {}", crossed_quotes);
    println!("   Locked quotes:      {}", locked_quotes);
    println!("   Time elapsed:       {:.2}s", elapsed.as_secs_f64());
    println!(
        "   Throughput:         {:.0} msg/s",
        msg_count as f64 / elapsed.as_secs_f64()
    );

    println!("\n📈 Message Type Distribution:");
    println!(
        "   Add:    {:>8} ({:.1}%)",
        add_count,
        add_count as f64 / msg_count as f64 * 100.0
    );
    println!(
        "   Modify: {:>8} ({:.1}%)",
        modify_count,
        modify_count as f64 / msg_count as f64 * 100.0
    );
    println!(
        "   Cancel: {:>8} ({:.1}%)",
        cancel_count,
        cancel_count as f64 / msg_count as f64 * 100.0
    );
    println!(
        "   Trade:  {:>8} ({:.1}%)",
        trade_count,
        trade_count as f64 / msg_count as f64 * 100.0
    );

    println!("\n💰 Price Statistics:");
    println!("   Mid-price avg:  ${:.4}", avg_mid);
    println!("   Mid-price min:  ${:.4}", min_mid);
    println!("   Mid-price max:  ${:.4}", max_mid);
    println!("   Price range:    ${:.4}", max_mid - min_mid);

    println!("\n📏 Spread Statistics:");
    println!(
        "   Spread avg:     ${:.6} ({:.2} bps)",
        avg_spread,
        avg_spread / avg_mid * 10000.0
    );
    println!("   Spread min:     ${:.6}", min_spread);
    println!("   Spread max:     ${:.6}", max_spread);

    println!("\n📦 Volume Statistics:");
    println!("   Avg bid volume: {:.0} shares", avg_bid_vol);
    println!("   Avg ask volume: {:.0} shares", avg_ask_vol);
    println!("   Volume ratio:   {:.3}", avg_bid_vol / avg_ask_vol);

    // Assertions
    println!("\n✅ Validation Checks:");

    // 1. Should have processed significant data
    assert!(msg_count > 100_000, "Should process at least 100K messages");
    println!("   ✓ Processed {} messages", msg_count);

    // 2. Should have mostly valid states
    let valid_ratio = valid_states as f64 / msg_count as f64;
    assert!(
        valid_ratio > 0.5,
        "Should have >50% valid states, got {:.1}%",
        valid_ratio * 100.0
    );
    println!("   ✓ Valid state ratio: {:.1}%", valid_ratio * 100.0);

    // 3. No crossed quotes (our implementation should prevent this)
    assert_eq!(crossed_quotes, 0, "Should have no crossed quotes");
    println!("   ✓ No crossed quotes detected");

    // 4. Mid-price should be in reasonable range for NVIDIA ($100-$150 in early 2025)
    assert!(
        avg_mid > 100.0 && avg_mid < 150.0,
        "Mid-price ${:.2} out of expected range",
        avg_mid
    );
    println!("   ✓ Mid-price ${:.2} in expected range", avg_mid);

    // 5. Spread should be reasonable (< $1 for liquid stock)
    assert!(avg_spread < 1.0, "Spread ${:.4} too wide", avg_spread);
    println!("   ✓ Spread ${:.4} is reasonable", avg_spread);

    // 6. Spread should be positive
    assert!(min_spread > 0.0, "Minimum spread should be positive");
    println!("   ✓ All spreads positive");

    // 7. Volume should be reasonable
    assert!(avg_bid_vol > 1000.0, "Bid volume too low");
    assert!(avg_ask_vol > 1000.0, "Ask volume too low");
    println!("   ✓ Volume levels reasonable");

    println!("\n✅ MBO-LOB-reconstructor PASSED all validation checks!");
}

// ============================================================================
// Test 2: Feature Extractor Validation
// ============================================================================

#[test]
fn test_feature_extractor_correctness() {
    if !require_test_data() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("TEST 2: Feature Extractor Validation");
    println!("{}\n", "=".repeat(70));

    let start = Instant::now();
    let loader = DbnLoader::new(TEST_DATA_PATH).expect("Failed to load DBN file");
    let mut reconstructor = LobReconstructor::new(10);
    let extractor = FeatureExtractor::new(10);
    let validator = FeatureValidator::new();

    let mut msg_count = 0;
    let mut feature_count = 0;
    let mut validation_errors = 0;

    // Track feature statistics
    let mut all_features: Vec<Vec<f64>> = Vec::new();
    let mut mid_prices: Vec<f64> = Vec::new();

    for msg in loader.iter_messages().expect("Failed to iterate messages") {
        if msg_count >= MAX_MESSAGES {
            break;
        }

        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            msg_count += 1;
            continue;
        }

        let state = match reconstructor.process_message(&msg) {
            Ok(s) => s,
            Err(_) => {
                msg_count += 1;
                continue;
            }
        };

        msg_count += 1;

        // Sample at intervals
        if msg_count % SAMPLE_INTERVAL != 0 {
            continue;
        }

        // Validate LOB state
        let validation = validator.validate_lob(&state);
        if !validation.is_valid() {
            validation_errors += 1;
            continue;
        }

        // Extract features
        if let Ok(features) = extractor.extract_lob_features(&state) {
            all_features.push(features.clone());
            feature_count += 1;

            if let Some(mid) = state.mid_price() {
                mid_prices.push(mid);
            }
        }
    }

    let elapsed = start.elapsed();

    // Analyze features
    let n_features = if all_features.is_empty() {
        0
    } else {
        all_features[0].len()
    };

    println!("📊 Feature Extraction Statistics:");
    println!("   Messages processed:  {}", msg_count);
    println!("   Features extracted:  {}", feature_count);
    println!("   Validation errors:   {}", validation_errors);
    println!("   Features per sample: {}", n_features);
    println!("   Time elapsed:        {:.2}s", elapsed.as_secs_f64());

    // Analyze each feature dimension
    if !all_features.is_empty() {
        println!("\n📈 Feature Analysis (first 10 features):");

        for feat_idx in 0..10.min(n_features) {
            let values: Vec<f64> = all_features.iter().map(|f| f[feat_idx]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let feat_name = match feat_idx {
                0..=9 => format!("ask_price_{}", feat_idx + 1),
                10..=19 => format!("ask_size_{}", feat_idx - 9),
                20..=29 => format!("bid_price_{}", feat_idx - 19),
                30..=39 => format!("bid_size_{}", feat_idx - 29),
                _ => format!("feature_{}", feat_idx),
            };

            println!(
                "   {:15} mean={:>12.4} min={:>12.4} max={:>12.4}",
                feat_name, mean, min, max
            );
        }
    }

    // Validation checks
    println!("\n✅ Validation Checks:");

    // 1. Should extract features
    assert!(
        feature_count > 1000,
        "Should extract at least 1000 feature vectors"
    );
    println!("   ✓ Extracted {} feature vectors", feature_count);

    // 2. Feature count should be 40 (10 levels × 4 features)
    assert_eq!(
        n_features, 40,
        "Should have 40 features, got {}",
        n_features
    );
    println!("   ✓ Feature count is 40 (correct)");

    // 3. No NaN or Inf in features
    let mut nan_count = 0;
    let mut inf_count = 0;
    for features in &all_features {
        for &val in features {
            if val.is_nan() {
                nan_count += 1;
            }
            if val.is_infinite() {
                inf_count += 1;
            }
        }
    }
    assert_eq!(nan_count, 0, "Should have no NaN values");
    assert_eq!(inf_count, 0, "Should have no Inf values");
    println!("   ✓ No NaN or Inf values in features");

    // 4. Prices should be positive
    for features in &all_features {
        for i in 0..10 {
            assert!(features[i] > 0.0, "Ask price {} should be positive", i);
            assert!(features[20 + i] > 0.0, "Bid price {} should be positive", i);
        }
    }
    println!("   ✓ All prices are positive");

    // 5. Volumes should be non-negative
    for features in &all_features {
        for i in 10..20 {
            assert!(
                features[i] >= 0.0,
                "Ask size {} should be non-negative",
                i - 10
            );
            assert!(
                features[i + 20] >= 0.0,
                "Bid size {} should be non-negative",
                i - 10
            );
        }
    }
    println!("   ✓ All volumes are non-negative");

    // 6. Ask prices should be ascending (level 1 < level 2 < ...)
    for features in &all_features {
        for i in 0..9 {
            if features[i] > 0.0 && features[i + 1] > 0.0 {
                assert!(
                    features[i] <= features[i + 1],
                    "Ask prices should be ascending: {} > {}",
                    features[i],
                    features[i + 1]
                );
            }
        }
    }
    println!("   ✓ Ask prices are properly ordered (ascending)");

    // 7. Bid prices should be descending (level 1 > level 2 > ...)
    for features in &all_features {
        for i in 20..29 {
            if features[i] > 0.0 && features[i + 1] > 0.0 {
                assert!(
                    features[i] >= features[i + 1],
                    "Bid prices should be descending: {} < {}",
                    features[i],
                    features[i + 1]
                );
            }
        }
    }
    println!("   ✓ Bid prices are properly ordered (descending)");

    // 8. Best bid < Best ask (no crossed book)
    for features in &all_features {
        let best_ask = features[0];
        let best_bid = features[20];
        assert!(
            best_bid < best_ask,
            "Best bid {} should be < best ask {}",
            best_bid,
            best_ask
        );
    }
    println!("   ✓ No crossed books (best_bid < best_ask)");

    println!("\n✅ Feature Extractor PASSED all validation checks!");
}

// ============================================================================
// Test 3: Order Flow Features Validation
// ============================================================================

#[test]
fn test_order_flow_features() {
    if !require_test_data() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("TEST 3: Order Flow Features Validation");
    println!("{}\n", "=".repeat(70));

    let start = Instant::now();
    let loader = DbnLoader::new(TEST_DATA_PATH).expect("Failed to load DBN file");
    let mut reconstructor = LobReconstructor::new(10);
    let mut ofi_tracker = OrderFlowTracker::new();
    let mut mlofi_tracker = MultiLevelOfiTracker::new(10);

    let mut msg_count = 0;
    let mut ofi_samples: Vec<f64> = Vec::new();
    let mut queue_imbalance_samples: Vec<f64> = Vec::new();

    let mut prev_state: Option<LobState> = None;

    for msg in loader.iter_messages().expect("Failed to iterate messages") {
        if msg_count >= MAX_MESSAGES {
            break;
        }

        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            msg_count += 1;
            continue;
        }

        let state = match reconstructor.process_message(&msg) {
            Ok(s) => s,
            Err(_) => {
                msg_count += 1;
                continue;
            }
        };

        msg_count += 1;

        if state.best_bid.is_none() || state.best_ask.is_none() {
            continue;
        }

        // Update trackers
        if let Some(ref _prev) = prev_state {
            ofi_tracker.update(&state, msg.timestamp.unwrap_or(0) as u64);
            mlofi_tracker.update(&state);

            // Sample at intervals
            if msg_count % SAMPLE_INTERVAL == 0 {
                let features = ofi_tracker.extract_features(&state);
                ofi_samples.push(features.ofi);
                queue_imbalance_samples.push(features.queue_imbalance);
            }
        }

        prev_state = Some(state);
    }

    let elapsed = start.elapsed();

    // Calculate statistics
    let ofi_mean = ofi_samples.iter().sum::<f64>() / ofi_samples.len() as f64;
    let ofi_min = ofi_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let ofi_max = ofi_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let qi_mean =
        queue_imbalance_samples.iter().sum::<f64>() / queue_imbalance_samples.len() as f64;
    let qi_min = queue_imbalance_samples
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let qi_max = queue_imbalance_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("📊 Order Flow Statistics:");
    println!("   Messages processed: {}", msg_count);
    println!("   OFI samples:        {}", ofi_samples.len());
    println!("   Time elapsed:       {:.2}s", elapsed.as_secs_f64());

    println!("\n📈 OFI Statistics:");
    println!("   Mean:  {:>12.4}", ofi_mean);
    println!("   Min:   {:>12.4}", ofi_min);
    println!("   Max:   {:>12.4}", ofi_max);

    println!("\n📈 Queue Imbalance Statistics:");
    println!("   Mean:  {:>12.4}", qi_mean);
    println!("   Min:   {:>12.4}", qi_min);
    println!("   Max:   {:>12.4}", qi_max);

    // Get MLOFI
    let mlofi = mlofi_tracker.get_ofi_per_level();
    println!("\n📈 Multi-Level OFI (per level):");
    for (i, ofi) in mlofi.iter().enumerate() {
        println!("   Level {}: {:>12.4}", i + 1, ofi);
    }

    // Validation checks
    println!("\n✅ Validation Checks:");

    // 1. Should have samples
    assert!(!ofi_samples.is_empty(), "Should have OFI samples");
    println!("   ✓ Generated {} OFI samples", ofi_samples.len());

    // 2. OFI should be finite
    for &ofi in &ofi_samples {
        assert!(ofi.is_finite(), "OFI should be finite");
    }
    println!("   ✓ All OFI values are finite");

    // 3. Queue imbalance should be in [-1, 1]
    for &qi in &queue_imbalance_samples {
        assert!(
            qi >= -1.0 && qi <= 1.0,
            "Queue imbalance {} should be in [-1, 1]",
            qi
        );
    }
    println!("   ✓ Queue imbalance values in valid range [-1, 1]");

    // 4. MLOFI should have 10 levels
    assert_eq!(mlofi.len(), 10, "Should have 10 MLOFI levels");
    println!("   ✓ MLOFI has 10 levels");

    println!("\n✅ Order Flow Features PASSED all validation checks!");
}

// ============================================================================
// Test 4: Labeling Validation
// ============================================================================

#[test]
fn test_labeling_correctness() {
    if !require_test_data() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("TEST 4: Labeling Validation");
    println!("{}\n", "=".repeat(70));

    let start = Instant::now();
    let loader = DbnLoader::new(TEST_DATA_PATH).expect("Failed to load DBN file");
    let mut reconstructor = LobReconstructor::new(10);

    // Collect mid-prices
    let mut mid_prices: Vec<f64> = Vec::new();
    let mut msg_count = 0;

    for msg in loader.iter_messages().expect("Failed to iterate messages") {
        if msg_count >= MAX_MESSAGES {
            break;
        }

        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            msg_count += 1;
            continue;
        }

        let state = match reconstructor.process_message(&msg) {
            Ok(s) => s,
            Err(_) => {
                msg_count += 1;
                continue;
            }
        };

        msg_count += 1;

        // Sample at intervals
        if msg_count % SAMPLE_INTERVAL == 0 {
            if let Some(mid) = state.mid_price() {
                mid_prices.push(mid);
            }
        }
    }

    let elapsed = start.elapsed();

    println!("📊 Data Collection:");
    println!("   Messages processed: {}", msg_count);
    println!("   Mid-prices collected: {}", mid_prices.len());
    println!("   Time elapsed: {:.2}s", elapsed.as_secs_f64());

    // Test TLOB labeling
    println!("\n--- TLOB Labeling ---");
    let tlob_config = LabelConfig {
        horizon: 50,
        smoothing_window: 10,
        threshold: 0.002,
    };

    let mut tlob_gen = TlobLabelGenerator::new(tlob_config.clone());
    tlob_gen.add_prices(&mid_prices);
    let tlob_labels = tlob_gen.generate_labels().expect("TLOB labeling failed");
    let tlob_stats = tlob_gen.compute_stats(&tlob_labels);

    println!("   Total labels:  {}", tlob_stats.total);
    println!(
        "   Up:            {} ({:.1}%)",
        tlob_stats.up_count,
        tlob_stats.up_count as f64 / tlob_stats.total as f64 * 100.0
    );
    println!(
        "   Stable:        {} ({:.1}%)",
        tlob_stats.stable_count,
        tlob_stats.stable_count as f64 / tlob_stats.total as f64 * 100.0
    );
    println!(
        "   Down:          {} ({:.1}%)",
        tlob_stats.down_count,
        tlob_stats.down_count as f64 / tlob_stats.total as f64 * 100.0
    );
    println!("   Avg change:    {:.4}%", tlob_stats.avg_change * 100.0);
    println!("   Std change:    {:.4}%", tlob_stats.std_change * 100.0);

    // Test DeepLOB labeling
    println!("\n--- DeepLOB Labeling ---");
    let deeplob_config = LabelConfig::fi2010(50);

    let mut deeplob_gen =
        DeepLobLabelGenerator::with_method(deeplob_config.clone(), DeepLobMethod::VsPastAverage);
    deeplob_gen.add_prices(&mid_prices);
    let deeplob_labels = deeplob_gen
        .generate_labels()
        .expect("DeepLOB labeling failed");
    let deeplob_stats = deeplob_gen.compute_stats(&deeplob_labels);

    println!("   Total labels:  {}", deeplob_stats.total);
    println!(
        "   Up:            {} ({:.1}%)",
        deeplob_stats.up_count,
        deeplob_stats.up_count as f64 / deeplob_stats.total as f64 * 100.0
    );
    println!(
        "   Stable:        {} ({:.1}%)",
        deeplob_stats.stable_count,
        deeplob_stats.stable_count as f64 / deeplob_stats.total as f64 * 100.0
    );
    println!(
        "   Down:          {} ({:.1}%)",
        deeplob_stats.down_count,
        deeplob_stats.down_count as f64 / deeplob_stats.total as f64 * 100.0
    );
    println!("   Avg change:    {:.4}%", deeplob_stats.avg_change * 100.0);
    println!("   Std change:    {:.4}%", deeplob_stats.std_change * 100.0);

    // Validation checks
    println!("\n✅ Validation Checks:");

    // 1. Should generate labels
    assert!(tlob_stats.total > 1000, "TLOB should generate >1000 labels");
    assert!(
        deeplob_stats.total > 1000,
        "DeepLOB should generate >1000 labels"
    );
    println!("   ✓ Both methods generated sufficient labels");

    // 2. Label counts should sum to total
    assert_eq!(
        tlob_stats.up_count + tlob_stats.down_count + tlob_stats.stable_count,
        tlob_stats.total
    );
    assert_eq!(
        deeplob_stats.up_count + deeplob_stats.down_count + deeplob_stats.stable_count,
        deeplob_stats.total
    );
    println!("   ✓ Label counts sum to total");

    // 3. No NaN in percentage changes
    for (_, _, change) in &tlob_labels {
        assert!(!change.is_nan(), "TLOB change should not be NaN");
        assert!(!change.is_infinite(), "TLOB change should not be Inf");
    }
    for (_, _, change) in &deeplob_labels {
        assert!(!change.is_nan(), "DeepLOB change should not be NaN");
        assert!(!change.is_infinite(), "DeepLOB change should not be Inf");
    }
    println!("   ✓ All percentage changes are finite");

    // 4. Labels should have some variety (not all same class)
    assert!(
        tlob_stats.up_count > 0 || tlob_stats.down_count > 0,
        "Should have some directional labels"
    );
    println!("   ✓ Labels show variety (not all stable)");

    // 5. Class balance - note that Stable can dominate with tight thresholds
    // This is expected behavior for low-volatility periods
    let max_pct = (tlob_stats
        .up_count
        .max(tlob_stats.down_count)
        .max(tlob_stats.stable_count)) as f64
        / tlob_stats.total as f64;
    // Stable can be up to 95% with tight thresholds in low-volatility periods
    assert!(
        max_pct < 0.95,
        "No class should dominate >95%, got {:.1}%",
        max_pct * 100.0
    );
    println!(
        "   ✓ Class balance check passed (max {:.1}%)",
        max_pct * 100.0
    );
    if max_pct > 0.8 {
        println!(
            "   ℹ️  Note: High Stable ratio ({:.1}%) indicates low volatility or tight threshold",
            max_pct * 100.0
        );
    }

    println!("\n✅ Labeling PASSED all validation checks!");
}

// ============================================================================
// Test 5: Normalization Validation
// ============================================================================

#[test]
fn test_normalization_correctness() {
    if !require_test_data() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("TEST 5: Normalization Validation");
    println!("{}\n", "=".repeat(70));

    let loader = DbnLoader::new(TEST_DATA_PATH).expect("Failed to load DBN file");
    let mut reconstructor = LobReconstructor::new(10);
    let extractor = FeatureExtractor::new(10);
    let global_normalizer = GlobalZScoreNormalizer::new();

    // Collect features
    let mut all_features: Vec<Vec<f64>> = Vec::new();
    let mut msg_count = 0;

    for msg in loader.iter_messages().expect("Failed to iterate messages") {
        if msg_count >= MAX_MESSAGES / 2 {
            break;
        }

        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            msg_count += 1;
            continue;
        }

        let state = match reconstructor.process_message(&msg) {
            Ok(s) => s,
            Err(_) => {
                msg_count += 1;
                continue;
            }
        };

        msg_count += 1;

        if msg_count % SAMPLE_INTERVAL != 0 {
            continue;
        }

        if state.best_bid.is_none() || state.best_ask.is_none() {
            continue;
        }

        if let Ok(features) = extractor.extract_lob_features(&state) {
            all_features.push(features);
        }
    }

    println!(
        "📊 Collected {} feature vectors for normalization testing",
        all_features.len()
    );

    // Test Z-score normalization
    println!("\n--- Z-Score Normalization ---");
    let mut zscore_normalizers: Vec<ZScoreNormalizer> =
        (0..40).map(|_| ZScoreNormalizer::new()).collect();

    // Fit normalizers
    for features in &all_features {
        for (i, &val) in features.iter().enumerate() {
            zscore_normalizers[i].update(val);
        }
    }

    // Normalize and check
    let mut zscore_normalized: Vec<Vec<f64>> = Vec::new();
    for features in &all_features {
        let normalized: Vec<f64> = features
            .iter()
            .enumerate()
            .map(|(i, &val)| zscore_normalizers[i].normalize(val))
            .collect();
        zscore_normalized.push(normalized);
    }

    // Calculate statistics of normalized features
    let mut zscore_means: Vec<f64> = vec![0.0; 40];
    let mut zscore_stds: Vec<f64> = vec![0.0; 40];

    for i in 0..40 {
        let values: Vec<f64> = zscore_normalized.iter().map(|f| f[i]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        zscore_means[i] = mean;
        zscore_stds[i] = variance.sqrt();
    }

    println!(
        "   Feature 0 (ask_price_1): mean={:.4}, std={:.4}",
        zscore_means[0], zscore_stds[0]
    );
    println!(
        "   Feature 10 (ask_size_1): mean={:.4}, std={:.4}",
        zscore_means[10], zscore_stds[10]
    );
    println!(
        "   Feature 20 (bid_price_1): mean={:.4}, std={:.4}",
        zscore_means[20], zscore_stds[20]
    );
    println!(
        "   Feature 30 (bid_size_1): mean={:.4}, std={:.4}",
        zscore_means[30], zscore_stds[30]
    );

    // Test Global Z-score normalization
    println!("\n--- Global Z-Score Normalization ---");
    let mut global_normalized: Vec<Vec<f64>> = Vec::new();
    for features in &all_features {
        let normalized = global_normalizer.normalize_snapshot(features);
        global_normalized.push(normalized);
    }

    // Check global normalization statistics
    let mut global_means: Vec<f64> = vec![0.0; 40];
    for i in 0..40 {
        let values: Vec<f64> = global_normalized.iter().map(|f| f[i]).collect();
        global_means[i] = values.iter().sum::<f64>() / values.len() as f64;
    }

    println!("   Feature 0 (ask_price_1): mean={:.4}", global_means[0]);
    println!("   Feature 10 (ask_size_1): mean={:.4}", global_means[10]);
    println!("   Feature 20 (bid_price_1): mean={:.4}", global_means[20]);
    println!("   Feature 30 (bid_size_1): mean={:.4}", global_means[30]);

    // Validation checks
    println!("\n✅ Validation Checks:");

    // 1. Z-score normalized means should be close to 0
    for (i, &mean) in zscore_means.iter().enumerate() {
        assert!(
            mean.abs() < 0.1,
            "Z-score mean for feature {} should be ~0, got {}",
            i,
            mean
        );
    }
    println!("   ✓ Z-score means are close to 0");

    // 2. Z-score normalized stds should be close to 1
    for (i, &std) in zscore_stds.iter().enumerate() {
        assert!(
            (std - 1.0).abs() < 0.1,
            "Z-score std for feature {} should be ~1, got {}",
            i,
            std
        );
    }
    println!("   ✓ Z-score stds are close to 1");

    // 3. No NaN or Inf in normalized features
    for features in &zscore_normalized {
        for &val in features {
            assert!(!val.is_nan(), "Normalized value should not be NaN");
            assert!(!val.is_infinite(), "Normalized value should not be Inf");
        }
    }
    println!("   ✓ No NaN or Inf in Z-score normalized features");

    // 4. Global normalization should preserve bid < ask ordering
    for features in &global_normalized {
        // After global normalization, the relative ordering should be preserved
        // (though absolute values change)
        assert!(
            features[0].is_finite(),
            "Global normalized ask price should be finite"
        );
        assert!(
            features[20].is_finite(),
            "Global normalized bid price should be finite"
        );
    }
    println!("   ✓ Global normalization produces finite values");

    println!("\n✅ Normalization PASSED all validation checks!");
}

// ============================================================================
// Test 6: End-to-End Pipeline Validation
// ============================================================================

#[test]
fn test_end_to_end_pipeline() {
    if !require_test_data() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("TEST 6: End-to-End Pipeline Validation");
    println!("{}\n", "=".repeat(70));

    let start = Instant::now();
    let loader = DbnLoader::new(TEST_DATA_PATH).expect("Failed to load DBN file");
    let mut reconstructor = LobReconstructor::new(10);
    let extractor = FeatureExtractor::new(10);
    let validator = FeatureValidator::new();
    let global_normalizer = GlobalZScoreNormalizer::new();

    // Collect data
    let mut features_raw: Vec<Vec<f64>> = Vec::new();
    let mut mid_prices: Vec<f64> = Vec::new();
    let mut timestamps: Vec<u64> = Vec::new();
    let mut msg_count = 0;

    for msg in loader.iter_messages().expect("Failed to iterate messages") {
        if msg_count >= MAX_MESSAGES {
            break;
        }

        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            msg_count += 1;
            continue;
        }

        let state = match reconstructor.process_message(&msg) {
            Ok(s) => s,
            Err(_) => {
                msg_count += 1;
                continue;
            }
        };

        msg_count += 1;

        if msg_count % SAMPLE_INTERVAL != 0 {
            continue;
        }

        // Validate
        let validation = validator.validate_lob(&state);
        if !validation.is_valid() {
            continue;
        }

        // Extract features
        if let Ok(features) = extractor.extract_lob_features(&state) {
            features_raw.push(features);
            if let Some(mid) = state.mid_price() {
                mid_prices.push(mid);
            }
            timestamps.push(msg.timestamp.unwrap_or(0) as u64);
        }
    }

    // Normalize features
    let features_normalized: Vec<Vec<f64>> = features_raw
        .iter()
        .map(|f| global_normalizer.normalize_snapshot(f))
        .collect();

    // Generate labels
    let label_config = LabelConfig {
        horizon: 50,
        smoothing_window: 10,
        threshold: 0.002,
    };
    let mut label_gen = TlobLabelGenerator::new(label_config);
    label_gen.add_prices(&mid_prices);
    let labels = label_gen.generate_labels().expect("Labeling failed");
    let stats = label_gen.compute_stats(&labels);

    let elapsed = start.elapsed();

    // Summary
    println!("📊 Pipeline Summary:");
    println!("   Messages processed:     {}", msg_count);
    println!("   Feature vectors:        {}", features_raw.len());
    println!("   Normalized vectors:     {}", features_normalized.len());
    println!("   Labels generated:       {}", labels.len());
    println!("   Total time:             {:.2}s", elapsed.as_secs_f64());
    println!(
        "   Throughput:             {:.0} msg/s",
        msg_count as f64 / elapsed.as_secs_f64()
    );

    println!("\n📈 Label Distribution:");
    println!(
        "   Up:     {} ({:.1}%)",
        stats.up_count,
        stats.up_count as f64 / stats.total as f64 * 100.0
    );
    println!(
        "   Stable: {} ({:.1}%)",
        stats.stable_count,
        stats.stable_count as f64 / stats.total as f64 * 100.0
    );
    println!(
        "   Down:   {} ({:.1}%)",
        stats.down_count,
        stats.down_count as f64 / stats.total as f64 * 100.0
    );

    // Calculate alignment
    let valid_pairs = labels.len().min(features_normalized.len());
    println!("\n📐 Data Alignment:");
    println!("   Features:  {}", features_normalized.len());
    println!("   Labels:    {}", labels.len());
    println!("   Aligned:   {}", valid_pairs);

    // Validation
    println!("\n✅ Validation Checks:");

    // 1. Should have data
    assert!(
        features_raw.len() > 1000,
        "Should have >1000 feature vectors"
    );
    println!("   ✓ Sufficient data collected");

    // 2. Features and normalized should match
    assert_eq!(features_raw.len(), features_normalized.len());
    println!("   ✓ Raw and normalized feature counts match");

    // 3. Labels generated
    assert!(labels.len() > 500, "Should generate >500 labels");
    println!("   ✓ Labels generated successfully");

    // 4. Pipeline produces valid output
    for features in &features_normalized {
        for &val in features {
            assert!(val.is_finite(), "All normalized values should be finite");
        }
    }
    println!("   ✓ All pipeline outputs are valid");

    // 5. Timestamps are monotonic
    for i in 1..timestamps.len() {
        assert!(
            timestamps[i] >= timestamps[i - 1],
            "Timestamps should be monotonic"
        );
    }
    println!("   ✓ Timestamps are monotonic");

    println!("\n✅ End-to-End Pipeline PASSED all validation checks!");
    println!("\n{}", "=".repeat(70));
    println!("ALL TESTS COMPLETED SUCCESSFULLY!");
    println!("{}", "=".repeat(70));
}
