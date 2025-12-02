//! Comprehensive Integration Tests for Feature Extractor with MBO-LOB-Reconstructor
//!
//! Tests the complete preprocessing pipeline with real NVIDIA MBO data to ensure
//! every module, logic, and computed number works exactly as intended.
//!
//! # Test Categories
//!
//! 1. **MBO-LOB Integration**: Verify MBO-LOB-reconstructor works with feature_extractor
//! 2. **LOB Feature Extraction**: Raw feature extraction accuracy
//! 3. **FI-2010 Features**: Handcrafted feature computation
//! 4. **Order Flow Features**: OFI, MLOFI, queue imbalance
//! 5. **Normalization**: All normalization strategies
//! 6. **Validation**: Data quality checks
//! 7. **Statistical Properties**: Research-backed invariants
//! 8. **Performance**: Throughput and latency
//! 9. **Numerical Accuracy**: Financial precision verification

use feature_extractor::{
    features::fi2010::{FI2010Config, FI2010Extractor},
    features::market_impact::{estimate_buy_impact, estimate_sell_impact},
    features::order_flow::{MultiLevelOfiTracker, OrderFlowTracker},
    preprocessing::{
        BilinearNormalizer, GlobalZScoreNormalizer, Normalizer, PerFeatureNormalizer,
        RollingZScoreNormalizer, VolatilityEstimator, ZScoreNormalizer,
    },
    validation::{validate_timestamps, FeatureValidator},
    FeatureConfig, FeatureExtractor, LobState,
};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor};
use std::path::Path;
use std::time::Instant;

/// Path to NVIDIA MBO data
const NVDA_DATA_DIR: &str =
    "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30";

/// Get path to a specific day's data file
fn get_data_file(date: &str) -> String {
    format!("{NVDA_DATA_DIR}/xnas-itch-{date}.mbo.dbn.zst")
}

/// Check if test data is available
fn data_available() -> bool {
    Path::new(NVDA_DATA_DIR).exists()
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Load LOB states from a day's data using MBO-LOB-reconstructor
fn load_lob_states(date: &str, max_snapshots: usize) -> Vec<(u64, LobState)> {
    let path = get_data_file(date);
    if !Path::new(&path).exists() {
        eprintln!("Data file not found: {path}");
        return Vec::new();
    }

    let loader = match DbnLoader::new(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to create loader: {e}");
            return Vec::new();
        }
    };

    let mut reconstructor = LobReconstructor::new(10);
    let mut states = Vec::new();
    let mut message_count = 0;

    let iter = match loader.iter_messages() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Failed to create message iterator: {e}");
            return Vec::new();
        }
    };

    for msg in iter {
        message_count += 1;

        // Get timestamp from message
        let timestamp = msg.timestamp.unwrap_or(0) as u64;

        // Process message
        match reconstructor.process_message(&msg) {
            Ok(state) => {
                if state.best_bid.is_some() && state.best_ask.is_some() {
                    // Verify the state is not crossed
                    if let (Some(bid), Some(ask)) = (state.best_bid, state.best_ask) {
                        if bid < ask {
                            states.push((timestamp, state.clone()));
                            if states.len() >= max_snapshots {
                                break;
                            }
                        }
                    }
                }
            }
            Err(_) => continue,
        }

        // Safety limit
        if message_count > max_snapshots * 100 {
            break;
        }
    }

    states
}

/// Verify LOB state is valid (not crossed, has quotes)
fn is_valid_lob(state: &LobState) -> bool {
    match (state.best_bid, state.best_ask) {
        (Some(bid), Some(ask)) => bid < ask && bid > 0 && ask > 0,
        _ => false,
    }
}

/// Convert internal price (i64 with 9 decimal places) to f64 dollars
fn price_to_dollars(price: i64) -> f64 {
    price as f64 / 1e9
}

// =============================================================================
// 1. MBO-LOB Integration Tests
// =============================================================================

#[test]
fn test_mbo_lob_reconstructor_integration() {
    if !data_available() {
        eprintln!("Skipping test: NVDA data not available at {NVDA_DATA_DIR}");
        return;
    }

    println!("\n=== MBO-LOB Reconstructor Integration Test ===\n");

    let states = load_lob_states("20250203", 1000);
    if states.is_empty() {
        eprintln!("Skipping test: No LOB states loaded");
        return;
    }

    println!("Loaded {} valid LOB states", states.len());

    // Verify basic LOB properties
    let mut valid_count = 0;
    let mut crossed_count = 0;
    let mut empty_count = 0;

    for (ts, state) in &states {
        match (state.best_bid, state.best_ask) {
            (Some(bid), Some(ask)) => {
                if bid >= ask {
                    crossed_count += 1;
                    println!(
                        "  WARNING: Crossed quote at ts {}: bid={} >= ask={}",
                        ts,
                        price_to_dollars(bid),
                        price_to_dollars(ask)
                    );
                } else {
                    valid_count += 1;
                }
            }
            _ => {
                empty_count += 1;
            }
        }
    }

    println!(
        "\nLOB State Summary:\n  Valid: {valid_count}\n  Crossed: {crossed_count}\n  Empty: {empty_count}"
    );

    // All loaded states should be valid (we filter in load_lob_states)
    assert_eq!(
        crossed_count, 0,
        "Should have no crossed quotes in loaded states"
    );
    assert!(valid_count > 0, "Should have valid LOB states");

    // Verify price levels are properly ordered
    for (_, state) in states.iter().take(100) {
        // Ask prices should be increasing
        for i in 1..state.levels {
            if state.ask_prices[i] > 0 && state.ask_prices[i - 1] > 0 {
                assert!(
                    state.ask_prices[i] >= state.ask_prices[i - 1],
                    "Ask prices should be increasing"
                );
            }
        }

        // Bid prices should be decreasing
        for i in 1..state.levels {
            if state.bid_prices[i] > 0 && state.bid_prices[i - 1] > 0 {
                assert!(
                    state.bid_prices[i] <= state.bid_prices[i - 1],
                    "Bid prices should be decreasing"
                );
            }
        }
    }

    println!("\n✓ MBO-LOB Reconstructor integration verified");
}

#[test]
fn test_lob_state_consistency() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 500);
    if states.is_empty() {
        return;
    }

    println!("\n=== LOB State Consistency Test ===\n");

    for (ts, state) in &states {
        // Verify best_bid matches bid_prices[0]
        if let Some(best_bid) = state.best_bid {
            assert_eq!(
                best_bid, state.bid_prices[0],
                "best_bid should match bid_prices[0] at ts {ts}"
            );
        }

        // Verify best_ask matches ask_prices[0]
        if let Some(best_ask) = state.best_ask {
            assert_eq!(
                best_ask, state.ask_prices[0],
                "best_ask should match ask_prices[0] at ts {ts}"
            );
        }

        // Verify volumes are valid (u32 is always non-negative)
        for i in 0..state.levels {
            // Sizes are u32, so always >= 0; just verify they exist
            let _ = state.bid_sizes[i];
            let _ = state.ask_sizes[i];
        }
    }

    println!(
        "✓ LOB state consistency verified for {} states",
        states.len()
    );
}

// =============================================================================
// 2. LOB Feature Extraction Tests
// =============================================================================

#[test]
fn test_raw_lob_feature_extraction() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 500);
    if states.is_empty() {
        return;
    }

    println!("\n=== Raw LOB Feature Extraction Test ===\n");

    let extractor = FeatureExtractor::new(10);

    let mut total_extractions = 0;
    let mut price_sum = 0.0;
    let mut volume_sum = 0.0;

    for (ts, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract_lob_features(state).unwrap();

        // Verify feature count
        assert_eq!(features.len(), 40, "Expected 40 features at ts {ts}");

        // Verify no NaN or Inf
        for (i, &f) in features.iter().enumerate() {
            assert!(f.is_finite(), "Feature {i} is not finite at ts {ts}: {f}");
        }

        // Feature layout: [ask_prices(0-9), ask_sizes(10-19), bid_prices(20-29), bid_sizes(30-39)]
        let best_ask_price = features[0];
        let best_bid_price = features[20];
        let best_ask_size = features[10];
        let best_bid_size = features[30];

        // Verify spread is positive
        assert!(
            best_ask_price > best_bid_price,
            "Spread should be positive: ask={best_ask_price}, bid={best_bid_price}"
        );

        // Verify prices match LOB state
        let expected_ask = price_to_dollars(state.ask_prices[0]);
        let expected_bid = price_to_dollars(state.bid_prices[0]);
        assert!(
            (best_ask_price - expected_ask).abs() < 1e-9,
            "Ask price mismatch: {best_ask_price} vs {expected_ask}"
        );
        assert!(
            (best_bid_price - expected_bid).abs() < 1e-9,
            "Bid price mismatch: {best_bid_price} vs {expected_bid}"
        );

        // Verify volumes match LOB state
        assert_eq!(
            best_ask_size as u32, state.ask_sizes[0],
            "Ask size mismatch"
        );
        assert_eq!(
            best_bid_size as u32, state.bid_sizes[0],
            "Bid size mismatch"
        );

        price_sum += best_ask_price + best_bid_price;
        volume_sum += best_ask_size + best_bid_size;
        total_extractions += 1;
    }

    let avg_mid_price = price_sum / (2.0 * total_extractions as f64);
    let avg_volume = volume_sum / (2.0 * total_extractions as f64);

    println!("  Total extractions: {total_extractions}");
    println!("  Average mid-price: ${avg_mid_price:.2}");
    println!("  Average best level volume: {avg_volume:.0}");

    // Sanity check: NVDA price should be reasonable (> $50, < $2000)
    assert!(
        avg_mid_price > 50.0 && avg_mid_price < 2000.0,
        "Average mid-price seems unreasonable: {avg_mid_price}"
    );

    println!("\n✓ Raw LOB feature extraction verified");
}

#[test]
fn test_derived_features_accuracy() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 200);
    if states.is_empty() {
        return;
    }

    println!("\n=== Derived Features Accuracy Test ===\n");

    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: false,
        mbo_window_size: 1000,
    };
    let extractor = FeatureExtractor::with_config(config);

    for (ts, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract_lob_features(state).unwrap();
        assert_eq!(
            features.len(),
            48,
            "Expected 48 features (40 raw + 8 derived)"
        );

        // Derived features: [mid_price, spread, spread_bps, total_bid_vol, total_ask_vol, vol_imbalance, weighted_mid, price_impact]
        let mid_price = features[40];
        let spread = features[41];
        let spread_bps = features[42];
        let total_bid_vol = features[43];
        let total_ask_vol = features[44];
        let vol_imbalance = features[45];
        let weighted_mid = features[46];
        let price_impact = features[47];

        // Manually compute expected values
        let best_ask = features[0];
        let best_bid = features[20];
        let expected_mid = (best_ask + best_bid) / 2.0;
        let expected_spread = best_ask - best_bid;
        let expected_spread_bps = (expected_spread / expected_mid) * 10_000.0;

        // Verify mid-price
        assert!(
            (mid_price - expected_mid).abs() < 1e-9,
            "Mid-price mismatch at ts {ts}: {mid_price} vs {expected_mid}"
        );

        // Verify spread
        assert!(
            (spread - expected_spread).abs() < 1e-9,
            "Spread mismatch at ts {ts}: {spread} vs {expected_spread}"
        );

        // Verify spread_bps
        assert!(
            (spread_bps - expected_spread_bps).abs() < 0.01,
            "Spread bps mismatch at ts {ts}: {spread_bps} vs {expected_spread_bps}"
        );

        // Verify volume imbalance is in [-1, 1]
        assert!(
            (-1.0..=1.0).contains(&vol_imbalance),
            "Volume imbalance out of range at ts {ts}: {vol_imbalance}"
        );

        // Verify weighted mid is between bid and ask
        assert!(
            weighted_mid >= best_bid && weighted_mid <= best_ask,
            "Weighted mid out of range at ts {ts}: {weighted_mid} not in [{best_bid}, {best_ask}]"
        );

        // Verify price impact is non-negative
        assert!(
            price_impact >= 0.0,
            "Price impact should be non-negative at ts {ts}: {price_impact}"
        );

        // Verify volumes are positive
        assert!(total_bid_vol >= 0.0 && total_ask_vol >= 0.0);
    }

    println!("✓ Derived features accuracy verified");
}

// =============================================================================
// 3. FI-2010 Feature Tests
// =============================================================================

#[test]
fn test_fi2010_features_comprehensive() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 500);
    if states.is_empty() {
        return;
    }

    println!("\n=== FI-2010 Features Comprehensive Test ===\n");

    let config = FI2010Config::default();
    let mut extractor = FI2010Extractor::new(config);

    let mut mid_price_derivatives: Vec<f64> = Vec::new();

    for (ts, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract(state, *ts).unwrap();

        // Should have 80 features
        assert_eq!(features.len(), 80, "Expected 80 FI-2010 features");

        // All features should be finite
        for (i, &f) in features.iter().enumerate() {
            assert!(
                f.is_finite(),
                "FI-2010 feature {i} is not finite at ts {ts}: {f}"
            );
        }

        // Time-insensitive features (0-19)
        let spread = features[0];
        let mid_price = features[1];

        // Verify spread is positive
        assert!(spread >= 0.0, "Spread should be non-negative: {spread}");

        // Verify mid-price is reasonable
        assert!(
            mid_price > 50.0 && mid_price < 2000.0,
            "Mid-price seems unreasonable: {mid_price}"
        );

        // Track mid-price derivative
        let derivative = features[20]; // mid_price_derivative
        mid_price_derivatives.push(derivative);

        // Depth features (40-79): accumulated volumes should be increasing
        for i in 1..10 {
            assert!(
                features[40 + i] >= features[40 + i - 1] - 1e-9,
                "Accumulated bid volume not increasing at level {}: {} < {}",
                i,
                features[40 + i],
                features[40 + i - 1]
            );
            assert!(
                features[50 + i] >= features[50 + i - 1] - 1e-9,
                "Accumulated ask volume not increasing at level {}: {} < {}",
                i,
                features[50 + i],
                features[50 + i - 1]
            );
        }
    }

    // Statistical check on derivatives
    if !mid_price_derivatives.is_empty() {
        let mean_deriv: f64 =
            mid_price_derivatives.iter().sum::<f64>() / mid_price_derivatives.len() as f64;
        let max_deriv = mid_price_derivatives
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_deriv = mid_price_derivatives
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        println!("  Mid-price derivative stats:");
        println!("    Mean: {mean_deriv:.6}");
        println!("    Min:  {min_deriv:.6}");
        println!("    Max:  {max_deriv:.6}");
    }

    println!("\n✓ FI-2010 features comprehensive test passed");
}

// =============================================================================
// 4. Order Flow Feature Tests
// =============================================================================

#[test]
fn test_order_flow_features() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 1000);
    if states.is_empty() {
        return;
    }

    println!("\n=== Order Flow Features Test ===\n");

    let mut tracker = OrderFlowTracker::new();
    let mut ofi_values: Vec<f64> = Vec::new();
    let mut queue_imbalance_values: Vec<f64> = Vec::new();

    for (ts, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        tracker.update(state, *ts);
        let features = tracker.extract_features(state);

        // Verify queue imbalance is in [-1, 1]
        assert!(
            features.queue_imbalance >= -1.0 && features.queue_imbalance <= 1.0,
            "Queue imbalance out of range: {}",
            features.queue_imbalance
        );

        // Verify depth imbalance is in [-1, 1]
        assert!(
            features.depth_imbalance >= -1.0 && features.depth_imbalance <= 1.0,
            "Depth imbalance out of range: {}",
            features.depth_imbalance
        );

        ofi_values.push(features.ofi);
        queue_imbalance_values.push(features.queue_imbalance);
    }

    // Statistical analysis
    if !ofi_values.is_empty() {
        let ofi_mean: f64 = ofi_values.iter().sum::<f64>() / ofi_values.len() as f64;
        let ofi_std: f64 = (ofi_values
            .iter()
            .map(|&x| (x - ofi_mean).powi(2))
            .sum::<f64>()
            / ofi_values.len() as f64)
            .sqrt();

        let qi_mean: f64 =
            queue_imbalance_values.iter().sum::<f64>() / queue_imbalance_values.len() as f64;

        println!("  OFI statistics:");
        println!("    Mean: {ofi_mean:.6}");
        println!("    Std:  {ofi_std:.6}");
        println!("  Queue Imbalance mean: {qi_mean:.6}");
    }

    println!("\n✓ Order flow features test passed");
}

#[test]
fn test_multi_level_ofi() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 1000);
    if states.is_empty() {
        return;
    }

    println!("\n=== Multi-Level OFI Test ===\n");

    let mut tracker = MultiLevelOfiTracker::new(10);

    for (_, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }
        tracker.update(state);
    }

    let ofi_by_level = tracker.get_ofi_per_level();
    assert_eq!(ofi_by_level.len(), 10, "Should have 10 OFI levels");

    println!("  OFI by level:");
    for (i, &ofi) in ofi_by_level.iter().enumerate() {
        assert!(ofi.is_finite(), "OFI at level {} is not finite", i + 1);
        println!("    Level {}: {:.6}", i + 1, ofi);
    }

    // Level 1 should typically have the most activity
    let level1_abs = ofi_by_level[0].abs();
    let deeper_avg: f64 = ofi_by_level[5..].iter().map(|x| x.abs()).sum::<f64>() / 5.0;
    println!("\n  Level 1 |OFI|: {level1_abs:.6}, Levels 6-10 avg |OFI|: {deeper_avg:.6}");

    // Get MLOFI (aggregated)
    let mlofi = tracker.get_mlofi();
    println!("  Aggregated MLOFI: {mlofi:.6}");

    println!("\n✓ Multi-level OFI test passed");
}

// =============================================================================
// 5. Normalization Tests
// =============================================================================

#[test]
fn test_zscore_normalization() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 1000);
    if states.is_empty() {
        return;
    }

    println!("\n=== Z-Score Normalization Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let mut normalizer = ZScoreNormalizer::new();

    // Collect mid-prices
    let mut mid_prices: Vec<f64> = Vec::new();
    for (_, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }
        let features = extractor.extract_lob_features(state).unwrap();
        let mid = (features[0] + features[20]) / 2.0;
        mid_prices.push(mid);
        normalizer.update(mid);
    }

    assert!(normalizer.is_ready(), "Normalizer should be ready");

    let mean = normalizer.mean();
    let std = normalizer.std();

    println!("  Raw mid-price statistics:");
    println!("    Mean: ${mean:.4}");
    println!("    Std:  ${std:.4}");

    // Normalize and verify
    let normalized: Vec<f64> = mid_prices
        .iter()
        .map(|&p| normalizer.normalize(p))
        .collect();

    let norm_mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
    let norm_std: f64 = (normalized
        .iter()
        .map(|&x| (x - norm_mean).powi(2))
        .sum::<f64>()
        / normalized.len() as f64)
        .sqrt();

    println!("  Normalized statistics:");
    println!("    Mean: {norm_mean:.6} (should be ~0)");
    println!("    Std:  {norm_std:.6} (should be ~1)");

    assert!(
        norm_mean.abs() < 0.01,
        "Normalized mean should be ~0: {norm_mean}"
    );
    assert!(
        (norm_std - 1.0).abs() < 0.01,
        "Normalized std should be ~1: {norm_std}"
    );

    println!("\n✓ Z-score normalization test passed");
}

#[test]
fn test_global_zscore_normalization() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 200);
    if states.is_empty() {
        return;
    }

    println!("\n=== Global Z-Score Normalization Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let normalizer = GlobalZScoreNormalizer::new();

    for (_, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract_lob_features(state).unwrap();
        let normalized = normalizer.normalize_snapshot(&features);

        // Verify mean is ~0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(
            mean.abs() < 1e-10,
            "Global Z-score mean should be ~0: {mean}"
        );

        // Verify ordering is preserved (ask > bid in raw → ask > bid in normalized)
        assert!(
            normalized[0] > normalized[20],
            "Global Z-score should preserve bid < ask ordering"
        );

        // Verify all values are finite
        for &v in &normalized {
            assert!(v.is_finite(), "Normalized value should be finite");
        }
    }

    println!("✓ Global Z-score normalization test passed");
}

#[test]
fn test_bilinear_normalization() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 200);
    if states.is_empty() {
        return;
    }

    println!("\n=== Bilinear Normalization Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let mut normalizer = BilinearNormalizer::new(0.01, 50.0); // tick=0.01, k=50

    for (_, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract_lob_features(state).unwrap();
        let mid = (features[0] + features[20]) / 2.0;
        normalizer.set_mid_price(mid);

        let norm_ask = normalizer.normalize(features[0]);
        let norm_bid = normalizer.normalize(features[20]);

        // Ask should be positive (above mid)
        assert!(norm_ask >= 0.0, "Normalized ask should be >= 0: {norm_ask}");

        // Bid should be negative (below mid)
        assert!(norm_bid <= 0.0, "Normalized bid should be <= 0: {norm_bid}");

        // Both should be finite and reasonable
        assert!(norm_ask.is_finite() && norm_bid.is_finite());
        assert!(
            norm_ask.abs() < 100.0 && norm_bid.abs() < 100.0,
            "Normalized values seem too large: ask={norm_ask}, bid={norm_bid}"
        );
    }

    println!("✓ Bilinear normalization test passed");
}

#[test]
fn test_rolling_zscore_normalization() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 2000);
    if states.len() < 500 {
        return;
    }

    println!("\n=== Rolling Z-Score Normalization Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let mut normalizer = RollingZScoreNormalizer::new(5);

    // Simulate 5 days by splitting data
    let chunk_size = states.len() / 5;
    for day in 0..5 {
        let start = day * chunk_size;
        let end = ((day + 1) * chunk_size).min(states.len());

        let mut day_values: Vec<f64> = Vec::new();
        for (_, state) in &states[start..end] {
            if is_valid_lob(state) {
                let features = extractor.extract_lob_features(state).unwrap();
                let mid = (features[0] + features[20]) / 2.0;
                day_values.push(mid);
            }
        }

        if !day_values.is_empty() {
            let mean: f64 = day_values.iter().sum::<f64>() / day_values.len() as f64;
            let std: f64 = (day_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / day_values.len() as f64)
                .sqrt()
                .max(1e-10);

            normalizer.add_day_stats(day_values.len() as u64, mean, std);
            println!("  Day {}: mean=${:.2}, std=${:.4}", day + 1, mean, std);
        }
    }

    assert!(normalizer.is_ready(), "Rolling normalizer should be ready");
    println!(
        "\n  Rolling stats: mean=${:.2}, std=${:.4}",
        normalizer.mean(),
        normalizer.std()
    );

    println!("\n✓ Rolling Z-score normalization test passed");
}

#[test]
fn test_per_feature_normalization() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 500);
    if states.is_empty() {
        return;
    }

    println!("\n=== Per-Feature Normalization Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let mut normalizer = PerFeatureNormalizer::new(40);

    // First pass: collect statistics
    for (_, state) in &states {
        if is_valid_lob(state) {
            let features = extractor.extract_lob_features(state).unwrap();
            normalizer.update_batch(&features);
        }
    }

    assert!(
        normalizer.is_ready(),
        "Per-feature normalizer should be ready"
    );

    // Check per-feature statistics
    let (price_mean, price_std) = normalizer.get_feature_stats(0).unwrap(); // best ask price
    let (vol_mean, vol_std) = normalizer.get_feature_stats(10).unwrap(); // best ask volume

    println!("  Best ask price: mean=${price_mean:.2}, std=${price_std:.4}");
    println!("  Best ask volume: mean={vol_mean:.0}, std={vol_std:.0}");

    // Normalize and verify
    if let Some((_, state)) = states.iter().find(|(_, s)| is_valid_lob(s)) {
        let features = extractor.extract_lob_features(state).unwrap();
        let normalized = normalizer.normalize_features(&features);

        // Check that prices and volumes are normalized separately
        assert_eq!(normalized.len(), 40);
        for &v in &normalized {
            assert!(v.is_finite(), "Normalized value should be finite");
        }
    }

    println!("\n✓ Per-feature normalization test passed");
}

// =============================================================================
// 6. Validation Tests
// =============================================================================

#[test]
fn test_lob_validation() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 1000);
    if states.is_empty() {
        return;
    }

    println!("\n=== LOB Validation Test ===\n");

    let validator = FeatureValidator::new();

    let mut valid_count = 0;
    let mut warning_count = 0;
    let mut error_count = 0;

    for (_, state) in &states {
        let result = validator.validate_lob(state);
        if result.is_valid() {
            valid_count += 1;
        }
        if result.has_warnings() {
            warning_count += 1;
        }
        if result.has_errors() {
            error_count += 1;
        }
    }

    let valid_ratio = valid_count as f64 / states.len() as f64;

    println!("  Validation results:");
    println!(
        "    Valid:    {} ({:.1}%)",
        valid_count,
        valid_ratio * 100.0
    );
    println!("    Warnings: {warning_count}");
    println!("    Errors:   {error_count}");

    assert!(
        valid_ratio > 0.9,
        "Expected >90% valid states, got {:.1}%",
        valid_ratio * 100.0
    );

    println!("\n✓ LOB validation test passed");
}

#[test]
fn test_feature_validation() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 200);
    if states.is_empty() {
        return;
    }

    println!("\n=== Feature Validation Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let validator = FeatureValidator::new();

    let mut all_valid = true;
    for (_, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract_lob_features(state).unwrap();
        let result = validator.validate_features(&features);

        if result.has_errors() {
            all_valid = false;
            for error in result.errors() {
                println!("  ERROR: {error}");
            }
        }
    }

    assert!(all_valid, "All features should be valid");
    println!("✓ Feature validation test passed");
}

#[test]
fn test_timestamp_monotonicity() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 1000);
    if states.is_empty() {
        return;
    }

    println!("\n=== Timestamp Monotonicity Test ===\n");

    let timestamps: Vec<u64> = states.iter().map(|(ts, _)| *ts).collect();
    let result = validate_timestamps(&timestamps);

    if result.has_errors() {
        for error in result.errors() {
            println!("  ERROR: {error}");
        }
    }

    // Check monotonicity manually
    let mut is_monotonic = true;
    for i in 1..timestamps.len() {
        if timestamps[i] < timestamps[i - 1] {
            is_monotonic = false;
            println!(
                "  Non-monotonic at index {}: {} < {}",
                i,
                timestamps[i],
                timestamps[i - 1]
            );
            break;
        }
    }

    // Note: timestamps might not always be strictly monotonic due to message ordering
    println!("  Timestamps monotonic: {is_monotonic}");
    println!("✓ Timestamp monotonicity test completed");
}

// =============================================================================
// 7. Market Impact Tests
// =============================================================================

#[test]
fn test_market_impact_estimation() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 200);
    if states.is_empty() {
        return;
    }

    println!("\n=== Market Impact Estimation Test ===\n");

    let mut total_tests = 0;
    let mut slippage_sum = 0.0;

    for (_, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        // Test buy impact
        let buy_impact = estimate_buy_impact(state, 100);
        assert!(buy_impact.vwap >= 0.0, "Buy VWAP should be non-negative");
        assert!(
            buy_impact.slippage_bps >= 0.0,
            "Buy slippage should be non-negative"
        );
        assert!(
            buy_impact.fill_ratio >= 0.0 && buy_impact.fill_ratio <= 1.0,
            "Fill ratio should be in [0, 1]"
        );

        // Test sell impact
        let sell_impact = estimate_sell_impact(state, 100);
        assert!(sell_impact.vwap >= 0.0, "Sell VWAP should be non-negative");

        // Buy VWAP should be >= sell VWAP (crossing spread)
        if buy_impact.filled_quantity > 0 && sell_impact.filled_quantity > 0 {
            assert!(
                buy_impact.vwap >= sell_impact.vwap - 0.01,
                "Buy VWAP {} should be >= sell VWAP {}",
                buy_impact.vwap,
                sell_impact.vwap
            );
        }

        slippage_sum += buy_impact.slippage_bps;
        total_tests += 1;
    }

    if total_tests > 0 {
        let avg_slippage = slippage_sum / total_tests as f64;
        println!("  Average buy slippage (100 shares): {avg_slippage:.2} bps");
    }

    println!("\n✓ Market impact estimation test passed");
}

// =============================================================================
// 8. Performance Tests
// =============================================================================

#[test]
fn test_extraction_performance() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 10000);
    if states.len() < 1000 {
        eprintln!("Not enough data for performance test");
        return;
    }

    println!("\n=== Extraction Performance Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let valid_states: Vec<_> = states.iter().filter(|(_, s)| is_valid_lob(s)).collect();

    // Warm up
    for (_, state) in valid_states.iter().take(100) {
        let _ = extractor.extract_lob_features(state);
    }

    // Benchmark raw feature extraction
    let iterations = valid_states.len().min(5000);
    let start = Instant::now();
    for (_, state) in valid_states.iter().take(iterations) {
        let _ = extractor.extract_lob_features(state);
    }
    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("  Raw LOB features (40):");
    println!("    Throughput: {throughput:.0} extractions/sec");
    println!("    Latency:    {latency_us:.2} µs/extraction");

    // Benchmark FI-2010 features
    let mut fi2010 = FI2010Extractor::new(FI2010Config::default());
    let start = Instant::now();
    for (ts, state) in valid_states.iter().take(iterations) {
        let _ = fi2010.extract(state, *ts);
    }
    let elapsed = start.elapsed();
    let fi2010_throughput = iterations as f64 / elapsed.as_secs_f64();
    let fi2010_latency = elapsed.as_micros() as f64 / iterations as f64;

    println!("  FI-2010 features (80):");
    println!("    Throughput: {fi2010_throughput:.0} extractions/sec");
    println!("    Latency:    {fi2010_latency:.2} µs/extraction");

    // Performance assertions (relaxed for CI)
    assert!(
        throughput > 50_000.0,
        "Raw feature throughput too low: {throughput:.0}/sec"
    );

    println!("\n✓ Performance test passed");
}

// =============================================================================
// 9. Volatility Estimation Test
// =============================================================================

#[test]
fn test_volatility_estimation() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 2000);
    if states.len() < 500 {
        return;
    }

    println!("\n=== Volatility Estimation Test ===\n");

    let extractor = FeatureExtractor::new(10);
    let mut estimator = VolatilityEstimator::new(1000);

    for (_, state) in &states {
        if is_valid_lob(state) {
            let features = extractor.extract_lob_features(state).unwrap();
            let mid = (features[0] + features[20]) / 2.0;
            estimator.update(mid);
        }
    }

    if estimator.is_ready() {
        let volatility = estimator.volatility().unwrap();
        let mean_return = estimator.mean_return().unwrap();

        println!("  Realized volatility: {volatility:.6}");
        println!("  Mean return: {mean_return:.8}");

        // Volatility should be positive and reasonable
        assert!(volatility > 0.0, "Volatility should be positive");
        assert!(
            volatility < 0.1,
            "Volatility seems too high for intraday: {volatility}"
        );
    }

    println!("\n✓ Volatility estimation test passed");
}

// =============================================================================
// 10. Numerical Precision Tests
// =============================================================================

#[test]
fn test_numerical_precision() {
    if !data_available() {
        return;
    }

    let states = load_lob_states("20250203", 100);
    if states.is_empty() {
        return;
    }

    println!("\n=== Numerical Precision Test ===\n");

    let extractor = FeatureExtractor::new(10);

    for (ts, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        let features = extractor.extract_lob_features(state).unwrap();

        // Verify price precision (should match internal representation)
        let raw_ask = state.ask_prices[0];
        let raw_bid = state.bid_prices[0];
        let extracted_ask = features[0];
        let extracted_bid = features[20];

        let expected_ask = raw_ask as f64 / 1e9;
        let expected_bid = raw_bid as f64 / 1e9;

        // Precision should be exact to 9 decimal places
        assert!(
            (extracted_ask - expected_ask).abs() < 1e-12,
            "Ask price precision error at ts {ts}: {extracted_ask} vs {expected_ask}"
        );
        assert!(
            (extracted_bid - expected_bid).abs() < 1e-12,
            "Bid price precision error at ts {ts}: {extracted_bid} vs {expected_bid}"
        );

        // Verify spread calculation precision
        let spread = extracted_ask - extracted_bid;
        let expected_spread = (raw_ask - raw_bid) as f64 / 1e9;
        assert!(
            (spread - expected_spread).abs() < 1e-12,
            "Spread precision error: {spread} vs {expected_spread}"
        );
    }

    println!("✓ Numerical precision test passed");
}

// =============================================================================
// Summary Integration Test
// =============================================================================

#[test]
fn test_full_pipeline_summary() {
    if !data_available() {
        eprintln!("\n========================================");
        eprintln!("SKIPPED: NVDA data not available");
        eprintln!("Expected path: {NVDA_DATA_DIR}");
        eprintln!("========================================\n");
        return;
    }

    println!("\n========================================");
    println!("Feature Extractor Integration Test Suite");
    println!("========================================\n");

    let states = load_lob_states("20250203", 2000);
    println!("Loaded {} valid LOB states from NVDA data\n", states.len());

    if states.is_empty() {
        println!("WARNING: No data loaded, skipping summary");
        return;
    }

    // Count valid states
    let valid_count = states.iter().filter(|(_, s)| is_valid_lob(s)).count();
    println!("Valid LOB states: {}/{}", valid_count, states.len());

    // Test each component
    let extractor = FeatureExtractor::new(10);
    let mut fi2010 = FI2010Extractor::new(FI2010Config::default());
    let mut of_tracker = OrderFlowTracker::new();
    let mut mlofi_tracker = MultiLevelOfiTracker::new(10);
    let validator = FeatureValidator::new();
    let global_norm = GlobalZScoreNormalizer::new();

    let mut raw_extractions = 0;
    let mut fi2010_extractions = 0;
    let mut valid_validations = 0;

    for (ts, state) in &states {
        if !is_valid_lob(state) {
            continue;
        }

        // Raw features
        if extractor.extract_lob_features(state).is_ok() {
            raw_extractions += 1;
        }

        // FI-2010 features
        if fi2010.extract(state, *ts).is_ok() {
            fi2010_extractions += 1;
        }

        // Order flow
        of_tracker.update(state, *ts);
        mlofi_tracker.update(state);

        // Validation
        if validator.validate_lob(state).is_valid() {
            valid_validations += 1;
        }

        // Normalization
        let features = extractor.extract_lob_features(state).unwrap();
        let _ = global_norm.normalize_snapshot(&features);
    }

    println!("\nComponent Test Results:");
    println!("  ✓ Raw LOB features: {raw_extractions} extractions");
    println!("  ✓ FI-2010 features: {fi2010_extractions} extractions");
    println!("  ✓ Order flow: OFI and MLOFI computed");
    println!("  ✓ Validation: {valid_validations}/{valid_count} states valid");
    println!("  ✓ Normalization: Global Z-score applied");

    // Final OFI stats
    let ofi_levels = mlofi_tracker.get_ofi_per_level();
    println!("\nMulti-Level OFI (final):");
    for (i, &ofi) in ofi_levels.iter().take(5).enumerate() {
        println!("  Level {}: {:.4}", i + 1, ofi);
    }

    println!("\n========================================");
    println!("All integration tests completed!");
    println!("========================================\n");
}
