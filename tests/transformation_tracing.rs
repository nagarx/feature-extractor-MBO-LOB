//! Level 1: Transformation Tracing Tests
//!
//! Processes the first 50,000 MBO events from Feb 3 through each pipeline stage
//! independently, verifying exact intermediate values with formula-based
//! expectations at every step.
//!
//! 50K events (~0.5-1% of a typical NVDA day) is sufficient for all 98 feature
//! formulas to be exercised, OFI to reach warm state, MBO windows to accumulate
//! meaningful rolling statistics, and order flow features to show variance.
//!
//! This is the highest-value test level: it catches bugs that modify a single
//! feature's formula without affecting aggregate statistics.
//!
//! Run with:
//! ```bash
//! cargo test --features "parallel,databento" --test transformation_tracing -- --test-threads=1
//! ```

#![cfg(feature = "parallel")]

mod common;

use feature_extractor::contract;
use feature_extractor::features::derived_features::compute_derived_features;
use feature_extractor::features::lob_features::extract_raw_features;
use feature_extractor::features::mbo_features::{MboAggregator, MboEvent};
use feature_extractor::features::signals;
use feature_extractor::{FeatureConfig, FeatureExtractor};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor, LobState, MboMessage};

const FEB3_DATE: &str = "20250203";
const EVENT_LIMIT: usize = 50_000;
const CHECKPOINT_EVENTS: &[usize] = &[100, 500, 1_000, 5_000, 10_000, 25_000, 50_000];

/// Absolute tolerance for deterministic formula checks (f64 precision).
const FORMULA_TOL: f64 = contract::FLOAT_CMP_EPS;

fn load_feb3_messages() -> Option<Vec<MboMessage>> {
    let path = common::find_mbo_file(FEB3_DATE)?;
    let loader = DbnLoader::new(path).ok()?;
    let msgs: Vec<MboMessage> = loader.iter_messages().ok()?.take(EVENT_LIMIT).collect();
    if msgs.is_empty() {
        None
    } else {
        Some(msgs)
    }
}

fn is_valid_msg(msg: &MboMessage) -> bool {
    msg.order_id != 0 && msg.size > 0 && msg.price > 0
}

/// Reconstruct LOB state by processing `count` valid messages.
/// Returns the LOB state after processing and the count of valid messages seen.
fn reconstruct_lob_at(messages: &[MboMessage], count: usize, levels: usize) -> (LobState, usize) {
    let mut lob = LobReconstructor::new(levels);
    let mut lob_state = LobState::new(levels);
    let mut valid_count = 0;

    for msg in messages {
        if !is_valid_msg(msg) {
            continue;
        }
        let _ = lob.process_message_into(msg, &mut lob_state);
        valid_count += 1;
        if valid_count >= count {
            break;
        }
    }

    (lob_state, valid_count)
}

// =========================================================================
// Test 1a: LOB Reconstruction Accuracy
// =========================================================================

#[test]
fn test_lob_reconstruction_at_checkpoints() {
    skip_if_no_data!();

    let messages = match load_feb3_messages() {
        Some(m) => m,
        None => {
            eprintln!("Skipping: no data for {FEB3_DATE}");
            return;
        }
    };

    let levels = 10;

    for &checkpoint in CHECKPOINT_EVENTS {
        let (lob_state, valid_count) = reconstruct_lob_at(&messages, checkpoint, levels);
        if valid_count < checkpoint {
            eprintln!(
                "Only {valid_count} valid messages in first {EVENT_LIMIT}, \
                 skipping checkpoint {checkpoint}"
            );
            continue;
        }

        if lob_state.best_bid.is_none() || lob_state.best_ask.is_none() {
            continue;
        }

        let best_bid = lob_state.best_bid.unwrap();
        let best_ask = lob_state.best_ask.unwrap();

        assert!(
            best_bid > 0,
            "Checkpoint {checkpoint}: best_bid must be positive, got {best_bid}"
        );
        assert!(
            best_ask > 0,
            "Checkpoint {checkpoint}: best_ask must be positive, got {best_ask}"
        );
        assert!(
            best_bid < best_ask,
            "Checkpoint {checkpoint}: best_bid ({best_bid}) must be < best_ask ({best_ask})"
        );

        assert_eq!(
            lob_state.bid_prices[0], best_bid,
            "Checkpoint {checkpoint}: bid_prices[0] must equal best_bid"
        );
        assert_eq!(
            lob_state.ask_prices[0], best_ask,
            "Checkpoint {checkpoint}: ask_prices[0] must equal best_ask"
        );

        for i in 0..levels {
            if lob_state.bid_prices[i] > 0 && i > 0 && lob_state.bid_prices[i - 1] > 0 {
                assert!(
                    lob_state.bid_prices[i] <= lob_state.bid_prices[i - 1],
                    "Checkpoint {checkpoint}: bid prices must be non-increasing: \
                     level {i} ({}) > level {} ({})",
                    lob_state.bid_prices[i],
                    i - 1,
                    lob_state.bid_prices[i - 1]
                );
            }
            if lob_state.ask_prices[i] > 0 && i > 0 && lob_state.ask_prices[i - 1] > 0 {
                assert!(
                    lob_state.ask_prices[i] >= lob_state.ask_prices[i - 1],
                    "Checkpoint {checkpoint}: ask prices must be non-decreasing: \
                     level {i} ({}) < level {} ({})",
                    lob_state.ask_prices[i],
                    i - 1,
                    lob_state.ask_prices[i - 1]
                );
            }
        }
    }
}

// =========================================================================
// Test 1b: Raw LOB Feature Extraction Verification
// =========================================================================

#[test]
fn test_raw_lob_feature_extraction() {
    skip_if_no_data!();

    let messages = match load_feb3_messages() {
        Some(m) => m,
        None => {
            eprintln!("Skipping: no data for {FEB3_DATE}");
            return;
        }
    };

    let levels = 10;
    let expected_raw_count = levels * 4; // ask_prices, ask_sizes, bid_prices, bid_sizes

    for &checkpoint in CHECKPOINT_EVENTS {
        let (lob_state, valid_count) = reconstruct_lob_at(&messages, checkpoint, levels);
        if valid_count < checkpoint {
            continue;
        }
        if lob_state.best_bid.is_none() || lob_state.best_ask.is_none() {
            continue;
        }

        let mut features = Vec::new();
        extract_raw_features(&lob_state, levels, &mut features);

        assert_eq!(
            features.len(),
            expected_raw_count,
            "Checkpoint {checkpoint}: expected {expected_raw_count} raw features, got {}",
            features.len()
        );

        for i in 0..levels {
            let expected_ask_price = lob_state.ask_prices[i] as f64 / 1e9;
            let expected_ask_size = lob_state.ask_sizes[i] as f64;
            let expected_bid_price = lob_state.bid_prices[i] as f64 / 1e9;
            let expected_bid_size = lob_state.bid_sizes[i] as f64;

            let actual_ask_price = features[i];
            let actual_ask_size = features[levels + i];
            let actual_bid_price = features[2 * levels + i];
            let actual_bid_size = features[3 * levels + i];

            let ctx = format!("Checkpoint {checkpoint}, level {i}");
            common::assertions::assert_f64_eq(
                actual_ask_price,
                expected_ask_price,
                &format!("{ctx} ask_price"),
            );
            common::assertions::assert_f64_eq(
                actual_ask_size,
                expected_ask_size,
                &format!("{ctx} ask_size"),
            );
            common::assertions::assert_f64_eq(
                actual_bid_price,
                expected_bid_price,
                &format!("{ctx} bid_price"),
            );
            common::assertions::assert_f64_eq(
                actual_bid_size,
                expected_bid_size,
                &format!("{ctx} bid_size"),
            );
        }

        common::assertions::assert_features_finite(
            &features,
            &format!("Checkpoint {checkpoint} raw features"),
        );
    }
}

// =========================================================================
// Test 1c: Derived Feature Formula Verification
// =========================================================================

#[test]
fn test_derived_feature_formulas() {
    skip_if_no_data!();

    let messages = match load_feb3_messages() {
        Some(m) => m,
        None => {
            eprintln!("Skipping: no data for {FEB3_DATE}");
            return;
        }
    };

    let levels = 10;

    for &checkpoint in CHECKPOINT_EVENTS {
        let (lob_state, valid_count) = reconstruct_lob_at(&messages, checkpoint, levels);
        if valid_count < checkpoint {
            continue;
        }
        if lob_state.best_bid.is_none() || lob_state.best_ask.is_none() {
            continue;
        }

        let derived = compute_derived_features(&lob_state, levels)
            .expect("compute_derived_features should succeed with valid LOB");

        let best_bid_f64 = lob_state.best_bid.unwrap() as f64 / 1e9;
        let best_ask_f64 = lob_state.best_ask.unwrap() as f64 / 1e9;

        // 1. Mid-price: (best_bid + best_ask) / 2
        let expected_mid = (best_bid_f64 + best_ask_f64) / 2.0;
        common::assertions::assert_f64_eq(
            derived[0],
            expected_mid,
            &format!("Checkpoint {checkpoint} mid_price"),
        );

        // 2. Spread: best_ask - best_bid
        let expected_spread = best_ask_f64 - best_bid_f64;
        common::assertions::assert_f64_eq(
            derived[1],
            expected_spread,
            &format!("Checkpoint {checkpoint} spread"),
        );

        // 3. Spread bps: (spread / mid_price) * 10_000
        let expected_spread_bps = if expected_mid > 0.0 {
            (expected_spread / expected_mid) * 10_000.0
        } else {
            0.0
        };
        common::assertions::assert_f64_eq(
            derived[2],
            expected_spread_bps,
            &format!("Checkpoint {checkpoint} spread_bps"),
        );

        // 4 & 5. Total bid/ask volumes
        let mut total_bid: f64 = 0.0;
        let mut total_ask: f64 = 0.0;
        for i in 0..levels.min(lob_state.levels) {
            total_bid += lob_state.bid_sizes[i] as u64 as f64;
            total_ask += lob_state.ask_sizes[i] as u64 as f64;
        }
        common::assertions::assert_f64_eq(
            derived[3],
            total_bid,
            &format!("Checkpoint {checkpoint} total_bid_volume"),
        );
        common::assertions::assert_f64_eq(
            derived[4],
            total_ask,
            &format!("Checkpoint {checkpoint} total_ask_volume"),
        );

        // 6. Volume imbalance: (bid - ask) / (bid + ask)
        let total = total_bid + total_ask;
        let expected_imbalance = if total > 0.0 {
            (total_bid - total_ask) / total
        } else {
            0.0
        };
        common::assertions::assert_f64_eq(
            derived[5],
            expected_imbalance,
            &format!("Checkpoint {checkpoint} volume_imbalance"),
        );

        // 7. Weighted mid-price: (bid_price * ask_size + ask_price * bid_size) / total_best_size
        let best_bid_size = lob_state.bid_sizes[0] as f64;
        let best_ask_size = lob_state.ask_sizes[0] as f64;
        let total_best = best_bid_size + best_ask_size;
        let expected_wmid = if total_best > 0.0 {
            (best_bid_f64 * best_ask_size + best_ask_f64 * best_bid_size) / total_best
        } else {
            expected_mid
        };
        common::assertions::assert_f64_eq(
            derived[6],
            expected_wmid,
            &format!("Checkpoint {checkpoint} weighted_mid_price"),
        );

        // 8. Price impact: |mid - weighted_mid|
        let expected_impact = (expected_mid - expected_wmid).abs();
        common::assertions::assert_f64_eq(
            derived[7],
            expected_impact,
            &format!("Checkpoint {checkpoint} price_impact"),
        );
    }
}

// =========================================================================
// MBO Feature Range Validation Helper
// =========================================================================

/// Validates that all 36 MBO features are within their mathematically valid ranges.
///
/// Each bound is derived from the feature's formula -- impossible values (negative
/// counts, ratios > 1, rates outside clamped bounds) indicate formula bugs even
/// when exact expected values are impractical to compute.
///
/// Ranges by MBO local index:
///   [0-5]  event rates (count/s)         >= 0
///   [6-8]  net flow imbalances           [-1, 1]
///   [9]    aggressive_order_ratio        [0, 1]
///   [10]   order_flow_volatility         >= 0
///   [11]   flow_regime_indicator         [-10, 10]  (clamped)
///   [12-15] size percentiles             >= 0
///   [16]   size_zscore                   unbounded
///   [17]   large_order_ratio             [0, 1]
///   [18]   size_skewness                 unbounded
///   [19]   size_concentration (HHI)      [0, 1]
///   [20]   avg_queue_position            >= 0
///   [21]   queue_volume_ahead            >= 0
///   [22]   orders_per_level              >= 0
///   [23]   level_concentration (HHI)     [0, 1]
///   [24-25] depth_ticks bid/ask          [0, MAX_LOB_LEVELS)
///   [26]   large_order_frequency         >= 0
///   [27]   large_order_imbalance         [-1, 1]
///   [28]   modification_score            >= 0
///   [29]   iceberg_proxy                 [0, 1]
///   [30]   avg_order_age                 >= 0
///   [31]   median_order_lifetime         >= 0
///   [32]   avg_fill_ratio                [0, 1]
///   [33]   avg_time_to_first_fill        >= 0
///   [34]   cancel_to_add_ratio           [0, 10]  (clamped)
///   [35]   active_order_count            >= 0
fn assert_mbo_feature_ranges(features: &[f64], context: &str) {
    assert_eq!(
        features.len(),
        36,
        "{context}: expected 36 MBO features, got {}",
        features.len()
    );

    let tol = FORMULA_TOL;

    for (i, &v) in features.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{context}: MBO feature[{i}] = {v} is not finite"
        );
    }

    // --- Flow features [0-11] ---

    for (i, &v) in features[..6].iter().enumerate() {
        assert!(v >= -tol, "{context}: event_rate[{i}] = {v} must be >= 0");
    }

    for &i in &[6, 7, 8] {
        assert!(
            (-1.0 - tol..=1.0 + tol).contains(&features[i]),
            "{context}: net_flow[{i}] = {} must be in [-1, 1]",
            features[i]
        );
    }

    assert!(
        (-tol..=1.0 + tol).contains(&features[9]),
        "{context}: aggressive_order_ratio = {} must be in [0, 1]",
        features[9]
    );

    assert!(
        features[10] >= -tol,
        "{context}: order_flow_volatility = {} must be >= 0",
        features[10]
    );

    assert!(
        (-10.0 - tol..=10.0 + tol).contains(&features[11]),
        "{context}: flow_regime_indicator = {} must be in [-10, 10]",
        features[11]
    );

    // --- Size features [12-19] ---

    for (j, &v) in features[12..16].iter().enumerate() {
        let i = 12 + j;
        assert!(
            v >= -tol,
            "{context}: size_percentile[{i}] = {v} must be >= 0"
        );
    }
    // [16] size_zscore: unbounded, no range check
    assert!(
        (-tol..=1.0 + tol).contains(&features[17]),
        "{context}: large_order_ratio = {} must be in [0, 1]",
        features[17]
    );
    // [18] size_skewness: unbounded, no range check
    assert!(
        (-tol..=1.0 + tol).contains(&features[19]),
        "{context}: size_concentration = {} must be in [0, 1]",
        features[19]
    );

    // --- Queue features [20-25] ---

    for &i in &[20, 21, 22] {
        assert!(
            features[i] >= -tol,
            "{context}: queue_feature[{i}] = {} must be >= 0",
            features[i]
        );
    }

    assert!(
        (-tol..=1.0 + tol).contains(&features[23]),
        "{context}: level_concentration = {} must be in [0, 1]",
        features[23]
    );

    for &i in &[24, 25] {
        assert!(
            features[i] >= -tol,
            "{context}: depth_ticks[{i}] = {} must be >= 0",
            features[i]
        );
    }

    // --- Institutional features [26-29] ---

    assert!(
        features[26] >= -tol,
        "{context}: large_order_frequency = {} must be >= 0",
        features[26]
    );

    assert!(
        (-1.0 - tol..=1.0 + tol).contains(&features[27]),
        "{context}: large_order_imbalance = {} must be in [-1, 1]",
        features[27]
    );

    assert!(
        features[28] >= -tol,
        "{context}: modification_score = {} must be >= 0",
        features[28]
    );

    assert!(
        (-tol..=1.0 + tol).contains(&features[29]),
        "{context}: iceberg_proxy = {} must be in [0, 1]",
        features[29]
    );

    // --- Lifecycle features [30-35] ---

    for &i in &[30, 31] {
        assert!(
            features[i] >= -tol,
            "{context}: lifecycle_time[{i}] = {} must be >= 0",
            features[i]
        );
    }

    assert!(
        (-tol..=1.0 + tol).contains(&features[32]),
        "{context}: avg_fill_ratio = {} must be in [0, 1]",
        features[32]
    );

    assert!(
        features[33] >= -tol,
        "{context}: avg_time_to_first_fill = {} must be >= 0",
        features[33]
    );

    assert!(
        (-tol..=10.0 + tol).contains(&features[34]),
        "{context}: cancel_to_add_ratio = {} must be in [0, 10]",
        features[34]
    );

    assert!(
        features[35] >= -tol,
        "{context}: active_order_count = {} must be >= 0",
        features[35]
    );
}

// =========================================================================
// Test 1d: MBO Feature Spot-Check
// =========================================================================

#[test]
fn test_mbo_feature_spot_check() {
    skip_if_no_data!();

    let messages = match load_feb3_messages() {
        Some(m) => m,
        None => {
            eprintln!("Skipping: no data for {FEB3_DATE}");
            return;
        }
    };

    let levels = 10;
    let mut aggregator = MboAggregator::new();
    let mut lob = LobReconstructor::new(levels);
    let mut lob_state = LobState::new(levels);
    let mut valid_count = 0;

    for msg in &messages {
        if !is_valid_msg(msg) {
            continue;
        }
        let _ = lob.process_message_into(msg, &mut lob_state);
        let mbo_event = MboEvent::from_mbo_message(msg);
        aggregator.process_event(mbo_event);
        valid_count += 1;

        if CHECKPOINT_EVENTS.contains(&valid_count) {
            if lob_state.best_bid.is_none() || lob_state.best_ask.is_none() {
                continue;
            }

            let mbo_feats = aggregator.extract_features(&lob_state);

            assert_mbo_feature_ranges(&mbo_feats, &format!("Checkpoint {valid_count}"));
        }
    }

    assert!(
        valid_count >= CHECKPOINT_EVENTS[0],
        "Not enough valid messages to reach first checkpoint: {valid_count}"
    );
}

// =========================================================================
// Test 1e: Signal Computation Verification
// =========================================================================

#[test]
fn test_signal_computation_verification() {
    skip_if_no_data!();

    let messages = match load_feb3_messages() {
        Some(m) => m,
        None => {
            eprintln!("Skipping: no data for {FEB3_DATE}");
            return;
        }
    };

    let levels = 10;
    let mut lob = LobReconstructor::new(levels);
    let mut lob_state = LobState::new(levels);
    let mut ofi_computer = signals::OfiComputer::new();
    let mut valid_count: usize = 0;
    let sample_interval: usize = 500;

    for msg in &messages {
        if !is_valid_msg(msg) {
            continue;
        }
        let _ = lob.process_message_into(msg, &mut lob_state);
        ofi_computer.update(&lob_state);
        valid_count += 1;

        if valid_count.is_multiple_of(sample_interval)
            && lob_state.best_bid.is_some()
            && lob_state.best_ask.is_some()
        {
            let timestamp_ns = msg.timestamp.unwrap_or(0);
            let ofi_sample = ofi_computer.sample_and_reset(timestamp_ns);

            let book_valid = signals::is_book_valid_from_lob(&lob_state);

            let mut base_features = Vec::new();
            extract_raw_features(&lob_state, levels, &mut base_features);
            let derived = compute_derived_features(&lob_state, levels).unwrap();
            base_features.extend_from_slice(&derived);
            base_features.resize(84, 0.0);

            let signal_vec = signals::compute_signals_with_book_valid(
                &base_features,
                &ofi_sample,
                timestamp_ns,
                0,
                book_valid,
            );

            let signals = signal_vec.to_vec();
            assert_eq!(
                signals.len(),
                contract::SIGNAL_COUNT,
                "Expected {}, got {} signals",
                contract::SIGNAL_COUNT,
                signals.len()
            );

            // book_valid (signal index 8 within signals, absolute 92) is 0.0 or 1.0
            let bv = signals[signals::indices::BOOK_VALID - 84];
            assert!(
                (bv - 0.0).abs() < FORMULA_TOL || (bv - 1.0).abs() < FORMULA_TOL,
                "book_valid must be 0.0 or 1.0, got {bv}"
            );

            // Verify book_valid matches our independent computation
            let expected_bv = if book_valid { 1.0 } else { 0.0 };
            common::assertions::assert_f64_eq(
                bv,
                expected_bv,
                &format!("Event {valid_count} book_valid signal"),
            );

            // time_regime (signal index 9, absolute 93) is in {0,1,2,3,4,5}
            let tr = signals[signals::indices::TIME_REGIME - 84];
            assert!(
                (0.0..=5.0 + FORMULA_TOL).contains(&tr),
                "time_regime must be in [0,5], got {tr}"
            );

            // schema_version (signal index 13, absolute 97) matches contract
            let sv = signals[signals::indices::SCHEMA_VERSION - 84];
            common::assertions::assert_f64_eq(
                sv,
                contract::SCHEMA_VERSION,
                &format!("Event {valid_count} schema_version signal"),
            );

            common::assertions::assert_signal_basics(
                &signals,
                &format!("Event {valid_count} signals"),
            );
        }
    }
}

// =========================================================================
// Test 1f: Full 98-Feature Vector Composition
// =========================================================================

#[test]
fn test_full_98_feature_composition() {
    skip_if_no_data!();

    let messages = match load_feb3_messages() {
        Some(m) => m,
        None => {
            eprintln!("Skipping: no data for {FEB3_DATE}");
            return;
        }
    };

    let levels = 10;
    let config = FeatureConfig::new(levels)
        .with_derived(true)
        .with_mbo(true)
        .with_signals(true);
    let mut extractor = FeatureExtractor::with_config(config);

    let mut lob = LobReconstructor::new(levels);
    let mut lob_state = LobState::new(levels);
    let mut valid_count: usize = 0;
    let sample_interval: usize = 500;
    let mut verified_samples: usize = 0;

    for msg in &messages {
        if !is_valid_msg(msg) {
            continue;
        }
        let _ = lob.process_message_into(msg, &mut lob_state);
        extractor.update_ofi(&lob_state);
        let mbo_event = MboEvent::from_mbo_message(msg);
        extractor.process_mbo_event(mbo_event);
        valid_count += 1;

        if valid_count.is_multiple_of(sample_interval)
            && lob_state.best_bid.is_some()
            && lob_state.best_ask.is_some()
        {
            let timestamp_ns = msg.timestamp.unwrap_or(0) as u64;
            let ctx = feature_extractor::SignalContext {
                timestamp_ns,
                invalidity_delta: 0u64,
            };
            let mut full_vec = Vec::new();
            extractor
                .extract_with_signals(&lob_state, &ctx, &mut full_vec)
                .expect("extract_with_signals should succeed");

            assert_eq!(
                full_vec.len(),
                contract::STABLE_FEATURE_COUNT,
                "Event {valid_count}: expected {} features, got {}",
                contract::STABLE_FEATURE_COUNT,
                full_vec.len()
            );

            // Verify LOB features (indices 0-39)
            let mut raw_features = Vec::new();
            extract_raw_features(&lob_state, levels, &mut raw_features);
            for i in 0..40 {
                common::assertions::assert_f64_eq(
                    full_vec[i],
                    raw_features[i],
                    &format!("Event {valid_count} LOB feature[{i}]"),
                );
            }

            // Verify derived features (indices 40-47)
            let derived = compute_derived_features(&lob_state, levels).unwrap();
            for (j, &d) in derived.iter().enumerate() {
                common::assertions::assert_f64_eq(
                    full_vec[40 + j],
                    d,
                    &format!("Event {valid_count} derived feature[{j}]"),
                );
            }

            // MBO features (indices 48-83): verify ranges at this checkpoint
            assert_mbo_feature_ranges(
                &full_vec[48..84],
                &format!("Event {valid_count} full-vector MBO"),
            );

            // Signal features (indices 84-97): already verified in test 1e,
            // here just verify composition
            assert!(
                (full_vec[signals::indices::SCHEMA_VERSION] - contract::SCHEMA_VERSION).abs()
                    < FORMULA_TOL,
                "Event {valid_count}: schema_version at index {} = {}, expected {}",
                signals::indices::SCHEMA_VERSION,
                full_vec[signals::indices::SCHEMA_VERSION],
                contract::SCHEMA_VERSION
            );

            verified_samples += 1;
        }
    }

    assert!(
        verified_samples >= 2,
        "Expected at least 2 verified samples, got {verified_samples}"
    );

    println!(
        "Full 98-feature composition verified at {verified_samples} sample points \
         across {valid_count} valid events"
    );
}
