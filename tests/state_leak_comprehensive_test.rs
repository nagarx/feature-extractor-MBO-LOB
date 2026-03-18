//! Comprehensive multi-day state leak integration test.
//!
//! Verifies that Pipeline.reset() isolates ALL stateful components between
//! trading days. Uses distinct price ranges per day ($100 vs $200) to detect
//! any leakage of Day 1 state into Day 2 features.
//!
//! Tests the ACTUAL production code path: Pipeline.process_messages() with
//! synthetic MboMessage sequences, not FeatureExtractor in isolation.
//!
//! Per Rule §7: "Stateful components must define and test reset semantics."

use feature_extractor::builder::PipelineBuilder;
use mbo_lob_reconstructor::{Action, MboMessage, Side};

// ============================================================================
// Synthetic MBO message generation
// ============================================================================

/// Generate synthetic MBO messages for a trading day.
///
/// Creates Add events that build a realistic order book, then alternates
/// Add/Cancel/Trade events to drive feature computation.
///
/// # Price ranges
/// - Day 1: base_price_nd = 100_000_000_000 ($100.00)
/// - Day 2: base_price_nd = 200_000_000_000 ($200.00)
///
/// Distinct price ranges make any cross-day leakage detectable:
/// if Day 2 features contain values near $100, Day 1 data leaked.
fn generate_day_messages(
    day: u32,
    base_price_nd: i64,
    base_size: u32,
    num_events: u64,
) -> Vec<MboMessage> {
    let base_ts: i64 = (day as i64) * 86_400_000_000_000; // Day offset in nanoseconds
    let spread_nd: i64 = 10_000_000; // $0.01 spread
    let order_id_base = (day as u64) * 1_000_000;
    let mut msgs = Vec::with_capacity(num_events as usize);

    for i in 0..num_events {
        let ts = base_ts + (i as i64) * 1_000_000; // 1ms apart
        let order_id = order_id_base + i;

        // Cycle through actions: Add(0,1,2) → Cancel(3) → Trade(4) → Add(5,6,7) → ...
        let action = match i % 5 {
            0 | 1 | 2 => Action::Add,
            3 => Action::Cancel,
            4 => Action::Trade,
            _ => Action::Add,
        };

        // Alternate sides
        let side = if i % 2 == 0 { Side::Bid } else { Side::Ask };

        // Price varies slightly around base to create depth
        let level_offset = (i % 5) as i64 * spread_nd;
        let price = match side {
            Side::Bid => base_price_nd - level_offset,
            Side::Ask => base_price_nd + spread_nd + level_offset,
            _ => base_price_nd,
        };

        // Size varies per day to aid leak detection
        let size = base_size + (i % 10) as u32;

        let msg = MboMessage::new(order_id, action, side, price, size)
            .with_timestamp(ts);
        msgs.push(msg);
    }

    msgs
}

// ============================================================================
// Phase 5: Multi-Day State Leak Tests
// ============================================================================

/// Helper: build a pipeline with signals enabled (98 features).
/// Uses event-based sampling every 10 events for dense sample generation.
fn build_signal_pipeline() -> feature_extractor::Pipeline {
    PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()
        .with_mbo_features()
        .with_trading_signals()
        .event_sampling(10)
        .window(20, 5) // small window for quick sequence generation
        .max_buffer(500)
        .build()
        .expect("Failed to build signal pipeline")
}

#[test]
fn test_full_pipeline_two_day_isolation() {
    // This test processes Day 1 ($100) then Day 2 ($200) through the full pipeline.
    // After reset, Day 2 features must reflect ONLY Day 2 data.
    let mut pipeline = build_signal_pipeline();

    let day1_msgs = generate_day_messages(1, 100_000_000_000, 100, 500);
    let day2_msgs = generate_day_messages(2, 200_000_000_000, 500, 500);

    // Process Day 1
    let output1 = pipeline
        .process_messages(day1_msgs.into_iter())
        .expect("Day 1 processing failed");

    assert!(
        !output1.sequences.is_empty(),
        "Day 1 should produce sequences"
    );

    // Verify Day 1 mid-prices are around $100
    if !output1.mid_prices.is_empty() {
        let day1_mid = output1.mid_prices[output1.mid_prices.len() / 2];
        assert!(
            day1_mid > 90.0 && day1_mid < 110.0,
            "Day 1 mid-price should be ~$100, got {}",
            day1_mid
        );
    }

    // Reset between days (CRITICAL)
    pipeline.reset();

    // Process Day 2
    let output2 = pipeline
        .process_messages(day2_msgs.into_iter())
        .expect("Day 2 processing failed");

    assert!(
        !output2.sequences.is_empty(),
        "Day 2 should produce sequences"
    );

    // ================================================================
    // Verify Day 2 features are INDEPENDENT of Day 1
    // ================================================================

    // Check mid-prices: should be ~$200, not ~$100
    if !output2.mid_prices.is_empty() {
        let day2_mid = output2.mid_prices[output2.mid_prices.len() / 2];
        assert!(
            day2_mid > 190.0 && day2_mid < 210.0,
            "Day 2 mid-price should be ~$200 (not ~$100 from Day 1). Got {}",
            day2_mid
        );
    }

    // Check that NO sequence in Day 2 contains features in Day 1 range
    // Feature index 0 = ask_price_L1 (should be ~$200, not ~$100)
    for (seq_idx, seq) in output2.sequences.iter().enumerate() {
        for (snap_idx, snapshot) in seq.features.iter().enumerate() {
            let ask_price = snapshot[0]; // ask_price_L1
            assert!(
                ask_price > 150.0 || ask_price == 0.0, // Allow 0.0 for uninitialized
                "Day 2 seq[{}][{}] ask_price_L1 = {} — Day 1 data leaked! (expected > 150)",
                seq_idx, snap_idx, ask_price
            );
        }
    }
}

#[test]
fn test_pipeline_reset_feature_count_stable() {
    // Feature count must be identical across days
    let mut pipeline = build_signal_pipeline();

    let day1_msgs = generate_day_messages(1, 100_000_000_000, 100, 300);
    let day2_msgs = generate_day_messages(2, 200_000_000_000, 500, 300);

    let output1 = pipeline
        .process_messages(day1_msgs.into_iter())
        .expect("Day 1 failed");
    pipeline.reset();
    let output2 = pipeline
        .process_messages(day2_msgs.into_iter())
        .expect("Day 2 failed");

    if !output1.sequences.is_empty() && !output2.sequences.is_empty() {
        let f1 = output1.sequences[0].features[0].len();
        let f2 = output2.sequences[0].features[0].len();
        assert_eq!(
            f1, f2,
            "Feature count must be identical across days. Day 1: {}, Day 2: {}",
            f1, f2
        );
    }
}

#[test]
fn test_pipeline_triple_reset_no_accumulation() {
    // Process 3 days. Each day's output should be independent.
    let mut pipeline = build_signal_pipeline();

    let mut mid_prices_per_day = Vec::new();
    for day in 1..=3u32 {
        let base_price = (100 + day as i64 * 50) * 1_000_000_000; // $150, $200, $250
        let msgs = generate_day_messages(day, base_price, 100 + day * 50, 400);

        let output = pipeline
            .process_messages(msgs.into_iter())
            .expect(&format!("Day {} failed", day));

        if !output.mid_prices.is_empty() {
            let mid = output.mid_prices[output.mid_prices.len() / 2];
            mid_prices_per_day.push(mid);
        }
        pipeline.reset();
    }

    // Mid-prices should be strictly increasing (~$150, ~$200, ~$250)
    for i in 0..mid_prices_per_day.len().saturating_sub(1) {
        assert!(
            mid_prices_per_day[i + 1] > mid_prices_per_day[i],
            "Day {} mid ({}) should be > Day {} mid ({}). State may have leaked.",
            i + 2,
            mid_prices_per_day[i + 1],
            i + 1,
            mid_prices_per_day[i]
        );
    }
}

#[test]
fn test_pipeline_reset_all_features_finite() {
    // After reset + new day, all features must be finite (no NaN/Inf from stale state)
    let mut pipeline = build_signal_pipeline();

    let day1_msgs = generate_day_messages(1, 100_000_000_000, 100, 300);
    let day2_msgs = generate_day_messages(2, 200_000_000_000, 500, 300);

    let _ = pipeline.process_messages(day1_msgs.into_iter());
    pipeline.reset();
    let output2 = pipeline
        .process_messages(day2_msgs.into_iter())
        .expect("Day 2 failed");

    // Check every feature in every sequence is finite
    for (seq_idx, seq) in output2.sequences.iter().enumerate() {
        for (snap_idx, snapshot) in seq.features.iter().enumerate() {
            for (feat_idx, &val) in snapshot.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Day 2 seq[{}][{}][{}] = {} is not finite! Possible stale state from Day 1.",
                    seq_idx, snap_idx, feat_idx, val
                );
            }
        }
    }
}
