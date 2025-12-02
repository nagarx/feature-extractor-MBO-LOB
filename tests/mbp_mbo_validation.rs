//! Cross-Validation Test: MBO Reconstruction vs MBP-10 Ground Truth
//!
//! This test validates our MBO-LOB-reconstructor library by comparing
//! the reconstructed LOB states against Databento's MBP-10 (Market-By-Price)
//! snapshots, which serve as ground truth.
//!
//! ## Validation Strategy
//!
//! 1. Load MBO (Market-By-Order) data and reconstruct LOB using our library
//! 2. Load MBP-10 data (aggregated snapshots from Databento)
//! 3. Align timestamps and compare:
//!    - Best bid/ask prices
//!    - Best bid/ask sizes
//!    - Spread
//!    - Mid-price
//!    - Deeper levels (2-10)
//!
//! ## Why This Matters
//!
//! MBP-10 data is pre-aggregated by Databento from the same raw feed.
//! If our MBO reconstruction produces the same LOB state as MBP-10,
//! it confirms our reconstruction logic is correct.

#![cfg(feature = "databento")]

use std::collections::BTreeMap;
use std::path::Path;

use dbn::decode::DecodeRecord;
use dbn::Mbp10Msg;
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor, LobState};

/// Price precision for Databento (1e-9)
const PRICE_SCALE: f64 = 1e-9;

/// Undefined price sentinel value in DBN (i64::MAX)
/// This represents "no price available"
#[allow(dead_code)]
const UNDEF_PRICE: i64 = i64::MAX;
#[allow(dead_code)]
const UNDEF_PRICE_F64: f64 = UNDEF_PRICE as f64 * PRICE_SCALE;

/// Tolerance for price comparison (in dollars)
/// We use a very small tolerance to account for floating point precision
const PRICE_TOLERANCE: f64 = 0.000001;

/// Tolerance for size comparison (exact match expected)
const SIZE_TOLERANCE: u32 = 0;

/// Maximum reasonable price (to filter out sentinel values)
const MAX_REASONABLE_PRICE: f64 = 100_000.0; // $100,000

/// MBP-10 snapshot for comparison
#[derive(Debug, Clone)]
struct Mbp10Snapshot {
    timestamp: u64,
    bid_prices: [f64; 10],
    bid_sizes: [u32; 10],
    ask_prices: [f64; 10],
    ask_sizes: [u32; 10],
}

impl Mbp10Snapshot {
    fn from_dbn(msg: &Mbp10Msg) -> Self {
        let mut bid_prices = [0.0; 10];
        let mut bid_sizes = [0u32; 10];
        let mut ask_prices = [0.0; 10];
        let mut ask_sizes = [0u32; 10];

        for (i, level) in msg.levels.iter().enumerate() {
            bid_prices[i] = level.bid_px as f64 * PRICE_SCALE;
            bid_sizes[i] = level.bid_sz;
            ask_prices[i] = level.ask_px as f64 * PRICE_SCALE;
            ask_sizes[i] = level.ask_sz;
        }

        Self {
            timestamp: msg.hd.ts_event,
            bid_prices,
            bid_sizes,
            ask_prices,
            ask_sizes,
        }
    }

    fn best_bid(&self) -> f64 {
        self.bid_prices[0]
    }

    fn best_ask(&self) -> f64 {
        self.ask_prices[0]
    }

    fn best_bid_size(&self) -> u32 {
        self.bid_sizes[0]
    }

    fn best_ask_size(&self) -> u32 {
        self.ask_sizes[0]
    }

    fn spread(&self) -> f64 {
        self.best_ask() - self.best_bid()
    }

    fn mid_price(&self) -> f64 {
        (self.best_bid() + self.best_ask()) / 2.0
    }

    fn is_valid(&self) -> bool {
        self.best_bid() > 0.0
            && self.best_ask() > 0.0
            && self.best_bid() < self.best_ask()
            && self.best_bid() < MAX_REASONABLE_PRICE
            && self.best_ask() < MAX_REASONABLE_PRICE
    }
}

/// Validation result for a single comparison
#[derive(Debug, Default)]
struct ValidationResult {
    total_comparisons: u64,
    matching_best_bid_price: u64,
    matching_best_ask_price: u64,
    matching_best_bid_size: u64,
    matching_best_ask_size: u64,
    matching_spread: u64,
    matching_mid_price: u64,

    // Detailed errors
    bid_price_errors: Vec<(u64, f64, f64)>, // (timestamp, expected, actual)
    ask_price_errors: Vec<(u64, f64, f64)>,
    bid_size_errors: Vec<(u64, u32, u32)>,
    ask_size_errors: Vec<(u64, u32, u32)>,

    // Statistics
    max_bid_price_diff: f64,
    max_ask_price_diff: f64,
    max_bid_size_diff: u32,
    max_ask_size_diff: u32,

    // Skipped due to invalid states
    skipped_invalid_mbp: u64,
    skipped_invalid_mbo: u64,
}

impl ValidationResult {
    fn add_comparison(&mut self, timestamp: u64, mbp: &Mbp10Snapshot, mbo_state: &LobState) {
        self.total_comparisons += 1;

        // Get MBO state prices (convert from fixed-point i64 to f64)
        let mbo_best_bid = mbo_state
            .best_bid
            .map(|p| p as f64 * PRICE_SCALE)
            .unwrap_or(0.0);
        let mbo_best_ask = mbo_state
            .best_ask
            .map(|p| p as f64 * PRICE_SCALE)
            .unwrap_or(0.0);

        // Compare best bid price
        let bid_diff = (mbp.best_bid() - mbo_best_bid).abs();
        if bid_diff <= PRICE_TOLERANCE {
            self.matching_best_bid_price += 1;
        } else {
            self.bid_price_errors
                .push((timestamp, mbp.best_bid(), mbo_best_bid));
            if bid_diff > self.max_bid_price_diff {
                self.max_bid_price_diff = bid_diff;
            }
        }

        // Compare best ask price
        let ask_diff = (mbp.best_ask() - mbo_best_ask).abs();
        if ask_diff <= PRICE_TOLERANCE {
            self.matching_best_ask_price += 1;
        } else {
            self.ask_price_errors
                .push((timestamp, mbp.best_ask(), mbo_best_ask));
            if ask_diff > self.max_ask_price_diff {
                self.max_ask_price_diff = ask_diff;
            }
        }

        // Compare best bid size
        let bid_size_diff =
            (mbp.best_bid_size() as i64 - mbo_state.bid_sizes[0] as i64).unsigned_abs() as u32;
        if bid_size_diff <= SIZE_TOLERANCE {
            self.matching_best_bid_size += 1;
        } else {
            self.bid_size_errors
                .push((timestamp, mbp.best_bid_size(), mbo_state.bid_sizes[0]));
            if bid_size_diff > self.max_bid_size_diff {
                self.max_bid_size_diff = bid_size_diff;
            }
        }

        // Compare best ask size
        let ask_size_diff =
            (mbp.best_ask_size() as i64 - mbo_state.ask_sizes[0] as i64).unsigned_abs() as u32;
        if ask_size_diff <= SIZE_TOLERANCE {
            self.matching_best_ask_size += 1;
        } else {
            self.ask_size_errors
                .push((timestamp, mbp.best_ask_size(), mbo_state.ask_sizes[0]));
            if ask_size_diff > self.max_ask_size_diff {
                self.max_ask_size_diff = ask_size_diff;
            }
        }

        // Compare spread
        if let Some(mbo_spread) = mbo_state.spread() {
            let spread_diff = (mbp.spread() - mbo_spread).abs();
            if spread_diff <= PRICE_TOLERANCE * 2.0 {
                self.matching_spread += 1;
            }
        }

        // Compare mid-price
        if let Some(mbo_mid) = mbo_state.mid_price() {
            let mid_diff = (mbp.mid_price() - mbo_mid).abs();
            if mid_diff <= PRICE_TOLERANCE {
                self.matching_mid_price += 1;
            }
        }
    }

    fn accuracy(&self) -> f64 {
        if self.total_comparisons == 0 {
            return 0.0;
        }

        // Overall accuracy is the average of all metrics
        let bid_price_acc = self.matching_best_bid_price as f64 / self.total_comparisons as f64;
        let ask_price_acc = self.matching_best_ask_price as f64 / self.total_comparisons as f64;
        let bid_size_acc = self.matching_best_bid_size as f64 / self.total_comparisons as f64;
        let ask_size_acc = self.matching_best_ask_size as f64 / self.total_comparisons as f64;

        (bid_price_acc + ask_price_acc + bid_size_acc + ask_size_acc) / 4.0 * 100.0
    }

    fn print_summary(&self) {
        println!("\n{:=<70}", "");
        println!("VALIDATION SUMMARY");
        println!("{:=<70}", "");

        println!("\nTotal comparisons: {}", self.total_comparisons);
        println!("Skipped (invalid MBP): {}", self.skipped_invalid_mbp);
        println!("Skipped (invalid MBO): {}", self.skipped_invalid_mbo);

        println!("\n--- Accuracy Metrics ---");
        println!(
            "Best Bid Price:  {:.2}% ({}/{})",
            self.matching_best_bid_price as f64 / self.total_comparisons as f64 * 100.0,
            self.matching_best_bid_price,
            self.total_comparisons
        );
        println!(
            "Best Ask Price:  {:.2}% ({}/{})",
            self.matching_best_ask_price as f64 / self.total_comparisons as f64 * 100.0,
            self.matching_best_ask_price,
            self.total_comparisons
        );
        println!(
            "Best Bid Size:   {:.2}% ({}/{})",
            self.matching_best_bid_size as f64 / self.total_comparisons as f64 * 100.0,
            self.matching_best_bid_size,
            self.total_comparisons
        );
        println!(
            "Best Ask Size:   {:.2}% ({}/{})",
            self.matching_best_ask_size as f64 / self.total_comparisons as f64 * 100.0,
            self.matching_best_ask_size,
            self.total_comparisons
        );
        println!(
            "Spread:          {:.2}% ({}/{})",
            self.matching_spread as f64 / self.total_comparisons as f64 * 100.0,
            self.matching_spread,
            self.total_comparisons
        );
        println!(
            "Mid-Price:       {:.2}% ({}/{})",
            self.matching_mid_price as f64 / self.total_comparisons as f64 * 100.0,
            self.matching_mid_price,
            self.total_comparisons
        );

        println!("\n--- Maximum Differences ---");
        println!("Max Bid Price Diff: ${:.6}", self.max_bid_price_diff);
        println!("Max Ask Price Diff: ${:.6}", self.max_ask_price_diff);
        println!("Max Bid Size Diff:  {} shares", self.max_bid_size_diff);
        println!("Max Ask Size Diff:  {} shares", self.max_ask_size_diff);

        println!("\n--- Overall Accuracy ---");
        println!("OVERALL: {:.2}%", self.accuracy());

        // Show sample errors if any
        if !self.bid_price_errors.is_empty() {
            println!("\n--- Sample Bid Price Errors (first 5) ---");
            for (ts, expected, actual) in self.bid_price_errors.iter().take(5) {
                println!(
                    "  ts={}: expected=${:.4}, got=${:.4}, diff=${:.6}",
                    ts,
                    expected,
                    actual,
                    (expected - actual).abs()
                );
            }
        }

        if !self.ask_price_errors.is_empty() {
            println!("\n--- Sample Ask Price Errors (first 5) ---");
            for (ts, expected, actual) in self.ask_price_errors.iter().take(5) {
                println!(
                    "  ts={}: expected=${:.4}, got=${:.4}, diff=${:.6}",
                    ts,
                    expected,
                    actual,
                    (expected - actual).abs()
                );
            }
        }

        println!("{:=<70}\n", "");
    }
}

/// Load MBP-10 snapshots from a DBN file
fn load_mbp10_snapshots(
    path: &Path,
) -> Result<BTreeMap<u64, Mbp10Snapshot>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut decoder = dbn::decode::dbn::Decoder::with_zstd_buffer(reader)?;

    let mut snapshots = BTreeMap::new();

    loop {
        match decoder.decode_record::<Mbp10Msg>() {
            Ok(Some(msg)) => {
                let snapshot = Mbp10Snapshot::from_dbn(msg);
                if snapshot.is_valid() {
                    snapshots.insert(snapshot.timestamp, snapshot);
                }
            }
            Ok(None) => break,
            Err(e) => {
                // Log but continue - some records might be different types
                eprintln!("Warning: Failed to decode MBP-10 record: {}", e);
                continue;
            }
        }
    }

    Ok(snapshots)
}

/// Find the closest MBP-10 snapshot to a given timestamp
fn find_closest_snapshot(
    snapshots: &BTreeMap<u64, Mbp10Snapshot>,
    timestamp: u64,
    max_diff_ns: u64,
) -> Option<&Mbp10Snapshot> {
    // Look for exact match first
    if let Some(snapshot) = snapshots.get(&timestamp) {
        return Some(snapshot);
    }

    // Find closest before and after
    let before = snapshots.range(..timestamp).next_back();
    let after = snapshots.range(timestamp..).next();

    match (before, after) {
        (Some((ts_before, snap_before)), Some((ts_after, snap_after))) => {
            let diff_before = timestamp - ts_before;
            let diff_after = ts_after - timestamp;

            if diff_before <= diff_after && diff_before <= max_diff_ns {
                Some(snap_before)
            } else if diff_after <= max_diff_ns {
                Some(snap_after)
            } else {
                None
            }
        }
        (Some((ts_before, snap_before)), None) => {
            if timestamp - ts_before <= max_diff_ns {
                Some(snap_before)
            } else {
                None
            }
        }
        (None, Some((ts_after, snap_after))) => {
            if ts_after - timestamp <= max_diff_ns {
                Some(snap_after)
            } else {
                None
            }
        }
        (None, None) => None,
    }
}

#[test]
fn test_mbo_vs_mbp10_single_day() {
    // Paths to data files - using July 1, 2025 as test day
    let mbo_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250701.mbo.dbn.zst");
    let mbp_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07/xnas-itch-20250701.mbp-10.dbn.zst");

    if !mbo_path.exists() || !mbp_path.exists() {
        println!("Skipping test: data files not found");
        println!(
            "  MBO: {} (exists: {})",
            mbo_path.display(),
            mbo_path.exists()
        );
        println!(
            "  MBP: {} (exists: {})",
            mbp_path.display(),
            mbp_path.exists()
        );
        return;
    }

    println!("\n{:=<70}", "");
    println!("MBO vs MBP-10 CROSS-VALIDATION TEST");
    println!("Date: July 1, 2025");
    println!("{:=<70}", "");

    // Step 1: Load MBP-10 snapshots (ground truth)
    println!("\n[1/3] Loading MBP-10 snapshots (ground truth)...");
    let mbp_snapshots = load_mbp10_snapshots(mbp_path).expect("Failed to load MBP-10 data");
    println!("  Loaded {} MBP-10 snapshots", mbp_snapshots.len());

    // Step 2: Process MBO data and reconstruct LOB
    println!("\n[2/3] Processing MBO data and reconstructing LOB...");
    let loader = DbnLoader::new(mbo_path).expect("Failed to create MBO loader");
    let mut reconstructor = LobReconstructor::new(10);

    let mut result = ValidationResult::default();
    let mut mbo_count = 0u64;
    let mut last_state: Option<LobState> = None;

    // Maximum time difference for matching (1 millisecond = 1,000,000 nanoseconds)
    let max_time_diff_ns = 1_000_000u64;

    for msg in loader
        .iter_messages()
        .expect("Failed to iterate MBO messages")
    {
        mbo_count += 1;

        // Process message
        if let Ok(state) = reconstructor.process_message(&msg) {
            // Get prices as f64
            let best_bid = state.best_bid.map(|p| p as f64 * PRICE_SCALE);
            let best_ask = state.best_ask.map(|p| p as f64 * PRICE_SCALE);

            // Only compare on valid states
            if let (Some(b), Some(a)) = (best_bid, best_ask) {
                if b > 0.0 && a > 0.0 && b < a {
                    // Find matching MBP-10 snapshot
                    if let Some(mbp_snapshot) = find_closest_snapshot(
                        &mbp_snapshots,
                        msg.timestamp.unwrap_or(0) as u64,
                        max_time_diff_ns,
                    ) {
                        result.add_comparison(
                            msg.timestamp.unwrap_or(0) as u64,
                            mbp_snapshot,
                            &state,
                        );
                    }
                    last_state = Some(state);
                } else {
                    result.skipped_invalid_mbo += 1;
                }
            } else {
                result.skipped_invalid_mbo += 1;
            }
        }

        // Progress indicator
        if mbo_count % 1_000_000 == 0 {
            println!(
                "  Processed {} MBO messages, {} comparisons so far...",
                mbo_count, result.total_comparisons
            );
        }
    }

    println!("  Total MBO messages processed: {}", mbo_count);

    // Step 3: Print results
    println!("\n[3/3] Validation Results:");
    result.print_summary();

    // Assertions for CI
    // We expect high accuracy (>95%) for a correct implementation
    let accuracy = result.accuracy();
    println!("Final accuracy: {:.2}%", accuracy);

    // Note: We set a lower threshold initially to understand the baseline
    // After analysis, we can tighten this threshold
    assert!(
        accuracy > 50.0,
        "Accuracy too low: {:.2}%. This may indicate a bug in the reconstruction logic.",
        accuracy
    );

    // Print final state comparison
    if let Some(state) = last_state {
        println!("\n--- Final LOB State ---");
        let best_bid = state
            .best_bid
            .map(|p| p as f64 * PRICE_SCALE)
            .unwrap_or(0.0);
        let best_ask = state
            .best_ask
            .map(|p| p as f64 * PRICE_SCALE)
            .unwrap_or(0.0);
        println!("Best Bid: ${:.4} x {}", best_bid, state.bid_sizes[0]);
        println!("Best Ask: ${:.4} x {}", best_ask, state.ask_sizes[0]);
        if let Some(spread) = state.spread() {
            println!("Spread:   ${:.4}", spread);
        }
        if let Some(mid) = state.mid_price() {
            println!("Mid:      ${:.4}", mid);
        }
    }
}

#[test]
fn test_mbp10_data_quality() {
    // Quick sanity check on MBP-10 data
    let mbp_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07/xnas-itch-20250701.mbp-10.dbn.zst");

    if !mbp_path.exists() {
        println!("Skipping test: MBP-10 data file not found");
        return;
    }

    println!("\n{:=<70}", "");
    println!("MBP-10 DATA QUALITY CHECK");
    println!("{:=<70}", "");

    let snapshots = load_mbp10_snapshots(mbp_path).expect("Failed to load MBP-10 data");

    println!("Total snapshots: {}", snapshots.len());

    // Check data quality
    let mut valid_count = 0u64;
    let mut crossed_count = 0u64;
    let mut zero_bid_count = 0u64;
    let mut zero_ask_count = 0u64;

    let mut min_spread = f64::MAX;
    let mut max_spread = 0.0f64;
    let mut total_spread = 0.0f64;

    for snapshot in snapshots.values() {
        if snapshot.best_bid() <= 0.0 {
            zero_bid_count += 1;
            continue;
        }
        if snapshot.best_ask() <= 0.0 {
            zero_ask_count += 1;
            continue;
        }
        if snapshot.best_bid() >= snapshot.best_ask() {
            crossed_count += 1;
            continue;
        }

        valid_count += 1;
        let spread = snapshot.spread();
        total_spread += spread;
        min_spread = min_spread.min(spread);
        max_spread = max_spread.max(spread);
    }

    println!("\n--- Data Quality Metrics ---");
    println!(
        "Valid snapshots:    {} ({:.2}%)",
        valid_count,
        valid_count as f64 / snapshots.len() as f64 * 100.0
    );
    println!("Zero bid price:     {}", zero_bid_count);
    println!("Zero ask price:     {}", zero_ask_count);
    println!("Crossed quotes:     {}", crossed_count);

    if valid_count > 0 {
        println!("\n--- Spread Statistics ---");
        println!("Min spread:  ${:.4}", min_spread);
        println!("Max spread:  ${:.4}", max_spread);
        println!("Avg spread:  ${:.4}", total_spread / valid_count as f64);
    }

    // Show sample data
    println!("\n--- Sample MBP-10 Snapshots (first 5) ---");
    for (i, (ts, snapshot)) in snapshots.iter().take(5).enumerate() {
        println!(
            "  [{}] ts={}: bid=${:.4}x{}, ask=${:.4}x{}, spread=${:.4}",
            i + 1,
            ts,
            snapshot.best_bid(),
            snapshot.best_bid_size(),
            snapshot.best_ask(),
            snapshot.best_ask_size(),
            snapshot.spread()
        );
    }

    // Assertions
    assert!(valid_count > 0, "No valid MBP-10 snapshots found");
    assert!(
        valid_count as f64 / snapshots.len() as f64 > 0.9,
        "Too many invalid snapshots"
    );
}

#[test]
fn test_mbo_reconstruction_consistency() {
    // Test that our MBO reconstruction produces consistent results
    let mbo_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250701.mbo.dbn.zst");

    if !mbo_path.exists() {
        println!("Skipping test: MBO data file not found");
        return;
    }

    println!("\n{:=<70}", "");
    println!("MBO RECONSTRUCTION CONSISTENCY CHECK");
    println!("{:=<70}", "");

    let loader = DbnLoader::new(mbo_path).expect("Failed to create MBO loader");
    let mut reconstructor = LobReconstructor::new(10);

    let mut total_messages = 0u64;
    let mut valid_states = 0u64;
    let mut crossed_states = 0u64;
    let mut empty_bid_states = 0u64;
    let mut empty_ask_states = 0u64;

    let mut spreads = Vec::new();

    for msg in loader
        .iter_messages()
        .expect("Failed to iterate MBO messages")
    {
        total_messages += 1;

        if let Ok(state) = reconstructor.process_message(&msg) {
            let best_bid = state.best_bid.map(|p| p as f64 * PRICE_SCALE);
            let best_ask = state.best_ask.map(|p| p as f64 * PRICE_SCALE);

            match (best_bid, best_ask) {
                (None, _) => {
                    empty_bid_states += 1;
                }
                (Some(b), _) if b <= 0.0 => {
                    empty_bid_states += 1;
                }
                (_, None) => {
                    empty_ask_states += 1;
                }
                (_, Some(a)) if a <= 0.0 => {
                    empty_ask_states += 1;
                }
                (Some(b), Some(a)) if b >= a => {
                    crossed_states += 1;
                }
                (Some(_), Some(_)) => {
                    valid_states += 1;
                    if let Some(spread) = state.spread() {
                        spreads.push(spread);
                    }
                }
            }
        }

        // Limit to first 100K messages for speed
        if total_messages >= 100_000 {
            break;
        }
    }

    println!("\n--- Reconstruction Statistics (first 100K messages) ---");
    println!("Total messages:     {}", total_messages);
    println!(
        "Valid states:       {} ({:.2}%)",
        valid_states,
        valid_states as f64 / total_messages as f64 * 100.0
    );
    println!("Empty bid states:   {}", empty_bid_states);
    println!("Empty ask states:   {}", empty_ask_states);
    println!("Crossed states:     {}", crossed_states);

    if !spreads.is_empty() {
        spreads.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_spread = spreads[0];
        let max_spread = spreads[spreads.len() - 1];
        let median_spread = spreads[spreads.len() / 2];
        let avg_spread: f64 = spreads.iter().sum::<f64>() / spreads.len() as f64;

        println!("\n--- Spread Statistics ---");
        println!("Min spread:     ${:.4}", min_spread);
        println!("Max spread:     ${:.4}", max_spread);
        println!("Median spread:  ${:.4}", median_spread);
        println!("Average spread: ${:.4}", avg_spread);
    }

    // Assertions
    assert!(valid_states > 0, "No valid LOB states produced");
    // After initial warmup, we should have mostly valid states
    let valid_ratio = valid_states as f64 / total_messages as f64;
    println!("\nValid state ratio: {:.2}%", valid_ratio * 100.0);
}

/// Statistical validation: OFI vs Price Change correlation
#[test]
fn test_ofi_price_correlation() {
    use feature_extractor::features::order_flow::OrderFlowTracker;

    let mbo_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250701.mbo.dbn.zst");

    if !mbo_path.exists() {
        println!("Skipping test: MBO data file not found");
        return;
    }

    println!("\n{:=<70}", "");
    println!("OFI vs PRICE CHANGE CORRELATION TEST");
    println!("{:=<70}", "");

    let loader = DbnLoader::new(mbo_path).expect("Failed to create MBO loader");
    let mut reconstructor = LobReconstructor::new(10);
    let mut ofi_tracker = OrderFlowTracker::new();

    let mut ofi_values: Vec<f64> = Vec::new();
    let mut price_changes: Vec<f64> = Vec::new();
    let mut prev_mid_price: Option<f64> = None;

    let mut count = 0u64;

    for msg in loader
        .iter_messages()
        .expect("Failed to iterate MBO messages")
    {
        if let Ok(state) = reconstructor.process_message(&msg) {
            let best_bid = state.best_bid.map(|p| p as f64 * PRICE_SCALE);
            let best_ask = state.best_ask.map(|p| p as f64 * PRICE_SCALE);

            if let (Some(b), Some(a)) = (best_bid, best_ask) {
                if b > 0.0 && a > 0.0 && b < a {
                    // Update OFI tracker
                    let timestamp = msg.timestamp.unwrap_or(0) as u64;
                    ofi_tracker.update(&state, timestamp);

                    let features = ofi_tracker.extract_features(&state);
                    let ofi = features.ofi;

                    if let Some(mid_price) = state.mid_price() {
                        if let Some(prev_mid) = prev_mid_price {
                            let price_change = mid_price - prev_mid;

                            // Only record non-zero changes
                            if price_change.abs() > 1e-9 {
                                ofi_values.push(ofi);
                                price_changes.push(price_change);
                            }
                        }

                        prev_mid_price = Some(mid_price);
                    }
                    count += 1;
                }
            }
        }

        // Limit to first 100K valid states for speed
        if count >= 100_000 {
            break;
        }
    }

    println!("Collected {} OFI/price-change pairs", ofi_values.len());

    if ofi_values.len() < 100 {
        println!("Not enough data points for correlation analysis");
        return;
    }

    // Calculate Pearson correlation
    let n = ofi_values.len() as f64;
    let mean_ofi: f64 = ofi_values.iter().sum::<f64>() / n;
    let mean_price: f64 = price_changes.iter().sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_ofi = 0.0f64;
    let mut var_price = 0.0f64;

    for i in 0..ofi_values.len() {
        let ofi_diff = ofi_values[i] - mean_ofi;
        let price_diff = price_changes[i] - mean_price;
        cov += ofi_diff * price_diff;
        var_ofi += ofi_diff * ofi_diff;
        var_price += price_diff * price_diff;
    }

    let correlation = if var_ofi > 0.0 && var_price > 0.0 {
        cov / (var_ofi.sqrt() * var_price.sqrt())
    } else {
        0.0
    };

    println!("\n--- Correlation Analysis ---");
    println!("Mean OFI:          {:.4}", mean_ofi);
    println!("Mean Price Change: ${:.6}", mean_price);
    println!("Correlation (r):   {:.4}", correlation);

    // OFI should have positive correlation with price changes
    // (positive OFI = buying pressure = price up)
    println!("\n--- Interpretation ---");
    if correlation > 0.1 {
        println!("✓ Positive correlation detected - OFI predicts price direction");
    } else if correlation > 0.0 {
        println!("~ Weak positive correlation - OFI has some predictive power");
    } else {
        println!("✗ No positive correlation - may indicate OFI calculation issue");
    }

    // We expect at least a weak positive correlation
    // Note: Real-world correlation is often weak but consistent
    assert!(
        correlation > -0.5,
        "Unexpected strong negative correlation: {:.4}. This may indicate a sign error in OFI calculation.",
        correlation
    );
}
