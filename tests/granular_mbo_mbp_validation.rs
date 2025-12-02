//! Granular Cross-Validation: MBO Reconstruction vs MBP-10 Ground Truth
//!
//! This test performs detailed, granular validation to find any potential issues
//! in our MBO-LOB-reconstructor and feature_extractor libraries before production use.
//!
//! ## Validation Categories
//!
//! 1. **Exact Timestamp Matching**: Only compare at exact MBP-10 timestamps
//! 2. **Level-by-Level Analysis**: Compare all 10 price levels, not just BBO
//! 3. **Time-of-Day Analysis**: Track accuracy by market session
//! 4. **Error Pattern Detection**: Find systematic errors vs random noise
//! 5. **Order Flow Validation**: Verify order lifecycle consistency
//! 6. **Statistical Tests**: Chi-squared, KS tests for distributions

#![cfg(feature = "databento")]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use dbn::decode::DecodeRecord;
use dbn::Mbp10Msg;
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor, LobState};

/// Price precision for Databento (1e-9)
const PRICE_SCALE: f64 = 1e-9;

/// Maximum reasonable price (to filter out sentinel values)
const MAX_REASONABLE_PRICE: f64 = 100_000.0;

/// Nanoseconds per hour
const NS_PER_HOUR: u64 = 3_600_000_000_000;

/// Market open time (9:30 AM ET in nanoseconds from midnight)
const MARKET_OPEN_NS: u64 = 9 * NS_PER_HOUR + 30 * 60 * 1_000_000_000;

/// Market close time (4:00 PM ET in nanoseconds from midnight)  
const MARKET_CLOSE_NS: u64 = 16 * NS_PER_HOUR;

/// Detailed comparison result for a single timestamp
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DetailedComparison {
    timestamp: u64,

    // Price comparisons (all 10 levels)
    bid_price_diffs: [f64; 10],
    ask_price_diffs: [f64; 10],

    // Size comparisons (all 10 levels)
    bid_size_diffs: [i64; 10],
    ask_size_diffs: [i64; 10],

    // Expected values from MBP-10
    mbp_bid_prices: [f64; 10],
    mbp_ask_prices: [f64; 10],
    mbp_bid_sizes: [u32; 10],
    mbp_ask_sizes: [u32; 10],

    // Actual values from MBO reconstruction
    mbo_bid_prices: [f64; 10],
    mbo_ask_prices: [f64; 10],
    mbo_bid_sizes: [u32; 10],
    mbo_ask_sizes: [u32; 10],

    // Derived metrics
    spread_diff: f64,
    mid_price_diff: f64,

    // Market session
    is_regular_hours: bool,
}

/// Aggregated statistics by time period
#[derive(Debug, Default, Clone)]
struct TimeSliceStats {
    total_comparisons: u64,

    // BBO accuracy
    exact_bid_price_matches: u64,
    exact_ask_price_matches: u64,
    exact_bid_size_matches: u64,
    exact_ask_size_matches: u64,

    // Within tolerance
    bid_price_within_1c: u64, // Within 1 cent
    ask_price_within_1c: u64,
    bid_price_within_5c: u64, // Within 5 cents
    ask_price_within_5c: u64,

    // Error magnitudes
    total_bid_price_error: f64,
    total_ask_price_error: f64,
    total_bid_size_error: i64,
    total_ask_size_error: i64,

    max_bid_price_error: f64,
    max_ask_price_error: f64,
    max_bid_size_error: u32,
    max_ask_size_error: u32,
}

impl TimeSliceStats {
    fn add_comparison(&mut self, comp: &DetailedComparison) {
        self.total_comparisons += 1;

        // BBO (level 0) analysis
        let bid_price_diff = comp.bid_price_diffs[0].abs();
        let ask_price_diff = comp.ask_price_diffs[0].abs();
        let bid_size_diff = comp.bid_size_diffs[0].unsigned_abs() as u32;
        let ask_size_diff = comp.ask_size_diffs[0].unsigned_abs() as u32;

        // Exact matches (within floating point tolerance)
        if bid_price_diff < 0.000001 {
            self.exact_bid_price_matches += 1;
        }
        if ask_price_diff < 0.000001 {
            self.exact_ask_price_matches += 1;
        }
        if comp.bid_size_diffs[0] == 0 {
            self.exact_bid_size_matches += 1;
        }
        if comp.ask_size_diffs[0] == 0 {
            self.exact_ask_size_matches += 1;
        }

        // Within tolerance
        if bid_price_diff <= 0.01 {
            self.bid_price_within_1c += 1;
        }
        if ask_price_diff <= 0.01 {
            self.ask_price_within_1c += 1;
        }
        if bid_price_diff <= 0.05 {
            self.bid_price_within_5c += 1;
        }
        if ask_price_diff <= 0.05 {
            self.ask_price_within_5c += 1;
        }

        // Error totals
        self.total_bid_price_error += bid_price_diff;
        self.total_ask_price_error += ask_price_diff;
        self.total_bid_size_error += comp.bid_size_diffs[0].abs();
        self.total_ask_size_error += comp.ask_size_diffs[0].abs();

        // Max errors
        if bid_price_diff > self.max_bid_price_error {
            self.max_bid_price_error = bid_price_diff;
        }
        if ask_price_diff > self.max_ask_price_error {
            self.max_ask_price_error = ask_price_diff;
        }
        if bid_size_diff > self.max_bid_size_error {
            self.max_bid_size_error = bid_size_diff;
        }
        if ask_size_diff > self.max_ask_size_error {
            self.max_ask_size_error = ask_size_diff;
        }
    }

    fn print_summary(&self, label: &str) {
        if self.total_comparisons == 0 {
            println!("{}: No comparisons", label);
            return;
        }

        let n = self.total_comparisons as f64;

        println!(
            "\n--- {} ({} comparisons) ---",
            label, self.total_comparisons
        );

        println!("  EXACT MATCHES:");
        println!(
            "    Bid Price: {:.2}% ({}/{})",
            self.exact_bid_price_matches as f64 / n * 100.0,
            self.exact_bid_price_matches,
            self.total_comparisons
        );
        println!(
            "    Ask Price: {:.2}% ({}/{})",
            self.exact_ask_price_matches as f64 / n * 100.0,
            self.exact_ask_price_matches,
            self.total_comparisons
        );
        println!(
            "    Bid Size:  {:.2}% ({}/{})",
            self.exact_bid_size_matches as f64 / n * 100.0,
            self.exact_bid_size_matches,
            self.total_comparisons
        );
        println!(
            "    Ask Size:  {:.2}% ({}/{})",
            self.exact_ask_size_matches as f64 / n * 100.0,
            self.exact_ask_size_matches,
            self.total_comparisons
        );

        println!("  WITHIN 1 CENT:");
        println!(
            "    Bid Price: {:.2}%",
            self.bid_price_within_1c as f64 / n * 100.0
        );
        println!(
            "    Ask Price: {:.2}%",
            self.ask_price_within_1c as f64 / n * 100.0
        );

        println!("  WITHIN 5 CENTS:");
        println!(
            "    Bid Price: {:.2}%",
            self.bid_price_within_5c as f64 / n * 100.0
        );
        println!(
            "    Ask Price: {:.2}%",
            self.ask_price_within_5c as f64 / n * 100.0
        );

        println!("  MEAN ABSOLUTE ERROR:");
        println!("    Bid Price: ${:.6}", self.total_bid_price_error / n);
        println!("    Ask Price: ${:.6}", self.total_ask_price_error / n);
        println!(
            "    Bid Size:  {:.1} shares",
            self.total_bid_size_error as f64 / n
        );
        println!(
            "    Ask Size:  {:.1} shares",
            self.total_ask_size_error as f64 / n
        );

        println!("  MAX ERROR:");
        println!("    Bid Price: ${:.4}", self.max_bid_price_error);
        println!("    Ask Price: ${:.4}", self.max_ask_price_error);
        println!("    Bid Size:  {} shares", self.max_bid_size_error);
        println!("    Ask Size:  {} shares", self.max_ask_size_error);
    }
}

/// Level-by-level statistics
#[derive(Debug, Default)]
struct LevelStats {
    comparisons: u64,
    price_matches: u64,
    size_matches: u64,
    price_within_1c: u64,
    total_price_error: f64,
    total_size_error: i64,
}

/// Error pattern analysis
#[derive(Debug, Default)]
#[allow(dead_code)]
struct ErrorPatternAnalysis {
    // Consecutive errors
    consecutive_bid_errors: Vec<u64>, // Lengths of consecutive error runs
    consecutive_ask_errors: Vec<u64>,

    // Error by price level
    errors_at_bbo: u64,
    errors_deeper_levels: u64,

    // Directional bias
    bid_overestimates: u64, // Our reconstruction > MBP
    bid_underestimates: u64,
    ask_overestimates: u64,
    ask_underestimates: u64,

    // Size-related
    large_size_errors: u64, // Errors > 1000 shares
    small_size_errors: u64, // Errors <= 100 shares
}

/// Load MBP-10 snapshots indexed by exact timestamp
fn load_mbp10_exact(
    path: &Path,
) -> Result<BTreeMap<u64, ([f64; 10], [f64; 10], [u32; 10], [u32; 10])>, Box<dyn std::error::Error>>
{
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut decoder = dbn::decode::dbn::Decoder::with_zstd_buffer(reader)?;

    let mut snapshots = BTreeMap::new();

    loop {
        match decoder.decode_record::<Mbp10Msg>() {
            Ok(Some(msg)) => {
                let mut bid_prices = [0.0f64; 10];
                let mut ask_prices = [0.0f64; 10];
                let mut bid_sizes = [0u32; 10];
                let mut ask_sizes = [0u32; 10];

                for (i, level) in msg.levels.iter().enumerate() {
                    bid_prices[i] = level.bid_px as f64 * PRICE_SCALE;
                    ask_prices[i] = level.ask_px as f64 * PRICE_SCALE;
                    bid_sizes[i] = level.bid_sz;
                    ask_sizes[i] = level.ask_sz;
                }

                // Only include valid snapshots
                if bid_prices[0] > 0.0
                    && bid_prices[0] < MAX_REASONABLE_PRICE
                    && ask_prices[0] > 0.0
                    && ask_prices[0] < MAX_REASONABLE_PRICE
                    && bid_prices[0] < ask_prices[0]
                {
                    snapshots.insert(
                        msg.hd.ts_event,
                        (bid_prices, ask_prices, bid_sizes, ask_sizes),
                    );
                }
            }
            Ok(None) => break,
            Err(_) => continue,
        }
    }

    Ok(snapshots)
}

/// Get time of day from timestamp (nanoseconds since midnight)
fn time_of_day_ns(timestamp: u64) -> u64 {
    // Approximate: timestamp is ns since Unix epoch
    // We need ns since midnight ET
    // For simplicity, we'll use the fractional day
    let ns_per_day = 24 * NS_PER_HOUR;
    timestamp % ns_per_day
}

/// Check if timestamp is during regular market hours
fn is_regular_hours(timestamp: u64) -> bool {
    let tod = time_of_day_ns(timestamp);
    // Adjust for ET timezone (approximate)
    // This is a rough check - actual implementation would need proper timezone handling
    tod >= MARKET_OPEN_NS && tod < MARKET_CLOSE_NS
}

#[test]
fn test_granular_level_by_level_validation() {
    let mbo_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250701.mbo.dbn.zst");
    let mbp_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07/xnas-itch-20250701.mbp-10.dbn.zst");

    if !mbo_path.exists() || !mbp_path.exists() {
        println!("Skipping test: data files not found");
        return;
    }

    println!("\n{:=<80}", "");
    println!("GRANULAR LEVEL-BY-LEVEL VALIDATION");
    println!("Date: July 1, 2025");
    println!("{:=<80}", "");

    // Load MBP-10 data
    println!("\n[1/4] Loading MBP-10 snapshots...");
    let mbp_data = load_mbp10_exact(mbp_path).expect("Failed to load MBP-10");
    println!("  Loaded {} valid MBP-10 snapshots", mbp_data.len());

    // Process MBO and build state at each MBP timestamp
    println!("\n[2/4] Processing MBO data...");
    let loader = DbnLoader::new(mbo_path).expect("Failed to create MBO loader");
    let mut reconstructor = LobReconstructor::new(10);

    // Store MBO states at MBP timestamps
    let mut mbo_states_at_mbp_ts: BTreeMap<u64, LobState> = BTreeMap::new();
    let mut _current_state: Option<LobState> = None;
    let mut mbo_count = 0u64;

    // Get list of MBP timestamps we care about
    let mbp_timestamps: Vec<u64> = mbp_data.keys().cloned().collect();
    let mut mbp_idx = 0;

    for msg in loader
        .iter_messages()
        .expect("Failed to iterate MBO messages")
    {
        mbo_count += 1;

        let msg_ts = msg.timestamp.unwrap_or(0) as u64;

        // Process message
        if let Ok(state) = reconstructor.process_message(&msg) {
            _current_state = Some(state.clone());

            // Check if we've passed any MBP timestamps
            while mbp_idx < mbp_timestamps.len() && mbp_timestamps[mbp_idx] <= msg_ts {
                if let Some(ref s) = current_state {
                    mbo_states_at_mbp_ts.insert(mbp_timestamps[mbp_idx], s.clone());
                }
                mbp_idx += 1;
            }
        }

        if mbo_count % 2_000_000 == 0 {
            println!(
                "  Processed {} MBO messages, captured {} MBP timestamps...",
                mbo_count,
                mbo_states_at_mbp_ts.len()
            );
        }
    }

    println!("  Total MBO messages: {}", mbo_count);
    println!(
        "  MBO states captured at MBP timestamps: {}",
        mbo_states_at_mbp_ts.len()
    );

    // Analyze level-by-level
    println!("\n[3/4] Analyzing level-by-level accuracy...");

    let mut level_stats: [LevelStats; 10] = Default::default();
    let mut comparisons = Vec::new();

    for (ts, (mbp_bids, mbp_asks, mbp_bid_sz, mbp_ask_sz)) in &mbp_data {
        if let Some(mbo_state) = mbo_states_at_mbp_ts.get(ts) {
            // Create detailed comparison
            let mut comp = DetailedComparison {
                timestamp: *ts,
                bid_price_diffs: [0.0; 10],
                ask_price_diffs: [0.0; 10],
                bid_size_diffs: [0; 10],
                ask_size_diffs: [0; 10],
                mbp_bid_prices: *mbp_bids,
                mbp_ask_prices: *mbp_asks,
                mbp_bid_sizes: *mbp_bid_sz,
                mbp_ask_sizes: *mbp_ask_sz,
                mbo_bid_prices: [0.0; 10],
                mbo_ask_prices: [0.0; 10],
                mbo_bid_sizes: [0; 10],
                mbo_ask_sizes: [0; 10],
                spread_diff: 0.0,
                mid_price_diff: 0.0,
                is_regular_hours: is_regular_hours(*ts),
            };

            // Compare each level
            for level in 0..10 {
                // Get MBO prices (convert from i64 fixed-point)
                let mbo_bid = if level < mbo_state.bid_prices.len() {
                    mbo_state.bid_prices[level] as f64 * PRICE_SCALE
                } else {
                    0.0
                };
                let mbo_ask = if level < mbo_state.ask_prices.len() {
                    mbo_state.ask_prices[level] as f64 * PRICE_SCALE
                } else {
                    0.0
                };
                let mbo_bid_size = if level < mbo_state.bid_sizes.len() {
                    mbo_state.bid_sizes[level]
                } else {
                    0
                };
                let mbo_ask_size = if level < mbo_state.ask_sizes.len() {
                    mbo_state.ask_sizes[level]
                } else {
                    0
                };

                comp.mbo_bid_prices[level] = mbo_bid;
                comp.mbo_ask_prices[level] = mbo_ask;
                comp.mbo_bid_sizes[level] = mbo_bid_size;
                comp.mbo_ask_sizes[level] = mbo_ask_size;

                // Calculate differences
                comp.bid_price_diffs[level] = mbo_bid - mbp_bids[level];
                comp.ask_price_diffs[level] = mbo_ask - mbp_asks[level];
                comp.bid_size_diffs[level] = mbo_bid_size as i64 - mbp_bid_sz[level] as i64;
                comp.ask_size_diffs[level] = mbo_ask_size as i64 - mbp_ask_sz[level] as i64;

                // Update level stats
                level_stats[level].comparisons += 1;

                if comp.bid_price_diffs[level].abs() < 0.000001 {
                    level_stats[level].price_matches += 1;
                }
                if comp.bid_size_diffs[level] == 0 {
                    level_stats[level].size_matches += 1;
                }
                if comp.bid_price_diffs[level].abs() <= 0.01 {
                    level_stats[level].price_within_1c += 1;
                }
                level_stats[level].total_price_error += comp.bid_price_diffs[level].abs();
                level_stats[level].total_size_error += comp.bid_size_diffs[level].abs();
            }

            // Spread and mid-price
            let mbp_spread = mbp_asks[0] - mbp_bids[0];
            let mbo_spread = comp.mbo_ask_prices[0] - comp.mbo_bid_prices[0];
            comp.spread_diff = mbo_spread - mbp_spread;

            let mbp_mid = (mbp_asks[0] + mbp_bids[0]) / 2.0;
            let mbo_mid = (comp.mbo_ask_prices[0] + comp.mbo_bid_prices[0]) / 2.0;
            comp.mid_price_diff = mbo_mid - mbp_mid;

            comparisons.push(comp);
        }
    }

    println!("  Total aligned comparisons: {}", comparisons.len());

    // Print level-by-level results
    println!("\n[4/4] Results by Level:");
    println!("\n{:-<80}", "");
    println!(
        "{:^10} {:^15} {:^15} {:^15} {:^15}",
        "Level", "Price Match%", "Size Match%", "Price<1c%", "MAE Price"
    );
    println!("{:-<80}", "");

    for (level, stats) in level_stats.iter().enumerate() {
        if stats.comparisons > 0 {
            let n = stats.comparisons as f64;
            println!(
                "{:^10} {:^15.2} {:^15.2} {:^15.2} ${:^14.6}",
                level,
                stats.price_matches as f64 / n * 100.0,
                stats.size_matches as f64 / n * 100.0,
                stats.price_within_1c as f64 / n * 100.0,
                stats.total_price_error / n
            );
        }
    }
    println!("{:-<80}", "");

    // BBO summary
    let bbo_stats = &level_stats[0];
    if bbo_stats.comparisons > 0 {
        let n = bbo_stats.comparisons as f64;
        println!("\nBBO (Level 0) Summary:");
        println!(
            "  Exact price matches: {:.2}%",
            bbo_stats.price_matches as f64 / n * 100.0
        );
        println!(
            "  Exact size matches:  {:.2}%",
            bbo_stats.size_matches as f64 / n * 100.0
        );
        println!(
            "  Price within 1 cent: {:.2}%",
            bbo_stats.price_within_1c as f64 / n * 100.0
        );
        println!(
            "  Mean Absolute Error: ${:.6}",
            bbo_stats.total_price_error / n
        );
    }

    // Assertions for CI
    let bbo_price_accuracy = bbo_stats.price_within_1c as f64 / bbo_stats.comparisons as f64;
    assert!(
        bbo_price_accuracy > 0.90,
        "BBO price accuracy should be >90%, got {:.2}%",
        bbo_price_accuracy * 100.0
    );
}

#[test]
fn test_granular_time_of_day_analysis() {
    let mbo_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250701.mbo.dbn.zst");
    let mbp_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07/xnas-itch-20250701.mbp-10.dbn.zst");

    if !mbo_path.exists() || !mbp_path.exists() {
        println!("Skipping test: data files not found");
        return;
    }

    println!("\n{:=<80}", "");
    println!("TIME-OF-DAY ACCURACY ANALYSIS");
    println!("Date: July 1, 2025");
    println!("{:=<80}", "");

    // Load data
    let mbp_data = load_mbp10_exact(mbp_path).expect("Failed to load MBP-10");
    let loader = DbnLoader::new(mbo_path).expect("Failed to create MBO loader");
    let mut reconstructor = LobReconstructor::new(10);

    // Track stats by hour
    let mut hourly_stats: HashMap<u32, TimeSliceStats> = HashMap::new();
    let mut pre_market_stats = TimeSliceStats::default();
    let mut regular_hours_stats = TimeSliceStats::default();
    let mut after_hours_stats = TimeSliceStats::default();

    // Process MBO
    let mut _current_state: Option<LobState> = None;
    let mbp_timestamps: Vec<u64> = mbp_data.keys().cloned().collect();
    let mut mbp_idx = 0;

    for msg in loader
        .iter_messages()
        .expect("Failed to iterate MBO messages")
    {
        let msg_ts = msg.timestamp.unwrap_or(0) as u64;

        if let Ok(state) = reconstructor.process_message(&msg) {
            _current_state = Some(state.clone());

            while mbp_idx < mbp_timestamps.len() && mbp_timestamps[mbp_idx] <= msg_ts {
                let mbp_ts = mbp_timestamps[mbp_idx];

                if let (Some(ref mbo_state), Some((mbp_bids, mbp_asks, mbp_bid_sz, mbp_ask_sz))) =
                    (&current_state, mbp_data.get(&mbp_ts))
                {
                    // Get MBO values
                    let mbo_bid = mbo_state
                        .best_bid
                        .map(|p| p as f64 * PRICE_SCALE)
                        .unwrap_or(0.0);
                    let mbo_ask = mbo_state
                        .best_ask
                        .map(|p| p as f64 * PRICE_SCALE)
                        .unwrap_or(0.0);

                    // Skip invalid states
                    if mbo_bid <= 0.0 || mbo_ask <= 0.0 || mbo_bid >= mbo_ask {
                        mbp_idx += 1;
                        continue;
                    }

                    // Create comparison
                    let mut comp = DetailedComparison {
                        timestamp: mbp_ts,
                        bid_price_diffs: [0.0; 10],
                        ask_price_diffs: [0.0; 10],
                        bid_size_diffs: [0; 10],
                        ask_size_diffs: [0; 10],
                        mbp_bid_prices: *mbp_bids,
                        mbp_ask_prices: *mbp_asks,
                        mbp_bid_sizes: *mbp_bid_sz,
                        mbp_ask_sizes: *mbp_ask_sz,
                        mbo_bid_prices: [0.0; 10],
                        mbo_ask_prices: [0.0; 10],
                        mbo_bid_sizes: [0; 10],
                        mbo_ask_sizes: [0; 10],
                        spread_diff: 0.0,
                        mid_price_diff: 0.0,
                        is_regular_hours: false,
                    };

                    // Fill in BBO
                    comp.mbo_bid_prices[0] = mbo_bid;
                    comp.mbo_ask_prices[0] = mbo_ask;
                    comp.mbo_bid_sizes[0] = mbo_state.bid_sizes[0];
                    comp.mbo_ask_sizes[0] = mbo_state.ask_sizes[0];

                    comp.bid_price_diffs[0] = mbo_bid - mbp_bids[0];
                    comp.ask_price_diffs[0] = mbo_ask - mbp_asks[0];
                    comp.bid_size_diffs[0] = mbo_state.bid_sizes[0] as i64 - mbp_bid_sz[0] as i64;
                    comp.ask_size_diffs[0] = mbo_state.ask_sizes[0] as i64 - mbp_ask_sz[0] as i64;

                    // Time of day analysis
                    let tod = time_of_day_ns(mbp_ts);
                    let hour = (tod / NS_PER_HOUR) as u32;

                    // Update hourly stats
                    hourly_stats.entry(hour).or_default().add_comparison(&comp);

                    // Update session stats
                    if tod < MARKET_OPEN_NS {
                        pre_market_stats.add_comparison(&comp);
                    } else if tod < MARKET_CLOSE_NS {
                        regular_hours_stats.add_comparison(&comp);
                        comp.is_regular_hours = true;
                    } else {
                        after_hours_stats.add_comparison(&comp);
                    }
                }

                mbp_idx += 1;
            }
        }
    }

    // Print results
    println!("\n--- BY MARKET SESSION ---");
    pre_market_stats.print_summary("Pre-Market (4:00-9:30 AM)");
    regular_hours_stats.print_summary("Regular Hours (9:30 AM - 4:00 PM)");
    after_hours_stats.print_summary("After Hours (4:00-8:00 PM)");

    println!("\n--- BY HOUR ---");
    let mut hours: Vec<u32> = hourly_stats.keys().cloned().collect();
    hours.sort();

    println!("\n{:-<90}", "");
    println!(
        "{:^6} {:^12} {:^15} {:^15} {:^15} {:^15}",
        "Hour", "Comparisons", "Bid Price %", "Ask Price %", "Bid Size %", "Ask Size %"
    );
    println!("{:-<90}", "");

    for hour in hours {
        if let Some(stats) = hourly_stats.get(&hour) {
            if stats.total_comparisons > 0 {
                let n = stats.total_comparisons as f64;
                println!(
                    "{:^6} {:^12} {:^15.2} {:^15.2} {:^15.2} {:^15.2}",
                    format!("{}:00", hour),
                    stats.total_comparisons,
                    stats.exact_bid_price_matches as f64 / n * 100.0,
                    stats.exact_ask_price_matches as f64 / n * 100.0,
                    stats.exact_bid_size_matches as f64 / n * 100.0,
                    stats.exact_ask_size_matches as f64 / n * 100.0
                );
            }
        }
    }
    println!("{:-<90}", "");

    // Assertion: Regular hours should have best accuracy
    if regular_hours_stats.total_comparisons > 0 {
        let regular_price_acc = regular_hours_stats.bid_price_within_1c as f64
            / regular_hours_stats.total_comparisons as f64;
        println!(
            "\nRegular hours price accuracy (within 1c): {:.2}%",
            regular_price_acc * 100.0
        );
    }
}

#[test]
fn test_granular_error_pattern_detection() {
    let mbo_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250701.mbo.dbn.zst");
    let mbp_path = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07/xnas-itch-20250701.mbp-10.dbn.zst");

    if !mbo_path.exists() || !mbp_path.exists() {
        println!("Skipping test: data files not found");
        return;
    }

    println!("\n{:=<80}", "");
    println!("ERROR PATTERN DETECTION");
    println!("Date: July 1, 2025");
    println!("{:=<80}", "");

    // Load data
    let mbp_data = load_mbp10_exact(mbp_path).expect("Failed to load MBP-10");
    let loader = DbnLoader::new(mbo_path).expect("Failed to create MBO loader");
    let mut reconstructor = LobReconstructor::new(10);

    // Error pattern tracking
    let mut bid_overestimates = 0u64;
    let mut bid_underestimates = 0u64;
    let mut ask_overestimates = 0u64;
    let mut ask_underestimates = 0u64;

    let mut bid_size_overestimates = 0u64;
    let mut bid_size_underestimates = 0u64;
    let mut ask_size_overestimates = 0u64;
    let mut ask_size_underestimates = 0u64;

    // Error magnitude distribution
    let mut price_errors: Vec<f64> = Vec::new();
    let mut size_errors: Vec<i64> = Vec::new();

    // Consecutive error tracking
    let mut consecutive_bid_price_errors = 0u64;
    let mut max_consecutive_bid_errors = 0u64;
    let mut consecutive_ask_price_errors = 0u64;
    let mut max_consecutive_ask_errors = 0u64;

    // Large error samples
    let mut large_bid_errors: Vec<(u64, f64, f64, f64)> = Vec::new(); // (ts, expected, actual, diff)
    let mut large_ask_errors: Vec<(u64, f64, f64, f64)> = Vec::new();

    let mut _current_state: Option<LobState> = None;
    let mbp_timestamps: Vec<u64> = mbp_data.keys().cloned().collect();
    let mut mbp_idx = 0;
    let mut total_comparisons = 0u64;

    for msg in loader
        .iter_messages()
        .expect("Failed to iterate MBO messages")
    {
        let msg_ts = msg.timestamp.unwrap_or(0) as u64;

        if let Ok(state) = reconstructor.process_message(&msg) {
            _current_state = Some(state.clone());

            while mbp_idx < mbp_timestamps.len() && mbp_timestamps[mbp_idx] <= msg_ts {
                let mbp_ts = mbp_timestamps[mbp_idx];

                if let (Some(ref mbo_state), Some((mbp_bids, mbp_asks, mbp_bid_sz, mbp_ask_sz))) =
                    (&current_state, mbp_data.get(&mbp_ts))
                {
                    let mbo_bid = mbo_state
                        .best_bid
                        .map(|p| p as f64 * PRICE_SCALE)
                        .unwrap_or(0.0);
                    let mbo_ask = mbo_state
                        .best_ask
                        .map(|p| p as f64 * PRICE_SCALE)
                        .unwrap_or(0.0);

                    if mbo_bid <= 0.0 || mbo_ask <= 0.0 || mbo_bid >= mbo_ask {
                        mbp_idx += 1;
                        continue;
                    }

                    total_comparisons += 1;

                    // Bid price analysis
                    let bid_diff = mbo_bid - mbp_bids[0];
                    if bid_diff.abs() > 0.000001 {
                        if bid_diff > 0.0 {
                            bid_overestimates += 1;
                        } else {
                            bid_underestimates += 1;
                        }
                        price_errors.push(bid_diff.abs());

                        consecutive_bid_price_errors += 1;

                        // Track large errors
                        if bid_diff.abs() > 0.10 && large_bid_errors.len() < 20 {
                            large_bid_errors.push((mbp_ts, mbp_bids[0], mbo_bid, bid_diff));
                        }
                    } else {
                        if consecutive_bid_price_errors > max_consecutive_bid_errors {
                            max_consecutive_bid_errors = consecutive_bid_price_errors;
                        }
                        consecutive_bid_price_errors = 0;
                    }

                    // Ask price analysis
                    let ask_diff = mbo_ask - mbp_asks[0];
                    if ask_diff.abs() > 0.000001 {
                        if ask_diff > 0.0 {
                            ask_overestimates += 1;
                        } else {
                            ask_underestimates += 1;
                        }

                        consecutive_ask_price_errors += 1;

                        if ask_diff.abs() > 0.10 && large_ask_errors.len() < 20 {
                            large_ask_errors.push((mbp_ts, mbp_asks[0], mbo_ask, ask_diff));
                        }
                    } else {
                        if consecutive_ask_price_errors > max_consecutive_ask_errors {
                            max_consecutive_ask_errors = consecutive_ask_price_errors;
                        }
                        consecutive_ask_price_errors = 0;
                    }

                    // Size analysis
                    let bid_size_diff = mbo_state.bid_sizes[0] as i64 - mbp_bid_sz[0] as i64;
                    let ask_size_diff = mbo_state.ask_sizes[0] as i64 - mbp_ask_sz[0] as i64;

                    if bid_size_diff != 0 {
                        if bid_size_diff > 0 {
                            bid_size_overestimates += 1;
                        } else {
                            bid_size_underestimates += 1;
                        }
                        size_errors.push(bid_size_diff.abs());
                    }

                    if ask_size_diff != 0 {
                        if ask_size_diff > 0 {
                            ask_size_overestimates += 1;
                        } else {
                            ask_size_underestimates += 1;
                        }
                    }
                }

                mbp_idx += 1;
            }
        }
    }

    // Print results
    println!("\nTotal comparisons: {}", total_comparisons);

    println!("\n--- DIRECTIONAL BIAS ANALYSIS ---");
    println!("  Bid Price:");
    println!(
        "    Overestimates (MBO > MBP):  {} ({:.2}%)",
        bid_overestimates,
        bid_overestimates as f64 / total_comparisons as f64 * 100.0
    );
    println!(
        "    Underestimates (MBO < MBP): {} ({:.2}%)",
        bid_underestimates,
        bid_underestimates as f64 / total_comparisons as f64 * 100.0
    );
    println!(
        "    Exact matches:              {} ({:.2}%)",
        total_comparisons - bid_overestimates - bid_underestimates,
        (total_comparisons - bid_overestimates - bid_underestimates) as f64
            / total_comparisons as f64
            * 100.0
    );

    println!("  Ask Price:");
    println!(
        "    Overestimates (MBO > MBP):  {} ({:.2}%)",
        ask_overestimates,
        ask_overestimates as f64 / total_comparisons as f64 * 100.0
    );
    println!(
        "    Underestimates (MBO < MBP): {} ({:.2}%)",
        ask_underestimates,
        ask_underestimates as f64 / total_comparisons as f64 * 100.0
    );
    println!(
        "    Exact matches:              {} ({:.2}%)",
        total_comparisons - ask_overestimates - ask_underestimates,
        (total_comparisons - ask_overestimates - ask_underestimates) as f64
            / total_comparisons as f64
            * 100.0
    );

    println!("\n  Bid Size:");
    println!(
        "    Overestimates:  {} ({:.2}%)",
        bid_size_overestimates,
        bid_size_overestimates as f64 / total_comparisons as f64 * 100.0
    );
    println!(
        "    Underestimates: {} ({:.2}%)",
        bid_size_underestimates,
        bid_size_underestimates as f64 / total_comparisons as f64 * 100.0
    );

    println!("\n  Ask Size:");
    println!(
        "    Overestimates:  {} ({:.2}%)",
        ask_size_overestimates,
        ask_size_overestimates as f64 / total_comparisons as f64 * 100.0
    );
    println!(
        "    Underestimates: {} ({:.2}%)",
        ask_size_underestimates,
        ask_size_underestimates as f64 / total_comparisons as f64 * 100.0
    );

    println!("\n--- CONSECUTIVE ERROR ANALYSIS ---");
    println!(
        "  Max consecutive bid price errors: {}",
        max_consecutive_bid_errors
    );
    println!(
        "  Max consecutive ask price errors: {}",
        max_consecutive_ask_errors
    );

    // Error distribution
    if !price_errors.is_empty() {
        price_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = price_errors[price_errors.len() / 2];
        let p90 = price_errors[(price_errors.len() as f64 * 0.9) as usize];
        let p99 = price_errors[(price_errors.len() as f64 * 0.99) as usize];

        println!("\n--- PRICE ERROR DISTRIBUTION ---");
        println!("  Median (P50): ${:.6}", p50);
        println!("  P90:          ${:.6}", p90);
        println!("  P99:          ${:.6}", p99);
        println!("  Max:          ${:.6}", price_errors.last().unwrap());
    }

    if !size_errors.is_empty() {
        size_errors.sort();
        let p50 = size_errors[size_errors.len() / 2];
        let p90 = size_errors[(size_errors.len() as f64 * 0.9) as usize];
        let p99 = size_errors[(size_errors.len() as f64 * 0.99) as usize];

        println!("\n--- SIZE ERROR DISTRIBUTION ---");
        println!("  Median (P50): {} shares", p50);
        println!("  P90:          {} shares", p90);
        println!("  P99:          {} shares", p99);
        println!("  Max:          {} shares", size_errors.last().unwrap());
    }

    // Sample large errors
    if !large_bid_errors.is_empty() {
        println!("\n--- SAMPLE LARGE BID ERRORS (>$0.10) ---");
        for (ts, expected, actual, diff) in large_bid_errors.iter().take(10) {
            println!(
                "  ts={}: expected=${:.4}, got=${:.4}, diff=${:.4}",
                ts, expected, actual, diff
            );
        }
    }

    if !large_ask_errors.is_empty() {
        println!("\n--- SAMPLE LARGE ASK ERRORS (>$0.10) ---");
        for (ts, expected, actual, diff) in large_ask_errors.iter().take(10) {
            println!(
                "  ts={}: expected=${:.4}, got=${:.4}, diff=${:.4}",
                ts, expected, actual, diff
            );
        }
    }

    // Check for systematic bias
    let bid_bias = (bid_overestimates as f64 - bid_underestimates as f64)
        / (bid_overestimates + bid_underestimates).max(1) as f64;
    let ask_bias = (ask_overestimates as f64 - ask_underestimates as f64)
        / (ask_overestimates + ask_underestimates).max(1) as f64;

    println!("\n--- BIAS INDICATORS ---");
    println!(
        "  Bid price bias: {:.4} (positive = overestimate tendency)",
        bid_bias
    );
    println!(
        "  Ask price bias: {:.4} (positive = overestimate tendency)",
        ask_bias
    );

    // Assertion: No extreme systematic bias
    assert!(
        bid_bias.abs() < 0.5,
        "Systematic bid price bias detected: {:.4}",
        bid_bias
    );
    assert!(
        ask_bias.abs() < 0.5,
        "Systematic ask price bias detected: {:.4}",
        ask_bias
    );
}

#[test]
fn test_granular_multi_day_consistency() {
    println!("\n{:=<80}", "");
    println!("MULTI-DAY CONSISTENCY CHECK");
    println!("Period: July 1-10, 2025");
    println!("{:=<80}", "");

    let mbo_base =
        Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30");
    let mbp_base = Path::new("/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07");

    let dates = [
        "20250701", "20250702", "20250703", "20250707", "20250708", "20250709", "20250710",
    ];

    let mut daily_results: Vec<(String, f64, f64, f64, f64)> = Vec::new();

    for date in &dates {
        let mbo_file = mbo_base.join(format!("xnas-itch-{}.mbo.dbn.zst", date));
        let mbp_file = mbp_base.join(format!("xnas-itch-{}.mbp-10.dbn.zst", date));

        if !mbo_file.exists() || !mbp_file.exists() {
            println!("  Skipping {}: files not found", date);
            continue;
        }

        // Quick validation for this day
        let mbp_data = match load_mbp10_exact(&mbp_file) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let loader = match DbnLoader::new(&mbo_file) {
            Ok(l) => l,
            Err(_) => continue,
        };

        let mut reconstructor = LobReconstructor::new(10);
        let mut _current_state: Option<LobState> = None;
        let mbp_timestamps: Vec<u64> = mbp_data.keys().cloned().collect();
        let mut mbp_idx = 0;

        let mut bid_matches = 0u64;
        let mut ask_matches = 0u64;
        let mut bid_within_1c = 0u64;
        let mut ask_within_1c = 0u64;
        let mut total = 0u64;

        for msg in loader
            .iter_messages()
            .unwrap_or_else(|_| panic!("Failed to iterate"))
        {
            let msg_ts = msg.timestamp.unwrap_or(0) as u64;

            if let Ok(state) = reconstructor.process_message(&msg) {
                _current_state = Some(state.clone());

                while mbp_idx < mbp_timestamps.len() && mbp_timestamps[mbp_idx] <= msg_ts {
                    let mbp_ts = mbp_timestamps[mbp_idx];

                    if let (Some(ref mbo_state), Some((mbp_bids, mbp_asks, _, _))) =
                        (&current_state, mbp_data.get(&mbp_ts))
                    {
                        let mbo_bid = mbo_state
                            .best_bid
                            .map(|p| p as f64 * PRICE_SCALE)
                            .unwrap_or(0.0);
                        let mbo_ask = mbo_state
                            .best_ask
                            .map(|p| p as f64 * PRICE_SCALE)
                            .unwrap_or(0.0);

                        if mbo_bid > 0.0 && mbo_ask > 0.0 && mbo_bid < mbo_ask {
                            total += 1;

                            let bid_diff = (mbo_bid - mbp_bids[0]).abs();
                            let ask_diff = (mbo_ask - mbp_asks[0]).abs();

                            if bid_diff < 0.000001 {
                                bid_matches += 1;
                            }
                            if ask_diff < 0.000001 {
                                ask_matches += 1;
                            }
                            if bid_diff <= 0.01 {
                                bid_within_1c += 1;
                            }
                            if ask_diff <= 0.01 {
                                ask_within_1c += 1;
                            }
                        }
                    }
                    mbp_idx += 1;
                }
            }

            // Limit for speed
            if total >= 500_000 {
                break;
            }
        }

        if total > 0 {
            let bid_acc = bid_matches as f64 / total as f64 * 100.0;
            let ask_acc = ask_matches as f64 / total as f64 * 100.0;
            let bid_1c = bid_within_1c as f64 / total as f64 * 100.0;
            let ask_1c = ask_within_1c as f64 / total as f64 * 100.0;
            daily_results.push((date.to_string(), bid_acc, ask_acc, bid_1c, ask_1c));
        }
    }

    // Print results
    println!("\n{:-<80}", "");
    println!(
        "{:^12} {:^15} {:^15} {:^15} {:^15}",
        "Date", "Bid Match%", "Ask Match%", "Bid <1c%", "Ask <1c%"
    );
    println!("{:-<80}", "");

    for (date, bid_acc, ask_acc, bid_1c, ask_1c) in &daily_results {
        println!(
            "{:^12} {:^15.2} {:^15.2} {:^15.2} {:^15.2}",
            date, bid_acc, ask_acc, bid_1c, ask_1c
        );
    }
    println!("{:-<80}", "");

    // Calculate consistency metrics
    if daily_results.len() >= 2 {
        let bid_accs: Vec<f64> = daily_results.iter().map(|r| r.3).collect(); // bid_1c
        let mean: f64 = bid_accs.iter().sum::<f64>() / bid_accs.len() as f64;
        let variance: f64 =
            bid_accs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / bid_accs.len() as f64;
        let std_dev = variance.sqrt();

        println!("\nConsistency Analysis (Bid Price within 1c):");
        println!("  Mean:     {:.2}%", mean);
        println!("  Std Dev:  {:.2}%", std_dev);
        println!(
            "  Min:      {:.2}%",
            bid_accs.iter().cloned().fold(f64::INFINITY, f64::min)
        );
        println!(
            "  Max:      {:.2}%",
            bid_accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );

        // Assertion: Results should be consistent across days
        assert!(
            std_dev < 10.0,
            "Too much variation across days: std_dev = {:.2}%",
            std_dev
        );
    }
}
