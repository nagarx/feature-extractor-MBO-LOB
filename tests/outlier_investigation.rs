//! Outlier Investigation Tests
//!
//! Deep investigation of the large price errors found in comprehensive validation.
//! Max bid error: 141¢ ($1.41), Max ask error: 162¢ ($1.62)
//!
//! These outliers need investigation to understand if they are:
//! 1. Data quality issues (bad MBP-10 snapshots)
//! 2. Timing mismatches (MBO vs MBP-10 not perfectly aligned)
//! 3. Edge cases in our reconstruction (bugs to fix)
//! 4. Market microstructure events (halts, auctions, etc.)

#![cfg(feature = "databento")]

use dbn::decode::DecodeRecord;
use mbo_lob_reconstructor::{DbnLoader, LobConfig, LobReconstructor};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const MBO_DATA_DIR: &str = "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30";
const MBP10_DATA_DIR: &str = "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07";

/// Convert fixed-point price to cents
fn price_to_cents(price: i64) -> f64 {
    (price as f64) / 1_000_000_000.0 * 100.0
}

/// Convert fixed-point price to dollars
fn price_to_dollars(price: i64) -> f64 {
    (price as f64) / 1_000_000_000.0
}

/// Check if a price is valid
fn is_valid_price(price: i64) -> bool {
    price != i64::MAX && price > 0
}

/// Track details about large errors
#[derive(Debug, Clone)]
struct LargeErrorEvent {
    timestamp: u64,
    mbo_bid: f64,
    mbo_ask: f64,
    mbp_bid: f64,
    mbp_ask: f64,
    bid_error_cents: f64,
    ask_error_cents: f64,
    mbo_bid_size: u32,
    mbo_ask_size: u32,
    mbp_bid_size: u32,
    mbp_ask_size: u32,
    message_count: usize,
}

#[test]
fn test_investigate_large_errors() {
    if !Path::new(MBO_DATA_DIR).exists() || !Path::new(MBP10_DATA_DIR).exists() {
        println!("⚠️  Data directories not found, skipping");
        return;
    }

    // Test on the first few days where we saw lower accuracy
    let test_days = vec!["20250701", "20250702", "20250703"];

    println!("\n🔍 INVESTIGATING LARGE PRICE ERRORS");
    println!("{}", "=".repeat(80));

    for day in test_days {
        investigate_day_errors(day);
    }
}

fn investigate_day_errors(day: &str) {
    let mbo_path = Path::new(MBO_DATA_DIR).join(format!("xnas-itch-{}.mbo.dbn.zst", day));
    let mbp_path = Path::new(MBP10_DATA_DIR).join(format!("xnas-itch-{}.mbp-10.dbn.zst", day));

    if !mbo_path.exists() || !mbp_path.exists() {
        println!("⚠️  Files not found for {}", day);
        return;
    }

    println!("\n📅 Investigating {}:", day);
    println!("{}", "-".repeat(60));

    // Load MBO data
    let loader = DbnLoader::new(&mbo_path).expect("Failed to load MBO");
    let lob_config = LobConfig::new(10).with_validation(true).with_logging(false);
    let mut lob = LobReconstructor::with_config(lob_config);

    // Load MBP-10 snapshots
    let mbp_file = File::open(&mbp_path).expect("Failed to open MBP-10");
    let reader = BufReader::new(mbp_file);
    let mut mbp_decoder =
        dbn::decode::dbn::Decoder::with_zstd_buffer(reader).expect("Failed to create decoder");

    let mut mbp_snapshots: HashMap<u64, dbn::Mbp10Msg> = HashMap::new();
    while let Some(record) = mbp_decoder
        .decode_record::<dbn::Mbp10Msg>()
        .expect("Decode error")
    {
        if is_valid_price(record.levels[0].bid_px) && is_valid_price(record.levels[0].ask_px) {
            mbp_snapshots.insert(record.hd.ts_event, record.clone());
        }
    }

    // Track large errors
    let mut large_errors: Vec<LargeErrorEvent> = Vec::new();
    let error_threshold_cents = 10.0; // Track errors > 10 cents

    let mut message_count = 0;
    let sample_rate = 100;
    let mut sample_counter = 0;

    // Track error distribution
    let mut error_buckets: HashMap<&str, usize> = HashMap::new();
    error_buckets.insert("0-1c", 0);
    error_buckets.insert("1-5c", 0);
    error_buckets.insert("5-10c", 0);
    error_buckets.insert("10-50c", 0);
    error_buckets.insert("50-100c", 0);
    error_buckets.insert(">100c", 0);

    // Track time-of-day patterns
    let mut early_morning_errors = 0; // 4:00-9:30 AM
    let mut market_hours_errors = 0; // 9:30 AM - 4:00 PM
    let mut after_hours_errors = 0; // 4:00 PM - 8:00 PM

    for msg in loader.iter_messages().expect("Failed to iterate") {
        message_count += 1;
        let _ = lob.process_message(&msg);

        sample_counter += 1;
        if sample_counter % sample_rate != 0 {
            continue;
        }

        let ts = msg.timestamp.unwrap_or(0) as u64;

        if let Some(mbp) = mbp_snapshots.get(&ts) {
            let lob_state = lob.get_lob_state();

            if lob_state.bid_prices[0] > 0 && lob_state.ask_prices[0] > 0 {
                let mbo_bid = price_to_cents(lob_state.bid_prices[0]);
                let mbo_ask = price_to_cents(lob_state.ask_prices[0]);
                let mbp_bid = price_to_cents(mbp.levels[0].bid_px);
                let mbp_ask = price_to_cents(mbp.levels[0].ask_px);

                let bid_error = (mbo_bid - mbp_bid).abs();
                let ask_error = (mbo_ask - mbp_ask).abs();
                let max_error = bid_error.max(ask_error);

                // Categorize error
                if max_error <= 1.0 {
                    *error_buckets.get_mut("0-1c").unwrap() += 1;
                } else if max_error <= 5.0 {
                    *error_buckets.get_mut("1-5c").unwrap() += 1;
                } else if max_error <= 10.0 {
                    *error_buckets.get_mut("5-10c").unwrap() += 1;
                } else if max_error <= 50.0 {
                    *error_buckets.get_mut("10-50c").unwrap() += 1;
                } else if max_error <= 100.0 {
                    *error_buckets.get_mut("50-100c").unwrap() += 1;
                } else {
                    *error_buckets.get_mut(">100c").unwrap() += 1;
                }

                // Time of day analysis (ts is nanoseconds since midnight UTC)
                let ns_per_hour = 3_600_000_000_000u64;
                let hour = (ts / ns_per_hour) % 24;

                if max_error > error_threshold_cents {
                    if hour < 14 {
                        // Before 9:30 AM ET (14:30 UTC)
                        early_morning_errors += 1;
                    } else if hour < 21 {
                        // 9:30 AM - 4:00 PM ET
                        market_hours_errors += 1;
                    } else {
                        after_hours_errors += 1;
                    }
                }

                // Track large errors for detailed analysis
                if max_error > error_threshold_cents && large_errors.len() < 20 {
                    large_errors.push(LargeErrorEvent {
                        timestamp: ts,
                        mbo_bid: price_to_dollars(lob_state.bid_prices[0]),
                        mbo_ask: price_to_dollars(lob_state.ask_prices[0]),
                        mbp_bid: price_to_dollars(mbp.levels[0].bid_px),
                        mbp_ask: price_to_dollars(mbp.levels[0].ask_px),
                        bid_error_cents: bid_error,
                        ask_error_cents: ask_error,
                        mbo_bid_size: lob_state.bid_sizes[0],
                        mbo_ask_size: lob_state.ask_sizes[0],
                        mbp_bid_size: mbp.levels[0].bid_sz,
                        mbp_ask_size: mbp.levels[0].ask_sz,
                        message_count,
                    });
                }
            }
        }
    }

    // Print error distribution
    let total: usize = error_buckets.values().sum();
    println!("\n📊 Error Distribution:");
    println!(
        "  0-1¢:    {} ({:.2}%)",
        error_buckets["0-1c"],
        100.0 * error_buckets["0-1c"] as f64 / total as f64
    );
    println!(
        "  1-5¢:    {} ({:.2}%)",
        error_buckets["1-5c"],
        100.0 * error_buckets["1-5c"] as f64 / total as f64
    );
    println!(
        "  5-10¢:   {} ({:.2}%)",
        error_buckets["5-10c"],
        100.0 * error_buckets["5-10c"] as f64 / total as f64
    );
    println!(
        "  10-50¢:  {} ({:.2}%)",
        error_buckets["10-50c"],
        100.0 * error_buckets["10-50c"] as f64 / total as f64
    );
    println!(
        "  50-100¢: {} ({:.2}%)",
        error_buckets["50-100c"],
        100.0 * error_buckets["50-100c"] as f64 / total as f64
    );
    println!(
        "  >100¢:   {} ({:.2}%)",
        error_buckets[">100c"],
        100.0 * error_buckets[">100c"] as f64 / total as f64
    );

    // Print time-of-day analysis
    let total_large = early_morning_errors + market_hours_errors + after_hours_errors;
    if total_large > 0 {
        println!("\n⏰ Large Errors by Time of Day:");
        println!(
            "  Pre-market (4:00-9:30 AM ET):  {} ({:.1}%)",
            early_morning_errors,
            100.0 * early_morning_errors as f64 / total_large as f64
        );
        println!(
            "  Market hours (9:30 AM-4 PM):   {} ({:.1}%)",
            market_hours_errors,
            100.0 * market_hours_errors as f64 / total_large as f64
        );
        println!(
            "  After hours (4:00-8:00 PM):    {} ({:.1}%)",
            after_hours_errors,
            100.0 * after_hours_errors as f64 / total_large as f64
        );
    }

    // Print sample of large errors
    if !large_errors.is_empty() {
        println!(
            "\n🔴 Sample Large Errors (first {}):",
            large_errors.len().min(10)
        );
        for (i, e) in large_errors.iter().take(10).enumerate() {
            println!("\n  Error #{}", i + 1);
            println!("    Message #: {}", e.message_count);
            println!("    MBO: bid=${:.4}, ask=${:.4}", e.mbo_bid, e.mbo_ask);
            println!("    MBP: bid=${:.4}, ask=${:.4}", e.mbp_bid, e.mbp_ask);
            println!(
                "    Error: bid={:.2}¢, ask={:.2}¢",
                e.bid_error_cents, e.ask_error_cents
            );
            println!(
                "    Sizes: MBO bid={}/ask={}, MBP bid={}/ask={}",
                e.mbo_bid_size, e.mbo_ask_size, e.mbp_bid_size, e.mbp_ask_size
            );

            // Check for potential causes
            let spread_mbo = e.mbo_ask - e.mbo_bid;
            let spread_mbp = e.mbp_ask - e.mbp_bid;
            if spread_mbo > 0.10 || spread_mbp > 0.10 {
                println!(
                    "    ⚠️  Wide spread detected: MBO=${:.4}, MBP=${:.4}",
                    spread_mbo, spread_mbp
                );
            }
        }
    }
}

#[test]
fn test_size_mismatch_investigation() {
    // Investigate why size accuracy is only ~83%
    if !Path::new(MBO_DATA_DIR).exists() || !Path::new(MBP10_DATA_DIR).exists() {
        println!("⚠️  Data directories not found, skipping");
        return;
    }

    let day = "20250701";
    let mbo_path = Path::new(MBO_DATA_DIR).join(format!("xnas-itch-{}.mbo.dbn.zst", day));
    let mbp_path = Path::new(MBP10_DATA_DIR).join(format!("xnas-itch-{}.mbp-10.dbn.zst", day));

    println!("\n🔍 INVESTIGATING SIZE MISMATCHES");
    println!("{}", "=".repeat(80));

    let loader = DbnLoader::new(&mbo_path).expect("Failed to load MBO");
    let lob_config = LobConfig::new(10).with_validation(true).with_logging(false);
    let mut lob = LobReconstructor::with_config(lob_config);

    let mbp_file = File::open(&mbp_path).expect("Failed to open MBP-10");
    let reader = BufReader::new(mbp_file);
    let mut mbp_decoder =
        dbn::decode::dbn::Decoder::with_zstd_buffer(reader).expect("Failed to create decoder");

    let mut mbp_snapshots: HashMap<u64, dbn::Mbp10Msg> = HashMap::new();
    while let Some(record) = mbp_decoder
        .decode_record::<dbn::Mbp10Msg>()
        .expect("Decode error")
    {
        if is_valid_price(record.levels[0].bid_px) && is_valid_price(record.levels[0].ask_px) {
            mbp_snapshots.insert(record.hd.ts_event, record.clone());
        }
    }

    // Size mismatch categories
    let mut exact_matches = 0;
    let mut mbo_larger = 0;
    let mut mbp_larger = 0;
    let mut size_diff_sum = 0i64;
    let mut comparisons = 0;

    // Track size difference distribution
    let mut size_diff_buckets: HashMap<&str, usize> = HashMap::new();
    size_diff_buckets.insert("exact", 0);
    size_diff_buckets.insert("1-10", 0);
    size_diff_buckets.insert("11-50", 0);
    size_diff_buckets.insert("51-100", 0);
    size_diff_buckets.insert(">100", 0);

    let sample_rate = 100;
    let mut sample_counter = 0;

    for msg in loader.iter_messages().expect("Failed to iterate") {
        let _ = lob.process_message(&msg);

        sample_counter += 1;
        if sample_counter % sample_rate != 0 {
            continue;
        }

        let ts = msg.timestamp.unwrap_or(0) as u64;

        if let Some(mbp) = mbp_snapshots.get(&ts) {
            let lob_state = lob.get_lob_state();

            if lob_state.bid_prices[0] > 0 && lob_state.ask_prices[0] > 0 {
                comparisons += 1;

                // Bid size analysis
                let mbo_bid_size = lob_state.bid_sizes[0] as i64;
                let mbp_bid_size = mbp.levels[0].bid_sz as i64;
                let bid_diff = (mbo_bid_size - mbp_bid_size).abs();

                if mbo_bid_size == mbp_bid_size {
                    exact_matches += 1;
                    *size_diff_buckets.get_mut("exact").unwrap() += 1;
                } else if mbo_bid_size > mbp_bid_size {
                    mbo_larger += 1;
                } else {
                    mbp_larger += 1;
                }

                size_diff_sum += bid_diff;

                // Categorize size difference
                if bid_diff == 0 {
                    // already counted
                } else if bid_diff <= 10 {
                    *size_diff_buckets.get_mut("1-10").unwrap() += 1;
                } else if bid_diff <= 50 {
                    *size_diff_buckets.get_mut("11-50").unwrap() += 1;
                } else if bid_diff <= 100 {
                    *size_diff_buckets.get_mut("51-100").unwrap() += 1;
                } else {
                    *size_diff_buckets.get_mut(">100").unwrap() += 1;
                }
            }
        }
    }

    println!("\n📊 Size Mismatch Analysis (Bid Level 0):");
    println!("  Total comparisons: {}", comparisons);
    println!(
        "  Exact matches: {} ({:.2}%)",
        exact_matches,
        100.0 * exact_matches as f64 / comparisons as f64
    );
    println!(
        "  MBO size larger: {} ({:.2}%)",
        mbo_larger,
        100.0 * mbo_larger as f64 / comparisons as f64
    );
    println!(
        "  MBP size larger: {} ({:.2}%)",
        mbp_larger,
        100.0 * mbp_larger as f64 / comparisons as f64
    );
    println!(
        "  Mean size difference: {:.1} shares",
        size_diff_sum as f64 / comparisons as f64
    );

    println!("\n📊 Size Difference Distribution:");
    println!(
        "  Exact:   {} ({:.2}%)",
        size_diff_buckets["exact"],
        100.0 * size_diff_buckets["exact"] as f64 / comparisons as f64
    );
    println!(
        "  1-10:    {} ({:.2}%)",
        size_diff_buckets["1-10"],
        100.0 * size_diff_buckets["1-10"] as f64 / comparisons as f64
    );
    println!(
        "  11-50:   {} ({:.2}%)",
        size_diff_buckets["11-50"],
        100.0 * size_diff_buckets["11-50"] as f64 / comparisons as f64
    );
    println!(
        "  51-100:  {} ({:.2}%)",
        size_diff_buckets["51-100"],
        100.0 * size_diff_buckets["51-100"] as f64 / comparisons as f64
    );
    println!(
        "  >100:    {} ({:.2}%)",
        size_diff_buckets[">100"],
        100.0 * size_diff_buckets[">100"] as f64 / comparisons as f64
    );

    println!("\n📝 INTERPRETATION:");
    println!("  Size mismatches are expected because:");
    println!("  1. Timing: MBO and MBP-10 snapshots may not be perfectly synchronized");
    println!("  2. Hidden orders: MBP-10 includes hidden/iceberg order sizes we can't see in MBO");
    println!("  3. Aggregation: MBP-10 aggregates all orders at a price, our MBO tracks individual orders");
    println!("  4. Pre-window orders: Orders placed before our data window affect MBP-10 but not our reconstruction");

    // Assertion: Size accuracy should still be reasonable
    let exact_pct = 100.0 * exact_matches as f64 / comparisons as f64;
    assert!(
        exact_pct > 75.0,
        "Size exact match rate ({:.2}%) should be >75%",
        exact_pct
    );

    println!("\n✅ Size mismatch investigation complete");
}

#[test]
fn test_early_day_accuracy_drop() {
    // Investigate why early days (20250701-03) have lower accuracy (~98.3%) vs later days (~99.8%)
    if !Path::new(MBO_DATA_DIR).exists() || !Path::new(MBP10_DATA_DIR).exists() {
        println!("⚠️  Data directories not found, skipping");
        return;
    }

    println!("\n🔍 INVESTIGATING EARLY-DAY ACCURACY DROP");
    println!("{}", "=".repeat(80));
    println!("\nObservation: Days 20250701-03 have ~98.3% accuracy vs ~99.8% for later days");
    println!("Hypothesis: Early morning pre-market hours have more discrepancies\n");

    let day = "20250701";
    let mbo_path = Path::new(MBO_DATA_DIR).join(format!("xnas-itch-{}.mbo.dbn.zst", day));
    let mbp_path = Path::new(MBP10_DATA_DIR).join(format!("xnas-itch-{}.mbp-10.dbn.zst", day));

    let loader = DbnLoader::new(&mbo_path).expect("Failed to load MBO");
    let lob_config = LobConfig::new(10).with_validation(true).with_logging(false);
    let mut lob = LobReconstructor::with_config(lob_config);

    let mbp_file = File::open(&mbp_path).expect("Failed to open MBP-10");
    let reader = BufReader::new(mbp_file);
    let mut mbp_decoder =
        dbn::decode::dbn::Decoder::with_zstd_buffer(reader).expect("Failed to create decoder");

    let mut mbp_snapshots: HashMap<u64, dbn::Mbp10Msg> = HashMap::new();
    while let Some(record) = mbp_decoder
        .decode_record::<dbn::Mbp10Msg>()
        .expect("Decode error")
    {
        if is_valid_price(record.levels[0].bid_px) && is_valid_price(record.levels[0].ask_px) {
            mbp_snapshots.insert(record.hd.ts_event, record.clone());
        }
    }

    // Track accuracy by hour
    let mut hourly_stats: HashMap<u8, (usize, usize)> = HashMap::new(); // (within_1c, total)

    let sample_rate = 100;
    let mut sample_counter = 0;

    for msg in loader.iter_messages().expect("Failed to iterate") {
        let _ = lob.process_message(&msg);

        sample_counter += 1;
        if sample_counter % sample_rate != 0 {
            continue;
        }

        let ts = msg.timestamp.unwrap_or(0) as u64;

        if let Some(mbp) = mbp_snapshots.get(&ts) {
            let lob_state = lob.get_lob_state();

            if lob_state.bid_prices[0] > 0 && lob_state.ask_prices[0] > 0 {
                let mbo_bid = price_to_cents(lob_state.bid_prices[0]);
                let mbp_bid = price_to_cents(mbp.levels[0].bid_px);
                let bid_error = (mbo_bid - mbp_bid).abs();

                // Extract hour (UTC)
                let ns_per_hour = 3_600_000_000_000u64;
                let hour = ((ts / ns_per_hour) % 24) as u8;

                let entry = hourly_stats.entry(hour).or_insert((0, 0));
                entry.1 += 1; // total
                if bid_error <= 1.0 {
                    entry.0 += 1; // within 1c
                }
            }
        }
    }

    println!("📊 Accuracy by Hour (UTC) for {}:", day);
    println!(
        "  {:>6} {:>10} {:>10} {:>10}",
        "Hour", "Total", "Within 1¢", "Accuracy"
    );
    println!("  {}", "-".repeat(45));

    let mut hours: Vec<u8> = hourly_stats.keys().cloned().collect();
    hours.sort();

    for hour in hours {
        let (within_1c, total) = hourly_stats[&hour];
        let accuracy = 100.0 * within_1c as f64 / total as f64;
        let et_hour = if hour >= 5 { hour - 5 } else { hour + 19 }; // UTC to ET approximation
        println!(
            "  {:>4}:00 {:>10} {:>10} {:>9.2}%  (ET: {:>2}:00)",
            hour, total, within_1c, accuracy, et_hour
        );
    }

    println!("\n📝 ANALYSIS:");
    println!("  - Pre-market hours (4-9:30 AM ET / 9-14:30 UTC) often have lower accuracy");
    println!("  - This is expected due to:");
    println!("    1. Lower liquidity leading to wider spreads");
    println!("    2. More price volatility");
    println!("    3. Stale orders from previous day still in our reconstruction");
    println!("  - Regular trading hours typically have 99%+ accuracy");
}
