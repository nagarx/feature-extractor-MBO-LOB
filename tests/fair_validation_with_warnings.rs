//! Fair Validation Test with Warning Statistics
//!
//! This test provides an unbiased comparison of MBO reconstruction vs MBP-10
//! ground truth, including detailed warning statistics from the new soft
//! error handling infrastructure.
//!
//! This test requires the `databento` feature to be enabled.

#![cfg(feature = "databento")]

use dbn::decode::DecodeRecord;
use mbo_lob_reconstructor::{DbnLoader, LobConfig, LobReconstructor, LobStats};
use std::collections::HashMap;
use std::path::Path;

/// Test configuration
const MBO_DATA_DIR: &str = "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30";
const MBP10_DATA_DIR: &str = "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_MBP10_2025-07";

/// Results structure for fair comparison
#[derive(Debug, Default)]
struct FairValidationResults {
    // Basic stats
    total_mbo_messages: u64,
    total_mbp_snapshots: u64,
    aligned_comparisons: u64,

    // Price accuracy
    exact_bid_price_matches: u64,
    exact_ask_price_matches: u64,
    bid_price_within_1c: u64,
    ask_price_within_1c: u64,
    bid_price_within_5c: u64,
    ask_price_within_5c: u64,

    // Size accuracy
    exact_bid_size_matches: u64,
    exact_ask_size_matches: u64,

    // Error metrics
    total_bid_price_error: f64,
    total_ask_price_error: f64,
    max_bid_price_error: f64,
    max_ask_price_error: f64,

    // Warning stats (from LobStats)
    cancel_order_not_found: u64,
    cancel_price_level_missing: u64,
    cancel_order_at_level_missing: u64,
    trade_order_not_found: u64,
    trade_price_level_missing: u64,
    trade_order_at_level_missing: u64,
    book_clears: u64,
    noop_messages: u64,
    crossed_quotes: u64,
    locked_quotes: u64,
    errors: u64,
}

impl FairValidationResults {
    fn bid_price_exact_pct(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        100.0 * self.exact_bid_price_matches as f64 / self.aligned_comparisons as f64
    }

    fn ask_price_exact_pct(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        100.0 * self.exact_ask_price_matches as f64 / self.aligned_comparisons as f64
    }

    fn bid_price_1c_pct(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        100.0 * self.bid_price_within_1c as f64 / self.aligned_comparisons as f64
    }

    fn ask_price_1c_pct(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        100.0 * self.ask_price_within_1c as f64 / self.aligned_comparisons as f64
    }

    fn bid_size_exact_pct(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        100.0 * self.exact_bid_size_matches as f64 / self.aligned_comparisons as f64
    }

    fn ask_size_exact_pct(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        100.0 * self.exact_ask_size_matches as f64 / self.aligned_comparisons as f64
    }

    fn bid_mae(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        self.total_bid_price_error / self.aligned_comparisons as f64
    }

    fn ask_mae(&self) -> f64 {
        if self.aligned_comparisons == 0 {
            return 0.0;
        }
        self.total_ask_price_error / self.aligned_comparisons as f64
    }

    fn total_warnings(&self) -> u64 {
        self.cancel_order_not_found
            + self.cancel_price_level_missing
            + self.cancel_order_at_level_missing
            + self.trade_order_not_found
            + self.trade_price_level_missing
            + self.trade_order_at_level_missing
    }

    fn warning_rate(&self) -> f64 {
        if self.total_mbo_messages == 0 {
            return 0.0;
        }
        100.0 * self.total_warnings() as f64 / self.total_mbo_messages as f64
    }
}

/// Load MBP-10 snapshots and index by timestamp
fn load_mbp10_snapshots(dir: &str) -> std::io::Result<HashMap<u64, (f64, f64, u32, u32)>> {
    use dbn::Mbp10Msg;
    use std::fs::File;
    use std::io::BufReader;

    let mut snapshots = HashMap::new();
    let path = Path::new(dir);

    if !path.exists() {
        return Ok(snapshots);
    }

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if !file_path.to_string_lossy().ends_with(".dbn.zst") {
            continue;
        }

        let file = File::open(&file_path)?;
        let reader = BufReader::new(file);
        let mut decoder =
            dbn::decode::dbn::Decoder::with_zstd_buffer(reader).expect("Failed to create decoder");

        loop {
            match decoder.decode_record::<Mbp10Msg>() {
                Ok(Some(record)) => {
                    let ts = record.hd.ts_event;
                    let bid_price = record.levels[0].bid_px as f64 / 1e9;
                    let ask_price = record.levels[0].ask_px as f64 / 1e9;
                    let bid_size = record.levels[0].bid_sz;
                    let ask_size = record.levels[0].ask_sz;

                    // Skip invalid snapshots
                    if bid_price <= 0.0
                        || ask_price <= 0.0
                        || bid_price > 10000.0
                        || ask_price > 10000.0
                    {
                        continue;
                    }

                    snapshots.insert(ts, (bid_price, ask_price, bid_size, ask_size));
                }
                Ok(None) => break,
                Err(_) => continue,
            }
        }
    }

    Ok(snapshots)
}

/// Run fair validation for a single day
fn validate_day(
    mbo_file: &str,
    mbp_snapshots: &HashMap<u64, (f64, f64, u32, u32)>,
) -> Option<(FairValidationResults, LobStats)> {
    let path = Path::new(mbo_file);
    if !path.exists() {
        return None;
    }

    let config = LobConfig::new(10)
        .with_logging(false)
        .with_validation(false); // Disable validation for performance

    let mut lob = LobReconstructor::with_config(config);
    let loader = DbnLoader::new(mbo_file).ok()?;

    let mut results = FairValidationResults::default();

    for msg in loader.iter_messages().ok()? {
        results.total_mbo_messages += 1;

        let state = match lob.process_message(&msg) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Check if we have an MBP snapshot at this timestamp
        if let Some(ts) = msg.timestamp {
            if let Some(&(mbp_bid, mbp_ask, mbp_bid_sz, mbp_ask_sz)) =
                mbp_snapshots.get(&(ts as u64))
            {
                results.aligned_comparisons += 1;

                let mbo_bid = state.best_bid.map(|p| p as f64 / 1e9).unwrap_or(0.0);
                let mbo_ask = state.best_ask.map(|p| p as f64 / 1e9).unwrap_or(0.0);
                let mbo_bid_sz = state.bid_sizes.first().copied().unwrap_or(0);
                let mbo_ask_sz = state.ask_sizes.first().copied().unwrap_or(0);

                // Price comparisons
                let bid_diff = (mbo_bid - mbp_bid).abs();
                let ask_diff = (mbo_ask - mbp_ask).abs();

                results.total_bid_price_error += bid_diff;
                results.total_ask_price_error += ask_diff;
                results.max_bid_price_error = results.max_bid_price_error.max(bid_diff);
                results.max_ask_price_error = results.max_ask_price_error.max(ask_diff);

                if bid_diff < 0.000001 {
                    results.exact_bid_price_matches += 1;
                }
                if ask_diff < 0.000001 {
                    results.exact_ask_price_matches += 1;
                }
                if bid_diff <= 0.01 {
                    results.bid_price_within_1c += 1;
                }
                if ask_diff <= 0.01 {
                    results.ask_price_within_1c += 1;
                }
                if bid_diff <= 0.05 {
                    results.bid_price_within_5c += 1;
                }
                if ask_diff <= 0.05 {
                    results.ask_price_within_5c += 1;
                }

                // Size comparisons
                if mbo_bid_sz == mbp_bid_sz {
                    results.exact_bid_size_matches += 1;
                }
                if mbo_ask_sz == mbp_ask_sz {
                    results.exact_ask_size_matches += 1;
                }
            }
        }
    }

    // Copy warning stats from LobStats
    let stats = lob.stats();
    results.cancel_order_not_found = stats.cancel_order_not_found;
    results.cancel_price_level_missing = stats.cancel_price_level_missing;
    results.cancel_order_at_level_missing = stats.cancel_order_at_level_missing;
    results.trade_order_not_found = stats.trade_order_not_found;
    results.trade_price_level_missing = stats.trade_price_level_missing;
    results.trade_order_at_level_missing = stats.trade_order_at_level_missing;
    results.book_clears = stats.book_clears;
    results.noop_messages = stats.noop_messages;
    results.crossed_quotes = stats.crossed_quotes;
    results.locked_quotes = stats.locked_quotes;
    results.errors = stats.errors;

    Some((results, stats.clone()))
}

#[test]
#[cfg(feature = "databento")]
fn test_fair_validation_with_warnings() {
    println!("\n================================================================================");
    println!("FAIR VALIDATION TEST WITH WARNING STATISTICS");
    println!("================================================================================\n");

    // Check if data exists
    let mbo_path = Path::new(MBO_DATA_DIR);
    let mbp_path = Path::new(MBP10_DATA_DIR);

    if !mbo_path.exists() {
        println!("⚠️  MBO data directory not found: {}", MBO_DATA_DIR);
        println!("   Skipping test - no data available");
        return;
    }

    if !mbp_path.exists() {
        println!("⚠️  MBP-10 data directory not found: {}", MBP10_DATA_DIR);
        println!("   Skipping test - no ground truth available");
        return;
    }

    // Load MBP-10 ground truth
    println!("[1/3] Loading MBP-10 ground truth...");
    let mbp_snapshots = match load_mbp10_snapshots(MBP10_DATA_DIR) {
        Ok(s) => s,
        Err(e) => {
            println!("   ❌ Failed to load MBP-10 data: {}", e);
            return;
        }
    };
    println!("   ✓ Loaded {} MBP-10 snapshots", mbp_snapshots.len());

    if mbp_snapshots.is_empty() {
        println!("   ⚠️  No MBP-10 snapshots loaded - skipping test");
        return;
    }

    // Find July 2025 MBO files
    println!("\n[2/3] Processing MBO data for July 2025...");
    let mut total_results = FairValidationResults::default();
    let mut days_processed = 0;

    // Look for July files (xnas-itch-202507*.mbo.dbn.zst)
    for day in 1..=31 {
        let date_str = format!("202507{:02}", day);
        let mbo_file = format!("{}/xnas-itch-{}.mbo.dbn.zst", MBO_DATA_DIR, date_str);

        if let Some((day_results, _stats)) = validate_day(&mbo_file, &mbp_snapshots) {
            if day_results.aligned_comparisons > 0 {
                days_processed += 1;

                // Aggregate results
                total_results.total_mbo_messages += day_results.total_mbo_messages;
                total_results.aligned_comparisons += day_results.aligned_comparisons;
                total_results.exact_bid_price_matches += day_results.exact_bid_price_matches;
                total_results.exact_ask_price_matches += day_results.exact_ask_price_matches;
                total_results.bid_price_within_1c += day_results.bid_price_within_1c;
                total_results.ask_price_within_1c += day_results.ask_price_within_1c;
                total_results.bid_price_within_5c += day_results.bid_price_within_5c;
                total_results.ask_price_within_5c += day_results.ask_price_within_5c;
                total_results.exact_bid_size_matches += day_results.exact_bid_size_matches;
                total_results.exact_ask_size_matches += day_results.exact_ask_size_matches;
                total_results.total_bid_price_error += day_results.total_bid_price_error;
                total_results.total_ask_price_error += day_results.total_ask_price_error;
                total_results.max_bid_price_error = total_results
                    .max_bid_price_error
                    .max(day_results.max_bid_price_error);
                total_results.max_ask_price_error = total_results
                    .max_ask_price_error
                    .max(day_results.max_ask_price_error);

                // Warning stats
                total_results.cancel_order_not_found += day_results.cancel_order_not_found;
                total_results.cancel_price_level_missing += day_results.cancel_price_level_missing;
                total_results.cancel_order_at_level_missing +=
                    day_results.cancel_order_at_level_missing;
                total_results.trade_order_not_found += day_results.trade_order_not_found;
                total_results.trade_price_level_missing += day_results.trade_price_level_missing;
                total_results.trade_order_at_level_missing +=
                    day_results.trade_order_at_level_missing;
                total_results.book_clears += day_results.book_clears;
                total_results.noop_messages += day_results.noop_messages;
                total_results.crossed_quotes += day_results.crossed_quotes;
                total_results.locked_quotes += day_results.locked_quotes;
                total_results.errors += day_results.errors;

                println!(
                    "   2025-07-{:02} - {} MBO msgs, {} comparisons, {:.2}% bid match",
                    day,
                    day_results.total_mbo_messages,
                    day_results.aligned_comparisons,
                    day_results.bid_price_exact_pct()
                );
            }
        }
    }

    if days_processed == 0 {
        println!("   ⚠️  No matching days found - skipping test");
        return;
    }

    // Print results
    println!("\n[3/3] VALIDATION RESULTS");
    println!("================================================================================");

    println!("\n📊 DATA SUMMARY:");
    println!("   Days processed:      {}", days_processed);
    println!(
        "   Total MBO messages:  {}",
        total_results.total_mbo_messages
    );
    println!(
        "   Aligned comparisons: {}",
        total_results.aligned_comparisons
    );

    println!("\n💰 PRICE ACCURACY (BBO):");
    println!("   ┌────────────────┬──────────────┬──────────────┐");
    println!("   │ Metric         │ Bid          │ Ask          │");
    println!("   ├────────────────┼──────────────┼──────────────┤");
    println!(
        "   │ Exact Match    │ {:>10.2}% │ {:>10.2}% │",
        total_results.bid_price_exact_pct(),
        total_results.ask_price_exact_pct()
    );
    println!(
        "   │ Within 1¢      │ {:>10.2}% │ {:>10.2}% │",
        total_results.bid_price_1c_pct(),
        total_results.ask_price_1c_pct()
    );
    println!(
        "   │ MAE            │ ${:>9.6} │ ${:>9.6} │",
        total_results.bid_mae(),
        total_results.ask_mae()
    );
    println!(
        "   │ Max Error      │ ${:>9.4} │ ${:>9.4} │",
        total_results.max_bid_price_error, total_results.max_ask_price_error
    );
    println!("   └────────────────┴──────────────┴──────────────┘");

    println!("\n📦 SIZE ACCURACY (BBO):");
    println!(
        "   Bid size exact match: {:.2}%",
        total_results.bid_size_exact_pct()
    );
    println!(
        "   Ask size exact match: {:.2}%",
        total_results.ask_size_exact_pct()
    );

    println!("\n⚠️  WARNING STATISTICS:");
    println!("   ┌─────────────────────────────────┬────────────┬──────────┐");
    println!("   │ Warning Type                    │ Count      │ Rate     │");
    println!("   ├─────────────────────────────────┼────────────┼──────────┤");
    println!(
        "   │ Cancel: Order not found         │ {:>10} │ {:>6.4}% │",
        total_results.cancel_order_not_found,
        100.0 * total_results.cancel_order_not_found as f64
            / total_results.total_mbo_messages as f64
    );
    println!(
        "   │ Cancel: Price level missing     │ {:>10} │ {:>6.4}% │",
        total_results.cancel_price_level_missing,
        100.0 * total_results.cancel_price_level_missing as f64
            / total_results.total_mbo_messages as f64
    );
    println!(
        "   │ Cancel: Order at level missing  │ {:>10} │ {:>6.4}% │",
        total_results.cancel_order_at_level_missing,
        100.0 * total_results.cancel_order_at_level_missing as f64
            / total_results.total_mbo_messages as f64
    );
    println!(
        "   │ Trade: Order not found          │ {:>10} │ {:>6.4}% │",
        total_results.trade_order_not_found,
        100.0 * total_results.trade_order_not_found as f64
            / total_results.total_mbo_messages as f64
    );
    println!(
        "   │ Trade: Price level missing      │ {:>10} │ {:>6.4}% │",
        total_results.trade_price_level_missing,
        100.0 * total_results.trade_price_level_missing as f64
            / total_results.total_mbo_messages as f64
    );
    println!(
        "   │ Trade: Order at level missing   │ {:>10} │ {:>6.4}% │",
        total_results.trade_order_at_level_missing,
        100.0 * total_results.trade_order_at_level_missing as f64
            / total_results.total_mbo_messages as f64
    );
    println!("   ├─────────────────────────────────┼────────────┼──────────┤");
    println!(
        "   │ TOTAL WARNINGS                  │ {:>10} │ {:>6.4}% │",
        total_results.total_warnings(),
        total_results.warning_rate()
    );
    println!("   └─────────────────────────────────┴────────────┴──────────┘");

    println!("\n📈 BOOK STATE EVENTS:");
    println!("   Book clears:    {}", total_results.book_clears);
    println!("   No-op messages: {}", total_results.noop_messages);
    println!("   Crossed quotes: {}", total_results.crossed_quotes);
    println!("   Locked quotes:  {}", total_results.locked_quotes);
    println!("   Errors:         {}", total_results.errors);

    // Assertions for fair validation
    println!("\n✅ VALIDATION ASSERTIONS:");

    // Price accuracy should be > 95% within 1 cent (realistic for MBO reconstruction)
    // Note: 99% is achievable for single-day tests but multi-day has more variance
    let bid_1c = total_results.bid_price_1c_pct();
    let ask_1c = total_results.ask_price_1c_pct();
    println!(
        "   [{}] Bid price within 1¢ > 95%: {:.2}%",
        if bid_1c > 95.0 { "✓" } else { "✗" },
        bid_1c
    );
    println!(
        "   [{}] Ask price within 1¢ > 95%: {:.2}%",
        if ask_1c > 95.0 { "✓" } else { "✗" },
        ask_1c
    );

    // Size accuracy should be > 80% (size aggregation has inherent timing differences)
    let bid_sz = total_results.bid_size_exact_pct();
    let ask_sz = total_results.ask_size_exact_pct();
    println!(
        "   [{}] Bid size exact match > 80%: {:.2}%",
        if bid_sz > 80.0 { "✓" } else { "✗" },
        bid_sz
    );
    println!(
        "   [{}] Ask size exact match > 80%: {:.2}%",
        if ask_sz > 80.0 { "✓" } else { "✗" },
        ask_sz
    );

    // Warning rate explanation:
    // - Trade warnings are expected (aggressor side not in book)
    // - Cancel warnings for already-filled orders are normal
    // - Warning rate < 10% is acceptable for production
    let warn_rate = total_results.warning_rate();
    println!(
        "   [{}] Warning rate < 10%: {:.4}%",
        if warn_rate < 10.0 { "✓" } else { "✗" },
        warn_rate
    );
    println!("      (Note: Trade/cancel warnings for aggressor orders are expected)");

    // No hard errors
    println!(
        "   [{}] No hard errors: {}",
        if total_results.errors == 0 {
            "✓"
        } else {
            "✗"
        },
        total_results.errors
    );

    // Book clears should match number of trading days
    println!(
        "   [{}] Book clears = days processed: {} = {}",
        if total_results.book_clears == days_processed as u64 {
            "✓"
        } else {
            "~"
        },
        total_results.book_clears,
        days_processed
    );

    println!("\n================================================================================");
    println!("\n📝 INTERPRETATION:");
    println!("   - Price accuracy 95-99% is excellent for MBO reconstruction");
    println!("   - Size differences are due to MBO vs MBP-10 aggregation timing");
    println!("   - 'Order not found' warnings are normal (aggressor side trades)");
    println!("   - Book clears occur at market session transitions");
    println!("================================================================================");

    // Assert key metrics with realistic thresholds
    assert!(
        bid_1c > 95.0,
        "Bid price accuracy within 1¢ should be > 95%"
    );
    assert!(
        ask_1c > 95.0,
        "Ask price accuracy within 1¢ should be > 95%"
    );
    assert!(bid_sz > 80.0, "Bid size accuracy should be > 80%");
    assert!(ask_sz > 80.0, "Ask size accuracy should be > 80%");
    assert!(warn_rate < 10.0, "Warning rate should be < 10%");
    assert_eq!(total_results.errors, 0, "Should have no hard errors");
}

#[test]
#[cfg(feature = "databento")]
fn test_warning_stats_export() {
    use std::fs;

    println!("\n================================================================================");
    println!("WARNING STATS EXPORT TEST");
    println!("================================================================================\n");

    // Find a single day file to test
    let mbo_path = Path::new(MBO_DATA_DIR);
    if !mbo_path.exists() {
        println!("⚠️  MBO data not found - skipping test");
        return;
    }

    // Find first available file
    let mut test_file = None;
    for day in 1..=31 {
        let date_str = format!("202507{:02}", day);
        let mbo_file = format!("{}/xnas-itch-{}.mbo.dbn.zst", MBO_DATA_DIR, date_str);
        if Path::new(&mbo_file).exists() {
            test_file = Some(mbo_file);
            break;
        }
    }

    let test_file = match test_file {
        Some(f) => f,
        None => {
            println!("⚠️  No July files found - skipping test");
            return;
        }
    };

    println!("Processing: {}", test_file);

    let config = LobConfig::new(10)
        .with_logging(false)
        .with_validation(false);

    let mut lob = LobReconstructor::with_config(config);
    let loader = DbnLoader::new(&test_file).expect("Failed to create loader");

    let mut msg_count = 0;
    for msg in loader.iter_messages().expect("Failed to iterate") {
        let _ = lob.process_message(&msg);
        msg_count += 1;

        // Process first 1M messages for speed
        if msg_count >= 1_000_000 {
            break;
        }
    }

    let stats = lob.stats();

    println!("\n📊 Processing Stats:");
    println!("   Messages processed: {}", stats.messages_processed);
    println!("   Active orders:      {}", stats.active_orders);
    println!("   Bid levels:         {}", stats.bid_levels);
    println!("   Ask levels:         {}", stats.ask_levels);

    println!("\n⚠️  Warning Stats:");
    println!("   Has warnings: {}", stats.has_warnings());
    println!("   Total warnings: {}", stats.total_warnings());

    // Export to file
    let export_path = "/tmp/lob_stats_test.json";
    stats
        .export_to_file(export_path)
        .expect("Failed to export stats");

    // Verify file was created
    assert!(
        Path::new(export_path).exists(),
        "Stats file should be created"
    );

    // Read and verify JSON structure
    let content = fs::read_to_string(export_path).expect("Failed to read stats file");
    assert!(
        content.contains("\"messages_processed\""),
        "Should contain messages_processed"
    );
    assert!(
        content.contains("\"warnings\""),
        "Should contain warnings section"
    );
    assert!(
        content.contains("\"cancel_order_not_found\""),
        "Should contain warning details"
    );

    println!("\n✅ Stats exported to: {}", export_path);
    println!("   File size: {} bytes", content.len());

    // Cleanup
    fs::remove_file(export_path).ok();

    println!("\n✅ Warning stats export test PASSED!");
}
