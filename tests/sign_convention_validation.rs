//! Sign Convention Validation for MBO Features (Schema v2.1)
//!
//! This test validates that `net_trade_flow` (index 56) and `net_cancel_flow`
//! (index 55) correctly follow the standard sign convention per RULE.md §9.
//!
//! ## Research Reference
//!
//! Per RULE.md §9 - Sign Conventions:
//! - `> 0` = Bullish / Buy pressure
//! - `< 0` = Bearish / Sell pressure
//! - `= 0` = Neutral / No signal
//!
//! ## MBO Data Semantics
//!
//! In MBO data, when a Trade message has `Side::Bid`:
//! - It means the **bid side was HIT**
//! - Which means a **seller aggressed** (sold into the bid)
//! - This is a SELL-initiated trade
//!
//! ## Correct Formula (Schema v2.1)
//!
//! ```
//! net_trade_flow = (trade_count_ask - trade_count_bid) / total
//! ```
//!
//! Where:
//! - `trade_count_ask` = buy-initiated trades (ask was hit)
//! - `trade_count_bid` = sell-initiated trades (bid was hit)
//!
//! So `net_trade_flow > 0` means MORE BUYS = BULLISH ✓
//!
//! ## Expected Results
//!
//! - `net_trade_flow` and `trade_asymmetry` should be POSITIVELY correlated (~1.0)
//! - `net_cancel_flow` and `cancel_asymmetry` should be POSITIVELY correlated (~1.0)
//! - Formula reconstruction should match exactly
//!
//! Run with:
//! ```bash
//! cargo test --features "parallel" --test sign_convention_validation --release -- --ignored --nocapture
//! ```

#![cfg(feature = "parallel")]

use feature_extractor::batch::{BatchConfig, BatchProcessor};
use feature_extractor::builder::PipelineBuilder;
use std::path::Path;

// Feature indices (from 03-FEATURE-INDEX-MAP-v2.md)
mod mbo_indices {
    pub const NET_ORDER_FLOW: usize = 54;
    pub const NET_CANCEL_FLOW: usize = 55;
    pub const NET_TRADE_FLOW: usize = 56;
}

mod signal_indices {
    // Trading signals start at index 84
    pub const TRADE_ASYMMETRY: usize = 88; // (ask - bid) / total
    pub const CANCEL_ASYMMETRY: usize = 89; // (ask - bid) / total
}

const HOT_STORE_DIR: &str = "../data/hot_store";

fn test_data_available() -> bool {
    Path::new(HOT_STORE_DIR).exists()
}

fn get_test_files(limit: usize) -> Vec<String> {
    let path = Path::new(HOT_STORE_DIR);
    if !path.exists() {
        return vec![];
    }

    let mut files: Vec<_> = std::fs::read_dir(path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "zst" || ext == "dbn")
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    files.sort();
    files.truncate(limit);
    files
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Arrays must have same length");
    let n = x.len() as f64;
    if n < 3.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute basic statistics
fn compute_stats(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max, mean, std_dev)
}

/// Data structure for a single observation
#[derive(Debug, Clone)]
struct Observation {
    // MBO Features (potentially inverted)
    net_order_flow: f64,
    net_cancel_flow: f64,
    net_trade_flow: f64,

    // Trading Signals (correctly signed)
    trade_asymmetry: f64,
    cancel_asymmetry: f64,

    // For price direction calculation
    mid_price: f64,
}

/// Extract observations from pipeline output
fn extract_observations(
    output: &feature_extractor::batch::BatchOutput,
) -> Vec<Observation> {
    let mut observations = Vec::new();

    for result in output.results.iter() {
        // Get mid-prices for this day
        let mid_prices = &result.output.mid_prices;

        // Each sequence's LAST feature vector corresponds to a mid-price sample
        for (seq_idx, seq) in result.output.sequences.iter().enumerate() {
            // The sequence's ending snapshot corresponds to:
            // ending_idx = seq_idx * stride + window_size - 1
            // But we need to map back to mid_price index
            let stride = result.output.stride;
            let window_size = result.output.window_size;
            let mid_price_idx = seq_idx * stride + window_size - 1;

            if mid_price_idx >= mid_prices.len() {
                continue;
            }

            // Get the last feature vector in the sequence (most recent)
            if let Some(feat_vec) = seq.features.last() {
                if feat_vec.len() >= 98 {
                    // Derived features: mid_price is at index 40
                    let mid_price = feat_vec[40];

                    observations.push(Observation {
                        net_order_flow: feat_vec[mbo_indices::NET_ORDER_FLOW],
                        net_cancel_flow: feat_vec[mbo_indices::NET_CANCEL_FLOW],
                        net_trade_flow: feat_vec[mbo_indices::NET_TRADE_FLOW],
                        trade_asymmetry: feat_vec[signal_indices::TRADE_ASYMMETRY],
                        cancel_asymmetry: feat_vec[signal_indices::CANCEL_ASYMMETRY],
                        mid_price,
                    });
                }
            }
        }
    }

    observations
}

/// Compute forward returns (price change as fraction)
fn compute_forward_returns(observations: &[Observation], horizon: usize) -> Vec<(Observation, f64)> {
    let mut result = Vec::new();

    for i in 0..(observations.len().saturating_sub(horizon)) {
        let current = &observations[i];
        let future = &observations[i + horizon];

        if current.mid_price > 1e-10 && future.mid_price > 1e-10 {
            let ret = (future.mid_price - current.mid_price) / current.mid_price;
            result.push((current.clone(), ret));
        }
    }

    result
}

#[test]
#[ignore = "Requires hot store data"]
fn test_net_trade_flow_sign_convention() {
    if !test_data_available() {
        eprintln!("Skipping: Hot store not available at {}", HOT_STORE_DIR);
        return;
    }

    let files = get_test_files(3); // Use 3 days for statistical power
    if files.is_empty() {
        eprintln!("Skipping: No data files found");
        return;
    }

    println!("\n{}", "=".repeat(80));
    println!("SIGN CONVENTION VALIDATION TEST");
    println!("{}", "=".repeat(80));
    println!("\nProcessing {} files...", files.len());

    // Build pipeline with full 98-feature set
    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()
        .with_mbo_features()
        .with_trading_signals()
        .event_sampling(1000)
        .window(100, 10)
        .build_config()
        .expect("Failed to build config");

    let batch_config = BatchConfig {
        num_threads: Some(4),
        error_mode: feature_extractor::batch::ErrorMode::CollectErrors,
        report_progress: false,
        stack_size: None,
        hot_store_dir: Some(HOT_STORE_DIR.into()),
    };

    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let output = processor
        .process_files(&files)
        .expect("Failed to process files");

    println!("Processed {} files successfully", output.successful_count());

    // Extract observations
    let observations = extract_observations(&output);
    println!("Extracted {} observations", observations.len());

    if observations.len() < 100 {
        panic!("Not enough observations for statistical analysis");
    }

    // Compute forward returns for horizon = 10
    let horizon = 10;
    let obs_with_returns = compute_forward_returns(&observations, horizon);
    println!(
        "Computed {} observations with forward returns (horizon={})",
        obs_with_returns.len(),
        horizon
    );

    // Extract signals and returns for correlation analysis
    let net_order_flow: Vec<f64> = obs_with_returns.iter().map(|(o, _)| o.net_order_flow).collect();
    let net_cancel_flow: Vec<f64> = obs_with_returns.iter().map(|(o, _)| o.net_cancel_flow).collect();
    let net_trade_flow: Vec<f64> = obs_with_returns.iter().map(|(o, _)| o.net_trade_flow).collect();
    let trade_asymmetry: Vec<f64> = obs_with_returns.iter().map(|(o, _)| o.trade_asymmetry).collect();
    let cancel_asymmetry: Vec<f64> = obs_with_returns.iter().map(|(o, _)| o.cancel_asymmetry).collect();
    let returns: Vec<f64> = obs_with_returns.iter().map(|(_, r)| *r).collect();

    // Filter out NaN/Inf values
    let valid_mask: Vec<bool> = (0..returns.len())
        .map(|i| {
            returns[i].is_finite()
                && net_order_flow[i].is_finite()
                && net_cancel_flow[i].is_finite()
                && net_trade_flow[i].is_finite()
                && trade_asymmetry[i].is_finite()
                && cancel_asymmetry[i].is_finite()
        })
        .collect();

    let filter = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .zip(valid_mask.iter())
            .filter_map(|(&val, &valid)| if valid { Some(val) } else { None })
            .collect()
    };

    let net_order_flow = filter(&net_order_flow);
    let net_cancel_flow = filter(&net_cancel_flow);
    let net_trade_flow = filter(&net_trade_flow);
    let trade_asymmetry = filter(&trade_asymmetry);
    let cancel_asymmetry = filter(&cancel_asymmetry);
    let returns = filter(&returns);

    println!("\nAfter filtering: {} valid observations", returns.len());

    // Compute correlations with returns
    let corr_net_order_flow = pearson_correlation(&net_order_flow, &returns);
    let corr_net_cancel_flow = pearson_correlation(&net_cancel_flow, &returns);
    let corr_net_trade_flow = pearson_correlation(&net_trade_flow, &returns);
    let corr_trade_asymmetry = pearson_correlation(&trade_asymmetry, &returns);
    let corr_cancel_asymmetry = pearson_correlation(&cancel_asymmetry, &returns);

    // Print results
    println!("\n{}", "=".repeat(80));
    println!("CORRELATION WITH FORWARD RETURNS (horizon={})", horizon);
    println!("{}", "=".repeat(80));
    println!("\nMBO Features (indices 54-56):");
    println!("  {:20} : {:+.6} {}", 
        "net_order_flow (54)", 
        corr_net_order_flow,
        sign_assessment(corr_net_order_flow, true)
    );
    println!("  {:20} : {:+.6} {}", 
        "net_cancel_flow (55)", 
        corr_net_cancel_flow,
        sign_assessment(corr_net_cancel_flow, true)
    );
    println!("  {:20} : {:+.6} {}",
        "net_trade_flow (56)",
        corr_net_trade_flow,
        sign_assessment(corr_net_trade_flow, true)
    );

    println!("\nTrading Signals (indices 88-89):");
    println!("  {:20} : {:+.6} {}",
        "trade_asymmetry (88)",
        corr_trade_asymmetry,
        sign_assessment(corr_trade_asymmetry, true)
    );
    println!("  {:20} : {:+.6} {}",
        "cancel_asymmetry (89)",
        corr_cancel_asymmetry,
        sign_assessment(corr_cancel_asymmetry, true)
    );

    // Cross-signal correlations to verify they measure similar things
    println!("\n{}", "=".repeat(80));
    println!("CROSS-SIGNAL CORRELATIONS");
    println!("{}", "=".repeat(80));
    
    let corr_trade_vs_asymmetry = pearson_correlation(&net_trade_flow, &trade_asymmetry);
    let corr_cancel_vs_asymmetry = pearson_correlation(&net_cancel_flow, &cancel_asymmetry);
    
    println!("\n  net_trade_flow vs trade_asymmetry:   {:+.6}", corr_trade_vs_asymmetry);
    println!("  net_cancel_flow vs cancel_asymmetry: {:+.6}", corr_cancel_vs_asymmetry);

    println!("\n{}", "=".repeat(80));
    println!("SIGN CONVENTION ANALYSIS");
    println!("{}", "=".repeat(80));

    // The key test: if net_trade_flow is inverted, it should be NEGATIVELY correlated
    // with trade_asymmetry (which is known to be correct)
    println!("\nIf net_trade_flow uses (bid - ask) and trade_asymmetry uses (ask - bid),");
    println!("they should be NEGATIVELY correlated (r ≈ -1.0).");
    println!("");
    println!("Observed correlation: {:+.6}", corr_trade_vs_asymmetry);

    if corr_trade_vs_asymmetry < -0.9 {
        println!("  ✓ CONFIRMED: net_trade_flow and trade_asymmetry are negatively correlated");
        println!("    This confirms net_trade_flow has INVERTED sign convention.");
    } else if corr_trade_vs_asymmetry < -0.5 {
        println!("  ⚠ LIKELY INVERTED: Strong negative correlation observed");
    } else {
        println!("  ? UNEXPECTED: Correlation is not strongly negative");
    }

    println!("\nSimilarly for cancel signals:");
    println!("Observed correlation (net_cancel_flow vs cancel_asymmetry): {:+.6}", corr_cancel_vs_asymmetry);

    if corr_cancel_vs_asymmetry < -0.9 {
        println!("  ✓ CONFIRMED: net_cancel_flow also has INVERTED sign convention");
    } else if corr_cancel_vs_asymmetry < -0.5 {
        println!("  ⚠ LIKELY INVERTED: Strong negative correlation observed");
    } else {
        println!("  ? UNEXPECTED or DIFFERENT: Correlation is not strongly negative");
    }

    // Print statistics
    println!("\n{}", "=".repeat(80));
    println!("SIGNAL STATISTICS");
    println!("{}", "=".repeat(80));
    
    let print_stats = |name: &str, vals: &[f64]| {
        let (min, max, mean, std) = compute_stats(vals);
        println!("  {:20} : min={:+.4}, max={:+.4}, mean={:+.4}, std={:.4}",
            name, min, max, mean, std);
    };

    print_stats("net_order_flow", &net_order_flow);
    print_stats("net_cancel_flow", &net_cancel_flow);
    print_stats("net_trade_flow", &net_trade_flow);
    print_stats("trade_asymmetry", &trade_asymmetry);
    print_stats("cancel_asymmetry", &cancel_asymmetry);
    print_stats("forward_returns", &returns);

    // Assertions
    println!("\n{}", "=".repeat(80));
    println!("ASSERTIONS (Post-Fix Validation)");
    println!("{}", "=".repeat(80));

    // After v2.1 fix: net_trade_flow uses (ask - bid), same as trade_asymmetry
    // They should now be POSITIVELY correlated (nearly identical)
    assert!(
        corr_trade_vs_asymmetry > 0.95,
        "ASSERTION FAILED: After v2.1 fix, net_trade_flow and trade_asymmetry should be \
         POSITIVELY correlated (expected r > 0.95, got r = {:.4}). \
         Both now use (ask - bid) / total formula.",
        corr_trade_vs_asymmetry
    );
    println!("✓ net_trade_flow vs trade_asymmetry: r = {:+.4} > 0.95 (SIGN CONVENTION CORRECT)", 
        corr_trade_vs_asymmetry);

    // After v2.1 fix: net_cancel_flow uses (ask - bid), same as cancel_asymmetry
    assert!(
        corr_cancel_vs_asymmetry > 0.95,
        "ASSERTION FAILED: After v2.1 fix, net_cancel_flow and cancel_asymmetry should be \
         POSITIVELY correlated (expected r > 0.95, got r = {:.4}). \
         Both now use (ask - bid) / total formula.",
        corr_cancel_vs_asymmetry
    );
    println!("✓ net_cancel_flow vs cancel_asymmetry: r = {:+.4} > 0.95 (SIGN CONVENTION CORRECT)",
        corr_cancel_vs_asymmetry);

    println!("\n{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("\nSign convention for net_trade_flow (56) and net_cancel_flow (55) is CORRECT.");
    println!("Both now use (ask - bid) / total formula, following RULE.md §9:");
    println!("  > 0 = BULLISH (more buyer activity)");
    println!("  < 0 = BEARISH (more seller activity)");
    println!("\nSchema version: 2.1");
}

fn sign_assessment(correlation: f64, expect_positive: bool) -> &'static str {
    if expect_positive {
        if correlation > 0.01 {
            "(EXPECTED: +correlation with returns)"
        } else if correlation < -0.01 {
            "(⚠ UNEXPECTED: -correlation, may be inverted)"
        } else {
            "(~0 correlation)"
        }
    } else {
        if correlation < -0.01 {
            "(EXPECTED: -correlation)"
        } else if correlation > 0.01 {
            "(⚠ UNEXPECTED: +correlation)"
        } else {
            "(~0 correlation)"
        }
    }
}

/// Additional test: verify the formulas directly by examining the raw counts
#[test]
#[ignore = "Requires hot store data"]
fn test_verify_formula_semantics() {
    if !test_data_available() {
        eprintln!("Skipping: Hot store not available at {}", HOT_STORE_DIR);
        return;
    }

    let files = get_test_files(1);
    if files.is_empty() {
        eprintln!("Skipping: No data files found");
        return;
    }

    println!("\n{}", "=".repeat(80));
    println!("FORMULA SEMANTICS VERIFICATION");
    println!("{}", "=".repeat(80));

    // Build pipeline with full features
    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .with_derived_features()
        .with_mbo_features()
        .with_trading_signals()
        .event_sampling(1000)
        .window(100, 10)
        .build_config()
        .expect("Failed to build config");

    let batch_config = BatchConfig {
        num_threads: Some(4),
        error_mode: feature_extractor::batch::ErrorMode::CollectErrors,
        report_progress: false,
        stack_size: None,
        hot_store_dir: Some(HOT_STORE_DIR.into()),
    };

    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let output = processor
        .process_files(&files)
        .expect("Failed to process files");

    // Trade rates are at indices 52 (bid) and 53 (ask)
    const TRADE_RATE_BID: usize = 52;
    const TRADE_RATE_ASK: usize = 53;

    // Verify formula by reconstruction (Post v2.1 fix)
    let mut formula_matches = 0;
    let mut formula_mismatches = 0;
    let mut samples_checked = 0;

    for result in output.results.iter() {
        for seq in result.output.sequences.iter() {
            for feat_vec in seq.features.iter() {
                if feat_vec.len() >= 98 {
                    let trade_rate_bid = feat_vec[TRADE_RATE_BID];
                    let trade_rate_ask = feat_vec[TRADE_RATE_ASK];
                    let net_trade_flow = feat_vec[mbo_indices::NET_TRADE_FLOW];
                    let trade_asymmetry = feat_vec[signal_indices::TRADE_ASYMMETRY];

                    let total = trade_rate_bid + trade_rate_ask;
                    if total > 1e-8 {
                        // After v2.1 fix: net_trade_flow uses (ask - bid) / total
                        let expected_net_trade_flow = (trade_rate_ask - trade_rate_bid) / total;

                        let tol = 1e-6;
                        if (net_trade_flow - expected_net_trade_flow).abs() < tol {
                            formula_matches += 1;
                        } else {
                            formula_mismatches += 1;
                            if formula_mismatches <= 3 {
                                println!("Formula mismatch for net_trade_flow:");
                                println!("  trade_rate_bid: {:.6}", trade_rate_bid);
                                println!("  trade_rate_ask: {:.6}", trade_rate_ask);
                                println!("  Expected (ask-bid)/total: {:.6}", expected_net_trade_flow);
                                println!("  Actual net_trade_flow: {:.6}", net_trade_flow);
                            }
                        }

                        // After v2.1 fix: net_trade_flow and trade_asymmetry should be EQUAL
                        if (net_trade_flow - trade_asymmetry).abs() < tol {
                            // Correct: they are equal
                        } else if samples_checked < 3 {
                            println!("\nSample {}:", samples_checked + 1);
                            println!("  trade_rate_bid (sells hitting bid): {:.4}", trade_rate_bid);
                            println!("  trade_rate_ask (buys hitting ask):  {:.4}", trade_rate_ask);
                            println!("  net_trade_flow  = (ask-bid)/total = {:+.4}", net_trade_flow);
                            println!("  trade_asymmetry = (ask-bid)/total = {:+.4}", trade_asymmetry);
                            println!("  Difference (should be 0): {:+.6}", net_trade_flow - trade_asymmetry);
                        }

                        samples_checked += 1;
                    }
                }
            }
        }
    }

    println!("\nFormula verification:");
    println!("  Formula matches:    {}", formula_matches);
    println!("  Formula mismatches: {}", formula_mismatches);
    println!("  Total samples:      {}", samples_checked);

    // The formulas should match our expectation
    assert!(
        formula_matches > formula_mismatches,
        "Formula verification failed: more mismatches ({}) than matches ({})",
        formula_mismatches, formula_matches
    );

    println!("\n{}", "=".repeat(80));
    println!("SEMANTIC ANALYSIS (Post v2.1 Fix)");
    println!("{}", "=".repeat(80));
    println!("\nFrom the MBO reconstructor documentation (trade_aggregator.rs):");
    println!("  - Trade with Side::Bid means the BID was HIT → SELLER aggressed");
    println!("  - Trade with Side::Ask means the ASK was HIT → BUYER aggressed");
    println!("\nTherefore:");
    println!("  - trade_rate_bid = rate of SELL-initiated trades");
    println!("  - trade_rate_ask = rate of BUY-initiated trades");
    println!("\nAfter v2.1 fix, net_trade_flow = (trade_rate_ask - trade_rate_bid) / total");
    println!("  → > 0 when more BUYS  → BULLISH pressure ✓");
    println!("  → < 0 when more SELLS → BEARISH pressure ✓");
    println!("\nThis correctly follows RULE.md §9:");
    println!("  → > 0 = BULLISH / Buy pressure");
    println!("  → < 0 = BEARISH / Sell pressure");
    println!("\n✓ Sign convention is now CORRECT.");
}

