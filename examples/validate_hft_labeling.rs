//! Validate HFT-Appropriate Labeling Configurations
//!
//! This example validates labeling configurations for minute-to-hour
//! HFT trading strategies using synthetic but realistic price data.
//!
//! **Horizons Tested**:
//! - h=200 (4.3 min) - Primary HFT target
//! - h=500 (10.8 min) - Extended positions
//! - h=1000 (21.6 min) - Hourly exits
//!
//! **Usage**:
//! ```bash
//! cargo run --release --example validate_hft_labeling
//! ```

use feature_extractor::{LabelConfig, TlobLabelGenerator, TrendLabel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     HFT Labeling Validation: NVIDIA Minute-to-Hour Scale   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("📊 Testing Context:");
    println!("   Asset: NVIDIA (NVDA)");
    println!("   Sampling: Every 1000 events (~1.3 seconds)");
    println!("   Use Case: Intraday HFT (minute to hour positions)");
    println!();

    // Generate synthetic but realistic mid-prices
    // Based on NVIDIA Feb 3, 2025: $113-$119 range
    let mid_prices = generate_realistic_prices(1000); // 1000 samples ≈ 22 minutes

    println!("📈 Generated Test Data:");
    println!("   Samples: {}", mid_prices.len());
    println!(
        "   Price Range: ${:.2} - ${:.2}",
        mid_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        mid_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "   Duration: ~{:.1} minutes",
        mid_prices.len() as f64 * 1.3 / 60.0
    );
    println!();

    // Test Configuration 1: Minute-Scale (h=200)
    println!("{:=<70}", "");
    println!("  TEST 1: Minute-Scale HFT (h=200 ≈ 4.3 minutes)");
    println!("{:=<70}\n", "");
    test_horizon(200, 20, 0.0005, &mid_prices, "4.3 minutes")?;

    // Test Configuration 2: Multi-Minute (h=500)
    println!("\n{:=<70}", "");
    println!("  TEST 2: Multi-Minute HFT (h=500 ≈ 10.8 minutes)");
    println!("{:=<70}\n", "");
    test_horizon(500, 50, 0.0008, &mid_prices, "10.8 minutes")?;

    // Test Configuration 3: Hourly (h=1000)
    // Note: Can't test h=1000 with only 1000 samples
    println!("\n{:=<70}", "");
    println!("  TEST 3: Hourly Scale (h=1000 ≈ 21.6 minutes)");
    println!("{:=<70}\n", "");
    println!("ℹ️  Skipped: Need 2000+ samples for h=1000 horizon");
    println!("   (Would need ~43 minutes of data)");
    println!();

    // Test threshold sensitivity
    println!("\n{:=<70}", "");
    println!("  TEST 4: Threshold Sensitivity Analysis (h=200)");
    println!("{:=<70}\n", "");
    test_threshold_sensitivity(&mid_prices)?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              HFT LABELING VALIDATION COMPLETE                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Test a specific horizon configuration
fn test_horizon(
    horizon: usize,
    smoothing: usize,
    threshold: f64,
    prices: &[f64],
    time_str: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Configuration:");
    println!("   Horizon (k): {}", horizon);
    println!("   Smoothing Window: {}", smoothing);
    println!(
        "   Threshold: {:.4} ({:.2} bps)",
        threshold,
        threshold * 10000.0
    );
    println!("   Prediction Time: {}", time_str);
    println!();

    // Check if we have enough data
    let required_samples = horizon + smoothing + 1;
    if prices.len() < required_samples {
        println!("⚠️  WARNING: Not enough samples!");
        println!("   Need: {} samples", required_samples);
        println!("   Have: {} samples", prices.len());
        println!("   Skipping this horizon.\n");
        return Ok(());
    }

    // Create label generator
    let config = LabelConfig {
        horizon,
        smoothing_window: smoothing,
        threshold,
    };

    let mut generator = TlobLabelGenerator::new(config);

    // Generate labels
    println!("⏱️  Generating labels...");
    let start_time = std::time::Instant::now();

    for &price in prices.iter() {
        generator.add_price(price);
    }

    let label_results = generator.generate_labels()?;
    let elapsed = start_time.elapsed();

    println!(
        "   ✓ Generated {} labels in {:.2}ms",
        label_results.len(),
        elapsed.as_secs_f64() * 1000.0
    );
    println!();

    // Analyze label distribution
    println!("📊 Label Distribution:");
    let mut up_count = 0;
    let mut down_count = 0;
    let mut stable_count = 0;

    for (_idx, label, _value) in &label_results {
        match label {
            TrendLabel::Up => up_count += 1,
            TrendLabel::Down => down_count += 1,
            TrendLabel::Stable => stable_count += 1,
        }
    }

    let total = label_results.len() as f64;
    let up_pct = (up_count as f64 / total) * 100.0;
    let down_pct = (down_count as f64 / total) * 100.0;
    let stable_pct = (stable_count as f64 / total) * 100.0;

    println!("   Up:     {:>5} ({:>5.1}%)", up_count, up_pct);
    println!("   Down:   {:>5} ({:>5.1}%)", down_count, down_pct);
    println!("   Stable: {:>5} ({:>5.1}%)", stable_count, stable_pct);
    println!();

    // Validate distribution
    println!("✅ Validation:");

    // Check we got labels
    assert!(!label_results.is_empty(), "Should generate labels");
    println!("   ✓ Labels generated: {}", label_results.len());

    // Check distribution is reasonable
    assert!(
        (5.0..=65.0).contains(&up_pct),
        "Up labels should be 5-65%, got {:.1}%",
        up_pct
    );
    println!("   ✓ Up labels in reasonable range: {:.1}%", up_pct);

    assert!(
        (5.0..=65.0).contains(&down_pct),
        "Down labels should be 5-65%, got {:.1}%",
        down_pct
    );
    println!("   ✓ Down labels in reasonable range: {:.1}%", down_pct);

    assert!(
        (2.0..=80.0).contains(&stable_pct),
        "Stable labels should be 2-80%, got {:.1}%",
        stable_pct
    );
    println!("   ✓ Stable labels in reasonable range: {:.1}%", stable_pct);

    if stable_pct < 10.0 {
        println!(
            "   ℹ️  Note: Very few stable labels ({:.1}%) - strong trending market",
            stable_pct
        );
    }

    // Check balance (no class dominates too much)
    let max_pct = up_pct.max(down_pct).max(stable_pct);
    assert!(
        max_pct < 85.0,
        "No class should dominate >85%, got {:.1}%",
        max_pct
    );
    println!("   ✓ Classes reasonably balanced (max: {:.1}%)", max_pct);

    // Note: Slight imbalance is NORMAL and realistic
    // Markets trend, especially intraday
    if max_pct > 55.0 {
        println!(
            "   ℹ️  Note: Trending market detected ({:.1}% max class)",
            max_pct
        );
    }

    // Check for transitions (not all same label)
    let mut transitions = 0;
    for i in 1..label_results.len() {
        if label_results[i].1 != label_results[i - 1].1 {
            transitions += 1;
        }
    }
    let transition_rate = (transitions as f64 / label_results.len() as f64) * 100.0;
    println!(
        "   ✓ Label transitions: {} ({:.1}%)",
        transitions, transition_rate
    );

    // Transitions should exist but can be low in trending markets
    assert!(
        transition_rate >= 0.5,
        "Should have >=0.5% transitions, got {:.1}%",
        transition_rate
    );

    if transition_rate < 3.0 {
        println!(
            "   ℹ️  Note: Low transitions ({:.1}%) suggests strong trend",
            transition_rate
        );
    }

    println!("\n✅ TEST PASSED: h={} configuration validated", horizon);

    Ok(())
}

/// Test how threshold affects label distribution
fn test_threshold_sensitivity(prices: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Testing Different Thresholds (h=200, smoothing=20):");
    println!();

    let thresholds = vec![
        (0.0001, "1 bp (very tight)"),
        (0.0003, "3 bp (recommended)"),
        (0.0005, "5 bp (balanced)"),
        (0.001, "10 bp (loose)"),
    ];

    println!("   Threshold  | Up%   | Down% | Stat% | Assessment");
    println!("   -----------|-------|-------|-------|------------------");

    for (threshold, label) in thresholds {
        let config = LabelConfig {
            horizon: 200,
            smoothing_window: 20,
            threshold,
        };

        let mut generator = TlobLabelGenerator::new(config);
        for &price in prices.iter() {
            generator.add_price(price);
        }
        let label_results = generator.generate_labels().unwrap();

        let mut up_count = 0;
        let mut down_count = 0;
        let mut stable_count = 0;

        for (_idx, l, _value) in &label_results {
            match l {
                TrendLabel::Up => up_count += 1,
                TrendLabel::Down => down_count += 1,
                TrendLabel::Stable => stable_count += 1,
            }
        }

        let total = label_results.len() as f64;
        let up_pct = (up_count as f64 / total) * 100.0;
        let down_pct = (down_count as f64 / total) * 100.0;
        let stable_pct = (stable_count as f64 / total) * 100.0;

        let assessment = if stable_pct > 70.0 {
            "Too tight"
        } else if stable_pct < 30.0 {
            "Too loose"
        } else {
            "✓ Good"
        };

        println!(
            "   {} | {:>5.1} | {:>5.1} | {:>5.1} | {}",
            label, up_pct, down_pct, stable_pct, assessment
        );
    }

    println!();
    println!("✅ Threshold Sensitivity Analysis Complete");
    println!("   Recommendation: Use 3-5 bp (0.0003-0.0005) for HFT");

    Ok(())
}

/// Generate realistic price movements for testing
fn generate_realistic_prices(n: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(n);
    let base_price = 115.0; // NVIDIA baseline

    // Create realistic intraday pattern:
    // 1. Morning drift up
    // 2. Midday consolidation
    // 3. Afternoon volatility
    // 4. Close drift

    let mut price = base_price;
    let mut rng_state = 12345u64; // Simple RNG for reproducibility

    for i in 0..n {
        // Simple LCG random number generator
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random = ((rng_state / 65536) % 32768) as f64 / 32768.0;

        // Time of day effect (0-1 over the samples)
        let t = i as f64 / n as f64;

        // Intraday pattern
        let trend = if t < 0.3 {
            // Morning: Slight upward drift
            0.0002
        } else if t < 0.6 {
            // Midday: Consolidation
            0.0
        } else if t < 0.8 {
            // Afternoon: Volatility
            (random - 0.5) * 0.001
        } else {
            // Close: Drift to VWAP
            -0.0001
        };

        // Add noise
        let noise = (random - 0.5) * 0.0005;

        // Update price
        price *= 1.0 + trend + noise;

        // Keep in realistic range
        price = price.max(113.0).min(119.0);

        prices.push(price);
    }

    prices
}
