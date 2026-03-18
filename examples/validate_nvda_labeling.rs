//! Validate Label Generation on Real NVDA Data
//!
//! This example:
//! 1. Loads NVDA MBO data from DBN file
//! 2. Reconstructs LOB state
//! 3. Collects mid-prices over time
//! 4. Generates TLOB labels with multiple configurations
//! 5. Validates label quality (no NaN, class balance, etc.)
//!
//! **Usage**:
//! ```bash
//! cargo run --release --example validate_nvda_labeling
//! ```
//!
//! **Note**: Requires the NVDA dataset to be present at the configured path.

use feature_extractor::{LabelConfig, LabelGenerator, TrendLabel};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor, Result};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔬 Testing TLOB Label Generation on NVDA Data");
    println!("═══════════════════════════════════════════════\n");

    // Configuration
    let data_path = "../data/hot_store/xnas-itch-20250203.mbo.dbn";
    let sample_interval = 100; // Sample every 100 messages

    println!("📂 Loading data from: {}", data_path);
    println!("📊 Sampling interval: every {} messages\n", sample_interval);

    // Initialize LOB reconstructor
    let loader = DbnLoader::new(data_path)?;
    let mut lob = LobReconstructor::new(10); // 10 price levels

    // Collect mid-prices
    println!("📈 Phase 1: Collecting mid-prices...");
    let start = Instant::now();

    let mut mid_prices = Vec::new();
    let mut msg_count = 0;
    let mut valid_snapshots = 0;

    for msg in loader.iter_messages()? {
        // Skip invalid messages
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            continue;
        }

        let lob_state = match lob.process_message(&msg) {
            Ok(state) => state,
            Err(_) => continue,
        };

        msg_count += 1;

        // Sample at intervals
        if msg_count % sample_interval == 0 {
            if let Some(mid_price) = lob_state.mid_price() {
                mid_prices.push(mid_price);
                valid_snapshots += 1;
            }
        }
    }

    let collection_time = start.elapsed();

    println!("✅ Collection complete!");
    println!("   - Total messages: {}", msg_count);
    println!("   - Valid snapshots: {}", valid_snapshots);
    println!("   - Mid-prices collected: {}", mid_prices.len());
    println!("   - Time: {:.2}s", collection_time.as_secs_f64());
    println!(
        "   - Throughput: {:.1}K msg/s\n",
        msg_count as f64 / collection_time.as_secs_f64() / 1000.0
    );

    if mid_prices.len() < 100 {
        println!("❌ Not enough mid-prices collected for labeling");
        return Ok(());
    }

    // Test multiple horizon configurations
    let test_configs = vec![
        (10, 5, 0.002, "h=10, k=5, θ=0.2%"),
        (20, 5, 0.002, "h=20, k=5, θ=0.2%"),
        (50, 10, 0.002, "h=50, k=10, θ=0.2%"),
        (100, 10, 0.002, "h=100, k=10, θ=0.2%"),
    ];

    println!("🏷️  Phase 2: Generating labels with multiple configurations...\n");

    for (horizon, smoothing_window, threshold, desc) in test_configs {
        println!("───────────────────────────────────────────────");
        println!("Configuration: {}", desc);
        println!("───────────────────────────────────────────────");

        let config = LabelConfig {
            horizon,
            smoothing_window,
            threshold,
        };

        let mut generator = LabelGenerator::new(config.clone());
        generator.add_prices(&mid_prices);

        // Generate labels
        let start = Instant::now();
        let labels = match generator.generate_labels() {
            Ok(labels) => labels,
            Err(e) => {
                println!("⚠️  Insufficient data: {}\n", e);
                continue;
            }
        };
        let label_time = start.elapsed();

        // Compute statistics
        let stats = generator.compute_stats(&labels);
        let (up_pct, stable_pct, down_pct) = stats.class_balance();

        println!("\n📊 Label Statistics:");
        println!("   Total labels:     {}", stats.total);
        println!(
            "   Generation time:  {:.3}ms",
            label_time.as_secs_f64() * 1000.0
        );
        println!(
            "   Throughput:       {:.1}K labels/s",
            stats.total as f64 / label_time.as_secs_f64() / 1000.0
        );

        println!("\n📈 Class Distribution:");
        println!("   Up:     {:6} ({:.1}%)", stats.up_count, up_pct * 100.0);
        println!(
            "   Stable: {:6} ({:.1}%)",
            stats.stable_count,
            stable_pct * 100.0
        );
        println!(
            "   Down:   {:6} ({:.1}%)",
            stats.down_count,
            down_pct * 100.0
        );

        let balanced = if stats.is_balanced() {
            "✅ Balanced"
        } else {
            "⚠️  Imbalanced"
        };
        println!("   Status: {}", balanced);

        println!("\n📉 Change Statistics:");
        println!(
            "   Mean:   {:.4}% ({:.1} bps)",
            stats.avg_change * 100.0,
            stats.avg_change * 10000.0
        );
        println!("   Std:    {:.4}%", stats.std_change * 100.0);
        println!("   Min:    {:.4}%", stats.min_change * 100.0);
        println!("   Max:    {:.4}%", stats.max_change * 100.0);

        // Validation checks
        println!("\n✅ Validation:");
        let mut nan_count = 0;
        let mut inf_count = 0;

        for (_, _, change) in &labels {
            if change.is_nan() {
                nan_count += 1;
            }
            if change.is_infinite() {
                inf_count += 1;
            }
        }

        println!(
            "   NaN values:      {} ({}%)",
            nan_count,
            nan_count as f64 / stats.total as f64 * 100.0
        );
        println!(
            "   Inf values:      {} ({}%)",
            inf_count,
            inf_count as f64 / stats.total as f64 * 100.0
        );

        // Show sample labels
        println!("\n📝 Sample Labels (first 5):");
        for (idx, label, change) in labels.iter().take(5) {
            let label_str = match label {
                TrendLabel::Up => "↑ Up  ",
                TrendLabel::Stable => "→ Stbl",
                TrendLabel::Down => "↓ Down",
            };
            println!(
                "   t={:5} | {} | change={:+.4}%",
                idx,
                label_str,
                change * 100.0
            );
        }

        println!();
    }

    println!("═══════════════════════════════════════════════");
    println!("✅ Label Generation Test Complete!");

    Ok(())
}
