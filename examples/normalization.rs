//! Simple example demonstrating different normalization strategies.
//!
//! This shows how to use each normalizer type and when to apply them.
//!
//! Usage:
//! ```bash
//! cargo run --example normalization_example
//! ```

use feature_extractor::preprocessing::{
    BilinearNormalizer, MinMaxNormalizer, Normalizer, PercentageChangeNormalizer, ZScoreNormalizer,
};

fn print_separator() {
    println!("{}", "=".repeat(70));
}

fn print_subseparator() {
    println!("{}", "-".repeat(70));
}

fn main() {
    println!("\n🔬 Feature Normalization Examples\n");
    print_separator();

    // =========================================================================
    // 1. Percentage Change Normalization (for Prices)
    // =========================================================================
    println!("\n1️⃣  Percentage Change Normalization");
    print_subseparator();
    println!("Use for: LOB prices (makes them comparable across stocks)\n");

    let mut price_norm = PercentageChangeNormalizer::new();
    price_norm.set_reference(100.0); // Mid-price = $100

    println!("  Reference (mid-price): ${:.2}", price_norm.reference());
    println!("\n  Price Normalization:");

    let prices = vec![99.0, 99.5, 100.0, 100.5, 101.0];
    for price in prices {
        let normalized = price_norm.normalize(price);
        println!(
            "    ${:.2} → {:.4} ({:.2}%)",
            price,
            normalized,
            normalized * 100.0
        );
    }

    println!("\n  💡 Insight: Normalizes to relative changes, removing absolute price level");

    // =========================================================================
    // 2. Z-Score Normalization (for Volumes, Spreads)
    // =========================================================================
    println!("\n\n2️⃣  Z-Score Normalization");
    print_subseparator();
    println!("Use for: Volumes, spreads, imbalances (statistical normalization)\n");

    let mut volume_norm = ZScoreNormalizer::new();

    // Simulate volume data
    let volumes = vec![100.0, 150.0, 200.0, 180.0, 220.0, 170.0, 190.0, 210.0];

    println!("  Training on volume data...");
    for &vol in &volumes {
        volume_norm.update(vol);
    }

    println!("  Mean: {:.2}", volume_norm.mean());
    println!("  Std:  {:.2}", volume_norm.std());
    println!("\n  Volume Normalization:");

    for &vol in &volumes[..5] {
        let normalized = volume_norm.normalize(vol);
        println!("    {vol:.0} shares → {normalized:.4} std deviations");
    }

    println!("\n  💡 Insight: Values in standard deviations from mean");

    // =========================================================================
    // 3. Z-Score with Windowed Statistics
    // =========================================================================
    println!("\n\n3️⃣  Windowed Z-Score (Adaptive)");
    print_subseparator();
    println!("Use for: Non-stationary data (recent statistics only)\n");

    let mut adaptive_norm = ZScoreNormalizer::with_window(5);

    // Simulate changing market conditions
    let changing_volumes = vec![
        100.0, 110.0, 105.0, 115.0, 120.0, // Period 1: low volume
        500.0, 520.0, 510.0, 530.0, 515.0, // Period 2: high volume
    ];

    println!("  Adapting to changing conditions:");
    for (i, &vol) in changing_volumes.iter().enumerate() {
        adaptive_norm.update(vol);

        if i == 4 || i == 9 {
            println!(
                "    After {} samples: mean={:.1}, std={:.1}",
                i + 1,
                adaptive_norm.mean(),
                adaptive_norm.std()
            );
        }
    }

    println!("\n  💡 Insight: Adapts to regime changes by using recent window only");

    // =========================================================================
    // 4. Bilinear Normalization (for LOB Structure)
    // =========================================================================
    println!("\n\n4️⃣  Bilinear Normalization");
    print_subseparator();
    println!("Use for: LOB price levels (preserves order book structure)\n");

    let mut lob_norm = BilinearNormalizer::new(0.01, 50.0);
    lob_norm.set_mid_price(100.0);

    println!("  Configuration:");
    println!("    Tick size: ${:.2}", 0.01);
    println!("    Scale factor: {}", 50.0);
    println!("    Mid-price: ${:.2}", lob_norm.mid_price());

    println!("\n  LOB Level Normalization:");

    let lob_prices = vec![
        ("Best Ask", 100.01),
        ("Ask L2", 100.02),
        ("Ask L3", 100.05),
        ("Best Bid", 99.99),
        ("Bid L2", 99.98),
        ("Bid L3", 99.95),
    ];

    for (level, price) in lob_prices {
        let normalized = lob_norm.normalize(price);
        let ticks = (price - lob_norm.mid_price()) / 0.01;
        println!("    {level} ${price:.2} ({ticks:+.0} ticks) → {normalized:.4}");
    }

    println!("\n  💡 Insight: Normalizes by distance from mid-price in tick units");

    // =========================================================================
    // 5. Min-Max Normalization
    // =========================================================================
    println!("\n\n5️⃣  Min-Max Normalization");
    print_subseparator();
    println!("Use for: Features with known bounds\n");

    // For [0, 1] range
    let spread_norm = MinMaxNormalizer::new(0.0, 0.10, false);
    println!("  Spread Normalization [0, 1]:");

    let spreads = vec![0.00, 0.01, 0.02, 0.05, 0.10];
    for spread in spreads {
        let normalized = spread_norm.normalize(spread);
        println!("    ${spread:.3} spread → {normalized:.2}");
    }

    // For [-1, 1] range (symmetric)
    let imbalance_norm = MinMaxNormalizer::new(-1.0, 1.0, true);
    println!("\n  Imbalance Normalization [-1, 1]:");

    let imbalances = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    for imb in imbalances {
        let normalized = imbalance_norm.normalize(imb);
        println!("    {imb:.2} imbalance → {normalized:.2}");
    }

    println!("\n  💡 Insight: Bounds values to known range");

    // =========================================================================
    // 6. Practical Recommendation
    // =========================================================================
    println!("\n\n6️⃣  TLOB Paper Recommendations");
    print_subseparator();
    println!("\n  Feature Type                Recommended Normalizer");
    print_subseparator();
    println!("  Bid/Ask Prices (L1-L10)     Percentage Change (vs mid-price)");
    println!("  Bid/Ask Volumes (L1-L10)    Z-Score (windowed)");
    println!("  Mid-Price                   Percentage Change (vs initial)");
    println!("  Spread                      Min-Max [0, 1] or Z-Score");
    println!("  Volume Imbalance            Z-Score or Min-Max [-1, 1]");
    println!("  Price Imbalance             Z-Score");
    println!("  Weighted Mid-Price          Percentage Change");
    println!("  MBO Features                Z-Score (windowed)");

    // =========================================================================
    // 7. Batch Processing
    // =========================================================================
    println!("\n\n7️⃣  Batch Processing");
    print_subseparator();

    let mut batch_norm = PercentageChangeNormalizer::new();
    batch_norm.set_reference(100.0);

    let batch_prices = vec![99.0, 100.0, 101.0, 102.0, 103.0];
    let normalized_batch = batch_norm.normalize_batch(&batch_prices);

    println!("  Original: {batch_prices:?}");
    println!(
        "  Normalized: {:?}",
        normalized_batch
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
    );

    println!("\n  💡 Insight: Batch operations for efficient processing");

    println!();
    print_separator();
    println!("\n✅ Normalization examples complete!");
    println!("\n💡 Key Takeaways:");
    println!("   • Different features need different normalization strategies");
    println!("   • Percentage change removes absolute price levels (critical for ML)");
    println!("   • Z-score handles varying scales with statistical properties");
    println!("   • Windowed statistics adapt to non-stationary markets");
    println!("   • All normalizers are trait-based for easy extension\n");
}
