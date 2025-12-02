//! Simple example showing how to use volume-based sampling in production.
//!
//! This demonstrates the recommended integration pattern for TLOB training pipeline.

use feature_extractor::{FeatureExtractor, VolumeBasedSampler};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <dbn_file>", args[0]);
        std::process::exit(1);
    }

    // Initialize components
    let loader = DbnLoader::new(&args[1])?.skip_invalid(true);
    let mut lob = LobReconstructor::new(10);
    let extractor = FeatureExtractor::new(10);

    // Volume-based sampler: 500 shares, 1ms minimum interval
    // Recommended by TLOB paper for liquid stocks like NVDA
    let mut sampler = VolumeBasedSampler::new(500, 1);

    println!("Processing with volume-based sampling (500 shares per sample)...\n");

    let mut samples = Vec::new();
    let mut msg_count = 0u64;

    for msg in loader.iter_messages()? {
        // Skip invalid
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            continue;
        }

        // Update LOB
        if lob.process_message(&msg).is_err() {
            continue;
        }

        msg_count += 1;

        // Sample based on volume
        let timestamp = msg.timestamp.unwrap_or(0) as u64;
        if sampler.should_sample(msg.size, timestamp) {
            let state = lob.get_lob_state();

            if state.is_valid() {
                // Extract features
                if let Ok(features) = extractor.extract_lob_features(&state) {
                    samples.push((timestamp, features));

                    // Show progress
                    if samples.len() % 10000 == 0 {
                        println!("Samples: {} (from {} messages)", samples.len(), msg_count);
                    }
                }
            }
        }

        // Limit for demo
        if msg_count >= 1_000_000 {
            break;
        }
    }

    // Statistics
    let (_sample_count, total_volume, avg_volume) = sampler.statistics();

    println!("\n✅ Complete!");
    println!("  Messages processed: {msg_count}");
    println!("  Samples generated:  {}", samples.len());
    println!("  Total volume:       {total_volume} shares");
    println!("  Avg volume/sample:  {avg_volume:.0} shares");
    println!(
        "  Sampling rate:      {:.2}%",
        (samples.len() as f64 / msg_count as f64) * 100.0
    );

    // Show first sample
    if let Some((ts, features)) = samples.first() {
        println!("\n📊 First sample features:");
        println!("  Timestamp:   {ts}");
        println!("  Mid-price:   ${:.4}", features[40]);
        println!("  Spread:      ${:.4}", features[41]);
        println!("  Bid volume:  {:.0}", features[43]);
        println!("  Ask volume:  {:.0}", features[44]);
    }

    Ok(())
}
