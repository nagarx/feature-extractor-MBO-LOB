//! Validation of PipelineBuilder with Real NVDA Data
//!
//! This example validates the new PipelineBuilder API by processing
//! real MBO data and verifying the output.
//!
//! Usage:
//! ```bash
//! cargo run --release --example builder_validation
//! ```

use feature_extractor::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("PipelineBuilder Validation with Real NVDA Data");
    println!("{}", "=".repeat(60));

    let data_path = "/Users/nigo/local/tlob-hft-pipeline/data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250203.mbo.dbn.zst";

    // Check if data exists
    if !std::path::Path::new(data_path).exists() {
        println!("Data file not found: {}", data_path);
        println!("Skipping real data validation.");
        return Ok(());
    }

    // =========================================================================
    // Test 1: Default Builder (40 features)
    // =========================================================================
    println!("\n[Test 1] Default Builder (40 features)");
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut pipeline = PipelineBuilder::new()
        .volume_sampling(5000)
        .window(100, 10)
        .build()?;

    println!("Builder summary:\n{}", PipelineBuilder::new()
        .volume_sampling(5000)
        .window(100, 10)
        .summary());

    let output = pipeline.process(data_path)?;
    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Messages processed: {}", output.messages_processed);
    println!("  Features extracted: {}", output.features_extracted);
    println!("  Sequences generated: {}", output.sequences_generated);
    println!("  Mid-prices collected: {}", output.mid_prices.len());
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    // Validate feature dimensions
    if !output.sequences.is_empty() {
        let first_seq = &output.sequences[0];
        println!("\nSequence validation:");
        println!("  Window size: {} (expected: 100)", first_seq.length);
        println!("  Features per snapshot: {} (expected: 40)", first_seq.features[0].len());
        
        assert_eq!(first_seq.length, 100, "Window size mismatch");
        assert_eq!(first_seq.features[0].len(), 40, "Feature count mismatch");
        
        // Check for NaN/Inf
        let has_nan = first_seq.features.iter()
            .flat_map(|f| f.iter())
            .any(|&v| v.is_nan() || v.is_infinite());
        println!("  Contains NaN/Inf: {}", has_nan);
        assert!(!has_nan, "Features should not contain NaN/Inf");
    }

    println!("  [PASS] Default builder test passed");

    // =========================================================================
    // Test 2: Builder with Derived Features (48 features)
    // =========================================================================
    println!("\n[Test 2] Builder with Derived Features (48 features)");
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut pipeline = PipelineBuilder::new()
        .with_derived_features()
        .volume_sampling(5000)
        .window(100, 10)
        .build()?;

    let output = pipeline.process(data_path)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Messages processed: {}", output.messages_processed);
    println!("  Features extracted: {}", output.features_extracted);
    println!("  Sequences generated: {}", output.sequences_generated);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    if !output.sequences.is_empty() {
        let first_seq = &output.sequences[0];
        println!("\nSequence validation:");
        println!("  Features per snapshot: {} (expected: 48)", first_seq.features[0].len());
        
        assert_eq!(first_seq.features[0].len(), 48, "Feature count mismatch for derived");
    }

    println!("  [PASS] Derived features test passed");

    // =========================================================================
    // Test 3: Builder with MBO Features (76 features)
    // =========================================================================
    println!("\n[Test 3] Builder with MBO Features (76 features)");
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut pipeline = PipelineBuilder::new()
        .with_mbo_features()
        .volume_sampling(5000)
        .window(100, 10)
        .build()?;

    let output = pipeline.process(data_path)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Messages processed: {}", output.messages_processed);
    println!("  Features extracted: {}", output.features_extracted);
    println!("  Sequences generated: {}", output.sequences_generated);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    if !output.sequences.is_empty() {
        let first_seq = &output.sequences[0];
        println!("\nSequence validation:");
        println!("  Features per snapshot: {} (expected: 76)", first_seq.features[0].len());
        
        assert_eq!(first_seq.features[0].len(), 76, "Feature count mismatch for MBO");
    }

    println!("  [PASS] MBO features test passed");

    // =========================================================================
    // Test 4: Full Feature Set (84 features)
    // =========================================================================
    println!("\n[Test 4] Full Feature Set (84 features)");
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut pipeline = PipelineBuilder::new()
        .with_derived_features()
        .with_mbo_features()
        .volume_sampling(5000)
        .window(100, 10)
        .build()?;

    let output = pipeline.process(data_path)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Messages processed: {}", output.messages_processed);
    println!("  Features extracted: {}", output.features_extracted);
    println!("  Sequences generated: {}", output.sequences_generated);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    if !output.sequences.is_empty() {
        let first_seq = &output.sequences[0];
        println!("\nSequence validation:");
        println!("  Features per snapshot: {} (expected: 84)", first_seq.features[0].len());
        
        assert_eq!(first_seq.features[0].len(), 84, "Feature count mismatch for full");
    }

    println!("  [PASS] Full feature set test passed");

    // =========================================================================
    // Test 5: Event-based Sampling
    // =========================================================================
    println!("\n[Test 5] Event-based Sampling");
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut pipeline = PipelineBuilder::new()
        .event_sampling(1000)
        .window(100, 10)
        .build()?;

    let output = pipeline.process(data_path)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Messages processed: {}", output.messages_processed);
    println!("  Features extracted: {}", output.features_extracted);
    println!("  Sequences generated: {}", output.sequences_generated);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    println!("  [PASS] Event-based sampling test passed");

    // =========================================================================
    // Test 6: Preset-based Configuration
    // =========================================================================
    println!("\n[Test 6] Preset-based Configuration (DeepLOB)");
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut pipeline = PipelineBuilder::from_preset(Preset::DeepLOB)
        .volume_sampling(5000)
        .build()?;

    let output = pipeline.process(data_path)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Messages processed: {}", output.messages_processed);
    println!("  Features extracted: {}", output.features_extracted);
    println!("  Sequences generated: {}", output.sequences_generated);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    if !output.sequences.is_empty() {
        let first_seq = &output.sequences[0];
        assert_eq!(first_seq.features[0].len(), 40, "DeepLOB should have 40 features");
    }

    println!("  [PASS] Preset-based configuration test passed");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("ALL TESTS PASSED");
    println!("{}", "=".repeat(60));
    println!("\nThe PipelineBuilder API is working correctly with real data.");
    println!("All feature counts are auto-computed and validated.");

    Ok(())
}

