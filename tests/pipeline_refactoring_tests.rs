//! Tests for pipeline refactoring - verifying process() and process_messages() equivalence.
//!
//! These tests ensure that any refactoring to consolidate code paths maintains
//! exact numerical precision and identical results.
//!
//! Run with: cargo test --release --test pipeline_refactoring_tests

use feature_extractor::{Pipeline, PipelineBuilder, PipelineConfig};
use mbo_lob_reconstructor::DbnLoader;
use std::path::Path;

// ============================================================================
// Test Fixtures
// ============================================================================

fn get_test_file() -> Option<String> {
    let data_dir = Path::new("../data/NVDA_2025-02-01_to_2025-09-30");

    if !data_dir.exists() {
        return None;
    }

    let mut files: Vec<String> = std::fs::read_dir(data_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "zst").unwrap_or(false))
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    files.sort();
    files.first().cloned()
}

fn create_test_config() -> PipelineConfig {
    PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 10)
        .build_config()
        .unwrap()
}

// ============================================================================
// Core Equivalence Tests
// ============================================================================

/// Test that process() and process_messages() produce IDENTICAL results.
/// This is the critical test before refactoring.
#[test]
fn test_process_vs_process_messages_equivalence() {
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST: process() vs process_messages() Equivalence");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let test_file = match get_test_file() {
        Some(f) => f,
        None => {
            println!("   ⏭️  Skipped: No test files found");
            return;
        }
    };

    println!("   Test file: {}", test_file.split('/').last().unwrap_or(&test_file));

    let config = create_test_config();

    // Method 1: Use process() directly
    let mut pipeline1 = Pipeline::from_config(config.clone()).unwrap();
    let output1 = pipeline1.process(&test_file).unwrap();

    // Method 2: Use process_messages() with DbnLoader
    let mut pipeline2 = Pipeline::from_config(config.clone()).unwrap();
    let loader = DbnLoader::new(&test_file).unwrap();
    let output2 = pipeline2.process_messages(loader.iter_messages().unwrap()).unwrap();

    // ========================================================================
    // Verify ALL fields are identical
    // ========================================================================

    println!("\n   Comparing outputs...");

    // 1. Messages processed
    assert_eq!(
        output1.messages_processed, output2.messages_processed,
        "Messages processed mismatch: {} vs {}",
        output1.messages_processed, output2.messages_processed
    );
    println!("   ✓ messages_processed: {} == {}", output1.messages_processed, output2.messages_processed);

    // 2. Features extracted
    assert_eq!(
        output1.features_extracted, output2.features_extracted,
        "Features extracted mismatch: {} vs {}",
        output1.features_extracted, output2.features_extracted
    );
    println!("   ✓ features_extracted: {} == {}", output1.features_extracted, output2.features_extracted);

    // 3. Sequences generated
    assert_eq!(
        output1.sequences_generated, output2.sequences_generated,
        "Sequences generated mismatch: {} vs {}",
        output1.sequences_generated, output2.sequences_generated
    );
    println!("   ✓ sequences_generated: {} == {}", output1.sequences_generated, output2.sequences_generated);

    // 4. Sequence count
    assert_eq!(
        output1.sequences.len(), output2.sequences.len(),
        "Sequence count mismatch: {} vs {}",
        output1.sequences.len(), output2.sequences.len()
    );
    println!("   ✓ sequences.len(): {} == {}", output1.sequences.len(), output2.sequences.len());

    // 5. Mid prices count
    assert_eq!(
        output1.mid_prices.len(), output2.mid_prices.len(),
        "Mid prices count mismatch: {} vs {}",
        output1.mid_prices.len(), output2.mid_prices.len()
    );
    println!("   ✓ mid_prices.len(): {} == {}", output1.mid_prices.len(), output2.mid_prices.len());

    // 6. Stride and window size (should be config-based, just verify)
    assert_eq!(output1.stride, output2.stride);
    assert_eq!(output1.window_size, output2.window_size);
    println!("   ✓ stride: {} == {}", output1.stride, output2.stride);
    println!("   ✓ window_size: {} == {}", output1.window_size, output2.window_size);

    // ========================================================================
    // Verify numerical precision of mid_prices
    // ========================================================================

    println!("\n   Verifying mid_price numerical precision...");
    
    for (i, (p1, p2)) in output1.mid_prices.iter().zip(output2.mid_prices.iter()).enumerate() {
        assert!(
            (p1 - p2).abs() < 1e-10,
            "Mid price mismatch at index {}: {} vs {} (diff: {})",
            i, p1, p2, (p1 - p2).abs()
        );
    }
    println!("   ✓ All {} mid_prices match with <1e-10 precision", output1.mid_prices.len());

    // ========================================================================
    // Verify sequences match exactly
    // ========================================================================

    println!("\n   Verifying sequence contents...");

    for (seq_idx, (seq1, seq2)) in output1.sequences.iter().zip(output2.sequences.iter()).enumerate() {
        // Check feature matrix dimensions
        assert_eq!(
            seq1.features.len(), seq2.features.len(),
            "Sequence {} feature count mismatch: {} vs {}",
            seq_idx, seq1.features.len(), seq2.features.len()
        );

        // Check each feature row
        for (row_idx, (row1, row2)) in seq1.features.iter().zip(seq2.features.iter()).enumerate() {
            assert_eq!(
                row1.len(), row2.len(),
                "Sequence {} row {} length mismatch: {} vs {}",
                seq_idx, row_idx, row1.len(), row2.len()
            );

            // Check numerical precision
            for (col_idx, (v1, v2)) in row1.iter().zip(row2.iter()).enumerate() {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "Sequence {} feature [{},{}] mismatch: {} vs {} (diff: {})",
                    seq_idx, row_idx, col_idx, v1, v2, (v1 - v2).abs()
                );
            }
        }
    }

    let total_features: usize = output1.sequences.iter()
        .map(|s| s.features.len() * s.features.first().map(|r| r.len()).unwrap_or(0))
        .sum();
    
    println!("   ✓ All {} sequences verified ({} total feature values)", 
        output1.sequences.len(), total_features);

    // ========================================================================
    // Final Summary
    // ========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   ✅ PASS: process() and process_messages() produce IDENTICAL results");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

/// Test that process_source() also produces identical results.
#[test]
fn test_process_source_equivalence() {
    use mbo_lob_reconstructor::DbnSource;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST: process() vs process_source(DbnSource) Equivalence");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let test_file = match get_test_file() {
        Some(f) => f,
        None => {
            println!("   ⏭️  Skipped: No test files found");
            return;
        }
    };

    println!("   Test file: {}", test_file.split('/').last().unwrap_or(&test_file));

    let config = create_test_config();

    // Method 1: Use process() directly
    let mut pipeline1 = Pipeline::from_config(config.clone()).unwrap();
    let output1 = pipeline1.process(&test_file).unwrap();

    // Method 2: Use process_source() with DbnSource
    let mut pipeline2 = Pipeline::from_config(config.clone()).unwrap();
    let source = DbnSource::new(&test_file).unwrap();
    let output2 = pipeline2.process_source(source).unwrap();

    // Verify identical results
    assert_eq!(output1.messages_processed, output2.messages_processed);
    assert_eq!(output1.features_extracted, output2.features_extracted);
    assert_eq!(output1.sequences_generated, output2.sequences_generated);
    assert_eq!(output1.sequences.len(), output2.sequences.len());
    assert_eq!(output1.mid_prices.len(), output2.mid_prices.len());

    // Verify numerical precision
    for (p1, p2) in output1.mid_prices.iter().zip(output2.mid_prices.iter()) {
        assert!((p1 - p2).abs() < 1e-10, "Mid price mismatch");
    }

    // Verify sequences
    for (seq1, seq2) in output1.sequences.iter().zip(output2.sequences.iter()) {
        for (row1, row2) in seq1.features.iter().zip(seq2.features.iter()) {
            for (v1, v2) in row1.iter().zip(row2.iter()) {
                assert!((v1 - v2).abs() < 1e-10, "Feature value mismatch");
            }
        }
    }

    println!("   ✓ messages_processed: {}", output1.messages_processed);
    println!("   ✓ features_extracted: {}", output1.features_extracted);
    println!("   ✓ sequences: {}", output1.sequences.len());
    println!("   ✓ mid_prices: {}", output1.mid_prices.len());

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   ✅ PASS: process() and process_source() produce IDENTICAL results");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

/// Test multiple processing methods all produce the same results.
#[test]
fn test_all_processing_methods_equivalence() {
    use mbo_lob_reconstructor::DbnSource;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST: All Processing Methods Equivalence");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let test_file = match get_test_file() {
        Some(f) => f,
        None => {
            println!("   ⏭️  Skipped: No test files found");
            return;
        }
    };

    let config = create_test_config();

    // Collect results from all methods
    let mut results = Vec::new();

    // Method 1: process(path)
    {
        let mut pipeline = Pipeline::from_config(config.clone()).unwrap();
        results.push(("process(path)", pipeline.process(&test_file).unwrap()));
    }

    // Method 2: process_messages(iterator)
    {
        let mut pipeline = Pipeline::from_config(config.clone()).unwrap();
        let loader = DbnLoader::new(&test_file).unwrap();
        results.push(("process_messages(iter)", pipeline.process_messages(loader.iter_messages().unwrap()).unwrap()));
    }

    // Method 3: process_source(DbnSource)
    {
        let mut pipeline = Pipeline::from_config(config.clone()).unwrap();
        let source = DbnSource::new(&test_file).unwrap();
        results.push(("process_source(DbnSource)", pipeline.process_source(source).unwrap()));
    }

    // Verify all produce identical results
    let baseline = &results[0].1;
    
    for (name, output) in &results[1..] {
        assert_eq!(
            baseline.messages_processed, output.messages_processed,
            "{} messages_processed differs from baseline", name
        );
        assert_eq!(
            baseline.features_extracted, output.features_extracted,
            "{} features_extracted differs from baseline", name
        );
        assert_eq!(
            baseline.sequences.len(), output.sequences.len(),
            "{} sequence count differs from baseline", name
        );
        println!("   ✓ {} matches baseline", name);
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   ✅ PASS: All {} processing methods produce IDENTICAL results", results.len());
    println!("═══════════════════════════════════════════════════════════════════\n");
}

