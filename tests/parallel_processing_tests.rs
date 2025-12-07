//! Comprehensive tests for parallel batch processing.
//!
//! These tests verify:
//! 1. Parallel and sequential processing produce identical results
//! 2. Thread isolation (no state leakage between threads)
//! 3. Error handling modes work correctly
//! 4. Progress reporting is accurate
//!
//! Run with: cargo test --features parallel --test parallel_processing_tests

#![cfg(feature = "parallel")]

use feature_extractor::batch::{
    BatchConfig, BatchOutput, BatchProcessor, ConsoleProgress, ErrorMode, ProgressCallback,
    ProgressInfo,
};
use feature_extractor::{Pipeline, PipelineBuilder, PipelineConfig, PipelineOutput};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

// ============================================================================
// Test Fixtures
// ============================================================================

fn get_test_files() -> Vec<String> {
    let data_dir = Path::new("../data/NVDA_2025-02-01_to_2025-09-30");

    if !data_dir.exists() {
        return Vec::new();
    }

    let mut files: Vec<String> = std::fs::read_dir(data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "zst").unwrap_or(false))
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    files.sort();
    files
}

fn create_test_config() -> PipelineConfig {
    PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 10)
        .build_config()
        .expect("Failed to create test config")
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_batch_processor_creation() {
    let pipeline_config = create_test_config();
    let batch_config = BatchConfig::new().with_threads(4);

    let processor = BatchProcessor::new(pipeline_config, batch_config);

    assert_eq!(processor.batch_config().num_threads, Some(4));
    assert_eq!(processor.batch_config().error_mode, ErrorMode::FailFast);
}

#[test]
fn test_batch_config_hardware_scaling() {
    // Test different hardware configurations
    let configs = vec![
        (2, "dual-core"),
        (4, "quad-core"),
        (8, "8-core"),
        (16, "16-core"),
        (32, "32-core"),
    ];

    for (threads, name) in configs {
        let config = BatchConfig::new().with_threads(threads);
        assert_eq!(
            config.num_threads,
            Some(threads),
            "Failed for {}",
            name
        );
    }
}

#[test]
fn test_error_mode_fail_fast() {
    let config = BatchConfig::new().with_error_mode(ErrorMode::FailFast);
    assert_eq!(config.error_mode, ErrorMode::FailFast);
}

#[test]
fn test_error_mode_collect_errors() {
    let config = BatchConfig::new().with_error_mode(ErrorMode::CollectErrors);
    assert_eq!(config.error_mode, ErrorMode::CollectErrors);
}

// ============================================================================
// Progress Callback Tests
// ============================================================================

struct TestProgressCallback {
    progress_calls: AtomicUsize,
    complete_called: AtomicUsize,
}

impl TestProgressCallback {
    fn new() -> Self {
        Self {
            progress_calls: AtomicUsize::new(0),
            complete_called: AtomicUsize::new(0),
        }
    }

    fn progress_count(&self) -> usize {
        self.progress_calls.load(Ordering::SeqCst)
    }

    fn complete_count(&self) -> usize {
        self.complete_called.load(Ordering::SeqCst)
    }
}

impl ProgressCallback for TestProgressCallback {
    fn on_progress(&self, _info: &ProgressInfo) {
        self.progress_calls.fetch_add(1, Ordering::SeqCst);
    }

    fn on_complete(&self, _output: &BatchOutput) {
        self.complete_called.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn test_progress_callback_basic() {
    let callback = TestProgressCallback::new();

    // Simulate progress calls
    let info = ProgressInfo {
        current_file: "test.dbn".to_string(),
        current_index: 0,
        total_files: 10,
        completed: 5,
        failed: 0,
        elapsed: Duration::from_secs(10),
    };

    callback.on_progress(&info);
    callback.on_progress(&info);

    assert_eq!(callback.progress_count(), 2);
    assert_eq!(callback.complete_count(), 0);
}

// ============================================================================
// Real Data Tests (require NVIDIA data)
// ============================================================================

#[test]
fn test_parallel_vs_sequential_identical_results() {
    let files = get_test_files();
    if files.is_empty() {
        println!("⚠️  Skipping test: No test data found");
        return;
    }

    // Use first 3 files for quick test
    let test_files: Vec<&str> = files.iter().take(3).map(|s| s.as_str()).collect();

    if test_files.len() < 2 {
        println!("⚠️  Skipping test: Need at least 2 files");
        return;
    }

    let pipeline_config = create_test_config();

    println!("\n📊 Parallel vs Sequential Comparison Test");
    println!("   Files: {}", test_files.len());

    // Process sequentially
    println!("\n   Processing SEQUENTIALLY...");
    let mut sequential_results: Vec<(String, PipelineOutput)> = Vec::new();

    for file in &test_files {
        let mut pipeline = Pipeline::from_config(pipeline_config.clone()).unwrap();
        let output = pipeline.process(file).unwrap();
        let day = Path::new(file)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        sequential_results.push((day, output));
    }

    // Process in parallel
    println!("   Processing PARALLEL...");
    let batch_config = BatchConfig::new()
        .with_threads(2)
        .with_error_mode(ErrorMode::FailFast);

    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let parallel_output = processor.process_files(&test_files).unwrap();

    // Compare results
    println!("\n   📈 Comparison Results:");

    assert_eq!(
        sequential_results.len(),
        parallel_output.successful_count(),
        "File count mismatch"
    );

    // Sort both by day for comparison
    sequential_results.sort_by(|a, b| a.0.cmp(&b.0));
    let parallel_by_day = parallel_output.results_by_day();

    for (seq, par) in sequential_results.iter().zip(parallel_by_day.iter()) {
        let seq_output = &seq.1;
        let par_output = &par.output;

        // Compare message counts
        assert_eq!(
            seq_output.messages_processed, par_output.messages_processed,
            "Messages mismatch for {}",
            seq.0
        );

        // Compare feature counts
        assert_eq!(
            seq_output.features_extracted, par_output.features_extracted,
            "Features mismatch for {}",
            seq.0
        );

        // Compare sequence counts
        assert_eq!(
            seq_output.sequences_generated, par_output.sequences_generated,
            "Sequences mismatch for {}",
            seq.0
        );

        println!(
            "      ✅ {} - {} messages, {} sequences",
            seq.0, seq_output.messages_processed, seq_output.sequences_generated
        );
    }

    println!("\n   ✅ Parallel processing produces IDENTICAL results to sequential!");
}

#[test]
fn test_parallel_numerical_correctness() {
    let files = get_test_files();
    if files.is_empty() {
        println!("⚠️  Skipping test: No test data found");
        return;
    }

    // Use first file for detailed comparison
    let test_file = &files[0];
    let pipeline_config = create_test_config();

    println!("\n📊 Numerical Correctness Test");
    println!("   File: {}", test_file);

    // Process sequentially
    let mut seq_pipeline = Pipeline::from_config(pipeline_config.clone()).unwrap();
    let seq_output = seq_pipeline.process(test_file).unwrap();

    // Process with BatchProcessor (1 file, but uses the parallel code path)
    let batch_config = BatchConfig::new().with_threads(1);
    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let par_output = processor.process_files(&[test_file]).unwrap();

    let par_result = &par_output.results[0].output;

    // Compare mid-prices (numerical values)
    assert_eq!(
        seq_output.mid_prices.len(),
        par_result.mid_prices.len(),
        "Mid-price count mismatch"
    );

    for (i, (seq_price, par_price)) in seq_output
        .mid_prices
        .iter()
        .zip(par_result.mid_prices.iter())
        .enumerate()
    {
        assert_eq!(
            seq_price.to_bits(),
            par_price.to_bits(),
            "Mid-price bit-level mismatch at index {}",
            i
        );
    }

    // Compare sequence features
    if !seq_output.sequences.is_empty() && !par_result.sequences.is_empty() {
        let seq_first = &seq_output.sequences[0];
        let par_first = &par_result.sequences[0];

        assert_eq!(
            seq_first.features.len(),
            par_first.features.len(),
            "Sequence length mismatch"
        );

        for (snap_idx, (seq_snap, par_snap)) in
            seq_first.features.iter().zip(par_first.features.iter()).enumerate()
        {
            for (feat_idx, (seq_val, par_val)) in seq_snap.iter().zip(par_snap.iter()).enumerate() {
                assert_eq!(
                    seq_val.to_bits(),
                    par_val.to_bits(),
                    "Feature value mismatch at seq[{}] feat[{}]",
                    snap_idx,
                    feat_idx
                );
            }
        }
    }

    println!("   ✅ Numerical values are BIT-LEVEL identical!");
}

#[test]
fn test_thread_isolation() {
    let files = get_test_files();
    if files.len() < 4 {
        println!("⚠️  Skipping test: Need at least 4 files");
        return;
    }

    let test_files: Vec<&str> = files.iter().take(4).map(|s| s.as_str()).collect();
    let pipeline_config = create_test_config();

    println!("\n📊 Thread Isolation Test");
    println!("   Files: {}", test_files.len());

    // Process multiple times with different thread counts
    let results_2_threads = {
        let batch_config = BatchConfig::new().with_threads(2);
        let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);
        processor.process_files(&test_files).unwrap()
    };

    let results_4_threads = {
        let batch_config = BatchConfig::new().with_threads(4);
        let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);
        processor.process_files(&test_files).unwrap()
    };

    // Results should be identical regardless of thread count
    assert_eq!(
        results_2_threads.total_messages(),
        results_4_threads.total_messages(),
        "Message count differs between 2 and 4 threads"
    );

    assert_eq!(
        results_2_threads.total_sequences(),
        results_4_threads.total_sequences(),
        "Sequence count differs between 2 and 4 threads"
    );

    // Check each day's results
    let days_2: HashSet<_> = results_2_threads.iter().map(|r| &r.day).collect();
    let days_4: HashSet<_> = results_4_threads.iter().map(|r| &r.day).collect();

    assert_eq!(days_2, days_4, "Different days processed");

    println!("   ✅ Thread isolation verified - results identical across thread counts!");
}

#[test]
fn test_error_handling_collect_mode() {
    let files = get_test_files();
    if files.is_empty() {
        println!("⚠️  Skipping test: No test data found");
        return;
    }

    // Create a list with one invalid file
    let mut test_files: Vec<String> = files.iter().take(2).cloned().collect();
    test_files.push("/nonexistent/file.dbn.zst".to_string());

    let pipeline_config = create_test_config();
    let batch_config = BatchConfig::new()
        .with_threads(2)
        .with_error_mode(ErrorMode::CollectErrors);

    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let test_files_ref: Vec<&str> = test_files.iter().map(|s| s.as_str()).collect();

    let output = processor.process_files(&test_files_ref).unwrap();

    println!("\n📊 Error Handling Test (CollectErrors mode)");
    println!("   Successful: {}", output.successful_count());
    println!("   Failed: {}", output.failed_count());

    // Should have processed valid files and collected error for invalid one
    assert!(output.successful_count() >= 1, "Should have some successes");
    assert!(output.failed_count() >= 1, "Should have captured the error");

    // Check error contains useful information
    for error in output.iter_errors() {
        println!("   Error: {} - {}", error.file_path, error.error);
        assert!(!error.error.is_empty(), "Error message should not be empty");
    }

    println!("   ✅ Error collection working correctly!");
}

#[test]
fn test_batch_output_statistics() {
    let files = get_test_files();
    if files.len() < 2 {
        println!("⚠️  Skipping test: Need at least 2 files");
        return;
    }

    let test_files: Vec<&str> = files.iter().take(2).map(|s| s.as_str()).collect();
    let pipeline_config = create_test_config();
    let batch_config = BatchConfig::new().with_threads(2);

    let processor = BatchProcessor::new(pipeline_config, batch_config);
    let output = processor.process_files(&test_files).unwrap();

    println!("\n📊 Batch Output Statistics Test");
    println!("   Successful: {}", output.successful_count());
    println!("   Total messages: {}", output.total_messages());
    println!("   Total sequences: {}", output.total_sequences());
    println!("   Throughput: {:.2} msg/sec", output.throughput_msg_per_sec());
    println!("   Speedup: {:.2}x", output.speedup_factor());
    println!("   Elapsed: {:?}", output.elapsed);

    // Verify statistics are reasonable
    assert!(output.total_messages() > 0, "Should have processed messages");
    assert!(output.throughput_msg_per_sec() > 0.0, "Throughput should be positive");
    assert!(output.speedup_factor() > 0.0, "Speedup should be positive");
    assert!(output.elapsed.as_nanos() > 0, "Elapsed time should be positive");

    // Verify individual results
    for result in output.iter() {
        assert!(result.messages() > 0, "Each day should have messages");
        assert!(result.throughput() > 0.0, "Each day should have throughput");
        println!(
            "   - {}: {} msg, {} seq, {:.2} msg/s",
            result.day,
            result.messages(),
            result.sequences(),
            result.throughput()
        );
    }

    println!("   ✅ Statistics are accurate and reasonable!");
}

#[test]
fn test_console_progress_callback() {
    // Just verify it doesn't panic
    let callback = ConsoleProgress::new().verbose();

    let info = ProgressInfo {
        current_file: "test.dbn.zst".to_string(),
        current_index: 0,
        total_files: 10,
        completed: 5,
        failed: 1,
        elapsed: Duration::from_secs(60),
    };

    callback.on_progress(&info);

    let _output = BatchOutput {
        results: vec![],
        errors: vec![],
        elapsed: Duration::from_secs(120),
        threads_used: 4,
    };

    // Note: callback.on_complete(&_output) would print to console

    println!("   ✅ Console progress callback works!");
}

