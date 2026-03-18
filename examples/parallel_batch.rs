//! Parallel Batch Processing Example
//!
//! This example demonstrates how to process multiple DBN files in parallel
//! using the BatchProcessor with cancellation support.
//!
//! ## Features Demonstrated
//!
//! - BatchProcessor configuration
//! - Parallel multi-file processing
//! - Progress reporting
//! - Graceful cancellation
//! - Error handling modes
//!
//! ## Usage
//!
//! ```bash
//! cargo run --features parallel --example parallel_batch
//! ```
//!
//! ## Requirements
//!
//! - The `parallel` feature must be enabled
//! - DBN files in the data directory

#![cfg(feature = "parallel")]

use feature_extractor::batch::{
    BatchConfig, BatchProcessor, CancellationToken, ConsoleProgress, ErrorMode,
};
use feature_extractor::PipelineBuilder;
use mbo_lob_reconstructor::Result;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("🚀 Parallel Batch Processing Example");
    println!("════════════════════════════════════════════════\n");

    // Find test files
    let data_dir = Path::new("../data/hot_store");
    if !data_dir.exists() {
        println!("⚠️  Data directory not found: {:?}", data_dir);
        println!("   Please ensure the NVDA dataset is available.");
        return Ok(());
    }

    let files: Vec<String> = std::fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "zst").unwrap_or(false))
        .take(3) // Process only 3 files for demo
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    if files.is_empty() {
        println!("⚠️  No .dbn.zst files found in data directory");
        return Ok(());
    }

    println!("📂 Found {} files to process:", files.len());
    for f in &files {
        println!("   - {}", f.split('/').last().unwrap_or(f));
    }
    println!();

    // ========================================================================
    // Example 1: Basic Parallel Processing
    // ========================================================================

    println!("═══════════════════════════════════════════════");
    println!("📊 Example 1: Basic Parallel Processing");
    println!("═══════════════════════════════════════════════\n");

    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 10)
        .build_config()?;

    let batch_config = BatchConfig::new()
        .with_threads(2) // Use 2 threads
        .with_error_mode(ErrorMode::CollectErrors);

    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

    println!(
        "🔄 Processing with {} threads...",
        processor.batch_config().effective_threads()
    );

    let output = processor.process_files(&files)?;

    println!("\n✅ Processing Complete!");
    println!("   Successful: {}", output.successful_count());
    println!("   Failed: {}", output.failed_count());
    println!("   Elapsed: {:?}", output.elapsed);
    println!(
        "   Throughput: {:.2} msg/sec",
        output.throughput_msg_per_sec()
    );

    for result in output.iter() {
        println!(
            "   - {}: {} msg, {} seq, {:.2} msg/s",
            result.day,
            result.messages(),
            result.sequences(),
            result.throughput()
        );
    }

    // ========================================================================
    // Example 2: With Progress Reporting
    // ========================================================================

    println!("\n═══════════════════════════════════════════════");
    println!("📊 Example 2: With Progress Reporting");
    println!("═══════════════════════════════════════════════\n");

    let batch_config = BatchConfig::new()
        .with_threads(2)
        .with_error_mode(ErrorMode::CollectErrors);

    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config)
        .with_progress_callback(Box::new(ConsoleProgress::new()));

    println!("🔄 Processing with progress reporting...\n");

    let output = processor.process_files(&files)?;

    println!("\n✅ Complete! Total messages: {}", output.total_messages());

    // ========================================================================
    // Example 3: With Cancellation Support
    // ========================================================================

    println!("\n═══════════════════════════════════════════════");
    println!("📊 Example 3: With Cancellation Support");
    println!("═══════════════════════════════════════════════\n");

    // Create cancellation token
    let token = CancellationToken::new();

    // Flag to simulate external cancellation trigger
    let should_cancel = Arc::new(AtomicBool::new(false));
    let should_cancel_clone = should_cancel.clone();
    let token_clone = token.clone();

    // Simulate external cancellation after 5 seconds
    // (In real usage, this could be triggered by user input or timeout)
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(5));
        if should_cancel_clone.load(Ordering::Relaxed) {
            println!("\n⚠️  Cancellation requested!");
            token_clone.cancel();
        }
    });

    let batch_config = BatchConfig::new()
        .with_threads(2)
        .with_error_mode(ErrorMode::CollectErrors);

    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config)
        .with_cancellation_token(token.clone());

    println!("🔄 Processing with cancellation support...");
    println!("   (Token is available but not triggered in this demo)\n");

    let output = processor.process_files(&files)?;

    if output.was_cancelled {
        println!("⚠️  Processing was cancelled!");
        println!("   Completed: {} files", output.successful_count());
        println!("   Skipped: {} files", output.skipped_count);
    } else {
        println!("✅ All files processed successfully!");
        println!("   Total: {} files", output.successful_count());
    }

    // ========================================================================
    // Example 4: Error Handling
    // ========================================================================

    println!("\n═══════════════════════════════════════════════");
    println!("📊 Example 4: Error Handling (CollectErrors)");
    println!("═══════════════════════════════════════════════\n");

    // Add a non-existent file to test error handling
    let mut files_with_error = files.clone();
    files_with_error.push("/nonexistent/file.dbn.zst".to_string());

    let batch_config = BatchConfig::new()
        .with_threads(2)
        .with_error_mode(ErrorMode::CollectErrors);

    let processor = BatchProcessor::new(pipeline_config, batch_config);

    println!("🔄 Processing with one invalid file...\n");

    let output = processor.process_files(&files_with_error)?;

    println!("✅ Processing Complete (with errors collected)!");
    println!("   Successful: {}", output.successful_count());
    println!("   Failed: {}", output.failed_count());

    for error in output.iter_errors() {
        println!("   ❌ Error: {} - {}", error.file_path, error.error);
    }

    // ========================================================================
    // Summary
    // ========================================================================

    println!("\n════════════════════════════════════════════════");
    println!("✅ All Examples Complete!");
    println!("════════════════════════════════════════════════");

    Ok(())
}
