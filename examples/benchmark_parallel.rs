//! Parallel Processing Benchmark
//!
//! Benchmarks different thread counts to find optimal configuration.
//!
//! Usage: cargo run --release --features parallel --example benchmark_parallel

fn main() {
    #[cfg(not(feature = "parallel"))]
    {
        eprintln!("══════════════════════════════════════════════════════════════════");
        eprintln!("  This example requires the 'parallel' feature to be enabled.");
        eprintln!("══════════════════════════════════════════════════════════════════");
        eprintln!();
        eprintln!("  Run with:");
        eprintln!("    cargo run --release --features parallel --example benchmark_parallel");
        eprintln!();
        std::process::exit(1);
    }

    #[cfg(feature = "parallel")]
    {
        if let Err(e) = run_benchmark() {
            eprintln!("Benchmark failed: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "parallel")]
use feature_extractor::batch::{BatchConfig, BatchProcessor, ErrorMode};
#[cfg(feature = "parallel")]
use feature_extractor::PipelineBuilder;
#[cfg(feature = "parallel")]
use mbo_lob_reconstructor::Result;
#[cfg(feature = "parallel")]
use std::path::Path;
#[cfg(feature = "parallel")]
use std::time::Instant;

#[cfg(feature = "parallel")]
fn run_benchmark() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("     PARALLEL PROCESSING BENCHMARK - Mac M1 Optimization");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // System info
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    
    println!("📊 System Configuration:");
    println!("   Available CPUs: {}", num_cpus);
    println!("   Architecture: Apple Silicon M1");
    println!("   Memory: 16 GB unified\n");

    // Find test files
    let data_dir = Path::new("../data/NVDA_2025-02-01_to_2025-09-30");
    if !data_dir.exists() {
        println!("❌ Data directory not found: {:?}", data_dir);
        return Ok(());
    }

    let mut files: Vec<String> = std::fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "zst").unwrap_or(false))
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    
    files.sort();
    
    // Use 4 files for the benchmark (enough to show parallelism benefits)
    let test_files: Vec<&str> = files.iter().take(4).map(|s| s.as_str()).collect();
    
    if test_files.len() < 4 {
        println!("⚠️  Need at least 4 files for benchmark, found {}", test_files.len());
        return Ok(());
    }

    println!("📂 Test Files ({}):", test_files.len());
    for f in &test_files {
        println!("   - {}", f.split('/').last().unwrap_or(f));
    }
    println!();

    // Pipeline configuration
    let pipeline_config = PipelineBuilder::new()
        .lob_levels(10)
        .event_sampling(1000)
        .window(100, 10)
        .build_config()?;

    // Thread counts to test: 1, 2, 4, 6, 8
    let thread_counts = vec![1, 2, 4, 6, 8];
    
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    BENCHMARK RESULTS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    struct BenchResult {
        threads: usize,
        elapsed_secs: f64,
        total_messages: usize,
        throughput: f64,
        speedup: f64,
    }

    let mut results: Vec<BenchResult> = Vec::new();
    let mut baseline_time: Option<f64> = None;

    for &threads in &thread_counts {
        println!("🔄 Testing {} thread(s)...", threads);
        
        let batch_config = BatchConfig::new()
            .with_threads(threads)
            .with_error_mode(ErrorMode::FailFast);

        let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

        // Run benchmark
        let start = Instant::now();
        let output = processor.process_files(&test_files)?;
        let elapsed = start.elapsed();
        
        let elapsed_secs = elapsed.as_secs_f64();
        let total_messages = output.total_messages();
        let throughput = total_messages as f64 / elapsed_secs;
        
        // Calculate speedup vs single thread
        let speedup = if let Some(base) = baseline_time {
            base / elapsed_secs
        } else {
            baseline_time = Some(elapsed_secs);
            1.0
        };

        results.push(BenchResult {
            threads,
            elapsed_secs,
            total_messages,
            throughput,
            speedup,
        });

        println!("   ✅ {} threads: {:.2}s, {:.0} msg/s, {:.2}x speedup",
            threads, elapsed_secs, throughput, speedup);
    }

    // Print summary table
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                    SUMMARY TABLE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("┌─────────┬──────────┬────────────────┬───────────────┬─────────┐");
    println!("│ Threads │  Time(s) │ Messages       │ Throughput    │ Speedup │");
    println!("├─────────┼──────────┼────────────────┼───────────────┼─────────┤");

    let mut best_throughput = 0.0;
    let mut best_threads = 1;
    let mut best_efficiency = 0.0;
    let mut best_efficiency_threads = 1;

    for r in &results {
        let efficiency = r.speedup / r.threads as f64 * 100.0;
        
        println!("│    {:2}   │ {:7.2} │ {:>14} │ {:>10.0}/s │  {:5.2}x │",
            r.threads, r.elapsed_secs, 
            format_number(r.total_messages),
            r.throughput, r.speedup);
        
        if r.throughput > best_throughput {
            best_throughput = r.throughput;
            best_threads = r.threads;
        }
        
        if efficiency > best_efficiency && r.threads > 1 {
            best_efficiency = efficiency;
            best_efficiency_threads = r.threads;
        }
    }

    println!("└─────────┴──────────┴────────────────┴───────────────┴─────────┘");

    // Efficiency analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                    EFFICIENCY ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("┌─────────┬─────────────────────────────────────────────────────┬──────────┐");
    println!("│ Threads │ Efficiency (speedup / threads × 100%)              │ Rating   │");
    println!("├─────────┼─────────────────────────────────────────────────────┼──────────┤");

    for r in &results {
        let efficiency = r.speedup / r.threads as f64 * 100.0;
        let bar_len = (efficiency / 2.0).min(25.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(25 - bar_len);
        
        let rating = if efficiency >= 80.0 { "Excellent" }
            else if efficiency >= 60.0 { "Good" }
            else if efficiency >= 40.0 { "Fair" }
            else { "Poor" };
        
        println!("│    {:2}   │ {} {:5.1}% │ {:>8} │",
            r.threads, bar, efficiency, rating);
    }

    println!("└─────────┴─────────────────────────────────────────────────────┴──────────┘");

    // Memory estimation
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                    MEMORY ESTIMATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Estimated memory usage per thread: ~10 MB (Pipeline + buffers)");
    println!();
    
    for threads in &[1, 2, 4, 6, 8] {
        let mem_mb = threads * 10;
        let mem_pct = mem_mb as f64 / 16384.0 * 100.0;
        println!("   {} threads: ~{} MB ({:.1}% of 16 GB)", threads, mem_mb, mem_pct);
    }

    // Recommendations
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                    RECOMMENDATIONS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("📌 For Mac M1 (8-core, 16 GB):\n");
    
    println!("   🏆 Best Throughput: {} threads ({:.0} msg/s)", 
        best_threads, best_throughput);
    println!("   ⚡ Best Efficiency: {} threads ({:.1}% efficiency)", 
        best_efficiency_threads, best_efficiency);
    
    println!("\n   Recommended configurations:");
    println!("   • Batch processing (max speed): 4-6 threads");
    println!("   • Background processing (with other apps): 2-4 threads");
    println!("   • Development/testing: 2 threads");
    
    println!("\n   Memory is NOT a bottleneck - you have plenty of headroom.");
    println!("   The limiting factor is likely I/O and decompression overhead.");

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                    BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(feature = "parallel")]
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

