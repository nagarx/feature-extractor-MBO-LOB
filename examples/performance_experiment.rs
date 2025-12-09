//! Performance Experiment: Compressed vs Decompressed Data + Parallel Scaling
//!
//! This experiment measures the performance impact of hot store optimization
//! and true multi-file parallel processing.
//!
//! Usage:
//!   # Phase 1: Baseline with compressed data
//!   cargo run --release --features parallel --example performance_experiment -- --mode compressed
//!
//!   # Phase 2: Decompress dataset (run separately)
//!   ../MBO-LOB-reconstructor/target/release/decompress_to_hot_store \
//!       -i ../data/NVDA_2025-02-01_to_2025-09-30 \
//!       -o ../data/hot_store
//!
//!   # Phase 3: Benchmark with decompressed data
//!   cargo run --release --features parallel --example performance_experiment -- --mode decompressed
//!
//!   # Phase 4: Full comparison (requires both compressed and decompressed data)
//!   cargo run --release --features parallel --example performance_experiment -- --mode compare
//!
//!   # Phase 5: Test true parallel scaling (16 files with 1-8 threads)
//!   cargo run --release --features parallel --example performance_experiment -- --mode parallel

fn main() {
    #[cfg(not(feature = "parallel"))]
    {
        eprintln!("This experiment requires the 'parallel' feature.");
        eprintln!("Run with: cargo run --release --features parallel --example performance_experiment");
        std::process::exit(1);
    }

    #[cfg(feature = "parallel")]
    {
        if let Err(e) = run_experiment() {
            eprintln!("Experiment failed: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "parallel")]
mod experiment {
    use feature_extractor::batch::{BatchConfig, BatchProcessor, ErrorMode};
    use feature_extractor::PipelineBuilder;
    use mbo_lob_reconstructor::Result;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const COMPRESSED_DIR: &str = "../data/NVDA_2025-02-01_to_2025-09-30";
    const HOT_STORE_DIR: &str = "../data/hot_store";
    const NUM_TEST_FILES: usize = 4;
    const NUM_PARALLEL_FILES: usize = 16;  // For true parallel scaling tests
    const THREAD_COUNTS: &[usize] = &[1, 2, 4, 6, 8];

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Mode {
        Compressed,
        Decompressed,
        Compare,
        Parallel,  // True parallel scaling with 16 files
    }

    impl Mode {
        pub fn from_arg(arg: &str) -> Option<Self> {
            match arg.to_lowercase().as_str() {
                "compressed" | "c" => Some(Mode::Compressed),
                "decompressed" | "d" | "hot" => Some(Mode::Decompressed),
                "compare" | "both" => Some(Mode::Compare),
                "parallel" | "p" | "scaling" => Some(Mode::Parallel),
                _ => None,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct BenchResult {
        pub threads: usize,
        pub elapsed_secs: f64,
        pub total_messages: usize,
        pub throughput_msg_per_sec: f64,
        pub speedup_vs_single: f64,
    }

    pub fn parse_args() -> Mode {
        let args: Vec<String> = std::env::args().collect();
        
        for i in 0..args.len() {
            if args[i] == "--mode" || args[i] == "-m" {
                if let Some(mode_str) = args.get(i + 1) {
                    if let Some(mode) = Mode::from_arg(mode_str) {
                        return mode;
                    }
                }
            }
        }
        
        // Default: show help
        println!("Usage: performance_experiment --mode <compressed|decompressed|compare|parallel>");
        println!();
        println!("Modes:");
        println!("  compressed   - Benchmark with .dbn.zst files (baseline, 4 files)");
        println!("  decompressed - Benchmark with .dbn files from hot store (4 files)");
        println!("  compare      - Run both and show comparison");
        println!("  parallel     - True parallel scaling test (16 files, shows real speedup)");
        std::process::exit(0);
    }

    pub fn find_compressed_files() -> Result<Vec<PathBuf>> {
        let dir = Path::new(COMPRESSED_DIR);
        if !dir.exists() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                format!("Compressed data directory not found: {}", COMPRESSED_DIR)
            ));
        }

        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "zst").unwrap_or(false))
            .collect();
        
        files.sort();
        Ok(files.into_iter().take(NUM_TEST_FILES).collect())
    }

    pub fn find_decompressed_files() -> Result<Vec<PathBuf>> {
        find_decompressed_files_n(NUM_TEST_FILES)
    }

    pub fn find_decompressed_files_n(count: usize) -> Result<Vec<PathBuf>> {
        let dir = Path::new(HOT_STORE_DIR);
        if !dir.exists() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                format!("Hot store directory not found: {}. Run decompression first.", HOT_STORE_DIR)
            ));
        }

        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map(|e| e == "dbn").unwrap_or(false)
                    && !p.to_string_lossy().ends_with(".dbn.zst")
            })
            .collect();
        
        files.sort();
        
        if files.is_empty() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "No decompressed .dbn files found in hot store"
            ));
        }
        
        Ok(files.into_iter().take(count).collect())
    }

    pub fn run_benchmark(files: &[PathBuf], label: &str) -> Result<Vec<BenchResult>> {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  BENCHMARK: {} ({} files)", label.to_uppercase(), files.len());
        println!("═══════════════════════════════════════════════════════════════════\n");

        println!("📂 Files:");
        for f in files {
            let size_mb = std::fs::metadata(f)
                .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            println!("   • {} ({:.1} MB)", f.file_name().unwrap().to_string_lossy(), size_mb);
        }
        println!();

        let pipeline_config = PipelineBuilder::new()
            .lob_levels(10)
            .event_sampling(1000)
            .window(100, 10)
            .build_config()?;

        let file_strs: Vec<String> = files.iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let file_refs: Vec<&str> = file_strs.iter().map(|s| s.as_str()).collect();

        let mut results = Vec::new();
        let mut baseline_time: Option<f64> = None;

        for &threads in THREAD_COUNTS {
            print!("   Testing {} thread(s)... ", threads);
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let batch_config = BatchConfig::new()
                .with_threads(threads)
                .with_error_mode(ErrorMode::FailFast);

            let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

            let start = Instant::now();
            let output = processor.process_files(&file_refs)?;
            let elapsed = start.elapsed();

            let elapsed_secs = elapsed.as_secs_f64();
            let total_messages = output.total_messages();
            let throughput = total_messages as f64 / elapsed_secs;

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
                throughput_msg_per_sec: throughput,
                speedup_vs_single: speedup,
            });

            println!("✓ {:.2}s ({:.0} msg/s, {:.2}x)", elapsed_secs, throughput, speedup);
        }

        Ok(results)
    }

    pub fn print_results_table(results: &[BenchResult], label: &str) {
        println!("\n┌─────────────────────────────────────────────────────────────────────┐");
        println!("│  {} Results", label);
        println!("├─────────┬──────────┬────────────────┬───────────────┬───────────────┤");
        println!("│ Threads │  Time(s) │    Messages    │  Throughput   │    Speedup    │");
        println!("├─────────┼──────────┼────────────────┼───────────────┼───────────────┤");

        for r in results {
            println!(
                "│    {:2}   │ {:7.2}s │ {:>14} │ {:>10.0}/s │     {:5.2}x     │",
                r.threads,
                r.elapsed_secs,
                format_number(r.total_messages),
                r.throughput_msg_per_sec,
                r.speedup_vs_single
            );
        }

        println!("└─────────┴──────────┴────────────────┴───────────────┴───────────────┘");
    }

    pub fn print_comparison(compressed: &[BenchResult], decompressed: &[BenchResult]) {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("                    PERFORMANCE COMPARISON");
        println!("═══════════════════════════════════════════════════════════════════\n");

        println!("┌─────────┬─────────────────────┬─────────────────────┬─────────────┐");
        println!("│ Threads │ Compressed (msg/s)  │ Decompressed (msg/s)│   Speedup   │");
        println!("├─────────┼─────────────────────┼─────────────────────┼─────────────┤");

        for (c, d) in compressed.iter().zip(decompressed.iter()) {
            let speedup = d.throughput_msg_per_sec / c.throughput_msg_per_sec;
            println!(
                "│    {:2}   │ {:>15.0}    │ {:>15.0}    │   {:5.2}x    │",
                c.threads,
                c.throughput_msg_per_sec,
                d.throughput_msg_per_sec,
                speedup
            );
        }

        println!("└─────────┴─────────────────────┴─────────────────────┴─────────────┘");

        // Summary statistics
        let best_compressed = compressed.iter()
            .max_by(|a, b| a.throughput_msg_per_sec.partial_cmp(&b.throughput_msg_per_sec).unwrap())
            .unwrap();
        let best_decompressed = decompressed.iter()
            .max_by(|a, b| a.throughput_msg_per_sec.partial_cmp(&b.throughput_msg_per_sec).unwrap())
            .unwrap();

        let overall_speedup = best_decompressed.throughput_msg_per_sec / best_compressed.throughput_msg_per_sec;

        println!("\n📊 Summary:");
        println!("   Best Compressed:   {:>10.0} msg/s @ {} threads",
            best_compressed.throughput_msg_per_sec, best_compressed.threads);
        println!("   Best Decompressed: {:>10.0} msg/s @ {} threads",
            best_decompressed.throughput_msg_per_sec, best_decompressed.threads);
        println!("   ────────────────────────────────────────────");
        println!("   🚀 Overall Speedup: {:.2}x faster with hot store", overall_speedup);
    }

    pub fn format_number(n: usize) -> String {
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

    /// Run parallel scaling benchmark with 16 files
    pub fn run_parallel_scaling_benchmark() -> Result<()> {
        use feature_extractor::batch::{BatchConfig, BatchProcessor, ErrorMode};

        let files = find_decompressed_files_n(NUM_PARALLEL_FILES)?;
        
        if files.len() < NUM_PARALLEL_FILES {
            println!("⚠️  Only {} files available in hot store, need {} for full scaling test", 
                files.len(), NUM_PARALLEL_FILES);
            println!("   Run: decompress_to_hot_store -i ../data/NVDA... -o ../data/hot_store");
            println!("   to decompress more files.\n");
        }

        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  TRUE PARALLEL SCALING BENCHMARK ({} files)", files.len());
        println!("═══════════════════════════════════════════════════════════════════\n");

        let total_size_mb: f64 = files.iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len() as f64 / 1024.0 / 1024.0)
            .sum();

        println!("📂 Test Configuration:");
        println!("   Files: {} decompressed .dbn files", files.len());
        println!("   Total size: {:.1} MB", total_size_mb);
        println!("   Thread counts: {:?}\n", THREAD_COUNTS);

        let pipeline_config = PipelineBuilder::new()
            .lob_levels(10)
            .event_sampling(1000)
            .window(100, 10)
            .build_config()?;

        let file_strs: Vec<String> = files.iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let file_refs: Vec<&str> = file_strs.iter().map(|s| s.as_str()).collect();

        let mut results: Vec<BenchResult> = Vec::new();
        let mut baseline_time: Option<f64> = None;

        for &threads in THREAD_COUNTS {
            print!("   Testing {} thread(s) on {} files... ", threads, files.len());
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let batch_config = BatchConfig::new()
                .with_threads(threads)
                .with_error_mode(ErrorMode::FailFast);

            let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);

            let start = Instant::now();
            let output = processor.process_files(&file_refs)?;
            let elapsed = start.elapsed();

            let elapsed_secs = elapsed.as_secs_f64();
            let total_messages = output.total_messages();
            let throughput = total_messages as f64 / elapsed_secs;

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
                throughput_msg_per_sec: throughput,
                speedup_vs_single: speedup,
            });

            println!("✓ {:.2}s ({:.0} msg/s, {:.2}x speedup)", 
                elapsed_secs, throughput, speedup);
        }

        // Results table
        println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
        println!("│  Parallel Scaling Results ({} files)", files.len());
        println!("├─────────┬──────────┬────────────────┬───────────────┬─────────┬─────────┤");
        println!("│ Threads │  Time(s) │    Messages    │  Throughput   │ Speedup │ Effic.  │");
        println!("├─────────┼──────────┼────────────────┼───────────────┼─────────┼─────────┤");

        for r in &results {
            let efficiency = (r.speedup_vs_single / r.threads as f64) * 100.0;
            println!(
                "│    {:2}   │ {:7.2}s │ {:>14} │ {:>10.0}/s │  {:5.2}x │ {:5.1}%  │",
                r.threads,
                r.elapsed_secs,
                format_number(r.total_messages),
                r.throughput_msg_per_sec,
                r.speedup_vs_single,
                efficiency
            );
        }

        println!("└─────────┴──────────┴────────────────┴───────────────┴─────────┴─────────┘");

        // Analysis
        let best = results.iter()
            .max_by(|a, b| a.throughput_msg_per_sec.partial_cmp(&b.throughput_msg_per_sec).unwrap())
            .unwrap();
        
        let single_thread = &results[0];
        let best_efficiency_result = results.iter()
            .filter(|r| r.threads > 1)
            .max_by(|a, b| {
                let eff_a = a.speedup_vs_single / a.threads as f64;
                let eff_b = b.speedup_vs_single / b.threads as f64;
                eff_a.partial_cmp(&eff_b).unwrap()
            });

        println!("\n📊 Analysis:");
        println!("   Single-thread baseline: {:>10.0} msg/s", single_thread.throughput_msg_per_sec);
        println!("   Best throughput:        {:>10.0} msg/s @ {} threads ({:.2}x speedup)", 
            best.throughput_msg_per_sec, best.threads, best.speedup_vs_single);
        
        if let Some(eff) = best_efficiency_result {
            let efficiency = (eff.speedup_vs_single / eff.threads as f64) * 100.0;
            println!("   Best efficiency:        {:>10.1}% @ {} threads", efficiency, eff.threads);
        }

        // Scaling assessment
        println!("\n📈 Scaling Assessment:");
        if best.speedup_vs_single >= 3.0 {
            println!("   ✅ Excellent parallel scaling (>3x speedup)");
        } else if best.speedup_vs_single >= 2.0 {
            println!("   ✅ Good parallel scaling (>2x speedup)");
        } else if best.speedup_vs_single >= 1.5 {
            println!("   ⚠️  Moderate parallel scaling (~1.5x speedup)");
            println!("      Bottleneck may be I/O or memory bandwidth");
        } else {
            println!("   ❌ Limited parallel scaling (<1.5x speedup)");
            println!("      Check if files are on SSD and hot store is populated");
        }

        Ok(())
    }
}

#[cfg(feature = "parallel")]
fn run_experiment() -> mbo_lob_reconstructor::Result<()> {
    use experiment::*;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("     PERFORMANCE EXPERIMENT: Compressed vs Decompressed Data");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // System info
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    
    println!("📊 System: Apple Silicon M1 Pro, {} cores, 16 GB RAM\n", num_cpus);

    let mode = parse_args();

    match mode {
        Mode::Compressed => {
            let files = find_compressed_files()?;
            let results = run_benchmark(&files, "Compressed (.dbn.zst)")?;
            print_results_table(&results, "Compressed");
            
            println!("\n💡 Next step: Decompress dataset to hot store, then run with --mode decompressed");
        }

        Mode::Decompressed => {
            let files = find_decompressed_files()?;
            let results = run_benchmark(&files, "Decompressed (.dbn)")?;
            print_results_table(&results, "Decompressed");
            
            println!("\n💡 To see comparison: Run with --mode compare");
        }

        Mode::Compare => {
            println!("Running full comparison...\n");
            
            let compressed_files = find_compressed_files()?;
            let decompressed_files = find_decompressed_files()?;
            
            let compressed_results = run_benchmark(&compressed_files, "Compressed (.dbn.zst)")?;
            let decompressed_results = run_benchmark(&decompressed_files, "Decompressed (.dbn)")?;
            
            print_results_table(&compressed_results, "Compressed");
            print_results_table(&decompressed_results, "Decompressed");
            print_comparison(&compressed_results, &decompressed_results);
        }

        Mode::Parallel => {
            println!("Running true parallel scaling benchmark...\n");
            run_parallel_scaling_benchmark()?;
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                    EXPERIMENT COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    Ok(())
}

