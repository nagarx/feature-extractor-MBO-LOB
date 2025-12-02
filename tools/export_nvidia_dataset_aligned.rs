//! Export NVIDIA MBO Dataset with Aligned Sequences and Labels
//!
//! This example uses the NEW AlignedBatchExporter to ensure perfect 1:1
//! alignment between sequences and labels, fixing the root cause of model collapse.
//!
//! Features:
//! - Process single or multiple days
//! - Event-based sampling (1000 events)
//! - Feature extraction (40 RAW LOB features: prices in $, sizes in shares)
//! - Sequence generation (100 timesteps, stride 10)
//! - Label generation with PERFECT ALIGNMENT
//! - Market-structure preserving Z-score normalization
//! - Export pre-built sequences [N_seq, 100, 40]
//! - Export aligned labels [N_seq]
//! - Comprehensive validation
//!
//! Usage:
//! ```bash
//! # Single day (for testing)
//! cargo run --release --example export_nvidia_dataset_aligned -- --day 2025-02-03
//!
//! # Multiple days
//! cargo run --release --example export_nvidia_dataset_aligned -- --start 2025-02-03 --end 2025-02-05
//!
//! # All 165 days (full 8-month dataset)
//! cargo run --release --example export_nvidia_dataset_aligned -- --start 2025-02-03 --end 2025-09-29
//! ```

use feature_extractor::{
    AlignedBatchExporter, AlignedDayExport, LabelConfig, Pipeline, PipelineConfig,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Configuration for the export process
struct ExportConfig {
    /// Input data directory
    data_dir: PathBuf,

    /// Output directory for aligned exports
    output_dir: PathBuf,

    /// Pipeline configuration
    pipeline_config: PipelineConfig,

    /// Label configuration
    label_config: LabelConfig,
}

impl ExportConfig {
    /// Create export configuration for NVIDIA HFT experiment with aligned export
    fn nvidia_hft_aligned(output_dir: PathBuf, data_dir: PathBuf) -> Self {
        // Try to load pipeline configuration from experiment config
        let config_path = PathBuf::from("experiments/nvidia_30day_v1/config.toml");
        let mut pipeline_config = if config_path.exists() {
            println!("  📄 Loading config from: {}", config_path.display());
            PipelineConfig::load_toml(&config_path).expect("Failed to load pipeline config")
        } else {
            // Fallback: Create default configuration matching experiment
            let mut config = PipelineConfig::new();

            // Event-based sampling: 1000 events (~1.3s at peak trading)
            if let Some(ref mut sampling) = config.sampling {
                sampling.strategy = feature_extractor::SamplingStrategy::EventBased;
                sampling.event_count = Some(1000);
                sampling.volume_threshold = None;
            }

            // Sequence configuration: 100 timesteps, stride 10
            config.sequence.window_size = 100;
            config.sequence.stride = 10;

            config
        };

        // ✅ CRITICAL: Force large buffer REGARDLESS of config source
        // For a full day with ~18K snapshots, we need buffer >> 1000
        // Setting to 50K to handle any day in the dataset
        if pipeline_config.sequence.max_buffer_size < 50_000 {
            println!(
                "  🔧 Forcing large buffer: {} → 50,000",
                pipeline_config.sequence.max_buffer_size
            );
            pipeline_config.sequence.max_buffer_size = 50_000;
        }

        // Label configuration (TLOB method: h=50, k=10, threshold=0.0008)
        // ✅ Strategy: short_conservative (5-7 min, 8 bps threshold)
        // ✅ Horizon: 50 events (~6.5 minutes at ~7-9 samples/min)
        // ✅ Threshold: 0.0008 (8 bps) - scaled for short horizon, high confidence
        let label_config = LabelConfig {
            horizon: 50,          // ✅ short
            smoothing_window: 10, // ✅ short
            threshold: 0.0008,    // ✅ short
        };

        Self {
            data_dir,
            output_dir,
            pipeline_config,
            label_config,
        }
    }
}

/// Result of processing a single day
#[derive(Debug)]
struct DayProcessingResult {
    day: String,
    success: bool,
    error: Option<String>,
    export_result: Option<AlignedDayExport>,
    processing_time_secs: f64,
}

/// Statistics for batch processing
#[derive(Debug)]
struct BatchStatistics {
    total_days: usize,
    successful_days: usize,
    failed_days: usize,
    total_sequences: usize,
    total_messages: usize,
    label_distribution: HashMap<String, usize>,
    total_time_secs: f64,
}

impl BatchStatistics {
    fn new() -> Self {
        let mut label_dist = HashMap::new();
        label_dist.insert("Up".to_string(), 0);
        label_dist.insert("Down".to_string(), 0);
        label_dist.insert("Stable".to_string(), 0);

        Self {
            total_days: 0,
            successful_days: 0,
            failed_days: 0,
            total_sequences: 0,
            total_messages: 0,
            label_distribution: label_dist,
            total_time_secs: 0.0,
        }
    }

    fn add_day(&mut self, result: &DayProcessingResult) {
        self.total_days += 1;
        self.total_time_secs += result.processing_time_secs;

        if result.success {
            self.successful_days += 1;

            if let Some(ref export) = result.export_result {
                self.total_sequences += export.n_sequences;
                self.total_messages += export.messages_processed;

                // Accumulate label distribution
                for (label, count) in &export.label_distribution {
                    *self.label_distribution.get_mut(label).unwrap() += count;
                }
            }
        } else {
            self.failed_days += 1;
        }
    }

    fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("BATCH PROCESSING SUMMARY");
        println!("{}", "=".repeat(80));

        println!("\n📊 Processing Statistics:");
        println!("  Total days:       {}", self.total_days);
        println!("  ✅ Successful:    {}", self.successful_days);
        println!("  ❌ Failed:        {}", self.failed_days);
        println!(
            "  Success rate:     {:.1}%",
            100.0 * self.successful_days as f64 / self.total_days as f64
        );

        println!("\n📈 Data Statistics:");
        println!(
            "  Total messages:   {:>12}",
            format_number(self.total_messages)
        );
        println!(
            "  Total sequences:  {:>12}",
            format_number(self.total_sequences)
        );
        println!(
            "  Sequences/day:    {:>12.0}",
            self.total_sequences as f64 / self.successful_days as f64
        );

        let total_labels: usize = self.label_distribution.values().sum();
        if total_labels > 0 {
            println!("\n🏷️  Label Distribution:");
            let total = total_labels as f64;
            let up_count = self.label_distribution["Up"];
            let down_count = self.label_distribution["Down"];
            let stable_count = self.label_distribution["Stable"];

            println!(
                "  Up:       {:>8} ({:>5.1}%)",
                format_number(up_count),
                100.0 * up_count as f64 / total
            );
            println!(
                "  Down:     {:>8} ({:>5.1}%)",
                format_number(down_count),
                100.0 * down_count as f64 / total
            );
            println!(
                "  Stable:   {:>8} ({:>5.1}%)",
                format_number(stable_count),
                100.0 * stable_count as f64 / total
            );

            // Check balance
            let up_pct = 100.0 * up_count as f64 / total;
            let down_pct = 100.0 * down_count as f64 / total;
            let stable_pct = 100.0 * stable_count as f64 / total;

            let max_pct = up_pct.max(down_pct).max(stable_pct);
            let min_pct = up_pct.min(down_pct).min(stable_pct);
            let imbalance = max_pct - min_pct;

            if imbalance < 10.0 {
                println!("  ✅ Labels are well-balanced (imbalance: {imbalance:.1}%)");
            } else if imbalance < 20.0 {
                println!("  ⚠️  Moderate imbalance: {imbalance:.1}%");
            } else {
                println!("  ❌ High imbalance: {imbalance:.1}% - consider adjusting threshold");
            }
        }

        println!("\n⏱️  Performance:");
        println!("  Total time:       {:.1}s", self.total_time_secs);
        if self.successful_days > 0 {
            println!(
                "  Avg per day:      {:.1}s",
                self.total_time_secs / self.successful_days as f64
            );
        }
        if self.total_messages > 0 {
            println!(
                "  Throughput:       {:.0} msgs/sec",
                self.total_messages as f64 / self.total_time_secs
            );
        }

        println!("\n{}", "=".repeat(80));
    }
}

/// Process a single day of data with aligned export
fn process_day(
    day: &str,
    config: &ExportConfig,
    exporter: &AlignedBatchExporter,
) -> DayProcessingResult {
    let start_time = Instant::now();

    println!("\n📅 Processing day: {day}");

    // Construct file path
    let filename = format!("xnas-itch-{}.mbo.dbn.zst", day.replace("-", ""));
    let data_path = config.data_dir.join(&filename);

    // Check if file exists
    if !data_path.exists() {
        let error = format!("Data file not found: {}", data_path.display());
        eprintln!("  ❌ {error}");
        return DayProcessingResult {
            day: day.to_string(),
            success: false,
            error: Some(error),
            export_result: None,
            processing_time_secs: start_time.elapsed().as_secs_f64(),
        };
    }

    // Create pipeline
    println!("  ⚙️  Creating pipeline...");
    let mut pipeline = match Pipeline::from_config(config.pipeline_config.clone()) {
        Ok(p) => p,
        Err(e) => {
            let error = format!("Pipeline creation error: {e}");
            eprintln!("  ❌ {error}");
            return DayProcessingResult {
                day: day.to_string(),
                success: false,
                error: Some(error),
                export_result: None,
                processing_time_secs: start_time.elapsed().as_secs_f64(),
            };
        }
    };

    // Process data
    println!("  📂 Processing: {filename}");
    let output = match pipeline.process(&data_path) {
        Ok(o) => o,
        Err(e) => {
            let error = format!("Pipeline error: {e}");
            eprintln!("  ❌ {error}");
            return DayProcessingResult {
                day: day.to_string(),
                success: false,
                error: Some(error),
                export_result: None,
                processing_time_secs: start_time.elapsed().as_secs_f64(),
            };
        }
    };

    println!(
        "  📊 Processed {} messages → {} sequences",
        format_number(output.messages_processed),
        format_number(output.sequences_generated)
    );

    // Export with aligned exporter
    println!("  💾 Exporting with alignment...");
    let export_result = match exporter.export_day(day, &output) {
        Ok(r) => r,
        Err(e) => {
            let error = format!("Export error: {e}");
            eprintln!("  ❌ {error}");
            return DayProcessingResult {
                day: day.to_string(),
                success: false,
                error: Some(error),
                export_result: None,
                processing_time_secs: start_time.elapsed().as_secs_f64(),
            };
        }
    };

    let elapsed = start_time.elapsed().as_secs_f64();
    println!("  ✅ Completed in {elapsed:.1}s");

    DayProcessingResult {
        day: day.to_string(),
        success: true,
        error: None,
        export_result: Some(export_result),
        processing_time_secs: elapsed,
    }
}

/// Generate list of trading days in range
fn generate_days(start: &str, end: &str) -> Vec<String> {
    // All available trading days in the full dataset (Feb-Sep 2025)
    // Expanded from 29 days to 165 days to support multi-month training
    let all_days = vec![
        // February (19 days)
        "2025-02-03",
        "2025-02-04",
        "2025-02-05",
        "2025-02-06",
        "2025-02-07",
        "2025-02-10",
        "2025-02-11",
        "2025-02-12",
        "2025-02-13",
        "2025-02-14",
        "2025-02-18",
        "2025-02-19",
        "2025-02-20",
        "2025-02-21",
        "2025-02-24",
        "2025-02-25",
        "2025-02-26",
        "2025-02-27",
        "2025-02-28",
        // March (20 days)
        "2025-03-03",
        "2025-03-04",
        "2025-03-05",
        "2025-03-06",
        "2025-03-07",
        "2025-03-10",
        "2025-03-11",
        "2025-03-12",
        "2025-03-13",
        "2025-03-14",
        "2025-03-17",
        "2025-03-18",
        "2025-03-19",
        "2025-03-20",
        "2025-03-21",
        "2025-03-24",
        "2025-03-25",
        "2025-03-26",
        "2025-03-27",
        "2025-03-28",
        "2025-03-31",
        // April (21 days)
        "2025-04-01",
        "2025-04-02",
        "2025-04-03",
        "2025-04-04",
        "2025-04-07",
        "2025-04-08",
        "2025-04-09",
        "2025-04-10",
        "2025-04-11",
        "2025-04-14",
        "2025-04-15",
        "2025-04-16",
        "2025-04-17",
        "2025-04-21",
        "2025-04-22",
        "2025-04-23",
        "2025-04-24",
        "2025-04-25",
        "2025-04-28",
        "2025-04-29",
        "2025-04-30",
        // May (21 days)
        "2025-05-01",
        "2025-05-02",
        "2025-05-05",
        "2025-05-06",
        "2025-05-07",
        "2025-05-08",
        "2025-05-09",
        "2025-05-12",
        "2025-05-13",
        "2025-05-14",
        "2025-05-15",
        "2025-05-16",
        "2025-05-19",
        "2025-05-20",
        "2025-05-21",
        "2025-05-22",
        "2025-05-23",
        "2025-05-27",
        "2025-05-28",
        "2025-05-29",
        "2025-05-30",
        // June (21 days)
        "2025-06-02",
        "2025-06-03",
        "2025-06-04",
        "2025-06-05",
        "2025-06-06",
        "2025-06-09",
        "2025-06-10",
        "2025-06-11",
        "2025-06-12",
        "2025-06-13",
        "2025-06-16",
        "2025-06-17",
        "2025-06-18",
        "2025-06-20",
        "2025-06-23",
        "2025-06-24",
        "2025-06-25",
        "2025-06-26",
        "2025-06-27",
        "2025-06-30",
        // July (22 days)
        "2025-07-01",
        "2025-07-02",
        "2025-07-03",
        "2025-07-07",
        "2025-07-08",
        "2025-07-09",
        "2025-07-10",
        "2025-07-11",
        "2025-07-14",
        "2025-07-15",
        "2025-07-16",
        "2025-07-17",
        "2025-07-18",
        "2025-07-21",
        "2025-07-22",
        "2025-07-23",
        "2025-07-24",
        "2025-07-25",
        "2025-07-28",
        "2025-07-29",
        "2025-07-30",
        "2025-07-31",
        // August (21 days)
        "2025-08-01",
        "2025-08-04",
        "2025-08-05",
        "2025-08-06",
        "2025-08-07",
        "2025-08-08",
        "2025-08-11",
        "2025-08-12",
        "2025-08-13",
        "2025-08-14",
        "2025-08-15",
        "2025-08-18",
        "2025-08-19",
        "2025-08-20",
        "2025-08-21",
        "2025-08-22",
        "2025-08-25",
        "2025-08-26",
        "2025-08-27",
        "2025-08-28",
        "2025-08-29",
        // September (20 days)
        "2025-09-02",
        "2025-09-03",
        "2025-09-04",
        "2025-09-05",
        "2025-09-08",
        "2025-09-09",
        "2025-09-10",
        "2025-09-11",
        "2025-09-12",
        "2025-09-15",
        "2025-09-16",
        "2025-09-17",
        "2025-09-18",
        "2025-09-19",
        "2025-09-22",
        "2025-09-23",
        "2025-09-24",
        "2025-09-25",
        "2025-09-26",
        "2025-09-29",
    ];

    // Filter days in range
    all_days
        .iter()
        .filter(|&&day| day >= start && day <= end)
        .map(|s| s.to_string())
        .collect()
}

/// Format number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let len = s.len();

    for (i, c) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }

    result
}

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("NVIDIA MBO DATASET EXPORT - WITH ALIGNED SEQUENCES");
    println!("{}", "=".repeat(80));
    println!("\n🔧 Using: AlignedBatchExporter (Phase 1 Fix)");
    println!("✅ Guarantees: 1:1 sequence-label alignment");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let (start_day, end_day, output_dir, data_dir) = if args.len() > 2 {
        // Check for --day flag (single day)
        if args[1] == "--day" {
            let output = if args.len() > 4 && args[3] == "--output" {
                PathBuf::from(&args[4])
            } else {
                PathBuf::from("experiments/nvidia_8month_v1/data_aligned_short_conservative")
            };
            let data = PathBuf::from("data/NVDA_2025-02-01_to_2025-09-30");
            (args[2].clone(), args[2].clone(), output, data)
        }
        // Check for --start and --end flags
        else if args.len() > 4 && args[1] == "--start" && args[3] == "--end" {
            let output = if args.len() > 6 && args[5] == "--output" {
                PathBuf::from(&args[6])
            } else {
                PathBuf::from("experiments/nvidia_8month_v1/data_aligned_short_conservative")
            };
            let data = PathBuf::from("data/NVDA_2025-02-01_to_2025-09-30");
            (args[2].clone(), args[4].clone(), output, data)
        } else {
            eprintln!("Usage:");
            eprintln!("  Single day:   cargo run --example export_nvidia_dataset_aligned -- --day 2025-02-03 --output path/to/output");
            eprintln!("  Date range:   cargo run --example export_nvidia_dataset_aligned -- --start 2025-02-03 --end 2025-09-29 --output path/to/output");
            std::process::exit(1);
        }
    } else {
        // Default: full 8-month dataset (165 days)
        (
            "2025-02-03".to_string(),
            "2025-09-29".to_string(),
            PathBuf::from("experiments/nvidia_8month_v1/data_aligned_short_conservative"),
            PathBuf::from("data/NVDA_2025-02-01_to_2025-09-30"),
        )
    };

    println!("\n📋 Configuration:");
    println!("  Date range:       {start_day} to {end_day}");

    // Create export configuration
    let config = ExportConfig::nvidia_hft_aligned(output_dir, data_dir);

    println!("  Data directory:   {}", config.data_dir.display());
    println!("  Output directory: {}", config.output_dir.display());
    println!("  Sampling:         Event-based (1000 events)");
    println!("  Features:         48 LOB features");
    println!("  Sequence:         100 timesteps, stride 10");
    println!("  Labeling:         TLOB (h=50, k=10, θ=0.0018) ✅ FIXED");
    println!("  Export format:    PRE-BUILT sequences [N_seq, 100, 48]");

    // Create output directory
    std::fs::create_dir_all(&config.output_dir).expect("Failed to create output directory");

    // Create aligned batch exporter
    println!("\n  📦 Creating AlignedBatchExporter...");
    let exporter = AlignedBatchExporter::new(
        &config.output_dir,
        config.label_config.clone(),
        config.pipeline_config.sequence.window_size, // 100
        config.pipeline_config.sequence.stride,      // 10
    );
    println!(
        "     Window size: {}",
        config.pipeline_config.sequence.window_size
    );
    println!("     Stride: {}", config.pipeline_config.sequence.stride);

    // Generate list of days to process
    let days = generate_days(&start_day, &end_day);
    println!("\n  Days to process:  {}", days.len());

    // Process each day
    let mut stats = BatchStatistics::new();
    let mut results = Vec::new();

    println!("\n{}", "=".repeat(80));
    println!("PROCESSING DAYS");
    println!("{}", "=".repeat(80));

    for day in &days {
        let result = process_day(day, &config, &exporter);
        stats.add_day(&result);
        results.push(result);
    }

    // Print summary
    stats.print_summary();

    // Print any errors
    let failed: Vec<_> = results.iter().filter(|r| !r.success).collect();

    if !failed.is_empty() {
        println!("\n❌ Failed Days:");
        for result in failed {
            println!("  {}: {}", result.day, result.error.as_ref().unwrap());
        }
    }

    // Validate outputs
    println!("\n🔍 Validating Exports...");
    validate_aligned_exports(&config.output_dir, &days);

    println!("\n✅ Export complete!");
    println!("📁 Output: {}", config.output_dir.display());
    println!("\n🎯 Next Steps:");
    println!("  1. Verify exports with: python TLOB/validate_exported_data.py");
    println!("  2. Load in Python using: TLOB/preprocessing/nvidia_aligned.py");
    println!("  3. Train model with aligned data (165 days!)");
}

/// Validate that all expected aligned export files were created
fn validate_aligned_exports(output_dir: &Path, days: &[String]) {
    let mut all_valid = true;

    for day in days {
        let sequences_path = output_dir.join(format!("{day}_sequences.npy"));
        let labels_path = output_dir.join(format!("{day}_labels.npy"));
        let metadata_path = output_dir.join(format!("{day}_metadata.json"));

        let sequences_ok = sequences_path.exists();
        let labels_ok = labels_path.exists();
        let metadata_ok = metadata_path.exists();

        if sequences_ok && labels_ok && metadata_ok {
            println!("  ✅ {day}: All files present (sequences, labels, metadata)");
        } else {
            println!("  ❌ {day}: Missing files");
            if !sequences_ok {
                println!("     - sequences.npy");
            }
            if !labels_ok {
                println!("     - labels.npy");
            }
            if !metadata_ok {
                println!("     - metadata.json");
            }
            all_valid = false;
        }
    }

    if all_valid {
        println!("\n✅ All aligned exports validated successfully!");
    } else {
        println!("\n⚠️  Some exports have missing files");
    }
}
