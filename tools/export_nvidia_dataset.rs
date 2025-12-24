//! ⚠️ DEPRECATED - Use `export_dataset` Instead
//!
//! This legacy tool uses `BatchExporter` which has critical bugs:
//! - 10x data inflation (every sample from every overlapping sequence)
//! - Label misalignment (labels don't match sequence endpoints)
//!
//! # Recommended Alternative
//!
//! Use the TOML-config-driven `export_dataset` tool which uses `AlignedBatchExporter`:
//!
//! ```bash
//! cargo run --release --bin export_dataset --features parallel -- --config configs/nvda_98feat.toml
//! ```
//!
//! Or use `export_aligned_legacy` for the aligned version:
//!
//! ```bash
//! cargo run --release --bin export_aligned_legacy -- --start 2025-02-03 --end 2025-02-05
//! ```
//!
//! ---
//!
//! # Legacy Tool Documentation (kept for reference)
//!
//! Export NVIDIA MBO Dataset to NumPy Format
//!
//! This example processes NVIDIA MBO data and exports it to NumPy format
//! for ML training with PyTorch/TensorFlow.
//!
//! Features:
//! - Process single or multiple days
//! - Event-based sampling (1000 events)
//! - Feature extraction (48 LOB features)
//! - Sequence generation (100 timesteps, stride 10)
//! - Label generation (TLOB method: h=200, k=20, threshold=0.0005)
//! - Export to NumPy format with metadata
//! - Progress tracking and validation
//!
//! Usage:
//! ```bash
//! # Single day
//! cargo run --release --example export_nvidia_dataset -- --day 2025-02-03
//!
//! # Multiple days
//! cargo run --release --example export_nvidia_dataset -- --start 2025-02-03 --end 2025-02-05
//!
//! # All 29 days
//! cargo run --release --example export_nvidia_dataset -- --start 2025-02-03 --end 2025-03-14
//! ```

use feature_extractor::{BatchExporter, DayExportResult, LabelConfig, Pipeline, PipelineConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Configuration for the export process
struct ExportConfig {
    /// Input data directory
    data_dir: PathBuf,

    /// Output directory for exports
    output_dir: PathBuf,

    /// Pipeline configuration
    pipeline_config: PipelineConfig,

    /// Label configuration
    label_config: LabelConfig,
}

impl ExportConfig {
    /// Create export configuration for NVIDIA HFT experiment
    fn nvidia_hft() -> Self {
        // Try to load pipeline configuration from experiment config
        let config_path = PathBuf::from("experiments/nvidia_30day_v1/config.toml");
        let pipeline_config = if config_path.exists() {
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

        // Label configuration (TLOB method: h=200, k=20, threshold)
        // EXPERIMENT 2 (CORRECTED): Threshold tuning 0.0025 → 0.0018 (18 bps)
        // Previous: 0.0025 (25 bps) → Stable=54.9%, imbalance=2.50:1 ❌
        // Logic: LOWER threshold = FEWER Stable (|Δp| ≤ θ → Stable)
        // Target: 0.0018 (18 bps) → Stable=40-45%, imbalance<2.0:1 ✓
        // See: experiments/nvidia_30day_v1/THRESHOLD_EXPERIMENTS.md
        let label_config = LabelConfig {
            horizon: 200,         // ~2.6 minutes at 1000 events
            smoothing_window: 20, // Noise reduction
            threshold: 0.0018,    // 0.18% change threshold (18 basis points)
        };

        Self {
            data_dir: PathBuf::from("data/NVDA_2025-02-01_to_2025-09-30"),
            output_dir: PathBuf::from("experiments/nvidia_30day_v1/data"),
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
    export_result: Option<DayExportResult>,
    processing_time_secs: f64,
}

/// Statistics for batch processing
#[derive(Debug)]
struct BatchStatistics {
    total_days: usize,
    successful_days: usize,
    failed_days: usize,
    total_features: usize,
    total_labels: usize,
    total_messages: usize,
    total_sequences: usize,
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
            total_features: 0,
            total_labels: 0,
            total_messages: 0,
            total_sequences: 0,
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
                self.total_features += export.n_features;
                self.total_labels += export.n_labels;
                self.total_messages += export.messages_processed;
                self.total_sequences += export.n_sequences;

                // Accumulate label distribution
                if let Some(ref dist) = export.label_distribution {
                    for (label, count) in dist {
                        *self.label_distribution.get_mut(label).unwrap() += count;
                    }
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
            "  Total features:   {:>12}",
            format_number(self.total_features)
        );
        println!(
            "  Total labels:     {:>12}",
            format_number(self.total_labels)
        );
        println!(
            "  Total sequences:  {:>12}",
            format_number(self.total_sequences)
        );

        if self.total_labels > 0 {
            println!("\n🏷️  Label Distribution:");
            let total = self.total_labels as f64;
            println!(
                "  Up:       {:>8} ({:>5.1}%)",
                format_number(self.label_distribution["Up"]),
                100.0 * self.label_distribution["Up"] as f64 / total
            );
            println!(
                "  Down:     {:>8} ({:>5.1}%)",
                format_number(self.label_distribution["Down"]),
                100.0 * self.label_distribution["Down"] as f64 / total
            );
            println!(
                "  Stable:   {:>8} ({:>5.1}%)",
                format_number(self.label_distribution["Stable"]),
                100.0 * self.label_distribution["Stable"] as f64 / total
            );

            // Check balance
            let up_pct = 100.0 * self.label_distribution["Up"] as f64 / total;
            let down_pct = 100.0 * self.label_distribution["Down"] as f64 / total;
            let stable_pct = 100.0 * self.label_distribution["Stable"] as f64 / total;

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
        println!(
            "  Avg per day:      {:.1}s",
            self.total_time_secs / self.successful_days as f64
        );
        if self.total_messages > 0 {
            println!(
                "  Throughput:       {:.0} msgs/sec",
                self.total_messages as f64 / self.total_time_secs
            );
        }

        println!("\n{}", "=".repeat(80));
    }
}

/// Process a single day of data
fn process_day(day: &str, config: &ExportConfig, exporter: &BatchExporter) -> DayProcessingResult {
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
        "  📊 Processed {} messages → {} features, {} sequences",
        format_number(output.messages_processed),
        format_number(output.features_extracted),
        output.sequences_generated
    );

    // Export
    println!("  💾 Exporting...");
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
    // For simplicity, we'll manually list the 29 trading days
    // In production, this would parse dates and skip weekends/holidays
    let all_days = vec![
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
    ];

    all_days
        .iter()
        .filter(|&&d| d >= start && d <= end)
        .map(|&s| s.to_string())
        .collect()
}

/// Format large numbers with commas
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

fn main() {
    env_logger::init();

    println!("🚀 NVIDIA MBO Dataset Export");
    println!("{}", "=".repeat(80));

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let (start_day, end_day) = if args.len() > 2 {
        // Check for --day flag (single day)
        if args[1] == "--day" {
            (args[2].clone(), args[2].clone())
        }
        // Check for --start and --end flags
        else if args.len() > 4 && args[1] == "--start" && args[3] == "--end" {
            (args[2].clone(), args[4].clone())
        } else {
            eprintln!("Usage:");
            eprintln!(
                "  Single day:   cargo run --example export_nvidia_dataset -- --day 2025-02-03"
            );
            eprintln!("  Date range:   cargo run --example export_nvidia_dataset -- --start 2025-02-03 --end 2025-03-14");
            std::process::exit(1);
        }
    } else {
        // Default: first day only (for testing)
        ("2025-02-03".to_string(), "2025-02-03".to_string())
    };

    println!("\n📋 Configuration:");
    println!("  Date range:       {start_day} to {end_day}");

    // Create export configuration
    let config = ExportConfig::nvidia_hft();

    println!("  Data directory:   {}", config.data_dir.display());
    println!("  Output directory: {}", config.output_dir.display());
    println!("  Sampling:         Event-based (1000 events)");
    println!("  Features:         48 LOB features");
    println!("  Sequence:         100 timesteps, stride 10");
    println!("  Labeling:         TLOB (h=200, k=20, θ=0.0005)");

    // Create output directory
    std::fs::create_dir_all(&config.output_dir).expect("Failed to create output directory");

    // Create batch exporter
    let exporter = BatchExporter::new(&config.output_dir, Some(config.label_config.clone()));

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
    validate_exports(&config.output_dir, &days);

    println!("\n✅ Export complete!");
    println!("📁 Output: {}", config.output_dir.display());
}

/// Validate that all expected files were created
fn validate_exports(output_dir: &Path, days: &[String]) {
    let mut all_valid = true;

    for day in days {
        let features_path = output_dir.join(format!("{day}_features.npy"));
        let labels_path = output_dir.join(format!("{day}_labels.npy"));
        let metadata_path = output_dir.join(format!("{day}_metadata.json"));

        let features_ok = features_path.exists();
        let labels_ok = labels_path.exists();
        let metadata_ok = metadata_path.exists();

        if features_ok && labels_ok && metadata_ok {
            println!("  ✅ {day}: All files present");
        } else {
            println!("  ❌ {day}: Missing files");
            if !features_ok {
                println!("     - features.npy");
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
        println!("\n✅ All exports validated successfully!");
    } else {
        println!("\n⚠️  Some exports have missing files");
    }
}
