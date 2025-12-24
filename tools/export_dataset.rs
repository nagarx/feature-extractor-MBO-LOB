//! Multi-Symbol Dataset Export Tool (Aligned)
//!
//! Configuration-driven, symbol-agnostic tool for exporting **aligned** feature datasets.
//!
//! # Key Feature: Correct Alignment
//!
//! This tool uses `AlignedBatchExporter` to ensure **perfect 1:1 alignment** between
//! sequences and labels. Each exported sequence has exactly one corresponding label.
//!
//! ## Output Format
//!
//! - **Sequences**: `{day}_sequences.npy` - Shape `[N_sequences, window_size, n_features]`
//! - **Labels**: `{day}_labels.npy` - Shape `[N_sequences]` - One label per sequence
//! - **Metadata**: `{day}_metadata.json` - Comprehensive validation info
//! - **Normalization**: `{day}_normalization.json` - Market-structure preserving params
//!
//! # Features
//!
//! - **Configuration-driven**: Load from TOML or command-line
//! - **Symbol-agnostic**: Works for any instrument (NVDA, AAPL, MSFT, etc.)
//! - **Flexible feature sets**: 40, 48, 84, or 98 features
//! - **Parallel processing**: Multi-threaded batch processing
//! - **Train/Val/Test splits**: Automatic chronological splitting
//! - **Multi-horizon labels**: FI-2010, DeepLOB benchmark support (via config)
//! - **Tensor formatting**: DeepLOB, HLOB, Image formats (via config)
//!
//! # Usage
//!
//! ```bash
//! # From TOML config
//! cargo run --release --bin export_dataset -- --config configs/nvda.toml
//!
//! # Generate sample config
//! cargo run --release --bin export_dataset -- --generate-config nvda.toml
//! ```
//!
//! # Configuration
//!
//! See `export::DatasetConfig` for full configuration options.

use feature_extractor::batch::{BatchConfig, BatchProcessor, ConsoleProgress, ErrorMode};
use feature_extractor::export::{
    DataPathConfig, DatasetConfig, DateRangeConfig, ExperimentInfo,
    ExportLabelConfig, SymbolConfig,
};
use feature_extractor::AlignedBatchExporter;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Main entry point for the export tool
fn main() {
    // Simple argument parsing
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "--config" => {
            if args.len() < 3 {
                eprintln!("Error: --config requires a path argument");
                std::process::exit(1);
            }
            run_from_config(&args[2]);
        }
        "--generate-config" => {
            if args.len() < 3 {
                eprintln!("Error: --generate-config requires a path argument");
                std::process::exit(1);
            }
            generate_sample_config(&args[2]);
        }
        "--help" | "-h" => {
            print_usage(&args[0]);
        }
        _ => {
            eprintln!("Unknown argument: {}", args[1]);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }
}

fn print_usage(program: &str) {
    eprintln!(
        r#"
Multi-Symbol Dataset Export Tool

Usage:
    {program} --config <path.toml>       Export dataset from config file
    {program} --generate-config <path>   Generate sample config file
    {program} --help                     Show this help

Examples:
    # Export NVDA 98-feature dataset
    {program} --config configs/nvda_98feat.toml

    # Generate sample config
    {program} --generate-config configs/my_dataset.toml

For configuration options, see the generated sample config.
"#
    );
}

/// Generate a sample configuration file
fn generate_sample_config(path: &str) {
    // Check if user wants multi-horizon example
    let is_multi_horizon = path.contains("multi");

    let label_config = if is_multi_horizon {
        // Multi-horizon configuration (FI-2010 style)
        ExportLabelConfig::multi(vec![10, 20, 50, 100, 200], 10, 0.0008)
    } else {
        // Single-horizon configuration (backward compatible)
        ExportLabelConfig::single(50, 10, 0.0008)
    };

    let sample_config = DatasetConfig::new(
        SymbolConfig::nasdaq("NVDA"),
        DataPathConfig::new(
            "/path/to/data/NVDA_raw", // User should modify
            "/path/to/exports/nvda_98feat",
        ),
        DateRangeConfig::from_range("2025-02-03", "2025-09-29"),
    )
    .with_experiment(ExperimentInfo {
        name: if is_multi_horizon {
            "NVDA Multi-Horizon Dataset".to_string()
        } else {
            "NVDA 98-Feature Dataset".to_string()
        },
        description: Some(if is_multi_horizon {
            "Full features with multi-horizon labels: [10, 20, 50, 100, 200]".to_string()
        } else {
            "Full feature set: LOB + Derived + MBO + Signals".to_string()
        }),
        version: "1.0.0".to_string(),
        tags: if is_multi_horizon {
            vec!["nvda".to_string(), "multi-horizon".to_string()]
        } else {
            vec!["nvda".to_string(), "98-features".to_string()]
        },
    })
    .with_full_features()
    .with_labels(label_config);

    match sample_config.save_toml(path) {
        Ok(()) => {
            println!("✅ Generated sample config: {}", path);
            println!("\nEdit the following fields before running:");
            println!("  - data.input_dir: Path to raw MBO data files");
            println!("  - data.output_dir: Path for exported datasets");
            println!("  - dates.start_date/end_date: Date range to process");
            if is_multi_horizon {
                println!("\nMulti-horizon mode enabled:");
                println!("  - labels.horizons: Prediction horizons (modify as needed)");
                println!("  - Output labels shape: (N, num_horizons)");
            }
        }
        Err(e) => {
            eprintln!("Error generating config: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run export from configuration file
fn run_from_config(config_path: &str) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Multi-Symbol Dataset Export Tool                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load and validate configuration
    let config = match DatasetConfig::load_toml(config_path) {
        Ok(c) => {
            println!("✅ Loaded configuration: {}", config_path);
            c
        }
        Err(e) => {
            eprintln!("❌ Failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    // Print configuration summary
    print_config_summary(&config);

    // Validate configuration
    if let Err(e) = config.validate() {
        eprintln!("❌ Configuration validation failed: {}", e);
        std::process::exit(1);
    }
    println!("✅ Configuration validated");
    println!();

    // Run the export
    if let Err(e) = run_export(&config) {
        eprintln!("❌ Export failed: {}", e);
        std::process::exit(1);
    }
}

fn print_config_summary(config: &DatasetConfig) {
    println!("┌─ Configuration Summary ───────────────────────────────────────┐");
    println!("│ Experiment: {:<49} │", config.experiment.name);
    println!("│ Symbol:     {:<49} │", config.symbol.name);
    println!("│ Exchange:   {:<49} │", config.symbol.exchange);
    println!("│ Features:   {:<49} │", config.feature_count());
    println!("│");
    println!("│ Feature Configuration:");
    println!(
        "│   LOB Levels:    {}",
        config.features.lob_levels
    );
    println!(
        "│   Include Derived: {}",
        config.features.include_derived
    );
    println!(
        "│   Include MBO:     {}",
        config.features.include_mbo
    );
    println!(
        "│   Include Signals: {}",
        config.features.include_signals
    );
    println!("│");
    println!("│ Label Configuration:");
    if config.labels.is_multi_horizon() {
        println!(
            "│   Mode:           Multi-horizon ({} horizons)",
            config.labels.horizons.len()
        );
        println!(
            "│   Horizons:       {:?}",
            config.labels.horizons
        );
    } else {
        println!(
            "│   Mode:           Single-horizon"
        );
        println!(
            "│   Horizon:        {}",
            config.labels.horizon
        );
    }
    println!(
        "│   Smoothing:      {}",
        config.labels.smoothing_window
    );
    println!(
        "│   Threshold:      {} ({:.1} bps)",
        config.labels.threshold,
        config.labels.threshold * 10000.0
    );
    println!("│");
    println!("│ Input:      {}", config.data.input_dir.display());
    println!("│ Output:     {}", config.data.output_dir.display());
    println!("└────────────────────────────────────────────────────────────────┘");
    println!();
}

/// Run the export process
fn run_export(config: &DatasetConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Get dates to process
    let dates = config.dates.get_dates()?;
    println!("📅 Processing {} trading days", dates.len());

    // Split into train/val/test
    let (train_dates, val_dates, test_dates) = config.split.split_days(&dates);
    println!(
        "   Train: {} days, Val: {} days, Test: {} days",
        train_dates.len(),
        val_dates.len(),
        test_dates.len()
    );
    println!();

    // Create output directory
    fs::create_dir_all(&config.data.output_dir)?;

    // Get pipeline configuration
    let pipeline_config = Arc::new(config.to_pipeline_config());

    // Determine number of threads
    let num_threads = config
        .processing
        .threads
        .unwrap_or_else(|| num_cpus::get().saturating_sub(2).max(1));

    // Determine error mode
    let error_mode = if config.processing.error_mode == "fail_fast" {
        ErrorMode::FailFast
    } else {
        ErrorMode::CollectErrors
    };

    let batch_config = BatchConfig {
        num_threads: Some(num_threads),
        error_mode,
        report_progress: config.processing.verbose,
        stack_size: None,
        hot_store_dir: config.data.hot_store_dir.clone(),
    };

    println!("🚀 Starting export (using {} threads)...", num_threads);
    println!();

    // Process each split
    let splits = [
        ("train", &train_dates[..]),
        ("val", &val_dates[..]),
        ("test", &test_dates[..]),
    ];

    let mut total_days_processed = 0;
    let mut total_sequences_exported = 0;

    for (split_name, split_dates) in splits.iter() {
        if split_dates.is_empty() {
            println!("⚠️  Skipping empty {} split", split_name);
            continue;
        }

        println!(
            "━━━ Processing {} split ({} days) ━━━",
            split_name,
            split_dates.len()
        );

        let split_output_dir = config.data.output_dir.join(split_name);
        fs::create_dir_all(&split_output_dir)?;

        // Build file paths for this split
        let files: Vec<std::path::PathBuf> = split_dates
            .iter()
            .map(|date| {
                // Check hot store first
                if let Some(ref hot_store) = config.data.hot_store_dir {
                    let hot_filename = config.symbol.filename_for_date(date).replace(".zst", "");
                    let hot_path = hot_store.join(&hot_filename);
                    if hot_path.exists() {
                        return hot_path;
                    }
                }
                // Fall back to input dir
                config.file_path_for_date(date)
            })
            .collect();

        // Check which files exist
        let existing_files: Vec<&std::path::PathBuf> =
            files.iter().filter(|f| f.exists()).collect();

        if existing_files.is_empty() {
            println!("  ⚠️  No data files found for {} split", split_name);
            continue;
        }

        println!(
            "  📁 Found {} of {} data files",
            existing_files.len(),
            files.len()
        );

        // Create batch processor with progress callback
        let processor = BatchProcessor::new((*pipeline_config).clone(), batch_config.clone());
        let processor = if config.processing.verbose {
            processor.with_progress_callback(Box::new(ConsoleProgress::new().verbose()))
        } else {
            // Always show basic progress (file count updates)
            processor.with_progress_callback(Box::new(ConsoleProgress::new()))
        };

        // Process files
        let file_refs: Vec<&Path> = existing_files.iter().map(|p| p.as_path()).collect();
        let output = processor.process_files(&file_refs)?;

        // Verify all files were processed (critical for financial data integrity)
        let processed_count = output.successful_count();
        let expected_count = existing_files.len();
        if processed_count != expected_count {
            eprintln!(
                "⚠️  WARNING: Processed {} files but expected {}. {} errors occurred.",
                processed_count,
                expected_count,
                output.failed_count()
            );
            for err in output.iter_errors() {
                eprintln!("    ❌ {}: {}", err.file_path, err.error);
            }
        }

        // Export results in chronological order for determinism
        // Using AlignedBatchExporter for correct 1:1 sequence-label alignment
        let base_exporter = AlignedBatchExporter::new(
            &split_output_dir,
            config.labels.to_label_config(),
            pipeline_config.sequence.window_size,
            pipeline_config.sequence.stride,
        );

        // Enable multi-horizon labeling if configured
        let exporter = if let Some(multi_config) = config.labels.to_multi_horizon_config() {
            println!(
                "    📊 Multi-horizon mode: {:?}",
                multi_config.horizons()
            );
            base_exporter.with_multi_horizon_labels(multi_config)
        } else {
            base_exporter
        };

        let mut split_sequences = 0;
        let mut days_exported = Vec::new();

        for day_result in output.results_by_day() {
            let day_name = &day_result.day;
            let result = exporter.export_day(day_name, &day_result.output)?;
            // AlignedDayExport has n_sequences (1:1 with labels)
            split_sequences += result.n_sequences;
            days_exported.push(day_name.clone());
            total_days_processed += 1;
        }

        // Verify exported days match expected (paranoia check for financial accuracy)
        if days_exported.len() != processed_count {
            return Err(format!(
                "Export mismatch: processed {} but exported {}",
                processed_count,
                days_exported.len()
            )
            .into());
        }

        // For aligned export: sequences == labels (1:1)
        total_sequences_exported += split_sequences;

        println!(
            "  ✅ {} complete: {} sequences (aligned 1:1 with labels)",
            split_name, split_sequences
        );
        println!();
    }

    // Save manifest
    save_manifest(config, total_days_processed)?;

    // Print summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Aligned Export Complete                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Days processed: {:<44} ║",
        total_days_processed
    );
    println!(
        "║  Sequences exported: {:<40} ║",
        total_sequences_exported
    );
    if config.labels.is_multi_horizon() {
        let horizons = &config.labels.horizons;
        println!(
            "║  Label mode: {:<48} ║",
            format!("Multi-horizon ({} horizons)", horizons.len())
        );
        println!(
            "║  Horizons: {:<50} ║",
            format!("{:?}", horizons).chars().take(50).collect::<String>()
        );
        println!(
            "║  Labels shape: {:<46} ║",
            format!("[N, {}]", horizons.len())
        );
    } else {
        println!(
            "║  Label mode: {:<48} ║",
            format!("Single-horizon (h={})", config.labels.horizon)
        );
        println!(
            "║  Labels shape: {:<46} ║",
            "[N]"
        );
    }
    println!(
        "║  Alignment: {:<49} ║",
        "1:1 (sequence ↔ label)"
    );
    println!(
        "║  Sequences format: {:<42} ║",
        format!("[N, {}, {}]", pipeline_config.sequence.window_size, config.feature_count())
    );
    println!(
        "║  Output directory: {:<42} ║",
        config.data.output_dir.display().to_string().chars().take(42).collect::<String>()
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Save dataset manifest for reproducibility
fn save_manifest(
    config: &DatasetConfig,
    days_processed: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(serde::Serialize)]
    struct DatasetManifest {
        experiment: ExperimentInfo,
        symbol: String,
        feature_count: usize,
        days_processed: usize,
        export_timestamp: String,
        config_hash: String,
    }

    let manifest = DatasetManifest {
        experiment: config.experiment.clone(),
        symbol: config.symbol.name.clone(),
        feature_count: config.feature_count(),
        days_processed,
        export_timestamp: chrono::Utc::now().to_rfc3339(),
        config_hash: format!(
            "{:x}",
            md5::compute(serde_json::to_string(config)?)
        ),
    };

    let manifest_path = config.data.output_dir.join("dataset_manifest.json");
    let file = fs::File::create(&manifest_path)?;
    serde_json::to_writer_pretty(file, &manifest)?;

    // Also save the full config for reproducibility
    let config_copy_path = config.data.output_dir.join("export_config.toml");
    config.save_toml(&config_copy_path)?;

    println!(
        "📋 Saved manifest: {}",
        manifest_path.display()
    );

    Ok(())
}

