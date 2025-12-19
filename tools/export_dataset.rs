//! Multi-Symbol Dataset Export Tool
//!
//! Configuration-driven, symbol-agnostic tool for exporting feature datasets.
//!
//! # Features
//!
//! - **Configuration-driven**: Load from TOML or command-line
//! - **Symbol-agnostic**: Works for any instrument
//! - **Flexible feature sets**: 40, 48, 84, or 98 features
//! - **Parallel processing**: Multi-threaded batch processing
//! - **Train/Val/Test splits**: Automatic chronological splitting
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

use feature_extractor::batch::{BatchConfig, BatchProcessor, ErrorMode};
use feature_extractor::export::{
    BatchExporter, DataPathConfig, DatasetConfig, DateRangeConfig, ExperimentInfo,
    ExportLabelConfig, SymbolConfig,
};
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
    let sample_config = DatasetConfig::new(
        SymbolConfig::nasdaq("NVDA"),
        DataPathConfig::new(
            "/path/to/data/NVDA_raw", // User should modify
            "/path/to/exports/nvda_98feat",
        ),
        DateRangeConfig::from_range("2025-02-03", "2025-09-29"),
    )
    .with_experiment(ExperimentInfo {
        name: "NVDA 98-Feature Dataset".to_string(),
        description: Some("Full feature set: LOB + Derived + MBO + Signals".to_string()),
        version: "1.0.0".to_string(),
        tags: vec!["nvda".to_string(), "98-features".to_string()],
    })
    .with_full_features()
    .with_labels(ExportLabelConfig {
        horizon: 50,
        smoothing_window: 10,
        threshold: 0.0008,
    });

    match sample_config.save_toml(path) {
        Ok(()) => {
            println!("✅ Generated sample config: {}", path);
            println!("\nEdit the following fields before running:");
            println!("  - data.input_dir: Path to raw MBO data files");
            println!("  - data.output_dir: Path for exported datasets");
            println!("  - dates.start_date/end_date: Date range to process");
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
    let mut total_features_exported = 0;
    let mut total_labels_exported = 0;

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

        // Create batch processor
        let processor = BatchProcessor::new((*pipeline_config).clone(), batch_config.clone());

        // Process files
        let file_refs: Vec<&Path> = existing_files.iter().map(|p| p.as_path()).collect();
        let output = processor.process_files(&file_refs)?;

        // Export results
        let exporter = BatchExporter::new(&split_output_dir, Some(config.labels.to_label_config()));

        let mut split_features = 0;
        let mut split_labels = 0;

        for day_result in &output.results {
            let day_name = &day_result.day;
            let result = exporter.export_day(day_name, &day_result.output)?;
            split_features += result.n_features;
            split_labels += result.n_labels;
            total_days_processed += 1;
        }

        total_features_exported += split_features;
        total_labels_exported += split_labels;

        println!(
            "  ✅ {} complete: {} features, {} labels",
            split_name, split_features, split_labels
        );
        println!();
    }

    // Save manifest
    save_manifest(config, total_days_processed)?;

    // Print summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     Export Complete                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Days processed: {:<44} ║",
        total_days_processed
    );
    println!(
        "║  Features exported: {:<41} ║",
        total_features_exported
    );
    println!(
        "║  Labels exported: {:<43} ║",
        total_labels_exported
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

