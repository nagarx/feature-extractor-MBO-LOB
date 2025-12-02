//! Test and validate the sequence builder with real market data.
//!
//! This example demonstrates the complete pipeline:
//! 1. Load DBN data
//! 2. Reconstruct LOB state
//! 3. Extract features
//! 4. Sample with volume-based strategy
//! 5. Build sequences for transformer input
//!
//! Usage:
//! ```bash
//! cargo run --release --example test_sequence_builder \
//!     data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250203.mbo.dbn.zst
//! ```

use std::env;
use std::path::Path;
use std::time::Instant;

use feature_extractor::{
    FeatureConfig, FeatureExtractor, Sequence, SequenceBuilder, SequenceConfig, VolumeBasedSampler,
};
use mbo_lob_reconstructor::{DbnLoader, LobReconstructor};

/// Statistics for the sequence building pipeline
struct PipelineStats {
    messages_processed: u64,
    features_extracted: u64,
    sequences_generated: u64,

    // Timing
    total_duration_secs: f64,

    // Sequence statistics
    avg_sequence_duration_secs: f64,
    avg_snapshots_per_second: f64,

    // Memory
    max_buffer_size: usize,
    final_buffer_utilization: f64,
}

impl PipelineStats {
    fn print_report(&self) {
        println!("\n{:=<50}", "");
        println!("=== Sequence Builder Validation Report ===");

        println!("\n📊 Processing Statistics:");
        println!(
            "  Messages processed:     {:>12}",
            format_num(self.messages_processed)
        );
        println!(
            "  Features extracted:     {:>12}",
            format_num(self.features_extracted)
        );
        println!(
            "  Sequences generated:    {:>12}",
            format_num(self.sequences_generated)
        );

        println!("\n⚡ Performance:");
        println!(
            "  Total duration:         {:>12.2} s",
            self.total_duration_secs
        );
        println!(
            "  Messages/sec:           {:>12.0}",
            self.messages_processed as f64 / self.total_duration_secs
        );
        println!(
            "  Sequences/sec:          {:>12.0}",
            self.sequences_generated as f64 / self.total_duration_secs
        );

        println!("\n📏 Sequence Characteristics:");
        println!(
            "  Avg sequence duration:  {:>12.2} s",
            self.avg_sequence_duration_secs
        );
        println!(
            "  Avg snapshots/sec:      {:>12.2}",
            self.avg_snapshots_per_second
        );

        println!("\n💾 Memory Usage:");
        println!(
            "  Max buffer size:        {:>12}",
            format_num(self.max_buffer_size as u64)
        );
        println!(
            "  Final utilization:      {:>12.1}%",
            self.final_buffer_utilization
        );

        println!("\n{:=<50}", "");
    }
}

fn format_num(n: u64) -> String {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <path-to-dbn-file>", args[0]);
        eprintln!("\nExample:");
        eprintln!(
            "  {} data/NVDA_2025-02-01_to_2025-09-30/xnas-itch-20250203.mbo.dbn.zst",
            args[0]
        );
        std::process::exit(1);
    }

    let dbn_path = Path::new(&args[1]);

    println!("🚀 Sequence Builder Validation Test");
    println!("====================================\n");
    println!("📁 Input file: {}", dbn_path.display());

    // Run the full pipeline test
    let stats = run_pipeline_test(dbn_path)?;

    // Print results
    stats.print_report();

    // Validation checks
    println!("\n✅ Validation Checks:");

    if stats.sequences_generated > 0 {
        println!("  ✓ Successfully generated sequences");
    } else {
        println!("  ✗ WARNING: No sequences generated!");
    }

    if stats.avg_sequence_duration_secs > 0.0 && stats.avg_sequence_duration_secs < 3600.0 {
        println!(
            "  ✓ Sequence durations are reasonable ({:.2}s avg)",
            stats.avg_sequence_duration_secs
        );
    } else {
        println!("  ⚠ WARNING: Unusual sequence durations");
    }

    if stats.final_buffer_utilization < 100.0 {
        println!(
            "  ✓ Buffer did not overflow ({:.1}% utilization)",
            stats.final_buffer_utilization
        );
    } else {
        println!("  ℹ Buffer reached capacity (expected for long data)");
    }

    let efficiency = (stats.sequences_generated as f64 / stats.features_extracted as f64) * 100.0;
    println!("  ℹ Sequence generation efficiency: {efficiency:.1}% (sequences/features)");

    println!("\n🎉 Sequence builder validation COMPLETE!");

    Ok(())
}

fn run_pipeline_test(dbn_path: &Path) -> Result<PipelineStats, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Initialize components
    println!("\n🔧 Initializing pipeline components...");

    // 1. LOB Reconstructor
    let mut reconstructor = LobReconstructor::new(10);
    println!("  ✓ LOB Reconstructor initialized");

    // 2. Feature Extractor (LOB features only for simplicity)
    let feature_config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: false,
        mbo_window_size: 1000,
    };
    let feature_extractor = FeatureExtractor::with_config(feature_config);
    println!("  ✓ Feature Extractor initialized (48 LOB features)");

    // 3. Volume-Based Sampler
    // Target: Sample every 10 shares traded, min 100μs between samples
    // Very aggressive sampling for testing - generates sequences quickly
    let mut sampler = VolumeBasedSampler::new(
        10,      // 10 shares per sample (very aggressive for testing)
        100_000, // 100μs min interval (nanoseconds)
    );
    println!("  ✓ Volume Sampler initialized (10 shares/sample, 100μs min)");

    // 4. Sequence Builder
    // For testing: smaller window (50 snapshots) to generate sequences faster
    // Production would use 100 snapshots per TLOB paper
    let sequence_config = SequenceConfig::new(50, 1) // 50 snapshots instead of 100
        .with_feature_count(48) // LOB features only
        .with_max_buffer_size(1000);

    let mut sequence_builder = SequenceBuilder::with_config(sequence_config);
    println!("  ✓ Sequence Builder initialized (50 snapshots/seq, stride 1, 1000 buffer)");

    // 5. DBN Loader
    let loader = DbnLoader::new(dbn_path.to_str().unwrap())?.skip_invalid(true);
    println!("  ✓ DBN Loader initialized\n");

    // Process data
    println!("📈 Processing market data...\n");

    let mut messages_processed = 0u64;
    let mut features_extracted = 0u64;
    let mut sequences: Vec<Sequence> = Vec::new();

    let mut last_report = Instant::now();

    for msg in loader.iter_messages()? {
        // Skip invalid messages
        if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
            continue;
        }

        messages_processed += 1;

        // 1. Update LOB state
        if reconstructor.process_message(&msg).is_err() {
            continue;
        }

        let lob_state = reconstructor.get_lob_state();

        // Skip if state is invalid
        if !lob_state.is_valid() {
            continue;
        }

        // 2. Check if we should sample
        let volume = msg.size;
        let timestamp = msg.timestamp.unwrap_or(0) as u64;

        if !sampler.should_sample(volume, timestamp) {
            continue;
        }

        // 3. Extract features
        let features = feature_extractor.extract_lob_features(&lob_state)?;
        features_extracted += 1;

        // 4. Push to sequence builder
        sequence_builder.push(timestamp, features);

        // 5. Try to build a sequence
        if let Some(seq) = sequence_builder.try_build_sequence() {
            sequences.push(seq);
        }

        // Progress reporting every 5 seconds
        if last_report.elapsed().as_secs() >= 5 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let msgs_per_sec = messages_processed as f64 / elapsed;
            let seqs_generated = sequences.len();

            println!(
                "  Progress: {} msgs | {} features | {} sequences | {:.0} msg/s",
                format_num(messages_processed),
                format_num(features_extracted),
                format_num(seqs_generated as u64),
                msgs_per_sec
            );

            last_report = Instant::now();
        }
    }

    // Final report
    let total_duration = start_time.elapsed().as_secs_f64();

    println!("\n✅ Processing complete!");
    println!("  Total messages: {}", format_num(messages_processed));
    println!("  Total features: {}", format_num(features_extracted));
    println!("  Total sequences: {}", format_num(sequences.len() as u64));

    // Calculate statistics
    let avg_sequence_duration = if !sequences.is_empty() {
        sequences.iter().map(|s| s.duration_seconds()).sum::<f64>() / sequences.len() as f64
    } else {
        0.0
    };

    let avg_snapshots_per_second = if !sequences.is_empty() {
        sequences
            .iter()
            .map(|s| s.length as f64 / s.duration_seconds())
            .sum::<f64>()
            / sequences.len() as f64
    } else {
        0.0
    };

    let (_buf_len, _, _, buf_util) = sequence_builder.statistics();

    Ok(PipelineStats {
        messages_processed,
        features_extracted,
        sequences_generated: sequences.len() as u64,
        total_duration_secs: total_duration,
        avg_sequence_duration_secs: avg_sequence_duration,
        avg_snapshots_per_second,
        max_buffer_size: sequence_builder.config().max_buffer_size,
        final_buffer_utilization: buf_util,
    })
}
