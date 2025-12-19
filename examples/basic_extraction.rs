//! Configuration serialization example.
//!
//! Demonstrates how to create, save, and load pipeline configurations
//! for reproducible experiments.
//!
//! Usage:
//! ```bash
//! cargo run --example config_example
//! ```

use feature_extractor::config::{
    ExperimentMetadata, PipelineConfig, SamplingConfig, SamplingStrategy,
};
use feature_extractor::{FeatureConfig, SequenceConfig};

fn main() {
    println!("🔧 Pipeline Configuration Example\n");
    println!("{}", "=".repeat(70));

    // =========================================================================
    // 1. Create Default Configuration
    // =========================================================================
    println!("\n1️⃣  Creating default configuration:\n");

    let config = PipelineConfig::default();
    println!("  Features:");
    println!("    - LOB levels: {}", config.features.lob_levels);
    println!("    - Tick size: ${:.4}", config.features.tick_size);
    println!("    - Include MBO: {}", config.features.include_mbo);

    println!("\n  Sequence:");
    println!(
        "    - Window size: {} snapshots",
        config.sequence.window_size
    );
    println!("    - Stride: {}", config.sequence.stride);
    println!("    - Feature count: {}", config.sequence.feature_count);

    if let Some(sampling) = &config.sampling {
        println!("\n  Sampling:");
        println!("    - Strategy: {:?}", sampling.strategy);
        if let Some(vol) = sampling.volume_threshold {
            println!("    - Volume threshold: {vol} shares");
        }
    }

    // =========================================================================
    // 2. Custom Configuration
    // =========================================================================
    println!("\n\n2️⃣  Creating custom configuration:\n");

    let custom_config = PipelineConfig::new()
        .with_features(FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true, // Enable MBO features
            mbo_window_size: 1000,
            include_signals: false,
        })
        .with_sequence(
            SequenceConfig::new(100, 1)
                .with_feature_count(84) // 48 LOB + 36 MBO = 84
                .with_max_buffer_size(2000),
        )
        .with_sampling(SamplingConfig {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: Some(10_000),
            min_time_interval_ns: Some(100_000_000), // 100ms
            event_count: None,
            adaptive: None,
            multiscale: None,
        })
        .with_metadata(ExperimentMetadata {
            name: "NVDA_TLOB_Experiment_1".to_string(),
            description: Some("Volume-based sampling with MBO features".to_string()),
            created_at: Some("2025-10-23T12:00:00Z".to_string()),
            version: Some("0.1.0".to_string()),
            tags: Some(vec![
                "nvda".to_string(),
                "mbo".to_string(),
                "volume-sampling".to_string(),
            ]),
        });

    println!("  Created custom configuration:");
    println!(
        "    - Name: {}",
        custom_config.metadata.as_ref().unwrap().name
    );
    println!("    - MBO enabled: {}", custom_config.features.include_mbo);
    println!(
        "    - Feature count: {}",
        custom_config.sequence.feature_count
    );
    println!(
        "    - Volume threshold: {} shares",
        custom_config
            .sampling
            .as_ref()
            .unwrap()
            .volume_threshold
            .unwrap()
    );

    // Validate
    match custom_config.validate() {
        Ok(()) => println!("\n  ✅ Configuration is valid!"),
        Err(e) => println!("\n  ❌ Configuration error: {e}"),
    }

    // =========================================================================
    // 3. Save to TOML
    // =========================================================================
    println!("\n\n3️⃣  Saving configuration to TOML:\n");

    let toml_path = "example_pipeline_config.toml";
    match custom_config.save_toml(toml_path) {
        Ok(()) => println!("  ✅ Saved to: {toml_path}"),
        Err(e) => println!("  ❌ Failed to save: {e}"),
    }

    // =========================================================================
    // 4. Save to JSON
    // =========================================================================
    println!("\n4️⃣  Saving configuration to JSON:\n");

    let json_path = "example_pipeline_config.json";
    match custom_config.save_json(json_path) {
        Ok(()) => println!("  ✅ Saved to: {json_path}"),
        Err(e) => println!("  ❌ Failed to save: {e}"),
    }

    // =========================================================================
    // 5. Load from TOML
    // =========================================================================
    println!("\n\n5️⃣  Loading configuration from TOML:\n");

    match PipelineConfig::load_toml(toml_path) {
        Ok(loaded) => {
            println!("  ✅ Loaded successfully!");
            println!(
                "    - Experiment: {}",
                loaded.metadata.as_ref().unwrap().name
            );
            println!("    - Window size: {}", loaded.sequence.window_size);
            println!("    - MBO enabled: {}", loaded.features.include_mbo);
        }
        Err(e) => println!("  ❌ Failed to load: {e}"),
    }

    // =========================================================================
    // 6. Load from JSON
    // =========================================================================
    println!("\n6️⃣  Loading configuration from JSON:\n");

    match PipelineConfig::load_json(json_path) {
        Ok(loaded) => {
            println!("  ✅ Loaded successfully!");
            println!(
                "    - Experiment: {}",
                loaded.metadata.as_ref().unwrap().name
            );
        }
        Err(e) => println!("  ❌ Failed to load: {e}"),
    }

    // =========================================================================
    // 7. Validation Example
    // =========================================================================
    println!("\n\n7️⃣  Configuration validation:\n");

    // Create invalid config (feature count mismatch)
    let mut invalid_config = PipelineConfig::default();
    invalid_config.features.include_mbo = true; // Enable MBO
    invalid_config.sequence.feature_count = 48; // Wrong! Should be 84

    match invalid_config.validate() {
        Ok(()) => println!("  ❌ Unexpected: invalid config passed validation"),
        Err(e) => println!("  ✅ Correctly caught error: {e}"),
    }

    // =========================================================================
    // 8. Different Sampling Strategies
    // =========================================================================
    println!("\n\n8️⃣  Different sampling strategies:\n");

    // Volume-based
    let volume_sampling = SamplingConfig {
        strategy: SamplingStrategy::VolumeBased,
        volume_threshold: Some(5000),
        min_time_interval_ns: Some(10_000_000),
        event_count: None,
        adaptive: None,
        multiscale: None,
    };
    println!(
        "  Volume-based: {} shares/sample",
        volume_sampling.volume_threshold.unwrap()
    );

    // Event-based
    let event_sampling = SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        volume_threshold: None,
        min_time_interval_ns: None,
        event_count: Some(100),
        adaptive: None,
        multiscale: None,
    };
    println!(
        "  Event-based: every {} events",
        event_sampling.event_count.unwrap()
    );

    // Time-based
    let time_sampling = SamplingConfig {
        strategy: SamplingStrategy::TimeBased,
        volume_threshold: None,
        min_time_interval_ns: Some(1_000_000_000), // 1 second
        event_count: None,
        adaptive: None,
        multiscale: None,
    };
    println!(
        "  Time-based: every {} ms",
        time_sampling.min_time_interval_ns.unwrap() / 1_000_000
    );

    // =========================================================================
    // Cleanup
    // =========================================================================
    println!("\n\n🧹 Cleaning up example files...");
    std::fs::remove_file(toml_path).ok();
    std::fs::remove_file(json_path).ok();
    println!("  ✅ Done\n");

    println!("{}", "=".repeat(70));
    println!("\n✅ Configuration example complete!\n");
    println!("💡 Key Benefits:");
    println!("   • Reproducible experiments (version control configs)");
    println!("   • Easy parameter sweeps (load/modify/save)");
    println!("   • Configuration validation before running");
    println!("   • Human-readable TOML or machine-readable JSON");
    println!("   • Experiment metadata tracking\n");
}
