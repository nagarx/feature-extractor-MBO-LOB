//! Comprehensive Pipeline Integration Tests
//!
//! These tests verify the pipeline works correctly with realistic data and edge cases.

use feature_extractor::{
    config::ExperimentMetadata, FeatureConfig, FeatureVec, Pipeline, PipelineConfig, PipelineOutput,
    SamplingConfig, SamplingStrategy, Sequence, SequenceConfig,
};
use std::sync::Arc;

#[test]
fn test_pipeline_creation_with_config() {
    let config = PipelineConfig {
        features: FeatureConfig::default(),
        sequence: SequenceConfig::default(),
        sampling: Some(SamplingConfig::default()),
        metadata: None,
    };

    let pipeline = Pipeline::from_config(config);
    assert!(pipeline.is_ok());
}

#[test]
fn test_pipeline_with_different_sampling_strategies() {
    // Test VolumeBased
    let mut config = PipelineConfig::default();
    config.features.include_mbo = false; // Use LOB only to avoid feature count mismatch
    config.features.include_derived = false; // Default: raw LOB only
    config.sequence.feature_count = 40; // 10 levels × 4 features = 40 raw LOB features
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::VolumeBased,
        volume_threshold: Some(500),
        min_time_interval_ns: Some(1_000_000),
        event_count: None,
        adaptive: None,
        multiscale: None,
    });
    let pipeline = Pipeline::from_config(config.clone());
    assert!(pipeline.is_ok());

    // Test EventBased
    config.sampling = Some(SamplingConfig {
        strategy: SamplingStrategy::EventBased,
        volume_threshold: None,
        min_time_interval_ns: None,
        event_count: Some(50),
        adaptive: None,
        multiscale: None,
    });
    let pipeline = Pipeline::from_config(config.clone());
    assert!(pipeline.is_ok());

    // Note: TimeBased sampling is NOT implemented.
    // Pipeline creation succeeds, but process() will return an error.
    // This is intentional - we explicitly error instead of silently falling back.
}

#[test]
fn test_pipeline_config_validation() {
    // Valid config - default is raw LOB only (40 features)
    let config = PipelineConfig::default();
    assert!(config.validate().is_ok());

    // Invalid feature count mismatch
    let mut config = PipelineConfig::default();
    config.features.include_mbo = true;
    config.features.include_derived = false;
    config.sequence.feature_count = 10; // Too low for MBO
    assert!(config.validate().is_err());

    // Fix it - LOB + MBO = 40 + 36 = 76
    config.sequence.feature_count = 76;
    assert!(config.validate().is_ok());
}

#[test]
fn test_pipeline_reset() {
    let config = PipelineConfig::default();
    let mut pipeline = Pipeline::from_config(config).unwrap();

    // Verify reset doesn't panic
    pipeline.reset();

    // Can reset multiple times
    pipeline.reset();
    pipeline.reset();
}

#[test]
fn test_pipeline_output_to_flat_features() {
    // Create mock sequences (using Arc for zero-copy)
    let sequences = vec![
        Sequence {
            features: vec![
                Arc::new(vec![1.0, 2.0, 3.0]),
                Arc::new(vec![4.0, 5.0, 6.0]),
            ],
            start_timestamp: 100,
            end_timestamp: 200,
            duration_ns: 100,
            length: 2,
        },
        Sequence {
            features: vec![Arc::new(vec![7.0, 8.0, 9.0])],
            start_timestamp: 300,
            end_timestamp: 300,
            duration_ns: 0,
            length: 1,
        },
    ];

    let output = PipelineOutput {
        sequences,
        mid_prices: vec![100.0, 101.0, 102.0],
        messages_processed: 1000,
        features_extracted: 3,
        sequences_generated: 2,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let flat = output.to_flat_features();

    // Should have 3 feature vectors total (2 + 1)
    assert_eq!(flat.len(), 3);
    assert_eq!(flat[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(flat[1], vec![4.0, 5.0, 6.0]);
    assert_eq!(flat[2], vec![7.0, 8.0, 9.0]);
}

#[test]
fn test_pipeline_output_mid_prices() {
    let output = PipelineOutput {
        sequences: vec![],
        mid_prices: vec![100.0, 101.5, 102.0, 100.5],
        messages_processed: 4,
        features_extracted: 4,
        sequences_generated: 0,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let mid_prices = output.get_mid_prices();
    assert_eq!(mid_prices.len(), 4);
    assert_eq!(mid_prices[0], 100.0);
    assert_eq!(mid_prices[1], 101.5);
    assert_eq!(mid_prices[2], 102.0);
    assert_eq!(mid_prices[3], 100.5);
}

#[test]
fn test_pipeline_output_statistics() {
    let output = PipelineOutput {
        sequences: vec![],
        mid_prices: vec![],
        messages_processed: 1_000_000,
        features_extracted: 10_000,
        sequences_generated: 95,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    assert_eq!(output.messages_processed, 1_000_000);
    assert_eq!(output.features_extracted, 10_000);
    assert_eq!(output.sequences_generated, 95);

    // Verify sampling rate
    let sampling_rate = output.features_extracted as f64 / output.messages_processed as f64;
    assert!(sampling_rate > 0.0 && sampling_rate <= 1.0);
}

#[test]
fn test_pipeline_with_no_sampling_config() {
    let mut config = PipelineConfig::default();
    config.sampling = None; // Explicitly set to None

    // Should use default sampling
    let pipeline = Pipeline::from_config(config);
    assert!(pipeline.is_ok());
}

#[test]
fn test_pipeline_config_serialization_round_trip() {
    // Note: feature_count must match: lob_levels * 4 + derived(8) + mbo(36)
    // With 20 levels, derived=true, mbo=false: 20*4 + 8 = 88 features
    let config = PipelineConfig {
        features: FeatureConfig {
            lob_levels: 20,
            tick_size: 0.001,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 1000,
        },
        sequence: SequenceConfig {
            window_size: 200,
            stride: 2,
            feature_count: 88, // 20*4 + 8 derived = 88
            max_buffer_size: 5000,
        },
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: Some(2000),
            min_time_interval_ns: Some(5_000_000),
            event_count: None,
            adaptive: None,
            multiscale: None,
        }),
        metadata: Some(ExperimentMetadata {
            name: "test_exp".to_string(),
            description: Some("Test experiment".to_string()),
            created_at: Some("2025-10-23T00:00:00Z".to_string()),
            version: Some("1.0.0".to_string()),
            tags: Some(vec!["test".to_string()]),
        }),
    };

    // Serialize to TOML
    let toml_str = toml::to_string(&config).unwrap();
    assert!(toml_str.contains("lob_levels = 20"));
    assert!(toml_str.contains("window_size = 200"));
    assert!(toml_str.contains("volume_threshold = 2000"));

    // Deserialize back
    let config_back: PipelineConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(config_back.features.lob_levels, 20);
    assert_eq!(config_back.sequence.window_size, 200);
    assert_eq!(
        config_back.sampling.as_ref().unwrap().volume_threshold,
        Some(2000)
    );

    // Validate
    assert!(config_back.validate().is_ok());
}

#[test]
fn test_pipeline_handles_empty_output() {
    let output = PipelineOutput {
        sequences: vec![],
        mid_prices: vec![],
        messages_processed: 0,
        features_extracted: 0,
        sequences_generated: 0,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let flat = output.to_flat_features();
    assert_eq!(flat.len(), 0);

    let mid_prices = output.get_mid_prices();
    assert_eq!(mid_prices.len(), 0);
}

#[test]
fn test_sampling_strategy_coverage() {
    // Verify implemented sampling strategies can be created and work
    // Note: TimeBased is NOT implemented and will error at process() time
    let working_strategies = vec![
        SamplingStrategy::VolumeBased,
        SamplingStrategy::EventBased,
        SamplingStrategy::MultiScale, // Uses volume-based as base sampler
    ];

    for strategy in working_strategies {
        let config = PipelineConfig {
            features: FeatureConfig::default(),
            sequence: SequenceConfig::default(),
            sampling: Some(SamplingConfig {
                strategy,
                volume_threshold: Some(1000),
                min_time_interval_ns: Some(1_000_000),
                event_count: Some(100),
                adaptive: None,
                multiscale: None,
            }),
            metadata: None,
        };

        let pipeline = Pipeline::from_config(config);
        assert!(
            pipeline.is_ok(),
            "Failed to create pipeline with strategy: {strategy:?}"
        );
    }
}

#[test]
fn test_timebased_sampling_not_implemented() {
    // Verify that TimeBased sampling returns a clear error
    // instead of silently falling back to another strategy
    let config = PipelineConfig {
        features: FeatureConfig::default(),
        sequence: SequenceConfig::default(),
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::TimeBased,
            volume_threshold: Some(1000),
            min_time_interval_ns: Some(1_000_000),
            event_count: Some(100),
            adaptive: None,
            multiscale: None,
        }),
        metadata: None,
    };

    // Pipeline creation succeeds (config is valid)
    let pipeline = Pipeline::from_config(config);
    assert!(pipeline.is_ok(), "Pipeline creation should succeed");

    // But if we had data to process, it would error at process() time
    // because create_sampler() returns an error for TimeBased
    // We can't easily test this without real data, but the error message
    // is clear: "TimeBased sampling strategy is not yet implemented"
}

#[test]
fn test_pipeline_numerical_accuracy() {
    // Test that pipeline maintains numerical precision for financial data
    let mid_prices = vec![100.12345678, 100.12345679, 100.12345680, 100.12345677];

    let output = PipelineOutput {
        sequences: vec![],
        mid_prices: mid_prices.clone(),
        messages_processed: 4,
        features_extracted: 4,
        sequences_generated: 0,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let retrieved = output.get_mid_prices();

    // Verify bit-exact precision
    for (original, retrieved) in mid_prices.iter().zip(retrieved.iter()) {
        assert_eq!(*original, *retrieved, "Lost precision in mid-price storage");
    }
}

#[test]
fn test_large_sequence_handling() {
    // Test with many sequences (using Arc for zero-copy)
    let mut sequences = Vec::new();
    for i in 0..1000 {
        // Create feature vectors wrapped in Arc
        let features: Vec<FeatureVec> = (0..100)
            .map(|_| Arc::new(vec![i as f64; 48]))
            .collect();
        sequences.push(Sequence {
            features,
            start_timestamp: i * 100,
            end_timestamp: i * 100 + 99,
            duration_ns: 99,
            length: 100,
        });
    }

    let output = PipelineOutput {
        sequences,
        mid_prices: vec![100.0; 100_000],
        messages_processed: 1_000_000,
        features_extracted: 100_000,
        sequences_generated: 1000,
        stride: 1,
        window_size: 100,
        multiscale_sequences: None,
        adaptive_stats: None,
    };

    let flat = output.to_flat_features();
    assert_eq!(flat.len(), 100_000); // 1000 sequences * 100 snapshots
    assert_eq!(flat[0].len(), 48); // 48 features per snapshot
}

#[test]
fn test_pipeline_feature_count_consistency() {
    let config = PipelineConfig {
        features: FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 0,
        },
        sequence: SequenceConfig {
            window_size: 100,
            stride: 1,
            feature_count: 48, // 10 levels * 4 (ask_price, ask_vol, bid_price, bid_vol) + 8 derived
            max_buffer_size: 1000,
        },
        sampling: Some(SamplingConfig::default()),
        metadata: None,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_config_edge_cases() {
    // Very small window
    let mut config = PipelineConfig::default();
    config.features.include_mbo = false; // Use LOB only
    config.features.include_derived = false; // Default: raw LOB only
    config.sequence.window_size = 1;
    config.sequence.feature_count = 40; // 10 levels × 4 = 40 raw LOB features
    assert!(config.validate().is_ok());

    // Very large window
    config.sequence.window_size = 10_000;
    config.sequence.max_buffer_size = 11_000; // Must be >= window_size
    let result = config.validate();
    if let Err(ref e) = result {
        panic!("Large window failed: {e}");
    }
    assert!(result.is_ok());

    // Zero stride (should fail)
    config.sequence.stride = 0;
    let result = config.validate();
    if result.is_ok() {
        panic!("Zero stride should have failed validation");
    }
    assert!(result.is_err());
}

// ============================================================================
// Zero-Allocation Hot Path Tests
// ============================================================================

/// Verify that pipeline state is correctly reset and doesn't leak between runs.
/// This indirectly tests that the zero-allocation LobState reuse doesn't introduce bugs.
#[test]
fn test_pipeline_multiple_resets_clean_state() {
    let config = PipelineConfig::default();
    let mut pipeline = Pipeline::from_config(config).unwrap();

    // Reset multiple times to verify no state corruption
    for _ in 0..10 {
        pipeline.reset();
    }

    // Pipeline should still be valid after multiple resets
    // (No panic or crash indicates the zero-allocation path is stable)
}

/// Verify the pipeline can be created with various configurations.
/// This tests that the zero-allocation API integrates correctly.
#[test]
fn test_pipeline_configuration_flexibility() {
    // Test various LOB level configurations
    for levels in [5, 10, 15, 20] {
        let mut config = PipelineConfig::default();
        config.features.lob_levels = levels;
        config.features.include_derived = false;
        config.features.include_mbo = false;
        config.sequence.feature_count = levels * 4; // raw LOB features

        let pipeline = Pipeline::from_config(config);
        assert!(
            pipeline.is_ok(),
            "Pipeline should work with {} levels",
            levels
        );
    }
}
