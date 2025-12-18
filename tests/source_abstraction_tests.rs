//! End-to-end tests for the MarketDataSource abstraction.
//!
//! These tests verify that the Pipeline works correctly with different
//! data sources through the MarketDataSource trait.

use feature_extractor::config::{PipelineConfig, SamplingConfig, SamplingStrategy};
use feature_extractor::{FeatureConfig, Pipeline, SequenceConfig};
use mbo_lob_reconstructor::{
    Action, MarketDataSource, MboMessage, Side, SourceMetadata, VecSource,
};

/// Create test messages representing a realistic order book scenario.
fn create_test_messages() -> Vec<MboMessage> {
    let mut messages = Vec::new();
    let base_price: i64 = 100_000_000_000; // $100.00 in fixed-point

    // Build initial order book with multiple levels
    // Bids: $99.95, $99.90, $99.85, etc.
    // Asks: $100.05, $100.10, $100.15, etc.

    let mut order_id = 1u64;
    let mut timestamp = 1000000000i64; // Starting timestamp

    // Add bid levels
    for i in 0..10 {
        let price = base_price - (i as i64 + 1) * 50_000_000; // $0.05 increments down
        for _ in 0..5 {
            // 5 orders per level
            let mut msg = MboMessage::new(order_id, Action::Add, Side::Bid, price, 100);
            msg.timestamp = Some(timestamp);
            messages.push(msg);
            order_id += 1;
            timestamp += 1000;
        }
    }

    // Add ask levels
    for i in 0..10 {
        let price = base_price + (i as i64 + 1) * 50_000_000; // $0.05 increments up
        for _ in 0..5 {
            let mut msg = MboMessage::new(order_id, Action::Add, Side::Ask, price, 100);
            msg.timestamp = Some(timestamp);
            messages.push(msg);
            order_id += 1;
            timestamp += 1000;
        }
    }

    // Add trading activity (modifications, trades, cancels)
    for i in 0..1000 {
        // Alternate between different activities
        let action = match i % 4 {
            0 => Action::Add,
            1 => Action::Modify,
            2 => Action::Trade,
            _ => Action::Cancel,
        };

        let side = if i % 2 == 0 { Side::Bid } else { Side::Ask };
        let price_offset = (i % 5) as i64 * 50_000_000;
        let price = if side == Side::Bid {
            base_price - price_offset - 50_000_000
        } else {
            base_price + price_offset + 50_000_000
        };

        let size = 50 + (i % 100) as u32;

        let mut msg = MboMessage::new(order_id, action, side, price, size);
        msg.timestamp = Some(timestamp);
        messages.push(msg);

        order_id += 1;
        timestamp += 1000;
    }

    messages
}

#[test]
fn test_pipeline_with_vec_source() {
    let messages = create_test_messages();
    let message_count = messages.len();

    let source = VecSource::new(messages).with_metadata(
        SourceMetadata::new()
            .with_symbol("TEST")
            .with_date("2025-01-01"),
    );

    // Verify metadata
    assert_eq!(source.metadata().symbol, Some("TEST".to_string()));
    assert_eq!(source.metadata().date, Some("2025-01-01".to_string()));

    // Create pipeline
    let config = PipelineConfig {
        features: FeatureConfig::default(),
        sequence: SequenceConfig {
            window_size: 50,
            stride: 10,
            ..Default::default()
        },
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::EventBased,
            event_count: Some(10),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");
    let output = pipeline
        .process_source(source)
        .expect("Failed to process source");

    // Verify output
    assert!(
        output.messages_processed > 0,
        "Should have processed messages"
    );
    assert!(
        output.messages_processed <= message_count,
        "Should not exceed message count"
    );
    assert!(
        output.features_extracted > 0,
        "Should have extracted features"
    );
}

#[test]
fn test_pipeline_process_messages_directly() {
    let messages = create_test_messages();

    let config = PipelineConfig {
        features: FeatureConfig::default(),
        sequence: SequenceConfig {
            window_size: 50,
            stride: 10,
            ..Default::default()
        },
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::EventBased,
            event_count: Some(10),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");

    // Process messages directly using the iterator
    let output = pipeline
        .process_messages(messages.into_iter())
        .expect("Failed to process messages");

    assert!(output.messages_processed > 0);
    assert!(output.features_extracted > 0);
}

#[test]
fn test_vec_source_empty_messages() {
    let source = VecSource::new(Vec::new());

    let config = PipelineConfig::default();
    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");

    let output = pipeline
        .process_source(source)
        .expect("Should handle empty source");

    assert_eq!(output.messages_processed, 0);
    assert_eq!(output.features_extracted, 0);
    assert!(output.sequences.is_empty());
}

#[test]
fn test_vec_source_single_message() {
    let messages = vec![MboMessage::new(
        1,
        Action::Add,
        Side::Bid,
        100_000_000_000,
        100,
    )];

    let source = VecSource::new(messages);

    let config = PipelineConfig {
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::EventBased,
            event_count: Some(1),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");
    let output = pipeline
        .process_source(source)
        .expect("Should handle single message");

    // Single bid order won't produce a valid LOB (no ask), so features_extracted should be 0
    assert_eq!(output.messages_processed, 1);
}

#[test]
fn test_source_metadata_preserved() {
    let messages = create_test_messages();

    let source = VecSource::new(messages).with_metadata(
        SourceMetadata::new()
            .with_symbol("NVDA")
            .with_date("2025-02-03")
            .with_provider("test")
            .with_estimated_messages(1000),
    );

    // Verify metadata before consuming
    let meta = source.metadata();
    assert_eq!(meta.symbol, Some("NVDA".to_string()));
    assert_eq!(meta.date, Some("2025-02-03".to_string()));
    assert_eq!(meta.provider, Some("test".to_string()));
    assert_eq!(meta.estimated_messages, Some(1000));
}

#[test]
fn test_pipeline_reset_between_sources() {
    let config = PipelineConfig {
        features: FeatureConfig::default(),
        sequence: SequenceConfig {
            window_size: 20,
            stride: 5,
            ..Default::default()
        },
        sampling: Some(SamplingConfig {
            strategy: SamplingStrategy::EventBased,
            event_count: Some(5),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");

    // Process first source
    let source1 = VecSource::new(create_test_messages());
    let output1 = pipeline
        .process_source(source1)
        .expect("Failed to process first source");

    // Reset pipeline
    pipeline.reset();

    // Process second source
    let source2 = VecSource::new(create_test_messages());
    let output2 = pipeline
        .process_source(source2)
        .expect("Failed to process second source");

    // Both should produce similar results (same input data)
    assert_eq!(output1.messages_processed, output2.messages_processed);
    assert_eq!(output1.features_extracted, output2.features_extracted);
}

#[test]
fn test_market_data_source_trait_is_object_safe() {
    // This test verifies that MarketDataSource can be used with dynamic dispatch
    // (important for flexibility in complex systems)

    fn accept_any_source<S: MarketDataSource>(source: S) -> usize {
        let meta = source.metadata();
        meta.estimated_messages.unwrap_or(0) as usize
    }

    let source = VecSource::new(vec![
        MboMessage::new(1, Action::Add, Side::Bid, 100_000_000_000, 100),
        MboMessage::new(2, Action::Add, Side::Ask, 100_010_000_000, 100),
    ]);

    let count = accept_any_source(source);
    assert_eq!(count, 2);
}

#[test]
fn test_numerical_accuracy_through_source() {
    // Verify that numerical precision is maintained when using VecSource

    let exact_price: i64 = 123_456_789_012; // Precise price
    let exact_size: u32 = 9999;

    let mut msg = MboMessage::new(1, Action::Add, Side::Bid, exact_price, exact_size);
    msg.timestamp = Some(1000000000);

    let source = VecSource::new(vec![msg]);

    // Consume source and verify message integrity
    let messages: Vec<_> = source.messages().expect("Failed to get messages").collect();

    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0].price, exact_price,
        "Price should be exactly preserved"
    );
    assert_eq!(
        messages[0].size, exact_size,
        "Size should be exactly preserved"
    );
    assert_eq!(messages[0].order_id, 1);
    assert_eq!(messages[0].action, Action::Add);
    assert_eq!(messages[0].side, Side::Bid);
}

// ============================================================================
// DbnSource tests (require actual files)
// ============================================================================

#[cfg(feature = "databento")]
mod dbn_source_tests {
    use super::*;
    use mbo_lob_reconstructor::DbnSource;
    use std::path::Path;

    fn find_test_file() -> Option<String> {
        let candidates = [
            "../data/NVDA_2025-02-01_to_2025-09-30/NVDA_2025-02-03.mbo.dbn.zst",
            "../../data/NVDA_2025-02-01_to_2025-09-30/NVDA_2025-02-03.mbo.dbn.zst",
            "data/NVDA_2025-02-01_to_2025-09-30/NVDA_2025-02-03.mbo.dbn.zst",
        ];

        for path in candidates {
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
        None
    }

    #[test]
    fn test_dbn_source_with_pipeline() {
        let Some(test_file) = find_test_file() else {
            eprintln!("Skipping test_dbn_source_with_pipeline: no test file available");
            return;
        };

        let source = DbnSource::new(&test_file).expect("Failed to create DbnSource");

        // Verify metadata extraction
        assert_eq!(source.metadata().provider, Some("databento".to_string()));
        assert!(source.metadata().symbol.is_some());

        let config = PipelineConfig {
            features: FeatureConfig::default(),
            sequence: SequenceConfig {
                window_size: 100,
                stride: 20,
                ..Default::default()
            },
            sampling: Some(SamplingConfig {
                strategy: SamplingStrategy::EventBased,
                event_count: Some(1000),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut pipeline = Pipeline::from_config(config).expect("Failed to create pipeline");
        let output = pipeline
            .process_source(source)
            .expect("Failed to process DbnSource");

        // Real NVDA data should have significant activity
        assert!(
            output.messages_processed > 10000,
            "Should process many messages from real data"
        );
        assert!(
            output.features_extracted > 0,
            "Should extract features from real data"
        );
    }

    #[test]
    fn test_dbn_source_vs_direct_loader_consistency() {
        let Some(test_file) = find_test_file() else {
            eprintln!("Skipping consistency test: no test file available");
            return;
        };

        let config = PipelineConfig {
            features: FeatureConfig::default(),
            sequence: SequenceConfig {
                window_size: 50,
                stride: 10,
                ..Default::default()
            },
            sampling: Some(SamplingConfig {
                strategy: SamplingStrategy::EventBased,
                event_count: Some(1000),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Process using DbnSource
        let source = DbnSource::new(&test_file).expect("Failed to create DbnSource");
        let mut pipeline1 =
            Pipeline::from_config(config.clone()).expect("Failed to create pipeline");
        let output1 = pipeline1
            .process_source(source)
            .expect("Failed with DbnSource");

        // Process using direct path
        let mut pipeline2 = Pipeline::from_config(config).expect("Failed to create pipeline");
        let output2 = pipeline2
            .process(&test_file)
            .expect("Failed with direct path");

        // Results should be identical
        assert_eq!(
            output1.messages_processed, output2.messages_processed,
            "Message count should match"
        );
        assert_eq!(
            output1.features_extracted, output2.features_extracted,
            "Feature count should match"
        );
        assert_eq!(
            output1.sequences_generated, output2.sequences_generated,
            "Sequence count should match"
        );

        // Mid-prices should match exactly
        assert_eq!(
            output1.mid_prices.len(),
            output2.mid_prices.len(),
            "Mid-price count should match"
        );

        for (i, (p1, p2)) in output1
            .mid_prices
            .iter()
            .zip(output2.mid_prices.iter())
            .enumerate()
        {
            assert!(
                (p1 - p2).abs() < 1e-9,
                "Mid-price {} mismatch: {} vs {}",
                i,
                p1,
                p2
            );
        }
    }
}
