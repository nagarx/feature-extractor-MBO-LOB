//! Test to expose and validate the multi-scale streaming issue.
//!
//! ISSUE: Pipeline::process_messages never calls try_build_all during streaming
//! for multi-scale mode. Only at finalization does it call try_build_all once,
//! which returns at most ONE sequence per scale.
//!
//! Expected: With 10,000 features pushed, we should get many sequences per scale.
//! Actual: Only 1 sequence per scale (fast, medium, slow) = 3 total.

use feature_extractor::sequence_builder::{MultiScaleConfig, MultiScaleWindow, ScaleConfig, Sequence};

/// Test that multi-scale window should produce multiple sequences during streaming.
///
/// With enough data, each scale should produce MULTIPLE sequences, not just one.
#[test]
fn test_multiscale_should_produce_multiple_sequences() {
    // Configure small windows for testing
    let config = MultiScaleConfig {
        fast: ScaleConfig {
            window_size: 10,
            stride: 5, // Every 5 events
            decimation: 1,
        },
        medium: ScaleConfig {
            window_size: 20,
            stride: 10,
            decimation: 2,
        },
        slow: ScaleConfig {
            window_size: 40,
            stride: 20,
            decimation: 4,
        },
    };

    let mut window = MultiScaleWindow::new(config, 4); // 4 features per sample

    // Push enough events for all scales
    // Slow scale: window=40, decimation=4 → needs 40*4=160 raw events minimum
    // Plus extra for multiple sequences
    for i in 0..500 {
        let features = vec![i as f64; 4];
        window.push(i * 1_000_000, features);
    }

    // Now build sequences
    let result = window.try_build_all();
    
    // The current implementation returns at most 1 sequence per scale
    // This test documents the expected vs actual behavior
    
    if let Some(ms) = &result {
        let (fast_count, medium_count, slow_count) = ms.sequence_counts();
        
        // With 500 events, decimation affects effective events per scale:
        // Fast: 500/1=500 effective, (500-10)/5 = ~98 sequences expected
        // Medium: 500/2=250 effective, (250-20)/10 = ~23 sequences expected  
        // Slow: 500/4=125 effective, (125-40)/20 = ~4 sequences expected
        
        // But currently we only get 1 each!
        // This assertion SHOULD pass once fixed, currently it will fail
        
        // For now, document the bug: we get exactly 1 per scale
        println!("Fast sequences: {} (expected ~18)", fast_count);
        println!("Medium sequences: {} (expected ~8)", medium_count);
        println!("Slow sequences: {} (expected ~3)", slow_count);
        
        // Assert that we get MORE than 1 sequence per scale
        // This will FAIL until the bug is fixed
        assert!(
            fast_count > 1,
            "BUG: Fast scale should produce multiple sequences with enough data. \
             Got {} sequences but expected ~18. Multi-scale streaming not implemented.",
            fast_count
        );
    } else {
        panic!("Expected Some(MultiScaleSequence), got None");
    }
}

/// Test to verify individual builders CAN produce multiple sequences.
///
/// This confirms the underlying SequenceBuilder works correctly,
/// isolating the bug to MultiScaleWindow integration.
#[test]
fn test_individual_builder_streaming_works() {
    use feature_extractor::sequence_builder::{SequenceBuilder, SequenceConfig};

    let config = SequenceConfig {
        window_size: 10,
        stride: 5,
        max_buffer_size: 100,
        feature_count: 4,
    };

    let mut builder = SequenceBuilder::with_config(config);
    let mut sequences: Vec<Sequence> = Vec::new();

    // Push 100 events, collecting sequences as they're built
    for i in 0..100 {
        let features = vec![i as f64; 4];
        if let Err(e) = builder.push(i * 1_000_000, features) {
            panic!("Push failed: {}", e);
        }
        
        // CRITICAL: Build sequence after each push
        if let Some(seq) = builder.try_build_sequence() {
            sequences.push(seq);
        }
    }

    // With stride 5 and 100 events, should get ~(100-10)/5 = ~18 sequences
    println!("Individual builder produced {} sequences", sequences.len());
    
    assert!(
        sequences.len() >= 15,
        "Individual builder should produce many sequences. Got {}",
        sequences.len()
    );
}

/// Document expected behavior after fix.
///
/// After fix, MultiScaleWindow should accumulate sequences during streaming
/// similar to how the regular pipeline does.
#[test]
fn test_document_expected_multiscale_behavior() {
    // This test documents what SHOULD happen:
    //
    // 1. During streaming (push_arc calls):
    //    - After each push, try_build_sequence() on each scale
    //    - Accumulate built sequences in internal vectors
    //
    // 2. At finalization (try_build_all):
    //    - Return accumulated sequences, not just one per scale
    //
    // Current behavior:
    //    - try_build_all() only returns current buffer state (1 seq each)
    //    - Sequences that could have been built during streaming are lost
    
    let config = MultiScaleConfig::default();
    let _window = MultiScaleWindow::new(config, 40);
    
    // Just documenting the expected API:
    // window.try_build_all_accumulated() -> MultiScaleSequence with all built sequences
    // Or: window should internally accumulate during push() and return all at try_build_all()
    
    // For now, this is a documentation-only test
    assert!(true, "Documented expected behavior");
}

