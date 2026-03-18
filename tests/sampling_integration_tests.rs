//! Integration tests for the trait-based sampling architecture.
//!
//! Tier 1: Synthetic data (always runnable, no real files needed).
//! Tier 2: Real data (guarded by `skip_if_no_data!()`, requires `--features databento`).
//!
//! These tests verify the sampling refactor (Phases 1-5) end-to-end:
//! - Sampler trait dispatch via `Box<dyn Sampler>`
//! - TimeBasedSampler grid alignment and interval correctness
//! - CompositeSampler algebraic properties
//! - MLOFI/Kolm OF sample_and_reset (interval-scoped, not cumulative)
//! - Feature count invariants across all sampling strategies
//! - Reset semantics between days

use feature_extractor::config::{PipelineConfig, SamplingConfig, SamplingStrategy};
use feature_extractor::preprocessing::sampling::{
    CompositeSampler, CompositeMode, EventBasedSampler, Sampler, SamplingContext,
    TimeBasedSampler, VolumeBasedSampler,
};
use feature_extractor::Pipeline;
use feature_extractor::SequenceConfig;

// ============================================================================
// Tier 1: Synthetic data tests (always runnable)
// ============================================================================

/// Helper: create a PipelineConfig with a specific sampling strategy.
fn config_with_sampling(sampling: SamplingConfig) -> PipelineConfig {
    PipelineConfig {
        features: Default::default(),
        sequence: SequenceConfig::default(),
        sampling: Some(sampling),
        metadata: None,
    }
}

#[test]
fn test_time_based_pipeline_creates_successfully() {
    let config = config_with_sampling(SamplingConfig {
        strategy: SamplingStrategy::TimeBased,
        volume_threshold: None,
        min_time_interval_ns: None,
        event_count: None,
        time_interval_ns: Some(5_000_000_000), // 5s
        utc_offset_hours: Some(-5),
        adaptive: None,
        multiscale: None,
    });

    let pipeline = Pipeline::from_config(config);
    assert!(
        pipeline.is_ok(),
        "Pipeline with TimeBased sampling should create successfully: {:?}",
        pipeline.err()
    );
}

#[test]
fn test_sampling_strategy_feature_count_invariant() {
    // All sampling strategies must produce the same expected feature count.
    // Feature count is determined by FeatureConfig, NOT by sampling strategy.
    let strategies = vec![
        SamplingConfig {
            strategy: SamplingStrategy::EventBased,
            event_count: Some(100),
            time_interval_ns: None,
            utc_offset_hours: None,
            ..Default::default()
        },
        SamplingConfig {
            strategy: SamplingStrategy::VolumeBased,
            volume_threshold: Some(500),
            time_interval_ns: None,
            utc_offset_hours: None,
            ..Default::default()
        },
        SamplingConfig {
            strategy: SamplingStrategy::TimeBased,
            volume_threshold: None,
            min_time_interval_ns: None,
            event_count: None,
            time_interval_ns: Some(1_000_000_000),
            utc_offset_hours: Some(-5),
            adaptive: None,
            multiscale: None,
        },
    ];

    let mut feature_counts = Vec::new();
    for sampling in strategies {
        let config = config_with_sampling(sampling.clone());
        let pipeline = Pipeline::from_config(config)
            .unwrap_or_else(|e| panic!("Failed for {:?}: {}", sampling.strategy, e));
        // Feature count is determined by FeatureConfig, not sampling strategy
        feature_counts.push(pipeline.config().features.feature_count());
    }

    assert!(
        feature_counts.windows(2).all(|w| w[0] == w[1]),
        "All strategies must produce same feature count. Got: {:?}",
        feature_counts
    );
}

// ============================================================================
// CompositeSampler algebraic properties
// ============================================================================

/// Helper: construct a SamplingContext with given timestamp and volume.
fn ctx(ts: u64, vol: u32) -> SamplingContext {
    SamplingContext {
        timestamp_ns: ts,
        event_volume: vol,
    }
}

/// Helper: run a sampler over N events and return sample count.
fn count_samples(sampler: &mut dyn Sampler, events: &[(u64, u32)]) -> u64 {
    for &(ts, vol) in events {
        sampler.should_sample(&ctx(ts, vol));
    }
    sampler.metrics().sample_count
}

/// Generate synthetic events: N events at 1ms intervals starting from a base timestamp.
fn make_events(base_ns: u64, count: usize) -> Vec<(u64, u32)> {
    (0..count)
        .map(|i| (base_ns + (i as u64) * 1_000_000, 100))
        .collect()
}

#[test]
fn test_composite_any_single_child_identity() {
    // Composite(Any, [E]) should produce the same sample count as E alone.
    let events = make_events(1_738_588_800_000_000_000, 1000); // ~14:40 UTC Feb 3

    let mut standalone = EventBasedSampler::new(10);
    let count_standalone = count_samples(&mut standalone, &events);

    let child = Box::new(EventBasedSampler::new(10));
    let mut composite = CompositeSampler::new(vec![child], CompositeMode::Any);
    let count_composite = count_samples(&mut composite, &events);

    assert_eq!(
        count_standalone, count_composite,
        "Composite(Any, [E(10)]) should equal E(10). Standalone={}, Composite={}",
        count_standalone, count_composite
    );
}

#[test]
fn test_composite_all_single_child_identity() {
    let events = make_events(1_738_588_800_000_000_000, 1000);

    let mut standalone = EventBasedSampler::new(10);
    let count_standalone = count_samples(&mut standalone, &events);

    let child = Box::new(EventBasedSampler::new(10));
    let mut composite = CompositeSampler::new(vec![child], CompositeMode::All);
    let count_composite = count_samples(&mut composite, &events);

    assert_eq!(
        count_standalone, count_composite,
        "Composite(All, [E(10)]) should equal E(10). Standalone={}, Composite={}",
        count_standalone, count_composite
    );
}

#[test]
fn test_composite_any_produces_superset() {
    // Composite(Any, [A, B]) should produce >= max(A_count, B_count) samples.
    let events = make_events(1_738_588_800_000_000_000, 1000);

    let mut a = EventBasedSampler::new(7);
    let count_a = count_samples(&mut a, &events);

    let mut b = EventBasedSampler::new(11);
    let count_b = count_samples(&mut b, &events);

    let child_a = Box::new(EventBasedSampler::new(7));
    let child_b = Box::new(EventBasedSampler::new(11));
    let mut composite = CompositeSampler::new(vec![child_a, child_b], CompositeMode::Any);
    let count_composite = count_samples(&mut composite, &events);

    let max_individual = count_a.max(count_b);
    assert!(
        count_composite >= max_individual,
        "Any-mode composite must produce >= max(A={}, B={}) = {}. Got {}",
        count_a, count_b, max_individual, count_composite
    );
}

#[test]
fn test_composite_all_produces_subset() {
    // Composite(All, [A, B]) should produce <= min(A_count, B_count) samples.
    let events = make_events(1_738_588_800_000_000_000, 1000);

    let mut a = EventBasedSampler::new(7);
    let count_a = count_samples(&mut a, &events);

    let mut b = EventBasedSampler::new(11);
    let count_b = count_samples(&mut b, &events);

    let child_a = Box::new(EventBasedSampler::new(7));
    let child_b = Box::new(EventBasedSampler::new(11));
    let mut composite = CompositeSampler::new(vec![child_a, child_b], CompositeMode::All);
    let count_composite = count_samples(&mut composite, &events);

    let min_individual = count_a.min(count_b);
    assert!(
        count_composite <= min_individual,
        "All-mode composite must produce <= min(A={}, B={}) = {}. Got {}",
        count_a, count_b, min_individual, count_composite
    );
}

// ============================================================================
// TimeBasedSampler grid and reset semantics
// ============================================================================

/// Nanoseconds per second.
const NS_PER_SECOND: u64 = 1_000_000_000;
const NS_PER_DAY: u64 = 86_400 * NS_PER_SECOND;

/// Compute UTC nanosecond timestamp for a given time on 2025-02-03.
/// Hours/minutes/seconds in UTC.
fn utc_time_ns(hours: u64, minutes: u64, seconds: u64) -> u64 {
    // 2025-02-03 00:00:00 UTC = 1738540800 seconds from epoch
    let day_base_s: u64 = 1_738_540_800;
    let time_s = hours * 3600 + minutes * 60 + seconds;
    (day_base_s + time_s) * NS_PER_SECOND
}

#[test]
fn test_time_based_reset_reinitializes_grid() {
    // After reset(), the grid must re-initialize from the next day's first event.
    let mut sampler = TimeBasedSampler::new(5 * NS_PER_SECOND, -5);

    // Day 1: 2025-02-03, market open at 14:30 UTC
    let day1_open = utc_time_ns(14, 30, 0);
    sampler.should_sample(&ctx(day1_open, 100));
    sampler.should_sample(&ctx(day1_open + 5 * NS_PER_SECOND, 100)); // triggers
    assert_eq!(sampler.metrics().sample_count, 1);

    // Reset between days
    sampler.reset();

    let m = sampler.metrics();
    assert_eq!(m.sample_count, 0, "sample_count must be 0 after reset");
    assert_eq!(m.events_observed, 0, "events_observed must be 0 after reset");

    // Day 2: 2025-02-04, market open at 14:30 UTC (next day)
    let day2_open = utc_time_ns(14, 30, 0) + NS_PER_DAY;
    sampler.should_sample(&ctx(day2_open, 100));
    sampler.should_sample(&ctx(day2_open + 5 * NS_PER_SECOND, 100)); // triggers
    assert_eq!(
        sampler.metrics().sample_count,
        1,
        "Day 2 should have its own independent sample count"
    );
}

#[test]
fn test_sampler_metrics_reset_to_zero() {
    // Verify all Sampler implementations zero metrics on reset.
    let samplers: Vec<Box<dyn Sampler>> = vec![
        Box::new(EventBasedSampler::new(5)),
        Box::new(VolumeBasedSampler::new(100, 0)),
        Box::new(TimeBasedSampler::new(NS_PER_SECOND, -5)),
    ];

    // Use enough events to span multiple seconds (3000 events at 1ms = 3 seconds)
    let events = make_events(utc_time_ns(14, 30, 0), 3000);

    for mut sampler in samplers {
        let name = sampler.strategy_name().to_string();

        // Generate some samples
        for &(ts, vol) in &events {
            sampler.should_sample(&ctx(ts, vol));
        }
        assert!(
            sampler.metrics().sample_count > 0,
            "{}: should have samples before reset",
            name
        );

        // Reset
        sampler.reset();

        let m = sampler.metrics();
        assert_eq!(m.sample_count, 0, "{}: sample_count after reset", name);
        assert_eq!(
            m.last_sample_timestamp_ns, 0,
            "{}: last_sample_timestamp_ns after reset",
            name
        );
    }
}

#[test]
fn test_time_based_deterministic_sample_points() {
    // Same input → same sample points (determinism requirement, Rule §7).
    let events = make_events(utc_time_ns(14, 30, 0), 5000);

    let run = |_| -> Vec<u64> {
        let mut sampler = TimeBasedSampler::new(NS_PER_SECOND, -5);
        let mut sample_timestamps = Vec::new();
        for &(ts, vol) in &events {
            if sampler.should_sample(&ctx(ts, vol)) {
                sample_timestamps.push(ts);
            }
        }
        sample_timestamps
    };

    let run1 = run(1);
    let run2 = run(2);

    assert_eq!(
        run1, run2,
        "Two runs with same input must produce identical sample points"
    );
    assert!(
        !run1.is_empty(),
        "Should produce at least some samples from 5000 events over 5 seconds"
    );
}

#[test]
fn test_time_based_sample_count_reasonable() {
    // 5-second interval over 6.5 trading hours ≈ 4680 samples.
    // With 1ms-spaced events, 6.5h = 23400 seconds = 23.4M events.
    // That's too many for a unit test. Instead verify the math with
    // a shorter window: 100 seconds of events at 1ms → 100/5 = 20 samples.
    let base = utc_time_ns(14, 30, 0);
    let events: Vec<(u64, u32)> = (0..100_000)
        .map(|i| (base + i * 1_000_000, 100)) // 1ms apart = 100 seconds
        .collect();

    let mut sampler = TimeBasedSampler::new(5 * NS_PER_SECOND, -5);
    for &(ts, vol) in &events {
        sampler.should_sample(&ctx(ts, vol));
    }

    let count = sampler.metrics().sample_count;
    // 100 seconds / 5s interval = 20 samples (first fires at 5s, last at 100s)
    assert!(
        (18..=22).contains(&count),
        "Expected ~20 samples for 100s of data at 5s intervals, got {}",
        count
    );
}
