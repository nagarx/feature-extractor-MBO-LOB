//! Benchmark suite for feature extraction performance.
//!
//! Run with: `cargo bench`
//!
//! This benchmark measures:
//! - Raw LOB feature extraction throughput
//! - Derived feature computation
//! - Order flow feature extraction
//! - Full pipeline performance
//! - Normalization overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use feature_extractor::{
    features::{
        derived_features::compute_derived_features,
        lob_features::extract_raw_features,
        order_flow::{compute_depth_imbalance, compute_queue_imbalance, OrderFlowTracker},
        FeatureConfig, FeatureExtractor,
    },
    preprocessing::{BilinearNormalizer, Normalizer, ZScoreNormalizer},
};
use mbo_lob_reconstructor::LobState;

/// Create a realistic test LOB state with the specified number of levels.
fn create_test_lob(levels: usize) -> LobState {
    let mut lob = LobState::new(levels);

    // Base price: $100.00 (in fixed-point: 100 * 1e9)
    let base_price: i64 = 100_000_000_000;
    let tick_size: i64 = 10_000_000; // $0.01

    // Fill bid side (prices decrease from best bid)
    for i in 0..levels {
        lob.bid_prices[i] = base_price - (i as i64 + 1) * tick_size;
        lob.bid_sizes[i] = ((100 + i * 50) % 500 + 100) as u32; // Varying sizes
    }

    // Fill ask side (prices increase from best ask)
    for i in 0..levels {
        lob.ask_prices[i] = base_price + (i as i64 + 1) * tick_size;
        lob.ask_sizes[i] = ((150 + i * 30) % 400 + 100) as u32; // Varying sizes
    }

    lob.best_bid = Some(base_price - tick_size);
    lob.best_ask = Some(base_price + tick_size);

    lob
}

/// Create a sequence of LOB states for order flow tracking.
fn create_lob_sequence(count: usize, levels: usize) -> Vec<LobState> {
    let mut states = Vec::with_capacity(count);
    let base_price: i64 = 100_000_000_000;
    let tick_size: i64 = 10_000_000;

    for seq in 0..count {
        let mut lob = LobState::new(levels);

        // Simulate price movement
        let price_offset = ((seq % 20) as i64 - 10) * tick_size;

        for i in 0..levels {
            lob.bid_prices[i] = base_price + price_offset - (i as i64 + 1) * tick_size;
            lob.bid_sizes[i] = ((100 + seq * 7 + i * 50) % 500 + 100) as u32;
            lob.ask_prices[i] = base_price + price_offset + (i as i64 + 1) * tick_size;
            lob.ask_sizes[i] = ((150 + seq * 11 + i * 30) % 400 + 100) as u32;
        }

        lob.best_bid = Some(base_price + price_offset - tick_size);
        lob.best_ask = Some(base_price + price_offset + tick_size);

        states.push(lob);
    }

    states
}

/// Benchmark raw LOB feature extraction.
fn bench_raw_lob_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_lob_features");

    for levels in [5, 10, 20].iter() {
        let lob = create_test_lob(*levels);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("extract", levels), levels, |b, &levels| {
            let mut features = Vec::with_capacity(levels * 4);
            b.iter(|| {
                features.clear();
                extract_raw_features(black_box(&lob), levels, &mut features);
                black_box(&features);
            });
        });
    }

    group.finish();
}

/// Benchmark derived feature computation.
fn bench_derived_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("derived_features");

    for levels in [5, 10, 20].iter() {
        let lob = create_test_lob(*levels);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("compute", levels), levels, |b, &levels| {
            b.iter(|| {
                let result = compute_derived_features(black_box(&lob), levels);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark order flow features.
fn bench_order_flow_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_flow_features");

    // Benchmark queue imbalance (single snapshot)
    let lob = create_test_lob(10);
    group.bench_function("queue_imbalance", |b| {
        b.iter(|| compute_queue_imbalance(black_box(&lob)))
    });

    // Benchmark depth imbalance
    group.bench_function("depth_imbalance_10_levels", |b| {
        b.iter(|| compute_depth_imbalance(black_box(&lob), 10))
    });

    // Benchmark order flow tracker with sequence
    for seq_len in [100, 1000, 10000].iter() {
        let states = create_lob_sequence(*seq_len, 10);

        group.throughput(Throughput::Elements(*seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("tracker_update", seq_len),
            &states,
            |b, states| {
                b.iter(|| {
                    let mut tracker = OrderFlowTracker::new();
                    for (i, lob) in states.iter().enumerate() {
                        tracker.update(black_box(lob), (i * 1_000_000) as u64);
                    }
                    let features = tracker.extract_features(&states[states.len() - 1]);
                    black_box(features)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full feature extractor.
fn bench_feature_extractor(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extractor");

    // LOB only (default config)
    let lob = create_test_lob(10);
    let extractor = FeatureExtractor::new(10);

    group.bench_function("lob_only_10_levels", |b| {
        b.iter(|| extractor.extract_lob_features(black_box(&lob)))
    });

    // LOB + derived
    let config_with_derived = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: false,
        mbo_window_size: 1000,
        include_signals: false,
    };
    let extractor_derived = FeatureExtractor::with_config(config_with_derived);

    group.bench_function("lob_with_derived_10_levels", |b| {
        b.iter(|| extractor_derived.extract_lob_features(black_box(&lob)))
    });

    // Full pipeline with MBO
    let config_full = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: false,
    };
    let mut extractor_full = FeatureExtractor::with_config(config_full);

    group.bench_function("full_pipeline_10_levels", |b| {
        b.iter(|| extractor_full.extract_all_features(black_box(&lob)))
    });

    group.finish();
}

/// Benchmark normalization strategies.
fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    // Generate test data
    let values: Vec<f64> = (0..1000).map(|i| 100.0 + (i as f64 * 0.01)).collect();

    // Z-Score normalization
    group.bench_function("zscore_update_1000", |b| {
        b.iter(|| {
            let mut normalizer = ZScoreNormalizer::new();
            for &v in &values {
                normalizer.update(black_box(v));
            }
            black_box(normalizer.mean())
        });
    });

    // Z-Score with window
    group.bench_function("zscore_windowed_100", |b| {
        b.iter(|| {
            let mut normalizer = ZScoreNormalizer::with_window(100);
            for &v in &values {
                normalizer.update(black_box(v));
            }
            black_box(normalizer.mean())
        });
    });

    // Bilinear normalization batch
    group.bench_function("bilinear_batch_1000", |b| {
        let mut normalizer = BilinearNormalizer::new(0.01, 50.0);
        normalizer.set_mid_price(100.0);
        b.iter(|| {
            let result = normalizer.normalize_batch(black_box(&values));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark throughput for realistic workload.
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Simulate processing 1 second of high-frequency data
    // Typical HFT: 10,000-100,000 updates per second
    for updates_per_sec in [10_000u64, 50_000, 100_000].iter() {
        let states = create_lob_sequence(*updates_per_sec as usize, 10);
        let extractor = FeatureExtractor::new(10);

        group.throughput(Throughput::Elements(*updates_per_sec));
        group.bench_with_input(
            BenchmarkId::new("process_updates", updates_per_sec),
            &states,
            |b, states| {
                b.iter(|| {
                    let mut all_features = Vec::with_capacity(states.len());
                    for lob in states {
                        let features = extractor.extract_lob_features(black_box(lob)).unwrap();
                        all_features.push(features);
                    }
                    black_box(all_features)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_raw_lob_features,
    bench_derived_features,
    bench_order_flow_features,
    bench_feature_extractor,
    bench_normalization,
    bench_throughput,
);

criterion_main!(benches);
