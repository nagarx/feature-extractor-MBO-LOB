//! Phase 3: Full-Day Statistical Invariants
//!
//! Validates NVIDIA-specific statistical properties across a full trading day
//! (Feb 3, 2025) with tightened assertions. Replaces the previous "3-day shallow
//! range check" approach with deep statistical invariants on one canonical day.
//!
//! Run with:
//! ```bash
//! cargo test --release --features "parallel,databento" --test phase3_real_data_validation -- --test-threads=1
//! ```
//!
//! These tests require real NVIDIA MBO data in `../data/hot_store/`.

#![cfg(feature = "parallel")]

mod common;

use feature_extractor::builder::PipelineBuilder;
use feature_extractor::contract;
use feature_extractor::{AlignedBatchExporter, LabelConfig, Pipeline, PipelineConfig};

const CANONICAL_DATE: &str = "20250203";

fn build_full_98_config() -> PipelineConfig {
    PipelineBuilder::new()
        .lob_levels(10)
        .with_trading_signals()
        .window(100, 10)
        .event_sampling(500)
        .build_config()
        .expect("Failed to build 98-feature config")
}

fn find_test_file(date: &str) -> Option<String> {
    common::find_mbo_file(date).map(|p| p.to_string_lossy().to_string())
}

// =============================================================================
// Test 3a: Feature Extraction Determinism (full 98-feature pipeline)
// =============================================================================

#[test]
fn test_determinism_full_pipeline() {
    skip_if_no_data!();

    let date = CANONICAL_DATE;
    let file = match find_test_file(date) {
        Some(f) => f,
        None => {
            eprintln!("Skipping: no data for {date}");
            return;
        }
    };

    let config = build_full_98_config();

    let mut pipeline1 = Pipeline::from_config(config.clone()).expect("pipeline1");
    let output1 = pipeline1.process(&file).expect("run1");

    let mut pipeline2 = Pipeline::from_config(config).expect("pipeline2");
    let output2 = pipeline2.process(&file).expect("run2");

    assert_eq!(
        output1.messages_processed, output2.messages_processed,
        "Message count mismatch: {} vs {}",
        output1.messages_processed, output2.messages_processed
    );
    assert_eq!(
        output1.features_extracted, output2.features_extracted,
        "Feature count mismatch: {} vs {}",
        output1.features_extracted, output2.features_extracted
    );
    assert_eq!(
        output1.sequences_generated, output2.sequences_generated,
        "Sequence count mismatch: {} vs {}",
        output1.sequences_generated, output2.sequences_generated
    );

    let mut mismatches = 0u64;
    let min_seqs = output1.sequences.len().min(output2.sequences.len());
    for i in 0..min_seqs {
        let s1 = &output1.sequences[i];
        let s2 = &output2.sequences[i];
        for (j, (f1, f2)) in s1.features.iter().zip(s2.features.iter()).enumerate() {
            for (k, (&v1, &v2)) in f1.iter().zip(f2.iter()).enumerate() {
                let diff = (v1 - v2).abs();
                if diff > contract::FLOAT_CMP_EPS {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "Mismatch seq[{}].features[{}][{}]: {} vs {} (diff={})",
                            i, j, k, v1, v2, diff
                        );
                    }
                }
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "Pipeline non-deterministic: {} mismatches across {} sequences. \
         All features must be bit-exact across runs (BTreeMap guarantees deterministic order).",
        mismatches, min_seqs
    );
}

// =============================================================================
// Test 3b: NVIDIA Statistical Invariants (tightened from generic range checks)
// =============================================================================

#[test]
fn test_nvidia_statistical_invariants() {
    skip_if_no_data!();

    let file = match find_test_file(CANONICAL_DATE) {
        Some(f) => f,
        None => {
            eprintln!("Skipping: no data for {CANONICAL_DATE}");
            return;
        }
    };

    let config = build_full_98_config();
    let mut pipeline = Pipeline::from_config(config).expect("pipeline");
    let output = pipeline.process(&file).expect("process");

    let warmup_skip = 100.min(output.sequences.len());
    let n = output.sequences.len() - warmup_skip;
    assert!(n > 500, "Need >500 post-warmup sequences, got {n}");

    let mut mid_prices = Vec::with_capacity(n);
    let mut spreads = Vec::with_capacity(n);
    let mut book_valid_count: usize = 0;
    let mut ofi_positive: usize = 0;
    let mut ofi_negative: usize = 0;
    let mut time_regimes = std::collections::HashSet::new();
    let mut feature_sums = vec![0.0f64; contract::STABLE_FEATURE_COUNT];
    let mut feature_sq_sums = vec![0.0f64; contract::STABLE_FEATURE_COUNT];
    let mut nan_inf_count: usize = 0;

    for seq_idx in warmup_skip..output.sequences.len() {
        let last = output.sequences[seq_idx].features.last().unwrap();
        assert_eq!(last.len(), contract::STABLE_FEATURE_COUNT);

        for (i, &v) in last.iter().enumerate() {
            if !v.is_finite() {
                nan_inf_count += 1;
                continue;
            }
            feature_sums[i] += v;
            feature_sq_sums[i] += v * v;
        }

        mid_prices.push(last[40]);
        spreads.push(last[41]);

        if last[92] == 1.0 {
            book_valid_count += 1;
        }

        let ofi = last[84];
        if ofi > 0.0 {
            ofi_positive += 1;
        } else if ofi < 0.0 {
            ofi_negative += 1;
        }

        time_regimes.insert(last[93] as u32);
    }

    let n_f = n as f64;

    // NVIDIA mid-price range for Feb 2025
    let mean_mid = feature_sums[40] / n_f;
    assert!(
        (50.0..=250.0).contains(&mean_mid),
        "Mean mid_price = {mean_mid}, expected in [50, 250] for NVIDIA"
    );

    // Spread: NVIDIA is liquid, typical spread < $0.10
    let mean_spread = feature_sums[41] / n_f;
    assert!(
        mean_spread > 0.001 && mean_spread < 0.10,
        "Mean spread = {mean_spread}, expected in (0.001, 0.10) for NVIDIA"
    );

    // No NaN/Inf anywhere
    assert_eq!(
        nan_inf_count, 0,
        "Found {nan_inf_count} NaN/Inf values in feature vectors"
    );

    // OFI sign balance: both positive and negative (two-sided market)
    assert!(
        ofi_positive > 0 && ofi_negative > 0,
        "OFI sign imbalance: +{ofi_positive}/-{ofi_negative} -- market must be two-sided"
    );

    // time_regime: at least 2 distinct regimes in a full trading day
    assert!(
        time_regimes.len() >= 2,
        "Only {} time_regime values ({:?}), expected >=2 for a full day",
        time_regimes.len(),
        time_regimes
    );

    // book_valid: > 95% valid for NVIDIA (rarely crossed)
    let book_valid_pct = book_valid_count as f64 / n_f;
    assert!(
        book_valid_pct > 0.95,
        "book_valid = {:.1}%, expected >95% for NVIDIA",
        book_valid_pct * 100.0
    );

    // Feature variance: every non-constant feature should have std > 0
    // Excludes categorical flags (92=book_valid, 93=time_regime, 94=mbo_ready, 97=schema_version),
    // diagnostic signals (96=invalidity_delta, zero on clean days where book is always valid),
    // and known-zero MBO features (68,69=queue, 76=modification, 77=iceberg)
    let excluded_indices: &[usize] = &[68, 69, 76, 77, 92, 93, 94, 96, 97];
    let mut zero_variance = Vec::new();
    for i in 0..contract::STABLE_FEATURE_COUNT {
        if excluded_indices.contains(&i) {
            continue;
        }
        let mean = feature_sums[i] / n_f;
        let variance = (feature_sq_sums[i] / n_f) - (mean * mean);
        if variance.abs() < 1e-20 {
            zero_variance.push(i);
        }
    }
    assert!(
        zero_variance.is_empty(),
        "Features with zero variance (dead computations): {:?}",
        zero_variance
    );

    println!(
        "NVIDIA invariants validated: {n} samples, mean_mid={mean_mid:.2}, \
         mean_spread={mean_spread:.6}, book_valid={:.1}%, regimes={:?}, \
         OFI +{ofi_positive}/-{ofi_negative}",
        book_valid_pct * 100.0,
        time_regimes
    );
}

// =============================================================================
// Test 3c: Cross-Day State Isolation (reset vs fresh pipeline)
// =============================================================================

#[test]
fn test_cross_day_state_isolation() {
    skip_if_no_data!();

    let day1 = CANONICAL_DATE;
    let file1 = match find_test_file(day1) {
        Some(f) => f,
        None => {
            eprintln!("Skipping: no data for {day1}");
            return;
        }
    };

    // Use a second available day (any hot_store file other than Feb 3)
    let day2_file = {
        let dir = std::path::Path::new(common::HOT_STORE_DIR);
        if !dir.is_dir() {
            eprintln!("Skipping: hot_store not found");
            return;
        }
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map_or(false, |ext| ext == "dbn")
                    && !p.to_string_lossy().contains("20250203")
            })
            .collect();
        entries.sort();
        match entries.into_iter().next() {
            Some(p) => p.to_string_lossy().to_string(),
            None => {
                eprintln!("Skipping: no second day available for state isolation test");
                return;
            }
        }
    };
    let file2 = &day2_file;

    let config = build_full_98_config();

    // Path A: process day1, reset, process day2
    let mut pipeline_a = Pipeline::from_config(config.clone()).expect("pipeline_a");
    let _out1 = pipeline_a.process(&file1).expect("day1");
    pipeline_a.reset();
    let out_a = pipeline_a.process(&file2).expect("day2 after reset");

    // Path B: fresh pipeline, process day2 only
    let mut pipeline_b = Pipeline::from_config(config).expect("pipeline_b");
    let out_b = pipeline_b.process(&file2).expect("day2 fresh");

    assert_eq!(
        out_a.messages_processed, out_b.messages_processed,
        "State isolation: message count {} vs {}",
        out_a.messages_processed, out_b.messages_processed
    );
    assert_eq!(
        out_a.features_extracted, out_b.features_extracted,
        "State isolation: feature count {} vs {}",
        out_a.features_extracted, out_b.features_extracted
    );
    assert_eq!(
        out_a.sequences_generated, out_b.sequences_generated,
        "State isolation: sequence count {} vs {}",
        out_a.sequences_generated, out_b.sequences_generated
    );

    let mut mismatches = 0u64;
    let min_seqs = out_a.sequences.len().min(out_b.sequences.len());
    for i in 0..min_seqs {
        let sa = &out_a.sequences[i];
        let sb = &out_b.sequences[i];
        for (j, (fa, fb)) in sa.features.iter().zip(sb.features.iter()).enumerate() {
            for (k, (&va, &vb)) in fa.iter().zip(fb.iter()).enumerate() {
                let diff = (va - vb).abs();
                if diff > contract::FLOAT_CMP_EPS {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "State leak: seq[{}][{}][{}] = {} vs {} (diff={})",
                            i, j, k, va, vb, diff
                        );
                    }
                }
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "State isolation violated: {} mismatches across {} sequences. \
         Reset must produce identical output to a fresh pipeline.",
        mismatches, min_seqs
    );

    println!(
        "State isolation verified: {} sequences match across {} features (exact at {:e})",
        min_seqs,
        contract::STABLE_FEATURE_COUNT,
        contract::FLOAT_CMP_EPS
    );
}

// =============================================================================
// Test 3d: Signal Computation Spot-Check
// =============================================================================

#[test]
fn test_signal_spot_check() {
    skip_if_no_data!();

    let file = match find_test_file(CANONICAL_DATE) {
        Some(f) => f,
        None => {
            eprintln!("Skipping: no data for {CANONICAL_DATE}");
            return;
        }
    };

    let config = build_full_98_config();
    let mut pipeline = Pipeline::from_config(config).expect("pipeline");
    let output = pipeline.process(&file).expect("process");

    assert!(
        output.sequences_generated > 100,
        "Need at least 100 sequences for signal spot-check, got {}",
        output.sequences_generated
    );

    let warmup_skip = 50;
    let mut ofi_positive = 0u64;
    let mut ofi_negative = 0u64;
    let mut ofi_zero = 0u64;
    let mut executed_positive = 0u64;
    let mut executed_negative = 0u64;
    let mut checked = 0u64;

    for seq_idx in warmup_skip..output.sequences.len() {
        let last = output.sequences[seq_idx].features.last().unwrap();

        // Index 84: true_ofi
        let true_ofi = last[84];
        assert!(
            true_ofi.is_finite(),
            "seq {}: true_ofi is not finite: {}",
            seq_idx,
            true_ofi
        );
        if true_ofi > 0.0 {
            ofi_positive += 1;
        } else if true_ofi < 0.0 {
            ofi_negative += 1;
        } else {
            ofi_zero += 1;
        }

        // Index 85: depth_norm_ofi (should be bounded roughly in [-1, 1])
        let depth_norm_ofi = last[85];
        assert!(
            depth_norm_ofi.is_finite(),
            "seq {}: depth_norm_ofi is not finite: {}",
            seq_idx,
            depth_norm_ofi
        );

        // Index 86: executed_pressure
        let executed_pressure = last[86];
        assert!(
            executed_pressure.is_finite(),
            "seq {}: executed_pressure is not finite: {}",
            seq_idx,
            executed_pressure
        );
        if executed_pressure > 0.0 {
            executed_positive += 1;
        } else if executed_pressure < 0.0 {
            executed_negative += 1;
        }

        // Sign convention validation (RULE.md section 10):
        // > 0 = Bullish / Buy pressure
        // < 0 = Bearish / Sell pressure
        // OFI and executed_pressure should show both signs in a full trading day
        checked += 1;
    }

    // Across a full trading day, OFI should have both positive and negative values.
    // If it's all one sign, something is wrong.
    assert!(
        ofi_positive > 0 && ofi_negative > 0,
        "OFI sign imbalance: +{}/-{}/zero={} -- expected both signs in a full day",
        ofi_positive,
        ofi_negative,
        ofi_zero
    );
    assert!(
        executed_positive > 0 && executed_negative > 0,
        "executed_pressure sign imbalance: +{}/{}- -- expected both signs",
        executed_positive,
        executed_negative
    );

    println!(
        "Signal spot-check passed: {} samples, OFI +{}/{}-/{}zero, exec_pressure +{}/{}",
        checked, ofi_positive, ofi_negative, ofi_zero, executed_positive, executed_negative
    );
}

// =============================================================================
// Test 3e: MBO Feature Statistical Properties
// =============================================================================

#[test]
fn test_mbo_feature_statistics() {
    skip_if_no_data!();

    let file = match find_test_file(CANONICAL_DATE) {
        Some(f) => f,
        None => {
            eprintln!("Skipping: no data for {CANONICAL_DATE}");
            return;
        }
    };

    let config = build_full_98_config();
    let mut pipeline = Pipeline::from_config(config).expect("pipeline");
    let output = pipeline.process(&file).expect("process");

    let warmup_skip = 100.min(output.sequences.len());
    let n = output.sequences.len() - warmup_skip;
    assert!(n > 100, "Need >100 post-warmup sequences, got {}", n);

    let mbo_start = 48;
    let mbo_end = 84;
    let mbo_count = mbo_end - mbo_start;

    let mut sums = vec![0.0f64; mbo_count];
    let mut sq_sums = vec![0.0f64; mbo_count];
    let mut mins = vec![f64::MAX; mbo_count];
    let mut maxs = vec![f64::MIN; mbo_count];
    let mut nan_counts = vec![0u64; mbo_count];
    let mut inf_counts = vec![0u64; mbo_count];

    for seq_idx in warmup_skip..output.sequences.len() {
        let last = output.sequences[seq_idx].features.last().unwrap();
        for (fi, &val) in last[mbo_start..mbo_end].iter().enumerate() {
            if val.is_nan() {
                nan_counts[fi] += 1;
                continue;
            }
            if val.is_infinite() {
                inf_counts[fi] += 1;
                continue;
            }
            sums[fi] += val;
            sq_sums[fi] += val * val;
            mins[fi] = mins[fi].min(val);
            maxs[fi] = maxs[fi].max(val);
        }
    }

    let n_f64 = n as f64;
    // Features expected to be zero by design:
    // - 68, 69: queue_position, queue_size_ahead (require include_queue_tracking=true)
    // - 76: modification_score (ITCH feeds use cancel+add, not modify)
    // - 77: iceberg_proxy (depends on modification_score)
    let known_zero_indices: &[usize] = &[68, 69, 76, 77];

    let mut zero_variance_features = Vec::new();
    let mut nan_features = Vec::new();
    let mut inf_features = Vec::new();

    for fi in 0..mbo_count {
        let global_idx = mbo_start + fi;

        if nan_counts[fi] > 0 {
            nan_features.push((global_idx, nan_counts[fi]));
        }
        if inf_counts[fi] > 0 {
            inf_features.push((global_idx, inf_counts[fi]));
        }

        if known_zero_indices.contains(&global_idx) {
            continue;
        }

        let mean = sums[fi] / n_f64;
        let variance = (sq_sums[fi] / n_f64) - (mean * mean);
        if variance.abs() < 1e-20 {
            zero_variance_features.push(global_idx);
        }
    }

    assert!(
        nan_features.is_empty(),
        "MBO features with NaN after warmup: {:?}",
        nan_features
    );
    assert!(
        inf_features.is_empty(),
        "MBO features with Inf after warmup: {:?}",
        inf_features
    );
    assert!(
        zero_variance_features.is_empty(),
        "MBO features with unexpected zero variance: {:?} (excluded known-zero: {:?})",
        zero_variance_features,
        known_zero_indices
    );

    // Cancel rate features (indices 50,51) should be in [0, 1]
    for &cancel_idx in &[50usize, 51] {
        let fi = cancel_idx - mbo_start;
        assert!(
            mins[fi] >= -0.001 && maxs[fi] <= 1.001,
            "Cancel rate feature[{}] range [{}, {}] outside [0, 1]",
            cancel_idx,
            mins[fi],
            maxs[fi]
        );
    }

    // Trade rate features (indices 52,53) should be >= 0
    for &trade_idx in &[52usize, 53] {
        let fi = trade_idx - mbo_start;
        assert!(
            mins[fi] >= -0.001,
            "Trade rate feature[{}] min = {} (expected >= 0)",
            trade_idx,
            mins[fi]
        );
    }

    println!(
        "MBO feature statistics validated: {} features x {} samples, no dead computations",
        mbo_count, n
    );
}

// =============================================================================
// Test 3f: End-to-End Export Validation (.npy + metadata JSON)
// =============================================================================

#[test]
fn test_export_end_to_end() {
    skip_if_no_data!();

    let date = CANONICAL_DATE;
    let file = match find_test_file(date) {
        Some(f) => f,
        None => {
            eprintln!("Skipping: no data for {date}");
            return;
        }
    };

    let config = build_full_98_config();
    let mut pipeline = Pipeline::from_config(config).expect("pipeline");
    let output = pipeline.process(&file).expect("process");

    assert!(
        output.sequences_generated > 0,
        "No sequences generated from real data"
    );

    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let label_config = LabelConfig {
        horizon: 50,
        smoothing_window: 10,
        threshold: 0.0008,
    };

    let exporter = AlignedBatchExporter::new(
        temp_dir.path(),
        label_config,
        100, // window_size
        10,  // stride
    );

    let result = exporter
        .export_day(date, &output)
        .expect("export_day failed");

    // --- Validate export result struct ---
    assert!(
        result.n_sequences > 0,
        "Export produced 0 sequences from {} pipeline sequences",
        output.sequences_generated
    );
    assert_eq!(
        result.seq_shape.1,
        contract::STABLE_FEATURE_COUNT,
        "Exported feature width {} != contract STABLE_FEATURE_COUNT {}",
        result.seq_shape.1,
        contract::STABLE_FEATURE_COUNT
    );
    assert_eq!(
        result.seq_shape.0, 100,
        "Exported window size {} != 100",
        result.seq_shape.0
    );

    // --- Validate output files exist ---
    let seq_path = temp_dir.path().join(format!("{date}_sequences.npy"));
    let label_path = temp_dir.path().join(format!("{date}_labels.npy"));
    let meta_path = temp_dir.path().join(format!("{date}_metadata.json"));
    let norm_path = temp_dir.path().join(format!("{date}_normalization.json"));

    assert!(seq_path.exists(), "sequences.npy not found");
    assert!(label_path.exists(), "labels.npy not found");
    assert!(meta_path.exists(), "metadata.json not found");
    assert!(norm_path.exists(), "normalization.json not found");

    // --- Validate .npy shapes via ndarray ---
    use ndarray::{Array1, Array3};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;

    let seq_file = File::open(&seq_path).expect("open sequences.npy");
    let sequences: Array3<f32> = Array3::read_npy(seq_file).expect("read sequences.npy");

    let label_file = File::open(&label_path).expect("open labels.npy");
    let labels: Array1<i8> = Array1::read_npy(label_file).expect("read labels.npy");

    assert_eq!(
        sequences.shape()[0],
        labels.len(),
        "sequences.shape[0]={} != labels.len()={}",
        sequences.shape()[0],
        labels.len()
    );
    assert_eq!(
        sequences.shape()[0],
        result.n_sequences,
        "sequences.shape[0]={} != result.n_sequences={}",
        sequences.shape()[0],
        result.n_sequences
    );
    assert_eq!(
        sequences.shape()[1],
        100,
        "window_size={} != 100",
        sequences.shape()[1]
    );
    assert_eq!(
        sequences.shape()[2],
        contract::STABLE_FEATURE_COUNT,
        "features={} != {}",
        sequences.shape()[2],
        contract::STABLE_FEATURE_COUNT
    );

    // --- Validate no NaN/Inf in sequences ---
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;
    for &v in sequences.iter() {
        if v.is_nan() {
            nan_count += 1;
        }
        if v.is_infinite() {
            inf_count += 1;
        }
    }
    assert_eq!(
        nan_count, 0,
        "Found {} NaN values in exported sequences",
        nan_count
    );
    assert_eq!(
        inf_count, 0,
        "Found {} Inf values in exported sequences",
        inf_count
    );

    // --- Validate labels in {-1, 0, 1} ---
    for &label in labels.iter() {
        assert!(
            (-1..=1).contains(&label),
            "Label {} not in valid range {{-1, 0, 1}}",
            label
        );
    }

    // --- Validate metadata JSON ---
    let meta_content = std::fs::read_to_string(&meta_path).expect("read metadata.json");
    let meta: serde_json::Value = serde_json::from_str(&meta_content).expect("parse metadata.json");

    let meta_n_seq = meta["n_sequences"].as_u64().expect("n_sequences");
    assert_eq!(
        meta_n_seq, result.n_sequences as u64,
        "metadata.n_sequences={} != result.n_sequences={}",
        meta_n_seq, result.n_sequences
    );

    let meta_window = meta["window_size"].as_u64().expect("window_size");
    assert_eq!(
        meta_window, 100,
        "metadata.window_size={} != 100",
        meta_window
    );

    let meta_features = meta["n_features"].as_u64().expect("n_features");
    assert_eq!(
        meta_features,
        contract::STABLE_FEATURE_COUNT as u64,
        "metadata.n_features={} != {}",
        meta_features,
        contract::STABLE_FEATURE_COUNT
    );

    let meta_schema = meta["schema_version"]
        .as_str()
        .expect("schema_version string");
    let expected_schema = contract::SCHEMA_VERSION.to_string();
    assert_eq!(
        meta_schema, expected_schema,
        "metadata.schema_version='{}' != '{}'",
        meta_schema, expected_schema
    );

    assert!(
        meta["label_strategy"].is_string(),
        "metadata.label_strategy missing"
    );
    assert!(
        meta["provenance"].is_object(),
        "metadata.provenance missing"
    );
    assert!(
        meta["validation"]["sequences_labels_match"].as_bool() == Some(true),
        "metadata.validation.sequences_labels_match != true"
    );

    // --- Validate normalization JSON ---
    let norm_content = std::fs::read_to_string(&norm_path).expect("read normalization.json");
    let _norm: serde_json::Value =
        serde_json::from_str(&norm_content).expect("parse normalization.json");

    println!(
        "End-to-end export validated: {} sequences, shape [{}, {}, {}], labels in {{-1,0,1}}, metadata consistent",
        result.n_sequences,
        sequences.shape()[0],
        sequences.shape()[1],
        sequences.shape()[2],
    );
}
