//! Level 4: Export Round-Trip Validation
//!
//! Validates that the export pipeline preserves numerical fidelity:
//! 1. In-memory features survive the f64→f32→npy→f32 round-trip within tolerance
//! 2. Multi-horizon labels produce correct shape and valid values
//! 3. Normalization config does not corrupt raw feature values
//!
//! Uses synthetic data (no real data dependency) so these tests always run in CI.

use feature_extractor::{
    contract, AlignedBatchExporter, FeatureVec, LabelConfig, MultiHorizonConfig, PipelineOutput,
    Sequence,
};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::sync::Arc;
use tempfile::TempDir;

const WINDOW: usize = 100;
const STRIDE: usize = 10;
const N_FEATURES: usize = 98;
const N_SEQUENCES: usize = 200;

fn make_synthetic_output() -> PipelineOutput {
    let base_mid = 130.0_f64;
    let spread = 0.02;
    let tick = 0.01;
    let mut sequences = Vec::with_capacity(N_SEQUENCES);
    let mut mid_prices = Vec::new();

    for s in 0..N_SEQUENCES {
        let mut seq_features: Vec<FeatureVec> = Vec::with_capacity(WINDOW);
        for t in 0..WINDOW {
            let mid = base_mid + (s as f64 * 0.01) + (t as f64 * 0.0001);
            let best_ask = mid + spread / 2.0;
            let best_bid = mid - spread / 2.0;

            let mut v = vec![0.0_f64; N_FEATURES];

            for lvl in 0..10 {
                v[lvl] = best_ask + (lvl as f64) * tick;
            }
            for lvl in 0..10 {
                v[10 + lvl] = (1000 - lvl * 50) as f64;
            }
            for lvl in 0..10 {
                v[20 + lvl] = best_bid - (lvl as f64) * tick;
            }
            for lvl in 0..10 {
                v[30 + lvl] = (1000 - lvl * 50) as f64;
            }

            v[40] = mid;
            v[41] = spread;
            v[42] = spread / mid * 10_000.0;
            for i in 43..48 {
                v[i] = 0.5 + (i as f64) * 0.1;
            }
            for i in 48..84 {
                v[i] = 100.0 + (i as f64) * 1.5 + (s as f64 * 0.01);
            }
            v[84] = if s % 3 == 0 { 1.0 } else { -0.5 }; // OFI
            v[85] = 0.3;
            v[86] = 0.1;
            for i in 87..92 {
                v[i] = (i as f64) * 0.01;
            }
            v[92] = 1.0; // book_valid
            v[93] = 2.0; // time_regime
            v[94] = 1.0; // mbo_ready
            v[95] = s as f64 % 5.0;
            v[96] = 0.0;
            v[97] = contract::SCHEMA_VERSION;

            seq_features.push(Arc::new(v));
            mid_prices.push(mid);
        }

        sequences.push(Sequence {
            features: seq_features,
            start_timestamp: s as u64 * 1_000_000_000,
            end_timestamp: (s + 1) as u64 * 1_000_000_000,
            duration_ns: 1_000_000_000,
            length: WINDOW,
        });
    }

    PipelineOutput {
        sequences,
        mid_prices,
        messages_processed: N_SEQUENCES * WINDOW,
        features_extracted: N_SEQUENCES * WINDOW,
        sequences_generated: N_SEQUENCES,
        stride: STRIDE,
        window_size: WINDOW,
        multiscale_sequences: None,
        adaptive_stats: None,
    }
}

// =============================================================================
// Test 4a: Numerical Fidelity (f64 → f32 → .npy → f32 round-trip)
// =============================================================================

#[test]
fn test_export_numerical_fidelity() {
    let output = make_synthetic_output();
    let tmp = TempDir::new().unwrap();

    let exporter = AlignedBatchExporter::new(
        tmp.path(),
        LabelConfig {
            horizon: 50,
            smoothing_window: 10,
            threshold: 0.001,
        },
        WINDOW,
        STRIDE,
    );

    let result = exporter.export_day("fidelity", &output).unwrap();
    assert!(result.n_sequences > 0);

    let f = File::open(tmp.path().join("fidelity_sequences.npy")).unwrap();
    let arr: Array3<f32> = Array3::read_npy(f).unwrap();

    assert_eq!(arr.shape()[1], WINDOW, "window mismatch");
    assert_eq!(arr.shape()[2], N_FEATURES, "feature count mismatch");

    let n_exported = arr.shape()[0];
    let mut max_rel_err: f64 = 0.0;
    let mut total_checked: u64 = 0;
    let mut violations: Vec<String> = Vec::new();

    for seq_idx in 0..n_exported {
        let source_idx = seq_idx;
        if source_idx >= output.sequences.len() {
            break;
        }
        let source_seq = &output.sequences[source_idx];

        for t in 0..WINDOW {
            let source_snap = &source_seq.features[t];
            for feat in 0..N_FEATURES {
                let original = source_snap[feat];
                let exported = arr[[seq_idx, t, feat]] as f64;

                if !original.is_finite() || !exported.is_finite() {
                    violations.push(format!(
                        "seq[{seq_idx}][{t}][{feat}]: non-finite original={original}, exported={exported}"
                    ));
                    continue;
                }

                let abs_err = (original - exported).abs();
                let scale = original.abs().max(1.0);
                let rel_err = abs_err / scale;

                // f32 has ~7 decimal digits of precision.
                // Maximum relative error for f64→f32 cast: ~6e-8 (half ULP at f32 precision).
                // We use 1e-5 as a generous bound that catches real precision bugs
                // while tolerating the expected ~1e-7 quantization noise.
                if rel_err > 1e-5 {
                    violations.push(format!(
                        "seq[{seq_idx}][{t}][{feat}]: original={original}, exported={exported}, \
                         rel_err={rel_err:.2e}"
                    ));
                }

                if rel_err > max_rel_err {
                    max_rel_err = rel_err;
                }
                total_checked += 1;
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Numerical fidelity violations ({}):\n{}",
        violations.len(),
        violations[..violations.len().min(10)].join("\n")
    );

    println!("Numerical fidelity: {total_checked} values checked, max_rel_err={max_rel_err:.2e}");
}

// =============================================================================
// Test 4b: Multi-Horizon Label Shape and Value Validation
// =============================================================================

#[test]
fn test_multi_horizon_label_shapes() {
    let output = make_synthetic_output();
    let tmp = TempDir::new().unwrap();

    let label_config = LabelConfig {
        horizon: 50,
        smoothing_window: 10,
        threshold: 0.001,
    };

    let mh_config = MultiHorizonConfig::deeplob();

    let exporter = AlignedBatchExporter::new(tmp.path(), label_config, WINDOW, STRIDE)
        .with_multi_horizon_labels(mh_config);

    let result = exporter.export_day("mh", &output).unwrap();
    assert!(result.n_sequences > 0, "no sequences exported");

    let labels_path = tmp.path().join("mh_labels.npy");
    assert!(labels_path.exists(), "labels.npy not created");

    let f = File::open(&labels_path).unwrap();
    let labels: Array2<i8> = Array2::read_npy(f).unwrap();

    // DeepLOB config: 4 horizons [10, 20, 50, 100]
    assert_eq!(
        labels.shape()[0],
        result.n_sequences,
        "labels rows {} != n_sequences {}",
        labels.shape()[0],
        result.n_sequences
    );
    assert_eq!(
        labels.shape()[1],
        4,
        "DeepLOB should have 4 horizons, got {}",
        labels.shape()[1]
    );

    for seq_idx in 0..labels.shape()[0] {
        for h in 0..labels.shape()[1] {
            let v = labels[[seq_idx, h]];
            assert!(
                (-1..=1).contains(&v),
                "label[{seq_idx}][{h}] = {v}, expected in {{-1, 0, 1}}"
            );
        }
    }

    // Sequences .npy must still match labels dimension 0
    let seq_f = File::open(tmp.path().join("mh_sequences.npy")).unwrap();
    let seqs: Array3<f32> = Array3::read_npy(seq_f).unwrap();
    assert_eq!(
        seqs.shape()[0],
        labels.shape()[0],
        "sequences.shape[0]={} != labels.shape[0]={}",
        seqs.shape()[0],
        labels.shape()[0]
    );

    // Metadata should document multi-horizon via labeling.label_mode or horizons.json
    let meta_str = std::fs::read_to_string(tmp.path().join("mh_metadata.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    let labeling_mode = meta["labeling"]["label_mode"].as_str().unwrap_or("");
    let has_horizons_json = tmp.path().join("mh_horizons.json").exists();
    assert!(
        labeling_mode == "multi_horizon" || has_horizons_json,
        "metadata should indicate multi-horizon labeling (label_mode={labeling_mode:?}, \
         horizons.json exists={has_horizons_json})"
    );

    println!(
        "Multi-horizon validated: {} sequences x {} horizons, all labels in {{-1,0,1}}",
        labels.shape()[0],
        labels.shape()[1]
    );
}

// =============================================================================
// Test 4c: Raw Export Preserves No-Normalization Boundary
// =============================================================================

#[test]
fn test_raw_export_no_normalization() {
    let output = make_synthetic_output();
    let tmp = TempDir::new().unwrap();

    let exporter = AlignedBatchExporter::new(
        tmp.path(),
        LabelConfig {
            horizon: 50,
            smoothing_window: 10,
            threshold: 0.001,
        },
        WINDOW,
        STRIDE,
    );

    let result = exporter.export_day("raw", &output).unwrap();
    assert!(result.n_sequences > 0);

    let f = File::open(tmp.path().join("raw_sequences.npy")).unwrap();
    let arr: Array3<f32> = Array3::read_npy(f).unwrap();

    // Mid-price (index 40) in the first exported sequence, last timestep,
    // should be the raw value (not z-scored or scaled).
    // In our synthetic data, mid-price is ~130 + small offsets.
    // If normalization were applied, it would be ~0 (z-score) or in [0,1] (min-max).
    let mid_price_exported = arr[[0, WINDOW - 1, 40]] as f64;
    assert!(
        mid_price_exported > 100.0,
        "Mid-price in raw export = {mid_price_exported}, looks normalized (expected ~130)"
    );

    // Ask prices (index 0) should be raw dollar values
    let ask_price_exported = arr[[0, WINDOW - 1, 0]] as f64;
    assert!(
        ask_price_exported > 100.0,
        "Ask price in raw export = {ask_price_exported}, looks normalized (expected ~130)"
    );

    // Verify normalization.json indicates raw
    let norm_path = tmp.path().join("raw_normalization.json");
    if norm_path.exists() {
        let norm_str = std::fs::read_to_string(&norm_path).unwrap();
        let norm: serde_json::Value = serde_json::from_str(&norm_str).unwrap();
        if let Some(applied) = norm.get("normalization_applied") {
            assert_eq!(
                applied.as_bool(),
                Some(false),
                "normalization_applied should be false for raw export"
            );
        }
    }

    // Single-horizon labels: 1D array
    let lf = File::open(tmp.path().join("raw_labels.npy")).unwrap();
    let labels: Array1<i8> = Array1::read_npy(lf).unwrap();
    assert_eq!(
        labels.len(),
        result.n_sequences,
        "label count mismatch: {} vs {}",
        labels.len(),
        result.n_sequences
    );

    for &v in labels.iter() {
        assert!((-1..=1).contains(&v), "label {v} outside {{-1, 0, 1}}");
    }

    println!(
        "Raw export validated: {} sequences, mid_price={mid_price_exported:.4} (raw), \
         labels in {{-1,0,1}}",
        result.n_sequences
    );
}
