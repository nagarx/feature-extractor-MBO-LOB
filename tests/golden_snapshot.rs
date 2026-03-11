//! Golden Snapshot Regression Test
//!
//! Captures the first 500 post-warmup feature vectors from Feb 3 NVIDIA MBO data
//! as a permanent regression guard. On first run, generates the fixture. On
//! subsequent runs, compares new pipeline output against the saved fixture.
//!
//! Per-feature-group checksums (LOB, derived, MBO, signals) are validated
//! separately so regressions report exactly which feature group broke.
//!
//! Run with:
//! ```bash
//! cargo test --release --features "parallel,databento" --test golden_snapshot -- --test-threads=1
//! ```
//!
//! To regenerate the fixture (after intentional changes):
//! ```bash
//! rm tests/fixtures/golden_feb3_500.json
//! cargo test --release --features "parallel,databento" --test golden_snapshot -- --test-threads=1
//! ```

#![cfg(feature = "parallel")]

mod common;

use feature_extractor::builder::PipelineBuilder;
use feature_extractor::contract;
use feature_extractor::{Pipeline, PipelineConfig};
use std::path::PathBuf;

const GOLDEN_DATE: &str = "20250203";
const GOLDEN_COUNT: usize = 500;
const WARMUP_SKIP: usize = 50;

/// Feature group ranges for per-group regression reporting.
const LOB_RANGE: std::ops::Range<usize> = 0..40;
const DERIVED_RANGE: std::ops::Range<usize> = 40..48;
const MBO_RANGE: std::ops::Range<usize> = 48..84;
const SIGNAL_RANGE: std::ops::Range<usize> = 84..98;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("golden_feb3_500.json")
}

fn build_config() -> PipelineConfig {
    PipelineBuilder::new()
        .lob_levels(10)
        .with_trading_signals()
        .window(100, 10)
        .event_sampling(500)
        .build_config()
        .expect("Failed to build 98-feature config")
}

fn extract_golden_vectors(output: &feature_extractor::PipelineOutput) -> Vec<Vec<f64>> {
    let start = WARMUP_SKIP;
    let end = (start + GOLDEN_COUNT).min(output.sequences.len());

    output.sequences[start..end]
        .iter()
        .map(|seq| {
            let last = seq.features.last().expect("empty sequence");
            last.iter().copied().collect::<Vec<f64>>()
        })
        .collect()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GoldenFixture {
    date: String,
    schema_version: f64,
    feature_count: usize,
    warmup_skip: usize,
    vector_count: usize,
    messages_processed: usize,
    sequences_generated: usize,
    /// Per-feature-group sum checksums for fast regression localization.
    /// Summing all values in a group across all vectors.
    #[serde(default)]
    group_checksums: GroupChecksums,
    vectors: Vec<Vec<f64>>,
}

#[derive(serde::Serialize, serde::Deserialize, Default, Debug)]
struct GroupChecksums {
    lob_sum: f64,
    derived_sum: f64,
    mbo_sum: f64,
    signal_sum: f64,
}

fn compute_group_checksums(vectors: &[Vec<f64>]) -> GroupChecksums {
    let mut cs = GroupChecksums::default();
    for v in vectors {
        for i in LOB_RANGE.clone() {
            cs.lob_sum += v[i];
        }
        for i in DERIVED_RANGE.clone() {
            cs.derived_sum += v[i];
        }
        for i in MBO_RANGE.clone() {
            cs.mbo_sum += v[i];
        }
        for i in SIGNAL_RANGE.clone() {
            cs.signal_sum += v[i];
        }
    }
    cs
}

#[test]
fn test_golden_snapshot_regression() {
    skip_if_no_data!();

    let file = match common::find_mbo_file(GOLDEN_DATE) {
        Some(p) => p.to_string_lossy().to_string(),
        None => {
            eprintln!("Skipping: no data for {}", GOLDEN_DATE);
            return;
        }
    };

    let config = build_config();
    let mut pipeline = Pipeline::from_config(config).expect("pipeline");
    let output = pipeline.process(&file).expect("process");

    assert!(
        output.sequences.len() > WARMUP_SKIP + GOLDEN_COUNT,
        "Need at least {} sequences, got {}",
        WARMUP_SKIP + GOLDEN_COUNT,
        output.sequences.len()
    );

    let current_vectors = extract_golden_vectors(&output);
    assert_eq!(current_vectors.len(), GOLDEN_COUNT);
    assert_eq!(current_vectors[0].len(), contract::STABLE_FEATURE_COUNT);

    let fixture_file = fixture_path();

    if !fixture_file.exists() {
        let group_checksums = compute_group_checksums(&current_vectors);
        let fixture = GoldenFixture {
            date: GOLDEN_DATE.to_string(),
            schema_version: contract::SCHEMA_VERSION,
            feature_count: contract::STABLE_FEATURE_COUNT,
            warmup_skip: WARMUP_SKIP,
            vector_count: GOLDEN_COUNT,
            messages_processed: output.messages_processed,
            sequences_generated: output.sequences_generated,
            group_checksums,
            vectors: current_vectors,
        };

        let json = serde_json::to_string_pretty(&fixture).expect("serialize fixture");
        if let Some(parent) = fixture_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&fixture_file, json).expect("write fixture");

        println!(
            "GENERATED golden fixture: {} ({} vectors x {} features)",
            fixture_file.display(),
            GOLDEN_COUNT,
            contract::STABLE_FEATURE_COUNT
        );
        println!("Re-run this test to validate against the fixture.");
        return;
    }

    let fixture_content = std::fs::read_to_string(&fixture_file).expect("read fixture");
    let fixture: GoldenFixture = serde_json::from_str(&fixture_content).expect("parse fixture");

    assert_eq!(
        fixture.schema_version,
        contract::SCHEMA_VERSION,
        "Fixture schema_version {} != contract {}. Regenerate fixture if intentional.",
        fixture.schema_version,
        contract::SCHEMA_VERSION
    );
    assert_eq!(
        fixture.feature_count,
        contract::STABLE_FEATURE_COUNT,
        "Fixture feature_count {} != contract {}. Regenerate fixture if intentional.",
        fixture.feature_count,
        contract::STABLE_FEATURE_COUNT
    );
    assert_eq!(
        fixture.vector_count, GOLDEN_COUNT,
        "Fixture vector_count mismatch"
    );

    assert_eq!(
        output.messages_processed, fixture.messages_processed,
        "Messages processed changed: {} vs fixture {}. Data file may have changed.",
        output.messages_processed, fixture.messages_processed
    );

    // Per-feature-group checksum validation (fast regression localization)
    let current_checksums = compute_group_checksums(&current_vectors);
    let cs_tol = 1e-6;
    if (current_checksums.lob_sum - fixture.group_checksums.lob_sum).abs() > cs_tol {
        eprintln!(
            "GROUP REGRESSION: LOB features (0-39) checksum changed: {} -> {}",
            fixture.group_checksums.lob_sum, current_checksums.lob_sum
        );
    }
    if (current_checksums.derived_sum - fixture.group_checksums.derived_sum).abs() > cs_tol {
        eprintln!(
            "GROUP REGRESSION: Derived features (40-47) checksum changed: {} -> {}",
            fixture.group_checksums.derived_sum, current_checksums.derived_sum
        );
    }
    if (current_checksums.mbo_sum - fixture.group_checksums.mbo_sum).abs() > cs_tol {
        eprintln!(
            "GROUP REGRESSION: MBO features (48-83) checksum changed: {} -> {}",
            fixture.group_checksums.mbo_sum, current_checksums.mbo_sum
        );
    }
    if (current_checksums.signal_sum - fixture.group_checksums.signal_sum).abs() > cs_tol {
        eprintln!(
            "GROUP REGRESSION: Signal features (84-97) checksum changed: {} -> {}",
            fixture.group_checksums.signal_sum, current_checksums.signal_sum
        );
    }

    // Per-value regression detection (all features are deterministic after BTreeMap migration)
    let mut mismatches = 0u64;
    let mut total_checked = 0u64;
    let mut group_mismatches: [u64; 4] = [0; 4]; // LOB, derived, MBO, signals

    for (vec_idx, (current, golden)) in current_vectors
        .iter()
        .zip(fixture.vectors.iter())
        .enumerate()
    {
        assert_eq!(
            current.len(),
            golden.len(),
            "Vector {} length mismatch: {} vs {}",
            vec_idx,
            current.len(),
            golden.len()
        );

        for (feat_idx, (&cv, &gv)) in current.iter().zip(golden.iter()).enumerate() {
            total_checked += 1;
            let diff = (cv - gv).abs();
            if diff > contract::FLOAT_CMP_EPS {
                mismatches += 1;

                let group_idx = if LOB_RANGE.contains(&feat_idx) {
                    0
                } else if DERIVED_RANGE.contains(&feat_idx) {
                    1
                } else if MBO_RANGE.contains(&feat_idx) {
                    2
                } else {
                    3
                };
                group_mismatches[group_idx] += 1;

                if mismatches <= 10 {
                    let group_name = ["LOB", "Derived", "MBO", "Signal"][group_idx];
                    eprintln!(
                        "REGRESSION [{group_name}]: vec[{vec_idx}][{feat_idx}] = {cv} vs golden {gv} (diff={diff})",
                    );
                }
            }
        }
    }

    assert_eq!(
        mismatches,
        0,
        "Golden snapshot REGRESSION: {} mismatches in {} checks \
         (LOB={}, Derived={}, MBO={}, Signal={}). \
         If changes are intentional, delete {} and re-run to regenerate.",
        mismatches,
        total_checked,
        group_mismatches[0],
        group_mismatches[1],
        group_mismatches[2],
        group_mismatches[3],
        fixture_file.display()
    );

    println!(
        "Golden snapshot validated: {} vectors x {} features, {} checks passed (exact match at {:e})",
        GOLDEN_COUNT,
        contract::STABLE_FEATURE_COUNT,
        total_checked,
        contract::FLOAT_CMP_EPS
    );
}
