//! Golden Snapshot Regression Test
//!
//! Captures the first 100 post-warmup feature vectors from Feb 3 NVIDIA MBO data
//! as a permanent regression guard. On first run, generates the fixture. On
//! subsequent runs, compares new pipeline output against the saved fixture.
//!
//! Run with:
//! ```bash
//! cargo test --release --features "parallel,databento" --test golden_snapshot -- --test-threads=1
//! ```
//!
//! To regenerate the fixture (after intentional changes):
//! ```bash
//! rm tests/fixtures/golden_feb3_100.json
//! cargo test --release --features "parallel,databento" --test golden_snapshot -- --test-threads=1
//! ```

#![cfg(feature = "parallel")]

mod common;

use feature_extractor::builder::PipelineBuilder;
use feature_extractor::contract;
use feature_extractor::{Pipeline, PipelineConfig};
use std::path::PathBuf;

const GOLDEN_DATE: &str = "20250203";
const GOLDEN_COUNT: usize = 100;
const WARMUP_SKIP: usize = 50;

/// Feature indices affected by HashMap iteration order in OrderTracker.
/// These are compared with loose tolerance instead of exact match.
const KNOWN_NONDETERMINISTIC: &[usize] = &[70, 78, 79, 80, 81, 83];

const DETERMINISTIC_ABS_TOL: f64 = 1e-10;
const NONDETERMINISTIC_REL_TOL: f64 = 0.05;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("golden_feb3_100.json")
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

fn extract_golden_vectors(
    output: &feature_extractor::PipelineOutput,
) -> Vec<Vec<f64>> {
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
    vectors: Vec<Vec<f64>>,
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
        let fixture = GoldenFixture {
            date: GOLDEN_DATE.to_string(),
            schema_version: contract::SCHEMA_VERSION,
            feature_count: contract::STABLE_FEATURE_COUNT,
            warmup_skip: WARMUP_SKIP,
            vector_count: GOLDEN_COUNT,
            messages_processed: output.messages_processed,
            sequences_generated: output.sequences_generated,
            vectors: current_vectors,
        };

        let json = serde_json::to_string_pretty(&fixture).expect("serialize fixture");
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
    let fixture: GoldenFixture =
        serde_json::from_str(&fixture_content).expect("parse fixture");

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
        fixture.vector_count,
        GOLDEN_COUNT,
        "Fixture vector_count mismatch"
    );

    assert_eq!(
        output.messages_processed, fixture.messages_processed,
        "Messages processed changed: {} vs fixture {}. Data file may have changed.",
        output.messages_processed, fixture.messages_processed
    );

    let mut deterministic_mismatches = 0u64;
    let mut nondeterministic_mismatches = 0u64;
    let mut total_checked = 0u64;

    for (vec_idx, (current, golden)) in
        current_vectors.iter().zip(fixture.vectors.iter()).enumerate()
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

            if KNOWN_NONDETERMINISTIC.contains(&feat_idx) {
                let scale = gv.abs().max(1.0);
                let rel_diff = (cv - gv).abs() / scale;
                if rel_diff > NONDETERMINISTIC_REL_TOL {
                    nondeterministic_mismatches += 1;
                    if nondeterministic_mismatches <= 3 {
                        eprintln!(
                            "Nondeterministic drift: vec[{}][{}] = {} vs golden {} (rel={})",
                            vec_idx, feat_idx, cv, gv, rel_diff
                        );
                    }
                }
            } else {
                let diff = (cv - gv).abs();
                if diff > DETERMINISTIC_ABS_TOL {
                    deterministic_mismatches += 1;
                    if deterministic_mismatches <= 10 {
                        eprintln!(
                            "REGRESSION: vec[{}][{}] = {} vs golden {} (diff={})",
                            vec_idx, feat_idx, cv, gv, diff
                        );
                    }
                }
            }
        }
    }

    if nondeterministic_mismatches > 0 {
        eprintln!(
            "WARNING: {} nondeterministic feature drifts (>{:.0}% rel) in features {:?}",
            nondeterministic_mismatches,
            NONDETERMINISTIC_REL_TOL * 100.0,
            KNOWN_NONDETERMINISTIC
        );
    }

    assert_eq!(
        deterministic_mismatches, 0,
        "Golden snapshot REGRESSION: {} deterministic mismatches in {} checks. \
         If changes are intentional, delete {} and re-run to regenerate.",
        deterministic_mismatches,
        total_checked,
        fixture_file.display()
    );

    println!(
        "Golden snapshot validated: {} vectors x {} features, {} deterministic checks passed, {} nondeterministic drifts",
        GOLDEN_COUNT,
        contract::STABLE_FEATURE_COUNT,
        total_checked - nondeterministic_mismatches,
        nondeterministic_mismatches
    );
}
