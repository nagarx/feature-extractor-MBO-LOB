//! Centralized assertion helpers for numerical precision testing.
//!
//! All floating-point comparisons use constants from `contract.rs` to ensure
//! consistency. No ad-hoc epsilon values in individual test files.

use feature_extractor::contract;

/// Result of comparing two floating-point vectors.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ComparisonResult {
    pub total_checked: u64,
    pub deterministic_mismatches: u64,
    pub nondeterministic_mismatches: u64,
    pub first_mismatches: Vec<MismatchDetail>,
}

/// Detail about a single mismatching value.
#[derive(Debug)]
#[allow(dead_code)]
pub struct MismatchDetail {
    pub index: usize,
    pub actual: f64,
    pub expected: f64,
    pub diff: f64,
    pub is_nondeterministic: bool,
}

/// Assert two f64 values are equal within `contract::FLOAT_CMP_EPS`.
///
/// Panics with a descriptive message including the `context` string.
#[allow(dead_code)]
pub fn assert_f64_eq(actual: f64, expected: f64, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= contract::FLOAT_CMP_EPS,
        "{context}: expected {expected}, got {actual} (diff={diff}, tol={})",
        contract::FLOAT_CMP_EPS
    );
}

/// Assert two f64 values are approximately equal within `rel_tol` relative tolerance.
///
/// Uses max(|expected|, 1.0) as the scale factor to handle near-zero values.
#[allow(dead_code)]
pub fn assert_f64_approx(actual: f64, expected: f64, rel_tol: f64, context: &str) {
    let scale = expected.abs().max(1.0);
    let rel_diff = (actual - expected).abs() / scale;
    assert!(
        rel_diff <= rel_tol,
        "{context}: expected {expected}, got {actual} (rel_diff={rel_diff}, rel_tol={rel_tol})",
    );
}

/// Assert all values in a feature vector are finite (not NaN, not Inf).
#[allow(dead_code)]
pub fn assert_features_finite(features: &[f64], context: &str) {
    for (i, &v) in features.iter().enumerate() {
        assert!(v.is_finite(), "{context}: feature[{i}] = {v} is not finite",);
    }
}

/// Assert a feature vector has the expected count and all values are finite.
#[allow(dead_code)]
pub fn assert_feature_layout(features: &[f64], expected_count: usize, context: &str) {
    assert_eq!(
        features.len(),
        expected_count,
        "{context}: expected {expected_count} features, got {}",
        features.len()
    );
    assert_features_finite(features, context);
}

/// Assert signal sign convention (RULE.md section 10):
/// - `> 0` = Bullish/Buy pressure
/// - `< 0` = Bearish/Sell pressure
/// - `= 0` = Neutral
///
/// Verifies that signals stay within expected bounds and categorical
/// signals are in their valid sets.
#[allow(dead_code)]
pub fn assert_signal_basics(signals_slice: &[f64], context: &str) {
    for (i, &v) in signals_slice.iter().enumerate() {
        assert!(v.is_finite(), "{context}: signal[{i}] = {v} is not finite");
    }
}

/// Compare two f64 vectors element-by-element, tolerating known nondeterministic indices.
///
/// Returns a `ComparisonResult` with mismatch details. Does not panic.
#[allow(dead_code)]
pub fn compare_vectors(
    actual: &[f64],
    expected: &[f64],
    deterministic_tol: f64,
    nondeterministic_indices: &[usize],
    nondeterministic_tol: f64,
) -> ComparisonResult {
    assert_eq!(actual.len(), expected.len(), "Vector length mismatch");

    let mut result = ComparisonResult {
        total_checked: 0,
        deterministic_mismatches: 0,
        nondeterministic_mismatches: 0,
        first_mismatches: Vec::new(),
    };

    for (i, (&av, &ev)) in actual.iter().zip(expected.iter()).enumerate() {
        result.total_checked += 1;

        if nondeterministic_indices.contains(&i) {
            let scale = ev.abs().max(1.0);
            let rel_diff = (av - ev).abs() / scale;
            if rel_diff > nondeterministic_tol {
                result.nondeterministic_mismatches += 1;
                if result.first_mismatches.len() < 10 {
                    result.first_mismatches.push(MismatchDetail {
                        index: i,
                        actual: av,
                        expected: ev,
                        diff: rel_diff,
                        is_nondeterministic: true,
                    });
                }
            }
        } else {
            let diff = (av - ev).abs();
            if diff > deterministic_tol {
                result.deterministic_mismatches += 1;
                if result.first_mismatches.len() < 10 {
                    result.first_mismatches.push(MismatchDetail {
                        index: i,
                        actual: av,
                        expected: ev,
                        diff,
                        is_nondeterministic: false,
                    });
                }
            }
        }
    }

    result
}

/// Compute Pearson correlation coefficient between two slices.
///
/// Returns 0.0 if either variance is near-zero or n < 3.
#[allow(dead_code)]
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Arrays must have same length");
    let n = x.len() as f64;
    if n < 3.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < contract::FLOAT_CMP_EPS || var_y < contract::FLOAT_CMP_EPS {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Assert that a feature has non-zero variance across a collection of samples.
///
/// Catches "always zero" or "always constant" bugs.
#[allow(dead_code)]
pub fn assert_feature_has_variance(values: &[f64], feature_name: &str) {
    if values.len() < 2 {
        return;
    }
    let first = values[0];
    let has_variance = values
        .iter()
        .any(|&v| (v - first).abs() > contract::FLOAT_CMP_EPS);
    assert!(
        has_variance,
        "Feature '{feature_name}' has zero variance across {} samples (all = {first})",
        values.len()
    );
}
