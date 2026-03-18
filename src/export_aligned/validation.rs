//! Export validation: alignment checks, spread verification, class balance, NaN/Inf scanning.

use super::types::LabelEncoding;
use super::AlignedBatchExporter;
use crate::contract;
use mbo_lob_reconstructor::Result;

/// Per-feature NaN/Inf counts from a full sequence scan.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct NanInfReport {
    pub is_clean: bool,
    pub total_values_scanned: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    /// Feature indices that contained at least one NaN or Inf.
    pub affected_features: Vec<usize>,
}

impl AlignedBatchExporter {
    /// Scan all aligned sequences for NaN/Inf values AFTER normalization.
    ///
    /// Normalization can itself introduce NaN (e.g., dividing by zero std for a
    /// constant feature), so this scan must run after normalization and before
    /// NPY write. Returns a hard error if any non-finite value is found --
    /// per RULE.md section 8: never silently write corrupt data.
    pub(super) fn scan_for_nan_inf(&self, sequences: &[Vec<Vec<f64>>]) -> Result<NanInfReport> {
        let mut total_scanned: usize = 0;
        let mut nan_count: usize = 0;
        let mut inf_count: usize = 0;
        let n_features = sequences
            .first()
            .and_then(|s| s.first())
            .map(|t| t.len())
            .unwrap_or(0);
        let mut feature_nan = vec![0usize; n_features];
        let mut feature_inf = vec![0usize; n_features];

        for seq in sequences {
            for timestep in seq {
                for (i, &val) in timestep.iter().enumerate() {
                    total_scanned += 1;
                    if val.is_nan() {
                        nan_count += 1;
                        if i < n_features {
                            feature_nan[i] += 1;
                        }
                    } else if val.is_infinite() {
                        inf_count += 1;
                        if i < n_features {
                            feature_inf[i] += 1;
                        }
                    }
                }
            }
        }

        let affected_features: Vec<usize> = (0..n_features)
            .filter(|&i| feature_nan[i] > 0 || feature_inf[i] > 0)
            .collect();

        let is_clean = nan_count == 0 && inf_count == 0;

        if !is_clean {
            let detail: Vec<String> = affected_features
                .iter()
                .map(|&i| {
                    format!(
                        "feature[{}]: {} NaN, {} Inf",
                        i, feature_nan[i], feature_inf[i]
                    )
                })
                .collect();
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Export integrity check FAILED: {} NaN + {} Inf in {} values. \
                     Affected features: [{}]",
                    nan_count,
                    inf_count,
                    total_scanned,
                    detail.join("; ")
                ),
            )
            .into());
        }

        Ok(NanInfReport {
            is_clean,
            total_values_scanned: total_scanned,
            nan_count,
            inf_count,
            affected_features,
        })
    }

    /// Assert that the feature count is within the valid contract range.
    ///
    /// Catches misconfiguration before corrupt data reaches disk. Valid counts
    /// are STABLE_FEATURE_COUNT (98) through FULL_FEATURE_COUNT (148), since
    /// experimental groups are mix-and-match (e.g., kolm_of without mlofi = 136).
    pub(super) fn validate_feature_count(&self, n_features: usize) -> Result<()> {
        if n_features < contract::STABLE_FEATURE_COUNT
            || n_features > contract::FULL_FEATURE_COUNT
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Feature count {} outside valid range [{}, {}]. \
                     Check FeatureConfig and experimental groups.",
                    n_features,
                    contract::STABLE_FEATURE_COUNT,
                    contract::FULL_FEATURE_COUNT,
                ),
            )
            .into());
        }
        Ok(())
    }
    /// Verify RAW spread integrity BEFORE normalization (DIAGNOSTIC)
    pub(super) fn verify_raw_spreads(&self, sequences: &[Vec<Vec<f64>>]) -> Result<()> {
        if sequences.is_empty() {
            return Ok(());
        }

        let mut total_spreads = 0;
        let mut positive_spreads = 0;
        let mut negative_spreads = 0;
        let mut spreads_sum = 0.0;

        // Feature layout (from lob_features.rs extract_raw_features):
        // [Ask_prices (10), Ask_sizes (10), Bid_prices (10), Bid_sizes (10)]
        //  0-9              10-19           20-29           30-39

        for seq in sequences {
            for timestep in seq {
                for level in 0..10 {
                    let ask_idx = level;
                    let bid_idx = 20 + level;

                    let ask_price = timestep[ask_idx];
                    let bid_price = timestep[bid_idx];
                    let spread = ask_price - bid_price;

                    total_spreads += 1;
                    spreads_sum += spread;

                    if spread > 0.0 {
                        positive_spreads += 1;
                    } else if spread < 0.0 {
                        negative_spreads += 1;
                    }
                }
            }
        }

        let pos_pct = 100.0 * positive_spreads as f64 / total_spreads as f64;
        let neg_pct = 100.0 * negative_spreads as f64 / total_spreads as f64;
        let mean_spread = spreads_sum / total_spreads as f64;

        println!("  🔍 RAW Spread Verification (BEFORE normalization):");
        println!("     Total spreads checked: {total_spreads}");
        println!("     Positive: {positive_spreads} ({pos_pct:.1}%)");
        println!("     Negative: {negative_spreads} ({neg_pct:.1}%)");
        println!("     Mean spread: ${mean_spread:.6}");

        if pos_pct < 99.0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "RAW spread integrity check FAILED: only {pos_pct:.1}% positive (expected >99%)"
                ),
            )
            .into());
        }

        println!("     ✅ RAW spreads are valid (ask > bid)");
        Ok(())
    }

    /// Validate label alignment between sequences and classification label matrix.
    ///
    /// For classification encodings, validates label range and reports class distribution.
    /// For regression encodings, skips classification-specific checks (regression labels
    /// are validated separately by `validate_regression_labels()`).
    pub(super) fn validate_label_alignment(
        &self,
        sequences: &[Vec<Vec<f64>>],
        label_matrix: &[Vec<i8>],
        encoding: &LabelEncoding,
    ) -> Result<()> {
        if sequences.len() != label_matrix.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Alignment error: {} sequences but {} label rows",
                    sequences.len(),
                    label_matrix.len()
                ),
            )
            .into());
        }

        if !label_matrix.is_empty() {
            let expected_width = label_matrix[0].len();
            for (i, row) in label_matrix.iter().enumerate() {
                if row.len() != expected_width {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Label row {} has wrong width: {} vs {}",
                            i,
                            row.len(),
                            expected_width
                        ),
                    )
                    .into());
                }
            }
        }

        // Classification-specific validation: range checks and distribution
        if let (Some((min_val, max_val)), Some(num_classes)) =
            (encoding.valid_range(), encoding.num_classes())
        {
            for (i, row) in label_matrix.iter().enumerate() {
                for (h, &label) in row.iter().enumerate() {
                    if label < min_val || label > max_val {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "Invalid {} label at row {}, horizon {}: {} (expected {})",
                                encoding.strategy_name(),
                                i,
                                h,
                                label,
                                encoding.expected_range_description()
                            ),
                        )
                        .into());
                    }
                }
            }

            let mut class_counts = vec![0usize; num_classes];
            for row in label_matrix {
                for &label in row {
                    let idx = (label - min_val) as usize;
                    if idx < num_classes {
                        class_counts[idx] += 1;
                    }
                }
            }
            let total = class_counts.iter().sum::<usize>();
            if total > 0 {
                if let Some(class_names) = encoding.class_names() {
                    let dist_str: Vec<String> = class_names
                        .iter()
                        .zip(class_counts.iter())
                        .map(|(name, &count)| {
                            format!("{}={:.1}%", name, count as f64 / total as f64 * 100.0)
                        })
                        .collect();
                    println!(
                        "    ✓ {} validation passed: {} sequences × {} horizons",
                        encoding.strategy_name(),
                        sequences.len(),
                        label_matrix.first().map(|r| r.len()).unwrap_or(0)
                    );
                    println!("      Class distribution: {}", dist_str.join(", "));
                }
            }
        }

        Ok(())
    }

    /// Validate regression label alignment, finiteness, shape, and range.
    ///
    /// Regression labels are float64 continuous values (forward returns in bps).
    /// This validation catches corrupt data before it reaches disk:
    /// - NaN/Inf values (silent computation failures)
    /// - Shape mismatches (regression labels vs sequences)
    /// - Extreme outliers (|value| > 10000 bps = 100% return, almost certainly a bug)
    /// - Inconsistent horizon widths across rows
    pub(super) fn validate_regression_labels(
        &self,
        sequences: &[Vec<Vec<f64>>],
        regression_labels: &[Vec<f64>],
    ) -> Result<()> {
        const MAX_REASONABLE_BPS: f64 = 10_000.0;

        if sequences.len() != regression_labels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Regression alignment error: {} sequences but {} regression label rows",
                    sequences.len(),
                    regression_labels.len()
                ),
            )
            .into());
        }

        if regression_labels.is_empty() {
            return Ok(());
        }

        let expected_width = regression_labels[0].len();
        let mut nan_count: usize = 0;
        let mut inf_count: usize = 0;
        let mut extreme_count: usize = 0;
        let mut total_values: usize = 0;

        for (i, row) in regression_labels.iter().enumerate() {
            if row.len() != expected_width {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Regression label row {} has wrong width: {} vs expected {}",
                        i,
                        row.len(),
                        expected_width
                    ),
                )
                .into());
            }

            for (h, &val) in row.iter().enumerate() {
                total_values += 1;
                if val.is_nan() {
                    nan_count += 1;
                    if nan_count <= 3 {
                        eprintln!(
                            "    Regression NaN at row {}, horizon {}: label is NaN",
                            i, h
                        );
                    }
                } else if val.is_infinite() {
                    inf_count += 1;
                    if inf_count <= 3 {
                        eprintln!(
                            "    Regression Inf at row {}, horizon {}: label is {:?}",
                            i, h, val
                        );
                    }
                } else if val.abs() > MAX_REASONABLE_BPS {
                    extreme_count += 1;
                }
            }
        }

        if nan_count > 0 || inf_count > 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Regression label integrity FAILED: {} NaN + {} Inf in {} values. \
                     Non-finite regression labels would corrupt model training.",
                    nan_count, inf_count, total_values
                ),
            )
            .into());
        }

        let extreme_pct = 100.0 * extreme_count as f64 / total_values.max(1) as f64;
        if extreme_pct > 1.0 {
            eprintln!(
                "    ⚠️  {:.1}% of regression labels exceed {} bps ({} of {}). \
                 This may indicate a computation bug.",
                extreme_pct, MAX_REASONABLE_BPS, extreme_count, total_values
            );
        }

        println!(
            "    ✓ Regression validation passed: {} sequences × {} horizons, all finite",
            sequences.len(),
            expected_width,
        );

        Ok(())
    }

    /// Validate class balance and warn if severely imbalanced.
    pub(super) fn validate_class_balance(
        &self,
        horizon: usize,
        profit_target_pct: f64,
        stop_loss_pct: f64,
        pt_rate: f64,
        sl_rate: f64,
        to_rate: f64,
    ) {
        const SEVERE_MAJORITY_THRESHOLD: f64 = 0.80;
        const MINORITY_THRESHOLD: f64 = 0.05;

        let mut warnings = Vec::new();

        if to_rate > SEVERE_MAJORITY_THRESHOLD {
            warnings.push(format!(
                "⚠️  SEVERE CLASS IMBALANCE at H{}: Timeout={:.1}% (>80%)",
                horizon,
                to_rate * 100.0
            ));
            warnings.push(format!(
                "    → Models will achieve ~{:.0}% accuracy by predicting only Timeout",
                to_rate * 100.0
            ));
            warnings.push(format!(
                "    → Current barriers: PT={:.1}bps, SL={:.1}bps are TOO TIGHT",
                profit_target_pct * 10000.0,
                stop_loss_pct * 10000.0
            ));
            warnings
                .push("    → Recommendation: REDUCE barriers. Run calibration tool:".to_string());
            warnings.push(format!(
                "       python tools/calibrate_triple_barrier.py --data-dir <export> --horizons {horizon}"
            ));
        }

        if pt_rate < MINORITY_THRESHOLD && pt_rate > 0.0 {
            warnings.push(format!(
                "⚠️  ProfitTarget only {:.1}% at H{}: May be too rare to learn",
                pt_rate * 100.0,
                horizon
            ));
        }
        if sl_rate < MINORITY_THRESHOLD && sl_rate > 0.0 {
            warnings.push(format!(
                "⚠️  StopLoss only {:.1}% at H{}: May be too rare to learn",
                sl_rate * 100.0,
                horizon
            ));
        }

        if pt_rate > 0.0 && sl_rate > 0.0 {
            let pt_sl_ratio = pt_rate / sl_rate;
            if !(0.2..=5.0).contains(&pt_sl_ratio) {
                warnings.push(format!(
                    "⚠️  PT/SL ratio {:.1}x at H{}: Consider adjusting barrier asymmetry",
                    pt_sl_ratio, horizon
                ));
            }
        }

        for warning in warnings {
            eprintln!("{}", warning);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::config::NormalizationConfig;
    use crate::labeling::LabelConfig;
    use std::path::PathBuf;

    fn test_exporter() -> AlignedBatchExporter {
        AlignedBatchExporter {
            output_dir: PathBuf::from("/tmp/test"),
            label_config: LabelConfig::new(50, 10, 0.0008),
            window_size: 100,
            stride: 10,
            tensor_format: None,
            feature_mapping: None,
            multi_horizon_config: None,
            regression_config: None,
            opportunity_configs: None,
            triple_barrier_configs: None,
            volatility_scaling: None,
            normalization_config: NormalizationConfig::default(),
            config_hash: None,
            export_forward_prices: false,
            smoothing_window_offset: 0,
        }
    }

    #[test]
    fn test_validate_regression_labels_valid() {
        let exporter = test_exporter();
        let sequences = vec![vec![vec![1.0; 98]; 100]; 5];
        let reg_labels = vec![vec![1.5, -2.3, 0.8]; 5];
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_ok(), "Valid regression labels should pass: {:?}", result.err());
    }

    #[test]
    fn test_validate_regression_labels_length_mismatch() {
        let exporter = test_exporter();
        let sequences = vec![vec![vec![1.0; 98]; 100]; 5];
        let reg_labels = vec![vec![1.0, 2.0]; 3];
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_err(), "Length mismatch should fail");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("5 sequences but 3"), "Error should mention counts: {err}");
    }

    #[test]
    fn test_validate_regression_labels_nan_rejected() {
        let exporter = test_exporter();
        let sequences = vec![vec![vec![1.0; 98]; 100]; 3];
        let reg_labels = vec![
            vec![1.0, 2.0],
            vec![f64::NAN, 3.0],
            vec![4.0, 5.0],
        ];
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_err(), "NaN should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN: {err}");
    }

    #[test]
    fn test_validate_regression_labels_inf_rejected() {
        let exporter = test_exporter();
        let sequences = vec![vec![vec![1.0; 98]; 100]; 2];
        let reg_labels = vec![
            vec![f64::INFINITY],
            vec![1.0],
        ];
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_err(), "Inf should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Inf"), "Error should mention Inf: {err}");
    }

    #[test]
    fn test_validate_regression_labels_inconsistent_width() {
        let exporter = test_exporter();
        let sequences = vec![vec![vec![1.0; 98]; 100]; 3];
        let reg_labels = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0, 8.0],
        ];
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_err(), "Inconsistent horizon width should fail");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("wrong width"), "Error should mention width: {err}");
    }

    #[test]
    fn test_validate_regression_labels_empty() {
        let exporter = test_exporter();
        let sequences: Vec<Vec<Vec<f64>>> = Vec::new();
        let reg_labels: Vec<Vec<f64>> = Vec::new();
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_ok(), "Empty should be valid");
    }

    #[test]
    fn test_validate_regression_labels_neg_inf_rejected() {
        let exporter = test_exporter();
        let sequences = vec![vec![vec![1.0; 98]; 100]; 1];
        let reg_labels = vec![vec![f64::NEG_INFINITY]];
        let result = exporter.validate_regression_labels(&sequences, &reg_labels);
        assert!(result.is_err(), "Negative infinity should be rejected");
    }
}
