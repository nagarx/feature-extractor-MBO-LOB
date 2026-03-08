//! Export validation: alignment checks, spread verification, class balance.

use super::types::LabelEncoding;
use super::AlignedBatchExporter;
use mbo_lob_reconstructor::Result;

impl AlignedBatchExporter {
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

    /// Validate label alignment between sequences and label matrix (multi-horizon).
    ///
    /// Unified validation for ALL labeling strategies. The `encoding` parameter
    /// defines the valid label range and class names.
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

        let (min_val, max_val) = encoding.valid_range();
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

        let num_classes = encoding.num_classes();
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
            let class_names = encoding.class_names();
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
