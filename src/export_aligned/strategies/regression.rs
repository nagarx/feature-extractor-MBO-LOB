//! Regression labeling strategy: continuous forward returns in bps.
//!
//! Supports multiple return formulas via `RegressionReturnType`:
//! - `SmoothedReturn`: TLOB-style smoothed average (via MultiHorizonLabelGenerator)
//! - `PointReturn`, `PeakReturn`, `MeanReturn`, `DominantReturn`: via MagnitudeGenerator
//!
//! Export format: `{day}_regression_labels.npy` as float64 [N, num_horizons].
//! Values are forward returns in basis points (1 bps = 0.01% = 0.0001).

use std::collections::HashMap;

use crate::export::config::{RegressionExportConfig, RegressionReturnType};
use crate::export_aligned::types::{LabelEncoding, LabelingResult};
use crate::export_aligned::AlignedBatchExporter;
use crate::labeling::{MagnitudeConfig, MagnitudeGenerator, MultiHorizonLabelGenerator, ReturnType};
use crate::prelude::PipelineOutput;
use mbo_lob_reconstructor::Result;

impl AlignedBatchExporter {
    /// Regression labeling: continuous forward returns in bps at each horizon.
    ///
    /// The return formula is selected by `config.return_type`:
    /// - `SmoothedReturn`: TLOB-style smoothed average (default, established path)
    /// - Other types: Use `MagnitudeGenerator` for point/peak/mean/dominant returns
    ///
    /// Pure regression mode: only produces float64 regression labels.
    pub(crate) fn labeling_regression(
        &self,
        output: &PipelineOutput,
        config: &RegressionExportConfig,
    ) -> Result<LabelingResult> {
        match config.return_type {
            RegressionReturnType::SmoothedReturn => {
                self.labeling_regression_smoothed(output, config)
            }
            _ => self.labeling_regression_magnitude(output, config),
        }
    }

    /// Smoothed-return regression: TLOB-style smoothed average returns in bps.
    ///
    /// Formula: `avg(price[t+h-k:t+h+1]) / avg(price[t-k:t+1]) - 1`
    /// where k = smoothing_window. Both windows have length k+1.
    fn labeling_regression_smoothed(
        &self,
        output: &PipelineOutput,
        config: &RegressionExportConfig,
    ) -> Result<LabelingResult> {
        let mh_config = &config.multi_horizon_config;
        let horizons = mh_config.horizons().to_vec();

        println!(
            "  📊 Generating regression labels (smoothed bps) for {} horizons: {:?}",
            horizons.len(),
            horizons
        );

        let mut generator = MultiHorizonLabelGenerator::new(mh_config.clone());
        generator.add_prices(&output.mid_prices);
        let multi_labels = generator.generate_labels()?;

        let summary = multi_labels.summary();
        println!(
            "    Generated {} total labels across {} horizons",
            summary.total_labels, summary.num_horizons
        );

        // Find indices valid across ALL horizons
        let valid_indices = Self::compute_valid_indices(&multi_labels, &horizons)?;

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            valid_indices.len()
        );

        // Build per-horizon return maps: index -> bps_return
        let mut horizon_return_maps: Vec<HashMap<usize, f64>> = Vec::new();
        for horizon in &horizons {
            let labels = multi_labels.labels_for_horizon(*horizon).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("No labels generated for horizon {horizon}"),
                )
            })?;
            let return_map: HashMap<usize, f64> = labels
                .iter()
                .map(|(idx, _, pct_change)| (*idx, *pct_change * 10000.0))
                .collect();
            horizon_return_maps.push(return_map);
        }

        let regression_matrix =
            Self::build_regression_matrix(&valid_indices, &horizons, &horizon_return_maps);

        Self::build_regression_result(
            valid_indices,
            regression_matrix,
            &horizons,
            mh_config.smoothing_window,
            &format!("{:?}", mh_config.threshold_strategy),
            config.return_type,
        )
    }

    /// Magnitude-based regression: uses MagnitudeGenerator for various return formulas.
    ///
    /// Supports PointReturn, PeakReturn, MeanReturn, DominantReturn.
    fn labeling_regression_magnitude(
        &self,
        output: &PipelineOutput,
        config: &RegressionExportConfig,
    ) -> Result<LabelingResult> {
        let horizons = config.multi_horizon_config.horizons().to_vec();
        let return_type = match config.return_type {
            RegressionReturnType::PointReturn => ReturnType::PointReturn,
            RegressionReturnType::PeakReturn | RegressionReturnType::DominantReturn => {
                ReturnType::DominantReturn
            }
            RegressionReturnType::MeanReturn => ReturnType::MeanReturn,
            RegressionReturnType::SmoothedReturn => unreachable!("handled in dispatch"),
        };

        println!(
            "  📊 Generating regression labels ({:?}, bps) for {} horizons: {:?}",
            config.return_type,
            horizons.len(),
            horizons
        );

        // Generate returns for each horizon using MagnitudeGenerator
        let mut per_horizon_returns: Vec<HashMap<usize, f64>> = Vec::new();
        let mut min_valid_count = usize::MAX;

        for &horizon in &horizons {
            let mag_config = MagnitudeConfig {
                horizons: vec![horizon],
                return_type,
                compute_all_stats: false,
                smoothing_window: None,
            };
            let mut generator = MagnitudeGenerator::new(mag_config);
            generator.add_prices(&output.mid_prices);
            let returns = generator.generate_returns()?;

            let return_map: HashMap<usize, f64> = returns
                .iter()
                .map(|(idx, ret_data)| {
                    let value = match return_type {
                        ReturnType::PointReturn => ret_data.point_return,
                        ReturnType::DominantReturn => ret_data.dominant_return(),
                        ReturnType::MeanReturn => ret_data.mean_return,
                        ReturnType::MaxReturn => ret_data.max_return,
                        ReturnType::MinReturn => ret_data.min_return,
                    };
                    (*idx, value * 10000.0)
                })
                .collect();

            min_valid_count = min_valid_count.min(return_map.len());
            per_horizon_returns.push(return_map);
        }

        println!(
            "    Generated returns for {} horizons (min valid: {})",
            horizons.len(),
            min_valid_count
        );

        // Find indices valid across ALL horizons
        let valid_indices = {
            let mut sets: Vec<std::collections::HashSet<usize>> = per_horizon_returns
                .iter()
                .map(|m| m.keys().copied().collect())
                .collect();
            let mut intersection = sets.remove(0);
            for set in &sets {
                intersection = intersection.intersection(set).copied().collect();
            }
            let mut sorted: Vec<usize> = intersection.into_iter().collect();
            sorted.sort_unstable();
            sorted
        };

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            valid_indices.len()
        );

        let regression_matrix =
            Self::build_regression_matrix(&valid_indices, &horizons, &per_horizon_returns);

        Self::build_regression_result(
            valid_indices,
            regression_matrix,
            &horizons,
            0,
            &format!("{:?}", config.return_type),
            config.return_type,
        )
    }

    // ========================================================================
    // Shared helpers
    // ========================================================================

    fn compute_valid_indices(
        multi_labels: &crate::labeling::MultiHorizonLabels,
        horizons: &[usize],
    ) -> Result<Vec<usize>> {
        let mut index_sets: Vec<std::collections::HashSet<usize>> = Vec::new();
        for horizon in horizons {
            let labels = multi_labels.labels_for_horizon(*horizon).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("No labels generated for horizon {horizon} in compute_valid_indices"),
                )
            })?;
            let set: std::collections::HashSet<usize> =
                labels.iter().map(|(idx, _, _)| *idx).collect();
            index_sets.push(set);
        }
        let mut intersection = index_sets[0].clone();
        for set in &index_sets[1..] {
            intersection = intersection.intersection(set).copied().collect();
        }
        let mut sorted: Vec<usize> = intersection.into_iter().collect();
        sorted.sort_unstable();
        Ok(sorted)
    }

    fn build_regression_matrix(
        valid_indices: &[usize],
        horizons: &[usize],
        horizon_return_maps: &[HashMap<usize, f64>],
    ) -> Vec<Vec<f64>> {
        let mut regression_matrix: Vec<Vec<f64>> = Vec::with_capacity(valid_indices.len());
        for &idx in valid_indices {
            let mut return_row: Vec<f64> = Vec::with_capacity(horizons.len());
            for (h_idx, _) in horizons.iter().enumerate() {
                return_row.push(*horizon_return_maps[h_idx].get(&idx).unwrap_or(&0.0));
            }
            regression_matrix.push(return_row);
        }
        regression_matrix
    }

    fn build_regression_result(
        valid_indices: Vec<usize>,
        regression_matrix: Vec<Vec<f64>>,
        horizons: &[usize],
        smoothing_window: usize,
        threshold_info: &str,
        return_type: RegressionReturnType,
    ) -> Result<LabelingResult> {
        let mut combined_dist = HashMap::new();
        let all_returns: Vec<f64> = regression_matrix
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        let n = all_returns.len().max(1) as f64;
        let mean_bps = all_returns.iter().sum::<f64>() / n;
        let abs_mean_bps = all_returns.iter().map(|v| v.abs()).sum::<f64>() / n;
        let positive = all_returns.iter().filter(|v| **v > 0.0).count();
        let negative = all_returns.iter().filter(|v| **v < 0.0).count();
        let zero = all_returns.iter().filter(|v| v.abs() < 1e-10).count();
        combined_dist.insert("positive".to_string(), positive);
        combined_dist.insert("negative".to_string(), negative);
        combined_dist.insert("zero".to_string(), zero);

        println!(
            "    Regression stats: mean={:.2} bps, |mean|={:.2} bps, positive={:.1}%, negative={:.1}%",
            mean_bps,
            abs_mean_bps,
            100.0 * positive as f64 / n,
            100.0 * negative as f64 / n,
        );

        let horizons_config = serde_json::json!({
            "horizons": horizons,
            "num_horizons": horizons.len(),
            "smoothing_window": smoothing_window,
            "threshold_strategy": threshold_info,
        });

        let strategy_metadata = serde_json::json!({
            "label_mode": "regression",
            "return_type": format!("{:?}", return_type),
            "horizons": horizons,
            "num_horizons": horizons.len(),
            "label_encoding": {
                "format": "continuous_bps",
                "dtype": "float64",
                "unit": "basis_points",
                "description": format!("{:?} forward return in bps at each horizon", return_type),
            },
            "regression_stats": {
                "mean_bps": mean_bps,
                "abs_mean_bps": abs_mean_bps,
                "n_positive": positive,
                "n_negative": negative,
                "n_zero": zero,
            },
        });

        Ok(LabelingResult {
            label_indices: valid_indices,
            classification_labels: None,
            regression_labels: Some(regression_matrix),
            encoding: LabelEncoding::ContinuousBps,
            distribution: combined_dist,
            strategy_name: "regression",
            strategy_metadata,
            is_multi_horizon: true,
            horizons_config: Some(horizons_config),
        })
    }
}
