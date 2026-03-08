//! TLOB labeling strategy: single-horizon and multi-horizon.

use crate::export_aligned::types::{LabelEncoding, LabelingResult};
use crate::export_aligned::AlignedBatchExporter;
use crate::labeling::{MultiHorizonConfig, MultiHorizonLabelGenerator};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use std::collections::HashMap;

impl AlignedBatchExporter {
    /// Single-horizon TLOB labeling strategy.
    pub(crate) fn labeling_single_horizon_tlob(
        &self,
        output: &PipelineOutput,
    ) -> Result<LabelingResult> {
        println!("  📊 Generating single-horizon labels...");
        let (label_indices, label_values, label_dist) = self.generate_labels(&output.mid_prices)?;

        println!(
            "    Generated {} labels from {} mid-prices",
            label_indices.len(),
            output.mid_prices.len()
        );
        println!(
            "    Label range: indices [{}, {}]",
            label_indices[0],
            label_indices[label_indices.len() - 1]
        );

        let label_matrix: Vec<Vec<i8>> = label_values.iter().map(|&v| vec![v]).collect();

        let strategy_metadata = serde_json::json!({
            "label_mode": "single_horizon",
            "label_encoding": {
                "format": "signed_int8",
                "values": { "-1": "Down", "0": "Stable", "1": "Up" },
            },
        });

        Ok(LabelingResult {
            label_indices,
            label_matrix,
            encoding: LabelEncoding::SignedTrend,
            distribution: label_dist,
            strategy_name: "tlob",
            strategy_metadata,
            is_multi_horizon: false,
            horizons_config: None,
        })
    }

    /// Multi-horizon TLOB labeling strategy (FI-2010, DeepLOB benchmarks).
    pub(crate) fn labeling_multi_horizon_tlob(
        &self,
        output: &PipelineOutput,
        config: &MultiHorizonConfig,
    ) -> Result<LabelingResult> {
        let horizons = config.horizons().to_vec();

        println!(
            "  📊 Generating multi-horizon labels for {} horizons: {:?}",
            horizons.len(),
            horizons
        );

        let mut generator = MultiHorizonLabelGenerator::new(config.clone());
        generator.add_prices(&output.mid_prices);
        let multi_labels = generator.generate_labels()?;

        let summary = multi_labels.summary();
        println!(
            "    Generated {} total labels across {} horizons",
            summary.total_labels, summary.num_horizons
        );

        let (label_indices, label_matrix) =
            self.build_multi_horizon_label_matrix(&multi_labels, &horizons)?;

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            label_indices.len()
        );

        // Per-horizon distributions
        let mut label_dist_per_horizon = HashMap::new();
        for (h_idx, horizon) in horizons.iter().enumerate() {
            let mut h_dist = HashMap::new();
            let (mut up_count, mut down_count, mut stable_count) = (0usize, 0usize, 0usize);
            for row in &label_matrix {
                match row[h_idx] {
                    1 => up_count += 1,
                    -1 => down_count += 1,
                    0 => stable_count += 1,
                    _ => {}
                }
            }
            h_dist.insert("Up".to_string(), up_count);
            h_dist.insert("Down".to_string(), down_count);
            h_dist.insert("Stable".to_string(), stable_count);
            label_dist_per_horizon.insert(format!("h{}", horizon), h_dist);
        }

        // Combined distribution from first horizon
        let mut combined_dist = HashMap::new();
        combined_dist.insert("Up".to_string(), 0);
        combined_dist.insert("Down".to_string(), 0);
        combined_dist.insert("Stable".to_string(), 0);
        if let Some(first_dist) = label_dist_per_horizon.get(&format!("h{}", horizons[0])) {
            combined_dist = first_dist.clone();
        }

        let horizons_config = serde_json::json!({
            "horizons": horizons,
            "num_horizons": horizons.len(),
            "smoothing_window": config.smoothing_window,
            "threshold_strategy": format!("{:?}", config.threshold_strategy),
        });

        let strategy_metadata = serde_json::json!({
            "label_mode": "multi_horizon",
            "horizons": horizons,
            "num_horizons": horizons.len(),
            "label_distribution_per_horizon": label_dist_per_horizon,
            "label_encoding": {
                "format": "signed_int8",
                "values": { "-1": "Down", "0": "Stable", "1": "Up" },
            },
        });

        Ok(LabelingResult {
            label_indices,
            label_matrix,
            encoding: LabelEncoding::SignedTrend,
            distribution: combined_dist,
            strategy_name: "tlob",
            strategy_metadata,
            is_multi_horizon: true,
            horizons_config: Some(horizons_config),
        })
    }
}
