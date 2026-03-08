//! Opportunity labeling strategy: big-move detection.

use crate::export_aligned::types::{LabelEncoding, LabelingResult};
use crate::export_aligned::AlignedBatchExporter;
use crate::labeling::{OpportunityConfig, OpportunityLabel, OpportunityLabelGenerator};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use std::collections::HashMap;

type LabelMatrix = (Vec<usize>, Vec<Vec<i8>>, HashMap<String, usize>);

impl AlignedBatchExporter {
    /// Opportunity-based labeling strategy.
    pub(crate) fn labeling_opportunity(
        &self,
        output: &PipelineOutput,
        configs: &[OpportunityConfig],
    ) -> Result<LabelingResult> {
        let horizons: Vec<usize> = configs.iter().map(|c| c.horizon).collect();
        let is_multi_horizon = configs.len() > 1;

        println!(
            "  📊 Generating OPPORTUNITY labels for {} horizon(s): {:?}",
            horizons.len(),
            horizons
        );
        println!(
            "    Threshold: {:.4}% ({:.1} bps)",
            configs[0].threshold * 100.0,
            configs[0].threshold * 10000.0
        );

        let mut all_labels: Vec<Vec<(usize, OpportunityLabel, f64, f64)>> = Vec::new();

        for config in configs {
            let mut generator = OpportunityLabelGenerator::new(config.clone());
            generator.add_prices(&output.mid_prices);
            let labels = generator.generate_labels()?;

            println!(
                "    Horizon {}: {} labels, {:.1}% opportunities",
                config.horizon,
                labels.len(),
                labels
                    .iter()
                    .filter(|(_, l, _, _)| l.is_opportunity())
                    .count() as f64
                    / labels.len() as f64
                    * 100.0
            );

            all_labels.push(labels);
        }

        let (label_indices, label_matrix, combined_dist) =
            Self::build_opportunity_label_matrix(&all_labels)?;

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            label_indices.len()
        );

        let horizons_config = serde_json::json!({
            "labeling_strategy": "opportunity",
            "horizons": horizons,
            "threshold": configs[0].threshold,
            "threshold_bps": configs[0].threshold * 10000.0,
            "conflict_priority": format!("{:?}", configs[0].conflict_priority),
        });

        let strategy_metadata = serde_json::json!({
            "strategy": "opportunity",
            "horizons": horizons,
            "threshold": configs[0].threshold,
            "threshold_bps": configs[0].threshold * 10000.0,
            "conflict_priority": format!("{:?}", configs[0].conflict_priority),
            "label_encoding": {
                "format": "signed_int8",
                "values": {
                    "-1": "BigDown",
                    "0": "NoOpportunity",
                    "1": "BigUp",
                },
                "class_index_mapping": "class_idx = label + 1  # For softmax: -1→0, 0→1, 1→2",
            },
        });

        Ok(LabelingResult {
            label_indices,
            label_matrix,
            encoding: LabelEncoding::SignedOpportunity,
            distribution: combined_dist,
            strategy_name: "opportunity",
            strategy_metadata,
            is_multi_horizon,
            horizons_config: Some(horizons_config),
        })
    }

    /// Build label matrix for opportunity labeling.
    fn build_opportunity_label_matrix(
        all_labels: &[Vec<(usize, OpportunityLabel, f64, f64)>],
    ) -> Result<LabelMatrix> {
        use std::collections::BTreeSet;

        let mut valid_indices: Option<BTreeSet<usize>> = None;

        for horizon_labels in all_labels {
            let indices: BTreeSet<usize> =
                horizon_labels.iter().map(|(idx, _, _, _)| *idx).collect();
            valid_indices = match valid_indices {
                None => Some(indices),
                Some(prev) => Some(prev.intersection(&indices).cloned().collect()),
            };
        }

        let valid_indices: Vec<usize> = valid_indices.unwrap_or_default().into_iter().collect();

        if valid_indices.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No valid indices found across all horizons",
            )
            .into());
        }

        let mut horizon_lookups: Vec<HashMap<usize, OpportunityLabel>> = Vec::new();
        for horizon_labels in all_labels {
            let lookup: HashMap<usize, OpportunityLabel> = horizon_labels
                .iter()
                .map(|(idx, label, _, _)| (*idx, *label))
                .collect();
            horizon_lookups.push(lookup);
        }

        let mut label_matrix = Vec::with_capacity(valid_indices.len());
        let mut dist = HashMap::new();
        dist.insert("BigDown".to_string(), 0);
        dist.insert("NoOpportunity".to_string(), 0);
        dist.insert("BigUp".to_string(), 0);

        for &idx in &valid_indices {
            let mut row = Vec::with_capacity(all_labels.len());
            for lookup in &horizon_lookups {
                let label = lookup.get(&idx).unwrap_or(&OpportunityLabel::NoOpportunity);
                row.push(label.as_int());
                *dist.entry(label.name().to_string()).or_insert(0) += 1;
            }
            label_matrix.push(row);
        }

        Ok((valid_indices, label_matrix, dist))
    }
}
