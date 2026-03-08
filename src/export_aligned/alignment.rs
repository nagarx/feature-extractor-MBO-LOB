//! Sequence-label alignment for single and multi-horizon labeling.

use super::AlignedBatchExporter;
use crate::labeling::TrendLabel;
use mbo_lob_reconstructor::Result;
use std::collections::HashMap;

impl AlignedBatchExporter {
    /// Generate labels from mid-prices (single-horizon TLOB).
    ///
    /// Returns: (indices, label_values, distribution)
    #[allow(clippy::type_complexity)]
    pub(super) fn generate_labels(
        &self,
        mid_prices: &[f64],
    ) -> Result<(Vec<usize>, Vec<i8>, HashMap<String, usize>)> {
        if mid_prices.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No mid-prices provided for labeling",
            )
            .into());
        }

        let mut generator = crate::labeling::TlobLabelGenerator::new(self.label_config.clone());
        generator.add_prices(mid_prices);

        let label_results = generator.generate_labels()?;

        let mut indices = Vec::with_capacity(label_results.len());
        let mut values = Vec::with_capacity(label_results.len());
        let mut distribution = HashMap::new();
        distribution.insert("Up".to_string(), 0);
        distribution.insert("Down".to_string(), 0);
        distribution.insert("Stable".to_string(), 0);

        for (idx, label, _pct_change) in label_results {
            indices.push(idx);

            let value = match label {
                TrendLabel::Up => {
                    *distribution.get_mut("Up").unwrap() += 1;
                    1
                }
                TrendLabel::Down => {
                    *distribution.get_mut("Down").unwrap() += 1;
                    -1
                }
                TrendLabel::Stable => {
                    *distribution.get_mut("Stable").unwrap() += 1;
                    0
                }
            };
            values.push(value);
        }

        Ok((indices, values, distribution))
    }

    /// Build label matrix for multi-horizon labeling.
    ///
    /// Returns (indices, label_matrix) where each row has labels for all horizons.
    #[allow(clippy::type_complexity)]
    pub(super) fn build_multi_horizon_label_matrix(
        &self,
        multi_labels: &crate::labeling::MultiHorizonLabels,
        horizons: &[usize],
    ) -> Result<(Vec<usize>, Vec<Vec<i8>>)> {
        use std::collections::BTreeSet;

        let mut valid_indices: Option<BTreeSet<usize>> = None;

        for horizon in horizons {
            let labels = multi_labels.labels_for_horizon(*horizon).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("No labels for horizon {}", horizon),
                )
            })?;

            let horizon_indices: BTreeSet<usize> = labels.iter().map(|(idx, _, _)| *idx).collect();

            valid_indices = Some(match valid_indices {
                Some(existing) => existing.intersection(&horizon_indices).copied().collect(),
                None => horizon_indices,
            });
        }

        let valid_indices: Vec<usize> = valid_indices
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "No valid indices found")
            })?
            .into_iter()
            .collect();

        if valid_indices.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No common indices across all horizons",
            )
            .into());
        }

        let mut label_matrix: Vec<Vec<i8>> = Vec::with_capacity(valid_indices.len());
        let mut horizon_maps: Vec<HashMap<usize, i8>> = Vec::with_capacity(horizons.len());

        for horizon in horizons {
            let labels = multi_labels.labels_for_horizon(*horizon).unwrap();
            let map: HashMap<usize, i8> = labels
                .iter()
                .map(|(idx, label, _)| {
                    let value = match label {
                        TrendLabel::Up => 1i8,
                        TrendLabel::Down => -1i8,
                        TrendLabel::Stable => 0i8,
                    };
                    (*idx, value)
                })
                .collect();
            horizon_maps.push(map);
        }

        for &idx in &valid_indices {
            let mut row: Vec<i8> = Vec::with_capacity(horizons.len());
            for map in &horizon_maps {
                row.push(*map.get(&idx).unwrap_or(&0));
            }
            label_matrix.push(row);
        }

        Ok((valid_indices, label_matrix))
    }

    /// Align sequences with multi-horizon labels.
    #[allow(clippy::type_complexity)]
    pub(super) fn align_sequences_with_multi_labels(
        &self,
        sequences: &[crate::sequence_builder::Sequence],
        label_indices: &[usize],
        label_matrix: &[Vec<i8>],
    ) -> Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<i8>>)> {
        let mut label_map: HashMap<usize, &Vec<i8>> = HashMap::new();
        for (i, &idx) in label_indices.iter().enumerate() {
            label_map.insert(idx, &label_matrix[i]);
        }

        let mut aligned_sequences = Vec::new();
        let mut aligned_labels = Vec::new();

        for (seq_idx, sequence) in sequences.iter().enumerate() {
            let ending_idx = seq_idx * self.stride + self.window_size - 1;

            if let Some(&label_row) = label_map.get(&ending_idx) {
                let features_owned: Vec<Vec<f64>> = sequence
                    .features
                    .iter()
                    .map(|arc_vec| arc_vec.to_vec())
                    .collect();
                aligned_sequences.push(features_owned);
                aligned_labels.push(label_row.clone());
            }
        }

        if aligned_sequences.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No sequences could be aligned with multi-horizon labels",
            )
            .into());
        }

        Ok((aligned_sequences, aligned_labels))
    }
}
