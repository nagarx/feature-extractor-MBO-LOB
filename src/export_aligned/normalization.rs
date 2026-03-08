//! Feature normalization engine with per-group strategy support.
//!
//! Supports multiple strategies to match different research papers:
//! - None (raw): TLOB paper (model handles normalization via BiN)
//! - ZScore / GlobalZScore / MarketStructure: Per-feature-group strategies
//! - Bilinear: Price-relative normalization

use super::types::{NormalizationParams, NormalizationStrategy};
use super::AlignedBatchExporter;
use crate::export::config::FeatureNormStrategy;
use mbo_lob_reconstructor::Result;

impl AlignedBatchExporter {
    /// Apply configurable normalization based on NormalizationConfig.
    ///
    /// # Feature Layout (98-feature mode)
    ///
    /// - `[0..10]`: Ask prices (levels 1-10)
    /// - `[10..20]`: Ask sizes (levels 1-10)
    /// - `[20..30]`: Bid prices (levels 1-10)
    /// - `[30..40]`: Bid sizes (levels 1-10)
    /// - `[40..48]`: Derived features
    /// - `[48..84]`: MBO features
    /// - `[84..98]`: Signal features (includes categoricals - NEVER normalize)
    ///
    /// # Returns
    ///
    /// Tuple of (normalized_sequences, normalization_params) for export.
    pub(super) fn normalize_sequences(
        &self,
        sequences: &[Vec<Vec<f64>>],
    ) -> Result<(Vec<Vec<Vec<f64>>>, NormalizationParams)> {
        let levels = 10;

        if sequences.is_empty() {
            let empty_params = NormalizationParams::new(
                vec![0.0; levels],
                vec![1.0; levels],
                vec![0.0; levels * 2],
                vec![1.0; levels * 2],
                0,
                levels,
            );
            return Ok((Vec::new(), empty_params));
        }

        let n_features = sequences[0][0].len();
        let config = &self.normalization_config;

        if !config.any_normalization() {
            println!("  🔧 Normalization: NONE (raw export for TLOB paper compatibility)");

            let norm_params = NormalizationParams {
                strategy: NormalizationStrategy::None,
                normalization_applied: false,
                price_means: vec![0.0; levels],
                price_stds: vec![1.0; levels],
                size_means: vec![0.0; levels * 2],
                size_stds: vec![1.0; levels * 2],
                sample_count: sequences.len() * sequences[0].len(),
                feature_layout: format!(
                    "raw_ask_prices_{}_ask_sizes_{}_bid_prices_{}_bid_sizes_{}",
                    levels, levels, levels, levels
                ),
                levels,
            };

            let copied: Vec<Vec<Vec<f64>>> = sequences
                .iter()
                .map(|seq| seq.iter().map(|ts| ts.clone()).collect())
                .collect();

            println!("    ✓ Raw values preserved (model handles normalization internally)");
            return Ok((copied, norm_params));
        }

        println!("  🔧 Applying configurable normalization:");
        println!("     LOB prices: {}", config.lob_prices.description());
        println!("     LOB sizes: {}", config.lob_sizes.description());
        println!("     Derived: {}", config.derived.description());
        println!("     MBO: {}", config.mbo.description());
        println!("     Signals: {}", config.signals.description());

        let epsilon = 1e-8;

        let (price_stats, size_stats, derived_stats, mbo_stats, total_samples) =
            self.compute_feature_statistics(sequences, levels)?;

        let strategy = if config.lob_prices == FeatureNormStrategy::MarketStructure {
            NormalizationStrategy::MarketStructureZScore
        } else if config.lob_prices == FeatureNormStrategy::GlobalZScore {
            NormalizationStrategy::GlobalZScore
        } else if config.lob_prices == FeatureNormStrategy::ZScore {
            NormalizationStrategy::PerFeatureZScore
        } else {
            NormalizationStrategy::None
        };

        let norm_params = NormalizationParams {
            normalization_applied: strategy != NormalizationStrategy::None,
            strategy,
            price_means: price_stats.0.clone(),
            price_stds: price_stats.1.clone(),
            size_means: size_stats.0.clone(),
            size_stds: size_stats.1.clone(),
            sample_count: total_samples,
            feature_layout: format!(
                "ask_prices_{}_ask_sizes_{}_bid_prices_{}_bid_sizes_{}",
                levels, levels, levels, levels
            ),
            levels,
        };

        let mut normalized = Vec::with_capacity(sequences.len());

        for seq in sequences {
            let mut norm_seq = Vec::with_capacity(seq.len());
            for timestep in seq {
                let mut norm_timestep = Vec::with_capacity(n_features);

                // Ask prices (0-9)
                for level in 0..levels {
                    let value = timestep[level];
                    let norm_value = self.apply_normalization(
                        value,
                        &config.lob_prices,
                        price_stats.0[level],
                        price_stats.1[level],
                        epsilon,
                    );
                    norm_timestep.push(norm_value);
                }

                // Ask sizes (10-19)
                for i in 0..levels {
                    let value = timestep[10 + i];
                    let norm_value = self.apply_normalization(
                        value,
                        &config.lob_sizes,
                        size_stats.0[i],
                        size_stats.1[i],
                        epsilon,
                    );
                    norm_timestep.push(norm_value);
                }

                // Bid prices (20-29)
                for level in 0..levels {
                    let value = timestep[20 + level];
                    let norm_value = self.apply_normalization(
                        value,
                        &config.lob_prices,
                        price_stats.0[level],
                        price_stats.1[level],
                        epsilon,
                    );
                    norm_timestep.push(norm_value);
                }

                // Bid sizes (30-39)
                for i in 0..levels {
                    let value = timestep[30 + i];
                    let norm_value = self.apply_normalization(
                        value,
                        &config.lob_sizes,
                        size_stats.0[levels + i],
                        size_stats.1[levels + i],
                        epsilon,
                    );
                    norm_timestep.push(norm_value);
                }

                // Derived features (40-47)
                for i in 0..8 {
                    let idx = 40 + i;
                    if idx < n_features {
                        let value = timestep[idx];
                        let norm_value = self.apply_normalization(
                            value,
                            &config.derived,
                            derived_stats.0.get(i).copied().unwrap_or(0.0),
                            derived_stats.1.get(i).copied().unwrap_or(1.0),
                            epsilon,
                        );
                        norm_timestep.push(norm_value);
                    }
                }

                // MBO features (48-83)
                for i in 0..36 {
                    let idx = 48 + i;
                    if idx < n_features {
                        let value = timestep[idx];
                        let norm_value = self.apply_normalization(
                            value,
                            &config.mbo,
                            mbo_stats.0.get(i).copied().unwrap_or(0.0),
                            mbo_stats.1.get(i).copied().unwrap_or(1.0),
                            epsilon,
                        );
                        norm_timestep.push(norm_value);
                    }
                }

                // Signal features (84+): NEVER normalize (includes categoricals)
                for i in 84..n_features {
                    norm_timestep.push(timestep[i]);
                }

                norm_seq.push(norm_timestep);
            }
            normalized.push(norm_seq);
        }

        println!("    ✓ Normalization applied successfully");
        println!("    📊 Stats: {} samples, {} levels", total_samples, levels);

        Ok((normalized, norm_params))
    }

    /// Compute statistics for each feature group.
    #[allow(clippy::type_complexity)]
    pub(super) fn compute_feature_statistics(
        &self,
        sequences: &[Vec<Vec<f64>>],
        levels: usize,
    ) -> Result<(
        (Vec<f64>, Vec<f64>), // price (means, stds)
        (Vec<f64>, Vec<f64>), // size (means, stds)
        (Vec<f64>, Vec<f64>), // derived (means, stds)
        (Vec<f64>, Vec<f64>), // mbo (means, stds)
        usize,                // sample count
    )> {
        let n_features = sequences[0][0].len();
        let epsilon = 1e-8;
        let config = &self.normalization_config;

        let mut price_level_values: Vec<Vec<f64>> = vec![Vec::new(); levels];
        let mut size_values: Vec<Vec<f64>> = vec![Vec::new(); levels * 2];
        let mut derived_values: Vec<Vec<f64>> = vec![Vec::new(); 8];
        let mut mbo_values: Vec<Vec<f64>> = vec![Vec::new(); 36];

        let mut all_prices: Vec<f64> = Vec::new();
        let mut all_sizes: Vec<f64> = Vec::new();

        let mut total_samples = 0usize;

        for seq in sequences {
            for timestep in seq {
                total_samples += 1;

                match config.lob_prices {
                    FeatureNormStrategy::GlobalZScore => {
                        for level in 0..levels {
                            all_prices.push(timestep[level]);
                            all_prices.push(timestep[20 + level]);
                        }
                    }
                    FeatureNormStrategy::MarketStructure => {
                        for level in 0..levels {
                            let ask_price = timestep[level];
                            let bid_price = timestep[20 + level];
                            price_level_values[level].push(ask_price);
                            price_level_values[level].push(bid_price);
                        }
                    }
                    _ => {
                        for level in 0..levels {
                            price_level_values[level].push(timestep[level]);
                        }
                    }
                }

                match config.lob_sizes {
                    FeatureNormStrategy::GlobalZScore => {
                        for i in 0..levels {
                            all_sizes.push(timestep[10 + i]);
                            all_sizes.push(timestep[30 + i]);
                        }
                    }
                    _ => {
                        for i in 0..levels {
                            size_values[i].push(timestep[10 + i]);
                            size_values[levels + i].push(timestep[30 + i]);
                        }
                    }
                }

                for i in 0..8 {
                    let idx = 40 + i;
                    if idx < n_features {
                        derived_values[i].push(timestep[idx]);
                    }
                }

                for i in 0..36 {
                    let idx = 48 + i;
                    if idx < n_features {
                        mbo_values[i].push(timestep[idx]);
                    }
                }
            }
        }

        let mut price_means = vec![0.0; levels];
        let mut price_stds = vec![1.0; levels];

        if config.lob_prices == FeatureNormStrategy::GlobalZScore && !all_prices.is_empty() {
            let global_mean: f64 = all_prices.iter().sum::<f64>() / all_prices.len() as f64;
            let global_variance: f64 = all_prices
                .iter()
                .map(|v| (v - global_mean).powi(2))
                .sum::<f64>()
                / all_prices.len() as f64;
            let global_std = if global_variance > epsilon {
                global_variance.sqrt()
            } else {
                1.0
            };

            for level in 0..levels {
                price_means[level] = global_mean;
                price_stds[level] = global_std;
            }
            println!(
                "    📊 GlobalZScore prices: mean={:.6}, std={:.6} (shared across all {} price columns)",
                global_mean, global_std, levels * 2
            );
        } else {
            for level in 0..levels {
                let values = &price_level_values[level];
                if !values.is_empty() {
                    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    price_means[level] = mean;
                    price_stds[level] = if variance > epsilon {
                        variance.sqrt()
                    } else {
                        1.0
                    };
                }
            }
        }

        let mut size_means = vec![0.0; levels * 2];
        let mut size_stds = vec![1.0; levels * 2];

        if config.lob_sizes == FeatureNormStrategy::GlobalZScore && !all_sizes.is_empty() {
            let global_mean: f64 = all_sizes.iter().sum::<f64>() / all_sizes.len() as f64;
            let global_variance: f64 = all_sizes
                .iter()
                .map(|v| (v - global_mean).powi(2))
                .sum::<f64>()
                / all_sizes.len() as f64;
            let global_std = if global_variance > epsilon {
                global_variance.sqrt()
            } else {
                1.0
            };

            for i in 0..(levels * 2) {
                size_means[i] = global_mean;
                size_stds[i] = global_std;
            }
            println!(
                "    📊 GlobalZScore sizes: mean={:.6}, std={:.6} (shared across all {} size columns)",
                global_mean, global_std, levels * 2
            );
        } else {
            for i in 0..(levels * 2) {
                let values = &size_values[i];
                if !values.is_empty() {
                    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    size_means[i] = mean;
                    size_stds[i] = if variance > epsilon {
                        variance.sqrt()
                    } else {
                        1.0
                    };
                }
            }
        }

        let mut derived_means = vec![0.0; 8];
        let mut derived_stds = vec![1.0; 8];
        for i in 0..8 {
            let values = &derived_values[i];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                derived_means[i] = mean;
                derived_stds[i] = if variance > epsilon {
                    variance.sqrt()
                } else {
                    1.0
                };
            }
        }

        let mut mbo_means = vec![0.0; 36];
        let mut mbo_stds = vec![1.0; 36];
        for i in 0..36 {
            let values = &mbo_values[i];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                mbo_means[i] = mean;
                mbo_stds[i] = if variance > epsilon {
                    variance.sqrt()
                } else {
                    1.0
                };
            }
        }

        Ok((
            (price_means, price_stds),
            (size_means, size_stds),
            (derived_means, derived_stds),
            (mbo_means, mbo_stds),
            total_samples,
        ))
    }

    /// Apply a single normalization strategy to a value.
    #[inline]
    pub(super) fn apply_normalization(
        &self,
        value: f64,
        strategy: &FeatureNormStrategy,
        mean: f64,
        std: f64,
        epsilon: f64,
    ) -> f64 {
        match strategy {
            FeatureNormStrategy::None => value,
            FeatureNormStrategy::ZScore
            | FeatureNormStrategy::GlobalZScore
            | FeatureNormStrategy::MarketStructure => (value - mean) / (std + epsilon),
            FeatureNormStrategy::PercentageChange => {
                if mean.abs() > epsilon {
                    (value - mean) / mean
                } else {
                    value
                }
            }
            FeatureNormStrategy::MinMax => {
                // FALLBACK: MinMax requires min/max tracking per feature. Falls back to Z-score.
                (value - mean) / (std + epsilon)
            }
            FeatureNormStrategy::Bilinear => {
                // FALLBACK: Bilinear requires per-timestep mid_price. Uses approximation.
                (value - mean) / (self.normalization_config.bilinear_scale_factor * 0.01)
            }
        }
    }
}
