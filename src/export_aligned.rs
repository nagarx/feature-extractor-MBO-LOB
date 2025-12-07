//! Aligned Export Module - Exports sequences with properly aligned labels
//!
//! This module fixes the critical data alignment issue in the original export.
//!
//! # The Problem
//! - Original export: flat features → generate labels separately → misalignment
//! - Labels have different length than features (due to horizon requirements)
//! - Python dataloader rebuilds sequences incorrectly
//!
//! # The Solution
//! 1. Generate labels FIRST from mid-prices
//! 2. Build sequences AND assign labels together (maintaining 1:1 correspondence)
//! 3. Export pre-built sequences [N_seq, window, features]
//! 4. Python loads directly (no sequence building)
//!
//! # Guarantees
//! - Every sequence has exactly one label
//! - Label corresponds to the prediction target for that sequence
//! - Exported arrays have matching lengths: len(sequences) == len(labels)

use crate::labeling::{LabelConfig, TlobLabelGenerator, TrendLabel};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use ndarray::{Array1, Array3};
use ndarray_npy::WriteNpyExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

/// Result of exporting a single day with aligned sequences and labels
#[derive(Debug)]
pub struct AlignedDayExport {
    /// Day identifier
    pub day: String,

    /// Number of sequences exported
    pub n_sequences: usize,

    /// Sequence dimensions: (window_size, n_features)
    pub seq_shape: (usize, usize),

    /// Label distribution
    pub label_distribution: HashMap<String, usize>,

    /// Messages processed
    pub messages_processed: usize,

    /// Export path
    pub export_path: PathBuf,

    /// VERIFICATION: Total features extracted (snapshots generated)
    pub features_extracted: usize,

    /// VERIFICATION: Buffer size used
    pub buffer_size: usize,

    /// VERIFICATION: Total sequences generated before alignment
    pub sequences_generated: usize,

    /// VERIFICATION: Sequences dropped during alignment
    pub sequences_dropped: usize,
}

/// Aligned batch exporter - exports sequences with perfectly aligned labels
pub struct AlignedBatchExporter {
    output_dir: PathBuf,
    label_config: LabelConfig,
    window_size: usize, // Sequence window size (e.g., 100)
    stride: usize,      // Sequence stride (e.g., 10)
}

impl AlignedBatchExporter {
    /// Create new aligned exporter
    ///
    /// # Arguments
    /// * `output_dir` - Where to save the exports
    /// * `label_config` - Configuration for label generation (h, k, threshold)
    /// * `window_size` - Sequence window size (from SequenceConfig)
    /// * `stride` - Sequence stride (from SequenceConfig)
    pub fn new<P: AsRef<Path>>(
        output_dir: P,
        label_config: LabelConfig,
        window_size: usize,
        stride: usize,
    ) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            label_config,
            window_size,
            stride,
        }
    }

    /// Export a single day with aligned sequences and labels
    ///
    /// # Process
    /// 1. Generate labels from mid-prices (establishes valid range)
    /// 2. Extract sequences that have corresponding labels
    /// 3. Align sequences with labels (1:1 mapping)
    /// 4. Export both with validation
    ///
    /// # Output Files
    /// - {day}_sequences.npy: \[N_seq, window, features\] float32
    /// - {day}_labels.npy: \[N_seq\] int8  
    /// - {day}_metadata.json: Metadata with validation info
    pub fn export_day(&self, day_name: &str, output: &PipelineOutput) -> Result<AlignedDayExport> {
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // VERIFICATION: Extract metrics from output
        let features_extracted = output.features_extracted;
        let sequences_generated = output.sequences.len();

        println!("  📊 Pipeline Output:");
        println!("    Features extracted: {features_extracted}");
        println!("    Sequences generated: {sequences_generated}");
        println!("    Mid-prices collected: {}", output.mid_prices.len());

        // Step 1: Generate labels from mid-prices
        println!("  📊 Generating labels...");
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

        // Step 2: Extract aligned sequences
        println!("  🔗 Aligning sequences with labels...");
        let (aligned_sequences, aligned_labels) =
            self.align_sequences_with_labels(&output.sequences, &label_indices, &label_values)?;

        let sequences_dropped = sequences_generated.saturating_sub(aligned_sequences.len());

        println!(
            "    Aligned {} sequences with labels",
            aligned_sequences.len()
        );
        println!("    Dropped {sequences_dropped} sequences (no label)");

        // Step 3: Validate alignment
        self.validate_alignment(&aligned_sequences, &aligned_labels)?;

        // Step 3.5: Verify RAW spreads BEFORE normalization (DIAGNOSTIC)
        self.verify_raw_spreads(&aligned_sequences)?;

        // Step 3.6: Apply Z-score normalization (CRITICAL: prevents NaN in training!)
        let aligned_sequences = self.normalize_sequences(&aligned_sequences)?;

        // Step 4: Export sequences
        let seq_shape = if !aligned_sequences.is_empty() {
            let window = aligned_sequences[0].len();
            let features = aligned_sequences[0][0].len();
            (window, features)
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No aligned sequences to export",
            )
            .into());
        };

        let sequences_path = self.output_dir.join(format!("{day_name}_sequences.npy"));
        self.export_sequences(&aligned_sequences, seq_shape, &sequences_path)?;

        // Step 5: Export labels
        let labels_path = self.output_dir.join(format!("{day_name}_labels.npy"));
        self.export_labels(&aligned_labels, &labels_path)?;

        // Step 6: Export metadata with verification info
        let drop_rate = if sequences_generated > 0 {
            (sequences_dropped as f64 / sequences_generated as f64) * 100.0
        } else {
            0.0
        };

        let metadata = serde_json::json!({
            "day": day_name,
            "n_sequences": aligned_sequences.len(),
            "window_size": seq_shape.0,
            "n_features": seq_shape.1,
            "label_distribution": label_dist,
            "messages_processed": output.messages_processed,
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
            "validation": {
                "sequences_labels_match": aligned_sequences.len() == aligned_labels.len(),
                "label_range_valid": true,
                "no_nan_inf": true,
            },
            "verification": {
                "features_extracted": features_extracted,
                "sequences_generated": sequences_generated,
                "sequences_aligned": aligned_sequences.len(),
                "sequences_dropped": sequences_dropped,
                "drop_rate_percent": format!("{:.2}", drop_rate),
                "buffer_coverage_ok": features_extracted <= 50000,
            }
        });

        let metadata_path = self.output_dir.join(format!("{day_name}_metadata.json"));
        let mut file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(&mut file, &metadata)
            .map_err(|e| std::io::Error::other(format!("Failed to write metadata: {e}")))?;

        println!(
            "  ✅ Export complete: {} sequences aligned with labels",
            aligned_sequences.len()
        );

        Ok(AlignedDayExport {
            day: day_name.to_string(),
            n_sequences: aligned_sequences.len(),
            seq_shape,
            label_distribution: label_dist,
            messages_processed: output.messages_processed,
            export_path: self.output_dir.clone(),
            features_extracted,
            buffer_size: 50_000, // From config
            sequences_generated,
            sequences_dropped,
        })
    }

    /// Generate labels from mid-prices
    ///
    /// Returns: (indices, label_values, distribution)
    /// - indices: Which mid-price index each label corresponds to
    /// - label_values: The actual labels {-1, 0, 1}
    /// - distribution: Count of each label type
    #[allow(clippy::type_complexity)]
    fn generate_labels(
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

        // Create label generator
        let mut generator = TlobLabelGenerator::new(self.label_config.clone());
        generator.add_prices(mid_prices);

        // Generate labels
        let label_results = generator.generate_labels()?;

        // Extract indices, values, and distribution
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

    /// Align sequences with labels
    ///
    /// Key insight: A sequence ending at snapshot i should use the label for snapshot i
    ///
    /// # Example
    /// ```text
    /// Sequence 0: snapshots [0, 1, 2, ..., 99]  → use label[99]
    /// Sequence 1: snapshots [10, 11, ..., 109]  → use label[109]
    /// Sequence 2: snapshots [20, 21, ..., 119]  → use label[119]
    /// ```
    ///
    /// Only keep sequences where the ending snapshot has a valid label.
    #[allow(clippy::type_complexity)]
    fn align_sequences_with_labels(
        &self,
        sequences: &[crate::sequence_builder::Sequence],
        label_indices: &[usize],
        label_values: &[i8],
    ) -> Result<(Vec<Vec<Vec<f64>>>, Vec<i8>)> {
        // Create a map: snapshot_index → label
        let mut label_map: HashMap<usize, i8> = HashMap::new();
        for (idx, &label) in label_indices.iter().zip(label_values) {
            label_map.insert(*idx, label);
        }

        let mut aligned_sequences = Vec::new();
        let mut aligned_labels = Vec::new();

        // For each sequence in pipeline output
        for (seq_idx, sequence) in sequences.iter().enumerate() {
            // Determine the index of the last snapshot in this sequence
            // CRITICAL: This must match how the sequence was built!
            //
            // If sequences are built with stride=10 from flat features:
            //   Seq 0: features[0:100]   → last snapshot = 99
            //   Seq 1: features[10:110]  → last snapshot = 109
            //   Seq 2: features[20:120]  → last snapshot = 119
            //
            // The label for a sequence should be based on where we want to predict
            // For TLOB: predict future from current state → use label at last snapshot

            // The sequence.length tells us how many snapshots in this sequence
            // We need to figure out which flat feature index this corresponds to
            //
            // Problem: Sequences don't store their original feature indices!
            // Solution: Assume sequences were built sequentially with known stride

            // Calculate the ending feature index for this sequence
            // If stride=10, window=100:
            //   Seq 0: features[0:100]   → ending_idx = 0 + 100 - 1 = 99
            //   Seq 1: features[10:110]  → ending_idx = 10 + 100 - 1 = 109
            //   Seq 2: features[20:120]  → ending_idx = 20 + 100 - 1 = 119
            let ending_idx = seq_idx * self.stride + self.window_size - 1;

            // Check if we have a label for this ending index
            if let Some(&label) = label_map.get(&ending_idx) {
                // We have a valid label for this sequence!
                // Deep-copy from Arc<Vec<f64>> to Vec<f64> for export format
                let features_owned: Vec<Vec<f64>> = sequence
                    .features
                    .iter()
                    .map(|arc_vec| arc_vec.to_vec())
                    .collect();
                aligned_sequences.push(features_owned);
                aligned_labels.push(label);
            }
            // If no label, skip this sequence (happens at end of day)
        }

        if aligned_sequences.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No sequences could be aligned with labels - check horizon/smoothing config",
            )
            .into());
        }

        Ok((aligned_sequences, aligned_labels))
    }

    /// Validate that sequences and labels are properly aligned
    fn validate_alignment(&self, sequences: &[Vec<Vec<f64>>], labels: &[i8]) -> Result<()> {
        // Check 1: Same length
        if sequences.len() != labels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Alignment error: {} sequences but {} labels",
                    sequences.len(),
                    labels.len()
                ),
            )
            .into());
        }

        // Check 2: All sequences have same shape
        if !sequences.is_empty() {
            let window = sequences[0].len();
            let features = sequences[0][0].len();

            for (i, seq) in sequences.iter().enumerate() {
                if seq.len() != window {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Sequence {} has wrong window size: {} vs {}",
                            i,
                            seq.len(),
                            window
                        ),
                    )
                    .into());
                }
                if seq[0].len() != features {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Sequence {} has wrong feature count: {} vs {}",
                            i,
                            seq[0].len(),
                            features
                        ),
                    )
                    .into());
                }
            }
        }

        // Check 3: All labels are valid {-1, 0, 1}
        for (i, &label) in labels.iter().enumerate() {
            if !(-1..=1).contains(&label) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid label at index {i}: {label}"),
                )
                .into());
            }
        }

        // Check 4: No NaN/Inf in sequences
        for (i, seq) in sequences.iter().enumerate() {
            for (t, timestep) in seq.iter().enumerate() {
                for (f, &value) in timestep.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "Invalid value in sequence {i} at timestep {t}, feature {f}: {value}"
                            ),
                        )
                        .into());
                    }
                }
            }
        }

        println!(
            "    ✓ Validation passed: {} aligned sequence-label pairs",
            sequences.len()
        );
        Ok(())
    }

    /// Verify RAW spread integrity BEFORE normalization (DIAGNOSTIC)
    fn verify_raw_spreads(&self, sequences: &[Vec<Vec<f64>>]) -> Result<()> {
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
                // Check all 10 levels
                for level in 0..10 {
                    let ask_idx = level; // 0, 1, 2, ..., 9
                    let bid_idx = 20 + level; // 20, 21, 22, ..., 29

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

        // CRITICAL: Raw spreads MUST be >99% positive
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

    /// Apply market-structure preserving Z-score normalization
    ///
    /// CRITICAL FIX: Normalizes ask+bid prices TOGETHER per level to preserve ordering.
    ///
    /// Feature layout: [Ask_prices(10), Ask_sizes(10), Bid_prices(10), Bid_sizes(10)]
    ///                  0-9            10-19          20-29          30-39
    ///
    /// For prices: Normalize ask_L and bid_L using SHARED mean/std for level L
    /// For sizes: Normalize independently (no ordering constraint)
    ///
    /// This ensures: if ask > bid, then ask_norm > bid_norm (market structure preserved!)
    fn normalize_sequences(&self, sequences: &[Vec<Vec<f64>>]) -> Result<Vec<Vec<Vec<f64>>>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        let n_features = sequences[0][0].len();
        let epsilon = 1e-8;

        println!("  🔧 Applying market-structure preserving Z-score normalization...");

        // Step 1: Compute statistics
        // For prices (ask+bid per level): shared mean/std
        // For sizes: independent mean/std

        // Collect all values per feature for statistics
        let mut price_level_values: Vec<Vec<f64>> = vec![Vec::new(); 10]; // 10 price levels
        let mut size_values: Vec<Vec<f64>> = vec![Vec::new(); 20]; // 20 size features (10 ask + 10 bid)

        for seq in sequences {
            for timestep in seq {
                // Collect ask prices (0-9) and bid prices (20-29) together per level
                for level in 0..10 {
                    let ask_price = timestep[level];
                    let bid_price = timestep[20 + level];

                    // Combine ask+bid for this level
                    price_level_values[level].push(ask_price);
                    price_level_values[level].push(bid_price);
                }

                // Collect sizes independently
                for i in 0..10 {
                    size_values[i].push(timestep[10 + i]); // Ask sizes
                    size_values[10 + i].push(timestep[30 + i]); // Bid sizes
                }
            }
        }

        // Compute mean/std for each price level (shared across ask+bid)
        let mut price_means = [0.0; 10];
        let mut price_stds = [1.0; 10];

        for level in 0..10 {
            let values = &price_level_values[level];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                    / values.len() as f64;
                let std = if variance > epsilon {
                    variance.sqrt()
                } else {
                    1.0
                };

                price_means[level] = mean;
                price_stds[level] = std;
            }
        }

        // Compute mean/std for each size feature (independent)
        let mut size_means = [0.0; 20];
        let mut size_stds = [1.0; 20];

        for i in 0..20 {
            let values = &size_values[i];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                    / values.len() as f64;
                let std = if variance > epsilon {
                    variance.sqrt()
                } else {
                    1.0
                };

                size_means[i] = mean;
                size_stds[i] = std;
            }
        }

        // Step 2: Normalize all sequences using computed statistics
        let mut normalized = Vec::with_capacity(sequences.len());

        for seq in sequences {
            let mut norm_seq = Vec::with_capacity(seq.len());
            for timestep in seq {
                let mut norm_timestep = Vec::with_capacity(n_features);

                // Normalize ask prices (0-9) using shared level stats
                for level in 0..10 {
                    let value = timestep[level];
                    let norm_value = (value - price_means[level]) / (price_stds[level] + epsilon);
                    norm_timestep.push(norm_value);
                }

                // Normalize ask sizes (10-19) independently
                for i in 0..10 {
                    let value = timestep[10 + i];
                    let norm_value = (value - size_means[i]) / (size_stds[i] + epsilon);
                    norm_timestep.push(norm_value);
                }

                // Normalize bid prices (20-29) using shared level stats
                for level in 0..10 {
                    let value = timestep[20 + level];
                    let norm_value = (value - price_means[level]) / (price_stds[level] + epsilon);
                    norm_timestep.push(norm_value);
                }

                // Normalize bid sizes (30-39) independently
                for i in 0..10 {
                    let value = timestep[30 + i];
                    let norm_value = (value - size_means[10 + i]) / (size_stds[10 + i] + epsilon);
                    norm_timestep.push(norm_value);
                }

                norm_seq.push(norm_timestep);
            }
            normalized.push(norm_seq);
        }

        println!("    ✓ Market-structure preserved (ask > bid maintained)");

        Ok(normalized)
    }

    /// Export sequences as 3D numpy array
    ///
    /// Shape: [n_sequences, window_size, n_features]
    fn export_sequences(
        &self,
        sequences: &[Vec<Vec<f64>>],
        shape: (usize, usize),
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let (window, features) = shape;

        // Flatten to 1D vec in row-major order
        let flat: Vec<f32> = sequences
            .iter()
            .flat_map(|seq| seq.iter())
            .flat_map(|timestep| timestep.iter().copied().map(|x| x as f32))
            .collect();

        // Create 3D array
        let array = Array3::from_shape_vec((n_seq, window, features), flat)
            .map_err(|e| format!("Failed to create 3D array: {e}"))?;

        // Write to file
        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write sequences: {e}"))?;

        println!(
            "  ✅ Sequences: {} [{} × {} × {}]",
            path.display(),
            n_seq,
            window,
            features
        );

        Ok(())
    }

    /// Export labels as 1D numpy array
    ///
    /// Shape: [n_sequences]
    fn export_labels(&self, labels: &[i8], path: &Path) -> Result<()> {
        let array = Array1::from_vec(labels.to_vec());

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write labels: {e}"))?;

        // Print distribution
        let up = labels.iter().filter(|&&l| l == 1).count();
        let down = labels.iter().filter(|&&l| l == -1).count();
        let stable = labels.iter().filter(|&&l| l == 0).count();

        println!("  ✅ Labels: {} [{} samples]", path.display(), labels.len());
        println!(
            "     Distribution: Up={} ({:.1}%), Down={} ({:.1}%), Stable={} ({:.1}%)",
            up,
            100.0 * up as f64 / labels.len() as f64,
            down,
            100.0 * down as f64 / labels.len() as f64,
            stable,
            100.0 * stable as f64 / labels.len() as f64
        );

        Ok(())
    }
}

/// Metadata for aligned export (reserved for future enhanced export)
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct AlignedMetadata {
    day: String,
    n_sequences: usize,
    window_size: usize,
    n_features: usize,
    label_distribution: HashMap<String, usize>,
    messages_processed: usize,
    export_timestamp: String,
    validation: ValidationInfo,
}

/// Validation information (reserved for future enhanced export)
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct ValidationInfo {
    sequences_labels_match: bool,
    label_range_valid: bool,
    no_nan_inf: bool,
}
