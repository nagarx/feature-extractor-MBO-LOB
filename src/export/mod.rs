//! Data Export Module
//!
//! Export processed features and labels to various formats for ML training.
//!
//! # Modules
//!
//! - **tensor_format**: Model-specific tensor reshaping (DeepLOB, HLOB, etc.)
//! - Core exports: NumPy (.npy) and JSON metadata
//!
//! # Supported Formats
//!
//! - NumPy (.npy) - For Python/PyTorch integration
//! - JSON - For metadata and configuration
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::export::{NumpyExporter, BatchExporter};
//! use feature_extractor::export::tensor_format::{TensorFormat, TensorFormatter};
//!
//! // Single file export
//! let exporter = NumpyExporter::new(output_dir);
//! exporter.export(&pipeline_output)?;
//!
//! // Batch export with labels
//! let batch_exporter = BatchExporter::new(output_dir, Some(label_config));
//! let result = batch_exporter.export_day("2025-02-03", &pipeline_output)?;
//!
//! // Format tensors for specific model
//! let formatter = TensorFormatter::deeplob(10);
//! let tensor = formatter.format_sequence(&features)?;
//! ```

pub mod tensor_format;

use crate::labeling::{LabelConfig, TlobLabelGenerator, TrendLabel};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use ndarray::{Array1, Array2};
use ndarray_npy::WriteNpyExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

// Re-export tensor formatting types
pub use tensor_format::{FeatureMapping, TensorFormat, TensorFormatter, TensorOutput};

// FeatureVec is used in test helper function
#[cfg(test)]
use crate::sequence_builder::FeatureVec;

/// Apply z-score normalization to features for neural network training.
///
/// Normalizes each feature column to have mean ≈ 0 and std ≈ 1.
/// This is critical for numerical stability in deep learning.
///
/// # Arguments
///
/// * `features` - [N_samples, N_features] array to normalize
///
/// # Returns
///
/// Normalized features with same shape
///
/// # Mathematical Details
///
/// For each feature column j:
/// ```text
/// normalized[i,j] = (features[i,j] - mean[j]) / std[j]
/// ```
///
/// Where:
/// - mean[j] = average of column j
/// - std[j] = standard deviation of column j
/// - If std[j] == 0 (constant feature), uses std[j] = 1 to avoid division by zero
fn z_score_normalize_batch(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let n_samples = features.len();
    let n_features = features[0].len();

    // Compute mean and std for each feature column
    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    // Compute means
    for feat_idx in 0..n_features {
        let mut sum = 0.0;
        for sample in features.iter() {
            sum += sample[feat_idx];
        }
        means[feat_idx] = sum / n_samples as f64;
    }

    // Compute standard deviations
    for feat_idx in 0..n_features {
        let mut sum_sq_diff = 0.0;
        for sample in features.iter() {
            let diff = sample[feat_idx] - means[feat_idx];
            sum_sq_diff += diff * diff;
        }
        let variance = sum_sq_diff / n_samples as f64;
        stds[feat_idx] = variance.sqrt();

        // Avoid division by zero for constant features
        if stds[feat_idx] < 1e-8 {
            stds[feat_idx] = 1.0;
        }
    }

    // Normalize all features
    let mut normalized = Vec::with_capacity(n_samples);
    for sample in features.iter() {
        let mut norm_sample = Vec::with_capacity(n_features);
        for feat_idx in 0..n_features {
            let norm_val = (sample[feat_idx] - means[feat_idx]) / stds[feat_idx];
            norm_sample.push(norm_val);
        }
        normalized.push(norm_sample);
    }

    normalized
}

/// Metadata about exported dataset
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Number of samples (snapshots)
    pub n_samples: usize,

    /// Number of features per sample
    pub n_features: usize,

    /// Number of sequences generated
    pub n_sequences: usize,

    /// Window size used
    pub window_size: usize,

    /// Stride used
    pub stride: usize,

    /// Whether labels are included
    pub has_labels: bool,

    /// Messages processed
    pub messages_processed: usize,

    /// Export timestamp
    pub export_timestamp: String,
}

/// NumPy exporter - exports to .npy files for Python
pub struct NumpyExporter {
    output_dir: PathBuf,
}

impl NumpyExporter {
    /// Create new NumPy exporter
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }

    /// Export pipeline output to NumPy format
    ///
    /// Creates:
    /// - features.npy: \[N_samples, N_features\] array
    /// - mid_prices.npy: \[N_samples\] array (for labeling)
    /// - metadata.json: Dataset metadata
    pub fn export(&self, output: &PipelineOutput) -> Result<()> {
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // Convert to flat arrays
        let features = output.to_flat_features();
        let mid_prices = &output.mid_prices;

        if features.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No features to export",
            )
            .into());
        }

        let n_samples = features.len();
        let n_features = features[0].len();

        // Export features
        self.export_features(&features, n_samples, n_features)?;

        // Export mid-prices (for labeling later)
        self.export_mid_prices(mid_prices)?;

        // Export metadata - use actual values from PipelineOutput
        self.export_metadata(ExportMetadata {
            n_samples,
            n_features,
            n_sequences: output.sequences_generated,
            window_size: output.window_size,
            stride: output.stride,
            has_labels: false, // Labels are generated separately
            messages_processed: output.messages_processed,
            export_timestamp: chrono::Utc::now().to_rfc3339(),
        })?;

        Ok(())
    }

    /// Export features as 2D NumPy array
    fn export_features(&self, features: &[Vec<f64>], rows: usize, cols: usize) -> Result<()> {
        // Flatten features into single vec
        let flat: Vec<f64> = features
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Create ndarray
        let array = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| format!("Failed to create array: {e}"))?;

        // Write to file
        let path = self.output_dir.join("features.npy");
        let mut file = File::create(&path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write features.npy: {e}"))?;

        println!(
            "✅ Exported features: {} [{} samples × {} features]",
            path.display(),
            rows,
            cols
        );

        Ok(())
    }

    /// Export mid-prices as 1D NumPy array (for labeling later)
    fn export_mid_prices(&self, mid_prices: &[f64]) -> Result<()> {
        let array = Array1::from_vec(mid_prices.to_vec());

        let path = self.output_dir.join("mid_prices.npy");
        let mut file = File::create(&path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write mid_prices.npy: {e}"))?;

        println!(
            "✅ Exported mid-prices: {} [{} samples]",
            path.display(),
            mid_prices.len()
        );

        Ok(())
    }

    /// Export metadata as JSON
    fn export_metadata(&self, metadata: ExportMetadata) -> Result<()> {
        let path = self.output_dir.join("metadata.json");
        let file = File::create(&path)?;
        serde_json::to_writer_pretty(file, &metadata)
            .map_err(|e| format!("Failed to write metadata: {e}"))?;

        println!("✅ Exported metadata: {}", path.display());

        Ok(())
    }
}

/// Convenience function for direct export
pub fn export_to_numpy<P: AsRef<Path>>(output: &PipelineOutput, output_dir: P) -> Result<()> {
    let exporter = NumpyExporter::new(output_dir);
    exporter.export(output)
}

//=============================================================================
// Batch Export Functionality
//=============================================================================

/// Result of exporting a single day
#[derive(Debug, Clone)]
pub struct DayExportResult {
    /// Day identifier (e.g., "2025-02-03")
    pub day: String,

    /// Number of features exported
    pub n_features: usize,

    /// Number of labels exported (if labeling enabled)
    pub n_labels: usize,

    /// Number of sequences exported
    pub n_sequences: usize,

    /// Messages processed
    pub messages_processed: usize,

    /// Label distribution (if labeling enabled)
    pub label_distribution: Option<HashMap<String, usize>>,

    /// Export path
    pub export_path: PathBuf,
}

/// Configuration for train/val/test splits
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Number of days for training
    pub train_days: usize,

    /// Number of days for validation
    pub val_days: usize,

    /// Number of days for testing (remaining days)
    pub test_days: usize,
}

/// Result of batch export operation
#[derive(Debug)]
pub struct BatchExportResult {
    /// Total days processed
    pub total_days: usize,

    /// Days in training split
    pub train_days: Vec<DayExportResult>,

    /// Days in validation split
    pub val_days: Vec<DayExportResult>,

    /// Days in test split
    pub test_days: Vec<DayExportResult>,

    /// Total features exported
    pub total_features: usize,

    /// Total labels exported
    pub total_labels: usize,

    /// Overall label distribution
    pub label_distribution: HashMap<String, usize>,
}

/// Batch exporter for processing multiple days
pub struct BatchExporter {
    output_dir: PathBuf,
    label_config: Option<LabelConfig>,
}

impl BatchExporter {
    /// Create new batch exporter
    ///
    /// # Arguments
    /// * `output_dir` - Base output directory
    /// * `label_config` - Optional label configuration for creating labels from mid-prices
    pub fn new<P: AsRef<Path>>(output_dir: P, label_config: Option<LabelConfig>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            label_config,
        }
    }

    /// Export a single day of data
    ///
    /// Creates files:
    /// - {day_name}_features.npy: \[N_samples, N_features\] features
    /// - {day_name}_labels.npy: \[N_labels\] labels (if label_generator provided)
    /// - {day_name}_metadata.json: Metadata for this day
    ///
    /// # Arguments
    /// * `day_name` - Day identifier (e.g., "2025-02-03")
    /// * `output` - Pipeline output to export
    pub fn export_day(&self, day_name: &str, output: &PipelineOutput) -> Result<DayExportResult> {
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // Convert features to flat array
        let mut features = output.to_flat_features();
        if features.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("No features to export for day {day_name}"),
            )
            .into());
        }

        // Apply z-score normalization (Stage 2: global normalization)
        // Stage 1 (relative normalization) already applied in feature extraction
        features = z_score_normalize_batch(&features);

        let n_samples = features.len();
        let n_features = features[0].len();

        // Export normalized features
        let features_path = self.output_dir.join(format!("{day_name}_features.npy"));
        self.export_features_to_file(&features, n_samples, n_features, &features_path)?;

        // Export labels (if label config provided)
        let (n_labels, label_dist) = if let Some(ref config) = self.label_config {
            let mid_prices = &output.mid_prices;
            if !mid_prices.is_empty() {
                let labels_path = self.output_dir.join(format!("{day_name}_labels.npy"));
                let distribution = self.export_labels(mid_prices, &labels_path, config)?;
                (distribution.values().sum::<usize>(), Some(distribution))
            } else {
                (0, None)
            }
        } else {
            (0, None)
        };

        // Export metadata for this day
        let metadata = DayMetadata {
            day: day_name.to_string(),
            n_features: n_samples,
            n_feature_dim: n_features,
            n_labels,
            n_sequences: output.sequences_generated,
            messages_processed: output.messages_processed,
            label_distribution: label_dist.clone(),
            export_timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let metadata_path = self.output_dir.join(format!("{day_name}_metadata.json"));
        let file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(file, &metadata)
            .map_err(|e| format!("Failed to write metadata: {e}"))?;

        Ok(DayExportResult {
            day: day_name.to_string(),
            n_features: n_samples,
            n_labels,
            n_sequences: output.sequences_generated,
            messages_processed: output.messages_processed,
            label_distribution: label_dist,
            export_path: self.output_dir.clone(),
        })
    }

    /// Export features to specific file path
    fn export_features_to_file(
        &self,
        features: &[Vec<f64>],
        rows: usize,
        cols: usize,
        path: &Path,
    ) -> Result<()> {
        // Flatten features into single vec
        let flat: Vec<f64> = features
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Create ndarray
        let array = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| format!("Failed to create array: {e}"))?;

        // Write to file
        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write features: {e}"))?;

        println!("  ✅ Features: {} [{} × {}]", path.display(), rows, cols);

        Ok(())
    }

    /// Export labels from mid-prices using label config
    fn export_labels(
        &self,
        mid_prices: &[f64],
        path: &Path,
        config: &LabelConfig,
    ) -> Result<HashMap<String, usize>> {
        // Create a new generator and add prices
        let mut generator = TlobLabelGenerator::new(config.clone());
        generator.add_prices(mid_prices);

        // Generate labels
        let label_results = generator.generate_labels()?;

        // Extract labels and count distribution
        let mut distribution = HashMap::new();
        distribution.insert("Up".to_string(), 0);
        distribution.insert("Down".to_string(), 0);
        distribution.insert("Stable".to_string(), 0);

        let labels: Vec<i8> = label_results
            .iter()
            .map(|(_idx, label, _value)| match label {
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
            })
            .collect();

        let n_labels = labels.len();

        // Convert to ndarray
        let array = Array1::from_vec(labels);

        // Write to file
        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write labels: {e}"))?;

        println!("  ✅ Labels: {} [{} samples]", path.display(), n_labels);
        println!(
            "     Distribution: Up={}, Down={}, Stable={}",
            distribution["Up"], distribution["Down"], distribution["Stable"]
        );

        Ok(distribution)
    }
}

/// Metadata for a single day's export
#[derive(Debug, Serialize, Deserialize)]
struct DayMetadata {
    day: String,
    n_features: usize,
    n_feature_dim: usize,
    n_labels: usize,
    n_sequences: usize,
    messages_processed: usize,
    label_distribution: Option<HashMap<String, usize>>,
    export_timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labeling::LabelConfig;
    use crate::Sequence;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Helper to create test pipeline output with realistic LOB data
    ///
    /// Creates feature vectors with proper LOB structure:
    /// - Indices 0-9: Ask prices (ascending from mid + spread/2)
    /// - Indices 10-19: Ask sizes
    /// - Indices 20-29: Bid prices (descending from mid - spread/2)
    /// - Indices 30-39: Bid sizes
    /// - Indices 40-47: Derived features (if 48 features)
    fn create_test_pipeline_output(n_features: usize, n_sequences: usize) -> PipelineOutput {
        let mut sequences = Vec::new();
        let mut mid_prices = Vec::new();

        let base_mid_price = 100.0;
        let spread = 0.01; // 1 cent spread
        let tick_size = 0.01;

        // Create test sequences
        for i in 0..n_sequences {
            let mut features: Vec<FeatureVec> = Vec::new();
            for t in 0..100 {
                // Create realistic LOB feature vector
                let mid_price = base_mid_price + (i + t) as f64 * 0.001;
                let best_ask = mid_price + spread / 2.0;
                let best_bid = mid_price - spread / 2.0;

                // Use 40 features for raw LOB (10 levels × 4)
                let feature_count = if n_features >= 40 { 40 } else { n_features };
                let mut feature_vec = vec![0.0; feature_count.max(40)];

                // Ask prices (indices 0-9): ascending from best ask
                for level in 0..10 {
                    feature_vec[level] = best_ask + (level as f64) * tick_size;
                }

                // Ask sizes (indices 10-19): decreasing with level
                for level in 0..10 {
                    feature_vec[10 + level] = (1000 - level * 50) as f64;
                }

                // Bid prices (indices 20-29): descending from best bid
                for level in 0..10 {
                    feature_vec[20 + level] = best_bid - (level as f64) * tick_size;
                }

                // Bid sizes (indices 30-39): decreasing with level
                for level in 0..10 {
                    feature_vec[30 + level] = (1000 - level * 50) as f64;
                }

                // Wrap in Arc for zero-copy sharing
                features.push(Arc::new(feature_vec));

                // Add corresponding mid-price
                mid_prices.push(mid_price);
            }

            let start_ts = i as u64 * 1000;
            let end_ts = start_ts + 100 * 1000;

            sequences.push(Sequence {
                features,
                start_timestamp: start_ts,
                end_timestamp: end_ts,
                duration_ns: end_ts - start_ts,
                length: 100,
            });
        }

        PipelineOutput {
            sequences,
            mid_prices,
            messages_processed: 100000,
            features_extracted: n_features,
            sequences_generated: n_sequences,
            stride: 1,        // Default stride for tests
            window_size: 100, // Default window size for tests
            multiscale_sequences: None,
            adaptive_stats: None,
        }
    }

    #[test]
    fn test_export_metadata_creation() {
        let metadata = ExportMetadata {
            n_samples: 1000,
            n_features: 48,
            n_sequences: 10,
            window_size: 100,
            stride: 1,
            has_labels: false,
            messages_processed: 100000,
            export_timestamp: "2025-10-23T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("1000"));
        assert!(json.contains("48"));
    }

    #[test]
    fn test_batch_exporter_creation() {
        let temp_dir = TempDir::new().unwrap();
        let _exporter = BatchExporter::new(temp_dir.path(), None);
        assert!(temp_dir.path().exists());
    }

    #[test]
    fn test_export_day_without_labels() {
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path(), None);

        let output = create_test_pipeline_output(1000, 10);
        let result = exporter.export_day("2025-02-03", &output).unwrap();

        // Verify result
        assert_eq!(result.day, "2025-02-03");
        assert_eq!(result.n_features, 1000); // 10 sequences * 100 timesteps
        assert_eq!(result.n_labels, 0); // No labels without generator
        assert_eq!(result.n_sequences, 10);
        assert_eq!(result.messages_processed, 100000);
        assert!(result.label_distribution.is_none());

        // Verify files exist
        let features_path = temp_dir.path().join("2025-02-03_features.npy");
        let metadata_path = temp_dir.path().join("2025-02-03_metadata.json");
        assert!(features_path.exists(), "Features file should exist");
        assert!(metadata_path.exists(), "Metadata file should exist");

        // Labels should NOT exist
        let labels_path = temp_dir.path().join("2025-02-03_labels.npy");
        assert!(
            !labels_path.exists(),
            "Labels file should not exist without generator"
        );
    }

    #[test]
    fn test_export_day_with_labels() {
        let temp_dir = TempDir::new().unwrap();

        // Create label config
        let label_config = LabelConfig {
            horizon: 200,
            smoothing_window: 20,
            threshold: 0.0005,
        };

        let exporter = BatchExporter::new(temp_dir.path(), Some(label_config));

        let output = create_test_pipeline_output(1000, 10);
        let result = exporter.export_day("2025-02-03", &output).unwrap();

        // Verify result
        assert_eq!(result.day, "2025-02-03");
        assert!(result.n_labels > 0, "Should have labels");
        assert!(
            result.label_distribution.is_some(),
            "Should have label distribution"
        );

        // Verify all three label classes are present in distribution
        let dist = result.label_distribution.unwrap();
        assert!(dist.contains_key("Up"));
        assert!(dist.contains_key("Down"));
        assert!(dist.contains_key("Stable"));

        // Verify files exist
        let features_path = temp_dir.path().join("2025-02-03_features.npy");
        let labels_path = temp_dir.path().join("2025-02-03_labels.npy");
        let metadata_path = temp_dir.path().join("2025-02-03_metadata.json");

        assert!(features_path.exists(), "Features file should exist");
        assert!(labels_path.exists(), "Labels file should exist");
        assert!(metadata_path.exists(), "Metadata file should exist");
    }

    #[test]
    fn test_feature_array_correctness() {
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path(), None);

        let output = create_test_pipeline_output(100, 1);
        exporter.export_day("test", &output).unwrap();

        // Read back the features
        let features_path = temp_dir.path().join("test_features.npy");
        let file = File::open(&features_path).unwrap();
        let array: Array2<f64> = ndarray_npy::ReadNpyExt::read_npy(file).unwrap();

        // Verify shape - 40 raw LOB features (10 levels × 4)
        assert_eq!(array.shape(), &[100, 40], "Shape should be [100, 40]");

        // Verify no NaN or Inf
        for &val in array.iter() {
            assert!(val.is_finite(), "All values should be finite (no NaN/Inf)");
        }

        // After z-score normalization, values should be in reasonable range
        // Z-scores typically fall within [-4, 4] for most data
        for &val in array.iter() {
            assert!(
                val.abs() < 10.0,
                "Normalized values should be in reasonable range, got {val}"
            );
        }
    }

    #[test]
    fn test_label_array_correctness() {
        let temp_dir = TempDir::new().unwrap();

        let label_config = LabelConfig {
            horizon: 10,
            smoothing_window: 5,
            threshold: 0.001,
        };
        let exporter = BatchExporter::new(temp_dir.path(), Some(label_config));

        let output = create_test_pipeline_output(100, 1);
        exporter.export_day("test", &output).unwrap();

        // Read back the labels
        let labels_path = temp_dir.path().join("test_labels.npy");
        let file = File::open(&labels_path).unwrap();
        let array: Array1<i8> = ndarray_npy::ReadNpyExt::read_npy(file).unwrap();

        // Verify labels are only -1, 0, or 1
        for &label in array.iter() {
            assert!(
                label == -1 || label == 0 || label == 1,
                "Labels must be -1 (Down), 0 (Stable), or 1 (Up), got {label}"
            );
        }
    }

    #[test]
    fn test_metadata_accuracy() {
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path(), None);

        let output = create_test_pipeline_output(500, 5);
        exporter.export_day("2025-02-03", &output).unwrap();

        // Read metadata
        let metadata_path = temp_dir.path().join("2025-02-03_metadata.json");
        let file = File::open(&metadata_path).unwrap();
        let metadata: DayMetadata = serde_json::from_reader(file).unwrap();

        // Verify metadata accuracy
        assert_eq!(metadata.day, "2025-02-03");
        assert_eq!(metadata.n_features, 500);
        assert_eq!(metadata.n_feature_dim, 40); // 10 levels × 4 = 40 raw LOB features
        assert_eq!(metadata.n_sequences, 5);
        assert_eq!(metadata.messages_processed, 100000);
    }

    #[test]
    fn test_empty_features_error() {
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path(), None);

        // Create empty output
        let empty_output = PipelineOutput {
            sequences: Vec::new(),
            mid_prices: Vec::new(),
            messages_processed: 0,
            features_extracted: 0,
            sequences_generated: 0,
            stride: 1,
            window_size: 100,
            multiscale_sequences: None,
            adaptive_stats: None,
        };

        // Should error on empty features
        let result = exporter.export_day("test", &empty_output);
        assert!(result.is_err(), "Should error on empty features");
    }

    #[test]
    fn test_label_distribution_sum() {
        let temp_dir = TempDir::new().unwrap();

        let label_config = LabelConfig {
            horizon: 10,
            smoothing_window: 5,
            threshold: 0.001,
        };
        let exporter = BatchExporter::new(temp_dir.path(), Some(label_config));

        let output = create_test_pipeline_output(100, 1);
        let result = exporter.export_day("test", &output).unwrap();

        // Verify label distribution sums to n_labels
        if let Some(dist) = result.label_distribution {
            let sum: usize = dist.values().sum();
            assert_eq!(
                sum, result.n_labels,
                "Label distribution should sum to n_labels"
            );
        }
    }

    #[test]
    fn test_concurrent_exports() {
        // Test that multiple exports to same directory don't interfere
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path(), None);

        let output1 = create_test_pipeline_output(100, 1);
        let output2 = create_test_pipeline_output(200, 2);

        exporter.export_day("day1", &output1).unwrap();
        exporter.export_day("day2", &output2).unwrap();

        // Both should exist
        assert!(temp_dir.path().join("day1_features.npy").exists());
        assert!(temp_dir.path().join("day2_features.npy").exists());
        assert!(temp_dir.path().join("day1_metadata.json").exists());
        assert!(temp_dir.path().join("day2_metadata.json").exists());
    }

    #[test]
    fn test_pipeline_output_stride_and_window_size_preserved() {
        // CRITICAL TEST: Verify that stride and window_size from PipelineOutput
        // are correctly preserved in export metadata.
        // This is essential for downstream consumers to know how sequences were built.

        let mut output = create_test_pipeline_output(40, 5);
        output.stride = 10; // Custom stride
        output.window_size = 100; // Custom window size

        let temp_dir = TempDir::new().unwrap();
        let exporter = NumpyExporter::new(temp_dir.path());
        exporter.export(&output).unwrap();

        // Read and verify metadata
        let metadata_path = temp_dir.path().join("metadata.json");
        let metadata_content = std::fs::read_to_string(&metadata_path).unwrap();
        let metadata: ExportMetadata = serde_json::from_str(&metadata_content).unwrap();

        // Verify stride is preserved (not hardcoded to 1)
        assert_eq!(
            metadata.stride, 10,
            "Stride should be preserved from PipelineOutput (was {} but expected 10)",
            metadata.stride
        );

        // Verify window_size is preserved
        assert_eq!(
            metadata.window_size, 100,
            "Window size should be preserved from PipelineOutput (was {} but expected 100)",
            metadata.window_size
        );
    }

    #[test]
    fn test_stride_consistency_across_export_methods() {
        // Verify stride is consistent between NumpyExporter and BatchExporter

        let mut output = create_test_pipeline_output(40, 5);
        output.stride = 25; // Non-default stride
        output.window_size = 50; // Non-default window size

        // Test NumpyExporter
        let temp_dir1 = TempDir::new().unwrap();
        let numpy_exporter = NumpyExporter::new(temp_dir1.path());
        numpy_exporter.export(&output).unwrap();

        let metadata1: ExportMetadata = serde_json::from_str(
            &std::fs::read_to_string(temp_dir1.path().join("metadata.json")).unwrap(),
        )
        .unwrap();

        // Test BatchExporter
        let temp_dir2 = TempDir::new().unwrap();
        let batch_exporter = BatchExporter::new(temp_dir2.path(), None);
        batch_exporter.export_day("test", &output).unwrap();

        let metadata2: DayMetadata = serde_json::from_str(
            &std::fs::read_to_string(temp_dir2.path().join("test_metadata.json")).unwrap(),
        )
        .unwrap();

        // Both should have the same stride
        assert_eq!(metadata1.stride, 25, "NumpyExporter should preserve stride");
        assert_eq!(
            metadata1.window_size, 50,
            "NumpyExporter should preserve window_size"
        );

        // Note: BatchExporter uses DayMetadata which has different fields
        // But the underlying data should be consistent
        assert_eq!(
            metadata2.n_sequences, 5,
            "BatchExporter should preserve sequence count"
        );
    }
}
