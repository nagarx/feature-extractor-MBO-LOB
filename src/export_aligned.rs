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

use crate::export::dataset_config::{FeatureNormStrategy, NormalizationConfig};
use crate::export::tensor_format::{FeatureMapping, TensorFormat};
use crate::labeling::{
    LabelConfig, MultiHorizonConfig, MultiHorizonLabelGenerator, OpportunityConfig,
    OpportunityLabel, OpportunityLabelGenerator, TlobLabelGenerator, TrendLabel,
};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::WriteNpyExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

// ============================================================================
// Normalization Types
// ============================================================================

/// Normalization strategy used for feature export.
///
/// This is included in metadata so consumers know how data was normalized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationStrategy {
    /// No normalization applied (raw values)
    None,
    /// Per-feature z-score: (x - mean) / std
    PerFeatureZScore,
    /// Market-structure preserving z-score:
    /// - Prices: shared mean/std per level (ask_L and bid_L)
    /// - Sizes: independent mean/std
    #[default]
    MarketStructureZScore,
    /// Global z-score: all features share same mean/std
    GlobalZScore,
    /// Bilinear: (price - mid_price) / (k * tick_size)
    Bilinear,
}

impl std::fmt::Display for NormalizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::PerFeatureZScore => write!(f, "per_feature_zscore"),
            Self::MarketStructureZScore => write!(f, "market_structure_zscore"),
            Self::GlobalZScore => write!(f, "global_zscore"),
            Self::Bilinear => write!(f, "bilinear"),
        }
    }
}

/// Normalization parameters for a single day export.
///
/// Contains mean/std values for each feature group, enabling:
/// - Python-side validation of normalization
/// - Denormalization for interpretability
/// - Transfer to inference pipeline
///
/// # Feature Layout
///
/// Standard 40-feature LOB layout:
/// - `price_params[0..10]`: Ask prices (levels 1-10)
/// - `size_params[0..10]`: Ask sizes (levels 1-10)
/// - `price_params[0..10]`: Also used for Bid prices (shared stats)
/// - `size_params[10..20]`: Bid sizes (levels 1-10)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Normalization strategy applied
    pub strategy: NormalizationStrategy,

    /// Mean values for price features (per-level, shared ask/bid)
    /// Length: 10 (one per LOB level)
    pub price_means: Vec<f64>,

    /// Std values for price features (per-level, shared ask/bid)
    /// Length: 10 (one per LOB level)
    pub price_stds: Vec<f64>,

    /// Mean values for size features (independent per feature)
    /// Length: 20 (10 ask sizes + 10 bid sizes)
    pub size_means: Vec<f64>,

    /// Std values for size features (independent per feature)
    /// Length: 20 (10 ask sizes + 10 bid sizes)
    pub size_stds: Vec<f64>,

    /// Number of samples used to compute statistics
    pub sample_count: usize,

    /// Feature layout description for validation
    pub feature_layout: String,

    /// Number of LOB levels
    pub levels: usize,
}

impl NormalizationParams {
    /// Create new normalization params with market-structure preserving stats
    pub fn new(
        price_means: Vec<f64>,
        price_stds: Vec<f64>,
        size_means: Vec<f64>,
        size_stds: Vec<f64>,
        sample_count: usize,
        levels: usize,
    ) -> Self {
        Self {
            strategy: NormalizationStrategy::MarketStructureZScore,
            price_means,
            price_stds,
            size_means,
            size_stds,
            sample_count,
            feature_layout: format!(
                "ask_prices_{}_ask_sizes_{}_bid_prices_{}_bid_sizes_{}",
                levels, levels, levels, levels
            ),
            levels,
        }
    }

    /// Save normalization params to JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        serde_json::to_writer_pretty(file, self).map_err(|e| {
            std::io::Error::other(format!("Failed to write normalization params: {e}"))
        })?;
        Ok(())
    }

    /// Load normalization params from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let params: Self = serde_json::from_reader(file).map_err(|e| {
            std::io::Error::other(format!("Failed to read normalization params: {e}"))
        })?;
        Ok(params)
    }
}

// ============================================================================
// Export Types
// ============================================================================

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
///
/// # Features
///
/// - **Single-horizon labeling**: Default mode using `LabelConfig`
/// - **Multi-horizon labeling**: FI-2010, DeepLOB benchmark support
/// - **Tensor formatting**: Model-specific shapes (DeepLOB, HLOB, Flat)
///
/// # Example
///
/// ```ignore
/// use feature_extractor::prelude::*;
///
/// // Basic single-horizon export
/// let exporter = AlignedBatchExporter::new("output/", label_config, 100, 10);
///
/// // Multi-horizon export for FI-2010 benchmark
/// let exporter = AlignedBatchExporter::new("output/", label_config, 100, 10)
///     .with_multi_horizon_labels(MultiHorizonConfig::fi2010());
///
/// // DeepLOB-formatted export
/// let exporter = AlignedBatchExporter::new("output/", label_config, 100, 10)
///     .with_tensor_format(TensorFormat::DeepLOB { levels: 10 });
/// ```
pub struct AlignedBatchExporter {
    output_dir: PathBuf,
    label_config: LabelConfig,
    window_size: usize,
    stride: usize,
    /// Optional tensor format for model-specific reshaping
    tensor_format: Option<TensorFormat>,
    /// Optional feature mapping for tensor formatting
    feature_mapping: Option<FeatureMapping>,
    /// Optional multi-horizon configuration (replaces single-horizon if set)
    multi_horizon_config: Option<MultiHorizonConfig>,
    /// Optional opportunity config for big-move detection labeling
    opportunity_configs: Option<Vec<OpportunityConfig>>,
    /// Normalization configuration for per-feature-group strategies.
    /// Default: raw (no normalization) for TLOB paper compatibility.
    normalization_config: NormalizationConfig,
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
            tensor_format: None,
            feature_mapping: None,
            multi_horizon_config: None,
            opportunity_configs: None,
            normalization_config: NormalizationConfig::default(), // Raw by default
        }
    }

    /// Set normalization configuration (builder pattern).
    ///
    /// Controls how each feature group is normalized during export.
    /// Default is raw (no normalization) for TLOB paper compatibility.
    ///
    /// # Arguments
    /// * `config` - Normalization configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::export::dataset_config::{NormalizationConfig, FeatureNormStrategy};
    ///
    /// // For TLOB paper (raw data, model handles normalization via BiN)
    /// let exporter = AlignedBatchExporter::new("output/", config, 100, 10)
    ///     .with_normalization(NormalizationConfig::raw());
    ///
    /// // For DeepLOB (per-feature z-score)
    /// let exporter = AlignedBatchExporter::new("output/", config, 100, 10)
    ///     .with_normalization(NormalizationConfig::deeplob());
    ///
    /// // Custom configuration
    /// let exporter = AlignedBatchExporter::new("output/", config, 100, 10)
    ///     .with_normalization(
    ///         NormalizationConfig::default()
    ///             .with_lob_prices(FeatureNormStrategy::GlobalZScore)
    ///             .with_lob_sizes(FeatureNormStrategy::ZScore)
    ///     );
    /// ```
    pub fn with_normalization(mut self, config: NormalizationConfig) -> Self {
        self.normalization_config = config;
        self
    }

    /// Get the current normalization configuration.
    #[inline]
    pub fn normalization_config(&self) -> &NormalizationConfig {
        &self.normalization_config
    }

    /// Enable tensor formatting for model-specific shapes (builder pattern).
    ///
    /// # Arguments
    /// * `format` - Target tensor format (DeepLOB, HLOB, Flat, Image)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let exporter = AlignedBatchExporter::new("output/", config, 100, 10)
    ///     .with_tensor_format(TensorFormat::DeepLOB { levels: 10 });
    /// ```
    ///
    /// # Output Shapes
    ///
    /// | Format | Shape | Use Case |
    /// |--------|-------|----------|
    /// | `Flat` | (N, T, F) | TLOB, LSTM, MLP |
    /// | `DeepLOB` | (N, T, 4, L) | DeepLOB CNN |
    /// | `HLOB` | (N, T, L, 4) | HLOB models |
    /// | `Image` | (N, T, C, H, W) | Vision models |
    pub fn with_tensor_format(mut self, format: TensorFormat) -> Self {
        self.tensor_format = Some(format);
        self
    }

    /// Set custom feature mapping for tensor formatting (builder pattern).
    ///
    /// Required when using non-default feature layouts.
    /// Default mapping assumes standard 40-feature LOB layout.
    ///
    /// # Arguments
    /// * `mapping` - Custom feature index mapping
    pub fn with_feature_mapping(mut self, mapping: FeatureMapping) -> Self {
        self.feature_mapping = Some(mapping);
        self
    }

    /// Enable multi-horizon label generation (builder pattern).
    ///
    /// When set, generates labels for multiple prediction horizons instead
    /// of single-horizon labeling. Required for FI-2010 benchmark reproduction.
    ///
    /// # Arguments
    /// * `config` - Multi-horizon configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// // FI-2010 benchmark: horizons [10, 20, 30, 50, 100]
    /// let exporter = AlignedBatchExporter::new("output/", config, 100, 10)
    ///     .with_multi_horizon_labels(MultiHorizonConfig::fi2010());
    ///
    /// // DeepLOB benchmark: horizons [10, 20, 50, 100]
    /// let exporter = AlignedBatchExporter::new("output/", config, 100, 10)
    ///     .with_multi_horizon_labels(MultiHorizonConfig::deeplob());
    /// ```
    ///
    /// # Output
    ///
    /// - `{day}_labels.npy`: Shape (N, num_horizons) instead of (N,)
    /// - `{day}_metadata.json`: Includes horizon configuration
    pub fn with_multi_horizon_labels(mut self, config: MultiHorizonConfig) -> Self {
        self.multi_horizon_config = Some(config);
        self
    }

    /// Enable opportunity-based labeling for big-move detection (builder pattern).
    ///
    /// When set, uses peak-return-based labeling instead of smoothed average labeling.
    /// This is designed for detecting "big moves" rather than trend direction.
    ///
    /// # Arguments
    /// * `configs` - Vector of OpportunityConfig, one per horizon
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::labeling::OpportunityConfig;
    ///
    /// // Detect 0.5% moves at horizons 50, 100, 200
    /// let configs = vec![
    ///     OpportunityConfig::new(50, 0.005),
    ///     OpportunityConfig::new(100, 0.005),
    ///     OpportunityConfig::new(200, 0.005),
    /// ];
    /// let exporter = AlignedBatchExporter::new("output/", label_config, 100, 10)
    ///     .with_opportunity_labels(configs);
    /// ```
    ///
    /// # Output
    ///
    /// - Labels: 0 = BigDown, 1 = NoOpportunity, 2 = BigUp
    /// - Multi-horizon: `{day}_labels.npy` shape (N, num_horizons)
    /// - Single-horizon: `{day}_labels.npy` shape (N,)
    pub fn with_opportunity_labels(mut self, configs: Vec<OpportunityConfig>) -> Self {
        self.opportunity_configs = Some(configs);
        self
    }

    /// Check if opportunity labeling is enabled.
    #[inline]
    pub fn is_opportunity_labeling(&self) -> bool {
        self.opportunity_configs.is_some()
    }

    /// Check if multi-horizon labeling is enabled.
    #[inline]
    pub fn is_multi_horizon(&self) -> bool {
        self.multi_horizon_config.is_some()
    }

    /// Check if tensor formatting is enabled.
    #[inline]
    pub fn is_tensor_formatted(&self) -> bool {
        self.tensor_format.is_some()
    }

    /// Get the configured tensor format, if any.
    #[inline]
    pub fn tensor_format(&self) -> Option<&TensorFormat> {
        self.tensor_format.as_ref()
    }

    /// Get the configured multi-horizon config, if any.
    #[inline]
    pub fn multi_horizon_config(&self) -> Option<&MultiHorizonConfig> {
        self.multi_horizon_config.as_ref()
    }

    /// Export a single day with aligned sequences and labels
    ///
    /// # Process
    /// 1. Generate labels from mid-prices (single or multi-horizon)
    /// 2. Extract sequences that have corresponding labels
    /// 3. Align sequences with labels (1:1 mapping)
    /// 4. Apply tensor formatting if configured
    /// 5. Export both with validation
    ///
    /// # Output Files
    ///
    /// ## Standard Mode (single-horizon)
    /// - `{day}_sequences.npy`: Shape depends on tensor format
    /// - `{day}_labels.npy`: `[N_seq]` int8
    /// - `{day}_metadata.json`: Metadata with validation info
    ///
    /// ## Multi-Horizon Mode
    /// - `{day}_sequences.npy`: Shape depends on tensor format
    /// - `{day}_labels.npy`: `[N_seq, num_horizons]` int8
    /// - `{day}_horizons.json`: Horizon configuration
    /// - `{day}_metadata.json`: Metadata with horizon info
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

        // Branch based on labeling mode (order matters: opportunity takes precedence)
        if let Some(opp_configs) = &self.opportunity_configs {
            // Opportunity labeling path (big-move detection)
            self.export_day_opportunity(day_name, output, opp_configs.clone())
        } else if let Some(multi_config) = &self.multi_horizon_config {
            // Multi-horizon TLOB labeling path
            self.export_day_multi_horizon(day_name, output, multi_config.clone())
        } else {
            // Single-horizon TLOB labeling path (original behavior)
            self.export_day_single_horizon(day_name, output)
        }
    }

    /// Export with single-horizon labeling (original behavior)
    fn export_day_single_horizon(
        &self,
        day_name: &str,
        output: &PipelineOutput,
    ) -> Result<AlignedDayExport> {
        let features_extracted = output.features_extracted;
        let sequences_generated = output.sequences.len();

        // Step 1: Generate labels from mid-prices
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
        let (aligned_sequences, norm_params) = self.normalize_sequences(&aligned_sequences)?;

        // Step 4: Export sequences (with optional tensor formatting)
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
        self.export_sequences_with_format(&aligned_sequences, seq_shape, &sequences_path)?;

        // Step 5: Export labels
        let labels_path = self.output_dir.join(format!("{day_name}_labels.npy"));
        self.export_labels(&aligned_labels, &labels_path)?;

        // Step 5.5: Export normalization params (enables Python-side denormalization)
        let norm_params_path = self
            .output_dir
            .join(format!("{day_name}_normalization.json"));
        norm_params.save_json(&norm_params_path)?;
        println!("  📊 Exported normalization params: {:?}", norm_params_path);

        // Step 6: Export metadata with verification info
        let drop_rate = if sequences_generated > 0 {
            (sequences_dropped as f64 / sequences_generated as f64) * 100.0
        } else {
            0.0
        };

        let tensor_format_str = self.tensor_format.as_ref().map(|f| format!("{:?}", f));

        let metadata = serde_json::json!({
            "day": day_name,
            "n_sequences": aligned_sequences.len(),
            "window_size": seq_shape.0,
            "n_features": seq_shape.1,
            "tensor_format": tensor_format_str,
            "label_mode": "single_horizon",
            "label_distribution": label_dist,
            "messages_processed": output.messages_processed,
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
            "normalization": {
                "strategy": norm_params.strategy.to_string(),
                "applied": true,
                "levels": norm_params.levels,
                "sample_count": norm_params.sample_count,
                "feature_layout": &norm_params.feature_layout,
                "params_file": format!("{day_name}_normalization.json"),
            },
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
            buffer_size: 50_000,
            sequences_generated,
            sequences_dropped,
        })
    }

    /// Export with multi-horizon labeling (FI-2010, DeepLOB benchmark mode)
    fn export_day_multi_horizon(
        &self,
        day_name: &str,
        output: &PipelineOutput,
        config: MultiHorizonConfig,
    ) -> Result<AlignedDayExport> {
        let features_extracted = output.features_extracted;
        let sequences_generated = output.sequences.len();
        let horizons = config.horizons().to_vec();

        // Step 1: Generate multi-horizon labels
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

        // Step 2: Build label matrix (N_prices × num_horizons)
        // We need to find the intersection of all valid label indices
        let (label_indices, label_matrix) =
            self.build_multi_horizon_label_matrix(&multi_labels, &horizons)?;

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            label_indices.len()
        );

        // Step 3: Extract aligned sequences
        println!("  🔗 Aligning sequences with multi-horizon labels...");
        let (aligned_sequences, aligned_label_matrix) = self.align_sequences_with_multi_labels(
            &output.sequences,
            &label_indices,
            &label_matrix,
        )?;

        let sequences_dropped = sequences_generated.saturating_sub(aligned_sequences.len());

        println!(
            "    Aligned {} sequences with {} horizons",
            aligned_sequences.len(),
            horizons.len()
        );
        println!("    Dropped {sequences_dropped} sequences (no label)");

        // Step 4: Validate alignment
        self.validate_multi_horizon_alignment(&aligned_sequences, &aligned_label_matrix)?;

        // Step 5: Verify RAW spreads
        self.verify_raw_spreads(&aligned_sequences)?;

        // Step 6: Normalize sequences
        let (aligned_sequences, norm_params) = self.normalize_sequences(&aligned_sequences)?;

        // Step 7: Export sequences (with optional tensor formatting)
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
        self.export_sequences_with_format(&aligned_sequences, seq_shape, &sequences_path)?;

        // Step 8: Export multi-horizon labels
        let labels_path = self.output_dir.join(format!("{day_name}_labels.npy"));
        self.export_multi_horizon_labels(&aligned_label_matrix, &labels_path)?;

        // Step 8.5: Export normalization params
        let norm_params_path = self
            .output_dir
            .join(format!("{day_name}_normalization.json"));
        norm_params.save_json(&norm_params_path)?;
        println!("  📊 Exported normalization params: {:?}", norm_params_path);

        // Step 9: Export horizons configuration
        let horizons_path = self.output_dir.join(format!("{day_name}_horizons.json"));
        let horizons_json = serde_json::json!({
            "horizons": horizons,
            "num_horizons": horizons.len(),
            "smoothing_window": config.smoothing_window,
            "threshold_strategy": format!("{:?}", config.threshold_strategy),
        });
        let mut file = File::create(&horizons_path)?;
        serde_json::to_writer_pretty(&mut file, &horizons_json)
            .map_err(|e| std::io::Error::other(format!("Failed to write horizons: {e}")))?;

        // Step 10: Compute label distribution per horizon
        let mut label_dist = HashMap::new();
        for (h_idx, horizon) in horizons.iter().enumerate() {
            let mut h_dist = HashMap::new();
            let mut up_count = 0usize;
            let mut down_count = 0usize;
            let mut stable_count = 0usize;

            for row in &aligned_label_matrix {
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
            label_dist.insert(format!("h{}", horizon), h_dist);
        }

        // Step 11: Export metadata
        let drop_rate = if sequences_generated > 0 {
            (sequences_dropped as f64 / sequences_generated as f64) * 100.0
        } else {
            0.0
        };

        let tensor_format_str = self.tensor_format.as_ref().map(|f| format!("{:?}", f));

        let metadata = serde_json::json!({
            "day": day_name,
            "n_sequences": aligned_sequences.len(),
            "window_size": seq_shape.0,
            "n_features": seq_shape.1,
            "tensor_format": tensor_format_str,
            "label_mode": "multi_horizon",
            "horizons": horizons,
            "num_horizons": horizons.len(),
            "label_distribution_per_horizon": label_dist,
            "messages_processed": output.messages_processed,
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
            "normalization": {
                "strategy": norm_params.strategy.to_string(),
                "applied": true,
                "levels": norm_params.levels,
                "sample_count": norm_params.sample_count,
                "feature_layout": &norm_params.feature_layout,
                "params_file": format!("{day_name}_normalization.json"),
            },
            "validation": {
                "sequences_labels_match": aligned_sequences.len() == aligned_label_matrix.len(),
                "all_horizons_valid": true,
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
            "  ✅ Multi-horizon export complete: {} sequences × {} horizons",
            aligned_sequences.len(),
            horizons.len()
        );

        // Return with combined distribution
        let mut combined_dist = HashMap::new();
        combined_dist.insert("Up".to_string(), 0);
        combined_dist.insert("Down".to_string(), 0);
        combined_dist.insert("Stable".to_string(), 0);

        // Use first horizon for summary distribution
        if let Some(first_dist) = label_dist.get(&format!("h{}", horizons[0])) {
            combined_dist = first_dist.clone();
        }

        Ok(AlignedDayExport {
            day: day_name.to_string(),
            n_sequences: aligned_sequences.len(),
            seq_shape,
            label_distribution: combined_dist,
            messages_processed: output.messages_processed,
            export_path: self.output_dir.clone(),
            features_extracted,
            buffer_size: 50_000,
            sequences_generated,
            sequences_dropped,
        })
    }

    // ========================================================================
    // Opportunity Labeling Methods
    // ========================================================================

    /// Export with opportunity-based labeling (big-move detection)
    fn export_day_opportunity(
        &self,
        day_name: &str,
        output: &PipelineOutput,
        configs: Vec<OpportunityConfig>,
    ) -> Result<AlignedDayExport> {
        let features_extracted = output.features_extracted;
        let sequences_generated = output.sequences.len();
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

        // Step 1: Generate opportunity labels for each horizon
        let mut all_labels: Vec<Vec<(usize, OpportunityLabel, f64, f64)>> = Vec::new();
        
        for config in &configs {
            let mut generator = OpportunityLabelGenerator::new(config.clone());
            generator.add_prices(&output.mid_prices);
            let labels = generator.generate_labels()?;
            
            println!(
                "    Horizon {}: {} labels, {:.1}% opportunities",
                config.horizon,
                labels.len(),
                labels.iter().filter(|(_, l, _, _)| l.is_opportunity()).count() as f64 / labels.len() as f64 * 100.0
            );
            
            all_labels.push(labels);
        }

        // Step 2: Build label matrix (intersect valid indices)
        let (label_indices, label_matrix, combined_dist) =
            self.build_opportunity_label_matrix(&all_labels, is_multi_horizon)?;

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            label_indices.len()
        );

        // Step 3: Extract aligned sequences
        println!("  🔗 Aligning sequences with opportunity labels...");
        let (aligned_sequences, aligned_label_matrix) = if is_multi_horizon {
            self.align_sequences_with_multi_labels(
                &output.sequences,
                &label_indices,
                &label_matrix,
            )?
        } else {
            // Single-horizon: convert to simpler format
            let single_labels: Vec<i8> = label_matrix.iter().map(|v| v[0]).collect();
            let (seqs, labels) = self.align_sequences_with_labels(
                &output.sequences,
                &label_indices,
                &single_labels,
            )?;
            (seqs, labels.iter().map(|&l| vec![l]).collect())
        };

        let sequences_dropped = sequences_generated.saturating_sub(aligned_sequences.len());

        println!(
            "    Aligned {} sequences with {} horizon(s)",
            aligned_sequences.len(),
            horizons.len()
        );
        println!("    Dropped {sequences_dropped} sequences (no label)");

        // Step 4: Validate alignment
        self.validate_multi_horizon_alignment(&aligned_sequences, &aligned_label_matrix)?;

        // Step 5: Verify RAW spreads
        self.verify_raw_spreads(&aligned_sequences)?;

        // Step 6: Normalize sequences
        let (aligned_sequences, norm_params) = self.normalize_sequences(&aligned_sequences)?;

        // Step 7: Export sequences
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
        self.export_sequences_with_format(&aligned_sequences, seq_shape, &sequences_path)?;

        // Step 8: Export labels
        let labels_path = self.output_dir.join(format!("{day_name}_labels.npy"));
        if is_multi_horizon {
            // Export multi-horizon labels
            let n_seq = aligned_label_matrix.len();
            let n_horizons = horizons.len();
            let mut labels_array = Array2::<i8>::zeros((n_seq, n_horizons));
            for (i, row) in aligned_label_matrix.iter().enumerate() {
                for (j, &label) in row.iter().enumerate() {
                    labels_array[[i, j]] = label;
                }
            }
            let file = File::create(&labels_path)?;
            labels_array.write_npy(file).map_err(|e| {
                std::io::Error::other(format!("Failed to write labels: {e}"))
            })?;
            println!("  💾 Exported labels: {} [{}, {}]", labels_path.display(), n_seq, n_horizons);
        } else {
            // Export single-horizon labels
            let labels_vec: Vec<i8> = aligned_label_matrix.iter().map(|v| v[0]).collect();
            let labels_array = Array1::from(labels_vec);
            let file = File::create(&labels_path)?;
            labels_array.write_npy(file).map_err(|e| {
                std::io::Error::other(format!("Failed to write labels: {e}"))
            })?;
            println!("  💾 Exported labels: {} [{}]", labels_path.display(), aligned_label_matrix.len());
        }

        // Step 9: Export normalization params
        norm_params.save_json(self.output_dir.join(format!("{day_name}_normalization.json")))?;

        // Step 10: Export metadata with opportunity-specific info
        let drop_rate = if sequences_generated > 0 {
            (sequences_dropped as f64 / sequences_generated as f64) * 100.0
        } else {
            0.0
        };

        let tensor_format_str = self.tensor_format.as_ref().map(|f| format!("{:?}", f));

        let metadata = serde_json::json!({
            "day": day_name,
            "n_sequences": aligned_sequences.len(),
            "window_size": seq_shape.0,
            "n_features": seq_shape.1,
            "tensor_format": tensor_format_str,
            "labeling": {
                "strategy": "opportunity",
                "horizons": horizons,
                "threshold": configs[0].threshold,
                "threshold_bps": configs[0].threshold * 10000.0,
                "conflict_priority": format!("{:?}", configs[0].conflict_priority),
            },
            "label_distribution": combined_dist,
            "label_encoding": {
                "format": "signed_int8",
                "values": {
                    "-1": "BigDown",
                    "0": "NoOpportunity",
                    "1": "BigUp",
                },
                "class_index_mapping": "class_idx = label + 1  # For softmax: -1→0, 0→1, 1→2",
            },
            "normalization": {
                "strategy": "market_structure_zscore",
                "params_file": format!("{day_name}_normalization.json"),
            },
            "processing": {
                "messages_processed": output.messages_processed,
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
        println!("  💾 Exported metadata: {}", metadata_path.display());

        // Step 11: Export horizon config
        let horizons_config = serde_json::json!({
            "labeling_strategy": "opportunity",
            "horizons": horizons,
            "threshold": configs[0].threshold,
            "threshold_bps": configs[0].threshold * 10000.0,
            "conflict_priority": format!("{:?}", configs[0].conflict_priority),
        });
        let horizons_path = self.output_dir.join(format!("{day_name}_horizons.json"));
        let horizons_file = File::create(&horizons_path)?;
        serde_json::to_writer_pretty(horizons_file, &horizons_config)
            .map_err(|e| std::io::Error::other(format!("Failed to write horizons config: {e}")))?;
        println!("  💾 Exported horizons config: {}", horizons_path.display());

        Ok(AlignedDayExport {
            day: day_name.to_string(),
            n_sequences: aligned_sequences.len(),
            seq_shape,
            label_distribution: combined_dist,
            messages_processed: output.messages_processed,
            export_path: self.output_dir.clone(),
            features_extracted,
            buffer_size: 50_000,
            sequences_generated,
            sequences_dropped,
        })
    }

    /// Build label matrix for opportunity labeling.
    ///
    /// Returns (indices, label_matrix, distribution) where:
    /// - indices: Valid snapshot indices that have labels for ALL horizons
    /// - label_matrix: Vec<Vec<i8>> where each row has labels for all horizons
    /// - distribution: Combined label distribution across all horizons
    #[allow(clippy::type_complexity)]
    fn build_opportunity_label_matrix(
        &self,
        all_labels: &[Vec<(usize, OpportunityLabel, f64, f64)>],
        _is_multi_horizon: bool,
    ) -> Result<(Vec<usize>, Vec<Vec<i8>>, HashMap<String, usize>)> {
        use std::collections::BTreeSet;

        // Find intersection of all valid indices across horizons
        let mut valid_indices: Option<BTreeSet<usize>> = None;

        for horizon_labels in all_labels {
            let indices: BTreeSet<usize> = horizon_labels.iter().map(|(idx, _, _, _)| *idx).collect();
            valid_indices = match valid_indices {
                None => Some(indices),
                Some(prev) => Some(prev.intersection(&indices).cloned().collect()),
            };
        }

        let valid_indices: Vec<usize> = valid_indices
            .unwrap_or_default()
            .into_iter()
            .collect();

        if valid_indices.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No valid indices found across all horizons",
            )
            .into());
        }

        // Create index lookup for each horizon
        let mut horizon_lookups: Vec<HashMap<usize, OpportunityLabel>> = Vec::new();
        for horizon_labels in all_labels {
            let lookup: HashMap<usize, OpportunityLabel> = horizon_labels
                .iter()
                .map(|(idx, label, _, _)| (*idx, *label))
                .collect();
            horizon_lookups.push(lookup);
        }

        // Build label matrix
        let mut label_matrix = Vec::with_capacity(valid_indices.len());
        let mut dist = HashMap::new();
        dist.insert("BigDown".to_string(), 0);
        dist.insert("NoOpportunity".to_string(), 0);
        dist.insert("BigUp".to_string(), 0);

        for &idx in &valid_indices {
            let mut row = Vec::with_capacity(all_labels.len());
            for lookup in &horizon_lookups {
                let label = lookup.get(&idx).unwrap_or(&OpportunityLabel::NoOpportunity);
                // Use as_int() for consistency with TLOB labels: -1=BigDown, 0=NoOpp, 1=BigUp
                row.push(label.as_int());
                *dist.entry(label.name().to_string()).or_insert(0) += 1;
            }
            label_matrix.push(row);
        }

        Ok((valid_indices, label_matrix, dist))
    }

    // ========================================================================
    // Multi-Horizon Helper Methods
    // ========================================================================

    /// Build label matrix for multi-horizon labeling.
    ///
    /// Returns (indices, label_matrix) where:
    /// - indices: Valid snapshot indices that have labels for ALL horizons
    /// - label_matrix: Vec<Vec<i8>> where each row has labels for all horizons
    #[allow(clippy::type_complexity)]
    fn build_multi_horizon_label_matrix(
        &self,
        multi_labels: &crate::labeling::MultiHorizonLabels,
        horizons: &[usize],
    ) -> Result<(Vec<usize>, Vec<Vec<i8>>)> {
        use std::collections::BTreeSet;

        // Find intersection of all valid indices across horizons
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

        // Build label matrix: for each valid index, collect labels from all horizons
        let mut label_matrix: Vec<Vec<i8>> = Vec::with_capacity(valid_indices.len());

        // Create lookup maps for each horizon
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
    fn align_sequences_with_multi_labels(
        &self,
        sequences: &[crate::sequence_builder::Sequence],
        label_indices: &[usize],
        label_matrix: &[Vec<i8>],
    ) -> Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<i8>>)> {
        // Create lookup map: index → label row
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

    /// Validate multi-horizon alignment.
    fn validate_multi_horizon_alignment(
        &self,
        sequences: &[Vec<Vec<f64>>],
        label_matrix: &[Vec<i8>],
    ) -> Result<()> {
        // Check 1: Same count
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

        // Check 2: All label rows have same width
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

        // Check 3: All labels are valid {-1, 0, 1}
        for (i, row) in label_matrix.iter().enumerate() {
            for (h, &label) in row.iter().enumerate() {
                if !(-1..=1).contains(&label) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid label at row {}, horizon {}: {}", i, h, label),
                    )
                    .into());
                }
            }
        }

        println!(
            "    ✓ Multi-horizon validation passed: {} sequences × {} horizons",
            sequences.len(),
            label_matrix.first().map(|r| r.len()).unwrap_or(0)
        );

        Ok(())
    }

    /// Export multi-horizon labels as 2D numpy array.
    ///
    /// Shape: [n_sequences, num_horizons]
    fn export_multi_horizon_labels(&self, label_matrix: &[Vec<i8>], path: &Path) -> Result<()> {
        let n_seq = label_matrix.len();
        let n_horizons = label_matrix.first().map(|r| r.len()).unwrap_or(0);

        // Flatten to 1D vec in row-major order
        let flat: Vec<i8> = label_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Create 2D array
        let array = Array2::from_shape_vec((n_seq, n_horizons), flat)
            .map_err(|e| format!("Failed to create 2D label array: {e}"))?;

        // Write to file
        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write multi-horizon labels: {e}"))?;

        println!(
            "  ✅ Multi-horizon labels: {} [{} × {}]",
            path.display(),
            n_seq,
            n_horizons
        );

        Ok(())
    }

    // ========================================================================
    // Tensor Formatting Helper Methods
    // ========================================================================

    /// Export sequences with optional tensor formatting.
    fn export_sequences_with_format(
        &self,
        sequences: &[Vec<Vec<f64>>],
        shape: (usize, usize),
        path: &Path,
    ) -> Result<()> {
        match &self.tensor_format {
            None => {
                // Default flat export (N, T, F)
                self.export_sequences(sequences, shape, path)
            }
            Some(TensorFormat::Flat) => {
                // Explicit flat format
                self.export_sequences(sequences, shape, path)
            }
            Some(TensorFormat::DeepLOB { levels }) => {
                self.export_sequences_deeplob(sequences, *levels, path)
            }
            Some(TensorFormat::HLOB { levels }) => {
                self.export_sequences_hlob(sequences, *levels, path)
            }
            Some(TensorFormat::Image {
                channels,
                height,
                width,
            }) => self.export_sequences_image(sequences, *channels, *height, *width, path),
        }
    }

    /// Export sequences in DeepLOB format: (N, T, 4, L)
    ///
    /// Channels: [ask_prices, ask_volumes, bid_prices, bid_volumes]
    fn export_sequences_deeplob(
        &self,
        sequences: &[Vec<Vec<f64>>],
        levels: usize,
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let n_timesteps = sequences.first().map(|s| s.len()).unwrap_or(0);

        // Reshape: (N, T, F) -> (N, T, 4, L)
        // Feature layout: [Ask_prices(L), Ask_sizes(L), Bid_prices(L), Bid_sizes(L)]
        let mut data: Vec<f32> = Vec::with_capacity(n_seq * n_timesteps * 4 * levels);

        for seq in sequences {
            for timestep in seq {
                // Channel 0: Ask prices (indices 0..levels)
                for l in 0..levels {
                    data.push(timestep.get(l).copied().unwrap_or(0.0) as f32);
                }
                // Channel 1: Ask volumes (indices levels..2*levels)
                for l in 0..levels {
                    data.push(timestep.get(levels + l).copied().unwrap_or(0.0) as f32);
                }
                // Channel 2: Bid prices (indices 2*levels..3*levels)
                for l in 0..levels {
                    data.push(timestep.get(2 * levels + l).copied().unwrap_or(0.0) as f32);
                }
                // Channel 3: Bid volumes (indices 3*levels..4*levels)
                for l in 0..levels {
                    data.push(timestep.get(3 * levels + l).copied().unwrap_or(0.0) as f32);
                }
            }
        }

        let array = Array4::from_shape_vec((n_seq, n_timesteps, 4, levels), data)
            .map_err(|e| format!("Failed to create DeepLOB array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write DeepLOB sequences: {e}"))?;

        println!(
            "  ✅ DeepLOB sequences: {} [{} × {} × 4 × {}]",
            path.display(),
            n_seq,
            n_timesteps,
            levels
        );

        Ok(())
    }

    /// Export sequences in HLOB format: (N, T, L, 4)
    ///
    /// Per-level features: [ask_price, ask_volume, bid_price, bid_volume]
    fn export_sequences_hlob(
        &self,
        sequences: &[Vec<Vec<f64>>],
        levels: usize,
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let n_timesteps = sequences.first().map(|s| s.len()).unwrap_or(0);

        // Reshape: (N, T, F) -> (N, T, L, 4)
        let mut data: Vec<f32> = Vec::with_capacity(n_seq * n_timesteps * levels * 4);

        for seq in sequences {
            for timestep in seq {
                for l in 0..levels {
                    // Feature 0: Ask price
                    data.push(timestep.get(l).copied().unwrap_or(0.0) as f32);
                    // Feature 1: Ask volume
                    data.push(timestep.get(levels + l).copied().unwrap_or(0.0) as f32);
                    // Feature 2: Bid price
                    data.push(timestep.get(2 * levels + l).copied().unwrap_or(0.0) as f32);
                    // Feature 3: Bid volume
                    data.push(timestep.get(3 * levels + l).copied().unwrap_or(0.0) as f32);
                }
            }
        }

        let array = Array4::from_shape_vec((n_seq, n_timesteps, levels, 4), data)
            .map_err(|e| format!("Failed to create HLOB array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write HLOB sequences: {e}"))?;

        println!(
            "  ✅ HLOB sequences: {} [{} × {} × {} × 4]",
            path.display(),
            n_seq,
            n_timesteps,
            levels
        );

        Ok(())
    }

    /// Export sequences in Image format: (N, T, C, H, W)
    fn export_sequences_image(
        &self,
        sequences: &[Vec<Vec<f64>>],
        channels: usize,
        height: usize,
        width: usize,
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let n_timesteps = sequences.first().map(|s| s.len()).unwrap_or(0);

        // For image format, we reshape the flat features into (C, H, W)
        let expected_features = channels * height * width;

        let mut data: Vec<f32> = Vec::with_capacity(n_seq * n_timesteps * expected_features);

        for seq in sequences {
            for timestep in seq {
                for i in 0..expected_features {
                    data.push(timestep.get(i).copied().unwrap_or(0.0) as f32);
                }
            }
        }

        // Create 5D array - ndarray doesn't support 5D directly, so we use dynamic
        let shape = ndarray::IxDyn(&[n_seq, n_timesteps, channels, height, width]);
        let array = ndarray::ArrayD::from_shape_vec(shape, data)
            .map_err(|e| format!("Failed to create Image array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write Image sequences: {e}"))?;

        println!(
            "  ✅ Image sequences: {} [{} × {} × {} × {} × {}]",
            path.display(),
            n_seq,
            n_timesteps,
            channels,
            height,
            width
        );

        Ok(())
    }

    // ========================================================================
    // Original Helper Methods
    // ========================================================================

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

    /// Apply configurable normalization based on NormalizationConfig.
    ///
    /// This method applies different normalization strategies to each feature group
    /// based on the configured settings. Supports multiple strategies to match
    /// requirements from different research papers (TLOB, DeepLOB, LOBench, etc.).
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
    /// # Strategies
    ///
    /// - `None`: Raw values (no normalization) - default for TLOB paper
    /// - `ZScore`: Per-feature z-score: (x - mean) / std
    /// - `GlobalZScore`: Shared mean/std across all features in group
    /// - `MarketStructure`: Ask+bid share stats per level (preserves ordering)
    /// - `PercentageChange`: (x - reference) / reference
    /// - `MinMax`: Scale to [0, 1] range
    /// - `Bilinear`: (price - mid) / (k * tick)
    ///
    /// # Returns
    ///
    /// Tuple of (normalized_sequences, normalization_params) for export.
    fn normalize_sequences(
        &self,
        sequences: &[Vec<Vec<f64>>],
    ) -> Result<(Vec<Vec<Vec<f64>>>, NormalizationParams)> {
        let levels = 10; // Standard LOB levels
        
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

        // Check if no normalization is needed (raw export for TLOB paper)
        if !config.any_normalization() {
            println!("  🔧 Normalization: NONE (raw export for TLOB paper compatibility)");
            
            // Return sequences as-is with placeholder params
            let norm_params = NormalizationParams {
                strategy: NormalizationStrategy::None,
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
            
            // Deep copy sequences (no transformation)
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

        // Step 1: Compute statistics for each feature group
        let (price_stats, size_stats, derived_stats, mbo_stats, total_samples) = 
            self.compute_feature_statistics(sequences, levels)?;

        // Step 2: Create normalization params (for metadata export)
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

        // Step 3: Normalize all sequences
        let mut normalized = Vec::with_capacity(sequences.len());

        for seq in sequences {
            let mut norm_seq = Vec::with_capacity(seq.len());
            for timestep in seq {
                let mut norm_timestep = Vec::with_capacity(n_features);

                // === LOB Prices (0-9, 20-29) ===
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
                        price_stats.0[level], // Same stats as ask for MarketStructure
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

                // === Derived features (40-47) ===
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

                // === MBO features (48-83) ===
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

                // === Signal features (84-97) ===
                // CRITICAL: Signals include categorical features (book_valid, time_regime, etc.)
                // that MUST NOT be normalized. Always copy as-is regardless of config.
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
    ///
    /// Returns ((price_means, price_stds), (size_means, size_stds), (derived_means, derived_stds), (mbo_means, mbo_stds), sample_count)
    #[allow(clippy::type_complexity)]
    fn compute_feature_statistics(
        &self,
        sequences: &[Vec<Vec<f64>>],
        levels: usize,
    ) -> Result<(
        (Vec<f64>, Vec<f64>),  // price (means, stds)
        (Vec<f64>, Vec<f64>),  // size (means, stds)
        (Vec<f64>, Vec<f64>),  // derived (means, stds)
        (Vec<f64>, Vec<f64>),  // mbo (means, stds)
        usize,                 // sample count
    )> {
        let n_features = sequences[0][0].len();
        let epsilon = 1e-8;
        let config = &self.normalization_config;

        // Collect values per feature group
        let mut price_level_values: Vec<Vec<f64>> = vec![Vec::new(); levels];
        let mut size_values: Vec<Vec<f64>> = vec![Vec::new(); levels * 2];
        let mut derived_values: Vec<Vec<f64>> = vec![Vec::new(); 8];
        let mut mbo_values: Vec<Vec<f64>> = vec![Vec::new(); 36];
        
        // For GlobalZScore: collect ALL prices/sizes into single containers
        let mut all_prices: Vec<f64> = Vec::new();
        let mut all_sizes: Vec<f64> = Vec::new();
        
        let mut total_samples = 0usize;

        for seq in sequences {
            for timestep in seq {
                total_samples += 1;

                // Collect LOB prices based on strategy
                match config.lob_prices {
                    FeatureNormStrategy::GlobalZScore => {
                        // GlobalZScore: ALL prices share ONE mean/std (TLOB repo style)
                        for level in 0..levels {
                            all_prices.push(timestep[level]);       // Ask price
                            all_prices.push(timestep[20 + level]);  // Bid price
                        }
                    }
                    FeatureNormStrategy::MarketStructure => {
                        // MarketStructure: Ask/Bid at same level share stats
                for level in 0..levels {
                    let ask_price = timestep[level];
                    let bid_price = timestep[20 + level];
                    price_level_values[level].push(ask_price);
                    price_level_values[level].push(bid_price);
                        }
                    }
                    _ => {
                        // Per-feature: each level has independent stats
                        for level in 0..levels {
                            price_level_values[level].push(timestep[level]);
                        }
                    }
                }

                // Collect LOB sizes based on strategy
                match config.lob_sizes {
                    FeatureNormStrategy::GlobalZScore => {
                        // GlobalZScore: ALL sizes share ONE mean/std (TLOB repo style)
                for i in 0..levels {
                            all_sizes.push(timestep[10 + i]);   // Ask sizes
                            all_sizes.push(timestep[30 + i]);   // Bid sizes
                        }
                    }
                    _ => {
                        // Per-feature: each size column has independent stats
                        for i in 0..levels {
                            size_values[i].push(timestep[10 + i]);           // Ask sizes
                            size_values[levels + i].push(timestep[30 + i]);  // Bid sizes
                        }
                    }
                }

                // Collect derived features
                for i in 0..8 {
                    let idx = 40 + i;
                    if idx < n_features {
                        derived_values[i].push(timestep[idx]);
                    }
                }

                // Collect MBO features
                for i in 0..36 {
                    let idx = 48 + i;
                    if idx < n_features {
                        mbo_values[i].push(timestep[idx]);
                    }
                }
            }
        }

        // Compute statistics for prices
        let mut price_means = vec![0.0; levels];
        let mut price_stds = vec![1.0; levels];

        if config.lob_prices == FeatureNormStrategy::GlobalZScore && !all_prices.is_empty() {
            // GlobalZScore: ONE mean/std for ALL prices (TLOB repo style)
            let global_mean: f64 = all_prices.iter().sum::<f64>() / all_prices.len() as f64;
            let global_variance: f64 = all_prices.iter()
                .map(|v| (v - global_mean).powi(2))
                .sum::<f64>() / all_prices.len() as f64;
            let global_std = if global_variance > epsilon { global_variance.sqrt() } else { 1.0 };
            
            // Fill all levels with the same global stats
            for level in 0..levels {
                price_means[level] = global_mean;
                price_stds[level] = global_std;
            }
            println!("    📊 GlobalZScore prices: mean={:.6}, std={:.6} (shared across all {} price columns)",
                global_mean, global_std, levels * 2);
        } else {
            // Per-level or MarketStructure stats
        for level in 0..levels {
            let values = &price_level_values[level];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                price_means[level] = mean;
                    price_stds[level] = if variance > epsilon { variance.sqrt() } else { 1.0 };
                }
            }
        }

        // Compute statistics for sizes
        let mut size_means = vec![0.0; levels * 2];
        let mut size_stds = vec![1.0; levels * 2];

        if config.lob_sizes == FeatureNormStrategy::GlobalZScore && !all_sizes.is_empty() {
            // GlobalZScore: ONE mean/std for ALL sizes (TLOB repo style)
            let global_mean: f64 = all_sizes.iter().sum::<f64>() / all_sizes.len() as f64;
            let global_variance: f64 = all_sizes.iter()
                .map(|v| (v - global_mean).powi(2))
                .sum::<f64>() / all_sizes.len() as f64;
            let global_std = if global_variance > epsilon { global_variance.sqrt() } else { 1.0 };
            
            // Fill all size indices with the same global stats
            for i in 0..(levels * 2) {
                size_means[i] = global_mean;
                size_stds[i] = global_std;
            }
            println!("    📊 GlobalZScore sizes: mean={:.6}, std={:.6} (shared across all {} size columns)",
                global_mean, global_std, levels * 2);
        } else {
            // Per-feature stats
        for i in 0..(levels * 2) {
            let values = &size_values[i];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                size_means[i] = mean;
                    size_stds[i] = if variance > epsilon { variance.sqrt() } else { 1.0 };
                }
            }
        }

        // Compute statistics for derived
        let mut derived_means = vec![0.0; 8];
        let mut derived_stds = vec![1.0; 8];
        for i in 0..8 {
            let values = &derived_values[i];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                derived_means[i] = mean;
                derived_stds[i] = if variance > epsilon { variance.sqrt() } else { 1.0 };
            }
        }

        // Compute statistics for MBO
        let mut mbo_means = vec![0.0; 36];
        let mut mbo_stds = vec![1.0; 36];
        for i in 0..36 {
            let values = &mbo_values[i];
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                mbo_means[i] = mean;
                mbo_stds[i] = if variance > epsilon { variance.sqrt() } else { 1.0 };
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
    fn apply_normalization(
        &self,
        value: f64,
        strategy: &FeatureNormStrategy,
        mean: f64,
        std: f64,
        epsilon: f64,
    ) -> f64 {
        match strategy {
            FeatureNormStrategy::None => value,
            FeatureNormStrategy::ZScore | FeatureNormStrategy::GlobalZScore | FeatureNormStrategy::MarketStructure => {
                (value - mean) / (std + epsilon)
            }
            FeatureNormStrategy::PercentageChange => {
                if mean.abs() > epsilon {
                    (value - mean) / mean
                } else {
                    value
                }
            }
            FeatureNormStrategy::MinMax => {
                // FALLBACK: MinMax requires min/max tracking per feature, which is not implemented.
                // Currently falls back to Z-score normalization.
                // For production use, prefer ZScore, GlobalZScore, or MarketStructure strategies.
                // See TODO.md section 3.3 for implementation plan.
                (value - mean) / (std + epsilon)
            }
            FeatureNormStrategy::Bilinear => {
                // FALLBACK: Bilinear requires per-timestep mid_price access, which is not available
                // in the batch normalization context. Currently uses an approximation:
                // (value - mean) / (scale_factor * tick_size)
                // For TLOB training, use NormalizationConfig::raw() to let the model's BiN layer
                // handle normalization internally (which is the intended design).
                // See TODO.md section 3.3 for implementation plan.
                (value - mean) / (self.normalization_config.bilinear_scale_factor * 0.01)
            }
        }
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
