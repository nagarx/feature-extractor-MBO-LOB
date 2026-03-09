//! Aligned Export Module - Exports sequences with properly aligned labels
//!
//! This module fixes the critical data alignment issue in the original export.
//!
//! # Architecture
//!
//! ```text
//! export_aligned/
//!   mod.rs          ← AlignedBatchExporter struct, builder, dispatch, common pipeline
//!   types.rs        ← LabelEncoding, NormalizationStrategy, NormalizationParams, AlignedDayExport
//!   metadata.rs     ← Provenance, normalization, and processing metadata builders
//!   alignment.rs    ← Sequence-label alignment (single + multi-horizon)
//!   validation.rs   ← Label validation, spread verification, class balance
//!   normalization.rs← Feature normalization engine (per-group strategies)
//!   npy_export.rs   ← NPY file writing (sequences, labels, tensor formats)
//!   strategies/     ← Per-strategy label generation (TLOB, opportunity, triple barrier)
//! ```
//!
//! # Guarantees
//! - Every sequence has exactly one label
//! - Label corresponds to the prediction target for that sequence
//! - Exported arrays have matching lengths: len(sequences) == len(labels)

mod alignment;
mod metadata;
mod normalization;
mod npy_export;
mod strategies;
pub mod types;
mod validation;

use metadata::{build_normalization_metadata, build_processing_metadata, build_provenance};
pub use types::{AlignedDayExport, LabelEncoding, NormalizationParams, NormalizationStrategy};

use crate::export::config::NormalizationConfig;
use crate::export::tensor_format::{FeatureMapping, TensorFormat};
use crate::labeling::{LabelConfig, MultiHorizonConfig, OpportunityConfig};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

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
    pub(super) output_dir: PathBuf,
    pub(super) label_config: LabelConfig,
    pub(super) window_size: usize,
    pub(super) stride: usize,
    pub(super) tensor_format: Option<TensorFormat>,
    pub(super) feature_mapping: Option<FeatureMapping>,
    pub(super) multi_horizon_config: Option<MultiHorizonConfig>,
    pub(super) opportunity_configs: Option<Vec<OpportunityConfig>>,
    /// Tuple of (configs, horizons). López de Prado (2018), Chapter 3.
    pub(super) triple_barrier_configs:
        Option<(Vec<crate::labeling::TripleBarrierConfig>, Vec<usize>)>,
    /// Volatility-adaptive scaling: (reference_volatility, floor, cap).
    pub(super) volatility_scaling: Option<(f64, f64, f64)>,
    pub(super) normalization_config: NormalizationConfig,
    pub(super) config_hash: Option<String>,
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
            triple_barrier_configs: None,
            volatility_scaling: None,
            normalization_config: NormalizationConfig::default(), // Raw by default
            config_hash: None,
        }
    }

    /// Set the config hash for provenance tracking (builder pattern).
    pub fn with_config_hash(mut self, hash: String) -> Self {
        self.config_hash = Some(hash);
        self
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
    /// use feature_extractor::export::config::{NormalizationConfig, FeatureNormStrategy};
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

    /// Enable Triple Barrier labeling for trade outcome prediction (builder pattern).
    ///
    /// Triple Barrier labeling determines which of three barriers is hit first:
    /// - **Upper barrier** (profit target): Price rises by `profit_target_pct`
    /// - **Lower barrier** (stop-loss): Price falls by `stop_loss_pct`
    /// - **Vertical barrier** (timeout): Maximum holding period reached
    ///
    /// # Arguments
    ///
    /// * `configs` - Triple Barrier configurations (one per horizon)
    /// * `horizons` - Corresponding horizon values for metadata
    ///
    /// # Label Encoding
    ///
    /// Labels are encoded for PyTorch CrossEntropyLoss:
    /// - 0: StopLoss (hit lower barrier first)
    /// - 1: Timeout (neither barrier hit within max_horizon)
    /// - 2: ProfitTarget (hit upper barrier first)
    ///
    /// # Output
    ///
    /// - Multi-horizon: `{day}_labels.npy` shape (N, num_horizons)
    /// - Single-horizon: `{day}_labels.npy` shape (N,)
    ///
    /// # Research Reference
    ///
    /// López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.
    /// The Triple Barrier Method.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::labeling::TripleBarrierConfig;
    ///
    /// let configs = vec![
    ///     TripleBarrierConfig::new(0.005, 0.003, 50),   // h=50
    ///     TripleBarrierConfig::new(0.005, 0.003, 100),  // h=100
    /// ];
    /// let horizons = vec![50, 100];
    ///
    /// let exporter = AlignedBatchExporter::new("output/", label_config, 100, 10)
    ///     .with_triple_barrier_labels(configs, horizons);
    /// ```
    pub fn with_triple_barrier_labels(
        mut self,
        configs: Vec<crate::labeling::TripleBarrierConfig>,
        horizons: Vec<usize>,
    ) -> Self {
        self.triple_barrier_configs = Some((configs, horizons));
        self
    }

    /// Enable volatility-adaptive barrier scaling (Schema 3.3+).
    ///
    /// When enabled, barriers are scaled per-day based on realized volatility:
    ///   `barrier_day = barrier_base * clamp(vol_day / vol_reference, floor, cap)`
    ///
    /// The `configs` passed to `with_triple_barrier_labels` serve as the base
    /// (unscaled) barriers.
    ///
    /// # Arguments
    ///
    /// * `reference_vol` - Reference volatility (median daily vol from calibration)
    /// * `floor` - Minimum scaling factor (e.g., 0.3)
    /// * `cap` - Maximum scaling factor (e.g., 3.0)
    pub fn with_volatility_scaling(mut self, reference_vol: f64, floor: f64, cap: f64) -> Self {
        self.volatility_scaling = Some((reference_vol, floor, cap));
        self
    }

    /// Check if opportunity labeling is enabled.
    #[inline]
    pub fn is_opportunity_labeling(&self) -> bool {
        self.opportunity_configs.is_some()
    }

    /// Check if Triple Barrier labeling is enabled.
    #[inline]
    pub fn is_triple_barrier_labeling(&self) -> bool {
        self.triple_barrier_configs.is_some()
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
        fs::create_dir_all(&self.output_dir)?;

        println!("  📊 Pipeline Output:");
        println!("    Features extracted: {}", output.features_extracted);
        println!("    Sequences generated: {}", output.sequences.len());
        println!("    Mid-prices collected: {}", output.mid_prices.len());

        // Strategy dispatch: generate labels → LabelingResult → common pipeline.
        // Priority order: Triple Barrier > Opportunity > Multi-horizon TLOB > Single-horizon TLOB
        let labeling = if let Some((tb_configs, horizons)) = &self.triple_barrier_configs {
            self.labeling_triple_barrier(output, tb_configs, horizons)?
        } else if let Some(opp_configs) = &self.opportunity_configs {
            self.labeling_opportunity(output, opp_configs)?
        } else if let Some(multi_config) = &self.multi_horizon_config {
            self.labeling_multi_horizon_tlob(output, multi_config)?
        } else {
            self.labeling_single_horizon_tlob(output)?
        };

        self.export_day_common(day_name, output, labeling)
    }

    /// Common export pipeline consuming a `LabelingResult` from any strategy.
    ///
    /// Steps: align → validate → spreads → normalize → export sequences →
    /// export labels → export normalization → export metadata → export horizons.
    fn export_day_common(
        &self,
        day_name: &str,
        output: &PipelineOutput,
        labeling: types::LabelingResult,
    ) -> Result<AlignedDayExport> {
        let features_extracted = output.features_extracted;
        let sequences_generated = output.sequences.len();

        // Step 1: Align sequences with labels
        println!("  🔗 Aligning sequences with labels...");
        let (aligned_sequences, aligned_label_matrix) = self.align_sequences_with_multi_labels(
            &output.sequences,
            &labeling.label_indices,
            &labeling.label_matrix,
        )?;

        let sequences_dropped = sequences_generated.saturating_sub(aligned_sequences.len());
        println!(
            "    Aligned {} sequences with labels",
            aligned_sequences.len()
        );
        println!("    Dropped {sequences_dropped} sequences (no label)");

        // Step 2: Validate alignment
        self.validate_label_alignment(
            &aligned_sequences,
            &aligned_label_matrix,
            &labeling.encoding,
        )?;

        // Step 3: Verify raw spreads
        self.verify_raw_spreads(&aligned_sequences)?;

        // Step 4: Normalize sequences
        let (aligned_sequences, norm_params) = self.normalize_sequences(&aligned_sequences)?;

        // Step 5: Validate data integrity AFTER normalization, BEFORE write
        let nan_inf_report = self.scan_for_nan_inf(&aligned_sequences)?;

        // Step 6: Compute shape and validate feature count against contract
        let seq_shape = if !aligned_sequences.is_empty() {
            let window = aligned_sequences[0].len();
            let features = aligned_sequences[0][0].len();
            self.validate_feature_count(features)?;
            (window, features)
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No aligned sequences to export",
            )
            .into());
        };

        // Step 7: Export sequences NPY
        let sequences_path = self.output_dir.join(format!("{day_name}_sequences.npy"));
        self.export_sequences_with_format(&aligned_sequences, seq_shape, &sequences_path)?;

        // Step 8: Export labels NPY
        let labels_path = self.output_dir.join(format!("{day_name}_labels.npy"));
        if labeling.is_multi_horizon {
            self.export_multi_horizon_labels(&aligned_label_matrix, &labels_path)?;
        } else {
            let single_labels: Vec<i8> = aligned_label_matrix.iter().map(|v| v[0]).collect();
            self.export_labels(&single_labels, &labels_path)?;
        }

        // Step 9: Export normalization params
        norm_params.save_json(
            self.output_dir
                .join(format!("{day_name}_normalization.json")),
        )?;

        // Step 10: Build and export metadata
        let tensor_format_str = self.tensor_format.as_ref().map(|f| format!("{:?}", f));
        let config_hash_str = self.config_hash.as_deref().unwrap_or("unknown");

        let metadata = serde_json::json!({
            "day": day_name,
            "n_sequences": aligned_sequences.len(),
            "window_size": seq_shape.0,
            "n_features": seq_shape.1,
            "schema_version": crate::contract::SCHEMA_VERSION.to_string(),
            "contract_version": crate::contract::SCHEMA_VERSION.to_string(),
            "label_strategy": labeling.strategy_name,
            "tensor_format": tensor_format_str,
            "labeling": labeling.strategy_metadata,
            "label_distribution": labeling.distribution,
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
            "normalization": build_normalization_metadata(&norm_params, day_name),
            "provenance": build_provenance(config_hash_str),
            "validation": {
                "sequences_labels_match": aligned_sequences.len() == aligned_label_matrix.len(),
                "label_range_valid": true,
                "no_nan_inf": nan_inf_report.is_clean,
                "values_scanned": nan_inf_report.total_values_scanned,
            },
            "processing": build_processing_metadata(
                output.messages_processed,
                features_extracted,
                sequences_generated,
                aligned_sequences.len(),
                sequences_dropped,
            ),
        });

        let metadata_path = self.output_dir.join(format!("{day_name}_metadata.json"));
        let mut file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(&mut file, &metadata)
            .map_err(|e| std::io::Error::other(format!("Failed to write metadata: {e}")))?;

        // Step 11: Export horizons config if present
        if let Some(horizons_config) = &labeling.horizons_config {
            let horizons_path = self.output_dir.join(format!("{day_name}_horizons.json"));
            let horizons_file = File::create(&horizons_path)?;
            serde_json::to_writer_pretty(horizons_file, horizons_config).map_err(|e| {
                std::io::Error::other(format!("Failed to write horizons config: {e}"))
            })?;
            println!("  💾 Exported horizons config: {}", horizons_path.display());
        }

        println!(
            "  ✅ Export complete: {} sequences × {} strategy",
            aligned_sequences.len(),
            labeling.strategy_name
        );

        Ok(AlignedDayExport {
            day: day_name.to_string(),
            n_sequences: aligned_sequences.len(),
            seq_shape,
            label_distribution: labeling.distribution,
            messages_processed: output.messages_processed,
            export_path: self.output_dir.clone(),
            features_extracted,
            buffer_size: 50_000,
            sequences_generated,
            sequences_dropped,
        })
    }
}
