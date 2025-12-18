//! Unified Pipeline for LOB Feature Extraction
//!
//! This module provides a simple, composable pipeline that connects all components:
//! - LOB Reconstruction (via `mbo-lob-reconstructor`)
//! - Feature Extraction (raw LOB, derived, MBO features)
//! - Sampling (volume-based, event-based, adaptive)
//! - Sequence Building (streaming mode)
//! - Label alignment support
//!
//! # Architecture
//!
//! ```text
//! MBO Messages → LobReconstructor → LobState → FeatureExtractor → Arc<Features>
//!                     ↑                             ↑                    ↓
//!            (reused buffer)              (reused buffer)      SequenceBuilder.push_arc()
//!                                                                        ↓
//!                                                        try_build_sequence() [streaming]
//!                                                                        ↓
//!                                                        Accumulated Sequences
//! ```
//!
//! # Zero-Allocation Hot Path
//!
//! The pipeline uses **zero-allocation optimizations** in the critical path:
//! 1. `LobState` - Stack-allocated, reused across all messages
//! 2. `feature_buffer` - Heap-allocated once, reused for extraction
//! 3. `Arc` wrapping - Features wrapped once, shared via `push_arc()`
//!
//! Memory allocation per sample: 1 Vec + Arc wrap (constant)
//! Multi-scale overhead: 16 bytes (Arc clones) vs 1,344 bytes (Vec clones)
//!
//! # Streaming Sequence Generation (Critical)
//!
//! The pipeline uses **streaming mode** for sequence generation:
//! - After each feature is pushed, `try_build_sequence()` is called
//! - Complete sequences are immediately accumulated
//! - This prevents data loss from buffer eviction
//!
//! Without streaming mode, ~98% of sequences would be lost due to the
//! bounded buffer (default 1000 snapshots).
//!
//! # Philosophy
//! - **Simple**: Easy to understand and use
//! - **Modular**: Components are independent and composable
//! - **Fast**: Zero-allocation hot path, Arc-based sharing
//! - **Correct**: No silent data loss
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::prelude::*;
//!
//! // Using PipelineBuilder (recommended)
//! let mut pipeline = PipelineBuilder::new()
//!     .with_derived_features()
//!     .event_sampling(1000)
//!     .build()?;
//!
//! let output = pipeline.process("data.dbn.zst")?;
//!
//! // Multi-day processing
//! for day in days {
//!     pipeline.reset();  // Critical: clear state between days
//!     let output = pipeline.process(&path)?;
//! }
//! ```
//!
//! # Output Structure
//!
//! | Field | Type | Description |
//! |-------|------|-------------|
//! | `sequences` | `Vec<Sequence>` | Generated sequences |
//! | `mid_prices` | `Vec<f64>` | Mid-prices for labeling |
//! | `messages_processed` | `usize` | Total MBO messages |
//! | `features_extracted` | `usize` | Sampled LOB snapshots |
//! | `sequences_generated` | `usize` | Complete sequences |

use crate::{
    config::{PipelineConfig, SamplingStrategy},
    export::tensor_format::{TensorFormat, TensorFormatter, TensorOutput},
    features::mbo_features::MboEvent,
    labeling::{MultiHorizonConfig, MultiHorizonLabelGenerator, MultiHorizonLabels},
    preprocessing::{AdaptiveVolumeThreshold, EventBasedSampler, VolumeBasedSampler},
    sequence_builder::{FeatureVec, MultiScaleSequence, MultiScaleWindow},
    FeatureExtractor, Sequence, SequenceBuilder,
};
use mbo_lob_reconstructor::{
    DbnLoader, LobReconstructor, LobState, MarketDataSource, MboMessage, Result,
};
use std::path::Path;
use std::sync::Arc;

/// Output from pipeline processing
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Generated sequences [N_sequences, window_size, features]
    pub sequences: Vec<Sequence>,

    /// Mid-prices extracted (for labeling)
    pub mid_prices: Vec<f64>,

    /// Total messages processed
    pub messages_processed: usize,

    /// Total features extracted
    pub features_extracted: usize,

    /// Total sequences generated
    pub sequences_generated: usize,

    /// Stride used for sequence generation (from SequenceConfig)
    ///
    /// This is important for downstream consumers to know how sequences were built.
    /// A stride of 1 means maximum overlap, stride = window_size means no overlap.
    pub stride: usize,

    /// Window size used for sequence generation (from SequenceConfig)
    pub window_size: usize,

    /// Phase 1: Multi-scale sequences (if enabled)
    pub multiscale_sequences: Option<MultiScaleSequence>,

    /// Phase 1: Adaptive sampling statistics (if enabled)
    pub adaptive_stats: Option<AdaptiveSamplingStats>,
}

/// Phase 1: Statistics from adaptive sampling
#[derive(Debug, Clone)]
pub struct AdaptiveSamplingStats {
    /// Current volatility estimate
    pub current_volatility: Option<f64>,

    /// Baseline volatility (after calibration)
    pub baseline_volatility: Option<f64>,

    /// Current threshold multiplier
    pub current_multiplier: Option<f64>,

    /// Current adaptive threshold
    pub current_threshold: Option<u64>,

    /// Whether calibration is complete
    pub is_calibrated: bool,
}

impl PipelineOutput {
    /// Get features as flat 2D array for export
    /// Returns [N_samples, N_features] where N_samples = sum of all sequence lengths
    ///
    /// Note: This creates deep copies of the feature vectors. For performance-critical
    /// code, consider working with the `Arc<Vec<f64>>` references in seq.features directly.
    pub fn to_flat_features(&self) -> Vec<Vec<f64>> {
        let mut flat = Vec::new();
        for seq in &self.sequences {
            // Each sequence has multiple feature vectors (Arc<Vec<f64>>)
            // We need to deep-copy here since the return type is Vec<Vec<f64>>
            for feature_vec in &seq.features {
                flat.push(feature_vec.to_vec());
            }
        }
        flat
    }

    /// Get mid-prices as flat array (for labeling later)
    pub fn get_mid_prices(&self) -> &[f64] {
        &self.mid_prices
    }

    // ========================================================================
    // Tensor Formatting API
    // ========================================================================

    /// Format all sequences using a TensorFormatter.
    ///
    /// This method converts sequences to model-specific tensor shapes for
    /// different deep learning architectures.
    ///
    /// # Arguments
    ///
    /// * `formatter` - TensorFormatter configured for the target model
    ///
    /// # Returns
    ///
    /// * `Ok(TensorOutput)` - Formatted tensor (Array3 for single sequences, Array4 for batch)
    /// * `Err(...)` - Formatting failed
    ///
    /// # Supported Formats
    ///
    /// | Format | Shape | Models |
    /// |--------|-------|--------|
    /// | Flat | (N, T, F) | TLOB, LSTM |
    /// | DeepLOB | (N, T, 4, L) | DeepLOB |
    /// | HLOB | (N, T, L, 4) | HLOB |
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::prelude::*;
    /// use feature_extractor::export::tensor_format::TensorFormatter;
    ///
    /// let output = pipeline.process("data.dbn.zst")?;
    ///
    /// // Format for DeepLOB (N, T, 4, L)
    /// let formatter = TensorFormatter::deeplob(10);
    /// let tensor = output.format_sequences(&formatter)?;
    /// ```
    pub fn format_sequences(&self, formatter: &TensorFormatter) -> Result<TensorOutput> {
        if self.sequences.is_empty() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "No sequences to format",
            ));
        }

        // Convert sequences to Vec<Vec<Vec<f64>>> format for batch formatting
        let batch: Vec<Vec<Vec<f64>>> = self
            .sequences
            .iter()
            .map(|seq| seq.features.iter().map(|f| f.to_vec()).collect())
            .collect();

        formatter.format_batch(&batch)
    }

    /// Format sequences for a specific tensor format.
    ///
    /// Convenience method that creates a TensorFormatter internally.
    ///
    /// # Arguments
    ///
    /// * `format` - Target tensor format
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::export::tensor_format::TensorFormat;
    ///
    /// let output = pipeline.process("data.dbn.zst")?;
    ///
    /// // Format for DeepLOB
    /// let tensor = output.format_as(TensorFormat::deeplob())?;
    /// ```
    pub fn format_as(&self, format: TensorFormat) -> Result<TensorOutput> {
        let levels = match &format {
            TensorFormat::DeepLOB { levels } => *levels,
            TensorFormat::HLOB { levels } => *levels,
            _ => 10, // Default for non-level formats
        };

        let formatter = TensorFormatter::new(
            format,
            crate::export::tensor_format::FeatureMapping::standard_lob(levels),
        );

        self.format_sequences(&formatter)
    }

    // ========================================================================
    // Multi-Horizon Labeling API
    // ========================================================================

    /// Generate multi-horizon labels from mid-prices.
    ///
    /// This method creates labels for multiple prediction horizons simultaneously,
    /// which is required for benchmark reproduction (FI-2010, DeepLOB, TLOB).
    ///
    /// # Arguments
    ///
    /// * `config` - Multi-horizon configuration specifying horizons and thresholds
    ///
    /// # Returns
    ///
    /// * `Ok(MultiHorizonLabels)` - Labels for all configured horizons
    /// * `Err(...)` - Label generation failed (insufficient data, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::prelude::*;
    ///
    /// let output = pipeline.process("data.dbn.zst")?;
    ///
    /// // Generate labels for FI-2010 horizons: [10, 20, 30, 50, 100]
    /// let labels = output.generate_multi_horizon_labels(MultiHorizonConfig::fi2010())?;
    ///
    /// // Access labels for specific horizon
    /// let h50_labels = labels.labels_for_horizon(50);
    /// ```
    pub fn generate_multi_horizon_labels(
        &self,
        config: MultiHorizonConfig,
    ) -> Result<MultiHorizonLabels> {
        if self.mid_prices.is_empty() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "No mid-prices available for label generation",
            ));
        }

        let mut generator = MultiHorizonLabelGenerator::new(config);
        generator.add_prices(&self.mid_prices);

        generator.generate_labels()
    }

    /// Get the number of features per timestep.
    ///
    /// Returns `None` if no sequences are available.
    pub fn feature_count(&self) -> Option<usize> {
        self.sequences
            .first()
            .and_then(|seq| seq.features.first().map(|f| f.len()))
    }

    /// Get the number of LOB levels (computed from feature count).
    ///
    /// Assumes standard LOB feature layout: 4 features per level.
    /// Returns `None` if feature count is not divisible by 4 or no sequences available.
    pub fn lob_levels(&self) -> Option<usize> {
        self.feature_count().and_then(|count| {
            if count % 4 == 0 && count >= 4 {
                Some(count / 4)
            } else {
                // May have derived features, try to infer from raw LOB portion
                // Standard: levels * 4 raw features, potentially + 8 derived + 36 MBO
                // For now, return None if not a clean multiple
                None
            }
        })
    }
}

/// Main Pipeline - connects all components
pub struct Pipeline {
    config: PipelineConfig,
    feature_extractor: FeatureExtractor,
    sequence_builder: SequenceBuilder,

    // Phase 1: Adaptive sampling (opt-in)
    // Note: AdaptiveVolumeThreshold has its own internal VolatilityEstimator
    adaptive_threshold: Option<AdaptiveVolumeThreshold>,

    // Phase 1: Multi-scale windowing (opt-in)
    multiscale_window: Option<MultiScaleWindow>,
}

impl Pipeline {
    /// Create pipeline from configuration
    pub fn from_config(config: PipelineConfig) -> Result<Self> {
        // Validate config
        config.validate()?;

        // Create core components
        let feature_extractor = FeatureExtractor::with_config(config.features.clone());
        let sequence_builder = SequenceBuilder::with_config(config.sequence.clone());

        // Phase 1: Initialize adaptive sampling if configured
        let adaptive_threshold = if let Some(ref sampling) = config.sampling {
            if let Some(ref adaptive_config) = sampling.adaptive {
                if adaptive_config.enabled {
                    // Note: AdaptiveVolumeThreshold creates its own internal VolatilityEstimator

                    // Create adaptive threshold with internal volatility estimator
                    let mut adaptive = AdaptiveVolumeThreshold::new(
                        adaptive_config.base_threshold,
                        adaptive_config.volatility_window,
                    );

                    // Configure parameters
                    adaptive.set_min_multiplier(adaptive_config.min_multiplier);
                    adaptive.set_max_multiplier(adaptive_config.max_multiplier);
                    adaptive.set_calibration_size(adaptive_config.calibration_size);

                    Some(adaptive)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Phase 1: Initialize multi-scale window if configured
        let multiscale_window = if let Some(ref sampling) = config.sampling {
            if let Some(ref ms_config) = sampling.multiscale {
                if ms_config.enabled {
                    use crate::sequence_builder::{MultiScaleConfig as MSConfig, ScaleConfig};

                    let ms_config_internal = MSConfig::new(
                        ScaleConfig::new(ms_config.fast_window, 1, 1),
                        ScaleConfig::new(ms_config.medium_window, ms_config.medium_decimation, 1),
                        ScaleConfig::new(ms_config.slow_window, ms_config.slow_decimation, 1),
                    );

                    Some(MultiScaleWindow::new(
                        ms_config_internal,
                        config.sequence.feature_count,
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            feature_extractor,
            sequence_builder,
            adaptive_threshold,
            multiscale_window,
        })
    }

    /// Process a single DBN file through the complete pipeline.
    ///
    /// This is a convenience method that loads a DBN file and processes it
    /// through the pipeline. For more flexibility, use [`process_source()`] or
    /// [`process_messages()`].
    ///
    /// # Feature Extraction
    ///
    /// This method extracts ALL configured features:
    /// - Raw LOB features (40 for 10 levels)
    /// - Derived features (8, if enabled)
    /// - MBO features (36, if enabled) - requires processing MBO events
    ///
    /// The feature count must match `sequence.feature_count` in the config.
    ///
    /// # Performance
    ///
    /// Uses **zero-allocation hot path** with multiple optimizations:
    /// 1. `process_message_into()` - Reuses single LobState across all messages
    /// 2. `extract_into()` - Reuses feature buffer across all samples
    /// 3. `push_arc()` - Zero-copy feature sharing across sequence builders
    ///
    /// Memory allocation pattern:
    /// - Per message: 0 allocations (LobState reused)
    /// - Per sample: 1 allocation (Vec for features) + Arc wrap
    /// - Multi-scale: 2 Arc clones (16 bytes) instead of 2 Vec clones (1,344 bytes)
    ///
    /// # Implementation Note
    ///
    /// This method delegates to [`process_messages()`], ensuring a single code
    /// path for all processing methods. For hot store support, use
    /// [`process_source()`] with [`DbnSource::with_hot_store()`].
    pub fn process<P: AsRef<Path>>(&mut self, input_path: P) -> Result<PipelineOutput> {
        let loader = DbnLoader::new(input_path)?;
        self.process_messages(loader.iter_messages()?)
    }

    /// Process a market data source through the complete pipeline.
    ///
    /// This is a generic version of `process()` that accepts any `MarketDataSource`
    /// implementation, enabling flexible data ingestion from different providers.
    ///
    /// # Type Parameters
    ///
    /// * `S` - Any type implementing `MarketDataSource`
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::prelude::*;
    /// use mbo_lob_reconstructor::{DbnSource, VecSource, MarketDataSource};
    ///
    /// let mut pipeline = PipelineBuilder::new().build()?;
    ///
    /// // With Databento DBN files
    /// let source = DbnSource::new("data/NVDA.mbo.dbn.zst")?;
    /// let output = pipeline.process_source(source)?;
    ///
    /// // With hot store (faster)
    /// let hot_store = HotStoreManager::for_dbn("data/hot/");
    /// let source = DbnSource::with_hot_store("data/raw/NVDA.mbo.dbn.zst", &hot_store)?;
    /// let output = pipeline.process_source(source)?;
    ///
    /// // With mock data (for testing)
    /// let source = VecSource::new(mock_messages);
    /// let output = pipeline.process_source(source)?;
    /// ```
    ///
    /// # Performance
    ///
    /// Same zero-allocation optimizations as `process()`:
    /// - `process_message_into()` - Reuses single LobState
    /// - `extract_into()` - Reuses feature buffer
    /// - `push_arc()` - Zero-copy feature sharing
    pub fn process_source<S: MarketDataSource>(&mut self, source: S) -> Result<PipelineOutput> {
        let messages = source.messages()?;
        self.process_messages(messages)
    }

    /// Process an iterator of MBO messages through the pipeline.
    ///
    /// This is the core processing method used by both `process()` and `process_source()`.
    ///
    /// # Arguments
    ///
    /// * `messages` - Iterator yielding `MboMessage`s
    ///
    /// # Returns
    ///
    /// * `Ok(PipelineOutput)` - Processing results
    /// * `Err(...)` - Processing failed
    pub fn process_messages<I>(&mut self, messages: I) -> Result<PipelineOutput>
    where
        I: Iterator<Item = MboMessage>,
    {
        let mut lob = LobReconstructor::new(self.config.features.lob_levels);
        let mut sampler = self.create_sampler()?;

        // ========================================================================
        // PERF: Pre-allocate reusable buffers (zero-allocation hot path)
        // ========================================================================

        let mut lob_state = LobState::new(self.config.features.lob_levels);
        let feature_count = self.feature_extractor.feature_count();
        let mut feature_buffer: Vec<f64> = Vec::with_capacity(feature_count);

        // Statistics
        let mut messages_processed = 0;
        let mut features_extracted = 0;
        let mut mid_prices = Vec::new();
        let mut accumulated_sequences: Vec<Sequence> = Vec::new();

        // Process messages
        for msg in messages {
            messages_processed += 1;

            // Skip invalid messages (system messages, metadata, etc.)
            if msg.order_id == 0 || msg.size == 0 || msg.price <= 0 {
                continue;
            }

            // ✅ PERF: Update LOB and fill pre-allocated state (zero-allocation)
            lob.process_message_into(&msg, &mut lob_state)?;

            // Process MBO event for MBO feature aggregation
            if self.feature_extractor.has_mbo() {
                let mbo_event = MboEvent::from_mbo_message(&msg);
                self.feature_extractor.process_mbo_event(mbo_event);
            }

            let ts = msg.timestamp.unwrap_or(0) as u64;

            // Determine threshold (adaptive or fixed)
            let current_threshold = if let (Some(ref mut adaptive), Some(mid_price)) =
                (&mut self.adaptive_threshold, lob_state.mid_price())
            {
                adaptive.update(mid_price)
            } else {
                self.config
                    .sampling
                    .as_ref()
                    .and_then(|s| s.volume_threshold)
                    .unwrap_or(1000)
            };

            // Check if we should sample
            let should_sample = match &mut sampler {
                SamplerType::Volume(s) => {
                    s.set_threshold(current_threshold);
                    s.should_sample(msg.size, ts)
                }
                SamplerType::Event(s) => s.should_sample(),
            };

            if should_sample {
                // Skip if LOB not ready (no bid/ask yet)
                if lob_state.mid_price().is_none() {
                    continue;
                }

                // ========================================================================
                // PERF: Zero-allocation feature extraction with Arc sharing
                // ========================================================================

                self.feature_extractor
                    .extract_into(&lob_state, &mut feature_buffer)?;
                features_extracted += 1;

                let features: FeatureVec = Arc::new(std::mem::take(&mut feature_buffer));
                feature_buffer = Vec::with_capacity(feature_count);

                if let Some(ref mut ms_window) = self.multiscale_window {
                    ms_window.push_arc(ts, features);
                } else if let Err(e) = self.sequence_builder.push_arc(ts, features) {
                    log::error!("Sequence builder push failed: {}", e);
                } else if let Some(seq) = self.sequence_builder.try_build_sequence() {
                    accumulated_sequences.push(seq);
                }

                if let Some(mid) = lob_state.mid_price() {
                    mid_prices.push(mid);
                }
            }
        }

        // Finalize sequences
        let (sequences, multiscale_sequences) =
            if let Some(ref mut ms_window) = self.multiscale_window {
                let ms_seq = ms_window.try_build_all();
                (Vec::new(), ms_seq)
            } else {
                (accumulated_sequences, None)
            };

        let sequences_generated = sequences.len()
            + multiscale_sequences
                .as_ref()
                .map(|ms| {
                    let (f, m, s) = ms.sequence_counts();
                    f + m + s
                })
                .unwrap_or(0);

        let adaptive_stats =
            self.adaptive_threshold
                .as_ref()
                .map(|adaptive| AdaptiveSamplingStats {
                    current_volatility: adaptive.current_volatility(),
                    baseline_volatility: adaptive.baseline_volatility(),
                    current_multiplier: adaptive.current_multiplier(),
                    current_threshold: Some(adaptive.current_threshold()),
                    is_calibrated: adaptive.is_calibrated(),
                });

        Ok(PipelineOutput {
            sequences,
            mid_prices,
            messages_processed,
            features_extracted,
            sequences_generated,
            stride: self.config.sequence.stride,
            window_size: self.config.sequence.window_size,
            multiscale_sequences,
            adaptive_stats,
        })
    }

    /// Process and return flat features (for direct export)
    pub fn process_flat<P: AsRef<Path>>(
        &mut self,
        input_path: P,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        let output = self.process(input_path)?;
        let features = output.to_flat_features();
        let mid_prices = output.mid_prices;
        Ok((features, mid_prices))
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Reset pipeline state (for processing multiple files/days)
    ///
    /// # What gets reset:
    /// - Sequence builder buffer
    /// - Feature extractor state (MBO aggregator, etc.)
    /// - Adaptive sampling calibration
    /// - Multi-scale window buffers
    ///
    /// # When to call:
    /// - Before processing a new trading day
    /// - Before processing a new data file
    /// - When switching between instruments
    ///
    /// # Important:
    /// This prevents state leakage between days, which could corrupt
    /// derivative calculations and MBO feature aggregations.
    pub fn reset(&mut self) {
        self.sequence_builder.reset();

        // ✅ FIX: Reset feature extractor state
        // This is critical to prevent state leakage between days
        self.feature_extractor.reset();

        // Phase 1: Reset adaptive sampling (uses internal reset method)
        if let Some(ref mut adaptive) = self.adaptive_threshold {
            adaptive.reset();
        }

        // Phase 1: Reset multi-scale window
        if let Some(ref mut ms_window) = self.multiscale_window {
            ms_window.reset();
        }
    }

    // Helper: Create sampler from config
    fn create_sampler(&self) -> Result<SamplerType> {
        // Use default sampling config if none provided
        let default_config = crate::config::SamplingConfig::default();
        let sampling_config = self.config.sampling.as_ref().unwrap_or(&default_config);

        match sampling_config.strategy {
            SamplingStrategy::VolumeBased => {
                let threshold = sampling_config.volume_threshold.unwrap_or(1000);
                let min_interval = sampling_config.min_time_interval_ns.unwrap_or(1_000_000);
                Ok(SamplerType::Volume(VolumeBasedSampler::new(
                    threshold,
                    min_interval,
                )))
            }
            SamplingStrategy::EventBased => {
                let count = sampling_config.event_count.unwrap_or(100) as u64;
                Ok(SamplerType::Event(EventBasedSampler::new(count)))
            }
            SamplingStrategy::TimeBased => {
                // TimeBased sampling is not yet implemented
                // Return an explicit error instead of silently falling back
                Err(mbo_lob_reconstructor::TlobError::generic(
                    "TimeBased sampling strategy is not yet implemented. \
                     Please use VolumeBased or EventBased sampling instead.",
                ))
            }
            SamplingStrategy::MultiScale => {
                // MultiScale uses the multi-scale window, not a traditional sampler
                // For the sampler itself, we use volume-based as the base
                let threshold = sampling_config.volume_threshold.unwrap_or(1000);
                let min_interval = sampling_config.min_time_interval_ns.unwrap_or(1_000_000);
                Ok(SamplerType::Volume(VolumeBasedSampler::new(
                    threshold,
                    min_interval,
                )))
            }
        }
    }
}

// Internal sampler enum (simple, no trait needed)
enum SamplerType {
    Volume(VolumeBasedSampler),
    Event(EventBasedSampler),
}
