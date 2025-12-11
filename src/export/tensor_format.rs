//! Tensor Formatting for Model-Specific Input Shapes
//!
//! Different deep learning models expect LOB data in different tensor shapes.
//! This module provides efficient tensor reshaping to support:
//!
//! # Supported Formats
//!
//! | Format | Shape | Models | Description |
//! |--------|-------|--------|-------------|
//! | Flat | (T, F) | TLOB, LSTM | Standard flat features |
//! | DeepLOB | (T, 4, L) | DeepLOB, CNN-LSTM | Channels: [ask_p, ask_v, bid_p, bid_v] |
//! | HLOB | (T, L, 4) | HLOB | Level-first ordering |
//! | Image | (T, C, H, W) | CNN | Image-like representation |
//!
//! # Research Reference
//!
//! - **DeepLOB**: "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
//!   Input: X ∈ R^{100×40}, reshaped to (100, 4, 10) for CNN layers
//!
//! - **HLOB**: "HLOB – Information Persistence and Structure in Limit Order Books"
//!   Uses level-first ordering for topological features
//!
//! - **LiT**: "Limit Order Book Transformer"
//!   Uses structured patches on (T, F) tensors
//!
//! # Performance
//!
//! - Uses ndarray for efficient memory layout
//! - Zero-copy when format matches source
//! - Single-pass reshaping for format conversion
//! - Contiguous memory for SIMD-friendly export
//!
//! # Example
//!
//! ```
//! use feature_extractor::export::tensor_format::{
//!     TensorFormat, TensorFormatter, FeatureMapping,
//! };
//!
//! // Configure for DeepLOB: 10 levels, 4 channels
//! let mapping = FeatureMapping::standard_lob(10);
//! let formatter = TensorFormatter::new(TensorFormat::DeepLOB { levels: 10 }, mapping);
//!
//! // Format a sequence of flat features
//! let flat_features: Vec<Vec<f64>> = vec![vec![0.0; 40]; 100]; // 100 timesteps, 40 features
//! let tensor = formatter.format_sequence(&flat_features).unwrap();
//!
//! // tensor is now (100, 4, 10) for DeepLOB
//! ```

use mbo_lob_reconstructor::{Result, TlobError};
use ndarray::{Array2, Array3, Array4};
use serde::{Deserialize, Serialize};

// ============================================================================
// Tensor Format Specification
// ============================================================================

/// Target tensor format for model input.
///
/// Each format specifies the expected shape and semantics of the output tensor.
///
/// # Format Details
///
/// ## Flat (T, F)
/// - Standard format for TLOB, LSTM, MLP models
/// - T = timesteps (sequence length)
/// - F = total features per timestep
/// - No reshaping required (most efficient)
///
/// ## DeepLOB (T, 4, L)
/// - DeepLOB paper format with explicit channels
/// - T = timesteps
/// - 4 = channels: [ask_price, ask_volume, bid_price, bid_volume]
/// - L = number of price levels (typically 10)
/// - Requires feature reordering
///
/// ## HLOB (T, L, 4)
/// - Level-first ordering for topological features
/// - T = timesteps
/// - L = number of price levels
/// - 4 = features per level
/// - Alternative to DeepLOB for level-aware models
///
/// ## Image (T, C, H, W)
/// - Image-like format for CNN models
/// - T = timesteps (batch-like dimension)
/// - C = channels
/// - H = height
/// - W = width
/// - Generic format for image-processing architectures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorFormat {
    /// Flat features: (T, F)
    ///
    /// No reshaping - returns as-is. Most efficient format.
    Flat,

    /// DeepLOB format: (T, 4, L)
    ///
    /// Channels are ordered: [ask_price, ask_vol, bid_price, bid_vol]
    /// Each channel contains L levels.
    DeepLOB {
        /// Number of price levels (typically 10)
        levels: usize,
    },

    /// HLOB format: (T, L, 4)
    ///
    /// Level-first ordering with 4 features per level.
    /// Features per level: [ask_price, ask_vol, bid_price, bid_vol]
    HLOB {
        /// Number of price levels (typically 10)
        levels: usize,
    },

    /// Image format: (T, C, H, W)
    ///
    /// For CNN models treating LOB as a 2D image.
    Image {
        /// Number of channels
        channels: usize,
        /// Image height
        height: usize,
        /// Image width
        width: usize,
    },
}

impl TensorFormat {
    /// Get the expected output shape for a given sequence length.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Number of timesteps (T)
    /// * `n_features` - Number of features per timestep (for Flat format validation)
    ///
    /// # Returns
    ///
    /// Output shape as a vector of dimensions.
    pub fn output_shape(&self, seq_len: usize, n_features: usize) -> Vec<usize> {
        match self {
            TensorFormat::Flat => vec![seq_len, n_features],
            TensorFormat::DeepLOB { levels } => vec![seq_len, 4, *levels],
            TensorFormat::HLOB { levels } => vec![seq_len, *levels, 4],
            TensorFormat::Image {
                channels,
                height,
                width,
            } => vec![seq_len, *channels, *height, *width],
        }
    }

    /// Get the number of dimensions in the output tensor.
    pub fn ndim(&self) -> usize {
        match self {
            TensorFormat::Flat => 2,
            TensorFormat::DeepLOB { .. } | TensorFormat::HLOB { .. } => 3,
            TensorFormat::Image { .. } => 4,
        }
    }

    /// Validate that input features can be formatted to this format.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features per timestep
    ///
    /// # Returns
    ///
    /// Ok(()) if valid, Err with explanation otherwise.
    pub fn validate_features(&self, n_features: usize) -> Result<()> {
        match self {
            TensorFormat::Flat => Ok(()),

            TensorFormat::DeepLOB { levels } => {
                let required = 4 * levels;
                if n_features < required {
                    return Err(TlobError::generic(format!(
                        "DeepLOB format requires at least {} features ({} levels × 4), got {}",
                        required, levels, n_features
                    )));
                }
                Ok(())
            }

            TensorFormat::HLOB { levels } => {
                let required = 4 * levels;
                if n_features < required {
                    return Err(TlobError::generic(format!(
                        "HLOB format requires at least {} features ({} levels × 4), got {}",
                        required, levels, n_features
                    )));
                }
                Ok(())
            }

            TensorFormat::Image {
                channels,
                height,
                width,
            } => {
                let required = channels * height * width;
                if n_features < required {
                    return Err(TlobError::generic(format!(
                        "Image format requires at least {} features ({}×{}×{}), got {}",
                        required, channels, height, width, n_features
                    )));
                }
                Ok(())
            }
        }
    }

    /// Check if this format requires feature reordering.
    pub fn requires_reordering(&self) -> bool {
        !matches!(self, TensorFormat::Flat)
    }

    /// Create DeepLOB format with standard 10 levels.
    pub fn deeplob() -> Self {
        TensorFormat::DeepLOB { levels: 10 }
    }

    /// Create HLOB format with standard 10 levels.
    pub fn hlob() -> Self {
        TensorFormat::HLOB { levels: 10 }
    }
}

impl Default for TensorFormat {
    fn default() -> Self {
        TensorFormat::Flat
    }
}

// ============================================================================
// Feature Mapping
// ============================================================================

/// Feature index mapping for tensor reshaping.
///
/// Specifies how flat feature indices map to structured tensor positions.
/// This is crucial for correctly reshaping (T, F) data to (T, C, L) formats.
///
/// # Standard LOB Feature Order
///
/// The standard feature order for 10-level LOB is:
/// ```text
/// [0-9]:   ask_price_1 to ask_price_10
/// [10-19]: ask_size_1 to ask_size_10  
/// [20-29]: bid_price_1 to bid_price_10
/// [30-39]: bid_size_1 to bid_size_10
/// ```
///
/// # Example
///
/// ```
/// use feature_extractor::export::tensor_format::FeatureMapping;
///
/// // Standard 10-level LOB mapping
/// let mapping = FeatureMapping::standard_lob(10);
///
/// // Get index for ask price at level 0
/// let idx = mapping.ask_price_index(0);
/// assert_eq!(idx, 0);
///
/// // Get index for bid size at level 5
/// let idx = mapping.bid_size_index(5);
/// assert_eq!(idx, 35);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMapping {
    /// Number of price levels
    pub levels: usize,

    /// Starting index for ask prices [0, 1, 2, ..., L-1]
    pub ask_price_offset: usize,

    /// Starting index for ask sizes [L, L+1, ..., 2L-1]
    pub ask_size_offset: usize,

    /// Starting index for bid prices [2L, 2L+1, ..., 3L-1]
    pub bid_price_offset: usize,

    /// Starting index for bid sizes [3L, 3L+1, ..., 4L-1]
    pub bid_size_offset: usize,

    /// Total features used for LOB (4 × levels)
    pub lob_features: usize,
}

impl FeatureMapping {
    /// Create standard LOB feature mapping.
    ///
    /// Assumes the standard feature order:
    /// [ask_prices..., ask_sizes..., bid_prices..., bid_sizes...]
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of price levels (typically 10)
    pub fn standard_lob(levels: usize) -> Self {
        Self {
            levels,
            ask_price_offset: 0,
            ask_size_offset: levels,
            bid_price_offset: 2 * levels,
            bid_size_offset: 3 * levels,
            lob_features: 4 * levels,
        }
    }

    /// Create custom feature mapping.
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of price levels
    /// * `ask_price_offset` - Starting index for ask prices
    /// * `ask_size_offset` - Starting index for ask sizes
    /// * `bid_price_offset` - Starting index for bid prices
    /// * `bid_size_offset` - Starting index for bid sizes
    pub fn custom(
        levels: usize,
        ask_price_offset: usize,
        ask_size_offset: usize,
        bid_price_offset: usize,
        bid_size_offset: usize,
    ) -> Self {
        Self {
            levels,
            ask_price_offset,
            ask_size_offset,
            bid_price_offset,
            bid_size_offset,
            lob_features: 4 * levels,
        }
    }

    /// Get feature index for ask price at given level.
    #[inline]
    pub fn ask_price_index(&self, level: usize) -> usize {
        self.ask_price_offset + level
    }

    /// Get feature index for ask size at given level.
    #[inline]
    pub fn ask_size_index(&self, level: usize) -> usize {
        self.ask_size_offset + level
    }

    /// Get feature index for bid price at given level.
    #[inline]
    pub fn bid_price_index(&self, level: usize) -> usize {
        self.bid_price_offset + level
    }

    /// Get feature index for bid size at given level.
    #[inline]
    pub fn bid_size_index(&self, level: usize) -> usize {
        self.bid_size_offset + level
    }

    /// Validate that the mapping is consistent with a given feature count.
    pub fn validate(&self, n_features: usize) -> Result<()> {
        let max_idx = [
            self.ask_price_offset + self.levels - 1,
            self.ask_size_offset + self.levels - 1,
            self.bid_price_offset + self.levels - 1,
            self.bid_size_offset + self.levels - 1,
        ]
        .into_iter()
        .max()
        .unwrap();

        if max_idx >= n_features {
            return Err(TlobError::generic(format!(
                "Feature mapping requires at least {} features, got {}",
                max_idx + 1,
                n_features
            )));
        }
        Ok(())
    }
}

impl Default for FeatureMapping {
    fn default() -> Self {
        Self::standard_lob(10)
    }
}

// ============================================================================
// Tensor Output
// ============================================================================

/// Output container for formatted tensors.
///
/// Holds the result of tensor formatting in the appropriate shape.
#[derive(Debug, Clone)]
pub enum TensorOutput {
    /// 2D tensor: (T, F) - Flat format
    Array2(Array2<f64>),

    /// 3D tensor: (T, C, L) or (T, L, C) - DeepLOB/HLOB format
    Array3(Array3<f64>),

    /// 4D tensor: (T, C, H, W) - Image format
    Array4(Array4<f64>),
}

impl TensorOutput {
    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorOutput::Array2(arr) => arr.shape().to_vec(),
            TensorOutput::Array3(arr) => arr.shape().to_vec(),
            TensorOutput::Array4(arr) => arr.shape().to_vec(),
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        match self {
            TensorOutput::Array2(_) => 2,
            TensorOutput::Array3(_) => 3,
            TensorOutput::Array4(_) => 4,
        }
    }

    /// Get total number of elements.
    pub fn len(&self) -> usize {
        match self {
            TensorOutput::Array2(arr) => arr.len(),
            TensorOutput::Array3(arr) => arr.len(),
            TensorOutput::Array4(arr) => arr.len(),
        }
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unwrap as Array2 (panics if wrong type).
    pub fn into_array2(self) -> Array2<f64> {
        match self {
            TensorOutput::Array2(arr) => arr,
            _ => panic!("Expected Array2, got Array{}", self.ndim()),
        }
    }

    /// Unwrap as Array3 (panics if wrong type).
    pub fn into_array3(self) -> Array3<f64> {
        match self {
            TensorOutput::Array3(arr) => arr,
            _ => panic!("Expected Array3, got Array{}", self.ndim()),
        }
    }

    /// Unwrap as Array4 (panics if wrong type).
    pub fn into_array4(self) -> Array4<f64> {
        match self {
            TensorOutput::Array4(arr) => arr,
            _ => panic!("Expected Array4, got Array{}", self.ndim()),
        }
    }

    /// Try to get as Array2 reference.
    pub fn as_array2(&self) -> Option<&Array2<f64>> {
        match self {
            TensorOutput::Array2(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to get as Array3 reference.
    pub fn as_array3(&self) -> Option<&Array3<f64>> {
        match self {
            TensorOutput::Array3(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to get as Array4 reference.
    pub fn as_array4(&self) -> Option<&Array4<f64>> {
        match self {
            TensorOutput::Array4(arr) => Some(arr),
            _ => None,
        }
    }
}

// ============================================================================
// Tensor Formatter
// ============================================================================

/// Tensor formatter for converting flat features to model-specific shapes.
///
/// # Design
///
/// - Stateless: all configuration is provided at construction
/// - Thread-safe: can be shared across threads (no interior mutability)
/// - Efficient: single-pass formatting with minimal allocations
///
/// # Usage
///
/// ```
/// use feature_extractor::export::tensor_format::{
///     TensorFormat, TensorFormatter, FeatureMapping,
/// };
///
/// // Create formatter for DeepLOB
/// let mapping = FeatureMapping::standard_lob(10);
/// let formatter = TensorFormatter::new(TensorFormat::deeplob(), mapping);
///
/// // Format a batch of sequences
/// let sequences: Vec<Vec<Vec<f64>>> = vec![
///     vec![vec![0.0; 40]; 100], // Sequence 1: 100 timesteps, 40 features
///     vec![vec![0.0; 40]; 100], // Sequence 2
/// ];
///
/// let batch = formatter.format_batch(&sequences).unwrap();
/// // batch.shape() == [2, 100, 4, 10] for DeepLOB
/// ```
#[derive(Debug, Clone)]
pub struct TensorFormatter {
    /// Target tensor format
    format: TensorFormat,

    /// Feature index mapping
    mapping: FeatureMapping,
}

impl TensorFormatter {
    /// Create a new tensor formatter.
    ///
    /// # Arguments
    ///
    /// * `format` - Target tensor format
    /// * `mapping` - Feature index mapping
    pub fn new(format: TensorFormat, mapping: FeatureMapping) -> Self {
        Self { format, mapping }
    }

    /// Create a formatter for flat output (no transformation).
    pub fn flat() -> Self {
        Self {
            format: TensorFormat::Flat,
            mapping: FeatureMapping::default(),
        }
    }

    /// Create a formatter for DeepLOB format.
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of price levels (typically 10)
    pub fn deeplob(levels: usize) -> Self {
        Self {
            format: TensorFormat::DeepLOB { levels },
            mapping: FeatureMapping::standard_lob(levels),
        }
    }

    /// Create a formatter for HLOB format.
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of price levels (typically 10)
    pub fn hlob(levels: usize) -> Self {
        Self {
            format: TensorFormat::HLOB { levels },
            mapping: FeatureMapping::standard_lob(levels),
        }
    }

    /// Get the target format.
    pub fn format(&self) -> &TensorFormat {
        &self.format
    }

    /// Get the feature mapping.
    pub fn mapping(&self) -> &FeatureMapping {
        &self.mapping
    }

    /// Format a single sequence of flat features.
    ///
    /// # Arguments
    ///
    /// * `features` - Sequence of feature vectors: [T][F]
    ///
    /// # Returns
    ///
    /// Formatted tensor in the target shape.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Features are empty
    /// - Feature count doesn't match format requirements
    /// - Feature vectors have inconsistent lengths
    pub fn format_sequence(&self, features: &[Vec<f64>]) -> Result<TensorOutput> {
        if features.is_empty() {
            return Err(TlobError::generic("Cannot format empty sequence"));
        }

        let seq_len = features.len();
        let n_features = features[0].len();

        // Validate consistent feature count
        for (i, f) in features.iter().enumerate() {
            if f.len() != n_features {
                return Err(TlobError::generic(format!(
                    "Inconsistent feature count at timestep {}: expected {}, got {}",
                    i,
                    n_features,
                    f.len()
                )));
            }
        }

        // Validate format requirements
        self.format.validate_features(n_features)?;
        self.mapping.validate(n_features)?;

        match &self.format {
            TensorFormat::Flat => self.format_flat(features, seq_len, n_features),
            TensorFormat::DeepLOB { levels } => {
                self.format_deeplob(features, seq_len, *levels)
            }
            TensorFormat::HLOB { levels } => {
                self.format_hlob(features, seq_len, *levels)
            }
            TensorFormat::Image {
                channels,
                height,
                width,
            } => self.format_image(features, seq_len, *channels, *height, *width),
        }
    }

    /// Format a batch of sequences.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Batch of sequences: [N][T][F]
    ///
    /// # Returns
    ///
    /// For non-Flat formats, returns a tensor with batch dimension prepended.
    /// For Flat format, returns concatenated 2D array.
    ///
    /// # Note
    ///
    /// All sequences must have the same shape (T, F).
    pub fn format_batch(&self, sequences: &[Vec<Vec<f64>>]) -> Result<TensorOutput> {
        if sequences.is_empty() {
            return Err(TlobError::generic("Cannot format empty batch"));
        }

        // Validate all sequences have same shape
        let seq_len = sequences[0].len();
        let n_features = if seq_len > 0 {
            sequences[0][0].len()
        } else {
            return Err(TlobError::generic("First sequence is empty"));
        };

        for (i, seq) in sequences.iter().enumerate() {
            if seq.len() != seq_len {
                return Err(TlobError::generic(format!(
                    "Sequence {} has different length: expected {}, got {}",
                    i,
                    seq_len,
                    seq.len()
                )));
            }
            for (j, f) in seq.iter().enumerate() {
                if f.len() != n_features {
                    return Err(TlobError::generic(format!(
                        "Sequence {}, timestep {} has different feature count: expected {}, got {}",
                        i,
                        j,
                        n_features,
                        f.len()
                    )));
                }
            }
        }

        // Format each sequence and combine
        let formatted: Vec<TensorOutput> = sequences
            .iter()
            .map(|seq| self.format_sequence(seq))
            .collect::<Result<Vec<_>>>()?;

        self.combine_batch(&formatted)
    }

    // ========================================================================
    // Private formatting methods
    // ========================================================================

    /// Format to flat (T, F) - essentially a no-op reshape.
    fn format_flat(
        &self,
        features: &[Vec<f64>],
        seq_len: usize,
        n_features: usize,
    ) -> Result<TensorOutput> {
        // Flatten into contiguous array
        let mut data = Vec::with_capacity(seq_len * n_features);
        for timestep in features {
            data.extend_from_slice(timestep);
        }

        let array = Array2::from_shape_vec((seq_len, n_features), data)
            .map_err(|e| TlobError::generic(format!("Failed to create Array2: {}", e)))?;

        Ok(TensorOutput::Array2(array))
    }

    /// Format to DeepLOB (T, 4, L).
    ///
    /// Channel order: [ask_price, ask_vol, bid_price, bid_vol]
    fn format_deeplob(
        &self,
        features: &[Vec<f64>],
        seq_len: usize,
        levels: usize,
    ) -> Result<TensorOutput> {
        let mut data = vec![0.0f64; seq_len * 4 * levels];

        for t in 0..seq_len {
            let timestep = &features[t];

            for l in 0..levels {
                let base_idx = t * 4 * levels;
                
                // Channel 0: ask_price
                data[base_idx + l] = timestep[self.mapping.ask_price_index(l)];

                // Channel 1: ask_vol
                data[base_idx + levels + l] = timestep[self.mapping.ask_size_index(l)];

                // Channel 2: bid_price
                data[base_idx + 2 * levels + l] = timestep[self.mapping.bid_price_index(l)];

                // Channel 3: bid_vol
                data[base_idx + 3 * levels + l] = timestep[self.mapping.bid_size_index(l)];
            }
        }

        let array = Array3::from_shape_vec((seq_len, 4, levels), data)
            .map_err(|e| TlobError::generic(format!("Failed to create DeepLOB Array3: {}", e)))?;

        Ok(TensorOutput::Array3(array))
    }

    /// Format to HLOB (T, L, 4).
    ///
    /// Feature order per level: [ask_price, ask_vol, bid_price, bid_vol]
    fn format_hlob(
        &self,
        features: &[Vec<f64>],
        seq_len: usize,
        levels: usize,
    ) -> Result<TensorOutput> {
        let mut data = vec![0.0f64; seq_len * levels * 4];

        for t in 0..seq_len {
            let timestep = &features[t];

            for l in 0..levels {
                // Feature 0: ask_price
                let idx_ask_p = t * levels * 4 + l * 4 + 0;
                data[idx_ask_p] = timestep[self.mapping.ask_price_index(l)];

                // Feature 1: ask_vol
                let idx_ask_v = t * levels * 4 + l * 4 + 1;
                data[idx_ask_v] = timestep[self.mapping.ask_size_index(l)];

                // Feature 2: bid_price
                let idx_bid_p = t * levels * 4 + l * 4 + 2;
                data[idx_bid_p] = timestep[self.mapping.bid_price_index(l)];

                // Feature 3: bid_vol
                let idx_bid_v = t * levels * 4 + l * 4 + 3;
                data[idx_bid_v] = timestep[self.mapping.bid_size_index(l)];
            }
        }

        let array = Array3::from_shape_vec((seq_len, levels, 4), data)
            .map_err(|e| TlobError::generic(format!("Failed to create HLOB Array3: {}", e)))?;

        Ok(TensorOutput::Array3(array))
    }

    /// Format to Image (T, C, H, W).
    fn format_image(
        &self,
        features: &[Vec<f64>],
        seq_len: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<TensorOutput> {
        let mut data = vec![0.0f64; seq_len * channels * height * width];

        for t in 0..seq_len {
            let timestep = &features[t];

            // Fill in row-major order: C, H, W
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let flat_idx = c * height * width + h * width + w;
                        let out_idx =
                            t * channels * height * width + c * height * width + h * width + w;

                        if flat_idx < timestep.len() {
                            data[out_idx] = timestep[flat_idx];
                        }
                        // Leave as 0.0 if flat_idx >= timestep.len()
                    }
                }
            }
        }

        let array = Array4::from_shape_vec((seq_len, channels, height, width), data)
            .map_err(|e| TlobError::generic(format!("Failed to create Image Array4: {}", e)))?;

        Ok(TensorOutput::Array4(array))
    }

    /// Combine batch of formatted tensors.
    fn combine_batch(&self, tensors: &[TensorOutput]) -> Result<TensorOutput> {
        if tensors.is_empty() {
            return Err(TlobError::generic("Empty tensor batch"));
        }

        // For now, we only support combining 3D tensors into 4D
        // This could be extended for other combinations
        match &tensors[0] {
            TensorOutput::Array2(_) => {
                // Stack 2D arrays into 3D: [N, T, F]
                let arrays: Vec<&Array2<f64>> = tensors
                    .iter()
                    .map(|t| t.as_array2().unwrap())
                    .collect();

                let n_batch = arrays.len();
                let (seq_len, n_features) = arrays[0].dim();

                let mut data = Vec::with_capacity(n_batch * seq_len * n_features);
                for arr in arrays {
                    data.extend(arr.iter());
                }

                let result = Array3::from_shape_vec((n_batch, seq_len, n_features), data)
                    .map_err(|e| TlobError::generic(format!("Failed to stack arrays: {}", e)))?;

                Ok(TensorOutput::Array3(result))
            }

            TensorOutput::Array3(_) => {
                // Stack 3D arrays into 4D: [N, T, C, L] or [N, T, L, C]
                let arrays: Vec<&Array3<f64>> = tensors
                    .iter()
                    .map(|t| t.as_array3().unwrap())
                    .collect();

                let n_batch = arrays.len();
                let (d0, d1, d2) = arrays[0].dim();

                let mut data = Vec::with_capacity(n_batch * d0 * d1 * d2);
                for arr in arrays {
                    data.extend(arr.iter());
                }

                let result = Array4::from_shape_vec((n_batch, d0, d1, d2), data)
                    .map_err(|e| TlobError::generic(format!("Failed to stack arrays: {}", e)))?;

                Ok(TensorOutput::Array4(result))
            }

            TensorOutput::Array4(_) => {
                // Cannot stack 4D into 5D - return error
                Err(TlobError::generic(
                    "Cannot batch 4D tensors (would require 5D output)",
                ))
            }
        }
    }
}

impl Default for TensorFormatter {
    fn default() -> Self {
        Self::flat()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TensorFormat Tests
    // ========================================================================

    #[test]
    fn test_tensor_format_output_shape() {
        assert_eq!(TensorFormat::Flat.output_shape(100, 40), vec![100, 40]);
        assert_eq!(
            TensorFormat::DeepLOB { levels: 10 }.output_shape(100, 40),
            vec![100, 4, 10]
        );
        assert_eq!(
            TensorFormat::HLOB { levels: 10 }.output_shape(100, 40),
            vec![100, 10, 4]
        );
        assert_eq!(
            TensorFormat::Image {
                channels: 2,
                height: 10,
                width: 20
            }
            .output_shape(100, 400),
            vec![100, 2, 10, 20]
        );
    }

    #[test]
    fn test_tensor_format_ndim() {
        assert_eq!(TensorFormat::Flat.ndim(), 2);
        assert_eq!(TensorFormat::DeepLOB { levels: 10 }.ndim(), 3);
        assert_eq!(TensorFormat::HLOB { levels: 10 }.ndim(), 3);
        assert_eq!(
            TensorFormat::Image {
                channels: 1,
                height: 10,
                width: 10
            }
            .ndim(),
            4
        );
    }

    #[test]
    fn test_tensor_format_validate_features() {
        assert!(TensorFormat::Flat.validate_features(40).is_ok());
        assert!(TensorFormat::Flat.validate_features(0).is_ok());

        assert!(TensorFormat::DeepLOB { levels: 10 }
            .validate_features(40)
            .is_ok());
        assert!(TensorFormat::DeepLOB { levels: 10 }
            .validate_features(39)
            .is_err());

        assert!(TensorFormat::HLOB { levels: 10 }
            .validate_features(40)
            .is_ok());
        assert!(TensorFormat::HLOB { levels: 10 }
            .validate_features(39)
            .is_err());
    }

    #[test]
    fn test_tensor_format_requires_reordering() {
        assert!(!TensorFormat::Flat.requires_reordering());
        assert!(TensorFormat::DeepLOB { levels: 10 }.requires_reordering());
        assert!(TensorFormat::HLOB { levels: 10 }.requires_reordering());
    }

    // ========================================================================
    // FeatureMapping Tests
    // ========================================================================

    #[test]
    fn test_feature_mapping_standard_lob() {
        let mapping = FeatureMapping::standard_lob(10);

        assert_eq!(mapping.ask_price_index(0), 0);
        assert_eq!(mapping.ask_price_index(9), 9);
        assert_eq!(mapping.ask_size_index(0), 10);
        assert_eq!(mapping.ask_size_index(9), 19);
        assert_eq!(mapping.bid_price_index(0), 20);
        assert_eq!(mapping.bid_price_index(9), 29);
        assert_eq!(mapping.bid_size_index(0), 30);
        assert_eq!(mapping.bid_size_index(9), 39);
    }

    #[test]
    fn test_feature_mapping_validate() {
        let mapping = FeatureMapping::standard_lob(10);

        assert!(mapping.validate(40).is_ok());
        assert!(mapping.validate(50).is_ok()); // More features is OK
        assert!(mapping.validate(39).is_err()); // Not enough features
    }

    // ========================================================================
    // TensorOutput Tests
    // ========================================================================

    #[test]
    fn test_tensor_output_shape() {
        let arr2 = Array2::<f64>::zeros((100, 40));
        let output = TensorOutput::Array2(arr2);
        assert_eq!(output.shape(), vec![100, 40]);
        assert_eq!(output.ndim(), 2);
        assert_eq!(output.len(), 4000);

        let arr3 = Array3::<f64>::zeros((100, 4, 10));
        let output = TensorOutput::Array3(arr3);
        assert_eq!(output.shape(), vec![100, 4, 10]);
        assert_eq!(output.ndim(), 3);
        assert_eq!(output.len(), 4000);
    }

    #[test]
    fn test_tensor_output_unwrap() {
        let arr2 = Array2::<f64>::zeros((10, 5));
        let output = TensorOutput::Array2(arr2);
        let unwrapped = output.into_array2();
        assert_eq!(unwrapped.dim(), (10, 5));
    }

    // ========================================================================
    // TensorFormatter Tests - Flat Format
    // ========================================================================

    #[test]
    fn test_formatter_flat() {
        let formatter = TensorFormatter::flat();

        let features: Vec<Vec<f64>> = (0..100)
            .map(|t| (0..40).map(|f| (t * 40 + f) as f64).collect())
            .collect();

        let output = formatter.format_sequence(&features).unwrap();

        assert_eq!(output.shape(), vec![100, 40]);

        let arr = output.into_array2();
        assert_eq!(arr[[0, 0]], 0.0);
        assert_eq!(arr[[0, 39]], 39.0);
        assert_eq!(arr[[99, 0]], 3960.0);
        assert_eq!(arr[[99, 39]], 3999.0);
    }

    // ========================================================================
    // TensorFormatter Tests - DeepLOB Format
    // ========================================================================

    #[test]
    fn test_formatter_deeplob_shape() {
        let formatter = TensorFormatter::deeplob(10);

        let features: Vec<Vec<f64>> = vec![vec![0.0; 40]; 100];

        let output = formatter.format_sequence(&features).unwrap();

        assert_eq!(output.shape(), vec![100, 4, 10]);
    }

    #[test]
    fn test_formatter_deeplob_values() {
        let formatter = TensorFormatter::deeplob(10);

        // Create features where we know the expected values
        // ask_prices: [1, 2, ..., 10]
        // ask_sizes:  [11, 12, ..., 20]
        // bid_prices: [21, 22, ..., 30]
        // bid_sizes:  [31, 32, ..., 40]
        let features: Vec<Vec<f64>> = vec![(1..=40).map(|x| x as f64).collect()];

        let output = formatter.format_sequence(&features).unwrap();
        let arr = output.into_array3();

        // Check channel 0 (ask_price): [1, 2, ..., 10]
        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 9]], 10.0);

        // Check channel 1 (ask_size): [11, 12, ..., 20]
        assert_eq!(arr[[0, 1, 0]], 11.0);
        assert_eq!(arr[[0, 1, 9]], 20.0);

        // Check channel 2 (bid_price): [21, 22, ..., 30]
        assert_eq!(arr[[0, 2, 0]], 21.0);
        assert_eq!(arr[[0, 2, 9]], 30.0);

        // Check channel 3 (bid_size): [31, 32, ..., 40]
        assert_eq!(arr[[0, 3, 0]], 31.0);
        assert_eq!(arr[[0, 3, 9]], 40.0);
    }

    #[test]
    fn test_formatter_deeplob_multiple_timesteps() {
        let formatter = TensorFormatter::deeplob(10);

        // Two timesteps with different values
        let features: Vec<Vec<f64>> = vec![
            (1..=40).map(|x| x as f64).collect(),
            (101..=140).map(|x| x as f64).collect(),
        ];

        let output = formatter.format_sequence(&features).unwrap();
        let arr = output.into_array3();

        // T=0: ask_price[0] = 1.0
        assert_eq!(arr[[0, 0, 0]], 1.0);

        // T=1: ask_price[0] = 101.0
        assert_eq!(arr[[1, 0, 0]], 101.0);
    }

    // ========================================================================
    // TensorFormatter Tests - HLOB Format
    // ========================================================================

    #[test]
    fn test_formatter_hlob_shape() {
        let formatter = TensorFormatter::hlob(10);

        let features: Vec<Vec<f64>> = vec![vec![0.0; 40]; 100];

        let output = formatter.format_sequence(&features).unwrap();

        assert_eq!(output.shape(), vec![100, 10, 4]);
    }

    #[test]
    fn test_formatter_hlob_values() {
        let formatter = TensorFormatter::hlob(10);

        // Same test data as DeepLOB
        let features: Vec<Vec<f64>> = vec![(1..=40).map(|x| x as f64).collect()];

        let output = formatter.format_sequence(&features).unwrap();
        let arr = output.into_array3();

        // Level 0: [ask_p, ask_v, bid_p, bid_v] = [1, 11, 21, 31]
        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 1]], 11.0);
        assert_eq!(arr[[0, 0, 2]], 21.0);
        assert_eq!(arr[[0, 0, 3]], 31.0);

        // Level 9: [10, 20, 30, 40]
        assert_eq!(arr[[0, 9, 0]], 10.0);
        assert_eq!(arr[[0, 9, 1]], 20.0);
        assert_eq!(arr[[0, 9, 2]], 30.0);
        assert_eq!(arr[[0, 9, 3]], 40.0);
    }

    // ========================================================================
    // TensorFormatter Tests - Image Format
    // ========================================================================

    #[test]
    fn test_formatter_image_shape() {
        let mapping = FeatureMapping::default();
        let formatter = TensorFormatter::new(
            TensorFormat::Image {
                channels: 2,
                height: 5,
                width: 4,
            },
            mapping,
        );

        let features: Vec<Vec<f64>> = vec![vec![0.0; 40]; 10];

        let output = formatter.format_sequence(&features).unwrap();

        assert_eq!(output.shape(), vec![10, 2, 5, 4]);
    }

    // ========================================================================
    // TensorFormatter Tests - Error Cases
    // ========================================================================

    #[test]
    fn test_formatter_empty_sequence_error() {
        let formatter = TensorFormatter::flat();
        let features: Vec<Vec<f64>> = vec![];

        let result = formatter.format_sequence(&features);
        assert!(result.is_err());
    }

    #[test]
    fn test_formatter_inconsistent_features_error() {
        let formatter = TensorFormatter::flat();
        let features: Vec<Vec<f64>> = vec![vec![0.0; 40], vec![0.0; 30]]; // Different lengths

        let result = formatter.format_sequence(&features);
        assert!(result.is_err());
    }

    #[test]
    fn test_formatter_insufficient_features_error() {
        let formatter = TensorFormatter::deeplob(10);
        let features: Vec<Vec<f64>> = vec![vec![0.0; 30]]; // Only 30 features, need 40

        let result = formatter.format_sequence(&features);
        assert!(result.is_err());
    }

    // ========================================================================
    // TensorFormatter Tests - Batch Processing
    // ========================================================================

    #[test]
    fn test_formatter_batch_flat() {
        let formatter = TensorFormatter::flat();

        let sequences: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 40]; 100], vec![vec![1.0; 40]; 100]];

        let output = formatter.format_batch(&sequences).unwrap();

        assert_eq!(output.shape(), vec![2, 100, 40]);
    }

    #[test]
    fn test_formatter_batch_deeplob() {
        let formatter = TensorFormatter::deeplob(10);

        let sequences: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 40]; 100], vec![vec![1.0; 40]; 100]];

        let output = formatter.format_batch(&sequences).unwrap();

        assert_eq!(output.shape(), vec![2, 100, 4, 10]);
    }

    #[test]
    fn test_formatter_batch_empty_error() {
        let formatter = TensorFormatter::flat();
        let sequences: Vec<Vec<Vec<f64>>> = vec![];

        let result = formatter.format_batch(&sequences);
        assert!(result.is_err());
    }

    // ========================================================================
    // Numerical Precision Tests
    // ========================================================================

    #[test]
    fn test_formatter_preserves_precision() {
        let formatter = TensorFormatter::deeplob(10);

        // Use values that require full f64 precision
        let precise_values: Vec<f64> = (0..40)
            .map(|i| 1.0000000000001_f64 + (i as f64) * 0.0000000000001)
            .collect();
        let features: Vec<Vec<f64>> = vec![precise_values.clone()];

        let output = formatter.format_sequence(&features).unwrap();
        let arr = output.into_array3();

        // Check that precision is preserved (within f64 epsilon)
        for l in 0..10 {
            let expected = precise_values[l]; // ask_price
            let actual = arr[[0, 0, l]];
            assert!(
                (expected - actual).abs() < f64::EPSILON,
                "Precision loss at level {}: expected {}, got {}",
                l,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_formatter_no_nan_inf() {
        let formatter = TensorFormatter::deeplob(10);

        // Test with extreme values
        let mut features: Vec<f64> = vec![0.0; 40];
        features[0] = f64::MAX;
        features[10] = f64::MIN_POSITIVE;
        features[20] = -f64::MAX;

        let features: Vec<Vec<f64>> = vec![features];

        let output = formatter.format_sequence(&features).unwrap();
        let arr = output.into_array3();

        // Verify no NaN or Inf in output
        for &val in arr.iter() {
            assert!(!val.is_nan(), "Found NaN in output");
            assert!(!val.is_infinite() || val == f64::MAX || val == -f64::MAX);
        }
    }

    // ========================================================================
    // Determinism Tests
    // ========================================================================

    #[test]
    fn test_formatter_deterministic() {
        let formatter = TensorFormatter::deeplob(10);

        let features: Vec<Vec<f64>> = (0..100)
            .map(|t| (0..40).map(|f| (t * 40 + f) as f64 * 0.001).collect())
            .collect();

        let output1 = formatter.format_sequence(&features).unwrap();
        let output2 = formatter.format_sequence(&features).unwrap();

        let arr1 = output1.into_array3();
        let arr2 = output2.into_array3();

        assert_eq!(arr1, arr2, "Formatting should be deterministic");
    }
}

