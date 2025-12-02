//! Sequence building for transformer models.
//!
//! This module provides efficient sequence generation for time-series transformers.
//! The TLOB model requires sequences of LOB snapshots (typically 100 snapshots)
//! rather than individual snapshots.
//!
//! # Architecture
//!
//! - **SequenceBuilder**: Maintains a circular buffer of feature vectors
//! - **SequenceConfig**: Configuration for window size, stride, etc.
//! - **Sequence**: Output type containing features and metadata
//!
//! # Memory Management
//!
//! The builder uses a **bounded circular buffer** to prevent unbounded memory growth:
//! - Fixed capacity (e.g., 1000 snapshots)
//! - Automatic eviction of old snapshots
//! - O(1) insertion and sequence extraction
//!
//! # Performance
//!
//! - Insertion: O(1) with minimal allocations
//! - Sequence extraction: O(window_size) - single allocation
//! - Memory: O(capacity × feature_count) - bounded and predictable
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::sequence::{SequenceBuilder, SequenceConfig};
//!
//! let config = SequenceConfig::new(100, 1); // 100 snapshots, stride 1
//! let mut builder = SequenceBuilder::with_config(config);
//!
//! // Add features as they're sampled
//! for features in sampled_features {
//!     builder.push(timestamp, features);
//!     
//!     // Try to get a complete sequence
//!     if let Some(seq) = builder.try_build_sequence() {
//!         // Feed to transformer
//!         model.predict(seq.features);
//!     }
//! }
//! ```

use std::collections::VecDeque;
use std::fmt;

/// Error type for sequence building operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceError {
    /// Feature vector length doesn't match configured feature_count.
    FeatureCountMismatch {
        /// Expected number of features
        expected: usize,
        /// Actual number of features received
        actual: usize,
    },
}

impl fmt::Display for SequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Feature vector length ({}) doesn't match configured count ({})",
                    actual, expected
                )
            }
        }
    }
}

impl std::error::Error for SequenceError {}

/// Configuration for sequence building.
///
/// Controls how sequences are generated from the feature buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SequenceConfig {
    /// Number of snapshots per sequence (transformer input length)
    ///
    /// TLOB paper uses 100 snapshots. Larger windows capture longer-term
    /// dependencies but require more memory and compute.
    ///
    /// Recommended values:
    /// - Short-term prediction: 50-100 snapshots
    /// - Long-term prediction: 100-200 snapshots
    pub window_size: usize,

    /// Stride for sliding window (number of snapshots to skip between sequences)
    ///
    /// - Stride 1: Maximum overlap, more training samples, slower
    /// - Stride = window_size: No overlap, fewer samples, faster
    /// - Stride 10-20: Good compromise for most use cases
    ///
    /// Recommended: 1 for online inference, 10-20 for training
    pub stride: usize,

    /// Maximum buffer capacity (prevents unbounded memory growth)
    ///
    /// Should be >= window_size + stride for proper operation.
    /// Recommended: 2-3× window_size
    pub max_buffer_size: usize,

    /// Number of features per snapshot (e.g., 40 for raw LOB, 48 for LOB+derived, 76 for LOB+MBO, 84 for LOB+derived+MBO)
    pub feature_count: usize,
}

impl SequenceConfig {
    /// Create a new sequence configuration.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of snapshots per sequence (e.g., 100)
    /// * `stride` - Number of snapshots to skip between sequences (e.g., 1)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::SequenceConfig;
    ///
    /// // TLOB standard: 100 snapshots, stride 1
    /// let config = SequenceConfig::new(100, 1);
    /// ```
    pub fn new(window_size: usize, stride: usize) -> Self {
        let max_buffer_size = window_size.max(1000); // At least 1000 or window_size

        Self {
            window_size,
            stride,
            max_buffer_size,
            feature_count: 40, // ✅ FIX: Default for raw LOB features (derived disabled)
        }
    }

    /// Set the feature count (40 for raw LOB, 48 for LOB+derived, 76 for LOB+MBO, 84 for LOB+derived+MBO).
    pub fn with_feature_count(mut self, count: usize) -> Self {
        self.feature_count = count;
        self
    }

    /// Set the maximum buffer size.
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size.max(self.window_size); // Ensure >= window_size
        self
    }

    /// Validate configuration.
    ///
    /// Returns Ok(()) if valid, Err(msg) otherwise.
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("window_size must be > 0".to_string());
        }

        if self.stride == 0 {
            return Err("stride must be > 0".to_string());
        }

        if self.max_buffer_size < self.window_size {
            return Err(format!(
                "max_buffer_size ({}) must be >= window_size ({})",
                self.max_buffer_size, self.window_size
            ));
        }

        if self.feature_count == 0 {
            return Err("feature_count must be > 0".to_string());
        }

        Ok(())
    }
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self::new(100, 1) // TLOB standard
    }
}

/// A snapshot of features at a specific timestamp.
///
/// This is the atomic unit stored in the sequence buffer.
#[derive(Debug, Clone)]
struct Snapshot {
    /// Timestamp (nanoseconds since epoch)
    timestamp: u64,

    /// Feature vector (length = feature_count)
    features: Vec<f64>,
}

/// A complete sequence ready for transformer input.
///
/// Contains a window of consecutive snapshots and metadata.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Sequence of feature vectors [window_size × feature_count]
    ///
    /// Stored as row-major: `features[snapshot_idx][feature_idx]`
    pub features: Vec<Vec<f64>>,

    /// Timestamp of the first snapshot in the sequence
    pub start_timestamp: u64,

    /// Timestamp of the last snapshot in the sequence
    pub end_timestamp: u64,

    /// Sequence duration (nanoseconds)
    pub duration_ns: u64,

    /// Number of snapshots in sequence (should equal window_size)
    pub length: usize,
}

impl Sequence {
    /// Get the sequence as a flattened vector [window_size * feature_count]
    ///
    /// Some models expect flattened input. This creates a copy.
    pub fn as_flat(&self) -> Vec<f64> {
        let total_size = self.features.len() * self.features[0].len();
        let mut flat = Vec::with_capacity(total_size);

        for snapshot in &self.features {
            flat.extend_from_slice(snapshot);
        }

        flat
    }

    /// Get the sequence duration in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        self.duration_ns as f64 / 1e9
    }

    /// Get average time between snapshots (seconds).
    #[inline]
    pub fn avg_sample_interval(&self) -> f64 {
        if self.length <= 1 {
            return 0.0;
        }
        self.duration_seconds() / (self.length - 1) as f64
    }
}

/// Efficient sequence builder for transformer models.
///
/// Maintains a circular buffer of feature snapshots and generates sequences
/// on demand using a sliding window approach.
///
/// # Memory Guarantees
///
/// - Maximum memory: `max_buffer_size × feature_count × 8 bytes`
/// - For 1000 buffer × 40 features: ~320 KB per builder
/// - Bounded and predictable, no unbounded growth
///
/// # Thread Safety
///
/// This struct is NOT thread-safe. Use one builder per thread or add
/// synchronization if sharing across threads.
pub struct SequenceBuilder {
    /// Configuration
    config: SequenceConfig,

    /// Circular buffer of snapshots
    ///
    /// Oldest snapshots are at the front, newest at the back.
    /// When at capacity, front elements are evicted.
    buffer: VecDeque<Snapshot>,

    /// Total snapshots pushed (for tracking)
    total_pushed: u64,

    /// Total sequences built (for tracking)
    total_sequences: u64,

    /// Position of last sequence extraction (for stride tracking)
    last_sequence_pos: usize,
}

impl SequenceBuilder {
    /// Create a new sequence builder with default configuration.
    ///
    /// Uses TLOB standard: 100 snapshots, stride 1, 48 features.
    pub fn new() -> Self {
        Self::with_config(SequenceConfig::default())
    }

    /// Create a new sequence builder with custom configuration.
    ///
    /// # Panics
    ///
    /// Panics if configuration is invalid (use `validate()` first).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{SequenceBuilder, SequenceConfig};
    ///
    /// let config = SequenceConfig::new(100, 1)
    ///     .with_feature_count(84)
    ///     .with_max_buffer_size(1000);
    ///
    /// config.validate().expect("Invalid config");
    /// let builder = SequenceBuilder::with_config(config);
    /// ```
    pub fn with_config(config: SequenceConfig) -> Self {
        config.validate().expect("Invalid sequence configuration");

        Self {
            buffer: VecDeque::with_capacity(config.max_buffer_size),
            config,
            total_pushed: 0,
            total_sequences: 0,
            last_sequence_pos: 0,
        }
    }

    /// Push a new snapshot into the buffer.
    ///
    /// If the buffer is at capacity, the oldest snapshot is evicted.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Snapshot timestamp (nanoseconds since epoch)
    /// * `features` - Feature vector (length must match feature_count)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the snapshot was successfully added
    /// - `Err(SequenceError)` if the feature vector length doesn't match
    ///
    /// # Performance
    ///
    /// - O(1) amortized insertion
    /// - May allocate if features.len() != capacity
    #[inline]
    pub fn push(&mut self, timestamp: u64, features: Vec<f64>) -> Result<(), SequenceError> {
        if features.len() != self.config.feature_count {
            return Err(SequenceError::FeatureCountMismatch {
                expected: self.config.feature_count,
                actual: features.len(),
            });
        }

        // Evict oldest if at capacity
        if self.buffer.len() >= self.config.max_buffer_size {
            self.buffer.pop_front();

            // Adjust last sequence position
            if self.last_sequence_pos > 0 {
                self.last_sequence_pos -= 1;
            }
        }

        // Add new snapshot
        self.buffer.push_back(Snapshot {
            timestamp,
            features,
        });
        self.total_pushed += 1;

        Ok(())
    }

    /// Try to build a sequence from the buffer.
    ///
    /// Returns `Some(Sequence)` if enough snapshots are available,
    /// `None` otherwise.
    ///
    /// This respects the configured stride: sequences are only built
    /// when enough new snapshots have been added since the last sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(seq) = builder.try_build_sequence() {
    ///     // Feed to model
    ///     model.predict(seq.features);
    /// }
    /// ```
    pub fn try_build_sequence(&mut self) -> Option<Sequence> {
        // Check if we have enough snapshots
        if self.buffer.len() < self.config.window_size {
            return None;
        }

        // Check if enough snapshots have been added since last sequence
        let snapshots_since_last = self.buffer.len() - self.last_sequence_pos;
        if snapshots_since_last < self.config.stride {
            return None;
        }

        // Build sequence from the most recent window_size snapshots
        let start_idx = self.buffer.len() - self.config.window_size;

        let sequence_snapshots: Vec<_> = self
            .buffer
            .iter()
            .skip(start_idx)
            .take(self.config.window_size)
            .collect();

        let start_timestamp = sequence_snapshots.first().unwrap().timestamp;
        let end_timestamp = sequence_snapshots.last().unwrap().timestamp;

        let features: Vec<Vec<f64>> = sequence_snapshots
            .iter()
            .map(|s| s.features.clone())
            .collect();

        // Update tracking
        self.last_sequence_pos = self.buffer.len();
        self.total_sequences += 1;

        Some(Sequence {
            features,
            start_timestamp,
            end_timestamp,
            duration_ns: end_timestamp.saturating_sub(start_timestamp),
            length: self.config.window_size,
        })
    }

    /// Generate all possible sequences from the buffer with sliding window.
    ///
    /// Useful for batch processing or offline training data generation.
    /// Returns all sequences with the configured stride.
    ///
    /// # Returns
    ///
    /// Vector of sequences, possibly empty if buffer has insufficient data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate training data in batch
    /// let sequences = builder.generate_all_sequences();
    /// for seq in sequences {
    ///     training_data.push((seq.features, label));
    /// }
    /// ```
    pub fn generate_all_sequences(&self) -> Vec<Sequence> {
        let mut sequences = Vec::new();

        if self.buffer.len() < self.config.window_size {
            return sequences;
        }

        // Generate sequences with stride
        let max_start = self.buffer.len() - self.config.window_size;

        for start_idx in (0..=max_start).step_by(self.config.stride) {
            let sequence_snapshots: Vec<_> = self
                .buffer
                .iter()
                .skip(start_idx)
                .take(self.config.window_size)
                .collect();

            let start_timestamp = sequence_snapshots.first().unwrap().timestamp;
            let end_timestamp = sequence_snapshots.last().unwrap().timestamp;

            let features: Vec<Vec<f64>> = sequence_snapshots
                .iter()
                .map(|s| s.features.clone())
                .collect();

            sequences.push(Sequence {
                features,
                start_timestamp,
                end_timestamp,
                duration_ns: end_timestamp.saturating_sub(start_timestamp),
                length: self.config.window_size,
            });
        }

        sequences
    }

    /// Get the number of snapshots currently in the buffer.
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Get the total number of snapshots pushed (lifetime).
    #[inline]
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Get the total number of sequences built (lifetime).
    #[inline]
    pub fn total_sequences(&self) -> u64 {
        self.total_sequences
    }

    /// Check if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.config.max_buffer_size
    }

    /// Reset the builder state.
    ///
    /// Clears the buffer and resets counters. Useful for starting a new
    /// trading session or data segment.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.total_pushed = 0;
        self.total_sequences = 0;
        self.last_sequence_pos = 0;
    }

    /// Get statistics about builder usage.
    ///
    /// Returns (buffer_len, total_pushed, total_sequences, buffer_utilization%)
    pub fn statistics(&self) -> (usize, u64, u64, f64) {
        let utilization = (self.buffer.len() as f64 / self.config.max_buffer_size as f64) * 100.0;
        (
            self.buffer.len(),
            self.total_pushed,
            self.total_sequences,
            utilization,
        )
    }

    /// Get the configuration.
    pub fn config(&self) -> &SequenceConfig {
        &self.config
    }
}

impl Default for SequenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features(value: f64, count: usize) -> Vec<f64> {
        vec![value; count]
    }

    #[test]
    fn test_sequence_config_default() {
        let config = SequenceConfig::default();
        assert_eq!(config.window_size, 100);
        assert_eq!(config.stride, 1);
        assert_eq!(config.feature_count, 40); // ✅ Updated for raw LOB features
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sequence_config_validation() {
        // Valid config
        let config = SequenceConfig::new(100, 1);
        assert!(config.validate().is_ok());

        // Invalid: zero window size
        let mut bad_config = config.clone();
        bad_config.window_size = 0;
        assert!(bad_config.validate().is_err());

        // Invalid: zero stride
        let mut bad_config = config.clone();
        bad_config.stride = 0;
        assert!(bad_config.validate().is_err());

        // Invalid: buffer smaller than window
        let mut bad_config = config.clone();
        bad_config.max_buffer_size = 50;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_sequence_builder_basic() {
        let config = SequenceConfig::new(5, 1)
            .with_feature_count(3)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        // Add snapshots
        for i in 0..5 {
            let timestamp = (i * 1000) as u64;
            let features = create_test_features(i as f64, 3);
            builder.push(timestamp, features).unwrap();
        }

        assert_eq!(builder.buffer_len(), 5);
        assert_eq!(builder.total_pushed(), 5);

        // Should be able to build a sequence now
        let seq = builder.try_build_sequence().expect("Should build sequence");
        assert_eq!(seq.length, 5);
        assert_eq!(seq.features.len(), 5);
        assert_eq!(seq.start_timestamp, 0);
        assert_eq!(seq.end_timestamp, 4000);
    }

    #[test]
    fn test_sequence_builder_insufficient_data() {
        let config = SequenceConfig::new(10, 1).with_feature_count(3);
        let mut builder = SequenceBuilder::with_config(config);

        // Add only 5 snapshots (need 10)
        for i in 0..5 {
            builder
                .push(i as u64, create_test_features(i as f64, 3))
                .unwrap();
        }

        // Should not be able to build sequence
        assert!(builder.try_build_sequence().is_none());
    }

    #[test]
    fn test_sequence_builder_stride() {
        let config = SequenceConfig::new(3, 2) // window=3, stride=2
            .with_feature_count(2)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        // Add 3 snapshots - should build first sequence
        for i in 0..3 {
            builder
                .push(i as u64, create_test_features(i as f64, 2))
                .unwrap();
        }

        let seq1 = builder.try_build_sequence();
        assert!(seq1.is_some());
        assert_eq!(builder.total_sequences(), 1);

        // Add 1 more (total 4) - not enough for stride=2
        builder.push(3, create_test_features(3.0, 2)).unwrap();
        assert!(builder.try_build_sequence().is_none());

        // Add 1 more (total 5) - now we have stride=2 new snapshots
        builder.push(4, create_test_features(4.0, 2)).unwrap();
        let seq2 = builder.try_build_sequence();
        assert!(seq2.is_some());
        assert_eq!(builder.total_sequences(), 2);
    }

    #[test]
    fn test_sequence_builder_buffer_eviction() {
        let config = SequenceConfig::new(3, 1)
            .with_feature_count(2)
            .with_max_buffer_size(5);

        let mut builder = SequenceBuilder::with_config(config);

        // Fill buffer to capacity
        for i in 0..5 {
            builder
                .push(i as u64, create_test_features(i as f64, 2))
                .unwrap();
        }

        assert_eq!(builder.buffer_len(), 5);
        assert!(builder.is_full());

        // Add one more - should evict oldest
        builder.push(5, create_test_features(5.0, 2)).unwrap();
        assert_eq!(builder.buffer_len(), 5); // Still at capacity

        // The oldest snapshot (0) should be gone
        let seq = builder.try_build_sequence().unwrap();
        // Sequence should be from latest window_size=3 snapshots: [3,4,5]
        assert_eq!(seq.features[0][0], 3.0);
        assert_eq!(seq.features[2][0], 5.0);
    }

    #[test]
    fn test_sequence_generate_all() {
        let config = SequenceConfig::new(3, 2) // window=3, stride=2
            .with_feature_count(2)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        // Add 7 snapshots
        for i in 0..7 {
            builder
                .push(i as u64, create_test_features(i as f64, 2))
                .unwrap();
        }

        let sequences = builder.generate_all_sequences();

        // With window=3, stride=2, from 7 snapshots:
        // Start positions: 0, 2, 4 (max_start = 7-3 = 4)
        // So we get 3 sequences
        assert_eq!(sequences.len(), 3);

        // Check first sequence is [0,1,2]
        assert_eq!(sequences[0].features[0][0], 0.0);
        assert_eq!(sequences[0].features[2][0], 2.0);

        // Check second sequence is [2,3,4]
        assert_eq!(sequences[1].features[0][0], 2.0);
        assert_eq!(sequences[1].features[2][0], 4.0);

        // Check third sequence is [4,5,6]
        assert_eq!(sequences[2].features[0][0], 4.0);
        assert_eq!(sequences[2].features[2][0], 6.0);
    }

    #[test]
    fn test_sequence_as_flat() {
        let config = SequenceConfig::new(2, 1)
            .with_feature_count(3)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        builder.push(0, vec![1.0, 2.0, 3.0]).unwrap();
        builder.push(1, vec![4.0, 5.0, 6.0]).unwrap();

        let seq = builder.try_build_sequence().unwrap();
        let flat = seq.as_flat();

        // Should be [1,2,3,4,5,6]
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sequence_metadata() {
        let config = SequenceConfig::new(3, 1)
            .with_feature_count(2)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        // Add snapshots 1 second apart
        for i in 0..3 {
            let timestamp = i * 1_000_000_000; // i seconds in nanoseconds
            builder
                .push(timestamp, create_test_features(i as f64, 2))
                .unwrap();
        }

        let seq = builder.try_build_sequence().unwrap();

        assert_eq!(seq.start_timestamp, 0);
        assert_eq!(seq.end_timestamp, 2_000_000_000);
        assert_eq!(seq.duration_ns, 2_000_000_000);
        assert_eq!(seq.duration_seconds(), 2.0);
        assert_eq!(seq.avg_sample_interval(), 1.0); // 2s / 2 intervals
    }

    #[test]
    fn test_sequence_builder_reset() {
        let config = SequenceConfig::new(3, 1)
            .with_feature_count(2)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        for i in 0..5 {
            builder
                .push(i as u64, create_test_features(i as f64, 2))
                .unwrap();
        }

        assert_eq!(builder.buffer_len(), 5);
        assert_eq!(builder.total_pushed(), 5);

        builder.reset();

        assert_eq!(builder.buffer_len(), 0);
        assert_eq!(builder.total_pushed(), 0);
        assert_eq!(builder.total_sequences(), 0);
    }

    #[test]
    fn test_sequence_builder_statistics() {
        let config = SequenceConfig::new(3, 1)
            .with_feature_count(2)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        for i in 0..5 {
            builder
                .push(i as u64, create_test_features(i as f64, 2))
                .unwrap();
        }

        builder.try_build_sequence();

        let (buf_len, total_pushed, total_seqs, utilization) = builder.statistics();
        assert_eq!(buf_len, 5);
        assert_eq!(total_pushed, 5);
        assert_eq!(total_seqs, 1);
        assert_eq!(utilization, 50.0); // 5/10 = 50%
    }

    #[test]
    fn test_sequence_builder_wrong_feature_count() {
        let config = SequenceConfig::new(3, 1)
            .with_feature_count(5)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        // Try to push with wrong feature count (should return error)
        let result = builder.push(0, vec![1.0, 2.0, 3.0]); // Only 3 features, expected 5

        assert!(result.is_err());
        match result {
            Err(SequenceError::FeatureCountMismatch { expected, actual }) => {
                assert_eq!(expected, 5);
                assert_eq!(actual, 3);
            }
            Ok(_) => panic!("Expected error but got Ok"),
        }

        // Buffer should remain empty after failed push
        assert_eq!(builder.buffer_len(), 0);
    }

    #[test]
    fn test_sequence_timestamp_ordering() {
        let config = SequenceConfig::new(3, 1)
            .with_feature_count(2)
            .with_max_buffer_size(10);

        let mut builder = SequenceBuilder::with_config(config);

        // Add with specific timestamps
        builder.push(1000, create_test_features(1.0, 2)).unwrap();
        builder.push(2000, create_test_features(2.0, 2)).unwrap();
        builder.push(3000, create_test_features(3.0, 2)).unwrap();

        let seq = builder.try_build_sequence().unwrap();

        // Verify chronological ordering
        assert!(seq.start_timestamp < seq.end_timestamp);
        assert_eq!(seq.start_timestamp, 1000);
        assert_eq!(seq.end_timestamp, 3000);
    }

    #[test]
    fn test_sequence_error_display() {
        let error = SequenceError::FeatureCountMismatch {
            expected: 40,
            actual: 48,
        };
        let display = format!("{}", error);
        assert!(display.contains("48"));
        assert!(display.contains("40"));
    }
}
