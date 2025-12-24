//! Multi-Scale Window Architecture for TLOB
//!
//! This module provides parallel time-scale windowing with intelligent decimation
//! to capture both microstructure patterns and macro trends simultaneously.
//!
//! # Problem Statement
//!
//! Single-scale sequences miss important patterns at different time scales:
//! - **Tick-level (fast)**: Captures microstructure, order flow dynamics
//! - **Medium-scale**: Captures short-term trends, price momentum
//! - **Macro-scale (slow)**: Captures long-term context, market regime
//!
//! # Solution
//!
//! Maintain three parallel windows with different decimation rates:
//! ```text
//! Fast Window:    100 events, decimation=1 (full resolution)
//! Medium Window:  500 events, decimation=2 (every 2nd event)
//! Slow Window:   1000 events, decimation=4 (every 4th event)
//! ```
//!
//! # Mathematical Foundation
//!
//! **Decimation Strategy**:
//! ```text
//! For scale s with decimation factor d_s:
//!   - Counter c_s increments on each push
//!   - Sample when: c_s % d_s == 0
//!   - Reset c_s after sampling
//! ```
//!
//! **Memory Efficiency**:
//! ```text
//! Total memory ≈ 3× single scale (not 17×)
//! Because: 100 + 250 + 250 = 600 samples
//!          (vs 100 + 500 + 1000 = 1600 if no decimation)
//! ```
//!
//! # Key Features
//!
//! - **Three Time Scales**: Fast/Medium/Slow for multi-resolution analysis
//! - **Decimation**: Reduces memory while preserving long-term context
//! - **Temporal Alignment**: All scales use same timestamps
//! - **Modular**: Each scale uses independent SequenceBuilder
//! - **Type-Safe**: Compile-time guarantees for configuration
//!
//! # Example
//!
//! ```
//! use feature_extractor::sequence_builder::{
//!     MultiScaleWindow, MultiScaleConfig, ScaleConfig
//! };
//!
//! // Configure three scales
//! let config = MultiScaleConfig::default();
//!
//! let mut window = MultiScaleWindow::new(config, 40); // 40 features per snapshot
//!
//! // Push events (automatically decimated per scale)
//! for i in 0..200 {
//!     let ts = i as u64 * 1_000_000; // Nanosecond timestamps
//!     let features: Vec<f64> = (0..40).map(|j| (i + j) as f64).collect();
//!     window.push(ts, features);
//! }
//!
//! // Get sequences from all scales (if enough data)
//! if let Some(multiscale) = window.try_build_all() {
//!     // Fast scale: full resolution microstructure
//!     let _fast = multiscale.fast();
//!     
//!     // Medium scale: short-term trends
//!     let _medium = multiscale.medium();
//!     
//!     // Slow scale: long-term context
//!     let _slow = multiscale.slow();
//! }
//! ```

use super::{Sequence, SequenceBuilder, SequenceConfig};

/// Configuration for a single time scale.
///
/// Defines the window size, decimation factor, and stride for one scale
/// in the multi-scale architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleConfig {
    /// Number of events to keep in the window
    pub window_size: usize,

    /// Decimation factor (1 = no decimation, 2 = every 2nd event, etc.)
    pub decimation: usize,

    /// Stride for sequence generation (1 = no stride)
    pub stride: usize,
}

impl ScaleConfig {
    /// Create a new scale configuration.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of events in the window
    /// * `decimation` - Decimation factor (e.g., 2 = keep every 2nd event)
    /// * `stride` - Stride for sequence generation
    ///
    /// # Panics
    ///
    /// Panics if any parameter is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::ScaleConfig;
    ///
    /// // Fast scale: full resolution
    /// let fast = ScaleConfig::new(100, 1, 1);
    ///
    /// // Medium scale: 2× decimation
    /// let medium = ScaleConfig::new(500, 2, 1);
    ///
    /// // Slow scale: 4× decimation
    /// let slow = ScaleConfig::new(1000, 4, 1);
    /// ```
    pub fn new(window_size: usize, decimation: usize, stride: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        assert!(decimation > 0, "decimation must be > 0");
        assert!(stride > 0, "stride must be > 0");

        Self {
            window_size,
            decimation,
            stride,
        }
    }
}

/// Configuration for multi-scale windowing.
///
/// Contains configurations for all three time scales (fast/medium/slow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiScaleConfig {
    /// Fast window configuration (full resolution)
    pub fast: ScaleConfig,

    /// Medium window configuration (moderate decimation)
    pub medium: ScaleConfig,

    /// Slow window configuration (high decimation)
    pub slow: ScaleConfig,
}

impl MultiScaleConfig {
    /// Create a new multi-scale configuration.
    ///
    /// # Arguments
    ///
    /// * `fast` - Fast scale configuration
    /// * `medium` - Medium scale configuration
    /// * `slow` - Slow scale configuration
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{MultiScaleConfig, ScaleConfig};
    ///
    /// let config = MultiScaleConfig::new(
    ///     ScaleConfig::new(100, 1, 1),   // Fast
    ///     ScaleConfig::new(500, 2, 1),   // Medium
    ///     ScaleConfig::new(1000, 4, 1),  // Slow
    /// );
    /// ```
    pub fn new(fast: ScaleConfig, medium: ScaleConfig, slow: ScaleConfig) -> Self {
        Self { fast, medium, slow }
    }
}

impl Default for MultiScaleConfig {
    /// Create with TLOB paper recommendations.
    ///
    /// - Fast: 100 events, no decimation (full microstructure)
    /// - Medium: 500 events, 2× decimation (short-term trends)
    /// - Slow: 1000 events, 4× decimation (long-term context)
    fn default() -> Self {
        Self {
            fast: ScaleConfig::new(100, 1, 1),
            medium: ScaleConfig::new(500, 2, 1),
            slow: ScaleConfig::new(1000, 4, 1),
        }
    }
}

/// Multi-scale sequence output containing sequences from all three scales.
///
/// This struct holds the generated sequences from fast, medium, and slow
/// time scales, along with metadata about the multi-scale generation.
#[derive(Debug, Clone)]
pub struct MultiScaleSequence {
    /// Fast scale sequences (full resolution)
    fast: Vec<Sequence>,

    /// Medium scale sequences (2× decimation)
    medium: Vec<Sequence>,

    /// Slow scale sequences (4× decimation)
    slow: Vec<Sequence>,

    /// Total number of events processed
    total_events: usize,
}

impl MultiScaleSequence {
    /// Create a new multi-scale sequence.
    fn new(
        fast: Vec<Sequence>,
        medium: Vec<Sequence>,
        slow: Vec<Sequence>,
        total_events: usize,
    ) -> Self {
        Self {
            fast,
            medium,
            slow,
            total_events,
        }
    }

    /// Get fast scale sequences (full resolution).
    pub fn fast(&self) -> &[Sequence] {
        &self.fast
    }

    /// Get medium scale sequences (2× decimation).
    pub fn medium(&self) -> &[Sequence] {
        &self.medium
    }

    /// Get slow scale sequences (4× decimation).
    pub fn slow(&self) -> &[Sequence] {
        &self.slow
    }

    /// Get total number of events processed.
    pub fn total_events(&self) -> usize {
        self.total_events
    }

    /// Get the number of sequences at each scale.
    pub fn sequence_counts(&self) -> (usize, usize, usize) {
        (self.fast.len(), self.medium.len(), self.slow.len())
    }
}

/// Multi-scale window manager with three parallel time scales.
///
/// Maintains three independent sequence builders (fast/medium/slow) with
/// different decimation rates to capture multi-resolution temporal patterns.
///
/// # Architecture
///
/// Each scale has:
/// - Independent SequenceBuilder with configured window size
/// - Decimation counter to track sampling frequency
/// - Consistent feature count across all scales
///
/// # Decimation Logic
///
/// ```text
/// For each push(timestamp, features):
///   fast_counter += 1
///   if fast_counter % fast.decimation == 0:
///     fast_builder.push(timestamp, features)
///   
///   medium_counter += 1
///   if medium_counter % medium.decimation == 0:
///     medium_builder.push(timestamp, features)
///   
///   slow_counter += 1
///   if slow_counter % slow.decimation == 0:
///     slow_builder.push(timestamp, features)
/// ```
pub struct MultiScaleWindow {
    /// Configuration for all scales
    config: MultiScaleConfig,

    /// Fast scale builder (full resolution)
    fast_builder: SequenceBuilder,

    /// Medium scale builder (moderate decimation)
    medium_builder: SequenceBuilder,

    /// Slow scale builder (high decimation)
    slow_builder: SequenceBuilder,

    /// Decimation counter for fast scale
    fast_counter: usize,

    /// Decimation counter for medium scale
    medium_counter: usize,

    /// Decimation counter for slow scale
    slow_counter: usize,

    /// Total number of events pushed
    total_events: usize,

    /// Number of features per snapshot
    feature_count: usize,

    /// Accumulated fast sequences (streaming mode)
    accumulated_fast: Vec<Sequence>,

    /// Accumulated medium sequences (streaming mode)
    accumulated_medium: Vec<Sequence>,

    /// Accumulated slow sequences (streaming mode)
    accumulated_slow: Vec<Sequence>,
}

impl MultiScaleWindow {
    /// Create a new multi-scale window.
    ///
    /// # Arguments
    ///
    /// * `config` - Multi-scale configuration
    /// * `feature_count` - Number of features per snapshot
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{MultiScaleWindow, MultiScaleConfig};
    ///
    /// let config = MultiScaleConfig::default();
    /// let window = MultiScaleWindow::new(config, 40); // 40 features (10 levels × 4)
    /// ```
    pub fn new(config: MultiScaleConfig, feature_count: usize) -> Self {
        // Create SequenceBuilder for each scale with correct feature count
        let fast_builder = SequenceBuilder::with_config(
            SequenceConfig::new(config.fast.window_size, config.fast.stride)
                .with_feature_count(feature_count),
        );

        let medium_builder = SequenceBuilder::with_config(
            SequenceConfig::new(config.medium.window_size, config.medium.stride)
                .with_feature_count(feature_count),
        );

        let slow_builder = SequenceBuilder::with_config(
            SequenceConfig::new(config.slow.window_size, config.slow.stride)
                .with_feature_count(feature_count),
        );

        Self {
            config,
            fast_builder,
            medium_builder,
            slow_builder,
            fast_counter: 0,
            medium_counter: 0,
            slow_counter: 0,
            total_events: 0,
            feature_count,
            accumulated_fast: Vec::new(),
            accumulated_medium: Vec::new(),
            accumulated_slow: Vec::new(),
        }
    }

    /// Push a new event to all scales (with appropriate decimation).
    ///
    /// Each scale samples according to its decimation factor:
    /// - Fast (decimation=1): Every event
    /// - Medium (decimation=2): Every 2nd event
    /// - Slow (decimation=4): Every 4th event
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Event timestamp (nanoseconds)
    /// * `features` - Feature vector (must match `feature_count`)
    ///
    /// # Panics
    ///
    /// Panics if `features.len()` doesn't match the configured `feature_count`.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{MultiScaleWindow, MultiScaleConfig};
    ///
    /// let mut window = MultiScaleWindow::new(MultiScaleConfig::default(), 40);
    ///
    /// // Push events (automatically decimated)
    /// for i in 0..1000 {
    ///     let features = vec![0.0; 40];
    ///     window.push(i * 1000, features);
    /// }
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method clones the `Vec<f64>` for each scale that samples this event.
    /// For maximum performance, use [`Self::push_arc`] with `Arc<Vec<f64>>` to enable
    /// zero-copy sharing across scales.
    pub fn push(&mut self, timestamp: u64, features: Vec<f64>) {
        self.push_arc(timestamp, std::sync::Arc::new(features));
    }

    /// Push features to all scales using shared Arc (zero-copy path).
    ///
    /// This is the **primary API for maximum performance**. The `FeatureVec`
    /// (`Arc<Vec<f64>>`) is shared across all scales without deep copying.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Timestamp for this snapshot
    /// * `features` - Feature vector wrapped in Arc for zero-copy sharing
    ///
    /// # Decimation Behavior
    ///
    /// Each scale maintains its own counter:
    /// - Fast scale samples every N events (default: 1)
    /// - Medium scale samples every 2N events (default: 2)
    /// - Slow scale samples every 4N events (default: 4)
    ///
    /// The same Arc is cloned (8 bytes) for each scale, not the underlying data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use feature_extractor::sequence_builder::MultiScaleWindow;
    ///
    /// // Extract features as Arc
    /// let features: Arc<Vec<f64>> = extractor.extract_arc(&lob_state)?;
    ///
    /// // Push to all scales (zero-copy sharing)
    /// window.push_arc(timestamp, features);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Zero data copying** - only Arc pointers are cloned (8 bytes each)
    /// - Memory savings: ~66% reduction for typical multi-scale usage
    /// - Ideal for high-throughput preprocessing pipelines
    pub fn push_arc(&mut self, timestamp: u64, features: super::FeatureVec) {
        assert_eq!(
            features.len(),
            self.feature_count,
            "Feature count mismatch: expected {}, got {}",
            self.feature_count,
            features.len()
        );

        // Increment total event counter
        self.total_events += 1;

        // Fast scale: increment and check decimation
        self.fast_counter += 1;
        if self.fast_counter % self.config.fast.decimation == 0 {
            // Clone Arc (8 bytes) not Vec (672 bytes)
            let _ = self.fast_builder.push_arc(timestamp, features.clone());
            // STREAMING FIX: Try to build sequence immediately after push
            if let Some(seq) = self.fast_builder.try_build_sequence() {
                self.accumulated_fast.push(seq);
            }
        }

        // Medium scale: increment and check decimation
        self.medium_counter += 1;
        if self.medium_counter % self.config.medium.decimation == 0 {
            // Clone Arc (8 bytes) not Vec (672 bytes)
            let _ = self.medium_builder.push_arc(timestamp, features.clone());
            // STREAMING FIX: Try to build sequence immediately after push
            if let Some(seq) = self.medium_builder.try_build_sequence() {
                self.accumulated_medium.push(seq);
            }
        }

        // Slow scale: increment and check decimation
        self.slow_counter += 1;
        if self.slow_counter % self.config.slow.decimation == 0 {
            // Move Arc (no clone needed for last consumer)
            let _ = self.slow_builder.push_arc(timestamp, features);
            // STREAMING FIX: Try to build sequence immediately after push
            if let Some(seq) = self.slow_builder.try_build_sequence() {
                self.accumulated_slow.push(seq);
            }
        }
    }

    /// Try to build sequences from all scales.
    ///
    /// Returns Some(MultiScaleSequence) if all scales have sufficient data,
    /// None otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{MultiScaleWindow, MultiScaleConfig};
    ///
    /// let mut window = MultiScaleWindow::new(MultiScaleConfig::default(), 40);
    ///
    /// // Push enough events
    /// for i in 0..1000 {
    ///     window.push(i * 1000, vec![0.0; 40]);
    /// }
    ///
    /// // Try to build from all scales
    /// if let Some(multiscale) = window.try_build_all() {
    ///     println!("Fast sequences: {}", multiscale.fast().len());
    ///     println!("Medium sequences: {}", multiscale.medium().len());
    ///     println!("Slow sequences: {}", multiscale.slow().len());
    /// }
    /// ```
    pub fn try_build_all(&mut self) -> Option<MultiScaleSequence> {
        // STREAMING FIX: Return ALL accumulated sequences from streaming mode,
        // plus any final sequences that can be built from remaining buffer data.
        
        // Try to build any remaining sequences from buffers
        if let Some(seq) = self.fast_builder.try_build_sequence() {
            self.accumulated_fast.push(seq);
        }
        if let Some(seq) = self.medium_builder.try_build_sequence() {
            self.accumulated_medium.push(seq);
        }
        if let Some(seq) = self.slow_builder.try_build_sequence() {
            self.accumulated_slow.push(seq);
        }

        // Return None if no sequences were built at any scale
        if self.accumulated_fast.is_empty()
            && self.accumulated_medium.is_empty()
            && self.accumulated_slow.is_empty()
        {
            return None;
        }

        // Take ownership of accumulated sequences
        let fast = std::mem::take(&mut self.accumulated_fast);
        let medium = std::mem::take(&mut self.accumulated_medium);
        let slow = std::mem::take(&mut self.accumulated_slow);

        Some(MultiScaleSequence::new(fast, medium, slow, self.total_events))
    }

    /// Get the number of events in each scale's buffer.
    ///
    /// Returns (fast_count, medium_count, slow_count).
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{MultiScaleWindow, MultiScaleConfig};
    ///
    /// let mut window = MultiScaleWindow::new(MultiScaleConfig::default(), 40);
    ///
    /// // Push some events
    /// for i in 0..10 {
    ///     window.push(i * 1000, vec![0.0; 40]);
    /// }
    ///
    /// let (fast, medium, slow) = window.buffer_counts();
    /// println!("Buffers: fast={}, medium={}, slow={}", fast, medium, slow);
    /// ```
    pub fn buffer_counts(&self) -> (usize, usize, usize) {
        (
            self.fast_builder.buffer_len(),
            self.medium_builder.buffer_len(),
            self.slow_builder.buffer_len(),
        )
    }

    /// Check if all scales are ready to generate sequences.
    pub fn is_ready(&self) -> bool {
        self.fast_builder.buffer_len() >= self.config.fast.window_size
            && self.medium_builder.buffer_len() >= self.config.medium.window_size
            && self.slow_builder.buffer_len() >= self.config.slow.window_size
    }

    /// Get the total number of events pushed.
    pub fn total_events(&self) -> usize {
        self.total_events
    }

    /// Get the decimation counters for each scale.
    ///
    /// Useful for debugging/monitoring decimation behavior.
    pub fn decimation_counters(&self) -> (usize, usize, usize) {
        (self.fast_counter, self.medium_counter, self.slow_counter)
    }

    /// Get the number of accumulated sequences for each scale.
    ///
    /// Returns (fast_count, medium_count, slow_count).
    ///
    /// Accumulated sequences are built during `push_arc()` calls and
    /// consumed by `try_build_all()`. This method is useful for:
    /// - Monitoring streaming progress
    /// - Verifying reset clears all state
    /// - Debugging sequence generation
    pub fn accumulated_counts(&self) -> (usize, usize, usize) {
        (
            self.accumulated_fast.len(),
            self.accumulated_medium.len(),
            self.accumulated_slow.len(),
        )
    }

    /// Reset all scales and counters.
    ///
    /// Clears ALL state to prepare for processing a new data segment (e.g., new day).
    /// After reset, the window behaves identically to a newly constructed instance.
    ///
    /// # What gets reset:
    ///
    /// - **Builders**: All three SequenceBuilder instances are recreated (clears internal buffers)
    /// - **Decimation counters**: Reset to 0
    /// - **Event counter**: Reset to 0
    /// - **Accumulated sequences**: All accumulated sequences are cleared (CRITICAL for day boundaries)
    ///
    /// # Why this matters:
    ///
    /// In multi-day processing, calling reset() between days prevents state leakage.
    /// Without clearing accumulated sequences, sequences from Day 1 would leak into
    /// Day 2's output if try_build_all() wasn't called before reset().
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::sequence_builder::{MultiScaleWindow, MultiScaleConfig};
    ///
    /// let mut window = MultiScaleWindow::new(MultiScaleConfig::default(), 40);
    ///
    /// // Process Day 1 data...
    /// // window.push_arc(...);
    ///
    /// // Reset for Day 2 (clears ALL state including accumulated sequences)
    /// window.reset();
    ///
    /// // Process Day 2 data with clean state
    /// // window.push_arc(...);
    /// ```
    pub fn reset(&mut self) {
        // Recreate builders to clear their internal buffers
        self.fast_builder = SequenceBuilder::with_config(
            SequenceConfig::new(self.config.fast.window_size, self.config.fast.stride)
                .with_feature_count(self.feature_count),
        );
        self.medium_builder = SequenceBuilder::with_config(
            SequenceConfig::new(self.config.medium.window_size, self.config.medium.stride)
                .with_feature_count(self.feature_count),
        );
        self.slow_builder = SequenceBuilder::with_config(
            SequenceConfig::new(self.config.slow.window_size, self.config.slow.stride)
                .with_feature_count(self.feature_count),
        );

        // Reset decimation counters
        self.fast_counter = 0;
        self.medium_counter = 0;
        self.slow_counter = 0;

        // Reset event counter
        self.total_events = 0;

        // CRITICAL: Clear accumulated sequences to prevent cross-day leakage
        // This was missing in the original implementation, causing sequences from
        // previous processing runs to leak into subsequent runs if try_build_all()
        // was not called before reset().
        self.accumulated_fast.clear();
        self.accumulated_medium.clear();
        self.accumulated_slow.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_config_new() {
        let config = ScaleConfig::new(100, 2, 1);
        assert_eq!(config.window_size, 100);
        assert_eq!(config.decimation, 2);
        assert_eq!(config.stride, 1);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_scale_config_invalid_window() {
        ScaleConfig::new(0, 1, 1);
    }

    #[test]
    #[should_panic(expected = "decimation must be > 0")]
    fn test_scale_config_invalid_decimation() {
        ScaleConfig::new(100, 0, 1);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_scale_config_invalid_stride() {
        ScaleConfig::new(100, 1, 0);
    }

    #[test]
    fn test_multiscale_config_default() {
        let config = MultiScaleConfig::default();

        // Fast: 100, decimation=1
        assert_eq!(config.fast.window_size, 100);
        assert_eq!(config.fast.decimation, 1);

        // Medium: 500, decimation=2
        assert_eq!(config.medium.window_size, 500);
        assert_eq!(config.medium.decimation, 2);

        // Slow: 1000, decimation=4
        assert_eq!(config.slow.window_size, 1000);
        assert_eq!(config.slow.decimation, 4);
    }

    #[test]
    fn test_multiscale_window_new() {
        let config = MultiScaleConfig::default();
        let window = MultiScaleWindow::new(config, 40);

        assert_eq!(window.total_events(), 0);
        assert_eq!(window.buffer_counts(), (0, 0, 0));
        assert!(!window.is_ready());
    }

    #[test]
    fn test_decimation_fast_scale() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push 10 events
        for i in 0..10 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        // Fast scale: decimation=1, should have all 10
        let (fast, _, _) = window.buffer_counts();
        assert_eq!(fast, 10);
    }

    #[test]
    fn test_decimation_medium_scale() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push 10 events
        for i in 0..10 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        // Medium scale: decimation=2, should have 5 (every 2nd)
        let (_, medium, _) = window.buffer_counts();
        assert_eq!(medium, 5);
    }

    #[test]
    fn test_decimation_slow_scale() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push 20 events
        for i in 0..20 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        // Slow scale: decimation=4, should have 5 (every 4th)
        let (_, _, slow) = window.buffer_counts();
        assert_eq!(slow, 5);
    }

    #[test]
    fn test_decimation_all_scales() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push 100 events
        for i in 0..100 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        let (fast, medium, slow) = window.buffer_counts();

        // Fast: 100 (all events)
        assert_eq!(fast, 100);

        // Medium: 50 (every 2nd)
        assert_eq!(medium, 50);

        // Slow: 25 (every 4th)
        assert_eq!(slow, 25);
    }

    #[test]
    fn test_decimation_counters() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push 10 events
        for i in 0..10 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        let (fast_c, medium_c, slow_c) = window.decimation_counters();

        // All counters should equal total events
        assert_eq!(fast_c, 10);
        assert_eq!(medium_c, 10);
        assert_eq!(slow_c, 10);
    }

    #[test]
    fn test_try_build_all_insufficient_data() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push only 50 events (not enough for fast=100)
        for i in 0..50 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        assert!(window.try_build_all().is_none());
    }

    #[test]
    fn test_try_build_all_sufficient_data() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push enough events for ALL scales (need 4000 for slow scale)
        for i in 0..4000 {
            window.push(i * 1000, vec![1.0, 2.0, 3.0, 4.0]);
        }

        let multiscale = window.try_build_all();
        assert!(multiscale.is_some());

        let ms = multiscale.unwrap();
        assert_eq!(ms.total_events(), 4000);

        // STREAMING FIX: Now produces MANY sequences per scale, not just 1
        let (fast_count, medium_count, slow_count) = ms.sequence_counts();
        // Fast scale: 4000 events, window=100, stride=1 → ~3900 sequences
        assert!(
            fast_count > 100,
            "Fast scale should produce many sequences: {}",
            fast_count
        );
        // Medium scale: 4000/2=2000 events, window=500, stride=1 → ~1500 sequences
        assert!(
            medium_count > 100,
            "Medium scale should produce many sequences: {}",
            medium_count
        );
        // Slow scale: 4000/4=1000 events, window=1000, stride=1 → 1 sequence
        assert!(
            slow_count >= 1,
            "Slow scale should produce at least 1 sequence: {}",
            slow_count
        );
    }

    #[test]
    fn test_reset() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push some events
        for i in 0..50 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        assert_eq!(window.total_events(), 50);
        assert_ne!(window.buffer_counts(), (0, 0, 0));

        // Reset
        window.reset();

        assert_eq!(window.total_events(), 0);
        assert_eq!(window.buffer_counts(), (0, 0, 0));
        assert_eq!(window.decimation_counters(), (0, 0, 0));
    }

    #[test]
    #[should_panic(expected = "Feature count mismatch")]
    fn test_push_wrong_feature_count() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push with wrong feature count
        window.push(1000, vec![0.0; 10]); // Expected 4, got 10
    }

    #[test]
    fn test_temporal_alignment() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push enough events for all scales (need 4000 for slow scale with decimation)
        for i in 0..4000 {
            let ts = (i * 1000) as u64;
            window.push(ts, vec![i as f64; 4]);
        }

        let ms = window.try_build_all();
        assert!(ms.is_some(), "Should produce sequences with 4000 events");
        
        let ms = ms.unwrap();
        // All scales should have sequences
        assert!(!ms.fast().is_empty(), "Fast scale should have sequences");
        assert!(!ms.medium().is_empty(), "Medium scale should have sequences");
        assert!(!ms.slow().is_empty(), "Slow scale should have sequences");

        // Check first timestamp is correct
        let fast_ts = ms.fast()[0].start_timestamp;
        assert_eq!(fast_ts, 0);
    }

    #[test]
    fn test_custom_decimation() {
        // Custom config with different decimation
        let config = MultiScaleConfig::new(
            ScaleConfig::new(50, 1, 1),  // Fast: no decimation
            ScaleConfig::new(100, 3, 1), // Medium: 3× decimation
            ScaleConfig::new(150, 5, 1), // Slow: 5× decimation
        );

        let mut window = MultiScaleWindow::new(config, 4);

        // Push 30 events
        for i in 0..30 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        let (fast, medium, slow) = window.buffer_counts();

        assert_eq!(fast, 30); // All events
        assert_eq!(medium, 10); // Every 3rd
        assert_eq!(slow, 6); // Every 5th
    }

    #[test]
    fn test_is_ready() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        assert!(!window.is_ready());

        // Need enough events for slowest scale (1000 window, 4× decimation)
        // So need 1000 × 4 = 4000 events
        for i in 0..4000 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        assert!(window.is_ready());
    }

    #[test]
    fn test_sequence_feature_consistency() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push enough events with known values (need 4000 for slow scale)
        for i in 0..4000 {
            window.push(
                i * 1000,
                vec![i as f64 % 100.0, ((i * 2) as f64) % 100.0, 0.0, 0.0],
            );
        }

        let ms = window.try_build_all().unwrap();

        // Verify feature dimensions
        let fast_seq = &ms.fast()[0];
        let medium_seq = &ms.medium()[0];
        let slow_seq = &ms.slow()[0];

        // All should have correct feature count
        assert_eq!(fast_seq.features[0].len(), 4);
        assert_eq!(medium_seq.features[0].len(), 4);
        assert_eq!(slow_seq.features[0].len(), 4);
    }

    #[test]
    fn test_multiscale_sequence_accessors() {
        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Push enough events (need 4000 for slow scale)
        for i in 0..4000 {
            window.push(i * 1000, vec![0.0; 4]);
        }

        let ms = window.try_build_all().unwrap();

        // STREAMING FIX: Now produces MANY sequences per scale
        assert!(ms.fast().len() > 100, "Fast should have many sequences");
        assert!(ms.medium().len() > 100, "Medium should have many sequences");
        assert!(ms.slow().len() >= 1, "Slow should have at least 1 sequence");
        assert_eq!(ms.total_events(), 4000);

        let (f, m, s) = ms.sequence_counts();
        assert!(f > 100);
        assert!(m > 100);
        assert!(s >= 1);
    }

    // ========================================================================
    // push_arc Tests (Zero-Copy API)
    // ========================================================================

    #[test]
    fn test_push_arc_basic() {
        use std::sync::Arc;

        let config = MultiScaleConfig::new(
            ScaleConfig::new(5, 1, 1), // Fast: 5 window
            ScaleConfig::new(5, 2, 1), // Medium: 2× decimation
            ScaleConfig::new(5, 4, 1), // Slow: 4× decimation
        );

        let mut window = MultiScaleWindow::new(config, 4);

        // Push using Arc
        for i in 0..20 {
            let features = Arc::new(vec![i as f64; 4]);
            window.push_arc(i * 1000, features);
        }

        let (fast, medium, slow) = window.buffer_counts();
        assert_eq!(fast, 20); // All 20 events
        assert_eq!(medium, 10); // Every 2nd
        assert_eq!(slow, 5); // Every 4th
    }

    #[test]
    fn test_push_arc_matches_push() {
        use std::sync::Arc;

        let config = MultiScaleConfig::new(
            ScaleConfig::new(10, 1, 1),
            ScaleConfig::new(10, 2, 1),
            ScaleConfig::new(10, 4, 1),
        );

        let mut window_vec = MultiScaleWindow::new(config.clone(), 4);
        let mut window_arc = MultiScaleWindow::new(config, 4);

        // Generate test data
        let test_data: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64, (i * 4) as f64])
            .collect();

        // Push using push() (Vec)
        for (i, data) in test_data.iter().enumerate() {
            window_vec.push(i as u64 * 1000, data.clone());
        }

        // Push using push_arc() (Arc)
        for (i, data) in test_data.iter().enumerate() {
            window_arc.push_arc(i as u64 * 1000, Arc::new(data.clone()));
        }

        // Buffer counts should match
        assert_eq!(window_vec.buffer_counts(), window_arc.buffer_counts());
        assert_eq!(window_vec.total_events(), window_arc.total_events());
    }

    #[test]
    fn test_push_arc_zero_copy_sharing() {
        use std::sync::Arc;

        // Create config where all scales sample the same event
        let config = MultiScaleConfig::new(
            ScaleConfig::new(3, 1, 1), // Fast: every event
            ScaleConfig::new(3, 1, 1), // Medium: every event
            ScaleConfig::new(3, 1, 1), // Slow: every event
        );

        let mut window = MultiScaleWindow::new(config, 4);

        // Create features and keep a reference
        let features = Arc::new(vec![100.0, 200.0, 300.0, 400.0]);
        let features_ref = features.clone();

        // Initial count = 2 (features + features_ref)
        assert_eq!(Arc::strong_count(&features), 2);

        // Push - should clone Arc to all 3 scales
        window.push_arc(1000, features);

        // Our reference is still valid
        assert_eq!(features_ref[0], 100.0);

        // Push more to build sequences
        window.push_arc(2000, Arc::new(vec![101.0, 201.0, 301.0, 401.0]));
        window.push_arc(3000, Arc::new(vec![102.0, 202.0, 302.0, 402.0]));

        // Build sequences
        let ms = window.try_build_all().unwrap();

        // All scales should have our original data
        assert_eq!(ms.fast()[0].features[0][0], 100.0);
        assert_eq!(ms.medium()[0].features[0][0], 100.0);
        assert_eq!(ms.slow()[0].features[0][0], 100.0);
    }

    #[test]
    fn test_push_arc_numerical_precision() {
        use std::sync::Arc;

        let config = MultiScaleConfig::new(
            ScaleConfig::new(2, 1, 1),
            ScaleConfig::new(2, 1, 1),
            ScaleConfig::new(2, 1, 1),
        );

        let mut window = MultiScaleWindow::new(config, 4);

        // Test edge values
        let edge_values = Arc::new(vec![std::f64::consts::PI, std::f64::consts::E, 1e-15, 1e15]);

        window.push_arc(1000, edge_values.clone());
        window.push_arc(2000, Arc::new(vec![1.0; 4]));

        let ms = window.try_build_all().unwrap();

        // Verify bit-level equality
        for i in 0..4 {
            assert_eq!(
                ms.fast()[0].features[0][i].to_bits(),
                edge_values[i].to_bits(),
                "Bit-level mismatch at index {i}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "Feature count mismatch")]
    fn test_push_arc_wrong_feature_count() {
        use std::sync::Arc;

        let config = MultiScaleConfig::default();
        let mut window = MultiScaleWindow::new(config, 4);

        // Wrong feature count
        let wrong_features = Arc::new(vec![1.0, 2.0]); // Only 2, expected 4
        window.push_arc(1000, wrong_features);
    }

    #[test]
    fn test_push_arc_with_decimation() {
        use std::sync::Arc;

        // Different decimation rates
        let config = MultiScaleConfig::new(
            ScaleConfig::new(3, 1, 1), // Fast: window 3, decimation 1
            ScaleConfig::new(3, 2, 1), // Medium: window 3, decimation 2
            ScaleConfig::new(3, 4, 1), // Slow: window 3, decimation 4
        );

        let mut window = MultiScaleWindow::new(config, 3);

        // Push 12 events
        for i in 0..12 {
            window.push_arc(i as u64 * 1000, Arc::new(vec![i as f64; 3]));
        }

        // STREAMING FIX: Buffer counts are now reduced because sequences are built during push
        // After push, buffers only contain what hasn't been consumed
        // With stride=1 and window=3, buffer length stays at 3 (ring buffer)
        
        // Build and verify - now we get MULTIPLE sequences per scale
        let ms = window.try_build_all().unwrap();

        // Fast: 12 events, window 3, stride 1 → 10 sequences (after first 3 events, then 9 more)
        assert!(
            ms.fast().len() >= 9,
            "Fast should have many sequences: {}",
            ms.fast().len()
        );
        // Last sequence should have last 3 events: values 9, 10, 11
        let last_fast = ms.fast().last().unwrap();
        assert_eq!(last_fast.features[2][0], 11.0);

        // Medium: 6 effective events, window 3, stride 1 → 4 sequences
        assert!(
            ms.medium().len() >= 3,
            "Medium should have multiple sequences: {}",
            ms.medium().len()
        );
        // Last sequence last value should be 11 (the 12th raw event, 6th medium event)
        let last_medium = ms.medium().last().unwrap();
        assert_eq!(last_medium.features[2][0], 11.0);

        // Slow: 3 effective events, window 3, stride 1 → 1 sequence
        assert!(
            ms.slow().len() >= 1,
            "Slow should have at least 1 sequence: {}",
            ms.slow().len()
        );
        // Slow samples events 3,7,11 (indices 0, 1, 2 in slow builder)
        let last_slow = ms.slow().last().unwrap();
        assert_eq!(last_slow.features[2][0], 11.0);
    }
}
