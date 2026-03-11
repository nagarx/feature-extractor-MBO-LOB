//! Sampling strategies for LOB feature extraction.
//!
//! This module provides various sampling strategies optimized for high-frequency
//! trading data. The goal is to capture meaningful market events while filtering
//! noise and maintaining computational efficiency.
//!
//! # Sampling Strategies
//!
//! - **Volume-Based**: Sample after a fixed volume has been traded
//!   - Recommended by TLOB paper (500 shares per sample)
//!   - Captures market-moving events regardless of message frequency
//!   - Better signal-to-noise ratio than event-based sampling
//!
//! - **Event-Based**: Sample every N messages (legacy, for comparison)
//!   - Simple but treats all events equally
//!   - May oversample during quiet periods, undersample during active trading
//!
//! # Performance
//!
//! All samplers are designed for zero-allocation hot paths:
//! - `should_sample()` performs O(1) operations with no heap allocations
//! - State updates use only primitive arithmetic
//! - Target throughput: >10M checks/second on modern CPU
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::sampling::VolumeBasedSampler;
//!
//! let mut sampler = VolumeBasedSampler::new(500, 1_000_000); // 500 shares, 1ms min interval
//!
//! for message in messages {
//!     if sampler.should_sample(message.size, message.timestamp) {
//!         // Extract features for this sample point
//!         let features = extractor.extract_features(&lob_state)?;
//!     }
//! }
//! ```

/// Volume-based sampler for LOB data.
///
/// Samples the order book after a predetermined volume of shares has been traded.
/// This approach aligns with the TLOB paper's recommendation and provides better
/// signal quality than fixed-interval sampling.
///
/// # Design Rationale
///
/// 1. **Market Impact**: Large volume trades have more price impact than small ones
/// 2. **Adaptive**: Automatically adjusts to market activity (more samples when active)
/// 3. **Noise Reduction**: Filters out low-volume events that don't move the market
///
/// # Performance
///
/// - Hot path (`should_sample`): ~2-3 CPU cycles, zero allocations
/// - State size: 24 bytes (3 × u64)
/// - Cache-friendly: All state fits in single cache line
#[derive(Debug, Clone)]
pub struct VolumeBasedSampler {
    /// Target volume per sample (in shares)
    ///
    /// Recommended values:
    /// - Liquid stocks (NVDA, TSLA, AAPL): 500-1000 shares
    /// - Less liquid stocks: 100-500 shares
    /// - Very liquid stocks (SPY): 1000-5000 shares
    target_volume: u64,

    /// Accumulated volume since last sample (in shares)
    accumulated_volume: u64,

    /// Minimum time between samples (nanoseconds)
    ///
    /// Prevents over-sampling during high-frequency bursts.
    /// Recommended: 1-10 milliseconds (1_000_000 - 10_000_000 ns)
    min_time_interval_ns: u64,

    /// Timestamp of last sample (nanoseconds since epoch)
    last_sample_time: u64,

    /// Total number of samples generated
    sample_count: u64,

    /// Total volume processed
    total_volume: u64,
}

impl VolumeBasedSampler {
    /// Create a new volume-based sampler.
    ///
    /// # Arguments
    ///
    /// * `target_volume` - Target volume per sample (shares). Recommended: 500 for liquid stocks
    /// * `min_time_interval_ns` - Minimum **nanoseconds** between samples. Recommended: 1_000_000 (1ms)
    ///
    /// # Units
    ///
    /// The `min_time_interval_ns` parameter is in **nanoseconds** to match:
    /// - `SamplingConfig.min_time_interval_ns` (pipeline config)
    /// - MBO message timestamps (nanoseconds since epoch)
    /// - RULE.md §2: "Timestamps must declare unit"
    ///
    /// Common values:
    /// - 1ms = 1_000_000 ns
    /// - 10ms = 10_000_000 ns
    /// - 100ms = 100_000_000 ns
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolumeBasedSampler;
    ///
    /// // Sample every 500 shares, with minimum 1ms (1_000_000 ns) between samples
    /// let sampler = VolumeBasedSampler::new(500, 1_000_000);
    /// ```
    #[inline]
    pub fn new(target_volume: u64, min_time_interval_ns: u64) -> Self {
        Self {
            target_volume,
            accumulated_volume: 0,
            min_time_interval_ns, // No multiplication - already in nanoseconds
            last_sample_time: 0,
            sample_count: 0,
            total_volume: 0,
        }
    }

    /// Check if we should sample at this event.
    ///
    /// Returns `true` when:
    /// 1. Accumulated volume >= target volume, AND
    /// 2. Sufficient time has passed since last sample
    ///
    /// When `true` is returned, the sampler automatically resets its state.
    ///
    /// # Arguments
    ///
    /// * `event_volume` - Volume of the current event (shares)
    /// * `timestamp_ns` - Event timestamp (nanoseconds since epoch)
    ///
    /// # Performance
    ///
    /// This is the hot path - called for every message. Optimized for:
    /// - Zero allocations
    /// - Branch prediction friendly
    /// - ~2-3 CPU cycles on modern processors
    ///
    /// # Example
    ///
    /// ```ignore
    /// if sampler.should_sample(message.size, message.timestamp) {
    ///     // Extract features here
    /// }
    /// ```
    #[inline]
    pub fn should_sample(&mut self, event_volume: u32, timestamp_ns: u64) -> bool {
        // Accumulate volume (always, even if we don't sample)
        self.accumulated_volume += event_volume as u64;
        self.total_volume += event_volume as u64;

        // Check volume threshold
        let volume_condition = self.accumulated_volume >= self.target_volume;

        // Check minimum time interval (skip on first sample)
        let time_condition = if self.last_sample_time == 0 {
            true
        } else {
            timestamp_ns.saturating_sub(self.last_sample_time) >= self.min_time_interval_ns
        };

        // Sample if both conditions met
        if volume_condition && time_condition {
            // Reset accumulated volume
            self.accumulated_volume = 0;

            // Update last sample time
            self.last_sample_time = timestamp_ns;

            // Increment sample count
            self.sample_count += 1;

            true
        } else {
            false
        }
    }

    /// Get the number of samples generated so far.
    #[inline]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get the total volume processed (in shares).
    #[inline]
    pub fn total_volume(&self) -> u64 {
        self.total_volume
    }

    /// Get the current accumulated volume since last sample.
    #[inline]
    pub fn accumulated_volume(&self) -> u64 {
        self.accumulated_volume
    }

    /// Reset the sampler state.
    ///
    /// Useful for starting a new trading day or data segment.
    #[inline]
    pub fn reset(&mut self) {
        self.accumulated_volume = 0;
        self.last_sample_time = 0;
        self.sample_count = 0;
        self.total_volume = 0;
    }

    /// Phase 1: Dynamically update the volume threshold.
    ///
    /// This allows adaptive sampling to adjust the threshold based on market conditions.
    /// The accumulated volume is preserved, so sampling continues smoothly.
    ///
    /// # Arguments
    ///
    /// * `new_threshold` - The new target volume threshold
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::VolumeBasedSampler;
    ///
    /// let mut sampler = VolumeBasedSampler::new(1000, 1_000_000);
    ///
    /// // Adaptive threshold adjusts based on volatility
    /// sampler.set_threshold(1500); // Increase in volatile markets
    /// sampler.set_threshold(500);  // Decrease in quiet markets
    /// ```
    #[inline]
    pub fn set_threshold(&mut self, new_threshold: u64) {
        self.target_volume = new_threshold;
    }

    /// Get statistics about sampling efficiency.
    ///
    /// Returns (sample_count, total_volume, avg_volume_per_sample)
    pub fn statistics(&self) -> (u64, u64, f64) {
        let avg_volume = if self.sample_count > 0 {
            self.total_volume as f64 / self.sample_count as f64
        } else {
            0.0
        };
        (self.sample_count, self.total_volume, avg_volume)
    }
}

/// Event-based sampler (legacy, for comparison).
///
/// Samples every N messages regardless of volume or market impact.
/// Kept for backward compatibility and performance comparison.
///
/// # When to Use
///
/// - Replicating FI-2010 dataset methodology (samples every 10 events)
/// - Benchmarking against volume-based sampling
/// - Fixed-rate sampling requirements
///
/// # Performance
///
/// Faster than volume-based (simpler logic) but lower quality samples.
#[derive(Debug, Clone)]
pub struct EventBasedSampler {
    /// Sample every N events
    sample_interval: u64,

    /// Current event count
    event_count: u64,

    /// Total samples generated
    sample_count: u64,
}

impl EventBasedSampler {
    /// Create a new event-based sampler.
    ///
    /// # Arguments
    ///
    /// * `sample_interval` - Sample every N events. FI-2010 uses 10.
    ///
    /// # Example
    ///
    /// ```
    /// use feature_extractor::preprocessing::EventBasedSampler;
    ///
    /// // Sample every 100 messages (current implementation)
    /// let sampler = EventBasedSampler::new(100);
    /// ```
    #[inline]
    pub fn new(sample_interval: u64) -> Self {
        Self {
            sample_interval,
            event_count: 0,
            sample_count: 0,
        }
    }

    /// Check if we should sample at this event.
    ///
    /// Returns `true` every N events.
    ///
    /// # Performance
    ///
    /// Faster than volume-based (~1 CPU cycle) due to simpler logic.
    #[inline]
    pub fn should_sample(&mut self) -> bool {
        self.event_count += 1;

        if self.event_count >= self.sample_interval {
            self.event_count = 0;
            self.sample_count += 1;
            true
        } else {
            false
        }
    }

    /// Get the number of samples generated.
    #[inline]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Reset the sampler state.
    #[inline]
    pub fn reset(&mut self) {
        self.event_count = 0;
        self.sample_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_based_sampler_basic() {
        // 500 shares, 1ms minimum interval (1_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(500, 1_000_000);

        // No sample on first small event
        assert!(!sampler.should_sample(100, 1_000_000));
        assert_eq!(sampler.accumulated_volume(), 100);
        assert_eq!(sampler.sample_count(), 0);

        // No sample when volume accumulated but not enough
        assert!(!sampler.should_sample(200, 2_000_000));
        assert_eq!(sampler.accumulated_volume(), 300);

        // Sample when volume threshold reached
        assert!(sampler.should_sample(300, 3_000_000));
        assert_eq!(sampler.accumulated_volume(), 0); // Reset after sample
        assert_eq!(sampler.sample_count(), 1);
        assert_eq!(sampler.total_volume(), 600);
    }

    #[test]
    fn test_volume_based_sampler_time_gating() {
        // 100 shares, 10ms minimum interval (10_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(100, 10_000_000);

        let base_time = 1_000_000_000; // 1 second

        // First sample
        assert!(sampler.should_sample(100, base_time));
        assert_eq!(sampler.sample_count(), 1);

        // Try to sample immediately after (volume met, but time not met)
        // Time delta: 1ms < 10ms minimum
        assert!(!sampler.should_sample(100, base_time + 1_000_000));
        assert_eq!(sampler.sample_count(), 1); // Still only 1 sample

        // Sample after sufficient time (11ms later)
        assert!(sampler.should_sample(0, base_time + 11_000_000));
        assert_eq!(sampler.sample_count(), 2);
    }

    #[test]
    fn test_volume_based_sampler_large_single_trade() {
        // 500 shares, 1ms minimum interval (1_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(500, 1_000_000);

        // Single large trade should trigger sample
        assert!(sampler.should_sample(1000, 1_000_000));
        assert_eq!(sampler.sample_count(), 1);
        assert_eq!(sampler.total_volume(), 1000);
    }

    #[test]
    fn test_volume_based_sampler_statistics() {
        // 500 shares, 1ms minimum interval (1_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(500, 1_000_000);

        // Generate several samples (10ms apart, which is > 1ms minimum)
        for i in 0..5 {
            let timestamp = (i + 1) * 10_000_000; // 10ms apart
            sampler.should_sample(100, timestamp);
            sampler.should_sample(400, timestamp + 1_000_000);
        }

        let (count, total_vol, avg_vol) = sampler.statistics();
        assert_eq!(count, 5);
        assert_eq!(total_vol, 2500); // 5 × 500
        assert!((avg_vol - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_volume_based_sampler_reset() {
        // 500 shares, 1ms minimum interval (1_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(500, 1_000_000);

        sampler.should_sample(600, 1_000_000);
        assert_eq!(sampler.sample_count(), 1);

        sampler.reset();
        assert_eq!(sampler.sample_count(), 0);
        assert_eq!(sampler.total_volume(), 0);
        assert_eq!(sampler.accumulated_volume(), 0);
    }

    #[test]
    fn test_event_based_sampler() {
        let mut sampler = EventBasedSampler::new(10);

        // No sample for first 9 events
        for _ in 0..9 {
            assert!(!sampler.should_sample());
        }
        assert_eq!(sampler.sample_count(), 0);

        // Sample on 10th event
        assert!(sampler.should_sample());
        assert_eq!(sampler.sample_count(), 1);

        // Cycle repeats
        for _ in 0..9 {
            assert!(!sampler.should_sample());
        }
        assert!(sampler.should_sample());
        assert_eq!(sampler.sample_count(), 2);
    }

    #[test]
    fn test_event_based_sampler_reset() {
        let mut sampler = EventBasedSampler::new(5);

        for _ in 0..5 {
            sampler.should_sample();
        }
        assert_eq!(sampler.sample_count(), 1);

        sampler.reset();
        assert_eq!(sampler.sample_count(), 0);
    }

    #[test]
    fn test_volume_sampler_zero_volume_events() {
        // 500 shares, 1ms minimum interval (1_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(500, 1_000_000);

        // Zero volume events (e.g., cancels) shouldn't trigger samples alone
        for i in 0..100 {
            assert!(!sampler.should_sample(0, i * 1_000_000));
        }
        assert_eq!(sampler.sample_count(), 0);

        // But accumulated volume + zero volume can trigger if threshold met
        sampler.should_sample(500, 101_000_000);
        assert_eq!(sampler.sample_count(), 1);
    }

    #[test]
    fn test_volume_sampler_timestamp_overflow_safety() {
        // 100 shares, 1ms minimum interval (1_000_000 ns)
        let mut sampler = VolumeBasedSampler::new(100, 1_000_000);

        // First sample at max timestamp
        assert!(sampler.should_sample(100, u64::MAX));

        // Second sample at lower timestamp (time went backwards - shouldn't panic)
        // Uses saturating_sub so this is safe, but won't sample (time condition fails)
        assert!(!sampler.should_sample(100, 0));

        // Third sample at slightly lower than max (still won't sample - not enough time)
        assert!(!sampler.should_sample(100, u64::MAX - 100_000));

        // But if we wait long enough after the backwards time, we can sample again
        // Reset to simulate starting fresh after clock correction
        sampler.reset();
        assert!(sampler.should_sample(100, 1_000_000_000));
    }
}
