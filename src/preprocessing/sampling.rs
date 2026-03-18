//! Sampling strategies for LOB feature extraction.
//!
//! This module provides a trait-based sampling architecture for high-frequency
//! trading data. All sampling strategies implement the [`Sampler`] trait, enabling
//! the pipeline to work with any strategy without modification.
//!
//! # Sampling Strategies
//!
//! - **Event-Based**: Sample every N MBO events (default, FI-2010 compatible)
//! - **Volume-Based**: Sample after N shares traded (TLOB paper, 500 shares)
//! - **Time-Based**: Sample at fixed wall-clock intervals aligned to market open
//!   (matches mbo-statistical-profiler's canonical grid; preserves OFI persistence)
//! - **Composite**: Combine strategies with OR/AND logic for experimentation
//!
//! # Architecture
//!
//! All strategies implement [`Sampler`], a trait with a unified interface:
//! ```ignore
//! trait Sampler: Send {
//!     fn should_sample(&mut self, ctx: &SamplingContext) -> bool;
//!     fn reset(&mut self);
//!     fn metrics(&self) -> SamplerMetrics;
//!     fn strategy_name(&self) -> &'static str;
//! }
//! ```
//!
//! The pipeline calls `should_sample()` on every MBO event. Adding a new strategy
//! requires only implementing the trait — zero changes to the pipeline loop.
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
//! use feature_extractor::preprocessing::sampling::{Sampler, SamplingContext, VolumeBasedSampler};
//!
//! let mut sampler = VolumeBasedSampler::new(500, 1_000_000);
//!
//! for message in messages {
//!     let ctx = SamplingContext {
//!         timestamp_ns: message.timestamp,
//!         event_volume: message.size,
//!     };
//!     if sampler.should_sample(&ctx) {
//!         let features = extractor.extract_features(&lob_state)?;
//!     }
//! }
//! ```

// ============================================================================
// Sampler Trait + Context Types
// ============================================================================

/// Context provided to the sampler on every MBO event.
///
/// Stack-allocated (16 bytes), zero-heap, passed by reference on the hot path.
/// Contains all information any sampling strategy might need.
#[derive(Debug, Clone, Copy)]
pub struct SamplingContext {
    /// MBO message timestamp in UTC nanoseconds since epoch.
    pub timestamp_ns: u64,
    /// Trade volume of this event in shares (0 for non-trade events like cancels).
    pub event_volume: u32,
}

/// Sampling metrics for experiment tracking and export metadata.
///
/// Every sampler reports these metrics via `metrics()`. Useful for:
/// - Export metadata (samples_per_day, avg_interval)
/// - Experiment comparison (different strategies produce different densities)
/// - Diagnostics (detect sampling anomalies)
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplerMetrics {
    /// Total samples emitted since last reset.
    pub sample_count: u64,
    /// Total events observed since last reset.
    pub events_observed: u64,
    /// Total volume observed since last reset (shares).
    pub volume_observed: u64,
    /// Timestamp of last sample (nanoseconds), 0 if none.
    pub last_sample_timestamp_ns: u64,
}

/// Unified sampling interface for the feature extraction pipeline.
///
/// All sampling strategies implement this trait. The pipeline loop calls
/// `should_sample()` on every MBO event. Implementations must be:
/// - **O(1)** time complexity per call
/// - **Zero heap allocations** on the hot path
/// - **Deterministic** given the same input sequence
///
/// # Lifecycle
///
/// 1. Created via constructor or `create_sampler()` in pipeline
/// 2. `should_sample()` called on every MBO event (~1M/sec for NVDA)
/// 3. `reset()` called between trading days (mandatory per Rule 7)
/// 4. `metrics()` queried for experiment tracking and export metadata
///
/// # Adding New Strategies
///
/// Implement this trait. No changes to `pipeline.rs` are needed — the pipeline
/// uses `Box<dyn Sampler>` for dispatch.
pub trait Sampler: Send {
    /// Check whether this event triggers a sample point.
    ///
    /// Called on EVERY MBO event. Must be O(1), zero-alloc.
    /// Returns `true` if the pipeline should extract features at this point.
    /// The sampler internally updates its own state (counters, timestamps)
    /// regardless of whether it returns true.
    fn should_sample(&mut self, ctx: &SamplingContext) -> bool;

    /// Reset all state for a new trading day.
    ///
    /// Clears counters, timestamps, and any accumulated state.
    /// Called by `Pipeline::reset()` between days to prevent cross-day leakage.
    fn reset(&mut self);

    /// Current metrics for experiment tracking and diagnostics.
    fn metrics(&self) -> SamplerMetrics;

    /// Human-readable name of this strategy (for logging and export metadata).
    fn strategy_name(&self) -> &'static str;

    /// Dynamically adjust the primary threshold.
    ///
    /// Used by adaptive sampling to adjust volume thresholds based on volatility.
    /// Default implementation is a no-op (most strategies ignore this).
    fn set_threshold(&mut self, _new_threshold: u64) {}
}

// ============================================================================
// Constants
// ============================================================================

/// Nanoseconds per second.
const NS_PER_SECOND: u64 = 1_000_000_000;

/// Nanoseconds per minute.
const NS_PER_MINUTE: u64 = 60 * NS_PER_SECOND;

/// Nanoseconds per hour.
const NS_PER_HOUR: u64 = 60 * NS_PER_MINUTE;

/// Nanoseconds per day.
const NS_PER_DAY: u64 = 24 * NS_PER_HOUR;

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

// ============================================================================
// Sampler Trait Implementations for Existing Samplers
// ============================================================================

impl Sampler for VolumeBasedSampler {
    #[inline]
    fn should_sample(&mut self, ctx: &SamplingContext) -> bool {
        self.should_sample(ctx.event_volume, ctx.timestamp_ns)
    }

    fn reset(&mut self) {
        VolumeBasedSampler::reset(self);
    }

    fn metrics(&self) -> SamplerMetrics {
        SamplerMetrics {
            sample_count: self.sample_count,
            events_observed: 0, // Volume sampler tracks volume, not events
            volume_observed: self.total_volume,
            last_sample_timestamp_ns: self.last_sample_time,
        }
    }

    fn strategy_name(&self) -> &'static str {
        "volume_based"
    }

    fn set_threshold(&mut self, new_threshold: u64) {
        VolumeBasedSampler::set_threshold(self, new_threshold);
    }
}

impl Sampler for EventBasedSampler {
    #[inline]
    fn should_sample(&mut self, _ctx: &SamplingContext) -> bool {
        EventBasedSampler::should_sample(self)
    }

    fn reset(&mut self) {
        EventBasedSampler::reset(self);
    }

    fn metrics(&self) -> SamplerMetrics {
        SamplerMetrics {
            sample_count: self.sample_count,
            events_observed: self.event_count + self.sample_count * self.sample_interval,
            volume_observed: 0,
            last_sample_timestamp_ns: 0,
        }
    }

    fn strategy_name(&self) -> &'static str {
        "event_based"
    }
}

// ============================================================================
// TimeBasedSampler (NEW — unlocks OFI persistence signal)
// ============================================================================

/// Time-based sampler: emits samples at fixed wall-clock intervals.
///
/// Grid is aligned to 09:30 ET market open, matching the canonical grid
/// used by `mbo-statistical-profiler`'s `resample_to_grid()`.
///
/// # Why This Exists
///
/// The profiler validated that OFI has ACF=0.266 at 5-minute time bins. Our
/// event-based sampling produces ACF=0.021 because non-overlapping event windows
/// are statistically independent. Time-based sampling preserves the temporal
/// structure that makes OFI persistence detectable and exploitable.
///
/// # Grid Alignment
///
/// The grid is aligned to market open (09:30 ET) so that:
/// - Consecutive trading days produce comparable sample points
/// - The profiler and feature extractor use the same time bins
/// - Cross-day analysis is aligned
///
/// # Gap Handling
///
/// If multiple time boundaries pass without events (e.g., market halt),
/// the sampler emits ONE sample on the first event after the gap. This matches
/// the profiler's `Last` aggregation mode.
///
/// # Performance
///
/// Same as other samplers: O(1), zero allocations, ~2 CPU cycles.
///
/// # Example
///
/// ```
/// use feature_extractor::preprocessing::sampling::{TimeBasedSampler, Sampler, SamplingContext};
///
/// // Sample every 5 seconds, EST timezone (UTC-5)
/// let mut sampler = TimeBasedSampler::new(5_000_000_000, -5);
///
/// let ctx = SamplingContext { timestamp_ns: 1_700_000_000_000_000_000, event_volume: 100 };
/// let _ = sampler.should_sample(&ctx);
/// ```
#[derive(Debug, Clone)]
pub struct TimeBasedSampler {
    /// Sampling interval in nanoseconds.
    ///
    /// Common values:
    /// - 1s = 1_000_000_000
    /// - 5s = 5_000_000_000
    /// - 1m = 60_000_000_000
    /// - 5m = 300_000_000_000
    interval_ns: u64,

    /// Next sample boundary (UTC nanoseconds since epoch).
    /// The sampler fires when `timestamp_ns >= next_boundary_ns`.
    next_boundary_ns: u64,

    /// Whether the grid has been initialized from the first event timestamp.
    initialized: bool,

    /// UTC offset for session alignment in hours.
    /// EST = -5, EDT = -4. Used to compute 09:30 ET in UTC.
    utc_offset_hours: i32,

    /// Total samples emitted since last reset.
    sample_count: u64,

    /// Total events observed since last reset.
    events_observed: u64,

    /// Total volume observed since last reset.
    volume_observed: u64,

    /// Timestamp of last sample.
    last_sample_timestamp_ns: u64,
}

impl TimeBasedSampler {
    /// Create a new time-based sampler.
    ///
    /// # Arguments
    ///
    /// * `interval_ns` - Sampling interval in nanoseconds (e.g., 5_000_000_000 for 5s)
    /// * `utc_offset_hours` - UTC offset for market open alignment (EST=-5, EDT=-4)
    ///
    /// # Panics
    ///
    /// Panics if `interval_ns` is 0.
    pub fn new(interval_ns: u64, utc_offset_hours: i32) -> Self {
        assert!(interval_ns > 0, "TimeBasedSampler interval must be > 0");
        Self {
            interval_ns,
            next_boundary_ns: 0,
            initialized: false,
            utc_offset_hours,
            sample_count: 0,
            events_observed: 0,
            volume_observed: 0,
            last_sample_timestamp_ns: 0,
        }
    }

    /// Compute the market open timestamp (09:30 local time) for the day containing `timestamp_ns`.
    ///
    /// Returns the UTC nanosecond timestamp of 09:30 local time.
    fn compute_market_open_ns(&self, timestamp_ns: u64) -> u64 {
        // UTC day boundary
        let day_epoch_ns = timestamp_ns - (timestamp_ns % NS_PER_DAY);

        // 09:30 local time in UTC nanoseconds
        // local 09:30 = UTC (09:30 - offset)
        let open_local_ns = 9 * NS_PER_HOUR + 30 * NS_PER_MINUTE;
        let offset_ns = (self.utc_offset_hours.unsigned_abs() as u64) * NS_PER_HOUR;

        if self.utc_offset_hours < 0 {
            // Negative offset (US timezones): UTC = local - offset → UTC = local + |offset|
            day_epoch_ns + open_local_ns + offset_ns
        } else {
            // Positive offset: UTC = local - offset
            day_epoch_ns + open_local_ns.saturating_sub(offset_ns)
        }
    }

    /// Initialize the grid from the first event's timestamp.
    ///
    /// Aligns `next_boundary_ns` to the canonical grid starting at market open.
    /// The grid is: open + k*interval for k = 1, 2, 3, ...
    ///
    /// The first boundary is always AFTER the open, because each sample captures
    /// the accumulated state over a full interval. This matches the profiler's
    /// `resample_to_grid()` semantics where each bin aggregates data from
    /// [boundary - interval, boundary).
    fn initialize_grid(&mut self, timestamp_ns: u64) {
        let open_ns = self.compute_market_open_ns(timestamp_ns);

        if timestamp_ns < open_ns {
            // Before market open: first boundary at open + interval
            self.next_boundary_ns = open_ns + self.interval_ns;
        } else {
            // At or after market open: align to next grid boundary
            let elapsed = timestamp_ns - open_ns;
            let intervals_passed = elapsed / self.interval_ns;
            self.next_boundary_ns = open_ns + (intervals_passed + 1) * self.interval_ns;
        }
    }

    /// Get the configured interval in nanoseconds.
    pub fn interval_ns(&self) -> u64 {
        self.interval_ns
    }
}

impl Sampler for TimeBasedSampler {
    #[inline]
    fn should_sample(&mut self, ctx: &SamplingContext) -> bool {
        self.events_observed += 1;
        self.volume_observed += ctx.event_volume as u64;

        if !self.initialized {
            self.initialize_grid(ctx.timestamp_ns);
            self.initialized = true;
        }

        if ctx.timestamp_ns >= self.next_boundary_ns {
            // Advance boundary past current time (handles gaps gracefully)
            while self.next_boundary_ns <= ctx.timestamp_ns {
                self.next_boundary_ns += self.interval_ns;
            }
            self.sample_count += 1;
            self.last_sample_timestamp_ns = ctx.timestamp_ns;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.next_boundary_ns = 0;
        self.initialized = false;
        self.sample_count = 0;
        self.events_observed = 0;
        self.volume_observed = 0;
        self.last_sample_timestamp_ns = 0;
    }

    fn metrics(&self) -> SamplerMetrics {
        SamplerMetrics {
            sample_count: self.sample_count,
            events_observed: self.events_observed,
            volume_observed: self.volume_observed,
            last_sample_timestamp_ns: self.last_sample_timestamp_ns,
        }
    }

    fn strategy_name(&self) -> &'static str {
        "time_based"
    }
}

// ============================================================================
// CompositeSampler (OR/AND logic for strategy composition)
// ============================================================================

/// How to combine multiple child samplers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositeMode {
    /// Sample when ANY child triggers (union of sample points).
    Any,
    /// Sample when ALL children have triggered since last sample (intersection).
    All,
}

/// Combines multiple sampling strategies with OR or AND logic.
///
/// # Modes
///
/// - **Any**: Fires when any child sampler fires. Produces the union of all
///   children's sample points. Use for "sample every 5 minutes OR every 10000 shares".
///
/// - **All**: Fires when all children have fired at least once since the last
///   composite sample. Use for "sample only when BOTH time AND volume conditions met".
///
/// # Example
///
/// ```ignore
/// let time = Box::new(TimeBasedSampler::new(300_000_000_000, -5)); // 5 min
/// let volume = Box::new(VolumeBasedSampler::new(10000, 0));
/// let sampler = CompositeSampler::new(vec![time, volume], CompositeMode::Any);
/// ```
pub struct CompositeSampler {
    children: Vec<Box<dyn Sampler>>,
    mode: CompositeMode,
    /// For All mode: tracks which children have triggered since last composite sample.
    triggered: Vec<bool>,
    sample_count: u64,
    events_observed: u64,
    volume_observed: u64,
    last_sample_timestamp_ns: u64,
}

impl CompositeSampler {
    /// Create a new composite sampler.
    ///
    /// # Panics
    ///
    /// Panics if `children` is empty.
    pub fn new(children: Vec<Box<dyn Sampler>>, mode: CompositeMode) -> Self {
        assert!(!children.is_empty(), "CompositeSampler requires at least one child");
        let n = children.len();
        Self {
            children,
            mode,
            triggered: vec![false; n],
            sample_count: 0,
            events_observed: 0,
            volume_observed: 0,
            last_sample_timestamp_ns: 0,
        }
    }
}

impl Sampler for CompositeSampler {
    fn should_sample(&mut self, ctx: &SamplingContext) -> bool {
        self.events_observed += 1;
        self.volume_observed += ctx.event_volume as u64;

        match self.mode {
            CompositeMode::Any => {
                let mut any_fired = false;
                for child in &mut self.children {
                    if child.should_sample(ctx) {
                        any_fired = true;
                    }
                }
                if any_fired {
                    self.sample_count += 1;
                    self.last_sample_timestamp_ns = ctx.timestamp_ns;
                }
                any_fired
            }
            CompositeMode::All => {
                for (i, child) in self.children.iter_mut().enumerate() {
                    if child.should_sample(ctx) {
                        self.triggered[i] = true;
                    }
                }
                if self.triggered.iter().all(|&t| t) {
                    // All children have triggered — emit sample and reset flags
                    self.triggered.fill(false);
                    self.sample_count += 1;
                    self.last_sample_timestamp_ns = ctx.timestamp_ns;
                    true
                } else {
                    false
                }
            }
        }
    }

    fn reset(&mut self) {
        for child in &mut self.children {
            child.reset();
        }
        self.triggered.fill(false);
        self.sample_count = 0;
        self.events_observed = 0;
        self.volume_observed = 0;
        self.last_sample_timestamp_ns = 0;
    }

    fn metrics(&self) -> SamplerMetrics {
        SamplerMetrics {
            sample_count: self.sample_count,
            events_observed: self.events_observed,
            volume_observed: self.volume_observed,
            last_sample_timestamp_ns: self.last_sample_timestamp_ns,
        }
    }

    fn strategy_name(&self) -> &'static str {
        match self.mode {
            CompositeMode::Any => "composite_any",
            CompositeMode::All => "composite_all",
        }
    }
}

// CompositeSampler can't derive Debug because Box<dyn Sampler> doesn't impl Debug
impl std::fmt::Debug for CompositeSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeSampler")
            .field("mode", &self.mode)
            .field("n_children", &self.children.len())
            .field("sample_count", &self.sample_count)
            .finish()
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

    // ====================================================================
    // Sampler trait conformance tests
    // ====================================================================

    /// Helper: create a SamplingContext
    fn ctx(ts: u64, vol: u32) -> SamplingContext {
        SamplingContext {
            timestamp_ns: ts,
            event_volume: vol,
        }
    }

    #[test]
    fn test_volume_sampler_trait_conformance() {
        // Verify trait-based should_sample matches direct call behavior
        let mut direct = VolumeBasedSampler::new(500, 1_000_000);
        let mut trait_obj: Box<dyn Sampler> = Box::new(VolumeBasedSampler::new(500, 1_000_000));

        let events = [
            (100u32, 1_000_000u64),
            (200, 2_000_000),
            (300, 3_000_000),
            (100, 4_000_000),
            (500, 5_000_000),
        ];

        for &(vol, ts) in &events {
            let direct_result = direct.should_sample(vol, ts);
            let trait_result = trait_obj.should_sample(&ctx(ts, vol));
            assert_eq!(
                direct_result, trait_result,
                "Mismatch at vol={vol}, ts={ts}: direct={direct_result}, trait={trait_result}"
            );
        }

        assert_eq!(trait_obj.strategy_name(), "volume_based");
    }

    #[test]
    fn test_event_sampler_trait_conformance() {
        let mut direct = EventBasedSampler::new(5);
        let mut trait_obj: Box<dyn Sampler> = Box::new(EventBasedSampler::new(5));

        for i in 0..20 {
            let direct_result = direct.should_sample();
            let trait_result = trait_obj.should_sample(&ctx(i * 1_000_000, 0));
            assert_eq!(
                direct_result, trait_result,
                "Mismatch at event {i}: direct={direct_result}, trait={trait_result}"
            );
        }

        assert_eq!(trait_obj.strategy_name(), "event_based");
    }

    #[test]
    fn test_trait_reset_clears_metrics() {
        let mut sampler: Box<dyn Sampler> = Box::new(EventBasedSampler::new(2));

        // Generate some samples
        sampler.should_sample(&ctx(1_000_000, 100));
        sampler.should_sample(&ctx(2_000_000, 200));

        let m = sampler.metrics();
        assert_eq!(m.sample_count, 1, "Should have 1 sample after 2 events with interval=2");

        sampler.reset();

        let m = sampler.metrics();
        assert_eq!(m.sample_count, 0, "sample_count must be 0 after reset");
    }

    #[test]
    fn test_trait_set_threshold_noop_for_event() {
        let mut sampler: Box<dyn Sampler> = Box::new(EventBasedSampler::new(5));

        // set_threshold is a no-op for event-based — should not panic
        sampler.set_threshold(9999);

        // Behavior unchanged
        for _ in 0..4 {
            assert!(!sampler.should_sample(&ctx(0, 0)));
        }
        assert!(sampler.should_sample(&ctx(0, 0)));
    }

    #[test]
    fn test_trait_set_threshold_affects_volume() {
        let mut sampler: Box<dyn Sampler> = Box::new(VolumeBasedSampler::new(100, 0));

        // Increase threshold mid-stream
        sampler.set_threshold(500);

        // 100 shares shouldn't trigger anymore
        assert!(!sampler.should_sample(&ctx(1_000_000, 100)));
        assert!(!sampler.should_sample(&ctx(2_000_000, 100)));
        assert!(!sampler.should_sample(&ctx(3_000_000, 100)));
        assert!(!sampler.should_sample(&ctx(4_000_000, 100)));
        // 5th 100-share event = 500 total → triggers
        assert!(sampler.should_sample(&ctx(5_000_000, 100)));
    }

    // ====================================================================
    // TimeBasedSampler tests
    // ====================================================================

    /// Helper: compute a timestamp for a given UTC time on 2025-02-03.
    /// Hours/minutes are in UTC.
    fn utc_time_ns(hours: u64, minutes: u64, seconds: u64) -> u64 {
        // 2025-02-03 is day 20122 from epoch (1970-01-01)
        // = 19 * 365 * 86400 + ... = complicated, just use a known epoch
        // 2025-02-03 00:00:00 UTC = 1738540800 seconds from epoch
        let day_base_s: u64 = 1_738_540_800;
        let time_s = hours * 3600 + minutes * 60 + seconds;
        (day_base_s + time_s) * NS_PER_SECOND
    }

    #[test]
    fn test_time_sampler_basic_interval() {
        // 5-second interval, EST (UTC-5)
        let mut sampler = TimeBasedSampler::new(5 * NS_PER_SECOND, -5);

        // 09:30:00 ET = 14:30:00 UTC on 2025-02-03
        let t_1430_00 = utc_time_ns(14, 30, 0);
        let t_1430_03 = utc_time_ns(14, 30, 3);
        let t_1430_05 = utc_time_ns(14, 30, 5);
        let t_1430_07 = utc_time_ns(14, 30, 7);
        let t_1430_10 = utc_time_ns(14, 30, 10);

        // First event at market open: initializes grid.
        // next_boundary = open + 1*5s = 14:30:05 (event is AT open, so next boundary is +5s)
        assert!(!sampler.should_sample(&ctx(t_1430_00, 100)));

        // 3s later: still before first boundary (14:30:05)
        assert!(!sampler.should_sample(&ctx(t_1430_03, 100)));

        // At 14:30:05: crosses first boundary → sample!
        assert!(sampler.should_sample(&ctx(t_1430_05, 100)));
        assert_eq!(sampler.metrics().sample_count, 1);

        // 7s mark: before second boundary (14:30:10)
        assert!(!sampler.should_sample(&ctx(t_1430_07, 100)));

        // 10s mark: crosses second boundary → sample!
        assert!(sampler.should_sample(&ctx(t_1430_10, 100)));
        assert_eq!(sampler.metrics().sample_count, 2);
    }

    #[test]
    fn test_time_sampler_grid_alignment_to_market_open() {
        // 1-minute interval, EST (UTC-5)
        let mut sampler = TimeBasedSampler::new(NS_PER_MINUTE, -5);

        // Market open: 09:30:00 ET = 14:30:00 UTC
        // First event at 14:30:30 (30s after open)
        // Grid: open + 1min = 14:31:00 is next boundary
        let t_first = utc_time_ns(14, 30, 30);
        assert!(!sampler.should_sample(&ctx(t_first, 100)));

        // 14:30:59 — still before boundary
        assert!(!sampler.should_sample(&ctx(utc_time_ns(14, 30, 59), 100)));

        // 14:31:00 — crosses boundary
        assert!(sampler.should_sample(&ctx(utc_time_ns(14, 31, 0), 100)));

        // 14:32:00 — next boundary
        assert!(sampler.should_sample(&ctx(utc_time_ns(14, 32, 0), 100)));

        assert_eq!(sampler.metrics().sample_count, 2);
    }

    #[test]
    fn test_time_sampler_first_event_after_open() {
        // 5-minute interval, EST
        let mut sampler = TimeBasedSampler::new(5 * NS_PER_MINUTE, -5);

        // First data arrives at 09:33:00 ET = 14:33:00 UTC (3 min after open)
        // Grid: open + 5min = 14:35:00 is next boundary
        let t_first = utc_time_ns(14, 33, 0);
        assert!(!sampler.should_sample(&ctx(t_first, 100)));

        // 14:34:59 — no sample
        assert!(!sampler.should_sample(&ctx(utc_time_ns(14, 34, 59), 100)));

        // 14:35:00 — first sample
        assert!(sampler.should_sample(&ctx(utc_time_ns(14, 35, 0), 100)));
        assert_eq!(sampler.metrics().sample_count, 1);
    }

    #[test]
    fn test_time_sampler_gap_handling() {
        // 5-minute interval: boundaries at :35, :40, :45, :50, ...
        let mut sampler = TimeBasedSampler::new(5 * NS_PER_MINUTE, -5);

        // First event at 14:30:00 (market open)
        sampler.should_sample(&ctx(utc_time_ns(14, 30, 0), 100));

        // Normal sample at 14:35:00
        assert!(sampler.should_sample(&ctx(utc_time_ns(14, 35, 0), 100)));
        assert_eq!(sampler.metrics().sample_count, 1);

        // GAP: next event at 14:55:00 (20 minutes later — 4 boundaries skipped)
        // Should emit ONE sample, not 4
        assert!(sampler.should_sample(&ctx(utc_time_ns(14, 55, 0), 100)));
        assert_eq!(
            sampler.metrics().sample_count, 2,
            "Gap should produce exactly 1 sample, not one per missed boundary"
        );
    }

    #[test]
    fn test_time_sampler_edt_offset() {
        // EDT = UTC-4 (summer time)
        let mut sampler = TimeBasedSampler::new(NS_PER_MINUTE, -4);

        // 09:30:00 ET during EDT = 13:30:00 UTC
        let t_1330 = utc_time_ns(13, 30, 0);
        assert!(!sampler.should_sample(&ctx(t_1330, 100)));

        // 13:31:00 UTC = 09:31:00 EDT → sample
        assert!(sampler.should_sample(&ctx(utc_time_ns(13, 31, 0), 100)));
        assert_eq!(sampler.metrics().sample_count, 1);
    }

    #[test]
    fn test_time_sampler_pre_market() {
        // Pre-market data: first boundary is at open + interval (bin-end semantics).
        // 1-minute interval, EST.
        let mut sampler = TimeBasedSampler::new(NS_PER_MINUTE, -5);

        // Pre-market: 09:00:00 ET = 14:00:00 UTC — no sample
        let t_pre = utc_time_ns(14, 0, 0);
        assert!(!sampler.should_sample(&ctx(t_pre, 100)));

        // Still pre-market: 14:29:59 — no sample
        assert!(!sampler.should_sample(&ctx(utc_time_ns(14, 29, 59), 100)));

        // Market open: 14:30:00 — no sample yet (first bin hasn't elapsed)
        assert!(!sampler.should_sample(&ctx(utc_time_ns(14, 30, 0), 100)));

        // 14:31:00: first full minute after open → sample
        assert!(sampler.should_sample(&ctx(utc_time_ns(14, 31, 0), 100)));
        assert_eq!(sampler.metrics().sample_count, 1);
    }

    #[test]
    fn test_time_sampler_reset() {
        let mut sampler = TimeBasedSampler::new(NS_PER_SECOND, -5);

        // Generate some samples
        sampler.should_sample(&ctx(utc_time_ns(14, 30, 0), 100));
        sampler.should_sample(&ctx(utc_time_ns(14, 30, 1), 100));
        assert_eq!(sampler.metrics().sample_count, 1);

        // Reset
        sampler.reset();

        let m = sampler.metrics();
        assert_eq!(m.sample_count, 0);
        assert_eq!(m.events_observed, 0);
        assert_eq!(m.volume_observed, 0);
        assert_eq!(m.last_sample_timestamp_ns, 0);

        // After reset, grid re-initializes from next event
        // Use a different day's timestamp to verify no leakage
        let new_day = utc_time_ns(14, 30, 0) + NS_PER_DAY;
        sampler.should_sample(&ctx(new_day, 100));
        sampler.should_sample(&ctx(new_day + NS_PER_SECOND, 100));
        assert_eq!(sampler.metrics().sample_count, 1);
    }

    #[test]
    fn test_time_sampler_metrics() {
        let mut sampler = TimeBasedSampler::new(NS_PER_SECOND, -5);

        let t0 = utc_time_ns(14, 30, 0);
        sampler.should_sample(&ctx(t0, 100));
        sampler.should_sample(&ctx(t0 + 500_000_000, 200)); // 0.5s
        let t1 = t0 + NS_PER_SECOND;
        sampler.should_sample(&ctx(t1, 300)); // triggers sample

        let m = sampler.metrics();
        assert_eq!(m.sample_count, 1);
        assert_eq!(m.events_observed, 3);
        assert_eq!(m.volume_observed, 600);
        assert_eq!(m.last_sample_timestamp_ns, t1);
    }

    #[test]
    fn test_time_sampler_strategy_name() {
        let sampler = TimeBasedSampler::new(NS_PER_SECOND, -5);
        assert_eq!(sampler.strategy_name(), "time_based");
    }

    #[test]
    #[should_panic(expected = "interval must be > 0")]
    fn test_time_sampler_zero_interval_panics() {
        TimeBasedSampler::new(0, -5);
    }

    #[test]
    fn test_time_sampler_rapid_events_between_boundaries() {
        // 1-second interval. Send 100 events in 0.5 seconds.
        // Only the event crossing the boundary should trigger.
        let mut sampler = TimeBasedSampler::new(NS_PER_SECOND, -5);
        let base = utc_time_ns(14, 30, 0);

        // Init
        sampler.should_sample(&ctx(base, 100));

        // 100 events at 5ms intervals within the first second
        let mut triggered = 0;
        for i in 0..100 {
            let ts = base + (i + 1) * 5_000_000; // 5ms apart
            if sampler.should_sample(&ctx(ts, 10)) {
                triggered += 1;
            }
        }
        // 100 * 5ms = 500ms. No events cross the 1s boundary.
        assert_eq!(triggered, 0, "No events in first 500ms should trigger a 1s boundary");

        // Event at 1.001s → triggers
        assert!(sampler.should_sample(&ctx(base + NS_PER_SECOND + 1_000_000, 10)));
    }

    // ====================================================================
    // CompositeSampler tests
    // ====================================================================

    #[test]
    fn test_composite_any_mode() {
        // Time: 1-second intervals. Event: every 5 events.
        let time = Box::new(TimeBasedSampler::new(NS_PER_SECOND, -5));
        let event = Box::new(EventBasedSampler::new(5));
        let mut sampler = CompositeSampler::new(vec![time, event], CompositeMode::Any);

        let base = utc_time_ns(14, 30, 0);

        // Init event (neither fires)
        assert!(!sampler.should_sample(&ctx(base, 100)));

        // Events 2-4 at 100ms intervals (event sampler not yet at 5)
        for i in 1..4 {
            assert!(!sampler.should_sample(&ctx(base + i * 100_000_000, 100)));
        }

        // Event 5: event sampler fires (5 total events)
        assert!(sampler.should_sample(&ctx(base + 400_000_000, 100)));
        assert_eq!(sampler.metrics().sample_count, 1);

        // Event 6 at 1.0s: time sampler fires (1s boundary)
        assert!(sampler.should_sample(&ctx(base + NS_PER_SECOND, 100)));
        assert_eq!(sampler.metrics().sample_count, 2);
    }

    #[test]
    fn test_composite_all_mode() {
        // Both time AND event must trigger before composite fires
        let time = Box::new(TimeBasedSampler::new(NS_PER_SECOND, -5));
        let event = Box::new(EventBasedSampler::new(3));
        let mut sampler = CompositeSampler::new(vec![time, event], CompositeMode::All);

        let base = utc_time_ns(14, 30, 0);

        // Event 1: neither fires
        assert!(!sampler.should_sample(&ctx(base, 100)));

        // Event 2: neither fires
        assert!(!sampler.should_sample(&ctx(base + 100_000_000, 100)));

        // Event 3: event sampler fires (3rd event), but time hasn't fired yet → no composite
        assert!(!sampler.should_sample(&ctx(base + 200_000_000, 100)));

        // Event 4 at 1.0s: time fires. Now both have triggered → composite fires!
        assert!(sampler.should_sample(&ctx(base + NS_PER_SECOND, 100)));
        assert_eq!(sampler.metrics().sample_count, 1);

        // After composite fires, triggered flags reset.
        // Event 5: neither has triggered yet again
        assert!(!sampler.should_sample(&ctx(base + NS_PER_SECOND + 100_000_000, 100)));
    }

    #[test]
    fn test_composite_reset() {
        let time = Box::new(TimeBasedSampler::new(NS_PER_SECOND, -5));
        let event = Box::new(EventBasedSampler::new(2));
        let mut sampler = CompositeSampler::new(vec![time, event], CompositeMode::Any);

        let base = utc_time_ns(14, 30, 0);

        sampler.should_sample(&ctx(base, 100));
        sampler.should_sample(&ctx(base + NS_PER_SECOND, 100));

        sampler.reset();

        let m = sampler.metrics();
        assert_eq!(m.sample_count, 0);
        assert_eq!(m.events_observed, 0);
        assert_eq!(m.volume_observed, 0);
    }

    #[test]
    fn test_composite_strategy_name() {
        let any_sampler = CompositeSampler::new(
            vec![Box::new(EventBasedSampler::new(10))],
            CompositeMode::Any,
        );
        assert_eq!(any_sampler.strategy_name(), "composite_any");

        let all_sampler = CompositeSampler::new(
            vec![Box::new(EventBasedSampler::new(10))],
            CompositeMode::All,
        );
        assert_eq!(all_sampler.strategy_name(), "composite_all");
    }

    #[test]
    #[should_panic(expected = "requires at least one child")]
    fn test_composite_empty_panics() {
        CompositeSampler::new(vec![], CompositeMode::Any);
    }
}
