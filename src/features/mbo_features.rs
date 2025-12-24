//! MBO feature extraction for institutional pattern detection.
//!
//! This module extracts 36 MBO-derived features designed to capture:
//! - Order flow dynamics and market pressure
//! - Institutional trading patterns (whale detection)
//! - Queue position and execution priority
//! - Order lifecycle and modification behavior
//!
//! # Architecture
//!
//! The implementation uses three key components:
//! - `MboWindow`: Rolling buffer with O(1) incremental statistics
//! - `OrderInfo`: Lifecycle tracking for individual orders
//! - `MboAggregator`: Main orchestrator combining windows and trackers
//!
//! # Performance
//!
//! - Event processing: <50 ns (maintains 1.6M+ msg/s throughput)
//! - Feature extraction: <100 ns per call
//! - Memory: ~2.6 MB per symbol (constant, no leaks)
//!
//! # Feature Categories (36 total)
//!
//! 1. **Order Flow Statistics (12)**: Capture buying/selling pressure
//!    - Add/cancel/trade rates by side
//!    - Net order flow and imbalances
//!    - Aggressive order ratios
//!
//! 2. **Size Distribution (8)**: Detect large institutional orders
//!    - Size percentiles (p25, p50, p75, p90)
//!    - Z-scores and skewness
//!    - Large order concentration
//!
//! 3. **Queue Position & Depth (6)**: Track execution priority
//!    - Average queue position
//!    - Volume ahead in queue
//!    - Level concentration metrics
//!
//! 4. **Institutional Detection (4)**: Identify whale patterns
//!    - Large order frequency and imbalance
//!    - Modification patterns
//!    - Iceberg order detection
//!
//! 5. **Core MBO Metrics (6)**: Order lifecycle characteristics
//!    - Average order age
//!    - Fill ratios and time to fill
//!    - Cancel-to-add ratio
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::features::mbo_features::MboAggregator;
//!
//! let mut aggregator = MboAggregator::new();
//!
//! // Process events
//! for msg in messages {
//!     aggregator.process_event(MboEvent::from_message(&msg));
//! }
//!
//! // Extract features (every N events or on demand)
//! let features = aggregator.extract_features(&lob_state);
//! assert_eq!(features.len(), 36);
//! ```

use ahash::AHashMap;
use mbo_lob_reconstructor::lob::queue_position::{QueuePositionConfig, QueuePositionTracker};
use mbo_lob_reconstructor::{Action, LobState, MboMessage, Side};
use std::collections::VecDeque;

// ============================================================================
// Core Data Structures
// ============================================================================

/// MBO event for aggregation.
///
/// Lightweight representation of an order book event for efficient storage
/// in rolling windows.
#[derive(Debug, Clone)]
pub struct MboEvent {
    /// Event timestamp (nanoseconds since epoch)
    pub timestamp: u64,

    /// Order action (Add, Modify, Cancel, Trade)
    pub action: Action,

    /// Order side (Bid or Ask)
    pub side: Side,

    /// Order price (fixed-point: divide by 1e9 for dollars)
    pub price: i64,

    /// Order size (shares)
    pub size: u32,

    /// Unique order identifier
    pub order_id: u64,
}

impl MboEvent {
    /// Create a new MBO event from message components.
    #[inline]
    pub fn new(
        timestamp: u64,
        action: Action,
        side: Side,
        price: i64,
        size: u32,
        order_id: u64,
    ) -> Self {
        Self {
            timestamp,
            action,
            side,
            price,
            size,
            order_id,
        }
    }

    /// Create an MBO event from an MboMessage.
    ///
    /// This is a convenience method for converting from the reconstructor's
    /// message format to the feature extractor's event format.
    ///
    /// # Arguments
    ///
    /// * `msg` - The MBO message from the reconstructor
    ///
    /// # Returns
    ///
    /// A new MboEvent with the same data
    #[inline]
    pub fn from_mbo_message(msg: &mbo_lob_reconstructor::MboMessage) -> Self {
        Self {
            timestamp: msg.timestamp.unwrap_or(0) as u64,
            action: msg.action,
            side: msg.side,
            price: msg.price,
            size: msg.size,
            order_id: msg.order_id,
        }
    }

    /// Convert to an MboMessage for use with the reconstructor's trackers.
    ///
    /// This enables integration with `QueuePositionTracker` and other
    /// reconstructor components that require `MboMessage` input.
    #[inline]
    pub fn to_mbo_message(&self) -> MboMessage {
        MboMessage {
            order_id: self.order_id,
            action: self.action,
            side: self.side,
            price: self.price,
            size: self.size,
            timestamp: Some(self.timestamp as i64),
        }
    }
}

/// Rolling window of MBO events with incremental statistics.
///
/// Maintains a fixed-size sliding window of recent events and incrementally
/// updates counters as events enter/leave the window. This enables O(1)
/// amortized performance for most operations.
///
/// # Performance Characteristics
///
/// - Push: O(1) amortized
/// - Counter access: O(1)
/// - Percentile recompute: O(n log n) but done infrequently
/// - Memory: O(capacity) = ~100 KB for typical window sizes
pub struct MboWindow {
    /// Circular buffer of events (oldest at front, newest at back)
    events: VecDeque<MboEvent>,

    /// Maximum number of events to store
    capacity: usize,

    // ---- Incremental Counters (O(1) updates) ----
    /// Count of Add orders on bid side
    add_count_bid: usize,

    /// Count of Add orders on ask side
    add_count_ask: usize,

    /// Count of Cancel orders on bid side
    cancel_count_bid: usize,

    /// Count of Cancel orders on ask side
    cancel_count_ask: usize,

    /// Count of Trade executions on bid side (buys)
    trade_count_bid: usize,

    /// Count of Trade executions on ask side (sells)
    trade_count_ask: usize,

    /// Total volume (shares) on bid side
    total_volume_bid: u64,

    /// Total volume (shares) on ask side
    total_volume_ask: u64,

    // ---- Timestamps ----
    /// Timestamp of first event in window
    first_ts: u64,

    /// Timestamp of last event in window
    last_ts: u64,

    // ---- Cached Statistics ----
    /// Cached size percentiles [p25, p50, p75, p90]
    size_percentiles: [f64; 4],

    /// Flag indicating percentiles need recomputation
    size_cache_dirty: bool,

    /// Cached mean size
    size_mean: f64,

    /// Cached size standard deviation
    size_std: f64,

    /// Flag indicating size stats need recomputation
    size_stats_dirty: bool,
}

impl MboWindow {
    /// Create a new rolling window with specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of events to store (e.g., 1000)
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            add_count_bid: 0,
            add_count_ask: 0,
            cancel_count_bid: 0,
            cancel_count_ask: 0,
            trade_count_bid: 0,
            trade_count_ask: 0,
            total_volume_bid: 0,
            total_volume_ask: 0,
            first_ts: 0,
            last_ts: 0,
            size_percentiles: [0.0; 4],
            size_cache_dirty: true,
            size_mean: 0.0,
            size_std: 0.0,
            size_stats_dirty: true,
        }
    }

    /// Add an event to the window (O(1) amortized).
    ///
    /// If the window is at capacity, the oldest event is evicted and its
    /// counters are decremented before the new event is added.
    #[inline]
    pub fn push(&mut self, event: MboEvent) {
        // Evict oldest if at capacity
        if self.events.len() == self.capacity {
            let old = self.events.pop_front().unwrap();
            self.decrement_counters(&old);
        }

        // Update timestamps
        if self.events.is_empty() {
            self.first_ts = event.timestamp;
        }
        self.last_ts = event.timestamp;

        // Increment counters for new event
        self.increment_counters(&event);

        // Add to window
        self.events.push_back(event);

        // Mark caches as dirty
        self.size_cache_dirty = true;
        self.size_stats_dirty = true;
    }

    /// Increment counters for a new event (O(1)).
    #[inline]
    fn increment_counters(&mut self, event: &MboEvent) {
        match (&event.action, &event.side) {
            (Action::Add, Side::Bid) => {
                self.add_count_bid += 1;
                self.total_volume_bid += event.size as u64;
            }
            (Action::Add, Side::Ask) => {
                self.add_count_ask += 1;
                self.total_volume_ask += event.size as u64;
            }
            (Action::Cancel, Side::Bid) => {
                self.cancel_count_bid += 1;
            }
            (Action::Cancel, Side::Ask) => {
                self.cancel_count_ask += 1;
            }
            (Action::Trade, Side::Bid) => {
                self.trade_count_bid += 1;
            }
            (Action::Trade, Side::Ask) => {
                self.trade_count_ask += 1;
            }
            _ => {}
        }
    }

    /// Decrement counters for an evicted event (O(1)).
    #[inline]
    fn decrement_counters(&mut self, event: &MboEvent) {
        match (&event.action, &event.side) {
            (Action::Add, Side::Bid) => {
                self.add_count_bid = self.add_count_bid.saturating_sub(1);
                self.total_volume_bid = self.total_volume_bid.saturating_sub(event.size as u64);
            }
            (Action::Add, Side::Ask) => {
                self.add_count_ask = self.add_count_ask.saturating_sub(1);
                self.total_volume_ask = self.total_volume_ask.saturating_sub(event.size as u64);
            }
            (Action::Cancel, Side::Bid) => {
                self.cancel_count_bid = self.cancel_count_bid.saturating_sub(1);
            }
            (Action::Cancel, Side::Ask) => {
                self.cancel_count_ask = self.cancel_count_ask.saturating_sub(1);
            }
            (Action::Trade, Side::Bid) => {
                self.trade_count_bid = self.trade_count_bid.saturating_sub(1);
            }
            (Action::Trade, Side::Ask) => {
                self.trade_count_ask = self.trade_count_ask.saturating_sub(1);
            }
            _ => {}
        }
    }

    /// Get window duration in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        ((self.last_ts - self.first_ts) as f64 / 1e9).max(0.001)
    }

    /// Get number of events in window.
    #[inline]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Recompute size percentiles (O(n log n), called infrequently).
    fn recompute_size_percentiles(&mut self) {
        if self.events.is_empty() {
            self.size_percentiles = [0.0; 4];
            self.size_cache_dirty = false;
            return;
        }

        let mut sizes: Vec<u32> = self.events.iter().map(|e| e.size).collect();
        sizes.sort_unstable();

        let len = sizes.len();
        self.size_percentiles[0] = sizes[len / 4] as f64; // p25
        self.size_percentiles[1] = sizes[len / 2] as f64; // p50
        self.size_percentiles[2] = sizes[len * 3 / 4] as f64; // p75
        self.size_percentiles[3] = sizes[len * 9 / 10] as f64; // p90

        self.size_cache_dirty = false;
    }

    /// Recompute size mean and standard deviation (O(n), called infrequently).
    fn recompute_size_stats(&mut self) {
        if self.events.is_empty() {
            self.size_mean = 0.0;
            self.size_std = 0.0;
            self.size_stats_dirty = false;
            return;
        }

        // Compute mean
        let sum: u64 = self.events.iter().map(|e| e.size as u64).sum();
        self.size_mean = sum as f64 / self.events.len() as f64;

        // Compute std dev
        let variance: f64 = self
            .events
            .iter()
            .map(|e| {
                let diff = e.size as f64 - self.size_mean;
                diff * diff
            })
            .sum::<f64>()
            / self.events.len() as f64;

        self.size_std = variance.sqrt();
        self.size_stats_dirty = false;
    }

    /// Get size percentile (lazy evaluation).
    #[inline]
    pub fn size_percentile(&mut self, index: usize) -> f64 {
        if self.size_cache_dirty {
            self.recompute_size_percentiles();
        }
        self.size_percentiles[index]
    }

    /// Get mean order size (lazy evaluation).
    #[inline]
    pub fn size_mean(&mut self) -> f64 {
        if self.size_stats_dirty {
            self.recompute_size_stats();
        }
        self.size_mean
    }

    /// Get order size standard deviation (lazy evaluation).
    #[inline]
    pub fn size_std(&mut self) -> f64 {
        if self.size_stats_dirty {
            self.recompute_size_stats();
        }
        self.size_std
    }
}

/// Order lifecycle information for tracking individual orders.
///
/// Tracks an order from creation to completion, recording modifications,
/// fills, and other lifecycle events. Used for computing institutional
/// pattern features.
#[derive(Debug, Clone)]
pub struct OrderInfo {
    /// Timestamp when order was created (nanoseconds)
    pub creation_time: u64,

    /// Original order size when created
    pub original_size: u32,

    /// Current remaining size (decreases with fills)
    pub current_size: u32,

    /// Order price (fixed-point)
    pub price: i64,

    /// Order side (Bid or Ask)
    pub side: Side,

    /// Number of times order was modified
    pub modifications: u8,

    /// Number of times order price changed
    pub price_changes: u8,

    /// Fill events: (timestamp, filled_size)
    pub fills: Vec<(u64, u32)>,
}

impl OrderInfo {
    /// Create new order info.
    pub fn new(creation_time: u64, size: u32, price: i64, side: Side) -> Self {
        Self {
            creation_time,
            original_size: size,
            current_size: size,
            price,
            side,
            modifications: 0,
            price_changes: 0,
            fills: Vec::new(),
        }
    }

    /// Record a fill event.
    #[inline]
    pub fn add_fill(&mut self, timestamp: u64, size: u32) {
        self.current_size = self.current_size.saturating_sub(size);
        self.fills.push((timestamp, size));
    }

    /// Record a modification.
    #[inline]
    pub fn add_modification(&mut self, new_price: i64) {
        self.modifications = self.modifications.saturating_add(1);
        if new_price != self.price {
            self.price_changes = self.price_changes.saturating_add(1);
            self.price = new_price;
        }
    }

    /// Get fill ratio (filled_size / original_size).
    #[inline]
    pub fn fill_ratio(&self) -> f64 {
        let filled = self.original_size - self.current_size;
        filled as f64 / self.original_size.max(1) as f64
    }

    /// Get time to first fill (seconds), or None if not filled.
    #[inline]
    pub fn time_to_first_fill(&self) -> Option<f64> {
        self.fills
            .first()
            .map(|(ts, _)| (*ts - self.creation_time) as f64 / 1e9)
    }

    /// Get order age (seconds since creation).
    #[inline]
    pub fn age(&self, current_ts: u64) -> f64 {
        (current_ts - self.creation_time) as f64 / 1e9
    }
}

/// Main MBO feature aggregator.
///
/// Combines multiple rolling windows and order tracking to extract
/// comprehensive MBO features. Uses multi-timescale windows to capture
/// patterns at different frequencies:
/// - Fast window (100 msgs ~2s): Capture immediate flow changes
/// - Medium window (1000 msgs ~20s): Primary feature extraction
/// - Slow window (5000 msgs ~100s): Long-term trends
///
/// # Memory Usage
///
/// - Fast window: ~100 KB
/// - Medium window: ~1 MB
/// - Slow window: ~5 MB
/// - Order tracker: ~2 MB (for ~22K active orders)
/// - Queue tracker (optional): ~1 MB
/// - **Total: ~8-9 MB per symbol**
pub struct MboAggregator {
    /// Fast window for short-term signals (100 messages)
    fast_window: MboWindow,

    /// Medium window for primary features (1000 messages)
    medium_window: MboWindow,

    /// Slow window for long-term trends (5000 messages)
    slow_window: MboWindow,

    /// Order lifecycle tracker
    order_tracker: AHashMap<u64, OrderInfo>,

    /// Optional queue position tracker for precise queue metrics.
    ///
    /// When enabled, provides accurate queue position and volume-ahead
    /// features using FIFO tracking. Adds ~1 MB memory overhead.
    queue_tracker: Option<QueuePositionTracker>,

    /// Total message count processed
    message_count: u64,

    /// Tick size in nanodollars for depth calculations.
    ///
    /// Default: 10_000_000 ($0.01 for US equities)
    /// 
    /// This is used by `depth_ticks_bid` and `depth_ticks_ask` to convert
    /// price distances into tick units. Different instruments have different
    /// tick sizes:
    /// - US equities: $0.01 = 10_000_000 nanodollars
    /// - Crypto: $0.001 = 1_000_000 nanodollars  
    /// - Forex: $0.0001 = 100_000 nanodollars
    /// - Some futures: $1.00 = 1_000_000_000 nanodollars
    tick_size_nanodollars: i64,
}

/// Default tick size in nanodollars: $0.01 for US equities
const DEFAULT_TICK_SIZE_NANODOLLARS: i64 = 10_000_000;

impl MboAggregator {
    /// Create a new MBO aggregator with default window sizes.
    ///
    /// Default tick size is $0.01 (US equity standard).
    /// Use `with_tick_size()` to set a different tick size for other instruments.
    /// Queue position tracking is disabled by default for performance.
    /// Use `with_queue_tracking()` to enable it.
    pub fn new() -> Self {
        Self {
            fast_window: MboWindow::new(100),
            medium_window: MboWindow::new(1000),
            slow_window: MboWindow::new(5000),
            order_tracker: AHashMap::new(),
            queue_tracker: None,
            message_count: 0,
            tick_size_nanodollars: DEFAULT_TICK_SIZE_NANODOLLARS,
        }
    }

    /// Create aggregator with custom window sizes.
    ///
    /// Default tick size is $0.01. Use `with_tick_size()` to customize.
    pub fn with_windows(fast_size: usize, medium_size: usize, slow_size: usize) -> Self {
        Self {
            fast_window: MboWindow::new(fast_size),
            medium_window: MboWindow::new(medium_size),
            slow_window: MboWindow::new(slow_size),
            order_tracker: AHashMap::new(),
            queue_tracker: None,
            message_count: 0,
            tick_size_nanodollars: DEFAULT_TICK_SIZE_NANODOLLARS,
        }
    }
    
    /// Set the tick size for depth calculations.
    ///
    /// The tick size is specified in **dollars** and is converted to nanodollars
    /// internally. This affects `depth_ticks_bid` and `depth_ticks_ask` features.
    ///
    /// # Arguments
    /// * `tick_size_dollars` - Tick size in dollars (e.g., 0.01 for US equities)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // US equity: $0.01 tick (default)
    /// let aggregator = MboAggregator::new(); // Uses $0.01 by default
    ///
    /// // Crypto: $0.001 tick
    /// let aggregator = MboAggregator::new().with_tick_size(0.001);
    ///
    /// // Forex: $0.0001 tick (pip)
    /// let aggregator = MboAggregator::new().with_tick_size(0.0001);
    ///
    /// // Futures: $1.00 tick
    /// let aggregator = MboAggregator::new().with_tick_size(1.0);
    /// ```
    ///
    /// # Panics
    /// Panics if tick_size_dollars is <= 0.
    pub fn with_tick_size(mut self, tick_size_dollars: f64) -> Self {
        assert!(
            tick_size_dollars > 0.0,
            "tick_size must be positive, got {}",
            tick_size_dollars
        );
        // Convert dollars to nanodollars: 1 dollar = 1e9 nanodollars
        self.tick_size_nanodollars = (tick_size_dollars * 1e9) as i64;
        self
    }
    
    /// Get the current tick size in nanodollars.
    ///
    /// Useful for debugging and testing.
    #[inline]
    pub fn tick_size_nanodollars(&self) -> i64 {
        self.tick_size_nanodollars
    }

    /// Enable queue position tracking for accurate queue metrics.
    ///
    /// When enabled, `average_queue_position` and `queue_size_ahead` features
    /// return accurate FIFO-based values instead of 0.0.
    ///
    /// # Performance Impact
    /// - Adds ~1 MB memory per symbol
    /// - Adds ~10-20 ns per event for queue tracking
    ///
    /// # Example
    /// ```ignore
    /// let aggregator = MboAggregator::new().with_queue_tracking();
    /// ```
    pub fn with_queue_tracking(mut self) -> Self {
        self.queue_tracker = Some(QueuePositionTracker::new(QueuePositionConfig::default()));
        self
    }

    /// Enable queue position tracking with custom configuration.
    pub fn with_queue_tracking_config(mut self, config: QueuePositionConfig) -> Self {
        self.queue_tracker = Some(QueuePositionTracker::new(config));
        self
    }

    /// Check if queue tracking is enabled.
    #[inline]
    pub fn has_queue_tracking(&self) -> bool {
        self.queue_tracker.is_some()
    }

    /// Process a single MBO event (O(1) amortized).
    ///
    /// Updates all windows and order tracker. This is the main hot path
    /// and must be extremely efficient to maintain throughput.
    ///
    /// If queue tracking is enabled, also updates the queue position tracker.
    #[inline]
    pub fn process_event(&mut self, event: MboEvent) {
        // Update all windows
        self.fast_window.push(event.clone());
        self.medium_window.push(event.clone());
        self.slow_window.push(event.clone());

        // Update order tracker
        self.update_order_tracker(&event);

        // Update queue tracker if enabled
        if let Some(ref mut tracker) = self.queue_tracker {
            let msg = event.to_mbo_message();
            tracker.process_message(&msg);
        }

        self.message_count += 1;
    }

    /// Update order lifecycle tracker based on event.
    #[inline]
    fn update_order_tracker(&mut self, event: &MboEvent) {
        match event.action {
            Action::Add => {
                // New order
                let info = OrderInfo::new(event.timestamp, event.size, event.price, event.side);
                self.order_tracker.insert(event.order_id, info);
            }
            Action::Modify => {
                // Modify existing order
                if let Some(info) = self.order_tracker.get_mut(&event.order_id) {
                    info.add_modification(event.price);
                }
            }
            Action::Cancel => {
                // Remove order
                self.order_tracker.remove(&event.order_id);
            }
            Action::Trade => {
                // Record fill
                if let Some(info) = self.order_tracker.get_mut(&event.order_id) {
                    info.add_fill(event.timestamp, event.size);

                    // Remove if fully filled
                    if info.current_size == 0 {
                        self.order_tracker.remove(&event.order_id);
                    }
                }
            }
            _ => {}
        }
    }

    /// Extract all 36 MBO features.
    ///
    /// Features are extracted primarily from the medium window (1000 messages),
    /// with some features using fast/slow windows or order tracker.
    ///
    /// # Returns
    ///
    /// Vector of 36 f64 features in the following order:
    /// - [0-11]: Order flow statistics
    /// - [12-19]: Size distribution
    /// - [20-25]: Queue & depth
    /// - [26-29]: Institutional detection
    /// - [30-35]: Core MBO metrics
    pub fn extract_features(&mut self, lob: &LobState) -> Vec<f64> {
        let mut features = Vec::with_capacity(36);

        // Category 1: Order flow (12 features)
        features.extend(self.extract_order_flow_features());

        // Category 2: Size distribution (8 features)
        features.extend(self.extract_size_features());

        // Category 3: Queue & depth (6 features)
        features.extend(self.extract_queue_features(lob));

        // Category 4: Institutional (4 features)
        features.extend(self.extract_institutional_features());

        // Category 5: Core MBO (6 features)
        features.extend(self.extract_core_features());

        features
    }

    /// Extract order flow statistics (12 features).
    fn extract_order_flow_features(&self) -> Vec<f64> {
        let w = &self.medium_window;
        let duration = w.duration_seconds();

        vec![
            // Event rates (6 features)
            w.add_count_bid as f64 / duration,
            w.add_count_ask as f64 / duration,
            w.cancel_count_bid as f64 / duration,
            w.cancel_count_ask as f64 / duration,
            w.trade_count_bid as f64 / duration,
            w.trade_count_ask as f64 / duration,
            // Order flow imbalances (4 features)
            self.net_order_flow(w),
            self.net_cancel_flow(w),
            self.net_trade_flow(w),
            self.aggressive_order_ratio(w),
            // Flow characteristics (2 features)
            self.order_flow_volatility(),
            self.flow_regime_indicator(),
        ]
    }

    /// Extract size distribution features (8 features).
    fn extract_size_features(&mut self) -> Vec<f64> {
        // Get percentiles first
        let p25 = self.medium_window.size_percentile(0);
        let p50 = self.medium_window.size_percentile(1);
        let p75 = self.medium_window.size_percentile(2);
        let p90 = self.medium_window.size_percentile(3);

        // Then compute other features
        let size_z = self.size_zscore();
        let large_ratio = self.large_order_ratio();
        let skew = self.size_skewness();
        let conc = self.size_concentration();

        vec![p25, p50, p75, p90, size_z, large_ratio, skew, conc]
    }

    /// Extract queue & depth features (6 features).
    fn extract_queue_features(&self, lob: &LobState) -> Vec<f64> {
        vec![
            self.average_queue_position(lob),
            self.queue_size_ahead(lob),
            self.orders_per_level(lob),
            self.level_concentration(lob),
            self.depth_ticks_bid(lob),
            self.depth_ticks_ask(lob),
        ]
    }

    /// Extract institutional detection features (4 features).
    fn extract_institutional_features(&mut self) -> Vec<f64> {
        vec![
            self.large_order_frequency(),
            self.large_order_imbalance(),
            self.modification_score(),
            self.iceberg_proxy(),
        ]
    }

    /// Extract core MBO metrics (6 features).
    fn extract_core_features(&self) -> Vec<f64> {
        vec![
            self.average_order_age(),
            self.median_order_lifetime(),
            self.average_fill_ratio(),
            self.average_time_to_first_fill(),
            self.cancel_to_add_ratio(),
            self.active_order_count(),
        ]
    }

    // ---- Helper functions for feature computation ----

    #[inline]
    fn net_order_flow(&self, w: &MboWindow) -> f64 {
        let net = (w.add_count_bid as i64) - (w.add_count_ask as i64);
        let total = (w.add_count_bid + w.add_count_ask) as f64;
        net as f64 / (total + 1e-8)
    }

    /// Net cancel flow: ratio of seller vs buyer order cancellations.
    ///
    /// Formula: (cancel_count_ask - cancel_count_bid) / total
    ///
    /// Interpretation:
    /// - Bid cancel = buyer pulling their order = less buying support = BEARISH
    /// - Ask cancel = seller pulling their order = less selling pressure = BULLISH
    ///
    /// Sign convention (RULE.md §9):
    /// - `> 0`: More ask cancels (sellers pulling) = reduced supply = BULLISH
    /// - `< 0`: More bid cancels (buyers pulling) = reduced demand = BEARISH
    /// - `= 0`: Balanced cancellation activity
    #[inline]
    fn net_cancel_flow(&self, w: &MboWindow) -> f64 {
        // CORRECT: (ask - bid) to follow > 0 = BULLISH convention
        // cancel_count_ask = sellers pulling, cancel_count_bid = buyers pulling
        let net = (w.cancel_count_ask as i64) - (w.cancel_count_bid as i64);
        let total = (w.cancel_count_bid + w.cancel_count_ask) as f64;
        net as f64 / (total + 1e-8)
    }

    /// Net trade flow: ratio of aggressive buying vs selling.
    ///
    /// Formula: (trade_count_ask - trade_count_bid) / total
    ///
    /// In MBO data:
    /// - Trade with Side::Bid = bid was HIT = seller aggressed = SELL-initiated
    /// - Trade with Side::Ask = ask was HIT = buyer aggressed = BUY-initiated
    ///
    /// Sign convention (RULE.md §9):
    /// - `> 0`: More BUY-initiated trades (buyers hitting asks) = BULLISH
    /// - `< 0`: More SELL-initiated trades (sellers hitting bids) = BEARISH
    /// - `= 0`: Balanced trading pressure
    #[inline]
    fn net_trade_flow(&self, w: &MboWindow) -> f64 {
        // CORRECT: (ask - bid) to follow > 0 = BULLISH convention
        // trade_count_ask = BUY-initiated, trade_count_bid = SELL-initiated
        let net = (w.trade_count_ask as i64) - (w.trade_count_bid as i64);
        let total = (w.trade_count_bid + w.trade_count_ask) as f64;
        net as f64 / (total + 1e-8)
    }

    #[inline]
    fn aggressive_order_ratio(&self, w: &MboWindow) -> f64 {
        let market_orders = w.trade_count_bid + w.trade_count_ask;
        let total_orders = w.add_count_bid + w.add_count_ask + market_orders;
        market_orders as f64 / (total_orders as f64 + 1e-8)
    }

    /// Compute volatility (std dev) of order flow across sub-windows.
    ///
    /// Divides the medium window into N_SUBWINDOWS sub-windows, computes
    /// net_order_flow for each, and returns the standard deviation.
    ///
    /// # Returns
    /// - Standard deviation of net_order_flow across sub-windows
    /// - Higher values indicate choppy, uncertain flow
    /// - Lower values indicate stable, directional flow
    /// - 0.0 if insufficient data for sub-windows
    ///
    /// # Implementation
    /// Uses 10 sub-windows. With medium_window of 1000 events,
    /// each sub-window contains 100 events.
    fn order_flow_volatility(&self) -> f64 {
        const N_SUBWINDOWS: usize = 10;
        const MIN_EVENTS_PER_SUBWINDOW: usize = 5;

        let w = &self.medium_window;
        let n_events = w.events.len();

        // Need enough events for meaningful sub-windows
        if n_events < N_SUBWINDOWS * MIN_EVENTS_PER_SUBWINDOW {
            return 0.0;
        }

        let subwindow_size = n_events / N_SUBWINDOWS;

        // Compute net_order_flow for each sub-window
        let mut subwindow_flows: Vec<f64> = Vec::with_capacity(N_SUBWINDOWS);

        for i in 0..N_SUBWINDOWS {
            let start = i * subwindow_size;
            let end = if i == N_SUBWINDOWS - 1 {
                n_events // Last subwindow gets remainder
            } else {
                (i + 1) * subwindow_size
            };

            // Count adds by side in this sub-window
            let mut add_bid: usize = 0;
            let mut add_ask: usize = 0;

            for j in start..end {
                if let Some(event) = w.events.get(j) {
                    if matches!(event.action, Action::Add) {
                        match event.side {
                            Side::Bid => add_bid += 1,
                            Side::Ask => add_ask += 1,
                            Side::None => {}
                        }
                    }
                }
            }

            // Net flow: (bid - ask) / (bid + ask + eps)
            let total = (add_bid + add_ask) as f64;
            let flow = if total > 0.0 {
                (add_bid as f64 - add_ask as f64) / total
            } else {
                0.0
            };

            subwindow_flows.push(flow);
        }

        // Compute standard deviation of sub-window flows
        if subwindow_flows.is_empty() {
            return 0.0;
        }

        let n = subwindow_flows.len() as f64;
        let mean: f64 = subwindow_flows.iter().sum::<f64>() / n;
        let variance: f64 = subwindow_flows
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        variance.sqrt()
    }

    fn flow_regime_indicator(&self) -> f64 {
        // Ratio of fast to slow window net flow
        let fast_flow = self.net_order_flow(&self.fast_window);
        let slow_flow = self.net_order_flow(&self.slow_window);
        fast_flow / (slow_flow.abs() + 1e-8)
    }

    fn size_zscore(&mut self) -> f64 {
        let w = &mut self.medium_window;
        if w.is_empty() {
            return 0.0;
        }
        // Use last event size
        let last_size = w.events.back().map(|e| e.size as f64).unwrap_or(0.0);
        let mean = w.size_mean();
        let std = w.size_std();
        (last_size - mean) / (std + 1e-8)
    }

    fn large_order_ratio(&mut self) -> f64 {
        let w = &mut self.medium_window;
        let threshold = w.size_percentile(3); // p90
        let large_count = w
            .events
            .iter()
            .filter(|e| e.size as f64 > threshold)
            .count();
        large_count as f64 / w.len().max(1) as f64
    }

    /// Compute skewness of order size distribution.
    ///
    /// Skewness = E[(X - μ)³] / σ³ = (1/n) Σ((x_i - μ)/σ)³
    ///
    /// # Returns
    /// - Positive: Right-skewed (many small orders, few large ones - institutional pattern)
    /// - Negative: Left-skewed (many large orders, few small ones)
    /// - Zero: Symmetric distribution
    /// - Returns 0.0 if insufficient data (< 3 samples) or zero variance
    ///
    /// # Interpretation
    /// High positive skewness (> 1.0) often indicates institutional activity,
    /// as large orders tend to be split into many smaller child orders with
    /// occasional larger fills.
    fn size_skewness(&mut self) -> f64 {
        let w = &mut self.medium_window;

        // Need at least 3 samples for meaningful skewness
        if w.events.len() < 3 {
            return 0.0;
        }

        // Get mean and std (triggers lazy recomputation if dirty)
        let mean = w.size_mean();
        let std = w.size_std();

        // Guard against division by zero
        if std < 1e-10 {
            return 0.0;
        }

        // Compute third moment: (1/n) Σ((x_i - μ)/σ)³
        let n = w.events.len() as f64;
        let skew: f64 = w
            .events
            .iter()
            .map(|e| {
                let z = (e.size as f64 - mean) / std;
                z * z * z // z³
            })
            .sum::<f64>()
            / n;

        skew
    }

    /// Compute Herfindahl-Hirschman Index of order size distribution.
    ///
    /// HHI = Σ(share_i)² where share_i = size_i / total_size
    ///
    /// # Returns
    /// - Range: [1/N, 1.0] where N = number of orders in window
    /// - 1/N = perfectly even sizes (retail-like)
    /// - 1.0 = single order has all volume (block trade)
    /// - 0.0 if no orders
    ///
    /// # Interpretation
    /// - Low HHI (< 0.1): Volume spread across many similar-sized orders
    /// - High HHI (> 0.3): Volume concentrated in few large orders (institutional)
    fn size_concentration(&self) -> f64 {
        let w = &self.medium_window;

        if w.events.is_empty() {
            return 0.0;
        }

        // Compute total size
        let total_size: u64 = w.events.iter().map(|e| e.size as u64).sum();

        if total_size == 0 {
            return 0.0;
        }

        let total_f = total_size as f64;

        // Compute HHI: sum of squared shares
        let hhi: f64 = w
            .events
            .iter()
            .map(|e| {
                let share = e.size as f64 / total_f;
                share * share
            })
            .sum();

        hhi
    }

    /// Compute average queue position across all tracked orders.
    ///
    /// Queue position is the FIFO order within a price level (0 = front, first to fill).
    ///
    /// # Returns
    /// - Average position across all orders (0.0 if no queue tracking or no orders)
    /// - Lower values indicate orders are closer to execution
    ///
    /// # Note
    /// Requires `with_queue_tracking()` to be enabled. Returns 0.0 otherwise.
    fn average_queue_position(&self, _lob: &LobState) -> f64 {
        match &self.queue_tracker {
            Some(tracker) => {
                // Compute weighted average across both sides
                let bid_avg = tracker.average_queue_position(Side::Bid);
                let ask_avg = tracker.average_queue_position(Side::Ask);

                match (bid_avg, ask_avg) {
                    (Some(b), Some(a)) => (b + a) / 2.0,
                    (Some(b), None) => b,
                    (None, Some(a)) => a,
                    (None, None) => 0.0,
                }
            }
            None => 0.0,
        }
    }

    /// Compute average volume ahead of tracked orders.
    ///
    /// Volume ahead is the total shares/contracts that must execute before
    /// a given order (at the same price level).
    ///
    /// # Returns
    /// - Average volume ahead across all tracked orders (0.0 if no tracking)
    /// - Higher values indicate more volume must execute before our orders
    ///
    /// # Note
    /// Requires `with_queue_tracking()` to be enabled. Returns 0.0 otherwise.
    fn queue_size_ahead(&self, _lob: &LobState) -> f64 {
        match &self.queue_tracker {
            Some(tracker) => {
                // Sum volume ahead for all tracked orders
                let mut total_volume_ahead: u64 = 0;
                let mut order_count: usize = 0;

                // We need to iterate through all tracked orders
                // The tracker's order_locations is private, so we use
                // the aggregator's order_tracker which has the same order_ids
                for &order_id in self.order_tracker.keys() {
                    if let Some(vol) = tracker.volume_ahead(order_id) {
                        total_volume_ahead += vol;
                        order_count += 1;
                    }
                }

                if order_count > 0 {
                    total_volume_ahead as f64 / order_count as f64
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    fn orders_per_level(&self, lob: &LobState) -> f64 {
        let active_orders = self.order_tracker.len();
        let active_levels = lob.bid_prices.iter().filter(|&&p| p > 0).count()
            + lob.ask_prices.iter().filter(|&&p| p > 0).count();
        active_orders as f64 / active_levels.max(1) as f64
    }

    /// Compute volume concentration across LOB levels using Herfindahl-Hirschman Index.
    ///
    /// HHI = Σ(share_i)² where share_i = volume_i / total_volume
    ///
    /// # Returns
    /// - Range: [1/N, 1.0] where N = number of active levels
    /// - 1/N = perfectly even distribution (liquid)
    /// - 1.0 = all volume concentrated at one level (thin)
    /// - 0.0 if no volume present
    ///
    /// # Interpretation
    /// - Low HHI (< 0.25): Well-distributed liquidity, easier to execute large orders
    /// - High HHI (> 0.5): Concentrated liquidity, higher price impact risk
    fn level_concentration(&self, lob: &LobState) -> f64 {
        // Compute total volume across both sides
        let total_bid: u64 = lob.bid_sizes[..lob.levels]
            .iter()
            .map(|&s| s as u64)
            .sum();
        let total_ask: u64 = lob.ask_sizes[..lob.levels]
            .iter()
            .map(|&s| s as u64)
            .sum();
        let total = total_bid + total_ask;

        if total == 0 {
            return 0.0;
        }

        let total_f = total as f64;

        // Compute HHI: sum of squared shares
        let mut hhi: f64 = 0.0;

        // Add bid side contributions
        for &size in &lob.bid_sizes[..lob.levels] {
            if size > 0 {
                let share = size as f64 / total_f;
                hhi += share * share;
            }
        }

        // Add ask side contributions
        for &size in &lob.ask_sizes[..lob.levels] {
            if size > 0 {
                let share = size as f64 / total_f;
                hhi += share * share;
            }
        }

        hhi
    }

    /// Compute volume-weighted average depth on bid side in ticks.
    ///
    /// Depth_ticks = Σ(vol_i × distance_i) / Σ(vol_i)
    /// where distance_i = (best_bid - bid_price[i]) / tick_size
    ///
    /// # Returns
    /// - Average depth in ticks (0.0 if no volume)
    /// - Higher values indicate liquidity is further from best price
    ///
    /// # Tick Size
    /// Uses the tick size configured via `with_tick_size()` (default: $0.01).
    /// This ensures correct depth measurements for different instrument types:
    /// - US equities: $0.01
    /// - Crypto: $0.001
    /// - Forex: $0.0001
    fn depth_ticks_bid(&self, lob: &LobState) -> f64 {
        let best_bid = match lob.best_bid {
            Some(p) if p > 0 => p,
            _ => return 0.0,
        };

        let mut weighted_depth: f64 = 0.0;
        let mut total_volume: u64 = 0;

        for i in 0..lob.levels {
            let price = lob.bid_prices[i];
            let size = lob.bid_sizes[i] as u64;

            if price > 0 && size > 0 {
                // Distance in ticks: (best_bid - this_price) / tick_size
                // Bid prices decrease from best (index 0) to deeper levels
                let distance_ticks = (best_bid - price) / self.tick_size_nanodollars;
                weighted_depth += size as f64 * distance_ticks as f64;
                total_volume += size;
            }
        }

        if total_volume > 0 {
            weighted_depth / total_volume as f64
        } else {
            0.0
        }
    }

    /// Compute volume-weighted average depth on ask side in ticks.
    ///
    /// Depth_ticks = Σ(vol_i × distance_i) / Σ(vol_i)
    /// where distance_i = (ask_price[i] - best_ask) / tick_size
    ///
    /// # Returns
    /// - Average depth in ticks (0.0 if no volume)
    /// - Higher values indicate liquidity is further from best price
    ///
    /// # Tick Size
    /// Uses the tick size configured via `with_tick_size()` (default: $0.01).
    /// This ensures correct depth measurements for different instrument types:
    /// - US equities: $0.01
    /// - Crypto: $0.001
    /// - Forex: $0.0001
    fn depth_ticks_ask(&self, lob: &LobState) -> f64 {
        let best_ask = match lob.best_ask {
            Some(p) if p > 0 => p,
            _ => return 0.0,
        };

        let mut weighted_depth: f64 = 0.0;
        let mut total_volume: u64 = 0;

        for i in 0..lob.levels {
            let price = lob.ask_prices[i];
            let size = lob.ask_sizes[i] as u64;

            if price > 0 && size > 0 {
                // Distance in ticks: (this_price - best_ask) / tick_size
                // Ask prices increase from best (index 0) to deeper levels
                let distance_ticks = (price - best_ask) / self.tick_size_nanodollars;
                weighted_depth += size as f64 * distance_ticks as f64;
                total_volume += size;
            }
        }

        if total_volume > 0 {
            weighted_depth / total_volume as f64
        } else {
            0.0
        }
    }

    fn large_order_frequency(&mut self) -> f64 {
        let w = &mut self.medium_window;
        let threshold = w.size_percentile(3); // p90
        let large_count = w
            .events
            .iter()
            .filter(|e| e.size as f64 > threshold)
            .count();
        large_count as f64 / w.duration_seconds()
    }

    fn large_order_imbalance(&mut self) -> f64 {
        let w = &mut self.medium_window;
        let threshold = w.size_percentile(3); // p90
        let (large_bid, large_ask) = w.events.iter().filter(|e| e.size as f64 > threshold).fold(
            (0u64, 0u64),
            |(bid, ask), e| {
                match e.side {
                    Side::Bid => (bid + e.size as u64, ask),
                    Side::Ask => (bid, ask + e.size as u64),
                    Side::None => (bid, ask), // Ignore non-directional orders
                }
            },
        );
        let total = large_bid + large_ask;
        (large_bid as f64 - large_ask as f64) / (total as f64 + 1e-8)
    }

    fn modification_score(&self) -> f64 {
        if self.order_tracker.is_empty() {
            return 0.0;
        }
        let total_mods: usize = self
            .order_tracker
            .values()
            .map(|info| info.modifications as usize)
            .sum();
        total_mods as f64 / self.order_tracker.len() as f64
    }

    fn iceberg_proxy(&self) -> f64 {
        if self.order_tracker.is_empty() {
            return 0.0;
        }
        // High fill ratio × high modifications suggests iceberg
        let avg_fill_ratio = self.average_fill_ratio();
        let mod_score = self.modification_score();
        avg_fill_ratio * (mod_score / 10.0).min(1.0)
    }

    fn average_order_age(&self) -> f64 {
        if self.order_tracker.is_empty() {
            return 0.0;
        }
        let current_ts = self.medium_window.last_ts;
        let total_age: f64 = self
            .order_tracker
            .values()
            .map(|info| info.age(current_ts))
            .sum();
        total_age / self.order_tracker.len() as f64
    }

    fn median_order_lifetime(&self) -> f64 {
        // Placeholder: track completed orders
        0.0
    }

    fn average_fill_ratio(&self) -> f64 {
        if self.order_tracker.is_empty() {
            return 0.0;
        }
        let total_ratio: f64 = self
            .order_tracker
            .values()
            .map(|info| info.fill_ratio())
            .sum();
        total_ratio / self.order_tracker.len() as f64
    }

    fn average_time_to_first_fill(&self) -> f64 {
        let fills: Vec<f64> = self
            .order_tracker
            .values()
            .filter_map(|info| info.time_to_first_fill())
            .collect();
        if fills.is_empty() {
            return 0.0;
        }
        fills.iter().sum::<f64>() / fills.len() as f64
    }

    fn cancel_to_add_ratio(&self) -> f64 {
        let w = &self.medium_window;
        let cancels = w.cancel_count_bid + w.cancel_count_ask;
        let adds = w.add_count_bid + w.add_count_ask;
        cancels as f64 / (adds as f64 + 1e-8)
    }

    fn active_order_count(&self) -> f64 {
        self.order_tracker.len() as f64
    }
}

impl Default for MboAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbo_event_creation() {
        let event = MboEvent::new(
            1000000000,
            Action::Add,
            Side::Bid,
            100_000_000_000,
            100,
            12345,
        );

        assert_eq!(event.timestamp, 1000000000);
        assert_eq!(event.size, 100);
        assert_eq!(event.order_id, 12345);
    }

    #[test]
    fn test_mbo_window_push() {
        let mut window = MboWindow::new(10);

        for i in 0..5 {
            let event = MboEvent::new(i * 1000000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            window.push(event);
        }

        assert_eq!(window.len(), 5);
        assert_eq!(window.add_count_bid, 5);
    }

    #[test]
    fn test_mbo_window_eviction() {
        let mut window = MboWindow::new(3);

        // Add 5 events (capacity is 3, so oldest 2 will be evicted)
        for i in 0..5 {
            let event = MboEvent::new(i * 1000000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            window.push(event);
        }

        assert_eq!(window.len(), 3);
        assert_eq!(window.add_count_bid, 3); // Only last 3 counted
    }

    #[test]
    fn test_order_info_fill_ratio() {
        let mut info = OrderInfo::new(0, 1000, 100_000_000_000, Side::Bid);

        info.add_fill(1000000, 300);
        assert_eq!(info.fill_ratio(), 0.3);

        info.add_fill(2000000, 400);
        assert_eq!(info.fill_ratio(), 0.7);
    }

    #[test]
    fn test_mbo_aggregator_creation() {
        let aggregator = MboAggregator::new();
        assert_eq!(aggregator.message_count, 0);
        assert!(aggregator.order_tracker.is_empty());
    }

    // =========================================================================
    // level_concentration tests
    // =========================================================================

    #[test]
    fn test_level_concentration_empty_lob() {
        let aggregator = MboAggregator::new();
        let lob = LobState::new(10);

        let result = aggregator.level_concentration(&lob);
        assert_eq!(result, 0.0, "Empty LOB should return 0.0");
    }

    #[test]
    fn test_level_concentration_single_level() {
        // All volume at one level = HHI of 1.0
        let aggregator = MboAggregator::new();
        let mut lob = LobState::new(10);
        lob.bid_sizes[0] = 1000; // All volume here

        let result = aggregator.level_concentration(&lob);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Single level concentration should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_level_concentration_perfectly_even() {
        // 4 levels with equal volume = HHI of 1/4 = 0.25
        // Each level has share = 0.25, HHI = 4 × (0.25)² = 4 × 0.0625 = 0.25
        let aggregator = MboAggregator::new();
        let mut lob = LobState::new(10);
        lob.bid_sizes[0] = 100;
        lob.bid_sizes[1] = 100;
        lob.ask_sizes[0] = 100;
        lob.ask_sizes[1] = 100;

        let result = aggregator.level_concentration(&lob);
        assert!(
            (result - 0.25).abs() < 1e-10,
            "4 equal levels should have HHI of 0.25, got {}",
            result
        );
    }

    #[test]
    fn test_level_concentration_asymmetric() {
        // 2 levels: one with 90% volume, one with 10%
        // HHI = (0.9)² + (0.1)² = 0.81 + 0.01 = 0.82
        let aggregator = MboAggregator::new();
        let mut lob = LobState::new(10);
        lob.bid_sizes[0] = 900; // 90%
        lob.bid_sizes[1] = 100; // 10%

        let result = aggregator.level_concentration(&lob);
        assert!(
            (result - 0.82).abs() < 1e-10,
            "90/10 split should have HHI of 0.82, got {}",
            result
        );
    }

    // =========================================================================
    // depth_ticks tests
    // =========================================================================

    #[test]
    fn test_depth_ticks_bid_no_best() {
        let aggregator = MboAggregator::new();
        let lob = LobState::new(10);

        let result = aggregator.depth_ticks_bid(&lob);
        assert_eq!(result, 0.0, "No best bid should return 0.0");
    }

    #[test]
    fn test_depth_ticks_bid_single_level() {
        // Single level at best price = 0 ticks away
        let aggregator = MboAggregator::new();
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000); // $100.00
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_sizes[0] = 100;

        let result = aggregator.depth_ticks_bid(&lob);
        assert!(
            result.abs() < 1e-10,
            "Single level at best should be 0 ticks, got {}",
            result
        );
    }

    #[test]
    fn test_depth_ticks_bid_multiple_levels() {
        // Level 0: $100.00 (best), 100 shares, 0 ticks away
        // Level 1: $99.99 (1 tick away = $0.01), 100 shares
        // Weighted avg = (100×0 + 100×1) / 200 = 0.5 ticks
        let aggregator = MboAggregator::new();
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000); // $100.00
        lob.bid_prices[0] = 100_000_000_000; // $100.00
        lob.bid_prices[1] = 99_990_000_000; // $99.99
        lob.bid_sizes[0] = 100;
        lob.bid_sizes[1] = 100;

        let result = aggregator.depth_ticks_bid(&lob);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "Should be 0.5 ticks weighted avg, got {}",
            result
        );
    }

    #[test]
    fn test_depth_ticks_ask_multiple_levels() {
        // Level 0: $100.00 (best), 200 shares, 0 ticks away
        // Level 1: $100.02 (2 ticks away = $0.02), 100 shares
        // Weighted avg = (200×0 + 100×2) / 300 = 200/300 = 0.6667 ticks
        let aggregator = MboAggregator::new();
        let mut lob = LobState::new(10);
        lob.best_ask = Some(100_000_000_000); // $100.00
        lob.ask_prices[0] = 100_000_000_000; // $100.00
        lob.ask_prices[1] = 100_020_000_000; // $100.02
        lob.ask_sizes[0] = 200;
        lob.ask_sizes[1] = 100;

        let result = aggregator.depth_ticks_ask(&lob);
        let expected = 200.0 / 300.0; // 0.6667
        assert!(
            (result - expected).abs() < 1e-6,
            "Should be ~0.6667 ticks, got {}",
            result
        );
    }

    // =========================================================================
    // size_skewness tests
    // =========================================================================

    #[test]
    fn test_size_skewness_insufficient_data() {
        let mut aggregator = MboAggregator::new();

        // Only 2 events - need at least 3
        for i in 0..2 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let result = aggregator.size_skewness();
        assert_eq!(result, 0.0, "Insufficient data should return 0.0");
    }

    #[test]
    fn test_size_skewness_symmetric() {
        // Symmetric distribution: equal amounts above and below mean
        // Sizes: 80, 100, 100, 100, 120 → mean = 100
        // Z-scores: -2σ, 0, 0, 0, +2σ should give skewness near 0
        let mut aggregator = MboAggregator::new();

        // Create symmetric distribution
        let sizes = [80u32, 100, 100, 100, 120];
        for (i, &size) in sizes.iter().enumerate() {
            let event = MboEvent::new(
                i as u64 * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                size,
                i as u64,
            );
            aggregator.process_event(event);
        }

        let result = aggregator.size_skewness();
        assert!(
            result.abs() < 0.1,
            "Symmetric distribution should have near-zero skewness, got {}",
            result
        );
    }

    #[test]
    fn test_size_skewness_right_skewed() {
        // Right-skewed: many small values, few large
        // This is typical of institutional order splitting
        // Sizes: 10, 10, 10, 10, 100 → right skewed (positive)
        let mut aggregator = MboAggregator::new();

        let sizes = [10u32, 10, 10, 10, 100];
        for (i, &size) in sizes.iter().enumerate() {
            let event = MboEvent::new(
                i as u64 * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                size,
                i as u64,
            );
            aggregator.process_event(event);
        }

        let result = aggregator.size_skewness();
        assert!(
            result > 0.5,
            "Right-skewed distribution should have positive skewness, got {}",
            result
        );
    }

    #[test]
    fn test_size_skewness_left_skewed() {
        // Left-skewed: many large values, few small
        // Sizes: 100, 100, 100, 100, 10 → left skewed (negative)
        let mut aggregator = MboAggregator::new();

        let sizes = [100u32, 100, 100, 100, 10];
        for (i, &size) in sizes.iter().enumerate() {
            let event = MboEvent::new(
                i as u64 * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                size,
                i as u64,
            );
            aggregator.process_event(event);
        }

        let result = aggregator.size_skewness();
        assert!(
            result < -0.5,
            "Left-skewed distribution should have negative skewness, got {}",
            result
        );
    }

    // =========================================================================
    // size_concentration tests
    // =========================================================================

    #[test]
    fn test_size_concentration_empty() {
        let aggregator = MboAggregator::new();
        let result = aggregator.size_concentration();
        assert_eq!(result, 0.0, "Empty window should return 0.0");
    }

    #[test]
    fn test_size_concentration_single_order() {
        // Single order has HHI = 1.0 (100% concentration)
        let mut aggregator = MboAggregator::new();
        let event = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        aggregator.process_event(event);

        let result = aggregator.size_concentration();
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Single order should have HHI of 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_size_concentration_equal_sizes() {
        // 4 orders of equal size = HHI of 1/4 = 0.25
        let mut aggregator = MboAggregator::new();

        for i in 0..4 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100, // All same size
                i,
            );
            aggregator.process_event(event);
        }

        let result = aggregator.size_concentration();
        assert!(
            (result - 0.25).abs() < 1e-10,
            "4 equal orders should have HHI of 0.25, got {}",
            result
        );
    }

    #[test]
    fn test_size_concentration_concentrated() {
        // 2 orders: 900 shares and 100 shares
        // Shares: 0.9 and 0.1
        // HHI = 0.9² + 0.1² = 0.81 + 0.01 = 0.82
        let mut aggregator = MboAggregator::new();

        let event1 = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 900, 1);
        let event2 = MboEvent::new(2_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 2);
        aggregator.process_event(event1);
        aggregator.process_event(event2);

        let result = aggregator.size_concentration();
        assert!(
            (result - 0.82).abs() < 1e-10,
            "900/100 split should have HHI of 0.82, got {}",
            result
        );
    }

    // =========================================================================
    // order_flow_volatility tests
    // =========================================================================

    #[test]
    fn test_order_flow_volatility_insufficient_data() {
        let mut aggregator = MboAggregator::new();

        // Need N_SUBWINDOWS × MIN_EVENTS = 10 × 5 = 50 events
        for i in 0..40 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let result = aggregator.order_flow_volatility();
        assert_eq!(result, 0.0, "Insufficient data should return 0.0");
    }

    #[test]
    fn test_order_flow_volatility_constant_flow() {
        // All bids or all asks = constant flow → low volatility
        let mut aggregator = MboAggregator::new();

        // 100 events, all bids → net_flow = 1.0 in every subwindow
        for i in 0..100 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let result = aggregator.order_flow_volatility();
        assert!(
            result < 0.01,
            "Constant flow should have near-zero volatility, got {}",
            result
        );
    }

    #[test]
    fn test_order_flow_volatility_alternating_flow() {
        // Alternating bid/ask subwindows → high volatility
        let mut aggregator = MboAggregator::new();

        // 100 events, each subwindow of 10 alternates between all-bid and all-ask
        for i in 0..100u64 {
            let subwindow = i / 10;
            let side = if subwindow % 2 == 0 {
                Side::Bid
            } else {
                Side::Ask
            };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let result = aggregator.order_flow_volatility();
        // With alternating +1 and -1 flows, std dev should be 1.0
        assert!(
            result > 0.5,
            "Alternating flow should have high volatility, got {}",
            result
        );
    }

    #[test]
    fn test_order_flow_volatility_balanced_flow() {
        // 50% bid, 50% ask in each subwindow → flow = 0.0, volatility = 0.0
        let mut aggregator = MboAggregator::new();

        // 100 events, perfectly alternating bid/ask
        for i in 0..100u64 {
            let side = if i % 2 == 0 { Side::Bid } else { Side::Ask };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let result = aggregator.order_flow_volatility();
        assert!(
            result < 0.1,
            "Balanced flow should have low volatility, got {}",
            result
        );
    }

    // =========================================================================
    // Integration test: verify feature extraction produces non-zero values
    // =========================================================================

    #[test]
    fn test_extract_features_with_populated_data() {
        let mut aggregator = MboAggregator::new();

        // Create a realistic LOB state
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000); // $100.00
        lob.best_ask = Some(100_010_000_000); // $100.01
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_prices[1] = 99_990_000_000;
        lob.bid_sizes[0] = 500;
        lob.bid_sizes[1] = 300;
        lob.ask_prices[0] = 100_010_000_000;
        lob.ask_prices[1] = 100_020_000_000;
        lob.ask_sizes[0] = 400;
        lob.ask_sizes[1] = 200;

        // Process enough events for all features
        for i in 0..200u64 {
            let side = if i % 3 == 0 { Side::Bid } else { Side::Ask };
            let size = 50 + ((i * 17) % 200) as u32; // Varied sizes
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, 100_000_000_000, size, i);
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&lob);
        assert_eq!(features.len(), 36, "Should have 36 features");

        // Check that the implemented features are non-zero
        // Feature indices from extract_queue_features (indices 20-25):
        // [20] average_queue_position (still stub)
        // [21] queue_size_ahead (still stub)
        // [22] orders_per_level
        // [23] level_concentration
        // [24] depth_ticks_bid
        // [25] depth_ticks_ask

        let level_conc = features[23];
        assert!(
            level_conc > 0.0 && level_conc <= 1.0,
            "level_concentration should be in (0,1], got {}",
            level_conc
        );

        let depth_bid = features[24];
        assert!(
            depth_bid >= 0.0,
            "depth_ticks_bid should be >= 0, got {}",
            depth_bid
        );

        let depth_ask = features[25];
        assert!(
            depth_ask >= 0.0,
            "depth_ticks_ask should be >= 0, got {}",
            depth_ask
        );

        // Size features (indices 12-19):
        // [18] size_skewness
        // [19] size_concentration
        let size_skew = features[18];
        assert!(
            size_skew.is_finite(),
            "size_skewness should be finite, got {}",
            size_skew
        );

        let size_conc = features[19];
        assert!(
            size_conc > 0.0 && size_conc <= 1.0,
            "size_concentration should be in (0,1], got {}",
            size_conc
        );

        // Order flow volatility (index 10)
        let flow_vol = features[10];
        assert!(
            flow_vol >= 0.0,
            "order_flow_volatility should be >= 0, got {}",
            flow_vol
        );
    }

    // =========================================================================
    // Queue tracking tests
    // =========================================================================

    #[test]
    fn test_queue_tracking_disabled_by_default() {
        let aggregator = MboAggregator::new();
        assert!(!aggregator.has_queue_tracking());
    }

    #[test]
    fn test_queue_tracking_enabled() {
        let aggregator = MboAggregator::new().with_queue_tracking();
        assert!(aggregator.has_queue_tracking());
    }

    #[test]
    fn test_average_queue_position_no_tracking() {
        // Without queue tracking, should return 0.0
        let aggregator = MboAggregator::new();
        let lob = LobState::new(10);

        let result = aggregator.average_queue_position(&lob);
        assert_eq!(result, 0.0, "Without tracking should return 0.0");
    }

    #[test]
    fn test_queue_size_ahead_no_tracking() {
        // Without queue tracking, should return 0.0
        let aggregator = MboAggregator::new();
        let lob = LobState::new(10);

        let result = aggregator.queue_size_ahead(&lob);
        assert_eq!(result, 0.0, "Without tracking should return 0.0");
    }

    #[test]
    fn test_average_queue_position_single_order() {
        // Single order at position 0 = avg position 0
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        let event = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        aggregator.process_event(event);

        let lob = LobState::new(10);
        let result = aggregator.average_queue_position(&lob);

        // Single order on one side, position 0
        assert!(
            result.abs() < 1e-10,
            "Single order should have avg position 0, got {}",
            result
        );
    }

    #[test]
    fn test_average_queue_position_multiple_orders_same_price() {
        // 3 orders at same price: positions 0, 1, 2
        // Average = (0 + 1 + 2) / 3 = 1.0
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        for i in 1..=3u64 {
            let event =
                MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let result = aggregator.average_queue_position(&lob);

        // Average position = (0 + 1 + 2) / 3 = 1.0
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Average of positions 0,1,2 should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_queue_size_ahead_fifo_order() {
        // Order 1: size 100, position 0, volume ahead = 0
        // Order 2: size 200, position 1, volume ahead = 100
        // Order 3: size 150, position 2, volume ahead = 300
        // Average volume ahead = (0 + 100 + 300) / 3 = 133.33
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        let event1 = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        let event2 = MboEvent::new(2_000_000, Action::Add, Side::Bid, 100_000_000_000, 200, 2);
        let event3 = MboEvent::new(3_000_000, Action::Add, Side::Bid, 100_000_000_000, 150, 3);

        aggregator.process_event(event1);
        aggregator.process_event(event2);
        aggregator.process_event(event3);

        let lob = LobState::new(10);
        let result = aggregator.queue_size_ahead(&lob);

        let expected = (0.0 + 100.0 + 300.0) / 3.0; // 133.33
        assert!(
            (result - expected).abs() < 1.0,
            "Average volume ahead should be ~133.33, got {}",
            result
        );
    }

    #[test]
    fn test_queue_position_after_cancel() {
        // Add 3 orders, then cancel the first one
        // Remaining orders should have positions 0 and 1
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        for i in 1..=3u64 {
            let event =
                MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        // Cancel order 1
        let cancel = MboEvent::new(4_000_000, Action::Cancel, Side::Bid, 100_000_000_000, 100, 1);
        aggregator.process_event(cancel);

        let lob = LobState::new(10);
        let result = aggregator.average_queue_position(&lob);

        // Now orders 2 and 3 are at positions 0 and 1
        // Average = (0 + 1) / 2 = 0.5
        assert!(
            (result - 0.5).abs() < 1e-10,
            "After cancel, avg should be 0.5, got {}",
            result
        );
    }

    #[test]
    fn test_queue_tracking_both_sides() {
        // Add orders on both sides and verify averaging works
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        // 2 bids: positions 0, 1 → avg = 0.5
        let bid1 = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        let bid2 = MboEvent::new(2_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 2);

        // 2 asks: positions 0, 1 → avg = 0.5
        let ask1 = MboEvent::new(3_000_000, Action::Add, Side::Ask, 101_000_000_000, 100, 3);
        let ask2 = MboEvent::new(4_000_000, Action::Add, Side::Ask, 101_000_000_000, 100, 4);

        aggregator.process_event(bid1);
        aggregator.process_event(bid2);
        aggregator.process_event(ask1);
        aggregator.process_event(ask2);

        let lob = LobState::new(10);
        let result = aggregator.average_queue_position(&lob);

        // Both sides have avg 0.5, so combined avg = (0.5 + 0.5) / 2 = 0.5
        assert!(
            (result - 0.5).abs() < 1e-10,
            "Combined avg should be 0.5, got {}",
            result
        );
    }

    #[test]
    fn test_extract_features_with_queue_tracking() {
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        // Create LOB state
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_sizes[0] = 500;
        lob.ask_prices[0] = 100_010_000_000;
        lob.ask_sizes[0] = 400;

        // Process some events
        for i in 0..50u64 {
            let side = if i % 2 == 0 { Side::Bid } else { Side::Ask };
            let price = if side == Side::Bid {
                100_000_000_000
            } else {
                100_010_000_000
            };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, price, 100, i + 1);
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&lob);

        // Feature indices 20 and 21 should now be non-zero
        let avg_pos = features[20];
        let vol_ahead = features[21];

        assert!(
            avg_pos > 0.0,
            "With queue tracking, avg_queue_position should be > 0, got {}",
            avg_pos
        );
        assert!(
            vol_ahead > 0.0,
            "With queue tracking, queue_size_ahead should be > 0, got {}",
            vol_ahead
        );
    }

    // =========================================================================
    // Sign Convention Tests (RULE.md §9)
    // =========================================================================
    // Standard convention: > 0 = BULLISH, < 0 = BEARISH
    //
    // For MBO data:
    // - Action::Add with Side::Bid = new buy limit order = BULLISH
    // - Action::Add with Side::Ask = new sell limit order = BEARISH
    // - Action::Cancel with Side::Bid = buyer pulling order = BEARISH
    // - Action::Cancel with Side::Ask = seller pulling order = BULLISH
    // - Action::Trade with Side::Bid = bid was HIT = seller aggressed = BEARISH
    // - Action::Trade with Side::Ask = ask was HIT = buyer aggressed = BULLISH

    #[test]
    fn test_net_order_flow_sign_convention_bullish() {
        // More bid adds = more buy orders = BULLISH → should be > 0
        let mut aggregator = MboAggregator::new();

        // 10 bid adds, 5 ask adds
        for i in 0..10u64 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }
        for i in 10..15u64 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Ask, 100_010_000_000, 100, i);
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_order_flow = features[6]; // Index 6 in MBO features = net_order_flow

        assert!(
            net_order_flow > 0.0,
            "More bid adds (buy orders) should give net_order_flow > 0 (BULLISH). \
             Got {:.4}. Sign convention: > 0 = BULLISH per RULE.md §9",
            net_order_flow
        );

        // Expected: (10 - 5) / 15 = 0.333...
        let expected = (10.0 - 5.0) / 15.0;
        assert!(
            (net_order_flow - expected).abs() < 0.01,
            "net_order_flow formula: (bid - ask) / total. Expected {:.4}, got {:.4}",
            expected,
            net_order_flow
        );
    }

    #[test]
    fn test_net_order_flow_sign_convention_bearish() {
        // More ask adds = more sell orders = BEARISH → should be < 0
        let mut aggregator = MboAggregator::new();

        // 5 bid adds, 10 ask adds
        for i in 0..5u64 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }
        for i in 5..15u64 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Ask, 100_010_000_000, 100, i);
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_order_flow = features[6];

        assert!(
            net_order_flow < 0.0,
            "More ask adds (sell orders) should give net_order_flow < 0 (BEARISH). \
             Got {:.4}. Sign convention: < 0 = BEARISH per RULE.md §9",
            net_order_flow
        );
    }

    #[test]
    fn test_net_cancel_flow_sign_convention_bullish() {
        // More ask cancels = sellers pulling orders = less selling pressure = BULLISH
        // Per standard convention: > 0 = BULLISH
        //
        // Current formula: (cancel_bid - cancel_ask) / total
        // If more ask cancels: cancel_ask > cancel_bid → result < 0 → WRONG
        //
        // CORRECT formula should be: (cancel_ask - cancel_bid) / total
        // If more ask cancels: result > 0 → BULLISH ✓
        let mut aggregator = MboAggregator::new();

        // First add some orders to cancel
        for i in 0..20u64 {
            let side = if i < 10 { Side::Bid } else { Side::Ask };
            let price = if side == Side::Bid {
                100_000_000_000
            } else {
                100_010_000_000
            };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, price, 100, i);
            aggregator.process_event(event);
        }

        // 3 bid cancels, 8 ask cancels
        for i in 0..3u64 {
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..18u64 {
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Cancel,
                Side::Ask,
                100_010_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_cancel_flow = features[7]; // Index 7 = net_cancel_flow

        // More ask cancels = sellers pulling = BULLISH → should be > 0
        assert!(
            net_cancel_flow > 0.0,
            "More ask cancels (sellers pulling) should give net_cancel_flow > 0 (BULLISH). \
             Got {:.4}. This test validates RULE.md §9 sign convention.",
            net_cancel_flow
        );
    }

    #[test]
    fn test_net_cancel_flow_sign_convention_bearish() {
        // More bid cancels = buyers pulling orders = less buying support = BEARISH
        // Per standard convention: < 0 = BEARISH
        let mut aggregator = MboAggregator::new();

        // First add orders
        for i in 0..20u64 {
            let side = if i < 10 { Side::Bid } else { Side::Ask };
            let price = if side == Side::Bid {
                100_000_000_000
            } else {
                100_010_000_000
            };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, price, 100, i);
            aggregator.process_event(event);
        }

        // 8 bid cancels, 3 ask cancels
        for i in 0..8u64 {
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..13u64 {
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Cancel,
                Side::Ask,
                100_010_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_cancel_flow = features[7];

        // More bid cancels = buyers pulling = BEARISH → should be < 0
        assert!(
            net_cancel_flow < 0.0,
            "More bid cancels (buyers pulling) should give net_cancel_flow < 0 (BEARISH). \
             Got {:.4}. This test validates RULE.md §9 sign convention.",
            net_cancel_flow
        );
    }

    #[test]
    fn test_net_trade_flow_sign_convention_bullish() {
        // In MBO data:
        // - Trade with Side::Bid = bid was HIT = seller aggressed = SELL-initiated
        // - Trade with Side::Ask = ask was HIT = buyer aggressed = BUY-initiated
        //
        // More BUY-initiated trades = BULLISH → should be > 0
        //
        // Current formula: (trade_count_bid - trade_count_ask) / total
        // trade_count_bid = SELL-initiated, trade_count_ask = BUY-initiated
        // If more buys: trade_count_ask > trade_count_bid → result < 0 → WRONG
        //
        // CORRECT formula: (trade_count_ask - trade_count_bid) / total
        // If more buys: result > 0 → BULLISH ✓
        let mut aggregator = MboAggregator::new();

        // First add orders to enable trades
        for i in 0..20u64 {
            let side = if i < 10 { Side::Bid } else { Side::Ask };
            let price = if side == Side::Bid {
                100_000_000_000
            } else {
                100_010_000_000
            };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, price, 100, i);
            aggregator.process_event(event);
        }

        // 3 bid-side trades (SELL-initiated), 8 ask-side trades (BUY-initiated)
        for i in 0..3u64 {
            // Seller hitting the bid = Side::Bid in MBO
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Trade,
                Side::Bid,
                100_000_000_000,
                50,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..18u64 {
            // Buyer hitting the ask = Side::Ask in MBO
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Trade,
                Side::Ask,
                100_010_000_000,
                50,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_trade_flow = features[8]; // Index 8 = net_trade_flow

        // More ask-side trades (BUY-initiated) = BULLISH → should be > 0
        assert!(
            net_trade_flow > 0.0,
            "More ask-side trades (buyer aggression) should give net_trade_flow > 0 (BULLISH). \
             Got {:.4}. In MBO: Side::Ask on Trade = buyer hitting ask = BUY-initiated. \
             Sign convention per RULE.md §9: > 0 = BULLISH.",
            net_trade_flow
        );
    }

    #[test]
    fn test_net_trade_flow_sign_convention_bearish() {
        // More SELL-initiated trades = BEARISH → should be < 0
        let mut aggregator = MboAggregator::new();

        // Add orders
        for i in 0..20u64 {
            let side = if i < 10 { Side::Bid } else { Side::Ask };
            let price = if side == Side::Bid {
                100_000_000_000
            } else {
                100_010_000_000
            };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, price, 100, i);
            aggregator.process_event(event);
        }

        // 8 bid-side trades (SELL-initiated), 3 ask-side trades (BUY-initiated)
        for i in 0..8u64 {
            // Seller hitting the bid
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Trade,
                Side::Bid,
                100_000_000_000,
                50,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..13u64 {
            // Buyer hitting the ask
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Trade,
                Side::Ask,
                100_010_000_000,
                50,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_trade_flow = features[8];

        // More bid-side trades (SELL-initiated) = BEARISH → should be < 0
        assert!(
            net_trade_flow < 0.0,
            "More bid-side trades (seller aggression) should give net_trade_flow < 0 (BEARISH). \
             Got {:.4}. In MBO: Side::Bid on Trade = seller hitting bid = SELL-initiated. \
             Sign convention per RULE.md §9: < 0 = BEARISH.",
            net_trade_flow
        );
    }

    #[test]
    fn test_net_flow_symmetry() {
        // When bid count == ask count, all net flows should be exactly 0
        let mut aggregator = MboAggregator::new();

        // 10 each for each action type
        for i in 0..10u64 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }
        for i in 10..20u64 {
            let event = MboEvent::new(i * 1_000_000, Action::Add, Side::Ask, 100_010_000_000, 100, i);
            aggregator.process_event(event);
        }

        for i in 0..5u64 {
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..15u64 {
            let event = MboEvent::new(
                (20 + i) * 1_000_000,
                Action::Cancel,
                Side::Ask,
                100_010_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        for i in 0..5u64 {
            let event = MboEvent::new(
                (30 + i) * 1_000_000,
                Action::Trade,
                Side::Bid,
                100_000_000_000,
                50,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..15u64 {
            let event = MboEvent::new(
                (30 + i) * 1_000_000,
                Action::Trade,
                Side::Ask,
                100_010_000_000,
                50,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);

        let net_order_flow = features[6];
        let net_cancel_flow = features[7];
        let net_trade_flow = features[8];

        assert!(
            net_order_flow.abs() < 0.01,
            "Equal bid/ask adds should give net_order_flow ≈ 0. Got {:.4}",
            net_order_flow
        );
        assert!(
            net_cancel_flow.abs() < 0.01,
            "Equal bid/ask cancels should give net_cancel_flow ≈ 0. Got {:.4}",
            net_cancel_flow
        );
        assert!(
            net_trade_flow.abs() < 0.01,
            "Equal bid/ask trades should give net_trade_flow ≈ 0. Got {:.4}",
            net_trade_flow
        );
    }
}
