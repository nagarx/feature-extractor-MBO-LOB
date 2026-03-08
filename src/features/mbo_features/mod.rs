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
//! - `OrderTracker`: Lifecycle tracking for individual orders
//! - `MboAggregator`: Main orchestrator combining windows and trackers
//!
//! # Sub-modules
//!
//! - `event`: MBO event representation (`MboEvent`)
//! - `window`: Rolling window with incremental statistics (`MboWindow`)
//! - `order_tracker`: Order lifecycle state (`OrderInfo`, `OrderTracker`)
//! - `flow_features`: 12 order flow features (indices 48-59)
//! - `size_features`: 8 size distribution features (indices 60-67)
//! - `queue_features`: 6 queue & depth features (indices 68-73)
//! - `institutional_features`: 4 institutional detection features (indices 74-77)
//! - `lifecycle_features`: 6 core MBO metrics (indices 78-83)
//!
//! # Performance
//!
//! - Event processing: <50 ns (maintains 1.6M+ msg/s throughput)
//! - Feature extraction: <100 ns per call
//! - Memory: ~2.6 MB per symbol (constant, no leaks)
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

mod event;
mod flow_features;
mod institutional_features;
mod lifecycle_features;
mod order_tracker;
mod queue_features;
mod size_features;
mod window;

pub use event::MboEvent;
pub use order_tracker::OrderInfo;

use mbo_lob_reconstructor::lob::queue_position::{QueuePositionConfig, QueuePositionTracker};
use mbo_lob_reconstructor::LobState;
use order_tracker::OrderTracker;
use window::MboWindow;

/// Default tick size in nanodollars: $0.01 for US equities
const DEFAULT_TICK_SIZE_NANODOLLARS: i64 = 10_000_000;

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
    fast_window: MboWindow,
    medium_window: MboWindow,
    slow_window: MboWindow,
    order_tracker: OrderTracker,
    queue_tracker: Option<QueuePositionTracker>,
    message_count: u64,
    tick_size_nanodollars: i64,
}

impl MboAggregator {
    /// Create a new MBO aggregator with default window sizes.
    ///
    /// Default tick size is $0.01 (US equity standard).
    /// Queue position tracking is disabled by default for performance.
    pub fn new() -> Self {
        Self {
            fast_window: MboWindow::new(100),
            medium_window: MboWindow::new(1000),
            slow_window: MboWindow::new(5000),
            order_tracker: OrderTracker::new(),
            queue_tracker: None,
            message_count: 0,
            tick_size_nanodollars: DEFAULT_TICK_SIZE_NANODOLLARS,
        }
    }

    /// Create aggregator with custom window sizes.
    pub fn with_windows(fast_size: usize, medium_size: usize, slow_size: usize) -> Self {
        Self {
            fast_window: MboWindow::new(fast_size),
            medium_window: MboWindow::new(medium_size),
            slow_window: MboWindow::new(slow_size),
            order_tracker: OrderTracker::new(),
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
    /// # Panics
    /// Panics if tick_size_dollars is <= 0.
    ///
    /// # Numerical Safety
    /// The minimum stored value is clamped to 1 nanodollar to prevent
    /// division-by-zero in depth_ticks calculations.
    pub fn with_tick_size(mut self, tick_size_dollars: f64) -> Self {
        assert!(
            tick_size_dollars > 0.0,
            "tick_size must be positive, got {}",
            tick_size_dollars
        );
        let raw_nanodollars = (tick_size_dollars * 1e9).round() as i64;
        self.tick_size_nanodollars = raw_nanodollars.max(1);
        self
    }

    /// Get the current tick size in nanodollars.
    #[inline]
    pub fn tick_size_nanodollars(&self) -> i64 {
        self.tick_size_nanodollars
    }

    /// Enable queue position tracking for accurate queue metrics.
    ///
    /// When enabled, `average_queue_position` and `queue_size_ahead` features
    /// return accurate FIFO-based values instead of 0.0.
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
    #[inline]
    pub fn process_event(&mut self, event: MboEvent) {
        self.fast_window.push(event.clone());
        self.medium_window.push(event.clone());
        self.slow_window.push(event.clone());

        self.order_tracker.process_event(&event);

        if let Some(ref mut tracker) = self.queue_tracker {
            let msg = event.to_mbo_message();
            tracker.process_message(&msg);
        }

        self.message_count += 1;

        if self.message_count.is_multiple_of(1000) {
            self.order_tracker.evict_old(event.timestamp);
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

        features.extend(
            flow_features::extract(&self.medium_window, &self.fast_window, &self.slow_window)
                .iter(),
        );

        features.extend(size_features::extract(&mut self.medium_window).iter());

        features.extend(
            queue_features::extract(
                lob,
                self.queue_tracker.as_ref(),
                &self.order_tracker,
                self.tick_size_nanodollars,
            )
            .iter(),
        );

        features.extend(
            institutional_features::extract(&mut self.medium_window, &self.order_tracker).iter(),
        );

        features
            .extend(lifecycle_features::extract(&self.order_tracker, &self.medium_window).iter());

        features
    }
}

impl Default for MboAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mbo_lob_reconstructor::{Action, Side};
    use order_tracker::{MAX_ORDER_AGE_NS, MAX_ORDER_TRACKER_SIZE};

    #[test]
    fn test_mbo_aggregator_creation() {
        let aggregator = MboAggregator::new();
        assert_eq!(aggregator.message_count, 0);
        assert!(aggregator.order_tracker.is_empty());
    }

    #[test]
    fn test_tick_size_normal_value() {
        let aggregator = MboAggregator::new().with_tick_size(0.01);
        assert_eq!(aggregator.tick_size_nanodollars(), 10_000_000);
    }

    #[test]
    fn test_tick_size_small_value() {
        let aggregator = MboAggregator::new().with_tick_size(0.0001);
        assert_eq!(aggregator.tick_size_nanodollars(), 100_000);
    }

    #[test]
    fn test_tick_size_very_small_clamped() {
        let aggregator = MboAggregator::new().with_tick_size(1e-12);
        assert_eq!(aggregator.tick_size_nanodollars(), 1);
    }

    #[test]
    fn test_tick_size_prevents_division_by_zero() {
        let mut aggregator = MboAggregator::new().with_tick_size(1e-15);
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000);
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_prices[1] = 99_990_000_000;
        lob.bid_sizes[0] = 100;
        lob.bid_sizes[1] = 100;

        let features = aggregator.extract_features(&lob);
        assert!(features[24].is_finite(), "depth_ticks_bid should be finite");
    }

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

    // Integration: verify 36 features with populated data
    #[test]
    fn test_extract_features_with_populated_data() {
        let mut aggregator = MboAggregator::new();

        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_prices[1] = 99_990_000_000;
        lob.bid_sizes[0] = 500;
        lob.bid_sizes[1] = 300;
        lob.ask_prices[0] = 100_010_000_000;
        lob.ask_prices[1] = 100_020_000_000;
        lob.ask_sizes[0] = 400;
        lob.ask_sizes[1] = 200;

        for i in 0..200u64 {
            let side = if i % 3 == 0 { Side::Bid } else { Side::Ask };
            let size = 50 + ((i * 17) % 200) as u32;
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, 100_000_000_000, size, i);
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&lob);
        assert_eq!(features.len(), 36, "Should have 36 features");

        let level_conc = features[23];
        assert!(level_conc > 0.0 && level_conc <= 1.0);

        let depth_bid = features[24];
        assert!(depth_bid >= 0.0);

        let depth_ask = features[25];
        assert!(depth_ask >= 0.0);

        let size_skew = features[18];
        assert!(size_skew.is_finite());

        let size_conc = features[19];
        assert!(size_conc > 0.0 && size_conc <= 1.0);

        let flow_vol = features[10];
        assert!(flow_vol >= 0.0);
    }

    // Sign convention integration tests
    #[test]
    fn test_net_order_flow_sign_convention_bullish() {
        let mut aggregator = MboAggregator::new();

        for i in 0..10u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..15u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Ask,
                100_010_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        let net_order_flow = features[6];

        assert!(
            net_order_flow > 0.0,
            "More bid adds should give net_order_flow > 0 (BULLISH). Got {:.4}",
            net_order_flow
        );

        let expected = (10.0 - 5.0) / 15.0;
        assert!(
            (net_order_flow - expected).abs() < 0.01,
            "Expected {:.4}, got {:.4}",
            expected,
            net_order_flow
        );
    }

    #[test]
    fn test_net_order_flow_sign_convention_bearish() {
        let mut aggregator = MboAggregator::new();

        for i in 0..5u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 5..15u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Ask,
                100_010_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        assert!(
            features[6] < 0.0,
            "More ask adds should be bearish, got {:.4}",
            features[6]
        );
    }

    #[test]
    fn test_net_cancel_flow_sign_convention_bullish() {
        let mut aggregator = MboAggregator::new();

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
        assert!(
            features[7] > 0.0,
            "More ask cancels should be bullish, got {:.4}",
            features[7]
        );
    }

    #[test]
    fn test_net_cancel_flow_sign_convention_bearish() {
        let mut aggregator = MboAggregator::new();

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
        assert!(
            features[7] < 0.0,
            "More bid cancels should be bearish, got {:.4}",
            features[7]
        );
    }

    #[test]
    fn test_net_trade_flow_sign_convention_bullish() {
        let mut aggregator = MboAggregator::new();

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

        for i in 0..3u64 {
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
        assert!(
            features[8] > 0.0,
            "More buy-initiated trades should be bullish, got {:.4}",
            features[8]
        );
    }

    #[test]
    fn test_net_trade_flow_sign_convention_bearish() {
        let mut aggregator = MboAggregator::new();

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

        for i in 0..8u64 {
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
        assert!(
            features[8] < 0.0,
            "More sell-initiated trades should be bearish, got {:.4}",
            features[8]
        );
    }

    #[test]
    fn test_net_flow_symmetry() {
        let mut aggregator = MboAggregator::new();

        for i in 0..10u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 10..20u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Ask,
                100_010_000_000,
                100,
                i,
            );
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
        assert!(
            features[6].abs() < 0.01,
            "Balanced adds should be ~0, got {:.4}",
            features[6]
        );
        assert!(
            features[7].abs() < 0.01,
            "Balanced cancels should be ~0, got {:.4}",
            features[7]
        );
        assert!(
            features[8].abs() < 0.01,
            "Balanced trades should be ~0, got {:.4}",
            features[8]
        );
    }

    // Numerical stability
    #[test]
    fn test_flow_regime_indicator_normal() {
        let mut aggregator = MboAggregator::with_windows(50, 200, 500);

        for i in 0..500u64 {
            let side = if i % 3 == 0 { Side::Ask } else { Side::Bid };
            let event = MboEvent::new(i * 1_000_000, Action::Add, side, 100_000_000_000, 100, i);
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&LobState::new(10));
        let regime = features[11];
        assert!(
            regime >= -10.0 && regime <= 10.0,
            "Should be clamped, got {}",
            regime
        );
    }

    #[test]
    fn test_flow_regime_indicator_empty_windows() {
        let mut aggregator = MboAggregator::new();
        let features = aggregator.extract_features(&LobState::new(10));
        let regime = features[11];
        assert!(
            regime.abs() <= 10.0,
            "Empty should be bounded, got {}",
            regime
        );
    }

    // cancel_to_add_ratio integration tests
    #[test]
    fn test_cancel_to_add_ratio_integration() {
        let mut aggregator = MboAggregator::new();

        for i in 0..10u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }
        for i in 0..5u64 {
            let event = MboEvent::new(
                (10 + i) * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&LobState::new(10));
        let ratio = features[34];
        assert!((ratio - 0.5).abs() < 0.01, "5/10 = 0.5, got {}", ratio);
    }

    #[test]
    fn test_cancel_to_add_ratio_capped_integration() {
        let mut aggregator = MboAggregator::new();

        let event = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        aggregator.process_event(event);

        for i in 0..100u64 {
            let event = MboEvent::new(
                (2 + i) * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&LobState::new(10));
        assert_eq!(
            features[34], 10.0,
            "Should be capped at 10.0, got {}",
            features[34]
        );
    }

    // median_order_lifetime integration
    #[test]
    fn test_median_order_lifetime_feature_extraction() {
        let mut aggregator = MboAggregator::new();

        for i in 0..3u64 {
            let lifetime_ns = (i + 1) * 1_000_000_000;
            let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(add);
            let cancel = MboEvent::new(
                lifetime_ns,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(cancel);
        }

        let features = aggregator.extract_features(&LobState::new(10));
        let median = features[31];
        assert!(
            (median - 2.0).abs() < 0.001,
            "Median of [1,2,3] should be 2.0, got {}",
            median
        );
    }

    // Queue tracking integration tests
    #[test]
    fn test_average_queue_position_no_tracking() {
        let mut aggregator = MboAggregator::new();
        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        assert_eq!(features[20], 0.0);
    }

    #[test]
    fn test_average_queue_position_single_order() {
        let mut aggregator = MboAggregator::new().with_queue_tracking();
        let event = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        aggregator.process_event(event);
        let lob = LobState::new(10);
        let features = aggregator.extract_features(&lob);
        assert!(
            features[20].abs() < 1e-10,
            "Single order should be at position 0, got {}",
            features[20]
        );
    }

    #[test]
    fn test_average_queue_position_multiple_orders_same_price() {
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        for i in 1..=3u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(event);
        }

        let features = aggregator.extract_features(&LobState::new(10));
        assert!(
            (features[20] - 1.0).abs() < 1e-10,
            "Avg of positions 0,1,2 should be 1.0, got {}",
            features[20]
        );
    }

    #[test]
    fn test_queue_size_ahead_fifo_order() {
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        let event1 = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        let event2 = MboEvent::new(2_000_000, Action::Add, Side::Bid, 100_000_000_000, 200, 2);
        let event3 = MboEvent::new(3_000_000, Action::Add, Side::Bid, 100_000_000_000, 150, 3);

        aggregator.process_event(event1);
        aggregator.process_event(event2);
        aggregator.process_event(event3);

        let features = aggregator.extract_features(&LobState::new(10));
        let expected = (0.0 + 100.0 + 300.0) / 3.0;
        assert!(
            (features[21] - expected).abs() < 1.0,
            "Expected ~{}, got {}",
            expected,
            features[21]
        );
    }

    #[test]
    fn test_queue_tracking_both_sides() {
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        let bid1 = MboEvent::new(1_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        let bid2 = MboEvent::new(2_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 2);
        let ask1 = MboEvent::new(3_000_000, Action::Add, Side::Ask, 101_000_000_000, 100, 3);
        let ask2 = MboEvent::new(4_000_000, Action::Add, Side::Ask, 101_000_000_000, 100, 4);

        aggregator.process_event(bid1);
        aggregator.process_event(bid2);
        aggregator.process_event(ask1);
        aggregator.process_event(ask2);

        let features = aggregator.extract_features(&LobState::new(10));
        assert!(
            (features[20] - 0.5).abs() < 1e-10,
            "Combined avg should be 0.5, got {}",
            features[20]
        );
    }

    #[test]
    fn test_extract_features_with_queue_tracking() {
        let mut aggregator = MboAggregator::new().with_queue_tracking();

        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_sizes[0] = 500;
        lob.ask_prices[0] = 100_010_000_000;
        lob.ask_sizes[0] = 400;

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
        assert!(
            features[20] > 0.0,
            "queue_position should be > 0 with tracking"
        );
        assert!(
            features[21] > 0.0,
            "volume_ahead should be > 0 with tracking"
        );
    }

    // Completed order lifetime buffer bound test
    #[test]
    fn test_median_order_lifetime_buffer_bounded() {
        let mut aggregator = MboAggregator::new();

        for i in 0..1100u64 {
            let lifetime_ns = (i + 1) * 1_000_000;
            let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            aggregator.process_event(add);
            let cancel = MboEvent::new(
                lifetime_ns,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(cancel);
        }

        let features = aggregator.extract_features(&LobState::new(10));
        let median = features[31];
        assert!(
            median > 0.5 && median < 0.7,
            "Should be ~0.6s, got {}",
            median
        );
    }

    // Fill ratio mixed outcomes
    #[test]
    fn test_average_fill_ratio_mixed_outcomes() {
        let mut aggregator = MboAggregator::new();

        let add1 = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        aggregator.process_event(add1);
        let cancel1 = MboEvent::new(
            1_000_000,
            Action::Cancel,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        aggregator.process_event(cancel1);

        let add2 = MboEvent::new(2_000_000, Action::Add, Side::Ask, 100_000_000_000, 200, 2);
        aggregator.process_event(add2);
        let fill2 = MboEvent::new(3_000_000, Action::Trade, Side::Ask, 100_000_000_000, 200, 2);
        aggregator.process_event(fill2);

        let add3 = MboEvent::new(4_000_000, Action::Add, Side::Bid, 100_000_000_000, 100, 3);
        aggregator.process_event(add3);
        let fill3 = MboEvent::new(5_000_000, Action::Trade, Side::Bid, 100_000_000_000, 50, 3);
        aggregator.process_event(fill3);
        let cancel3 = MboEvent::new(6_000_000, Action::Cancel, Side::Bid, 100_000_000_000, 50, 3);
        aggregator.process_event(cancel3);

        let features = aggregator.extract_features(&LobState::new(10));
        let fill_ratio = features[32]; // average_fill_ratio is Core[2]
        assert!(
            (fill_ratio - 0.5).abs() < 0.001,
            "Average of [0, 1.0, 0.5] should be 0.5, got {}",
            fill_ratio
        );
    }

    // Order eviction tests
    #[test]
    fn test_order_eviction_called_periodically() {
        let mut aggregator = MboAggregator::new();

        for i in 0..1100u64 {
            let add = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(add);
        }

        assert_eq!(aggregator.message_count, 1100);
        assert!(aggregator.order_tracker.active_count() <= MAX_ORDER_TRACKER_SIZE);
    }

    #[test]
    fn test_order_eviction_removes_old_orders() {
        let mut aggregator = MboAggregator::new();

        let old_timestamp = 0u64;
        let new_timestamp = MAX_ORDER_AGE_NS + 1_000_000_000;

        for i in 0..1000u64 {
            let add = MboEvent::new(
                old_timestamp,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            aggregator.process_event(add);
        }

        aggregator.order_tracker.evict_old(new_timestamp);
        assert!(aggregator.order_tracker.active_count() <= MAX_ORDER_TRACKER_SIZE);
    }

    #[test]
    fn test_evicted_orders_recorded_in_completed_buffers() {
        let mut aggregator = MboAggregator::new();

        let old_timestamp = 0u64;
        let new_timestamp = MAX_ORDER_AGE_NS + 1_000_000_000;

        let add = MboEvent::new(
            old_timestamp,
            Action::Add,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        aggregator.process_event(add);

        for i in 2..(MAX_ORDER_TRACKER_SIZE as u64 + 2) {
            aggregator.order_tracker.active_mut().insert(
                i,
                OrderInfo::new(old_timestamp, 100, 100_000_000_000, Side::Bid),
            );
        }

        aggregator.order_tracker.evict_old(new_timestamp);

        assert!(
            !aggregator.order_tracker.completed_fill_ratios().is_empty()
                || aggregator.order_tracker.active_count() < MAX_ORDER_TRACKER_SIZE,
            "Eviction should either record completed orders or reduce tracker size"
        );
    }
}
