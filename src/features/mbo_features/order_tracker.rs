//! Order lifecycle tracking for MBO feature extraction.
//!
//! Encapsulates order lifecycle state: active orders, completed order statistics
//! (lifetimes, fill ratios, modification counts), and eviction of stale orders.
//! This is the single source of truth for all order-level tracking in the
//! MBO feature pipeline.

use super::event::MboEvent;
use mbo_lob_reconstructor::{Action, Side};
use std::collections::{BTreeMap, VecDeque};

/// Buffer size for tracking completed order lifetimes.
///
/// Matches the medium window size for consistency.
/// Memory cost: 1000 x 8 bytes = 8 KB (negligible).
pub(super) const COMPLETED_ORDER_BUFFER_SIZE: usize = 1000;

/// Maximum age (in nanoseconds) for orders in the tracker.
/// Orders older than this are evicted to prevent unbounded growth.
/// Default: 1 hour = 3.6e12 nanoseconds
pub(super) const MAX_ORDER_AGE_NS: u64 = 3_600_000_000_000;

/// Maximum number of orders to track before forcing eviction.
/// Prevents memory blowup if eviction based on age alone is insufficient.
/// Default: 50,000 orders (typical LOB has ~1,000-10,000)
pub(super) const MAX_ORDER_TRACKER_SIZE: usize = 50_000;

/// Eviction batch size: how many old orders to remove when limit is reached.
const EVICTION_BATCH_SIZE: usize = 5_000;

/// Order lifecycle information for tracking individual orders.
///
/// Tracks an order from creation to completion, recording modifications,
/// fills, and other lifecycle events.
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

/// Encapsulates all order lifecycle tracking state.
///
/// Manages the active order BTreeMap and completed order rolling buffers
/// (lifetimes, fill ratios, modification counts). Provides methods for
/// processing events, recording completions, and evicting stale orders.
pub(super) struct OrderTracker {
    /// Active orders keyed by order_id.
    /// BTreeMap ensures deterministic iteration order (ascending order_id),
    /// which makes eviction and aggregation reproducible across runs.
    active: BTreeMap<u64, OrderInfo>,

    /// Rolling buffer of completed order lifetimes (in seconds)
    completed_lifetimes: VecDeque<f64>,

    /// Rolling buffer of completed order fill ratios
    completed_fill_ratios: VecDeque<f64>,

    /// Rolling buffer of completed order modification counts
    completed_modifications: VecDeque<u8>,
}

impl OrderTracker {
    pub(super) fn new() -> Self {
        Self {
            active: BTreeMap::new(),
            completed_lifetimes: VecDeque::with_capacity(COMPLETED_ORDER_BUFFER_SIZE),
            completed_fill_ratios: VecDeque::with_capacity(COMPLETED_ORDER_BUFFER_SIZE),
            completed_modifications: VecDeque::with_capacity(COMPLETED_ORDER_BUFFER_SIZE),
        }
    }

    /// Process an MBO event to update order lifecycle state.
    #[inline]
    pub(super) fn process_event(&mut self, event: &MboEvent) {
        match event.action {
            Action::Add => {
                let info = OrderInfo::new(event.timestamp, event.size, event.price, event.side);
                self.active.insert(event.order_id, info);
            }
            Action::Modify => {
                if let Some(info) = self.active.get_mut(&event.order_id) {
                    info.add_modification(event.price);
                }
            }
            Action::Cancel => {
                if let Some(info) = self.active.remove(&event.order_id) {
                    self.record_completed(&info, event.timestamp);
                }
            }
            Action::Trade => {
                if let Some(info) = self.active.get_mut(&event.order_id) {
                    info.add_fill(event.timestamp, event.size);

                    if info.current_size == 0 {
                        if let Some(completed_info) = self.active.remove(&event.order_id) {
                            self.record_completed(&completed_info, event.timestamp);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Record a completed order's statistics in rolling buffers.
    #[inline]
    fn record_completed(&mut self, info: &OrderInfo, completion_time: u64) {
        let lifetime_ns = completion_time.saturating_sub(info.creation_time);
        let lifetime_secs = lifetime_ns as f64 / 1e9;
        let fill_ratio = info.fill_ratio();

        if self.completed_lifetimes.len() >= COMPLETED_ORDER_BUFFER_SIZE {
            self.completed_lifetimes.pop_front();
        }
        self.completed_lifetimes.push_back(lifetime_secs);

        if self.completed_fill_ratios.len() >= COMPLETED_ORDER_BUFFER_SIZE {
            self.completed_fill_ratios.pop_front();
        }
        self.completed_fill_ratios.push_back(fill_ratio);

        if self.completed_modifications.len() >= COMPLETED_ORDER_BUFFER_SIZE {
            self.completed_modifications.pop_front();
        }
        self.completed_modifications.push_back(info.modifications);
    }

    /// Evict old orders from the tracker to prevent unbounded growth.
    pub(super) fn evict_old(&mut self, current_time: u64) {
        if self.active.len() <= MAX_ORDER_TRACKER_SIZE {
            return;
        }

        let cutoff_time = current_time.saturating_sub(MAX_ORDER_AGE_NS);
        let orders_to_evict: Vec<u64> = self
            .active
            .iter()
            .filter(|(_, info)| info.creation_time < cutoff_time)
            .take(EVICTION_BATCH_SIZE)
            .map(|(&id, _)| id)
            .collect();

        for order_id in orders_to_evict {
            if let Some(info) = self.active.remove(&order_id) {
                self.record_completed(&info, current_time);
            }
        }
    }

    // ---- Accessors ----

    #[inline]
    pub(super) fn active_orders(&self) -> &BTreeMap<u64, OrderInfo> {
        &self.active
    }

    #[inline]
    pub(super) fn active_count(&self) -> usize {
        self.active.len()
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    #[inline]
    pub(super) fn completed_lifetimes(&self) -> &VecDeque<f64> {
        &self.completed_lifetimes
    }

    #[inline]
    pub(super) fn completed_fill_ratios(&self) -> &VecDeque<f64> {
        &self.completed_fill_ratios
    }

    #[inline]
    pub(super) fn completed_modifications(&self) -> &VecDeque<u8> {
        &self.completed_modifications
    }

    /// Direct mutable access to `active` for tests that need to insert directly.
    #[cfg(test)]
    pub(super) fn active_mut(&mut self) -> &mut BTreeMap<u64, OrderInfo> {
        &mut self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_info_fill_ratio() {
        let mut info = OrderInfo::new(0, 1000, 100_000_000_000, Side::Bid);

        info.add_fill(1000000, 300);
        assert_eq!(info.fill_ratio(), 0.3);

        info.add_fill(2000000, 400);
        assert_eq!(info.fill_ratio(), 0.7);
    }

    #[test]
    fn test_order_tracker_add_and_cancel() {
        let mut tracker = OrderTracker::new();

        let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&add);
        assert_eq!(tracker.active_count(), 1);

        let cancel = MboEvent::new(
            1_000_000_000,
            Action::Cancel,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        tracker.process_event(&cancel);
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_lifetimes().len(), 1);

        let lifetime = tracker.completed_lifetimes()[0];
        assert!(
            (lifetime - 1.0).abs() < 0.001,
            "Lifetime should be 1.0s, got {}",
            lifetime
        );
    }

    #[test]
    fn test_order_tracker_full_fill() {
        let mut tracker = OrderTracker::new();

        let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&add);

        let fill = MboEvent::new(
            2_000_000_000,
            Action::Trade,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        tracker.process_event(&fill);

        assert_eq!(tracker.active_count(), 0);
        assert!((tracker.completed_fill_ratios()[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_order_eviction_removes_old_orders() {
        let mut tracker = OrderTracker::new();

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
            tracker.process_event(&add);
        }

        tracker.evict_old(new_timestamp);

        assert!(
            tracker.active_count() <= MAX_ORDER_TRACKER_SIZE,
            "Order tracker should be bounded after eviction"
        );
    }
}
