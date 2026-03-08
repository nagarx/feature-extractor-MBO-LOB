//! Rolling window of MBO events with incremental statistics.
//!
//! Maintains a fixed-size sliding window of recent events and incrementally
//! updates counters as events enter/leave the window. This enables O(1)
//! amortized performance for most operations.
//!
//! # Performance Characteristics
//!
//! - Push: O(1) amortized
//! - Counter access: O(1)
//! - Percentile recompute: O(n log n) but done infrequently
//! - Memory: O(capacity) = ~100 KB for typical window sizes

use super::event::MboEvent;
use mbo_lob_reconstructor::{Action, Side};
use std::collections::VecDeque;

pub struct MboWindow {
    /// Circular buffer of events (oldest at front, newest at back)
    pub(super) events: VecDeque<MboEvent>,

    /// Maximum number of events to store
    pub(super) capacity: usize,

    // ---- Incremental Counters (O(1) updates) ----
    pub(super) add_count_bid: usize,
    pub(super) add_count_ask: usize,
    pub(super) cancel_count_bid: usize,
    pub(super) cancel_count_ask: usize,
    pub(super) trade_count_bid: usize,
    pub(super) trade_count_ask: usize,
    pub(super) total_volume_bid: u64,
    pub(super) total_volume_ask: u64,

    // ---- Timestamps ----
    pub(super) first_ts: u64,
    pub(super) last_ts: u64,

    // ---- Cached Statistics ----
    size_percentiles: [f64; 4],
    size_cache_dirty: bool,
    size_mean: f64,
    size_std: f64,
    size_stats_dirty: bool,
}

impl MboWindow {
    /// Create a new rolling window with specified capacity.
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
        if self.events.len() == self.capacity {
            let old = self.events.pop_front().unwrap();
            self.decrement_counters(&old);
        }

        if self.events.is_empty() {
            self.first_ts = event.timestamp;
        }
        self.last_ts = event.timestamp;

        self.increment_counters(&event);
        self.events.push_back(event);

        self.size_cache_dirty = true;
        self.size_stats_dirty = true;
    }

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

        let sum: u64 = self.events.iter().map(|e| e.size as u64).sum();
        self.size_mean = sum as f64 / self.events.len() as f64;

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

#[cfg(test)]
mod tests {
    use super::*;

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

        for i in 0..5 {
            let event = MboEvent::new(i * 1000000, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            window.push(event);
        }

        assert_eq!(window.len(), 3);
        assert_eq!(window.add_count_bid, 3);
    }
}
