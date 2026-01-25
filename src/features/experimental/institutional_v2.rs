//! Enhanced Institutional Detection Features (V2)
//!
//! This module provides 8 features designed to detect institutional
//! trading patterns that are not captured by the base MBO features.
//!
//! # Feature Index Layout (when experimental enabled)
//!
//! | Offset | Name | Description | Sign Convention |
//! |--------|------|-------------|-----------------|
//! | 0 | round_lot_ratio | % orders divisible by round_lot_size | N/A |
//! | 1 | odd_lot_ratio | % orders < round_lot_size | N/A |
//! | 2 | size_clustering | Std dev of consecutive sizes (low = splitting) | N/A |
//! | 3 | price_clustering | Orders at same price / total orders | N/A |
//! | 4 | mod_before_cancel | Orders modified then cancelled / cancelled | N/A |
//! | 5 | sweep_ratio | Multi-level consuming orders / total trades | N/A |
//! | 6 | fill_patience_bid | Avg fill time for large bid orders (seconds) | N/A |
//! | 7 | fill_patience_ask | Avg fill time for large ask orders (seconds) | N/A |
//!
//! # Institutional Patterns Detected
//!
//! 1. **Order Splitting**: Large orders split into many similar-sized pieces
//!    - Detected via `size_clustering` (low std dev of consecutive sizes)
//!
//! 2. **Round Lot Preference**: Institutions often trade round lots (100, 200, etc.)
//!    - Detected via `round_lot_ratio` (high = more institutional)
//!
//! 3. **Price Clustering**: Many orders at same price = iceberg or accumulation
//!    - Detected via `price_clustering`
//!
//! 4. **Modification Patterns**: HFT/algos modify orders before cancelling
//!    - Detected via `mod_before_cancel`
//!
//! 5. **Sweep Orders**: Orders that consume multiple price levels
//!    - Detected via `sweep_ratio`
//!
//! 6. **Patient Execution**: Institutions are patient, wait for fills
//!    - Detected via `fill_patience_*` (higher = more patient)
//!
//! # References
//!
//! - Hasbrouck, J. (2018). "High-Frequency Quoting: Short-Term Volatility in Bids and Offers"
//! - Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events"

use crate::features::mbo_features::MboEvent;
use mbo_lob_reconstructor::{Action, Side};
use std::collections::{HashMap, VecDeque};

/// Number of features in this group.
pub const FEATURE_COUNT: usize = 8;

/// Feature indices (relative to group start).
pub mod indices {
    pub const ROUND_LOT_RATIO: usize = 0;
    pub const ODD_LOT_RATIO: usize = 1;
    pub const SIZE_CLUSTERING: usize = 2;
    pub const PRICE_CLUSTERING: usize = 3;
    pub const MOD_BEFORE_CANCEL: usize = 4;
    pub const SWEEP_RATIO: usize = 5;
    pub const FILL_PATIENCE_BID: usize = 6;
    pub const FILL_PATIENCE_ASK: usize = 7;
}

/// Numerical stability constant.
const EPSILON: f64 = 1e-10;

/// Order state for tracking modifications and fills.
#[derive(Debug, Clone)]
struct OrderState {
    creation_time: u64,
    original_size: u32,
    #[allow(dead_code)] // Reserved for future price-based clustering detection
    price: i64,
    side: Side,
    was_modified: bool,
    first_fill_time: Option<u64>,
}

/// Enhanced institutional detector with additional pattern recognition.
pub struct InstitutionalDetectorV2 {
    /// Window of recent events.
    events: VecDeque<MboEvent>,
    
    /// Maximum window size.
    window_size: usize,
    
    /// Threshold for large orders (percentile, e.g., 90.0).
    large_percentile: f64,
    
    /// Round lot size (e.g., 100 for US equities).
    round_lot_size: u32,
    
    /// Active order tracker.
    orders: HashMap<u64, OrderState>,
    
    /// Count of orders modified before cancel.
    mod_before_cancel_count: usize,
    
    /// Total cancel count.
    cancel_count: usize,
    
    /// Count of sweep-like trades (consuming multiple levels).
    sweep_count: usize,
    
    /// Total trade count.
    trade_count: usize,
    
    /// Fill times for large bid orders (in nanoseconds).
    large_bid_fill_times: VecDeque<u64>,
    
    /// Fill times for large ask orders.
    large_ask_fill_times: VecDeque<u64>,
    
    /// Consecutive order sizes for clustering detection.
    consecutive_sizes: VecDeque<u32>,
    
    /// Price counts for clustering detection.
    price_counts: HashMap<i64, usize>,
    
    /// Last trade price (for sweep detection).
    last_trade_price: Option<i64>,
    
    /// Warmup counter.
    event_count: usize,
}

impl InstitutionalDetectorV2 {
    /// Create a new institutional detector.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of events to track
    /// * `large_percentile` - Percentile threshold for "large" orders (e.g., 90.0)
    /// * `round_lot_size` - Round lot size (e.g., 100 for US equities)
    pub fn new(window_size: usize, large_percentile: f64, round_lot_size: u32) -> Self {
        Self {
            events: VecDeque::with_capacity(window_size),
            window_size,
            large_percentile,
            round_lot_size,
            orders: HashMap::with_capacity(1000),
            mod_before_cancel_count: 0,
            cancel_count: 0,
            sweep_count: 0,
            trade_count: 0,
            large_bid_fill_times: VecDeque::with_capacity(100),
            large_ask_fill_times: VecDeque::with_capacity(100),
            consecutive_sizes: VecDeque::with_capacity(50),
            price_counts: HashMap::with_capacity(100),
            last_trade_price: None,
            event_count: 0,
        }
    }

    /// Process an MBO event.
    #[inline]
    pub fn process_event(&mut self, event: &MboEvent) {
        self.event_count += 1;

        // Evict old event if at capacity
        if self.events.len() >= self.window_size {
            if let Some(old) = self.events.pop_front() {
                self.remove_event_stats(&old);
            }
        }

        // Add event stats
        self.add_event_stats(event);
        self.events.push_back(event.clone());
    }

    /// Add statistics for a new event.
    fn add_event_stats(&mut self, event: &MboEvent) {
        // Track consecutive sizes for Add orders
        if matches!(event.action, Action::Add) {
            self.consecutive_sizes.push_back(event.size);
            if self.consecutive_sizes.len() > 50 {
                self.consecutive_sizes.pop_front();
            }

            // Track price clustering
            *self.price_counts.entry(event.price).or_insert(0) += 1;

            // Track order creation
            self.orders.insert(
                event.order_id,
                OrderState {
                    creation_time: event.timestamp,
                    original_size: event.size,
                    price: event.price,
                    side: event.side,
                    was_modified: false,
                    first_fill_time: None,
                },
            );
        }

        // Track modifications
        if matches!(event.action, Action::Modify) {
            if let Some(order) = self.orders.get_mut(&event.order_id) {
                order.was_modified = true;
            }
        }

        // Track cancellations
        if matches!(event.action, Action::Cancel) {
            self.cancel_count += 1;

            if let Some(order) = self.orders.remove(&event.order_id) {
                if order.was_modified {
                    self.mod_before_cancel_count += 1;
                }
            }

            // Decrement price count
            if let Some(count) = self.price_counts.get_mut(&event.price) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.price_counts.remove(&event.price);
                }
            }
        }

        // Track trades
        if matches!(event.action, Action::Trade) {
            self.trade_count += 1;

            // Sweep detection: trade at different price than last
            if let Some(last_price) = self.last_trade_price {
                if event.price != last_price {
                    self.sweep_count += 1;
                }
            }
            self.last_trade_price = Some(event.price);

            // Compute size threshold BEFORE borrowing orders to avoid borrow conflict
            let size_threshold = self.compute_large_threshold();

            // Track fill times for large orders
            if let Some(order) = self.orders.get_mut(&event.order_id) {
                if order.first_fill_time.is_none() {
                    order.first_fill_time = Some(event.timestamp);

                    // Check if this is a "large" order
                    if order.original_size >= size_threshold {
                        let fill_time = event.timestamp.saturating_sub(order.creation_time);
                        match order.side {
                            Side::Bid => {
                                self.large_bid_fill_times.push_back(fill_time);
                                if self.large_bid_fill_times.len() > 50 {
                                    self.large_bid_fill_times.pop_front();
                                }
                            }
                            Side::Ask => {
                                self.large_ask_fill_times.push_back(fill_time);
                                if self.large_ask_fill_times.len() > 50 {
                                    self.large_ask_fill_times.pop_front();
                                }
                            }
                            Side::None => {
                                // Ignore orders with no side
                            }
                        }
                    }
                }
            }
        }
    }

    /// Remove statistics for an evicted event.
    fn remove_event_stats(&mut self, event: &MboEvent) {
        // Decrement counters based on event type
        match event.action {
            Action::Cancel => {
                self.cancel_count = self.cancel_count.saturating_sub(1);
            }
            Action::Trade => {
                self.trade_count = self.trade_count.saturating_sub(1);
            }
            _ => {}
        }
    }

    /// Compute the size threshold for "large" orders.
    fn compute_large_threshold(&self) -> u32 {
        if self.events.is_empty() {
            return 1000; // Default
        }

        // Get sizes of Add orders
        let mut sizes: Vec<u32> = self
            .events
            .iter()
            .filter(|e| matches!(e.action, Action::Add))
            .map(|e| e.size)
            .collect();

        if sizes.is_empty() {
            return 1000;
        }

        sizes.sort_unstable();
        let idx = ((sizes.len() as f64 * self.large_percentile / 100.0) as usize)
            .min(sizes.len() - 1);
        sizes[idx]
    }

    /// Extract all 8 features into the output buffer.
    pub fn extract_into(&mut self, output: &mut Vec<f64>) {
        let n = self.events.len() as f64;
        
        if n < 10.0 {
            // Not enough data, output zeros
            output.extend_from_slice(&[0.0; FEATURE_COUNT]);
            return;
        }

        // Feature 0: Round lot ratio
        let add_events: Vec<_> = self
            .events
            .iter()
            .filter(|e| matches!(e.action, Action::Add))
            .collect();
        let n_adds = add_events.len() as f64;
        
        let round_lots = add_events
            .iter()
            .filter(|e| e.size % self.round_lot_size == 0 && e.size > 0)
            .count() as f64;
        let round_lot_ratio = if n_adds > 0.0 {
            round_lots / n_adds
        } else {
            0.0
        };
        output.push(round_lot_ratio);

        // Feature 1: Odd lot ratio
        let odd_lots = add_events
            .iter()
            .filter(|e| e.size < self.round_lot_size && e.size > 0)
            .count() as f64;
        let odd_lot_ratio = if n_adds > 0.0 {
            odd_lots / n_adds
        } else {
            0.0
        };
        output.push(odd_lot_ratio);

        // Feature 2: Size clustering (std dev of consecutive sizes)
        let size_clustering = if self.consecutive_sizes.len() >= 2 {
            let sizes: Vec<f64> = self.consecutive_sizes.iter().map(|&s| s as f64).collect();
            let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
            let variance =
                sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
            // Normalize by mean to get coefficient of variation
            if mean > EPSILON {
                variance.sqrt() / mean
            } else {
                1.0
            }
        } else {
            1.0 // High uncertainty when no data
        };
        output.push(size_clustering);

        // Feature 3: Price clustering
        let total_at_prices: usize = self.price_counts.values().sum();
        let max_at_price = self.price_counts.values().max().copied().unwrap_or(0);
        let price_clustering = if total_at_prices > 0 {
            max_at_price as f64 / total_at_prices as f64
        } else {
            0.0
        };
        output.push(price_clustering);

        // Feature 4: Modification before cancel ratio
        let mod_before_cancel = if self.cancel_count > 0 {
            self.mod_before_cancel_count as f64 / self.cancel_count as f64
        } else {
            0.0
        };
        output.push(mod_before_cancel);

        // Feature 5: Sweep ratio
        let sweep_ratio = if self.trade_count > 1 {
            self.sweep_count as f64 / (self.trade_count - 1).max(1) as f64
        } else {
            0.0
        };
        output.push(sweep_ratio);

        // Feature 6: Fill patience for large bid orders (in seconds)
        let fill_patience_bid = if !self.large_bid_fill_times.is_empty() {
            let sum: u64 = self.large_bid_fill_times.iter().sum();
            (sum as f64 / self.large_bid_fill_times.len() as f64) / 1e9
        } else {
            0.0
        };
        output.push(fill_patience_bid);

        // Feature 7: Fill patience for large ask orders (in seconds)
        let fill_patience_ask = if !self.large_ask_fill_times.is_empty() {
            let sum: u64 = self.large_ask_fill_times.iter().sum();
            (sum as f64 / self.large_ask_fill_times.len() as f64) / 1e9
        } else {
            0.0
        };
        output.push(fill_patience_ask);
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.events.clear();
        self.orders.clear();
        self.mod_before_cancel_count = 0;
        self.cancel_count = 0;
        self.sweep_count = 0;
        self.trade_count = 0;
        self.large_bid_fill_times.clear();
        self.large_ask_fill_times.clear();
        self.consecutive_sizes.clear();
        self.price_counts.clear();
        self.last_trade_price = None;
        self.event_count = 0;
    }

    /// Check if enough data for valid features.
    pub fn is_warm(&self) -> bool {
        self.event_count >= self.window_size / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(action: Action, side: Side, size: u32, price: i64, order_id: u64) -> MboEvent {
        MboEvent {
            timestamp: 1_000_000_000,
            action,
            side,
            price,
            size,
            order_id,
        }
    }

    #[test]
    fn test_round_lot_detection() {
        let mut detector = InstitutionalDetectorV2::new(100, 90.0, 100);

        // Add some round lot orders
        for i in 0..50 {
            let size = if i % 2 == 0 { 100 } else { 200 }; // All round lots
            detector.process_event(&make_event(
                Action::Add,
                Side::Bid,
                size,
                100_000_000_000,
                i,
            ));
        }

        let mut output = Vec::new();
        detector.extract_into(&mut output);

        assert_eq!(output.len(), FEATURE_COUNT);
        // Round lot ratio should be 1.0 (all round lots)
        assert!((output[indices::ROUND_LOT_RATIO] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_odd_lot_detection() {
        let mut detector = InstitutionalDetectorV2::new(100, 90.0, 100);

        // Add some odd lot orders (< 100)
        for i in 0..50 {
            let size = 50; // All odd lots
            detector.process_event(&make_event(
                Action::Add,
                Side::Bid,
                size,
                100_000_000_000,
                i,
            ));
        }

        let mut output = Vec::new();
        detector.extract_into(&mut output);

        // Odd lot ratio should be 1.0 (all odd lots)
        assert!((output[indices::ODD_LOT_RATIO] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_feature_count() {
        assert_eq!(FEATURE_COUNT, 8);
    }
}
