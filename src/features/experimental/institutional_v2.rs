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

use crate::contract::FLOAT_CMP_EPS;

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

/// Event metadata stored alongside the raw event for windowed computation.
/// This enables accurate rolling statistics without cumulative counter bugs.
#[derive(Debug, Clone)]
struct EventMetadata {
    /// The original event.
    event: MboEvent,
    /// For Trade events: was this a sweep (price changed from previous trade)?
    is_sweep: bool,
    /// For Cancel events: was the order modified before being cancelled?
    was_modified_before_cancel: bool,
}

/// Enhanced institutional detector with additional pattern recognition.
///
/// # Design Notes
///
/// This module uses windowed computation for accurate rolling statistics.
/// Rather than maintaining cumulative counters (which can become unsynchronized
/// when events are evicted), we store event metadata and compute ratios
/// on-demand from the current window.
pub struct InstitutionalDetectorV2 {
    /// Window of recent events with metadata for windowed computation.
    events: VecDeque<EventMetadata>,
    
    /// Maximum window size.
    window_size: usize,
    
    /// Threshold for large orders (percentile, e.g., 90.0).
    large_percentile: f64,
    
    /// Round lot size (e.g., 100 for US equities).
    round_lot_size: u32,
    
    /// Active order tracker.
    orders: HashMap<u64, OrderState>,
    
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
                self.remove_event_stats(&old.event);
            }
        }

        // Compute metadata flags BEFORE adding event stats (for sweep detection)
        let is_sweep = if matches!(event.action, Action::Trade) {
            if let Some(last_price) = self.last_trade_price {
                event.price != last_price
            } else {
                false
            }
        } else {
            false
        };

        let was_modified_before_cancel = if matches!(event.action, Action::Cancel) {
            self.orders
                .get(&event.order_id)
                .map(|o| o.was_modified)
                .unwrap_or(false)
        } else {
            false
        };

        // Add event stats (updates orders, price_counts, etc.)
        self.add_event_stats(event);

        // Store event with computed metadata
        self.events.push_back(EventMetadata {
            event: event.clone(),
            is_sweep,
            was_modified_before_cancel,
        });
    }

    /// Add statistics for a new event.
    /// 
    /// Note: sweep_count and mod_before_cancel_count are computed on-demand
    /// from event metadata to ensure accurate windowed statistics.
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
            // Remove from active orders (metadata already captured was_modified)
            self.orders.remove(&event.order_id);

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
            // Update last trade price for sweep detection
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
    /// 
    /// Note: sweep_count and mod_before_cancel are computed on-demand from
    /// event metadata, so no counter updates needed here.
    fn remove_event_stats(&mut self, _event: &MboEvent) {
        // Statistics are computed on-demand from the windowed events with metadata.
        // This avoids counter synchronization bugs that occurred with cumulative counters.
        //
        // The following stats ARE still updated incrementally:
        // - consecutive_sizes (bounded deque)
        // - price_counts (for add/cancel only, cleaned up in add_event_stats)
        // - orders (active order tracker)
        // - large_*_fill_times (bounded deques)
        //
        // The following are computed on-demand in extract_into:
        // - sweep_ratio (from Trade events with is_sweep flag)
        // - mod_before_cancel (from Cancel events with was_modified_before_cancel flag)
        // - cancel_count / trade_count (counted from events)
    }

    /// Compute the size threshold for "large" orders.
    fn compute_large_threshold(&self) -> u32 {
        if self.events.is_empty() {
            return 1000; // Default
        }

        // Get sizes of Add orders from windowed events
        let mut sizes: Vec<u32> = self
            .events
            .iter()
            .filter(|em| matches!(em.event.action, Action::Add))
            .map(|em| em.event.size)
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
    ///
    /// All ratios are computed on-demand from the current window to ensure
    /// accurate windowed statistics without counter synchronization issues.
    pub fn extract_into(&mut self, output: &mut Vec<f64>) {
        let n = self.events.len() as f64;
        
        if n < 10.0 {
            // Not enough data, output zeros
            output.extend_from_slice(&[0.0; FEATURE_COUNT]);
            return;
        }

        // Collect Add events from windowed events
        let add_events: Vec<_> = self
            .events
            .iter()
            .filter(|em| matches!(em.event.action, Action::Add))
            .map(|em| &em.event)
            .collect();
        let n_adds = add_events.len() as f64;
        
        // Feature 0: Round lot ratio
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
            if mean > FLOAT_CMP_EPS {
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
        // Computed on-demand from windowed events to ensure accurate rolling statistics
        let cancel_events: Vec<_> = self
            .events
            .iter()
            .filter(|em| matches!(em.event.action, Action::Cancel))
            .collect();
        let cancel_count = cancel_events.len();
        let mod_before_cancel_count = cancel_events
            .iter()
            .filter(|em| em.was_modified_before_cancel)
            .count();
        let mod_before_cancel = if cancel_count > 0 {
            mod_before_cancel_count as f64 / cancel_count as f64
        } else {
            0.0
        };
        output.push(mod_before_cancel);

        // Feature 5: Sweep ratio (multi-level trades / total trade transitions)
        // Computed on-demand from windowed events to ensure accurate rolling statistics
        //
        // A "sweep" is a trade at a different price than the previous trade,
        // indicating that multiple price levels were consumed.
        //
        // Note: We cap at 1.0 because the first trade in the window may be marked
        // as a sweep if last_trade_price persists from before the window.
        let trade_events: Vec<_> = self
            .events
            .iter()
            .filter(|em| matches!(em.event.action, Action::Trade))
            .collect();
        let trade_count = trade_events.len();
        let sweep_count = trade_events.iter().filter(|em| em.is_sweep).count();
        // Denominator is (trade_count - 1) because first trade cannot be a sweep
        // Cap at 1.0 to handle edge case where last_trade_price persists from before window
        let sweep_ratio = if trade_count > 1 {
            (sweep_count as f64 / (trade_count - 1) as f64).min(1.0)
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

    fn make_event_with_ts(
        action: Action,
        side: Side,
        size: u32,
        price: i64,
        order_id: u64,
        timestamp: u64,
    ) -> MboEvent {
        MboEvent {
            timestamp,
            action,
            side,
            price,
            size,
            order_id,
        }
    }

    #[test]
    fn test_sweep_ratio_basic() {
        // Test sweep detection: trades at different prices = sweeps
        let mut detector = InstitutionalDetectorV2::new(100, 90.0, 100);

        // First trade at price 100
        detector.process_event(&make_event_with_ts(
            Action::Trade,
            Side::Bid,
            100,
            100_000_000_000,
            1,
            1_000_000,
        ));

        // Second trade at different price (sweep)
        detector.process_event(&make_event_with_ts(
            Action::Trade,
            Side::Bid,
            100,
            100_010_000_000, // Different price
            2,
            2_000_000,
        ));

        // Third trade at same price as second (not a sweep)
        detector.process_event(&make_event_with_ts(
            Action::Trade,
            Side::Bid,
            100,
            100_010_000_000, // Same price
            3,
            3_000_000,
        ));

        // Fourth trade at different price (sweep)
        detector.process_event(&make_event_with_ts(
            Action::Trade,
            Side::Bid,
            100,
            100_020_000_000, // Different price
            4,
            4_000_000,
        ));

        // Add some padding events to meet minimum threshold
        for i in 5..15 {
            detector.process_event(&make_event_with_ts(
                Action::Add,
                Side::Bid,
                100,
                100_000_000_000,
                i,
                (i * 1_000_000) as u64,
            ));
        }

        let mut output = Vec::new();
        detector.extract_into(&mut output);

        // 4 trades total, 2 sweeps (trades 2 and 4 changed price)
        // sweep_ratio = 2 / (4 - 1) = 2/3 ≈ 0.667
        let sweep_ratio = output[indices::SWEEP_RATIO];
        assert!(
            (sweep_ratio - 0.667).abs() < 0.01,
            "Sweep ratio should be ~0.667, got {}",
            sweep_ratio
        );
    }

    #[test]
    fn test_sweep_ratio_windowed() {
        // Test that sweep ratio is correctly windowed (old events are evicted)
        let mut detector = InstitutionalDetectorV2::new(20, 90.0, 100); // Small window

        // Fill window with sweeps (alternating prices)
        for i in 0..20 {
            let price = if i % 2 == 0 {
                100_000_000_000
            } else {
                100_010_000_000
            };
            detector.process_event(&make_event_with_ts(
                Action::Trade,
                Side::Bid,
                100,
                price,
                i,
                (i * 1_000_000) as u64,
            ));
        }

        let mut output1 = Vec::new();
        detector.extract_into(&mut output1);
        let sweep_ratio_before = output1[indices::SWEEP_RATIO];

        // Now add non-sweep trades (same price), evicting the sweeps
        for i in 20..40 {
            detector.process_event(&make_event_with_ts(
                Action::Trade,
                Side::Bid,
                100,
                100_000_000_000, // All same price
                i,
                (i * 1_000_000) as u64,
            ));
        }

        let mut output2 = Vec::new();
        detector.extract_into(&mut output2);
        let sweep_ratio_after = output2[indices::SWEEP_RATIO];

        // Before: high sweep ratio (alternating prices)
        // After: low sweep ratio (same prices) - should approach 0
        assert!(
            sweep_ratio_before > 0.5,
            "Before eviction, sweep ratio should be high: {}",
            sweep_ratio_before
        );
        assert!(
            sweep_ratio_after < 0.2,
            "After eviction, sweep ratio should be low (window now has same-price trades): {}",
            sweep_ratio_after
        );
    }

    #[test]
    fn test_mod_before_cancel_basic() {
        // Test modification before cancel detection
        let mut detector = InstitutionalDetectorV2::new(100, 90.0, 100);

        // Order 1: Add, then Cancel without modification
        detector.process_event(&make_event_with_ts(
            Action::Add,
            Side::Bid,
            100,
            100_000_000_000,
            1,
            1_000_000,
        ));
        detector.process_event(&make_event_with_ts(
            Action::Cancel,
            Side::Bid,
            100,
            100_000_000_000,
            1,
            2_000_000,
        ));

        // Order 2: Add, Modify, then Cancel (mod-before-cancel)
        detector.process_event(&make_event_with_ts(
            Action::Add,
            Side::Bid,
            100,
            100_000_000_000,
            2,
            3_000_000,
        ));
        detector.process_event(&make_event_with_ts(
            Action::Modify,
            Side::Bid,
            150, // Size changed
            100_000_000_000,
            2,
            4_000_000,
        ));
        detector.process_event(&make_event_with_ts(
            Action::Cancel,
            Side::Bid,
            150,
            100_000_000_000,
            2,
            5_000_000,
        ));

        // Order 3: Add, then Cancel without modification
        detector.process_event(&make_event_with_ts(
            Action::Add,
            Side::Bid,
            100,
            100_000_000_000,
            3,
            6_000_000,
        ));
        detector.process_event(&make_event_with_ts(
            Action::Cancel,
            Side::Bid,
            100,
            100_000_000_000,
            3,
            7_000_000,
        ));

        // Add padding to meet minimum threshold
        for i in 4..14 {
            detector.process_event(&make_event_with_ts(
                Action::Add,
                Side::Bid,
                100,
                100_000_000_000,
                i,
                ((i + 3) * 1_000_000) as u64,
            ));
        }

        let mut output = Vec::new();
        detector.extract_into(&mut output);

        // 3 cancels total, 1 was modified before cancel
        // mod_before_cancel = 1/3 ≈ 0.333
        let mod_before_cancel = output[indices::MOD_BEFORE_CANCEL];
        assert!(
            (mod_before_cancel - 0.333).abs() < 0.01,
            "Mod before cancel should be ~0.333, got {}",
            mod_before_cancel
        );
    }

    #[test]
    fn test_mod_before_cancel_windowed() {
        // Test that mod_before_cancel is correctly windowed
        let mut detector = InstitutionalDetectorV2::new(30, 90.0, 100);

        // Add orders that will be modified then cancelled
        for i in 0..10 {
            detector.process_event(&make_event_with_ts(
                Action::Add,
                Side::Bid,
                100,
                100_000_000_000,
                i,
                (i * 3_000_000) as u64,
            ));
            detector.process_event(&make_event_with_ts(
                Action::Modify,
                Side::Bid,
                150,
                100_000_000_000,
                i,
                (i * 3_000_000 + 1_000_000) as u64,
            ));
            detector.process_event(&make_event_with_ts(
                Action::Cancel,
                Side::Bid,
                150,
                100_000_000_000,
                i,
                (i * 3_000_000 + 2_000_000) as u64,
            ));
        }

        let mut output1 = Vec::new();
        detector.extract_into(&mut output1);
        let mod_before_cancel_before = output1[indices::MOD_BEFORE_CANCEL];

        // Now add orders that are cancelled WITHOUT modification (evicting the modified ones)
        for i in 10..20 {
            detector.process_event(&make_event_with_ts(
                Action::Add,
                Side::Bid,
                100,
                100_000_000_000,
                i,
                (i * 3_000_000) as u64,
            ));
            // No modify
            detector.process_event(&make_event_with_ts(
                Action::Cancel,
                Side::Bid,
                100,
                100_000_000_000,
                i,
                (i * 3_000_000 + 1_000_000) as u64,
            ));
            // Add padding
            detector.process_event(&make_event_with_ts(
                Action::Add,
                Side::Bid,
                100,
                100_000_000_000,
                i + 100,
                (i * 3_000_000 + 2_000_000) as u64,
            ));
        }

        let mut output2 = Vec::new();
        detector.extract_into(&mut output2);
        let mod_before_cancel_after = output2[indices::MOD_BEFORE_CANCEL];

        // Before: All cancels had modifications (ratio = 1.0)
        // After: Window should have mostly non-modified cancels
        assert!(
            mod_before_cancel_before > 0.8,
            "Before eviction, mod_before_cancel should be high: {}",
            mod_before_cancel_before
        );
        assert!(
            mod_before_cancel_after < 0.3,
            "After eviction, mod_before_cancel should be low: {}",
            mod_before_cancel_after
        );
    }

    #[test]
    fn test_ratios_bounded_zero_to_one() {
        // Ensure all ratios are properly bounded [0, 1]
        let mut detector = InstitutionalDetectorV2::new(100, 90.0, 100);

        // Add a variety of events
        for i in 0..50 {
            let action = match i % 4 {
                0 => Action::Add,
                1 => Action::Modify,
                2 => Action::Cancel,
                _ => Action::Trade,
            };
            let price = if i % 3 == 0 {
                100_000_000_000
            } else {
                100_010_000_000
            };
            detector.process_event(&make_event_with_ts(
                action,
                Side::Bid,
                (i * 10 + 50) as u32,
                price,
                i,
                (i * 1_000_000) as u64,
            ));
        }

        let mut output = Vec::new();
        detector.extract_into(&mut output);

        // Check bounds
        assert!(
            output[indices::ROUND_LOT_RATIO] >= 0.0 && output[indices::ROUND_LOT_RATIO] <= 1.0,
            "round_lot_ratio out of bounds: {}",
            output[indices::ROUND_LOT_RATIO]
        );
        assert!(
            output[indices::ODD_LOT_RATIO] >= 0.0 && output[indices::ODD_LOT_RATIO] <= 1.0,
            "odd_lot_ratio out of bounds: {}",
            output[indices::ODD_LOT_RATIO]
        );
        assert!(
            output[indices::MOD_BEFORE_CANCEL] >= 0.0 && output[indices::MOD_BEFORE_CANCEL] <= 1.0,
            "mod_before_cancel out of bounds: {}",
            output[indices::MOD_BEFORE_CANCEL]
        );
        assert!(
            output[indices::SWEEP_RATIO] >= 0.0 && output[indices::SWEEP_RATIO] <= 1.0,
            "sweep_ratio out of bounds: {}",
            output[indices::SWEEP_RATIO]
        );
    }
}
