//! Order Flow Features
//!
//! Implements order flow features from "The Price Impact of Order Book Events" (Cont et al., 2014).
//!
//! # Key Features
//!
//! - **Order Flow Imbalance (OFI)**: Net order flow at best bid/ask
//! - **Multi-Level OFI (MLOFI)**: OFI computed across multiple LOB levels (1-10)
//! - **Queue Imbalance**: Relative queue size imbalance
//! - **Trade Imbalance**: Signed trade volume
//!
//! # Research Background
//!
//! The OFI explains ~65% of mid-price variance according to Cont et al. (2014).
//! Queue imbalance is a strong predictor of one-tick-ahead price direction.
//! Multi-level OFI (MLOFI) captures demand/supply shifts across multiple price levels,
//! providing richer information than single-level OFI.
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::features::order_flow::{OrderFlowTracker, OrderFlowFeatures, MultiLevelOfiTracker};
//!
//! let mut tracker = OrderFlowTracker::new();
//!
//! // Process LOB updates
//! tracker.update(&lob, timestamp);
//!
//! // Extract features
//! let features = tracker.extract_features(&lob);
//! println!("OFI: {}", features.ofi);
//! println!("Queue Imbalance: {}", features.queue_imbalance);
//!
//! // Multi-Level OFI
//! let mut mlofi_tracker = MultiLevelOfiTracker::new(10);
//! mlofi_tracker.update(&lob);
//! let mlofi = mlofi_tracker.get_mlofi();
//! ```

use mbo_lob_reconstructor::LobState;

/// Order flow features extracted from LOB state transitions.
#[derive(Debug, Clone, Default)]
pub struct OrderFlowFeatures {
    /// Order Flow Imbalance (total)
    /// OFI = Σ e_n where e_n tracks queue changes
    pub ofi: f64,

    /// Order Flow Imbalance (bid side only)
    pub ofi_bid: f64,

    /// Order Flow Imbalance (ask side only)
    pub ofi_ask: f64,

    /// Queue Imbalance: (n_bid - n_ask) / (n_bid + n_ask)
    /// Strong predictor of one-tick-ahead price direction
    pub queue_imbalance: f64,

    /// Trade Imbalance: net signed trade volume
    pub trade_imbalance: f64,

    /// Depth Imbalance: (bid_vol - ask_vol) / total_vol
    pub depth_imbalance: f64,

    /// Order arrival rate on bid side (orders per second)
    pub arrival_rate_bid: f64,

    /// Order arrival rate on ask side (orders per second)
    pub arrival_rate_ask: f64,
}

impl OrderFlowFeatures {
    /// Convert to feature vector (8 features).
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.ofi,
            self.ofi_bid,
            self.ofi_ask,
            self.queue_imbalance,
            self.trade_imbalance,
            self.depth_imbalance,
            self.arrival_rate_bid,
            self.arrival_rate_ask,
        ]
    }

    /// Number of features.
    pub const fn count() -> usize {
        8
    }
}

/// Tracks order flow across LOB state transitions.
///
/// Maintains state needed to compute OFI and related features.
pub struct OrderFlowTracker {
    /// Previous best bid price
    prev_best_bid: Option<i64>,

    /// Previous best ask price
    prev_best_ask: Option<i64>,

    /// Previous best bid size
    prev_best_bid_size: u32,

    /// Previous best ask size
    prev_best_ask_size: u32,

    /// Cumulative OFI (bid side)
    cumulative_ofi_bid: f64,

    /// Cumulative OFI (ask side)
    cumulative_ofi_ask: f64,

    /// Cumulative trade imbalance
    cumulative_trade_imbalance: f64,

    /// Event count for arrival rate calculation
    bid_events: u64,
    ask_events: u64,

    /// First timestamp for rate calculation
    first_timestamp: Option<u64>,

    /// Last timestamp
    last_timestamp: Option<u64>,
}

impl OrderFlowTracker {
    /// Create a new order flow tracker.
    pub fn new() -> Self {
        Self {
            prev_best_bid: None,
            prev_best_ask: None,
            prev_best_bid_size: 0,
            prev_best_ask_size: 0,
            cumulative_ofi_bid: 0.0,
            cumulative_ofi_ask: 0.0,
            cumulative_trade_imbalance: 0.0,
            bid_events: 0,
            ask_events: 0,
            first_timestamp: None,
            last_timestamp: None,
        }
    }

    /// Update tracker with new LOB state.
    ///
    /// Computes OFI contribution from the state transition.
    ///
    /// # OFI Calculation (from Cont et al., 2014)
    ///
    /// For each update n:
    /// ```text
    /// e_n = I{P^b_n >= P^b_{n-1}} × ΔV^b_n - I{P^b_n <= P^b_{n-1}} × ΔV^b_{n-1}
    ///     - I{P^a_n <= P^a_{n-1}} × ΔV^a_n + I{P^a_n >= P^a_{n-1}} × ΔV^a_{n-1}
    /// ```
    ///
    /// Simplified: Track queue changes at best bid/ask
    pub fn update(&mut self, lob: &LobState, timestamp: u64) {
        // Initialize timestamps
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(timestamp);
        }
        self.last_timestamp = Some(timestamp);

        let curr_best_bid = lob.best_bid;
        let curr_best_ask = lob.best_ask;
        let curr_bid_size = lob.bid_sizes[0];
        let curr_ask_size = lob.ask_sizes[0];

        // Compute OFI contribution
        if let (Some(prev_bid), Some(curr_bid)) = (self.prev_best_bid, curr_best_bid) {
            let ofi_bid = self.compute_ofi_contribution(
                prev_bid,
                curr_bid,
                self.prev_best_bid_size,
                curr_bid_size,
                true, // is_bid
            );
            self.cumulative_ofi_bid += ofi_bid;
            if ofi_bid.abs() > 1e-10 {
                self.bid_events += 1;
            }
        }

        if let (Some(prev_ask), Some(curr_ask)) = (self.prev_best_ask, curr_best_ask) {
            let ofi_ask = self.compute_ofi_contribution(
                prev_ask,
                curr_ask,
                self.prev_best_ask_size,
                curr_ask_size,
                false, // is_bid
            );
            self.cumulative_ofi_ask += ofi_ask;
            if ofi_ask.abs() > 1e-10 {
                self.ask_events += 1;
            }
        }

        // Update previous state
        self.prev_best_bid = curr_best_bid;
        self.prev_best_ask = curr_best_ask;
        self.prev_best_bid_size = curr_bid_size;
        self.prev_best_ask_size = curr_ask_size;
    }

    /// Compute OFI contribution for one side.
    ///
    /// # Arguments
    ///
    /// * `prev_price` - Previous best price
    /// * `curr_price` - Current best price
    /// * `prev_size` - Previous best size
    /// * `curr_size` - Current best size
    /// * `is_bid` - Whether this is the bid side
    fn compute_ofi_contribution(
        &self,
        prev_price: i64,
        curr_price: i64,
        prev_size: u32,
        curr_size: u32,
        is_bid: bool,
    ) -> f64 {
        let size_change = curr_size as f64 - prev_size as f64;

        if is_bid {
            // Bid side: positive OFI when price improves or size increases at same price
            if curr_price > prev_price {
                // Price improved: new queue, count new size
                curr_size as f64
            } else if curr_price < prev_price {
                // Price worsened: lost queue, subtract old size
                -(prev_size as f64)
            } else {
                // Same price: track size change
                size_change
            }
        } else {
            // Ask side: negative OFI when price improves (goes down) or size increases
            if curr_price < prev_price {
                // Price improved (lower ask): new queue, negative contribution
                -(curr_size as f64)
            } else if curr_price > prev_price {
                // Price worsened (higher ask): lost queue, positive contribution
                prev_size as f64
            } else {
                // Same price: track size change (negative for asks)
                -size_change
            }
        }
    }

    /// Record a trade event.
    ///
    /// # Arguments
    ///
    /// * `size` - Trade size
    /// * `is_buy` - Whether this was a buy (taker bought)
    pub fn record_trade(&mut self, size: u32, is_buy: bool) {
        let signed_size = if is_buy { size as f64 } else { -(size as f64) };
        self.cumulative_trade_imbalance += signed_size;
    }

    /// Extract current order flow features.
    pub fn extract_features(&self, lob: &LobState) -> OrderFlowFeatures {
        // Compute queue imbalance
        let bid_size = lob.bid_sizes[0] as f64;
        let ask_size = lob.ask_sizes[0] as f64;
        let total_size = bid_size + ask_size;

        let queue_imbalance = if total_size > 0.0 {
            (bid_size - ask_size) / total_size
        } else {
            0.0
        };

        // Compute depth imbalance (across all levels)
        let total_bid_vol: f64 = lob.bid_sizes.iter().map(|&s| s as f64).sum();
        let total_ask_vol: f64 = lob.ask_sizes.iter().map(|&s| s as f64).sum();
        let total_vol = total_bid_vol + total_ask_vol;

        let depth_imbalance = if total_vol > 0.0 {
            (total_bid_vol - total_ask_vol) / total_vol
        } else {
            0.0
        };

        // Compute arrival rates
        let duration_secs = self.duration_seconds();
        let arrival_rate_bid = if duration_secs > 0.0 {
            self.bid_events as f64 / duration_secs
        } else {
            0.0
        };
        let arrival_rate_ask = if duration_secs > 0.0 {
            self.ask_events as f64 / duration_secs
        } else {
            0.0
        };

        OrderFlowFeatures {
            ofi: self.cumulative_ofi_bid + self.cumulative_ofi_ask,
            ofi_bid: self.cumulative_ofi_bid,
            ofi_ask: self.cumulative_ofi_ask,
            queue_imbalance,
            trade_imbalance: self.cumulative_trade_imbalance,
            depth_imbalance,
            arrival_rate_bid,
            arrival_rate_ask,
        }
    }

    /// Get the tracking duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        match (self.first_timestamp, self.last_timestamp) {
            (Some(first), Some(last)) if last > first => (last - first) as f64 / 1e9,
            _ => 0.0,
        }
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for OrderFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute queue imbalance from LOB state.
///
/// # Formula
///
/// ```text
/// I(t) = (n^b - n^a) / (n^b + n^a)
/// ```
///
/// Where:
/// - n^b = best bid size
/// - n^a = best ask size
///
/// # Returns
///
/// Value in [-1, 1]:
/// - Positive: More volume on bid side
/// - Negative: More volume on ask side
/// - Zero: Balanced
#[inline]
pub fn compute_queue_imbalance(lob: &LobState) -> f64 {
    let bid_size = lob.bid_sizes[0] as f64;
    let ask_size = lob.ask_sizes[0] as f64;
    let total = bid_size + ask_size;

    if total > 0.0 {
        (bid_size - ask_size) / total
    } else {
        0.0
    }
}

/// Compute depth imbalance across all levels.
///
/// # Formula
///
/// ```text
/// DI(t) = (Σ V^b_i - Σ V^a_i) / (Σ V^b_i + Σ V^a_i)
/// ```
#[inline]
pub fn compute_depth_imbalance(lob: &LobState, levels: usize) -> f64 {
    let levels = levels.min(lob.levels);

    let total_bid: f64 = lob.bid_sizes[..levels].iter().map(|&s| s as f64).sum();
    let total_ask: f64 = lob.ask_sizes[..levels].iter().map(|&s| s as f64).sum();
    let total = total_bid + total_ask;

    if total > 0.0 {
        (total_bid - total_ask) / total
    } else {
        0.0
    }
}

// =============================================================================
// Multi-Level Order Flow Imbalance (MLOFI)
// =============================================================================

/// Multi-Level Order Flow Imbalance (MLOFI) Tracker
///
/// Computes OFI across multiple LOB levels (not just level 1), capturing
/// demand/supply shifts throughout the order book depth.
///
/// # Research Background
///
/// From "LOB-feature-analysis" and "Price Impact of Order Book Events":
/// - Level 1 OFI captures ~65% of mid-price variance
/// - Multi-level OFI adds 5-10% additional explanatory power
/// - Deeper levels (2-5) are particularly informative for large orders
///
/// # Formula
///
/// For each level k:
/// ```text
/// OFI_k = Σ e_n,k
/// ```
///
/// Where e_n,k is the OFI contribution at level k:
/// - Price improved: +new_size (for bids), -new_size (for asks)
/// - Price worsened: -old_size (for bids), +old_size (for asks)
/// - Price unchanged: size_change
///
/// MLOFI = Σ OFI_k (sum across all levels)
///
/// Weighted MLOFI = Σ w_k × OFI_k where w_k = 1/k (level 1 has weight 1, level 2 has weight 0.5, etc.)
#[derive(Debug, Clone)]
pub struct MultiLevelOfiTracker {
    /// Number of levels to track
    levels: usize,

    /// Previous bid prices per level
    prev_bid_prices: Vec<i64>,

    /// Previous ask prices per level
    prev_ask_prices: Vec<i64>,

    /// Previous bid sizes per level
    prev_bid_sizes: Vec<u32>,

    /// Previous ask sizes per level
    prev_ask_sizes: Vec<u32>,

    /// Cumulative OFI per level (bid side)
    ofi_bid_per_level: Vec<f64>,

    /// Cumulative OFI per level (ask side)
    ofi_ask_per_level: Vec<f64>,

    /// Whether we have received at least one update
    initialized: bool,

    /// Number of updates processed
    update_count: u64,
}

impl MultiLevelOfiTracker {
    /// Create a new multi-level OFI tracker.
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of LOB levels to track (typically 5-10)
    pub fn new(levels: usize) -> Self {
        let levels = levels.max(1); // At least 1 level
        Self {
            levels,
            prev_bid_prices: vec![0; levels],
            prev_ask_prices: vec![0; levels],
            prev_bid_sizes: vec![0; levels],
            prev_ask_sizes: vec![0; levels],
            ofi_bid_per_level: vec![0.0; levels],
            ofi_ask_per_level: vec![0.0; levels],
            initialized: false,
            update_count: 0,
        }
    }

    /// Update tracker with new LOB state.
    ///
    /// Computes OFI contribution for each level from the state transition.
    #[inline]
    pub fn update(&mut self, lob: &LobState) {
        let track_levels = self.levels.min(lob.levels);

        if !self.initialized {
            // First update: just store state
            for i in 0..track_levels {
                self.prev_bid_prices[i] = lob.bid_prices[i];
                self.prev_ask_prices[i] = lob.ask_prices[i];
                self.prev_bid_sizes[i] = lob.bid_sizes[i];
                self.prev_ask_sizes[i] = lob.ask_sizes[i];
            }
            self.initialized = true;
            self.update_count = 1;
            return;
        }

        // Compute OFI for each level
        for i in 0..track_levels {
            // Bid side OFI
            let ofi_bid = self.compute_level_ofi(
                self.prev_bid_prices[i],
                lob.bid_prices[i],
                self.prev_bid_sizes[i],
                lob.bid_sizes[i],
                true,
            );
            self.ofi_bid_per_level[i] += ofi_bid;

            // Ask side OFI
            let ofi_ask = self.compute_level_ofi(
                self.prev_ask_prices[i],
                lob.ask_prices[i],
                self.prev_ask_sizes[i],
                lob.ask_sizes[i],
                false,
            );
            self.ofi_ask_per_level[i] += ofi_ask;

            // Update previous state
            self.prev_bid_prices[i] = lob.bid_prices[i];
            self.prev_ask_prices[i] = lob.ask_prices[i];
            self.prev_bid_sizes[i] = lob.bid_sizes[i];
            self.prev_ask_sizes[i] = lob.ask_sizes[i];
        }

        self.update_count += 1;
    }

    /// Compute OFI contribution for a single level.
    #[inline]
    fn compute_level_ofi(
        &self,
        prev_price: i64,
        curr_price: i64,
        prev_size: u32,
        curr_size: u32,
        is_bid: bool,
    ) -> f64 {
        // Handle empty levels
        if prev_price == 0 && curr_price == 0 {
            return 0.0;
        }

        let size_change = curr_size as f64 - prev_size as f64;

        if is_bid {
            if curr_price > prev_price || (prev_price == 0 && curr_price > 0) {
                // Price improved or level appeared: new queue
                curr_size as f64
            } else if curr_price < prev_price || (curr_price == 0 && prev_price > 0) {
                // Price worsened or level disappeared: lost queue
                -(prev_size as f64)
            } else {
                // Same price: track size change
                size_change
            }
        } else {
            // Ask side (inverted logic: lower price = improvement)
            if curr_price < prev_price || (prev_price == 0 && curr_price > 0) {
                // Price improved (lower) or level appeared: new queue (negative contribution)
                -(curr_size as f64)
            } else if curr_price > prev_price || (curr_price == 0 && prev_price > 0) {
                // Price worsened (higher) or level disappeared: lost queue (positive contribution)
                prev_size as f64
            } else {
                // Same price: track size change (negative for asks)
                -size_change
            }
        }
    }

    /// Get total MLOFI (sum across all levels).
    ///
    /// # Returns
    ///
    /// Sum of OFI across all tracked levels.
    #[inline]
    pub fn get_mlofi(&self) -> f64 {
        let total_bid: f64 = self.ofi_bid_per_level.iter().sum();
        let total_ask: f64 = self.ofi_ask_per_level.iter().sum();
        total_bid + total_ask
    }

    /// Get weighted MLOFI with level-based weights.
    ///
    /// Uses weights w_k = 1/k where k is the level (1-indexed).
    /// Level 1 has weight 1.0, level 2 has weight 0.5, etc.
    ///
    /// # Returns
    ///
    /// Weighted sum of OFI across all tracked levels.
    #[inline]
    pub fn get_weighted_mlofi(&self) -> f64 {
        let mut weighted_sum = 0.0;
        for (i, (&bid, &ask)) in self
            .ofi_bid_per_level
            .iter()
            .zip(self.ofi_ask_per_level.iter())
            .enumerate()
        {
            let weight = 1.0 / (i as f64 + 1.0);
            weighted_sum += weight * (bid + ask);
        }
        weighted_sum
    }

    /// Get OFI per level (combined bid + ask).
    ///
    /// # Returns
    ///
    /// Vector of OFI values, one per level.
    #[inline]
    pub fn get_ofi_per_level(&self) -> Vec<f64> {
        self.ofi_bid_per_level
            .iter()
            .zip(self.ofi_ask_per_level.iter())
            .map(|(&bid, &ask)| bid + ask)
            .collect()
    }

    /// Get OFI per level for bid side only.
    #[inline]
    pub fn get_ofi_bid_per_level(&self) -> &[f64] {
        &self.ofi_bid_per_level
    }

    /// Get OFI per level for ask side only.
    #[inline]
    pub fn get_ofi_ask_per_level(&self) -> &[f64] {
        &self.ofi_ask_per_level
    }

    /// Get the number of updates processed.
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Get the number of levels being tracked.
    #[inline]
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.ofi_bid_per_level.fill(0.0);
        self.ofi_ask_per_level.fill(0.0);
        self.prev_bid_prices.fill(0);
        self.prev_ask_prices.fill(0);
        self.prev_bid_sizes.fill(0);
        self.prev_ask_sizes.fill(0);
        self.initialized = false;
        self.update_count = 0;
    }

    /// Extract MLOFI features as a vector.
    ///
    /// Returns:
    /// - [0]: Total MLOFI (sum across all levels)
    /// - [1]: Weighted MLOFI (level-weighted sum)
    /// - [2..2+levels]: OFI per level (bid + ask combined)
    #[inline]
    pub fn extract_features(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(2 + self.levels);
        features.push(self.get_mlofi());
        features.push(self.get_weighted_mlofi());
        features.extend(self.get_ofi_per_level());
        features
    }
}

impl Default for MultiLevelOfiTracker {
    fn default() -> Self {
        Self::new(10) // Default to 10 levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lob(bid_price: i64, ask_price: i64, bid_size: u32, ask_size: u32) -> LobState {
        let mut lob = LobState::new(10);
        lob.bid_prices[0] = bid_price;
        lob.ask_prices[0] = ask_price;
        lob.bid_sizes[0] = bid_size;
        lob.ask_sizes[0] = ask_size;
        lob.best_bid = Some(bid_price);
        lob.best_ask = Some(ask_price);
        lob
    }

    #[test]
    fn test_queue_imbalance_balanced() {
        let lob = create_test_lob(100_000_000_000, 100_010_000_000, 100, 100);
        let imbalance = compute_queue_imbalance(&lob);
        assert!((imbalance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_queue_imbalance_bid_heavy() {
        let lob = create_test_lob(100_000_000_000, 100_010_000_000, 200, 100);
        let imbalance = compute_queue_imbalance(&lob);
        // (200 - 100) / 300 = 0.333...
        assert!((imbalance - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_queue_imbalance_ask_heavy() {
        let lob = create_test_lob(100_000_000_000, 100_010_000_000, 100, 200);
        let imbalance = compute_queue_imbalance(&lob);
        // (100 - 200) / 300 = -0.333...
        assert!((imbalance + 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_order_flow_tracker_basic() {
        let mut tracker = OrderFlowTracker::new();

        // Initial state
        let lob1 = create_test_lob(100_000_000_000, 100_010_000_000, 100, 100);
        tracker.update(&lob1, 1_000_000_000);

        // Size increase on bid
        let lob2 = create_test_lob(100_000_000_000, 100_010_000_000, 150, 100);
        tracker.update(&lob2, 2_000_000_000);

        let features = tracker.extract_features(&lob2);

        // OFI should be positive (bid size increased)
        assert!(features.ofi_bid > 0.0);
    }

    #[test]
    fn test_order_flow_tracker_price_improvement() {
        let mut tracker = OrderFlowTracker::new();

        // Initial state
        let lob1 = create_test_lob(100_000_000_000, 100_010_000_000, 100, 100);
        tracker.update(&lob1, 1_000_000_000);

        // Bid price improves (new higher bid)
        let lob2 = create_test_lob(100_005_000_000, 100_010_000_000, 50, 100);
        tracker.update(&lob2, 2_000_000_000);

        let features = tracker.extract_features(&lob2);

        // OFI should be positive (bid improved)
        assert!(features.ofi_bid > 0.0);
    }

    #[test]
    fn test_order_flow_features_to_vec() {
        let features = OrderFlowFeatures {
            ofi: 100.0,
            ofi_bid: 60.0,
            ofi_ask: 40.0,
            queue_imbalance: 0.2,
            trade_imbalance: 50.0,
            depth_imbalance: 0.1,
            arrival_rate_bid: 10.0,
            arrival_rate_ask: 8.0,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), OrderFlowFeatures::count());
        assert_eq!(vec[0], 100.0); // ofi
        assert_eq!(vec[3], 0.2); // queue_imbalance
    }

    #[test]
    fn test_depth_imbalance() {
        let mut lob = LobState::new(10);

        // Set up multi-level book
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_sizes[0] = 100;
        lob.bid_prices[1] = 99_990_000_000;
        lob.bid_sizes[1] = 200;

        lob.ask_prices[0] = 100_010_000_000;
        lob.ask_sizes[0] = 50;
        lob.ask_prices[1] = 100_020_000_000;
        lob.ask_sizes[1] = 50;

        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);

        // Total bid = 300, Total ask = 100
        // Imbalance = (300 - 100) / 400 = 0.5
        let imbalance = compute_depth_imbalance(&lob, 2);
        assert!((imbalance - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = OrderFlowTracker::new();

        let lob = create_test_lob(100_000_000_000, 100_010_000_000, 100, 100);
        tracker.update(&lob, 1_000_000_000);
        tracker.record_trade(50, true);

        tracker.reset();

        let features = tracker.extract_features(&lob);
        assert_eq!(features.ofi, 0.0);
        assert_eq!(features.trade_imbalance, 0.0);
    }

    #[test]
    fn test_mlofi_basic() {
        let mut tracker = MultiLevelOfiTracker::new(5);

        // Initial state
        let lob1 = create_multi_level_lob(
            100_000_000_000,
            100_010_000_000,
            &[100, 200, 300],
            &[100, 200, 300],
        );
        tracker.update(&lob1);

        // Size increase on all levels
        let lob2 = create_multi_level_lob(
            100_000_000_000,
            100_010_000_000,
            &[150, 250, 350],
            &[100, 200, 300],
        );
        tracker.update(&lob2);

        let mlofi = tracker.get_mlofi();

        // MLOFI should be positive (bid sizes increased)
        assert!(mlofi > 0.0);
    }

    #[test]
    fn test_mlofi_per_level() {
        let mut tracker = MultiLevelOfiTracker::new(3);

        // Initial state
        let lob1 = create_multi_level_lob(
            100_000_000_000,
            100_010_000_000,
            &[100, 200, 300],
            &[100, 200, 300],
        );
        tracker.update(&lob1);

        // Size increase on level 0 only
        let lob2 = create_multi_level_lob(
            100_000_000_000,
            100_010_000_000,
            &[200, 200, 300],
            &[100, 200, 300],
        );
        tracker.update(&lob2);

        let per_level = tracker.get_ofi_per_level();
        assert_eq!(per_level.len(), 3);
        assert!(per_level[0] > 0.0); // Level 0 increased
        assert!((per_level[1] - 0.0).abs() < 1e-10); // Level 1 unchanged
        assert!((per_level[2] - 0.0).abs() < 1e-10); // Level 2 unchanged
    }

    #[test]
    fn test_mlofi_weighted() {
        let mut tracker = MultiLevelOfiTracker::new(3);

        // Initial state
        let lob1 = create_multi_level_lob(
            100_000_000_000,
            100_010_000_000,
            &[100, 100, 100],
            &[100, 100, 100],
        );
        tracker.update(&lob1);

        // Size increase on all levels equally
        let lob2 = create_multi_level_lob(
            100_000_000_000,
            100_010_000_000,
            &[200, 200, 200],
            &[100, 100, 100],
        );
        tracker.update(&lob2);

        let weighted = tracker.get_weighted_mlofi();
        let unweighted = tracker.get_mlofi();

        // Weighted should give more importance to level 0
        // Both should be positive
        assert!(weighted > 0.0);
        assert!(unweighted > 0.0);
    }

    fn create_multi_level_lob(
        bid_price: i64,
        ask_price: i64,
        bid_sizes: &[u32],
        ask_sizes: &[u32],
    ) -> LobState {
        let levels = bid_sizes.len().max(ask_sizes.len()).max(10);
        let mut lob = LobState::new(levels);

        for (i, &size) in bid_sizes.iter().enumerate() {
            lob.bid_prices[i] = bid_price - (i as i64 * 10_000_000); // 1 cent apart
            lob.bid_sizes[i] = size;
        }

        for (i, &size) in ask_sizes.iter().enumerate() {
            lob.ask_prices[i] = ask_price + (i as i64 * 10_000_000); // 1 cent apart
            lob.ask_sizes[i] = size;
        }

        lob.best_bid = Some(bid_price);
        lob.best_ask = Some(ask_price);
        lob
    }
}
