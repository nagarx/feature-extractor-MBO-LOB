//! Queue position and depth feature extraction (6 features, contract indices 68-73).
//!
//! Tracks execution priority through queue position, volume ahead,
//! orders per level, volume concentration, and depth-in-ticks metrics.

use super::order_tracker::OrderTracker;
use mbo_lob_reconstructor::lob::queue_position::QueuePositionTracker;
use mbo_lob_reconstructor::{LobState, Side};

/// Extract 6 queue & depth features.
///
/// Layout:
/// - [0]: Average queue position (FIFO-based, requires queue tracking)
/// - [1]: Average volume ahead in queue (requires queue tracking)
/// - [2]: Orders per active LOB level
/// - [3]: Volume concentration across levels (HHI)
/// - [4]: Volume-weighted average depth on bid side (in ticks)
/// - [5]: Volume-weighted average depth on ask side (in ticks)
pub(super) fn extract(
    lob: &LobState,
    queue_tracker: Option<&QueuePositionTracker>,
    order_tracker: &OrderTracker,
    tick_size_nanodollars: i64,
) -> [f64; 6] {
    [
        average_queue_position(queue_tracker),
        queue_size_ahead(queue_tracker, order_tracker),
        orders_per_level(order_tracker, lob),
        level_concentration(lob),
        depth_ticks_bid(lob, tick_size_nanodollars),
        depth_ticks_ask(lob, tick_size_nanodollars),
    ]
}

fn average_queue_position(queue_tracker: Option<&QueuePositionTracker>) -> f64 {
    match queue_tracker {
        Some(tracker) => {
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

fn queue_size_ahead(
    queue_tracker: Option<&QueuePositionTracker>,
    order_tracker: &OrderTracker,
) -> f64 {
    match queue_tracker {
        Some(tracker) => {
            let mut total_volume_ahead: u64 = 0;
            let mut order_count: usize = 0;

            for &order_id in order_tracker.active_orders().keys() {
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

fn orders_per_level(order_tracker: &OrderTracker, lob: &LobState) -> f64 {
    let active_orders = order_tracker.active_count();
    let active_levels = lob.bid_prices.iter().filter(|&&p| p > 0).count()
        + lob.ask_prices.iter().filter(|&&p| p > 0).count();
    active_orders as f64 / active_levels.max(1) as f64
}

/// Volume concentration across LOB levels using HHI.
///
/// HHI = Σ(share_i)² where share_i = volume_i / total_volume
/// Range: [1/N, 1.0]. Low = well-distributed, High = concentrated.
fn level_concentration(lob: &LobState) -> f64 {
    let total_bid: u64 = lob.bid_sizes[..lob.levels].iter().map(|&s| s as u64).sum();
    let total_ask: u64 = lob.ask_sizes[..lob.levels].iter().map(|&s| s as u64).sum();
    let total = total_bid + total_ask;

    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut hhi: f64 = 0.0;

    for &size in &lob.bid_sizes[..lob.levels] {
        if size > 0 {
            let share = size as f64 / total_f;
            hhi += share * share;
        }
    }

    for &size in &lob.ask_sizes[..lob.levels] {
        if size > 0 {
            let share = size as f64 / total_f;
            hhi += share * share;
        }
    }

    hhi
}

/// Volume-weighted average depth on bid side in ticks.
///
/// Depth_ticks = Σ(vol_i × distance_i) / Σ(vol_i)
/// where distance_i = (best_bid - bid_price[i]) / tick_size
fn depth_ticks_bid(lob: &LobState, tick_size_nanodollars: i64) -> f64 {
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
            let distance_ticks = (best_bid - price) / tick_size_nanodollars;
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

/// Volume-weighted average depth on ask side in ticks.
///
/// Depth_ticks = Σ(vol_i × distance_i) / Σ(vol_i)
/// where distance_i = (ask_price[i] - best_ask) / tick_size
fn depth_ticks_ask(lob: &LobState, tick_size_nanodollars: i64) -> f64 {
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
            let distance_ticks = (price - best_ask) / tick_size_nanodollars;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_concentration_empty_lob() {
        let lob = LobState::new(10);
        assert_eq!(level_concentration(&lob), 0.0);
    }

    #[test]
    fn test_level_concentration_single_level() {
        let mut lob = LobState::new(10);
        lob.bid_sizes[0] = 1000;
        assert!((level_concentration(&lob) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_level_concentration_perfectly_even() {
        let mut lob = LobState::new(10);
        lob.bid_sizes[0] = 100;
        lob.bid_sizes[1] = 100;
        lob.ask_sizes[0] = 100;
        lob.ask_sizes[1] = 100;
        assert!((level_concentration(&lob) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_depth_ticks_bid_no_best() {
        let lob = LobState::new(10);
        assert_eq!(depth_ticks_bid(&lob, 10_000_000), 0.0);
    }

    #[test]
    fn test_depth_ticks_bid_single_level() {
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000);
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_sizes[0] = 100;
        assert!(depth_ticks_bid(&lob, 10_000_000).abs() < 1e-10);
    }

    #[test]
    fn test_depth_ticks_bid_multiple_levels() {
        let mut lob = LobState::new(10);
        lob.best_bid = Some(100_000_000_000);
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_prices[1] = 99_990_000_000;
        lob.bid_sizes[0] = 100;
        lob.bid_sizes[1] = 100;
        let result = depth_ticks_bid(&lob, 10_000_000);
        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_depth_ticks_ask_multiple_levels() {
        let mut lob = LobState::new(10);
        lob.best_ask = Some(100_000_000_000);
        lob.ask_prices[0] = 100_000_000_000;
        lob.ask_prices[1] = 100_020_000_000;
        lob.ask_sizes[0] = 200;
        lob.ask_sizes[1] = 100;
        let result = depth_ticks_ask(&lob, 10_000_000);
        let expected = 200.0 / 300.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            result
        );
    }
}
