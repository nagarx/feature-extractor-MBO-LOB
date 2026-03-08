//! Core MBO lifecycle feature extraction (6 features, contract indices 78-83).
//!
//! Order lifecycle characteristics: age, lifetime, fill ratio,
//! time to fill, cancel-to-add ratio, and active order count.

use super::order_tracker::OrderTracker;
use super::window::MboWindow;

/// Extract 6 core MBO lifecycle features.
///
/// Layout:
/// - [0]: Average order age (seconds)
/// - [1]: Median order lifetime (seconds, from completed orders)
/// - [2]: Average fill ratio (from completed orders)
/// - [3]: Average time to first fill (seconds)
/// - [4]: Cancel-to-add ratio (clamped to [0, 10])
/// - [5]: Active order count
pub(super) fn extract(tracker: &OrderTracker, medium_window: &MboWindow) -> [f64; 6] {
    [
        average_order_age(tracker, medium_window.last_ts),
        median_order_lifetime(tracker),
        average_fill_ratio(tracker),
        average_time_to_first_fill(tracker),
        cancel_to_add_ratio(medium_window),
        tracker.active_count() as f64,
    ]
}

fn average_order_age(tracker: &OrderTracker, current_ts: u64) -> f64 {
    if tracker.is_empty() {
        return 0.0;
    }
    let total_age: f64 = tracker
        .active_orders()
        .values()
        .map(|info| info.age(current_ts))
        .sum();
    total_age / tracker.active_count() as f64
}

/// Median of completed order lifetimes (seconds).
///
/// O(n log n) where n <= COMPLETED_ORDER_BUFFER_SIZE (1000).
fn median_order_lifetime(tracker: &OrderTracker) -> f64 {
    let lifetimes = tracker.completed_lifetimes();
    let n = lifetimes.len();
    if n == 0 {
        return 0.0;
    }

    let mut sorted: Vec<f64> = lifetimes.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn average_fill_ratio(tracker: &OrderTracker) -> f64 {
    let ratios = tracker.completed_fill_ratios();
    if ratios.is_empty() {
        return 0.0;
    }
    let total: f64 = ratios.iter().sum();
    total / ratios.len() as f64
}

fn average_time_to_first_fill(tracker: &OrderTracker) -> f64 {
    let fills: Vec<f64> = tracker
        .active_orders()
        .values()
        .filter_map(|info| info.time_to_first_fill())
        .collect();
    if fills.is_empty() {
        return 0.0;
    }
    fills.iter().sum::<f64>() / fills.len() as f64
}

/// Cancel-to-add ratio: cancels / adds, clamped to [0, 10].
///
/// - `< 1.0`: Book growing (more adds)
/// - `= 1.0`: Stable book
/// - `> 1.0`: Book shrinking (more cancels)
/// - No activity: 1.0 (neutral)
/// - Cancels only: 10.0 (capped)
fn cancel_to_add_ratio(w: &MboWindow) -> f64 {
    const MAX_RATIO: f64 = 10.0;

    let cancels = w.cancel_count_bid + w.cancel_count_ask;
    let adds = w.add_count_bid + w.add_count_ask;

    if adds == 0 {
        if cancels == 0 {
            return 1.0;
        } else {
            return MAX_RATIO;
        }
    }

    let ratio = cancels as f64 / adds as f64;
    ratio.clamp(0.0, MAX_RATIO)
}

#[cfg(test)]
mod tests {
    use super::super::event::MboEvent;
    use super::super::order_tracker::OrderTracker;
    use super::*;
    use mbo_lob_reconstructor::{Action, Side};

    #[test]
    fn test_median_order_lifetime_no_completed() {
        let tracker = OrderTracker::new();
        assert_eq!(median_order_lifetime(&tracker), 0.0);
    }

    #[test]
    fn test_median_order_lifetime_single_order() {
        let mut tracker = OrderTracker::new();
        let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&add);
        let cancel = MboEvent::new(
            1_000_000_000,
            Action::Cancel,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        tracker.process_event(&cancel);

        let median = median_order_lifetime(&tracker);
        assert!(
            (median - 1.0).abs() < 0.001,
            "Expected 1.0s, got {}",
            median
        );
    }

    #[test]
    fn test_median_order_lifetime_odd_count() {
        let mut tracker = OrderTracker::new();
        for i in 0..5u64 {
            let lifetime_ns = (i + 1) * 1_000_000_000;
            let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            tracker.process_event(&add);
            let cancel = MboEvent::new(
                lifetime_ns,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            tracker.process_event(&cancel);
        }
        let median = median_order_lifetime(&tracker);
        assert!(
            (median - 3.0).abs() < 0.001,
            "Median of [1,2,3,4,5] should be 3.0, got {}",
            median
        );
    }

    #[test]
    fn test_median_order_lifetime_even_count() {
        let mut tracker = OrderTracker::new();
        for i in 0..4u64 {
            let lifetime_ns = (i + 1) * 1_000_000_000;
            let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, i);
            tracker.process_event(&add);
            let cancel = MboEvent::new(
                lifetime_ns,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            tracker.process_event(&cancel);
        }
        let median = median_order_lifetime(&tracker);
        assert!(
            (median - 2.5).abs() < 0.001,
            "Median of [1,2,3,4] should be 2.5, got {}",
            median
        );
    }

    #[test]
    fn test_average_fill_ratio_empty() {
        let tracker = OrderTracker::new();
        assert_eq!(average_fill_ratio(&tracker), 0.0);
    }

    #[test]
    fn test_average_fill_ratio_fully_filled() {
        let mut tracker = OrderTracker::new();
        let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&add);
        let fill = MboEvent::new(1_000_000, Action::Trade, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&fill);

        let ratio = average_fill_ratio(&tracker);
        assert!((ratio - 1.0).abs() < 0.001, "Expected 1.0, got {}", ratio);
    }

    #[test]
    fn test_cancel_to_add_ratio_normal() {
        let mut w = MboWindow::new(1000);
        for i in 0..10u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            w.push(event);
        }
        for i in 10..15u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            w.push(event);
        }

        let ratio = cancel_to_add_ratio(&w);
        assert!((ratio - 0.5).abs() < 0.01, "5/10 = 0.5, got {}", ratio);
    }

    #[test]
    fn test_cancel_to_add_ratio_no_adds() {
        let mut w = MboWindow::new(1000);
        for i in 0..5u64 {
            let event = MboEvent::new(
                i * 1_000_000,
                Action::Cancel,
                Side::Bid,
                100_000_000_000,
                100,
                i,
            );
            w.push(event);
        }
        assert_eq!(cancel_to_add_ratio(&w), 10.0);
    }

    #[test]
    fn test_cancel_to_add_ratio_no_activity() {
        let w = MboWindow::new(1000);
        assert_eq!(cancel_to_add_ratio(&w), 1.0);
    }
}
