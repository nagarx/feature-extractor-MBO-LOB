//! Institutional detection feature extraction (4 features, contract indices 74-77).
//!
//! Identifies whale trading patterns through large order frequency/imbalance,
//! modification patterns, and iceberg order proxy detection.

use super::order_tracker::OrderTracker;
use super::window::MboWindow;
use crate::contract::DIVISION_GUARD_EPS;
use mbo_lob_reconstructor::Side;

/// Extract 4 institutional detection features.
///
/// Layout:
/// - [0]: Large order frequency (large orders per second)
/// - [1]: Large order imbalance (bid vs ask large order volume)
/// - [2]: Modification score (avg modifications per completed order)
/// - [3]: Iceberg proxy (fill_ratio * modification_score)
pub(super) fn extract(window: &mut MboWindow, tracker: &OrderTracker) -> [f64; 4] {
    let mod_score = modification_score(tracker);
    let fill_ratio = average_fill_ratio(tracker);

    [
        large_order_frequency(window),
        large_order_imbalance(window),
        mod_score,
        iceberg_proxy(fill_ratio, mod_score, tracker),
    ]
}

fn large_order_frequency(w: &mut MboWindow) -> f64 {
    let threshold = w.size_percentile(3); // p90
    let large_count = w
        .events
        .iter()
        .filter(|e| e.size as f64 > threshold)
        .count();
    large_count as f64 / w.duration_seconds()
}

fn large_order_imbalance(w: &mut MboWindow) -> f64 {
    let threshold = w.size_percentile(3); // p90
    let (large_bid, large_ask) = w.events.iter().filter(|e| e.size as f64 > threshold).fold(
        (0u64, 0u64),
        |(bid, ask), e| match e.side {
            Side::Bid => (bid + e.size as u64, ask),
            Side::Ask => (bid, ask + e.size as u64),
            Side::None => (bid, ask),
        },
    );
    let total = large_bid + large_ask;
    (large_bid as f64 - large_ask as f64) / (total as f64 + DIVISION_GUARD_EPS)
}

/// Average number of modifications per completed order.
fn modification_score(tracker: &OrderTracker) -> f64 {
    let mods = tracker.completed_modifications();
    if mods.is_empty() {
        return 0.0;
    }
    let total_mods: usize = mods.iter().map(|&m| m as usize).sum();
    total_mods as f64 / mods.len() as f64
}

/// Computes fill_ratio * (modification_score / 10).min(1.0).
/// High fill ratio combined with frequent modifications suggests iceberg orders.
fn iceberg_proxy(fill_ratio: f64, mod_score: f64, tracker: &OrderTracker) -> f64 {
    if tracker.is_empty() {
        return 0.0;
    }
    fill_ratio * (mod_score / 10.0).min(1.0)
}

/// Average fill ratio of completed orders (used internally for iceberg proxy).
fn average_fill_ratio(tracker: &OrderTracker) -> f64 {
    let ratios = tracker.completed_fill_ratios();
    if ratios.is_empty() {
        return 0.0;
    }
    let total: f64 = ratios.iter().sum();
    total / ratios.len() as f64
}

#[cfg(test)]
mod tests {
    use super::super::event::MboEvent;
    use super::super::order_tracker::OrderTracker;
    use super::*;
    use mbo_lob_reconstructor::Action;

    #[test]
    fn test_modification_score_empty() {
        let tracker = OrderTracker::new();
        assert_eq!(modification_score(&tracker), 0.0);
    }

    #[test]
    fn test_modification_score_no_modifications() {
        let mut tracker = OrderTracker::new();
        let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&add);
        let cancel = MboEvent::new(
            1_000_000,
            Action::Cancel,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        tracker.process_event(&cancel);
        assert_eq!(modification_score(&tracker), 0.0);
    }

    #[test]
    fn test_modification_score_with_modifications() {
        let mut tracker = OrderTracker::new();
        let add = MboEvent::new(0, Action::Add, Side::Bid, 100_000_000_000, 100, 1);
        tracker.process_event(&add);

        let modify1 = MboEvent::new(1_000_000, Action::Modify, Side::Bid, 99_990_000_000, 100, 1);
        tracker.process_event(&modify1);

        let modify2 = MboEvent::new(2_000_000, Action::Modify, Side::Bid, 99_980_000_000, 100, 1);
        tracker.process_event(&modify2);

        let cancel = MboEvent::new(
            3_000_000,
            Action::Cancel,
            Side::Bid,
            100_000_000_000,
            100,
            1,
        );
        tracker.process_event(&cancel);

        let score = modification_score(&tracker);
        assert!(
            (score - 2.0).abs() < 0.001,
            "Order with 2 modifications should have score 2.0, got {}",
            score
        );
    }
}
