//! Order flow feature extraction (12 features, contract indices 48-59).
//!
//! Captures buying/selling pressure through event rates, net flow
//! imbalances, aggressive order ratios, flow volatility, and regime indicators.
//!
//! Sign convention (RULE.md §9):
//! - `> 0` = Bullish / Buy pressure
//! - `< 0` = Bearish / Sell pressure
//! - `= 0` = Neutral / No signal

use super::window::MboWindow;
use crate::contract::DIVISION_GUARD_EPS;
use mbo_lob_reconstructor::{Action, Side};

/// Extract 12 order flow features from three windows.
///
/// Returns `[f64; 12]` to avoid heap allocation in the hot path.
///
/// Layout:
/// - [0-5]: Event rates (add_rate_bid/ask, cancel_rate_bid/ask, trade_rate_bid/ask)
/// - [6-9]: Net flow imbalances (order, cancel, trade, aggressive ratio)
/// - [10-11]: Flow characteristics (volatility, regime)
pub(super) fn extract(medium: &MboWindow, fast: &MboWindow, slow: &MboWindow) -> [f64; 12] {
    let duration = medium.duration_seconds();

    [
        // Event rates (6)
        medium.add_count_bid as f64 / duration,
        medium.add_count_ask as f64 / duration,
        medium.cancel_count_bid as f64 / duration,
        medium.cancel_count_ask as f64 / duration,
        medium.trade_count_bid as f64 / duration,
        medium.trade_count_ask as f64 / duration,
        // Net flow imbalances (4)
        net_order_flow(medium),
        net_cancel_flow(medium),
        net_trade_flow(medium),
        aggressive_order_ratio(medium),
        // Flow characteristics (2)
        order_flow_volatility(medium),
        flow_regime_indicator(fast, slow),
    ]
}

#[inline]
fn net_order_flow(w: &MboWindow) -> f64 {
    let net = (w.add_count_bid as i64) - (w.add_count_ask as i64);
    let total = (w.add_count_bid + w.add_count_ask) as f64;
    net as f64 / (total + DIVISION_GUARD_EPS)
}

/// Net cancel flow: (cancel_ask - cancel_bid) / total
///
/// `> 0`: More ask cancels (sellers pulling) = BULLISH
/// `< 0`: More bid cancels (buyers pulling) = BEARISH
#[inline]
fn net_cancel_flow(w: &MboWindow) -> f64 {
    let net = (w.cancel_count_ask as i64) - (w.cancel_count_bid as i64);
    let total = (w.cancel_count_bid + w.cancel_count_ask) as f64;
    net as f64 / (total + DIVISION_GUARD_EPS)
}

/// Net trade flow: (trade_ask - trade_bid) / total
///
/// In MBO data:
/// - Trade with Side::Bid = bid was HIT = SELL-initiated
/// - Trade with Side::Ask = ask was HIT = BUY-initiated
///
/// `> 0`: More BUY-initiated trades = BULLISH
/// `< 0`: More SELL-initiated trades = BEARISH
#[inline]
fn net_trade_flow(w: &MboWindow) -> f64 {
    let net = (w.trade_count_ask as i64) - (w.trade_count_bid as i64);
    let total = (w.trade_count_bid + w.trade_count_ask) as f64;
    net as f64 / (total + DIVISION_GUARD_EPS)
}

#[inline]
fn aggressive_order_ratio(w: &MboWindow) -> f64 {
    let market_orders = w.trade_count_bid + w.trade_count_ask;
    let total_orders = w.add_count_bid + w.add_count_ask + market_orders;
    market_orders as f64 / (total_orders as f64 + DIVISION_GUARD_EPS)
}

/// Standard deviation of net_order_flow across sub-windows.
fn order_flow_volatility(w: &MboWindow) -> f64 {
    const N_SUBWINDOWS: usize = 10;
    const MIN_EVENTS_PER_SUBWINDOW: usize = 5;

    let n_events = w.events.len();
    if n_events < N_SUBWINDOWS * MIN_EVENTS_PER_SUBWINDOW {
        return 0.0;
    }

    let subwindow_size = n_events / N_SUBWINDOWS;
    let mut subwindow_flows: Vec<f64> = Vec::with_capacity(N_SUBWINDOWS);

    for i in 0..N_SUBWINDOWS {
        let start = i * subwindow_size;
        let end = if i == N_SUBWINDOWS - 1 {
            n_events
        } else {
            (i + 1) * subwindow_size
        };

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

        let total = (add_bid + add_ask) as f64;
        let flow = if total > 0.0 {
            (add_bid as f64 - add_ask as f64) / total
        } else {
            0.0
        };
        subwindow_flows.push(flow);
    }

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

/// Flow regime indicator: ratio of fast to slow window net order flow.
///
/// Uses proper epsilon (0.01) and output clamping [-10, 10] to prevent
/// extreme values when slow_flow is near zero.
fn flow_regime_indicator(fast: &MboWindow, slow: &MboWindow) -> f64 {
    const MIN_DENOM: f64 = 0.01;
    const MAX_ABS: f64 = 10.0;

    let fast_flow = net_order_flow(fast);
    let slow_flow = net_order_flow(slow);

    let denom = slow_flow.abs().max(MIN_DENOM);
    let ratio = fast_flow / denom;
    ratio.clamp(-MAX_ABS, MAX_ABS)
}

#[cfg(test)]
mod tests {
    use super::super::event::MboEvent;
    use super::super::window::MboWindow;
    use super::*;

    fn make_window_with_events(events: &[(Action, Side)]) -> MboWindow {
        let mut w = MboWindow::new(1000);
        for (i, (action, side)) in events.iter().enumerate() {
            let event = MboEvent::new(
                i as u64 * 1_000_000,
                *action,
                *side,
                100_000_000_000,
                100,
                i as u64,
            );
            w.push(event);
        }
        w
    }

    #[test]
    fn test_net_order_flow_bullish() {
        let mut events = Vec::new();
        for _ in 0..10 {
            events.push((Action::Add, Side::Bid));
        }
        for _ in 0..5 {
            events.push((Action::Add, Side::Ask));
        }
        let w = make_window_with_events(&events);
        let flow = net_order_flow(&w);
        assert!(flow > 0.0, "More bid adds should be bullish, got {}", flow);
        let expected = (10.0 - 5.0) / 15.0;
        assert!((flow - expected).abs() < 0.01);
    }

    #[test]
    fn test_net_cancel_flow_bullish() {
        let mut events = Vec::new();
        for _ in 0..3 {
            events.push((Action::Cancel, Side::Bid));
        }
        for _ in 0..8 {
            events.push((Action::Cancel, Side::Ask));
        }
        let w = make_window_with_events(&events);
        let flow = net_cancel_flow(&w);
        assert!(flow > 0.0, "More ask cancels = bullish, got {}", flow);
    }

    #[test]
    fn test_net_trade_flow_bullish() {
        let mut events = Vec::new();
        for _ in 0..3 {
            events.push((Action::Trade, Side::Bid));
        }
        for _ in 0..8 {
            events.push((Action::Trade, Side::Ask));
        }
        let w = make_window_with_events(&events);
        let flow = net_trade_flow(&w);
        assert!(
            flow > 0.0,
            "More ask trades (buy-initiated) = bullish, got {}",
            flow
        );
    }

    #[test]
    fn test_flow_symmetry() {
        let mut events = Vec::new();
        for _ in 0..10 {
            events.push((Action::Add, Side::Bid));
        }
        for _ in 0..10 {
            events.push((Action::Add, Side::Ask));
        }
        let w = make_window_with_events(&events);
        assert!(net_order_flow(&w).abs() < 0.01);
    }

    #[test]
    fn test_order_flow_volatility_insufficient_data() {
        let mut w = MboWindow::new(1000);
        for i in 0..40u64 {
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
        assert_eq!(order_flow_volatility(&w), 0.0);
    }

    #[test]
    fn test_order_flow_volatility_constant_flow() {
        let mut w = MboWindow::new(1000);
        for i in 0..100u64 {
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
        let vol = order_flow_volatility(&w);
        assert!(
            vol < 0.01,
            "Constant flow should have near-zero volatility, got {}",
            vol
        );
    }

    #[test]
    fn test_flow_regime_indicator_bounded() {
        let fast = MboWindow::new(50);
        let slow = MboWindow::new(500);
        let result = flow_regime_indicator(&fast, &slow);
        assert!(result.abs() <= 10.0, "Must be bounded, got {}", result);
    }
}
