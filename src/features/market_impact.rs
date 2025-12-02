//! Market Impact Estimation
//!
//! Provides features for estimating the market impact of hypothetical orders.
//! Based on OrderBook-rs market_impact module.
//!
//! # Key Features
//!
//! - **Slippage**: Price deviation from best price when executing a large order
//! - **VWAP**: Volume-Weighted Average Price for order execution
//! - **Levels Consumed**: Number of price levels needed to fill an order
//! - **Fill Ratio**: Percentage of order that can be filled from available liquidity
//!
//! # Research Background
//!
//! Market impact estimation is crucial for:
//! - Execution quality prediction
//! - Optimal order sizing
//! - Liquidity analysis
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::features::market_impact::{estimate_buy_impact, MarketImpactFeatures};
//!
//! let impact = estimate_buy_impact(&lob, 1000);
//! println!("VWAP: {}", impact.vwap);
//! println!("Slippage (bps): {}", impact.slippage_bps);
//! println!("Levels consumed: {}", impact.levels_consumed);
//! ```

use mbo_lob_reconstructor::LobState;

/// Market impact estimation results.
#[derive(Debug, Clone, Default)]
pub struct MarketImpactFeatures {
    /// Volume-Weighted Average Price for the simulated execution
    pub vwap: f64,

    /// Best price at time of estimation (best ask for buy, best bid for sell)
    pub best_price: f64,

    /// Slippage in absolute price terms
    pub slippage: f64,

    /// Slippage in basis points (relative to best price)
    pub slippage_bps: f64,

    /// Number of price levels consumed to fill the order
    pub levels_consumed: usize,

    /// Quantity that could be filled
    pub filled_quantity: u64,

    /// Requested quantity
    pub requested_quantity: u64,

    /// Fill ratio (0.0 to 1.0)
    pub fill_ratio: f64,

    /// Total cost of execution
    pub total_cost: f64,

    /// Whether the order can be fully filled
    pub can_fill: bool,
}

impl MarketImpactFeatures {
    /// Convert to feature vector (10 features).
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.vwap,
            self.best_price,
            self.slippage,
            self.slippage_bps,
            self.levels_consumed as f64,
            self.filled_quantity as f64,
            self.requested_quantity as f64,
            self.fill_ratio,
            self.total_cost,
            if self.can_fill { 1.0 } else { 0.0 },
        ]
    }

    /// Number of features.
    pub const fn count() -> usize {
        10
    }
}

/// Estimate market impact for a buy order.
///
/// Simulates walking through the ask side of the order book to fill
/// the specified quantity.
///
/// # Arguments
///
/// * `lob` - Current LOB state
/// * `quantity` - Quantity to buy
///
/// # Returns
///
/// Market impact features for the simulated buy order.
#[inline]
pub fn estimate_buy_impact(lob: &LobState, quantity: u64) -> MarketImpactFeatures {
    estimate_impact_internal(lob, quantity, false)
}

/// Estimate market impact for a sell order.
///
/// Simulates walking through the bid side of the order book to fill
/// the specified quantity.
///
/// # Arguments
///
/// * `lob` - Current LOB state
/// * `quantity` - Quantity to sell
///
/// # Returns
///
/// Market impact features for the simulated sell order.
#[inline]
pub fn estimate_sell_impact(lob: &LobState, quantity: u64) -> MarketImpactFeatures {
    estimate_impact_internal(lob, quantity, true)
}

/// Internal implementation for market impact estimation.
#[inline]
fn estimate_impact_internal(lob: &LobState, quantity: u64, is_sell: bool) -> MarketImpactFeatures {
    let (prices, sizes, best_price_opt) = if is_sell {
        (&lob.bid_prices, &lob.bid_sizes, lob.best_bid)
    } else {
        (&lob.ask_prices, &lob.ask_sizes, lob.best_ask)
    };

    let best_price = match best_price_opt {
        Some(p) => p as f64 / 1e9,
        None => {
            return MarketImpactFeatures {
                requested_quantity: quantity,
                ..Default::default()
            };
        }
    };

    if quantity == 0 {
        return MarketImpactFeatures {
            best_price,
            vwap: best_price,
            fill_ratio: 1.0,
            can_fill: true,
            ..Default::default()
        };
    }

    let mut remaining = quantity;
    let mut total_cost = 0.0;
    let mut filled = 0u64;
    let mut levels_consumed = 0;

    for i in 0..lob.levels {
        if remaining == 0 {
            break;
        }

        let price = prices[i];
        let size = sizes[i] as u64;

        if price == 0 || size == 0 {
            continue;
        }

        let price_f64 = price as f64 / 1e9;
        let fill_at_level = remaining.min(size);

        total_cost += price_f64 * fill_at_level as f64;
        filled += fill_at_level;
        remaining -= fill_at_level;
        levels_consumed += 1;
    }

    let vwap = if filled > 0 {
        total_cost / filled as f64
    } else {
        best_price
    };

    let slippage = if is_sell {
        best_price - vwap // Selling: VWAP < best bid is bad
    } else {
        vwap - best_price // Buying: VWAP > best ask is bad
    };

    let slippage_bps = if best_price > 0.0 {
        (slippage / best_price) * 10_000.0
    } else {
        0.0
    };

    let fill_ratio = if quantity > 0 {
        filled as f64 / quantity as f64
    } else {
        1.0
    };

    MarketImpactFeatures {
        vwap,
        best_price,
        slippage,
        slippage_bps,
        levels_consumed,
        filled_quantity: filled,
        requested_quantity: quantity,
        fill_ratio,
        total_cost,
        can_fill: remaining == 0,
    }
}

/// Estimate market impact for both buy and sell sides.
///
/// # Arguments
///
/// * `lob` - Current LOB state
/// * `quantity` - Quantity to trade
///
/// # Returns
///
/// Tuple of (buy_impact, sell_impact)
#[inline]
pub fn estimate_both_sides(
    lob: &LobState,
    quantity: u64,
) -> (MarketImpactFeatures, MarketImpactFeatures) {
    (
        estimate_buy_impact(lob, quantity),
        estimate_sell_impact(lob, quantity),
    )
}

/// Compute market impact features for multiple order sizes.
///
/// This is useful for understanding liquidity across different order sizes.
///
/// # Arguments
///
/// * `lob` - Current LOB state
/// * `quantities` - Slice of quantities to estimate
///
/// # Returns
///
/// Vector of (buy_impact, sell_impact) tuples for each quantity.
pub fn estimate_multiple_sizes(
    lob: &LobState,
    quantities: &[u64],
) -> Vec<(MarketImpactFeatures, MarketImpactFeatures)> {
    quantities
        .iter()
        .map(|&q| estimate_both_sides(lob, q))
        .collect()
}

/// Compute liquidity depth features.
///
/// Analyzes how much liquidity is available at different price depths.
///
/// # Arguments
///
/// * `lob` - Current LOB state
/// * `depth_bps` - Slice of basis point depths to analyze (e.g., [10, 25, 50, 100])
///
/// # Returns
///
/// Vector of (bid_volume, ask_volume) tuples for each depth.
pub fn compute_liquidity_at_depths(lob: &LobState, depth_bps: &[f64]) -> Vec<(f64, f64)> {
    let mid_price = match lob.mid_price() {
        Some(p) => p,
        None => return vec![(0.0, 0.0); depth_bps.len()],
    };

    depth_bps
        .iter()
        .map(|&bps| {
            let price_range = mid_price * bps / 10_000.0;

            // Count bid volume within range
            let bid_vol: f64 = (0..lob.levels)
                .filter_map(|i| {
                    let price = lob.bid_prices[i] as f64 / 1e9;
                    if price > 0.0 && (mid_price - price) <= price_range {
                        Some(lob.bid_sizes[i] as f64)
                    } else {
                        None
                    }
                })
                .sum();

            // Count ask volume within range
            let ask_vol: f64 = (0..lob.levels)
                .filter_map(|i| {
                    let price = lob.ask_prices[i] as f64 / 1e9;
                    if price > 0.0 && (price - mid_price) <= price_range {
                        Some(lob.ask_sizes[i] as f64)
                    } else {
                        None
                    }
                })
                .sum();

            (bid_vol, ask_vol)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lob() -> LobState {
        let mut lob = LobState::new(10);

        // Ask side: 100.01 × 100, 100.02 × 200, 100.03 × 300
        lob.ask_prices[0] = 100_010_000_000;
        lob.ask_sizes[0] = 100;
        lob.ask_prices[1] = 100_020_000_000;
        lob.ask_sizes[1] = 200;
        lob.ask_prices[2] = 100_030_000_000;
        lob.ask_sizes[2] = 300;

        // Bid side: 100.00 × 100, 99.99 × 200, 99.98 × 300
        lob.bid_prices[0] = 100_000_000_000;
        lob.bid_sizes[0] = 100;
        lob.bid_prices[1] = 99_990_000_000;
        lob.bid_sizes[1] = 200;
        lob.bid_prices[2] = 99_980_000_000;
        lob.bid_sizes[2] = 300;

        lob.best_ask = Some(100_010_000_000);
        lob.best_bid = Some(100_000_000_000);

        lob
    }

    #[test]
    fn test_buy_impact_single_level() {
        let lob = create_test_lob();

        // Buy 50 shares (less than level 1)
        let impact = estimate_buy_impact(&lob, 50);

        assert_eq!(impact.requested_quantity, 50);
        assert_eq!(impact.filled_quantity, 50);
        assert!(impact.can_fill);
        assert_eq!(impact.levels_consumed, 1);
        assert!((impact.vwap - 100.01).abs() < 1e-6);
        assert!(impact.slippage.abs() < 1e-6); // No slippage at single level
    }

    #[test]
    fn test_buy_impact_multiple_levels() {
        let lob = create_test_lob();

        // Buy 150 shares (spans 2 levels)
        let impact = estimate_buy_impact(&lob, 150);

        assert_eq!(impact.requested_quantity, 150);
        assert_eq!(impact.filled_quantity, 150);
        assert!(impact.can_fill);
        assert_eq!(impact.levels_consumed, 2);

        // VWAP: (100 × 100.01 + 50 × 100.02) / 150 = 100.0133...
        assert!(impact.vwap > 100.01);
        assert!(impact.vwap < 100.02);

        // Slippage should be positive
        assert!(impact.slippage > 0.0);
        assert!(impact.slippage_bps > 0.0);
    }

    #[test]
    fn test_buy_impact_exceeds_liquidity() {
        let lob = create_test_lob();

        // Buy 1000 shares (more than available: 100+200+300=600)
        let impact = estimate_buy_impact(&lob, 1000);

        assert_eq!(impact.requested_quantity, 1000);
        assert_eq!(impact.filled_quantity, 600);
        assert!(!impact.can_fill);
        assert_eq!(impact.levels_consumed, 3);
        assert!((impact.fill_ratio - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_sell_impact_single_level() {
        let lob = create_test_lob();

        // Sell 50 shares (less than level 1)
        let impact = estimate_sell_impact(&lob, 50);

        assert_eq!(impact.requested_quantity, 50);
        assert_eq!(impact.filled_quantity, 50);
        assert!(impact.can_fill);
        assert_eq!(impact.levels_consumed, 1);
        assert!((impact.vwap - 100.00).abs() < 1e-6);
        assert!(impact.slippage.abs() < 1e-6);
    }

    #[test]
    fn test_sell_impact_multiple_levels() {
        let lob = create_test_lob();

        // Sell 150 shares (spans 2 levels)
        let impact = estimate_sell_impact(&lob, 150);

        assert_eq!(impact.levels_consumed, 2);

        // VWAP: (100 × 100.00 + 50 × 99.99) / 150 = 99.9966...
        assert!(impact.vwap < 100.00);
        assert!(impact.vwap > 99.99);

        // Slippage should be positive (selling at worse prices)
        assert!(impact.slippage > 0.0);
    }

    #[test]
    fn test_empty_lob() {
        let lob = LobState::new(10);

        let buy_impact = estimate_buy_impact(&lob, 100);
        assert_eq!(buy_impact.filled_quantity, 0);
        assert!(!buy_impact.can_fill);

        let sell_impact = estimate_sell_impact(&lob, 100);
        assert_eq!(sell_impact.filled_quantity, 0);
        assert!(!sell_impact.can_fill);
    }

    #[test]
    fn test_zero_quantity() {
        let lob = create_test_lob();

        let impact = estimate_buy_impact(&lob, 0);
        assert!(impact.can_fill);
        assert_eq!(impact.fill_ratio, 1.0);
    }

    #[test]
    fn test_both_sides() {
        let lob = create_test_lob();

        let (buy, sell) = estimate_both_sides(&lob, 100);

        assert_eq!(buy.filled_quantity, 100);
        assert_eq!(sell.filled_quantity, 100);
        assert!(buy.can_fill);
        assert!(sell.can_fill);
    }

    #[test]
    fn test_multiple_sizes() {
        let lob = create_test_lob();

        let results = estimate_multiple_sizes(&lob, &[50, 100, 200, 500]);

        assert_eq!(results.len(), 4);

        // Smaller orders should have less slippage
        assert!(results[0].0.slippage_bps <= results[1].0.slippage_bps);
        assert!(results[1].0.slippage_bps <= results[2].0.slippage_bps);
    }

    #[test]
    fn test_liquidity_at_depths() {
        let lob = create_test_lob();

        let depths = compute_liquidity_at_depths(&lob, &[10.0, 50.0, 100.0]);

        assert_eq!(depths.len(), 3);

        // More depth should include more volume
        assert!(depths[0].0 <= depths[1].0);
        assert!(depths[1].0 <= depths[2].0);
    }

    #[test]
    fn test_to_vec() {
        let impact = MarketImpactFeatures {
            vwap: 100.01,
            best_price: 100.00,
            slippage: 0.01,
            slippage_bps: 1.0,
            levels_consumed: 2,
            filled_quantity: 150,
            requested_quantity: 150,
            fill_ratio: 1.0,
            total_cost: 15001.5,
            can_fill: true,
        };

        let vec = impact.to_vec();
        assert_eq!(vec.len(), MarketImpactFeatures::count());
        assert_eq!(vec[0], 100.01);
        assert_eq!(vec[9], 1.0); // can_fill
    }
}
