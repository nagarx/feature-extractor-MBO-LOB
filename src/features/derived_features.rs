//! Derived feature computation from LOB snapshots.
//!
//! Computes meaningful derived features such as:
//! - Mid-price and spread
//! - Volume imbalances
//! - Weighted prices
//! - Market depth metrics

use mbo_lob_reconstructor::{LobState, Result, TlobError};

/// Compute derived features from a LOB state.
///
/// Returns 8 derived features:
/// 1. Mid-price
/// 2. Spread (absolute)
/// 3. Spread (basis points)
/// 4. Total bid volume
/// 5. Total ask volume
/// 6. Volume imbalance
/// 7. Weighted mid-price
/// 8. Price impact
///
/// # Performance
///
/// All computations are done in a single pass where possible.
/// Fixed-size array allocation for predictable performance.
#[inline]
pub fn compute_derived_features(lob_state: &LobState, levels: usize) -> Result<[f64; 8]> {
    let extract_levels = levels.min(lob_state.levels);

    // Get best prices
    let best_bid = lob_state
        .best_bid
        .ok_or_else(|| TlobError::generic("Cannot compute derived features: no best bid"))?;
    let best_ask = lob_state
        .best_ask
        .ok_or_else(|| TlobError::generic("Cannot compute derived features: no best ask"))?;

    let best_bid_f64 = best_bid as f64 / 1e9;
    let best_ask_f64 = best_ask as f64 / 1e9;

    // 1. Mid-price
    let mid_price = (best_bid_f64 + best_ask_f64) / 2.0;

    // 2. Spread (absolute)
    let spread = best_ask_f64 - best_bid_f64;

    // 3. Spread (basis points)
    let spread_bps = if mid_price > 0.0 {
        (spread / mid_price) * 10_000.0
    } else {
        0.0
    };

    // 4 & 5. Total bid and ask volumes (single pass)
    let mut total_bid_volume = 0u64;
    let mut total_ask_volume = 0u64;

    for i in 0..extract_levels {
        total_bid_volume += lob_state.bid_sizes[i] as u64;
        total_ask_volume += lob_state.ask_sizes[i] as u64;
    }

    let total_bid_f64 = total_bid_volume as f64;
    let total_ask_f64 = total_ask_volume as f64;

    // 6. Volume imbalance
    let total_volume = total_bid_f64 + total_ask_f64;
    let volume_imbalance = if total_volume > 0.0 {
        (total_bid_f64 - total_ask_f64) / total_volume
    } else {
        0.0
    };

    // 7. Weighted mid-price (volume-weighted)
    let best_bid_size = lob_state.bid_sizes[0] as f64;
    let best_ask_size = lob_state.ask_sizes[0] as f64;
    let total_best_size = best_bid_size + best_ask_size;

    let weighted_mid_price = if total_best_size > 0.0 {
        (best_bid_f64 * best_ask_size + best_ask_f64 * best_bid_size) / total_best_size
    } else {
        mid_price
    };

    // 8. Price impact (difference between mid-price and weighted mid-price)
    let price_impact = (mid_price - weighted_mid_price).abs();

    Ok([
        mid_price,
        spread,
        spread_bps,
        total_bid_f64,
        total_ask_f64,
        volume_imbalance,
        weighted_mid_price,
        price_impact,
    ])
}

/// Compute market depth features.
///
/// Returns depth metrics:
/// - Number of active bid levels
/// - Number of active ask levels
/// - Average bid level depth (ticks from best)
/// - Average ask level depth (ticks from best)
#[inline]
pub fn compute_depth_features(lob_state: &LobState, levels: usize, tick_size: f64) -> [f64; 4] {
    let extract_levels = levels.min(lob_state.levels);

    let mut active_bid_levels = 0;
    let mut active_ask_levels = 0;
    let mut total_bid_depth_ticks = 0.0;
    let mut total_ask_depth_ticks = 0.0;

    let best_bid_f64 = lob_state.best_bid.map(|p| p as f64 / 1e9).unwrap_or(0.0);
    let best_ask_f64 = lob_state.best_ask.map(|p| p as f64 / 1e9).unwrap_or(0.0);

    for i in 0..extract_levels {
        // Count active bid levels and compute depth
        if lob_state.bid_prices[i] > 0 {
            active_bid_levels += 1;
            let price_f64 = lob_state.bid_prices[i] as f64 / 1e9;
            let depth_ticks = (best_bid_f64 - price_f64) / tick_size;
            total_bid_depth_ticks += depth_ticks;
        }

        // Count active ask levels and compute depth
        if lob_state.ask_prices[i] > 0 {
            active_ask_levels += 1;
            let price_f64 = lob_state.ask_prices[i] as f64 / 1e9;
            let depth_ticks = (price_f64 - best_ask_f64) / tick_size;
            total_ask_depth_ticks += depth_ticks;
        }
    }

    let avg_bid_depth = if active_bid_levels > 0 {
        total_bid_depth_ticks / active_bid_levels as f64
    } else {
        0.0
    };

    let avg_ask_depth = if active_ask_levels > 0 {
        total_ask_depth_ticks / active_ask_levels as f64
    } else {
        0.0
    };

    [
        active_bid_levels as f64,
        active_ask_levels as f64,
        avg_bid_depth,
        avg_ask_depth,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lob_state() -> LobState {
        let mut state = LobState::new(10);

        // Best bid: $100.00 × 100
        state.bid_prices[0] = 100_000_000_000;
        state.bid_sizes[0] = 100;

        // Best ask: $100.01 × 200
        state.ask_prices[0] = 100_010_000_000;
        state.ask_sizes[0] = 200;

        // Level 2 bid: $99.99 × 150
        state.bid_prices[1] = 99_990_000_000;
        state.bid_sizes[1] = 150;

        // Level 2 ask: $100.02 × 180
        state.ask_prices[1] = 100_020_000_000;
        state.ask_sizes[1] = 180;

        // Level 3 bid: $99.98 × 200
        state.bid_prices[2] = 99_980_000_000;
        state.bid_sizes[2] = 200;

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        state
    }

    #[test]
    fn test_compute_derived_features() {
        let state = create_test_lob_state();
        let features = compute_derived_features(&state, 10).unwrap();

        // 1. Mid-price should be $100.005
        assert!((features[0] - 100.005).abs() < 1e-6);

        // 2. Spread should be $0.01
        assert!((features[1] - 0.01).abs() < 1e-6);

        // 3. Spread (bps) should be ~1.0 bps
        assert!((features[2] - 0.9995).abs() < 0.01);

        // 4. Total bid volume
        assert_eq!(features[3], 450.0); // 100 + 150 + 200

        // 5. Total ask volume
        assert_eq!(features[4], 380.0); // 200 + 180

        // 6. Volume imbalance (more bids than asks)
        assert!(features[5] > 0.0); // Positive imbalance
        assert!((features[5] - 0.0843).abs() < 0.001); // ~8.43%

        // 7. Weighted mid-price (weighted toward larger size at best ask)
        assert!(features[6] > 100.00 && features[6] < 100.01);

        // 8. Price impact (small)
        assert!(features[7] < 0.01);
    }

    #[test]
    fn test_volume_imbalance_balanced() {
        let mut state = LobState::new(10);

        state.bid_prices[0] = 100_000_000_000;
        state.bid_sizes[0] = 100;
        state.ask_prices[0] = 100_010_000_000;
        state.ask_sizes[0] = 100;

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        let features = compute_derived_features(&state, 10).unwrap();

        // Volume imbalance should be ~0 (balanced)
        assert!(features[5].abs() < 1e-6);
    }

    #[test]
    fn test_compute_depth_features() {
        let state = create_test_lob_state();
        let features = compute_depth_features(&state, 10, 0.01);

        // Active bid levels: 3
        assert_eq!(features[0], 3.0);

        // Active ask levels: 2
        assert_eq!(features[1], 2.0);

        // Average bid depth in ticks
        // Level 1: 0 ticks, Level 2: 1 tick, Level 3: 2 ticks
        // Average: (0 + 1 + 2) / 3 = 1.0
        assert!((features[2] - 1.0).abs() < 1e-6);

        // Average ask depth in ticks
        // Level 1: 0 ticks, Level 2: 1 tick
        // Average: (0 + 1) / 2 = 0.5
        assert!((features[3] - 0.5).abs() < 1e-6); // Fixed: should be 0.5, not 1.0
    }

    #[test]
    fn test_no_best_prices() {
        let state = LobState::new(10);
        let result = compute_derived_features(&state, 10);

        // Should return error when no best prices
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_mid_price() {
        let mut state = LobState::new(10);

        // Large size on bid side
        state.bid_prices[0] = 100_000_000_000;
        state.bid_sizes[0] = 1000; // Large
        state.ask_prices[0] = 100_010_000_000;
        state.ask_sizes[0] = 10; // Small

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        let features = compute_derived_features(&state, 10).unwrap();

        // Weighted mid-price should be closer to ask (where volume is smaller)
        // Because formula is: (bid × ask_size + ask × bid_size) / total_size
        assert!(features[6] > features[0]); // Weighted > simple mid
        assert!(features[6] < 100.01); // But still less than best ask
    }

    // ====================================================================
    // Golden value tests — hand-calculated expected values (Rule §6)
    // ====================================================================

    /// Helper: create a LobState with specified bid/ask prices and sizes.
    /// Prices in nanodollars. Only populates level 0.
    fn make_lob(
        bid_price_nd: i64,
        ask_price_nd: i64,
        bid_size: u32,
        ask_size: u32,
    ) -> LobState {
        let mut state = LobState::new(10);
        state.bid_prices[0] = bid_price_nd;
        state.bid_sizes[0] = bid_size;
        state.ask_prices[0] = ask_price_nd;
        state.ask_sizes[0] = ask_size;
        state.best_bid = Some(bid_price_nd);
        state.best_ask = Some(ask_price_nd);
        state
    }

    #[test]
    fn test_golden_mid_price() {
        // bid = $100.00, ask = $100.02
        // mid = (100.00 + 100.02) / 2 = 100.01
        let state = make_lob(100_000_000_000, 100_020_000_000, 100, 100);
        let f = compute_derived_features(&state, 10).unwrap();
        assert!(
            (f[0] - 100.01).abs() < 1e-10,
            "mid_price: expected 100.01, got {}. Formula: (bid + ask) / 2",
            f[0]
        );
    }

    #[test]
    fn test_golden_spread_bps() {
        // bid = $100.00, ask = $100.02 → spread = $0.02, mid = $100.01
        // spread_bps = (0.02 / 100.01) * 10000 = 1.99980002... bps
        let state = make_lob(100_000_000_000, 100_020_000_000, 100, 100);
        let f = compute_derived_features(&state, 10).unwrap();
        let expected_bps = (0.02 / 100.01) * 10_000.0;
        assert!(
            (f[2] - expected_bps).abs() < 1e-8,
            "spread_bps: expected {}, got {}. Formula: (spread / mid) * 10000",
            expected_bps, f[2]
        );
    }

    #[test]
    fn test_golden_volume_imbalance() {
        // bid_vol = 300, ask_vol = 100 → total = 400
        // vi = (300 - 100) / 400 = 0.5
        let state = make_lob(100_000_000_000, 100_010_000_000, 300, 100);
        let f = compute_derived_features(&state, 10).unwrap();
        assert!(
            (f[5] - 0.5).abs() < 1e-10,
            "volume_imbalance: expected 0.5, got {}. Formula: (bid-ask)/total",
            f[5]
        );
    }

    #[test]
    fn test_golden_weighted_mid_price() {
        // bid=$100.00 × 200, ask=$100.02 × 100
        // wmp = (100.00 * 100 + 100.02 * 200) / (200 + 100)
        //     = (10000 + 20004) / 300 = 30004 / 300 = 100.01333...
        let state = make_lob(100_000_000_000, 100_020_000_000, 200, 100);
        let f = compute_derived_features(&state, 10).unwrap();
        let expected_wmp = (100.00 * 100.0 + 100.02 * 200.0) / 300.0;
        assert!(
            (f[6] - expected_wmp).abs() < 1e-8,
            "weighted_mid_price: expected {}, got {}. Formula: (bid*ask_size + ask*bid_size) / total",
            expected_wmp, f[6]
        );
    }

    // ====================================================================
    // Edge case tests — zero/boundary values (Rule §2)
    // ====================================================================

    #[test]
    fn test_zero_total_volume() {
        // All sizes = 0 → volume_imbalance = 0.0, no div-by-zero
        let state = make_lob(100_000_000_000, 100_010_000_000, 0, 0);
        let f = compute_derived_features(&state, 10).unwrap();
        assert_eq!(f[3], 0.0, "total_bid_volume should be 0");
        assert_eq!(f[4], 0.0, "total_ask_volume should be 0");
        assert_eq!(
            f[5], 0.0,
            "volume_imbalance must be 0.0 when total_volume=0, not NaN/Inf"
        );
        assert!(f[5].is_finite(), "volume_imbalance must be finite");
    }

    #[test]
    fn test_zero_best_sizes() {
        // Best sizes = 0 → weighted_mid_price falls back to mid_price
        let state = make_lob(100_000_000_000, 100_010_000_000, 0, 0);
        let f = compute_derived_features(&state, 10).unwrap();
        let mid_price = f[0];
        assert_eq!(
            f[6], mid_price,
            "weighted_mid_price must fall back to mid_price when best sizes are 0"
        );
        assert_eq!(
            f[7], 0.0,
            "price_impact must be 0.0 when wmp == mid (no imbalance)"
        );
    }

    #[test]
    fn test_single_level_depth() {
        // Only level 0 populated → avg depth = 0 ticks (only best level)
        let state = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        let depth = compute_depth_features(&state, 10, 0.01);
        assert_eq!(depth[0], 1.0, "active_bid_levels should be 1");
        assert_eq!(depth[1], 1.0, "active_ask_levels should be 1");
        assert_eq!(
            depth[2], 0.0,
            "avg_bid_depth should be 0 ticks (only best level)"
        );
        assert_eq!(
            depth[3], 0.0,
            "avg_ask_depth should be 0 ticks (only best level)"
        );
    }

    #[test]
    fn test_all_features_finite() {
        // Comprehensive finiteness check on standard inputs
        let state = create_test_lob_state();
        let f = compute_derived_features(&state, 10).unwrap();
        for (i, &val) in f.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Feature {} must be finite, got {}",
                i, val
            );
        }
    }
}
