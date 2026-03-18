//! Multi-Level Order Flow Imbalance (MLOFI)
//!
//! Wraps `MultiLevelOfiTracker` from `order_flow.rs` as an experimental feature group.
//!
//! # Reference
//!
//! Kolm, Turiel & Westray (2023), "Deep Order Flow Imbalance: Extracting Alpha
//! at Multiple Horizons from the Limit Order Book"
//! - R²_OOS ~0.60 for 10-level OFI vs ~0.10 for single-level OFI
//! - Per-level OFI captures depth-dependent order flow dynamics
//! - Weighted MLOFI (w_k = 1/k) gives higher weight to levels closer to BBO
//!
//! # Feature Layout (12 features)
//!
//! | Offset | Name | Description | Sign Convention |
//! |--------|------|-------------|-----------------|
//! | 0 | total_mlofi | Sum of OFI across all 10 levels | >0 bullish |
//! | 1 | weighted_mlofi | Level-weighted sum (w_k = 1/k) | >0 bullish |
//! | 2-11 | ofi_level_1..10 | Per-level OFI (bid+ask combined) | >0 bullish |

use mbo_lob_reconstructor::LobState;

use crate::features::order_flow::MultiLevelOfiTracker;

/// Number of features in this group.
pub const FEATURE_COUNT: usize = 12;

/// Number of LOB levels tracked for MLOFI.
pub const MLOFI_LEVELS: usize = 10;

/// Feature indices (relative to group start).
pub mod indices {
    pub const TOTAL_MLOFI: usize = 0;
    pub const WEIGHTED_MLOFI: usize = 1;
    pub const OFI_LEVEL_1: usize = 2;
    pub const OFI_LEVEL_2: usize = 3;
    pub const OFI_LEVEL_3: usize = 4;
    pub const OFI_LEVEL_4: usize = 5;
    pub const OFI_LEVEL_5: usize = 6;
    pub const OFI_LEVEL_6: usize = 7;
    pub const OFI_LEVEL_7: usize = 8;
    pub const OFI_LEVEL_8: usize = 9;
    pub const OFI_LEVEL_9: usize = 10;
    pub const OFI_LEVEL_10: usize = 11;
}

/// Multi-level OFI feature computer.
///
/// Must be updated on every LOB state transition via `update()`.
/// Extraction produces exactly `FEATURE_COUNT` (12) values.
#[derive(Debug, Clone)]
pub struct MlofiComputer {
    tracker: MultiLevelOfiTracker,
}

impl MlofiComputer {
    pub fn new() -> Self {
        Self {
            tracker: MultiLevelOfiTracker::new(MLOFI_LEVELS),
        }
    }

    /// Update on every LOB state transition.
    #[inline]
    pub fn update(&mut self, lob: &LobState) {
        self.tracker.update(lob);
    }

    /// Extract MLOFI features into the output buffer.
    ///
    /// Pushes exactly `FEATURE_COUNT` values:
    /// [total_mlofi, weighted_mlofi, ofi_level_1, ..., ofi_level_10]
    pub fn extract_into(&self, output: &mut Vec<f64>) {
        output.push(self.tracker.get_mlofi());
        output.push(self.tracker.get_weighted_mlofi());

        let per_level = self.tracker.get_ofi_per_level();
        for i in 0..MLOFI_LEVELS {
            output.push(if i < per_level.len() { per_level[i] } else { 0.0 });
        }
    }

    /// Full reset: clear all state including previous book snapshot.
    /// Used between trading days.
    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    /// Extract current MLOFI features AND reset accumulators for next sample period.
    ///
    /// Call this at each sample point instead of `extract_into()` to get
    /// interval-scoped OFI (the sum of LOB transitions since the last sample),
    /// rather than cumulative OFI since day start.
    ///
    /// Follows the same pattern as `OfiComputer::sample_and_reset()`.
    pub fn sample_and_reset(&mut self, output: &mut Vec<f64>) {
        self.extract_into(output);
        self.tracker.sample_and_reset_accumulators();
    }

    /// Returns true if the tracker has received at least 2 LOB updates
    /// (minimum for meaningful OFI deltas).
    pub fn is_warm(&self) -> bool {
        self.tracker.update_count() >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count_matches_extraction() {
        let computer = MlofiComputer::new();
        let mut output = Vec::new();
        computer.extract_into(&mut output);
        assert_eq!(
            output.len(),
            FEATURE_COUNT,
            "extract_into produced {} features, expected {}",
            output.len(),
            FEATURE_COUNT
        );
    }

    #[test]
    fn test_initial_state_is_zero() {
        let computer = MlofiComputer::new();
        let mut output = Vec::new();
        computer.extract_into(&mut output);
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 0.0, "Feature {} should be 0.0 initially, got {}", i, val);
        }
    }

    #[test]
    fn test_warmup_requires_two_updates() {
        let mut computer = MlofiComputer::new();
        assert!(!computer.is_warm());

        let mut lob = LobState::new(10);
        lob.bid_prices[0] = 100_000_000_000;
        lob.ask_prices[0] = 100_010_000_000;
        lob.bid_sizes[0] = 100;
        lob.ask_sizes[0] = 100;
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);

        computer.update(&lob);
        assert!(!computer.is_warm());

        lob.bid_sizes[0] = 150;
        computer.update(&lob);
        assert!(computer.is_warm());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut computer = MlofiComputer::new();

        let mut lob = LobState::new(10);
        lob.bid_prices[0] = 100_000_000_000;
        lob.ask_prices[0] = 100_010_000_000;
        lob.bid_sizes[0] = 100;
        lob.ask_sizes[0] = 100;
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);
        computer.update(&lob);
        lob.bid_sizes[0] = 200;
        computer.update(&lob);

        computer.reset();
        assert!(!computer.is_warm());

        let mut output = Vec::new();
        computer.extract_into(&mut output);
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_bid_size_increase_produces_positive_ofi() {
        let mut computer = MlofiComputer::new();

        let mut lob = LobState::new(10);
        lob.bid_prices[0] = 100_000_000_000;
        lob.ask_prices[0] = 100_010_000_000;
        lob.bid_sizes[0] = 100;
        lob.ask_sizes[0] = 100;
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);

        computer.update(&lob);

        lob.bid_sizes[0] = 200;
        computer.update(&lob);

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        assert!(
            output[indices::TOTAL_MLOFI] > 0.0,
            "Bid size increase should produce positive total MLOFI, got {}",
            output[indices::TOTAL_MLOFI]
        );
        assert!(
            output[indices::WEIGHTED_MLOFI] > 0.0,
            "Bid size increase should produce positive weighted MLOFI, got {}",
            output[indices::WEIGHTED_MLOFI]
        );
        assert!(
            output[indices::OFI_LEVEL_1] > 0.0,
            "Bid size increase at L1 should produce positive L1 OFI, got {}",
            output[indices::OFI_LEVEL_1]
        );
    }

    #[test]
    fn test_indices_are_contiguous() {
        assert_eq!(indices::TOTAL_MLOFI, 0);
        assert_eq!(indices::WEIGHTED_MLOFI, 1);
        assert_eq!(indices::OFI_LEVEL_1, 2);
        assert_eq!(indices::OFI_LEVEL_10, 11);
        assert_eq!(indices::OFI_LEVEL_10 + 1, FEATURE_COUNT);
    }

    #[test]
    fn test_sample_and_reset_interval_scoped() {
        let mut computer = MlofiComputer::new();

        let mut lob = LobState::new(10);
        lob.bid_prices[0] = 100_000_000_000;
        lob.ask_prices[0] = 100_010_000_000;
        lob.bid_sizes[0] = 100;
        lob.ask_sizes[0] = 100;
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);

        // Interval 1: init + bid increase
        computer.update(&lob);
        lob.bid_sizes[0] = 200;
        computer.update(&lob);

        let mut interval_1 = Vec::new();
        computer.sample_and_reset(&mut interval_1);
        assert_eq!(interval_1.len(), FEATURE_COUNT);
        let mlofi_1 = interval_1[indices::TOTAL_MLOFI];
        assert!(mlofi_1 > 0.0, "Interval 1 MLOFI should be positive (bid increase)");

        // Interval 2: no change in sizes → OFI should be 0
        computer.update(&lob); // same lob state
        let mut interval_2 = Vec::new();
        computer.sample_and_reset(&mut interval_2);
        assert_eq!(
            interval_2[indices::TOTAL_MLOFI], 0.0,
            "Interval 2 MLOFI should be 0 (no size change). Got {}. \
             Without sample_and_reset, this would be {} (cumulative).",
            interval_2[indices::TOTAL_MLOFI], mlofi_1
        );

        // Interval 3: ask decrease → bullish (less selling pressure)
        // Tracker convention: ask_ofi = prev_size - curr_size (positive when ask shrinks)
        lob.ask_sizes[0] = 50;
        computer.update(&lob);
        let mut interval_3 = Vec::new();
        computer.sample_and_reset(&mut interval_3);
        assert!(
            interval_3[indices::TOTAL_MLOFI] > 0.0,
            "Interval 3 MLOFI should be positive (ask decrease = less supply = bullish), got {}",
            interval_3[indices::TOTAL_MLOFI]
        );
    }

    #[test]
    fn test_mlofi_per_level_isolation() {
        // Changing L1 only should NOT affect L5 OFI.
        let mut computer = MlofiComputer::new();

        // Create LOB with 5 bid/ask levels populated
        let mut lob = LobState::new(10);
        for i in 0..5 {
            lob.bid_prices[i] = 100_000_000_000 - (i as i64 * 10_000_000); // $100.00, $99.99, ...
            lob.bid_sizes[i] = 100;
            lob.ask_prices[i] = 100_010_000_000 + (i as i64 * 10_000_000); // $100.01, $100.02, ...
            lob.ask_sizes[i] = 100;
        }
        lob.best_bid = Some(100_000_000_000);
        lob.best_ask = Some(100_010_000_000);

        // First update: initialize
        computer.update(&lob);

        // Second update: change ONLY L1 bid size
        lob.bid_sizes[0] = 200; // L1 bid +100
        // L2-L5 unchanged
        computer.update(&lob);

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        // L1 OFI should be positive (bid increase)
        assert!(
            output[indices::OFI_LEVEL_1] > 0.0,
            "L1 OFI should be positive after L1 bid increase. Got {}",
            output[indices::OFI_LEVEL_1]
        );

        // L5 OFI should be zero (no change at level 5)
        assert_eq!(
            output[indices::OFI_LEVEL_5], 0.0,
            "L5 OFI should be 0 when only L1 changed. Got {} (level isolation violated)",
            output[indices::OFI_LEVEL_5]
        );

        // Total MLOFI should equal L1 contribution only
        let total = output[indices::TOTAL_MLOFI];
        let l1 = output[indices::OFI_LEVEL_1];
        assert!(
            (total - l1).abs() < 1e-10,
            "total_mlofi ({}) should equal L1 OFI ({}) when only L1 changed",
            total, l1
        );
    }
}
