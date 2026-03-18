//! Per-Level Order Flow (Kolm et al. 2023)
//!
//! Exposes 20-dimensional Order Flow with bid and ask kept separate per level,
//! matching the representation from Kolm, Turiel & Westray (2023).
//!
//! # Reference
//!
//! Kolm, Turiel & Westray (2023), "Deep Order Flow Imbalance: Extracting Alpha
//! at Multiple Horizons from the Limit Order Book", Mathematical Finance.
//! - Section 2.1.2: OF = [bOF; aOF] in R^20 (10 bid + 10 ask, kept separate)
//! - OF significantly outperforms OFI (combined bid-ask) across 115 NASDAQ stocks
//! - Keeping bid/ask separate lets the model learn asymmetric effects
//!
//! # Difference from MLOFI
//!
//! MLOFI (this codebase, indices 116-127) combines bid+ask per level into 10-dim OFI.
//! This module keeps them separate as 20-dim OF, matching Kolm's finding that the
//! 20-dim representation outperforms the 10-dim combined form.
//!
//! # Sign Conventions (Kolm, Section 2.1.2)
//!
//! **Bid OF (bOF):**
//! - Price improved (up): `+curr_bid_size` (new queue, bullish)
//! - Price unchanged: `curr_size - prev_size` (size change)
//! - Price worsened (down): `-prev_bid_size` (lost queue, bearish)
//!
//! **Ask OF (aOF):**
//! - Price improved (down): `+curr_ask_size` (new supply, bearish pressure absorbed)
//! - Price unchanged: `curr_size - prev_size` (size change)
//! - Price worsened (up): `-prev_ask_size` (lost supply)
//!
//! Note: Kolm's aOF is POSITIVE when ask improves (price drops). Our internal
//! `MultiLevelOfiTracker` uses the OPPOSITE convention for asks (negative when
//! ask improves). We negate the tracker's ask values to match Kolm.
//!
//! # Feature Layout (20 features)
//!
//! | Offset | Name | Description | Sign Convention |
//! |--------|------|-------------|-----------------|
//! | 0-9 | bof_level_1..10 | Bid order flow per level | >0 bid pressure |
//! | 10-19 | aof_level_1..10 | Ask order flow per level | >0 ask improved |

use mbo_lob_reconstructor::LobState;

use crate::features::order_flow::MultiLevelOfiTracker;

/// Number of features in this group.
pub const FEATURE_COUNT: usize = 20;

/// Number of LOB levels tracked.
pub const KOLM_OF_LEVELS: usize = 10;

/// Feature indices (relative to group start).
pub mod indices {
    pub const BOF_LEVEL_1: usize = 0;
    pub const BOF_LEVEL_2: usize = 1;
    pub const BOF_LEVEL_3: usize = 2;
    pub const BOF_LEVEL_4: usize = 3;
    pub const BOF_LEVEL_5: usize = 4;
    pub const BOF_LEVEL_6: usize = 5;
    pub const BOF_LEVEL_7: usize = 6;
    pub const BOF_LEVEL_8: usize = 7;
    pub const BOF_LEVEL_9: usize = 8;
    pub const BOF_LEVEL_10: usize = 9;
    pub const AOF_LEVEL_1: usize = 10;
    pub const AOF_LEVEL_2: usize = 11;
    pub const AOF_LEVEL_3: usize = 12;
    pub const AOF_LEVEL_4: usize = 13;
    pub const AOF_LEVEL_5: usize = 14;
    pub const AOF_LEVEL_6: usize = 15;
    pub const AOF_LEVEL_7: usize = 16;
    pub const AOF_LEVEL_8: usize = 17;
    pub const AOF_LEVEL_9: usize = 18;
    pub const AOF_LEVEL_10: usize = 19;
}

/// Per-level Order Flow computer (Kolm et al. 2023).
///
/// Must be updated on every LOB state transition via `update()`.
/// Extraction produces exactly `FEATURE_COUNT` (20) values:
/// 10 bid OF + 10 ask OF, with Kolm sign conventions.
#[derive(Debug, Clone)]
pub struct KolmOfComputer {
    tracker: MultiLevelOfiTracker,
}

impl KolmOfComputer {
    pub fn new() -> Self {
        Self {
            tracker: MultiLevelOfiTracker::new(KOLM_OF_LEVELS),
        }
    }

    /// Update on every LOB state transition.
    #[inline]
    pub fn update(&mut self, lob: &LobState) {
        self.tracker.update(lob);
    }

    /// Extract 20-dim OF features into the output buffer.
    ///
    /// Layout: [bof_level_1, ..., bof_level_10, aof_level_1, ..., aof_level_10]
    ///
    /// Ask values are negated relative to the internal tracker to match
    /// Kolm's convention where aOF is positive when the ask improves.
    pub fn extract_into(&self, output: &mut Vec<f64>) {
        let bid_of = self.tracker.get_ofi_bid_per_level();
        let ask_of = self.tracker.get_ofi_ask_per_level();

        for i in 0..KOLM_OF_LEVELS {
            output.push(if i < bid_of.len() { bid_of[i] } else { 0.0 });
        }

        for i in 0..KOLM_OF_LEVELS {
            // Negate: tracker's ask convention is opposite to Kolm's
            output.push(if i < ask_of.len() { -ask_of[i] } else { 0.0 });
        }
    }

    /// Full reset: clear all state including previous book snapshot.
    /// Used between trading days.
    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    /// Extract current Kolm OF features AND reset accumulators for next sample period.
    ///
    /// Call this at each sample point instead of `extract_into()` to get
    /// interval-scoped OF (the sum of LOB transitions since the last sample),
    /// rather than cumulative OF since day start.
    ///
    /// Follows the same pattern as `OfiComputer::sample_and_reset()`.
    pub fn sample_and_reset(&mut self, output: &mut Vec<f64>) {
        self.extract_into(output);
        self.tracker.sample_and_reset_accumulators();
    }

    /// Returns true if the tracker has received at least 2 LOB updates.
    pub fn is_warm(&self) -> bool {
        self.tracker.update_count() >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lob(
        bid_price: i64,
        ask_price: i64,
        bid_size: u32,
        ask_size: u32,
    ) -> LobState {
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
    fn test_feature_count_matches_extraction() {
        let computer = KolmOfComputer::new();
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
        let computer = KolmOfComputer::new();
        let mut output = Vec::new();
        computer.extract_into(&mut output);
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 0.0, "Feature {} should be 0.0 initially, got {}", i, val);
        }
    }

    #[test]
    fn test_bid_size_increase_produces_positive_bof() {
        let mut computer = KolmOfComputer::new();

        let lob1 = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000_000, 100_010_000_000, 200, 100);
        computer.update(&lob2);

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        assert!(
            output[indices::BOF_LEVEL_1] > 0.0,
            "Bid size increase should produce positive bof_level_1, got {}",
            output[indices::BOF_LEVEL_1]
        );
        assert_eq!(
            output[indices::BOF_LEVEL_1], 100.0,
            "Bid size increase of 100 should give bof=100"
        );
    }

    #[test]
    fn test_ask_size_increase_produces_positive_aof() {
        let mut computer = KolmOfComputer::new();

        let lob1 = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000_000, 100_010_000_000, 100, 200);
        computer.update(&lob2);

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        // Kolm convention: ask size increase at same price = positive aOF
        // Internal tracker: ask size change at same price = -size_change (negative)
        // We negate → positive
        assert!(
            output[indices::AOF_LEVEL_1] > 0.0,
            "Ask size increase should produce positive aof_level_1 (Kolm convention), got {}",
            output[indices::AOF_LEVEL_1]
        );
    }

    #[test]
    fn test_bid_ask_separation() {
        let mut computer = KolmOfComputer::new();

        let lob1 = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        computer.update(&lob1);

        // Only bid changes
        let lob2 = make_lob(100_000_000_000, 100_010_000_000, 200, 100);
        computer.update(&lob2);

        let mut output = Vec::new();
        computer.extract_into(&mut output);

        assert!(output[indices::BOF_LEVEL_1] != 0.0, "Bid OF should be non-zero");
        assert_eq!(output[indices::AOF_LEVEL_1], 0.0, "Ask OF should be zero when only bid changed");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut computer = KolmOfComputer::new();

        let lob1 = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        computer.update(&lob1);
        let lob2 = make_lob(100_000_000_000, 100_010_000_000, 200, 200);
        computer.update(&lob2);

        computer.reset();
        assert!(!computer.is_warm());

        let mut output = Vec::new();
        computer.extract_into(&mut output);
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_warmup_requires_two_updates() {
        let mut computer = KolmOfComputer::new();
        assert!(!computer.is_warm());

        let lob = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        computer.update(&lob);
        assert!(!computer.is_warm());

        let lob2 = make_lob(100_000_000_000, 100_010_000_000, 150, 100);
        computer.update(&lob2);
        assert!(computer.is_warm());
    }

    #[test]
    fn test_indices_are_contiguous() {
        assert_eq!(indices::BOF_LEVEL_1, 0);
        assert_eq!(indices::BOF_LEVEL_10, 9);
        assert_eq!(indices::AOF_LEVEL_1, 10);
        assert_eq!(indices::AOF_LEVEL_10, 19);
        assert_eq!(indices::AOF_LEVEL_10 + 1, FEATURE_COUNT);
    }

    #[test]
    fn test_sample_and_reset_interval_scoped() {
        let mut computer = KolmOfComputer::new();

        let lob1 = make_lob(100_000_000_000, 100_010_000_000, 100, 100);
        computer.update(&lob1);

        // Interval 1: bid increase
        let lob2 = make_lob(100_000_000_000, 100_010_000_000, 200, 100);
        computer.update(&lob2);

        let mut interval_1 = Vec::new();
        computer.sample_and_reset(&mut interval_1);
        assert_eq!(interval_1.len(), FEATURE_COUNT);
        let bof1 = interval_1[indices::BOF_LEVEL_1];
        assert!(bof1 > 0.0, "Interval 1 bof_level_1 should be positive (bid increase)");

        // Interval 2: no change → all zeros
        computer.update(&lob2); // same state
        let mut interval_2 = Vec::new();
        computer.sample_and_reset(&mut interval_2);
        assert_eq!(
            interval_2[indices::BOF_LEVEL_1], 0.0,
            "Interval 2 bof should be 0 (no size change). Got {}. \
             Without sample_and_reset, this would be {} (cumulative).",
            interval_2[indices::BOF_LEVEL_1], bof1
        );
        assert_eq!(
            interval_2[indices::AOF_LEVEL_1], 0.0,
            "Interval 2 aof should also be 0"
        );
    }
}
