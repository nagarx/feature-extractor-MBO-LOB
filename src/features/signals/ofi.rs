//! Streaming OFI computer per Cont et al. (2014).
//!
//! Accumulates Order Flow Imbalance on every LOB state transition and provides
//! sample-and-reset functionality for discrete sampling.

use mbo_lob_reconstructor::LobState;

use crate::contract::FLOAT_CMP_EPS;

/// Nanoseconds per second (for dt_seconds computation).
const NS_PER_SECOND: i64 = 1_000_000_000;

/// Minimum effective state changes before OFI is considered "warm".
/// This counts ACTUAL LOB state transitions, not raw messages.
pub const MIN_WARMUP_STATE_CHANGES: u64 = 100;

/// Sample returned at sampling points.
///
/// Contains OFI and metadata accumulated since last sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct OfiSample {
    /// Accumulated OFI since last sample (positive = buy pressure)
    pub ofi: f64,

    /// OFI from bid side only
    pub ofi_bid: f64,

    /// OFI from ask side only
    pub ofi_ask: f64,

    /// Average depth (bid + ask L0 size) over sampling interval
    pub avg_depth: f64,

    /// Number of state transitions in this interval
    pub event_count: u64,

    /// Whether MBO/OFI has sufficient warmup
    pub is_warm: bool,

    /// Seconds since last sample
    pub dt_seconds: f64,
}

impl OfiSample {
    /// Compute depth-normalized OFI.
    ///
    /// Per Cont et al. (2014): beta is proportional to 1/AD, so we normalize by average depth.
    #[inline]
    pub fn depth_norm_ofi(&self) -> f64 {
        if self.avg_depth > FLOAT_CMP_EPS {
            self.ofi / self.avg_depth
        } else {
            0.0
        }
    }
}

/// Streaming OFI computer per Cont et al. (2014).
///
/// Accumulates OFI on every LOB state transition and provides
/// sample-and-reset functionality for discrete sampling.
///
/// # Key Features
///
/// - **Streaming accumulation**: Updates on every state change
/// - **Warmup tracking**: Counts effective state changes for `mbo_ready`
/// - **Sample duration**: Tracks `dt_seconds` between samples
/// - **Clear handling**: Resets on `Action::Clear` events
///
/// # OFI Formula (Cont et al. 2014, Section 2.1)
///
/// For each state transition n:
/// ```text
/// e_n = bid_contribution - ask_contribution
///
/// where:
///   bid_contribution:
///     if curr_bid > prev_bid: +curr_bid_size  (price improved)
///     if curr_bid < prev_bid: -prev_bid_size  (price dropped)
///     else: curr_bid_size - prev_bid_size     (size change only)
///
///   ask_contribution (note: inverted sign):
///     if curr_ask < prev_ask: +curr_ask_size  (price improved)
///     if curr_ask > prev_ask: -prev_ask_size  (price dropped)
///     else: curr_ask_size - prev_ask_size     (size change only)
///
/// OFI = Sigma(e_n) over sampling interval
/// ```
#[derive(Debug, Clone)]
pub struct OfiComputer {
    prev_bid_price: Option<i64>,
    prev_bid_size: u32,
    prev_ask_price: Option<i64>,
    prev_ask_size: u32,

    cumulative_ofi_bid: f64,
    cumulative_ofi_ask: f64,
    depth_sum: f64,
    event_count: u64,

    // Warmup tracking (persists across samples, reset on Clear)
    state_changes_since_reset: u64,

    last_sample_timestamp_ns: Option<i64>,
}

impl Default for OfiComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl OfiComputer {
    /// Create a new OFI computer.
    pub fn new() -> Self {
        Self {
            prev_bid_price: None,
            prev_bid_size: 0,
            prev_ask_price: None,
            prev_ask_size: 0,
            cumulative_ofi_bid: 0.0,
            cumulative_ofi_ask: 0.0,
            depth_sum: 0.0,
            event_count: 0,
            state_changes_since_reset: 0,
            last_sample_timestamp_ns: None,
        }
    }

    /// Update OFI with a new LOB state.
    ///
    /// **CRITICAL**: Call this on EVERY LOB state transition, not just at sampling points.
    ///
    /// Only counts as a "state change" for warmup if L0 prices/sizes actually changed.
    #[inline]
    pub fn update(&mut self, lob: &LobState) {
        let curr_bid_price = lob.best_bid;
        let curr_ask_price = lob.best_ask;
        let curr_bid_size = lob.bid_sizes[0];
        let curr_ask_size = lob.ask_sizes[0];

        self.depth_sum += (curr_bid_size + curr_ask_size) as f64;
        self.event_count += 1;

        let (prev_bid, prev_ask) = match (self.prev_bid_price, self.prev_ask_price) {
            (Some(b), Some(a)) => (b, a),
            _ => {
                self.prev_bid_price = curr_bid_price;
                self.prev_bid_size = curr_bid_size;
                self.prev_ask_price = curr_ask_price;
                self.prev_ask_size = curr_ask_size;
                return;
            }
        };

        let (curr_bid, curr_ask) = match (curr_bid_price, curr_ask_price) {
            (Some(b), Some(a)) => (b, a),
            _ => {
                self.prev_bid_price = curr_bid_price;
                self.prev_bid_size = curr_bid_size;
                self.prev_ask_price = curr_ask_price;
                self.prev_ask_size = curr_ask_size;
                return;
            }
        };

        let bid_changed = curr_bid != prev_bid || curr_bid_size != self.prev_bid_size;
        let ask_changed = curr_ask != prev_ask || curr_ask_size != self.prev_ask_size;

        if bid_changed || ask_changed {
            self.state_changes_since_reset += 1;
        }

        // === BID SIDE CONTRIBUTION (demand) ===
        let ofi_bid = if curr_bid > prev_bid {
            curr_bid_size as f64
        } else if curr_bid < prev_bid {
            -(self.prev_bid_size as f64)
        } else {
            (curr_bid_size as i64 - self.prev_bid_size as i64) as f64
        };

        // === ASK SIDE CONTRIBUTION (supply) ===
        // Signs inverted: ask improvement means price DOWN
        let ofi_ask = if curr_ask < prev_ask {
            curr_ask_size as f64
        } else if curr_ask > prev_ask {
            -(self.prev_ask_size as f64)
        } else {
            (curr_ask_size as i64 - self.prev_ask_size as i64) as f64
        };

        self.cumulative_ofi_bid += ofi_bid;
        self.cumulative_ofi_ask -= ofi_ask;

        self.prev_bid_price = curr_bid_price;
        self.prev_bid_size = curr_bid_size;
        self.prev_ask_price = curr_ask_price;
        self.prev_ask_size = curr_ask_size;
    }

    /// Sample accumulated OFI and reset accumulators.
    ///
    /// Call this at each sampling point to get the OFI for the interval.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    ///
    /// # Returns
    ///
    /// An `OfiSample` containing OFI and metadata for the interval.
    #[inline]
    pub fn sample_and_reset(&mut self, timestamp_ns: i64) -> OfiSample {
        let dt_seconds = match self.last_sample_timestamp_ns {
            Some(prev_ts) => (timestamp_ns - prev_ts) as f64 / NS_PER_SECOND as f64,
            None => 0.0,
        };

        let avg_depth = if self.event_count > 0 {
            self.depth_sum / self.event_count as f64
        } else {
            0.0
        };

        let sample = OfiSample {
            ofi: self.cumulative_ofi_bid + self.cumulative_ofi_ask,
            ofi_bid: self.cumulative_ofi_bid,
            ofi_ask: self.cumulative_ofi_ask,
            avg_depth,
            event_count: self.event_count,
            is_warm: self.is_warm(),
            dt_seconds,
        };

        // Reset accumulators (but NOT warmup counter)
        self.cumulative_ofi_bid = 0.0;
        self.cumulative_ofi_ask = 0.0;
        self.depth_sum = 0.0;
        self.event_count = 0;
        self.last_sample_timestamp_ns = Some(timestamp_ns);

        sample
    }

    /// Reset on Clear action (day boundary, session reset).
    ///
    /// Resets ALL state including warmup counter.
    pub fn reset_on_clear(&mut self) {
        self.prev_bid_price = None;
        self.prev_bid_size = 0;
        self.prev_ask_price = None;
        self.prev_ask_size = 0;
        self.cumulative_ofi_bid = 0.0;
        self.cumulative_ofi_ask = 0.0;
        self.depth_sum = 0.0;
        self.event_count = 0;
        self.state_changes_since_reset = 0;
        self.last_sample_timestamp_ns = None;
    }

    /// Check if OFI has sufficient warmup.
    ///
    /// Returns `true` if at least `MIN_WARMUP_STATE_CHANGES` effective
    /// state changes have occurred since the last reset.
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.state_changes_since_reset >= MIN_WARMUP_STATE_CHANGES
    }

    /// Get count of state changes since last reset.
    #[inline]
    pub fn state_changes_since_reset(&self) -> u64 {
        self.state_changes_since_reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lob(bid_price: i64, bid_size: u32, ask_price: i64, ask_size: u32) -> LobState {
        let mut lob = LobState::new(10);
        lob.best_bid = Some(bid_price);
        lob.best_ask = Some(ask_price);
        lob.bid_sizes[0] = bid_size;
        lob.ask_sizes[0] = ask_size;
        lob
    }

    #[test]
    fn test_ofi_computer_new() {
        let computer = OfiComputer::new();
        assert!(!computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), 0);
    }

    #[test]
    fn test_ofi_computer_bid_size_increase() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 150, 100_010_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_bid, 50.0);
        assert_eq!(sample.ofi, 50.0);
        assert!(sample.ofi > 0.0, "Bid size increase should be positive OFI");
    }

    #[test]
    fn test_ofi_computer_bid_size_decrease() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 50, 100_010_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_bid, -50.0);
        assert!(sample.ofi < 0.0, "Bid size decrease should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_bid_price_improvement() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_020_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_010_000, 80, 100_020_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_bid, 80.0);
        assert!(sample.ofi > 0.0, "Bid price improvement should be positive OFI");
    }

    #[test]
    fn test_ofi_computer_bid_price_drop() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_010_000, 100, 100_020_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 120, 100_020_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_bid, -100.0);
        assert!(sample.ofi < 0.0, "Bid price drop should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_ask_size_increase() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 100, 100_010_000, 150);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_ask, -50.0);
        assert!(sample.ofi < 0.0, "Ask size increase should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_ask_price_improvement() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_020_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 100, 100_010_000, 80);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_ask, -80.0);
        assert!(sample.ofi < 0.0, "Ask price improvement should be negative OFI");
    }

    #[test]
    fn test_ofi_computer_warmup_tracking() {
        let mut computer = OfiComputer::new();

        assert!(!computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), 0);

        let lob_init = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob_init);
        assert_eq!(computer.state_changes_since_reset(), 0);

        for i in 1..=MIN_WARMUP_STATE_CHANGES {
            let size = (100 + i) as u32;
            let lob = make_lob(100_000_000, size, 100_010_000, 100);
            computer.update(&lob);
        }

        assert!(computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), MIN_WARMUP_STATE_CHANGES);
    }

    #[test]
    fn test_ofi_computer_warmup_persists_across_samples() {
        let mut computer = OfiComputer::new();

        let lob_init = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob_init);

        for i in 1..=MIN_WARMUP_STATE_CHANGES + 10 {
            let size = (100 + i) as u32;
            let lob = make_lob(100_000_000, size, 100_010_000, 100);
            computer.update(&lob);
        }

        assert!(computer.is_warm());
        let changes_before = computer.state_changes_since_reset();

        let _ = computer.sample_and_reset(1_000_000_000);

        assert!(computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), changes_before);
    }

    #[test]
    fn test_ofi_computer_reset_on_clear() {
        let mut computer = OfiComputer::new();

        let lob_init = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob_init);

        for i in 1..=MIN_WARMUP_STATE_CHANGES + 10 {
            let size = (100 + i) as u32;
            let lob = make_lob(100_000_000, size, 100_010_000, 100);
            computer.update(&lob);
        }

        assert!(computer.is_warm());

        computer.reset_on_clear();

        assert!(!computer.is_warm());
        assert_eq!(computer.state_changes_since_reset(), 0);
    }

    #[test]
    fn test_ofi_computer_dt_seconds() {
        let mut computer = OfiComputer::new();

        let lob = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob);

        let sample1 = computer.sample_and_reset(1_000_000_000);
        assert_eq!(sample1.dt_seconds, 0.0);

        computer.update(&lob);

        let sample2 = computer.sample_and_reset(2_000_000_000);
        assert_eq!(sample2.dt_seconds, 1.0);

        computer.update(&lob);
        let sample3 = computer.sample_and_reset(2_500_000_000);
        assert_eq!(sample3.dt_seconds, 0.5);
    }

    #[test]
    fn test_ofi_computer_avg_depth() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 200, 100_010_000, 300);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        // Average = (200 + 500) / 2 = 350
        assert_eq!(sample.avg_depth, 350.0);
    }

    #[test]
    fn test_ofi_computer_depth_norm_ofi() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 200, 100_010_000, 100);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        let depth_norm = sample.depth_norm_ofi();
        assert!((depth_norm - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_ofi_computer_no_change_no_warmup_increment() {
        let mut computer = OfiComputer::new();

        let lob = make_lob(100_000_000, 100, 100_010_000, 100);

        computer.update(&lob);
        let changes1 = computer.state_changes_since_reset();

        computer.update(&lob);
        let changes2 = computer.state_changes_since_reset();

        assert_eq!(changes2, changes1);
    }

    #[test]
    fn test_ofi_computer_balanced_flow() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 150, 100_010_000, 150);
        computer.update(&lob2);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_bid, 50.0);
        assert_eq!(sample.ofi_ask, -50.0);
        assert_eq!(sample.ofi, 0.0);
    }

    #[test]
    fn test_ofi_computer_accumulation_multiple_events() {
        let mut computer = OfiComputer::new();

        let lob1 = make_lob(100_000_000, 100, 100_010_000, 100);
        computer.update(&lob1);

        let lob2 = make_lob(100_000_000, 150, 100_010_000, 100);
        computer.update(&lob2);

        let lob3 = make_lob(100_000_000, 180, 100_010_000, 100);
        computer.update(&lob3);

        let lob4 = make_lob(100_000_000, 180, 100_010_000, 120);
        computer.update(&lob4);

        let sample = computer.sample_and_reset(1_000_000_000);

        assert_eq!(sample.ofi_bid, 80.0);
        assert_eq!(sample.ofi_ask, -20.0);
        assert_eq!(sample.ofi, 60.0);
        assert_eq!(sample.event_count, 4);
    }
}
