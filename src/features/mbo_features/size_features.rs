//! Size distribution feature extraction (8 features, contract indices 60-67).
//!
//! Detects institutional order patterns through size percentiles,
//! z-scores, skewness, and concentration (HHI) metrics.

use super::window::MboWindow;
use crate::contract::{DIVISION_GUARD_EPS, FLOAT_CMP_EPS};

/// Extract 8 size distribution features from the medium window.
///
/// Layout:
/// - [0-3]: Size percentiles (p25, p50, p75, p90)
/// - [4]: Size z-score (last event vs window mean)
/// - [5]: Large order ratio (fraction above p90)
/// - [6]: Size skewness (third standardized moment)
/// - [7]: Size concentration (HHI of order sizes)
pub(super) fn extract(window: &mut MboWindow) -> [f64; 8] {
    let p25 = window.size_percentile(0);
    let p50 = window.size_percentile(1);
    let p75 = window.size_percentile(2);
    let p90 = window.size_percentile(3);

    [
        p25,
        p50,
        p75,
        p90,
        size_zscore(window),
        large_order_ratio(window),
        size_skewness(window),
        size_concentration(window),
    ]
}

fn size_zscore(w: &mut MboWindow) -> f64 {
    if w.is_empty() {
        return 0.0;
    }
    let last_size = w.events.back().map(|e| e.size as f64).unwrap_or(0.0);
    let mean = w.size_mean();
    let std = w.size_std();
    (last_size - mean) / (std + DIVISION_GUARD_EPS)
}

fn large_order_ratio(w: &mut MboWindow) -> f64 {
    let threshold = w.size_percentile(3); // p90
    let large_count = w
        .events
        .iter()
        .filter(|e| e.size as f64 > threshold)
        .count();
    large_count as f64 / w.len().max(1) as f64
}

/// Skewness = E[(X - μ)³] / σ³
///
/// Positive: Right-skewed (institutional order splitting pattern)
/// Negative: Left-skewed
/// Returns 0.0 if insufficient data (< 3 samples) or zero variance.
fn size_skewness(w: &mut MboWindow) -> f64 {
    if w.events.len() < 3 {
        return 0.0;
    }

    let mean = w.size_mean();
    let std = w.size_std();

    if std < FLOAT_CMP_EPS {
        return 0.0;
    }

    let n = w.events.len() as f64;
    w.events
        .iter()
        .map(|e| {
            let z = (e.size as f64 - mean) / std;
            z * z * z
        })
        .sum::<f64>()
        / n
}

/// HHI of order size distribution.
///
/// HHI = Σ(share_i)² where share_i = size_i / total_size
/// Range: [1/N, 1.0]. Low = even sizes, High = concentrated.
fn size_concentration(w: &MboWindow) -> f64 {
    if w.events.is_empty() {
        return 0.0;
    }

    let total_size: u64 = w.events.iter().map(|e| e.size as u64).sum();
    if total_size == 0 {
        return 0.0;
    }

    let total_f = total_size as f64;
    w.events
        .iter()
        .map(|e| {
            let share = e.size as f64 / total_f;
            share * share
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::super::event::MboEvent;
    use super::super::window::MboWindow;
    use super::*;
    use mbo_lob_reconstructor::{Action, Side};

    fn push_events(w: &mut MboWindow, sizes: &[u32]) {
        for (i, &size) in sizes.iter().enumerate() {
            let event = MboEvent::new(
                i as u64 * 1_000_000,
                Action::Add,
                Side::Bid,
                100_000_000_000,
                size,
                i as u64,
            );
            w.push(event);
        }
    }

    #[test]
    fn test_size_skewness_insufficient_data() {
        let mut w = MboWindow::new(1000);
        push_events(&mut w, &[100, 200]);
        assert_eq!(size_skewness(&mut w), 0.0);
    }

    #[test]
    fn test_size_skewness_right_skewed() {
        let mut w = MboWindow::new(1000);
        push_events(&mut w, &[10, 10, 10, 10, 100]);
        let skew = size_skewness(&mut w);
        assert!(skew > 0.5, "Right-skewed should be positive, got {}", skew);
    }

    #[test]
    fn test_size_skewness_left_skewed() {
        let mut w = MboWindow::new(1000);
        push_events(&mut w, &[100, 100, 100, 100, 10]);
        let skew = size_skewness(&mut w);
        assert!(skew < -0.5, "Left-skewed should be negative, got {}", skew);
    }

    #[test]
    fn test_size_concentration_empty() {
        let w = MboWindow::new(1000);
        assert_eq!(size_concentration(&w), 0.0);
    }

    #[test]
    fn test_size_concentration_single_order() {
        let mut w = MboWindow::new(1000);
        push_events(&mut w, &[100]);
        assert!((size_concentration(&w) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_size_concentration_equal_sizes() {
        let mut w = MboWindow::new(1000);
        push_events(&mut w, &[100, 100, 100, 100]);
        assert!((size_concentration(&w) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_size_concentration_concentrated() {
        let mut w = MboWindow::new(1000);
        push_events(&mut w, &[900, 100]);
        let expected = 0.9f64 * 0.9 + 0.1 * 0.1; // 0.82
        assert!((size_concentration(&w) - expected).abs() < 1e-10);
    }
}
