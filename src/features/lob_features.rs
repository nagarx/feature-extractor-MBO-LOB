//! LOB feature extraction.
//!
//! Extracts raw features from LOB snapshots according to the TLOB paper format:
//! L(t) = (P^ask(t), V^ask(t), P^bid(t), V^bid(t))
//!
//! Features are extracted in fixed-point format (prices) and converted to f64.

use mbo_lob_reconstructor::LobState;

/// Extract raw LOB features from a snapshot.
///
/// Extracts features in the following order:
/// 1. Ask prices (levels 1 to L)
/// 2. Ask sizes (levels 1 to L)
/// 3. Bid prices (levels 1 to L)
/// 4. Bid sizes (levels 1 to L)
///
/// # Arguments
///
/// * `lob_state` - The LOB state snapshot
/// * `levels` - Number of levels to extract (typically 10)
/// * `output` - Pre-allocated vector to append features to
///
/// # Performance
///
/// This function is optimized for speed:
/// - No allocations (writes to pre-allocated vector)
/// - Inline operations
/// - Direct memory access
#[inline]
pub fn extract_raw_features(lob_state: &LobState, levels: usize, output: &mut Vec<f64>) {
    // Ensure we don't exceed available levels
    let extract_levels = levels.min(lob_state.levels);

    // Reserve space for all features at once (avoid reallocations)
    output.reserve(extract_levels * 4);

    // Extract ask prices (convert from fixed-point i64 to f64 dollars)
    for i in 0..extract_levels {
        let price = if lob_state.ask_prices[i] > 0 {
            lob_state.ask_prices[i] as f64 / 1e9
        } else {
            0.0 // No ask at this level
        };
        output.push(price);
    }

    // Extract ask sizes
    for i in 0..extract_levels {
        output.push(lob_state.ask_sizes[i] as f64);
    }

    // Extract bid prices (convert from fixed-point i64 to f64 dollars)
    for i in 0..extract_levels {
        let price = if lob_state.bid_prices[i] > 0 {
            lob_state.bid_prices[i] as f64 / 1e9
        } else {
            0.0 // No bid at this level
        };
        output.push(price);
    }

    // Extract bid sizes
    for i in 0..extract_levels {
        output.push(lob_state.bid_sizes[i] as f64);
    }
}

/// Extract normalized LOB features (for multi-instrument training).
///
/// Normalizes prices relative to mid-price and sizes relative to total volume.
/// This is useful when training models on multiple instruments with different price scales.
///
/// # Arguments
///
/// * `lob_state` - The LOB state snapshot
/// * `levels` - Number of levels to extract
/// * `output` - Pre-allocated vector to append features to
#[inline]
pub fn extract_normalized_features(lob_state: &LobState, levels: usize, output: &mut Vec<f64>) {
    let extract_levels = levels.min(lob_state.levels);
    output.reserve(extract_levels * 4);

    // Calculate mid-price for normalization
    let mid_price = lob_state.mid_price().unwrap_or(0.0);
    if mid_price == 0.0 {
        // No valid mid-price, return zeros
        for _ in 0..(extract_levels * 4) {
            output.push(0.0);
        }
        return;
    }

    // Calculate total volume for normalization
    let total_volume: f64 = (0..extract_levels)
        .map(|i| (lob_state.ask_sizes[i] + lob_state.bid_sizes[i]) as f64)
        .sum();
    let total_volume = if total_volume > 0.0 {
        total_volume
    } else {
        1.0
    };

    // Normalize ask prices (as percentage deviation from mid-price)
    for i in 0..extract_levels {
        let price = if lob_state.ask_prices[i] > 0 {
            let price_f64 = lob_state.ask_prices[i] as f64 / 1e9;
            (price_f64 - mid_price) / mid_price
        } else {
            0.0
        };
        output.push(price);
    }

    // Normalize ask sizes (as proportion of total volume)
    for i in 0..extract_levels {
        output.push(lob_state.ask_sizes[i] as f64 / total_volume);
    }

    // Normalize bid prices (as percentage deviation from mid-price)
    for i in 0..extract_levels {
        let price = if lob_state.bid_prices[i] > 0 {
            let price_f64 = lob_state.bid_prices[i] as f64 / 1e9;
            (price_f64 - mid_price) / mid_price
        } else {
            0.0
        };
        output.push(price);
    }

    // Normalize bid sizes (as proportion of total volume)
    for i in 0..extract_levels {
        output.push(lob_state.bid_sizes[i] as f64 / total_volume);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lob_state() -> LobState {
        let mut state = LobState::new(10);

        // Set up a simple LOB
        // Best bid: $100.00
        state.bid_prices[0] = 100_000_000_000;
        state.bid_sizes[0] = 100;

        // Best ask: $100.01
        state.ask_prices[0] = 100_010_000_000;
        state.ask_sizes[0] = 200;

        // Second level bid: $99.99
        state.bid_prices[1] = 99_990_000_000;
        state.bid_sizes[1] = 150;

        // Second level ask: $100.02
        state.ask_prices[1] = 100_020_000_000;
        state.ask_sizes[1] = 180;

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        state
    }

    #[test]
    fn test_extract_raw_features() {
        let state = create_test_lob_state();
        let mut features = Vec::new();

        extract_raw_features(&state, 2, &mut features);

        // Should have 2 levels × 4 features = 8 features
        assert_eq!(features.len(), 8);

        // Check ask prices
        assert!((features[0] - 100.01).abs() < 1e-6);
        assert!((features[1] - 100.02).abs() < 1e-6);

        // Check ask sizes
        assert_eq!(features[2], 200.0);
        assert_eq!(features[3], 180.0);

        // Check bid prices
        assert!((features[4] - 100.00).abs() < 1e-6);
        assert!((features[5] - 99.99).abs() < 1e-6);

        // Check bid sizes
        assert_eq!(features[6], 100.0);
        assert_eq!(features[7], 150.0);
    }

    #[test]
    fn test_extract_normalized_features() {
        let state = create_test_lob_state();
        let mut features = Vec::new();

        extract_normalized_features(&state, 2, &mut features);

        // Should have 2 levels × 4 features = 8 features
        assert_eq!(features.len(), 8);

        // Check normalized ask prices (should be small positive values)
        // Mid-price should be $100.005, so asks are slightly above
        assert!(features[0] > 0.0 && features[0] < 0.01); // ~0.005%
        assert!(features[1] > 0.0 && features[1] < 0.01); // ~0.015%

        // Check normalized sizes (should sum to approximately 1.0)
        let total_size_fraction: f64 = features[2] + features[3] + features[6] + features[7];
        assert!((total_size_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_lob() {
        let state = LobState::new(10);
        let mut features = Vec::new();

        extract_raw_features(&state, 10, &mut features);

        // Should have 10 levels × 4 = 40 features, all zeros
        assert_eq!(features.len(), 40);
        assert!(features.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn test_performance_no_allocation() {
        let state = create_test_lob_state();
        let mut features = Vec::with_capacity(40);

        // Extract features multiple times
        for _ in 0..1000 {
            features.clear();
            extract_raw_features(&state, 10, &mut features);
        }

        // Should not have reallocated
        assert_eq!(features.capacity(), 40);
    }
}
