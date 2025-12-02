//! FI-2010 Benchmark Feature Extraction
//!
//! Implements the 104 handcrafted features from the FI-2010 benchmark paper:
//! "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data"
//! (Ntakaris et al., 2018)
//!
//! # Feature Categories
//!
//! 1. **Time-Insensitive (20 features)**: Spread, mid-price, price/volume differences
//! 2. **Time-Sensitive (20 features)**: Derivatives, intensity measures
//! 3. **Depth Features (40 features)**: Accumulated volumes and price differences
//! 4. **Regression Features (24 features)**: Linear regression coefficients (optional)
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::features::fi2010::{FI2010Extractor, FI2010Config};
//!
//! let config = FI2010Config::default();
//! let mut extractor = FI2010Extractor::new(config);
//!
//! // Extract features from LOB state
//! let features = extractor.extract(&lob_state)?;
//! ```

use mbo_lob_reconstructor::{LobState, Result, TlobError};

/// Configuration for FI-2010 feature extraction.
#[derive(Debug, Clone)]
pub struct FI2010Config {
    /// Number of LOB levels to use (default: 10)
    pub levels: usize,

    /// Include time-insensitive features (20 features)
    pub include_time_insensitive: bool,

    /// Include time-sensitive features (20 features)
    pub include_time_sensitive: bool,

    /// Include depth features (40 features)
    pub include_depth: bool,

    /// Time delta for derivative computation (in nanoseconds)
    pub time_delta_ns: u64,
}

impl Default for FI2010Config {
    fn default() -> Self {
        Self {
            levels: 10,
            include_time_insensitive: true,
            include_time_sensitive: true,
            include_depth: true,
            time_delta_ns: 1_000_000, // 1ms
        }
    }
}

/// FI-2010 feature extractor.
///
/// Extracts handcrafted features as defined in the FI-2010 benchmark paper.
/// Maintains state for time-sensitive feature computation.
pub struct FI2010Extractor {
    config: FI2010Config,

    // Previous state for time-sensitive features
    prev_mid_price: Option<f64>,
    prev_spread: Option<f64>,
    prev_relative_spread: Option<f64>,
    prev_volume_ratio: Option<f64>,
    prev_ask_prices: Vec<f64>,
    prev_timestamp: Option<u64>,
}

impl FI2010Extractor {
    /// Create a new FI-2010 extractor.
    pub fn new(config: FI2010Config) -> Self {
        Self {
            prev_ask_prices: vec![0.0; config.levels],
            config,
            prev_mid_price: None,
            prev_spread: None,
            prev_relative_spread: None,
            prev_volume_ratio: None,
            prev_timestamp: None,
        }
    }

    /// Get the total number of features that will be extracted.
    pub fn feature_count(&self) -> usize {
        let mut count = 0;
        if self.config.include_time_insensitive {
            count += 20;
        }
        if self.config.include_time_sensitive {
            count += 20;
        }
        if self.config.include_depth {
            count += 40;
        }
        count
    }

    /// Extract all configured FI-2010 features.
    pub fn extract(&mut self, lob_state: &LobState, timestamp: u64) -> Result<Vec<f64>> {
        let mut features = Vec::with_capacity(self.feature_count());

        if self.config.include_time_insensitive {
            let ti_features = self.extract_time_insensitive(lob_state)?;
            features.extend_from_slice(&ti_features);
        }

        if self.config.include_time_sensitive {
            let ts_features = self.extract_time_sensitive(lob_state, timestamp)?;
            features.extend_from_slice(&ts_features);
        }

        if self.config.include_depth {
            let depth_features = self.extract_depth(lob_state);
            features.extend_from_slice(&depth_features);
        }

        // Update previous state
        self.update_state(lob_state, timestamp);

        Ok(features)
    }

    /// Extract time-insensitive features (20 features).
    ///
    /// Features:
    /// - 1: spread
    /// - 2: mid_price
    /// - 3-11: ask_price_diff_1 to ask_price_diff_9 (level i - level i+1)
    /// - 12-20: bid_price_diff_1 to bid_price_diff_9 (level i - level i+1)
    pub fn extract_time_insensitive(&self, lob_state: &LobState) -> Result<[f64; 20]> {
        let levels = self.config.levels.min(lob_state.levels);

        let best_bid = lob_state
            .best_bid
            .ok_or_else(|| TlobError::generic("No best bid for FI-2010 features"))?;
        let best_ask = lob_state
            .best_ask
            .ok_or_else(|| TlobError::generic("No best ask for FI-2010 features"))?;

        let best_bid_f64 = best_bid as f64 / 1e9;
        let best_ask_f64 = best_ask as f64 / 1e9;

        let mut features = [0.0f64; 20];

        // 1. Spread
        features[0] = best_ask_f64 - best_bid_f64;

        // 2. Mid-price
        features[1] = (best_ask_f64 + best_bid_f64) / 2.0;

        // 3-11. Ask price differences (level i - level i+1)
        for i in 0..(levels - 1).min(9) {
            let price_i = lob_state.ask_prices[i] as f64 / 1e9;
            let price_next = lob_state.ask_prices[i + 1] as f64 / 1e9;
            features[2 + i] = price_next - price_i; // Higher level has higher price
        }

        // 12-20. Bid price differences (level i - level i+1)
        for i in 0..(levels - 1).min(9) {
            let price_i = lob_state.bid_prices[i] as f64 / 1e9;
            let price_next = lob_state.bid_prices[i + 1] as f64 / 1e9;
            features[11 + i] = price_i - price_next; // Higher level has lower price
        }

        Ok(features)
    }

    /// Extract time-sensitive features (20 features).
    ///
    /// Features:
    /// 1. mid_price_derivative
    /// 2. spread_derivative
    /// 3. price_intensity
    /// - 4: volume_intensity
    /// - 5: bid_intensity
    /// - 6: ask_intensity
    /// - 7: relative_spread
    /// - 8: relative_spread_derivative
    /// - 9: bid_ask_volume_ratio
    /// - 10: volume_ratio_derivative
    /// - 11-20: ask_price_derivative_1 to ask_price_derivative_10
    pub fn extract_time_sensitive(
        &self,
        lob_state: &LobState,
        timestamp: u64,
    ) -> Result<[f64; 20]> {
        let levels = self.config.levels.min(lob_state.levels);

        let best_bid = lob_state
            .best_bid
            .ok_or_else(|| TlobError::generic("No best bid for FI-2010 features"))?;
        let best_ask = lob_state
            .best_ask
            .ok_or_else(|| TlobError::generic("No best ask for FI-2010 features"))?;

        let best_bid_f64 = best_bid as f64 / 1e9;
        let best_ask_f64 = best_ask as f64 / 1e9;
        let mid_price = (best_ask_f64 + best_bid_f64) / 2.0;
        let spread = best_ask_f64 - best_bid_f64;

        // Compute time delta in seconds
        let dt = if let Some(prev_ts) = self.prev_timestamp {
            let delta_ns = timestamp.saturating_sub(prev_ts);
            if delta_ns > 0 {
                delta_ns as f64 / 1e9
            } else {
                1.0 // Default to 1 second if no time difference
            }
        } else {
            1.0
        };

        let mut features = [0.0f64; 20];

        // 1. Mid-price derivative
        features[0] = if let Some(prev) = self.prev_mid_price {
            (mid_price - prev) / dt
        } else {
            0.0
        };

        // 2. Spread derivative
        features[1] = if let Some(prev) = self.prev_spread {
            (spread - prev) / dt
        } else {
            0.0
        };

        // 3. Price intensity (absolute mid-price change rate)
        features[2] = features[0].abs();

        // 4. Volume intensity
        let total_bid_vol: f64 = (0..levels).map(|i| lob_state.bid_sizes[i] as f64).sum();
        let total_ask_vol: f64 = (0..levels).map(|i| lob_state.ask_sizes[i] as f64).sum();
        features[3] = total_bid_vol + total_ask_vol;

        // 5. Bid intensity (bid volume)
        features[4] = total_bid_vol;

        // 6. Ask intensity (ask volume)
        features[5] = total_ask_vol;

        // 7. Relative spread
        let relative_spread = if mid_price > 0.0 {
            spread / mid_price
        } else {
            0.0
        };
        features[6] = relative_spread;

        // 8. Relative spread derivative
        features[7] = if let Some(prev) = self.prev_relative_spread {
            (relative_spread - prev) / dt
        } else {
            0.0
        };

        // 9. Bid/ask volume ratio
        let volume_ratio = if total_ask_vol > 0.0 {
            total_bid_vol / total_ask_vol
        } else {
            1.0
        };
        features[8] = volume_ratio;

        // 10. Volume ratio derivative
        features[9] = if let Some(prev) = self.prev_volume_ratio {
            (volume_ratio - prev) / dt
        } else {
            0.0
        };

        // 11-20. Ask price derivatives for each level
        for i in 0..levels.min(10) {
            let price = lob_state.ask_prices[i] as f64 / 1e9;
            features[10 + i] = if i < self.prev_ask_prices.len() && self.prev_ask_prices[i] > 0.0 {
                (price - self.prev_ask_prices[i]) / dt
            } else {
                0.0
            };
        }

        Ok(features)
    }

    /// Extract depth features (40 features).
    ///
    /// Features:
    /// 1-10. acc_bid_volume_1 to acc_bid_volume_10 (accumulated bid volume)
    /// 11-20. acc_ask_volume_1 to acc_ask_volume_10 (accumulated ask volume)
    /// 21-30. acc_bid_price_diff_1 to acc_bid_price_diff_10 (accumulated price diff from best)
    /// 31-40. acc_ask_price_diff_1 to acc_ask_price_diff_10 (accumulated price diff from best)
    pub fn extract_depth(&self, lob_state: &LobState) -> [f64; 40] {
        let levels = self.config.levels.min(lob_state.levels);

        let mut features = [0.0f64; 40];

        let best_bid_f64 = lob_state.best_bid.map(|p| p as f64 / 1e9).unwrap_or(0.0);
        let best_ask_f64 = lob_state.best_ask.map(|p| p as f64 / 1e9).unwrap_or(0.0);

        // Accumulated volumes and price differences
        let mut acc_bid_vol = 0.0;
        let mut acc_ask_vol = 0.0;
        let mut acc_bid_price_diff = 0.0;
        let mut acc_ask_price_diff = 0.0;

        for i in 0..levels.min(10) {
            // Accumulated bid volume
            acc_bid_vol += lob_state.bid_sizes[i] as f64;
            features[i] = acc_bid_vol;

            // Accumulated ask volume
            acc_ask_vol += lob_state.ask_sizes[i] as f64;
            features[10 + i] = acc_ask_vol;

            // Accumulated bid price difference from best
            let bid_price = lob_state.bid_prices[i] as f64 / 1e9;
            acc_bid_price_diff += best_bid_f64 - bid_price;
            features[20 + i] = acc_bid_price_diff;

            // Accumulated ask price difference from best
            let ask_price = lob_state.ask_prices[i] as f64 / 1e9;
            acc_ask_price_diff += ask_price - best_ask_f64;
            features[30 + i] = acc_ask_price_diff;
        }

        features
    }

    /// Update internal state for next extraction.
    fn update_state(&mut self, lob_state: &LobState, timestamp: u64) {
        let best_bid = lob_state.best_bid.map(|p| p as f64 / 1e9).unwrap_or(0.0);
        let best_ask = lob_state.best_ask.map(|p| p as f64 / 1e9).unwrap_or(0.0);

        let mid_price = (best_ask + best_bid) / 2.0;
        let spread = best_ask - best_bid;
        let relative_spread = if mid_price > 0.0 {
            spread / mid_price
        } else {
            0.0
        };

        let levels = self.config.levels.min(lob_state.levels);
        let total_bid_vol: f64 = (0..levels).map(|i| lob_state.bid_sizes[i] as f64).sum();
        let total_ask_vol: f64 = (0..levels).map(|i| lob_state.ask_sizes[i] as f64).sum();
        let volume_ratio = if total_ask_vol > 0.0 {
            total_bid_vol / total_ask_vol
        } else {
            1.0
        };

        self.prev_mid_price = Some(mid_price);
        self.prev_spread = Some(spread);
        self.prev_relative_spread = Some(relative_spread);
        self.prev_volume_ratio = Some(volume_ratio);
        self.prev_timestamp = Some(timestamp);

        // Update previous ask prices
        for i in 0..levels.min(self.prev_ask_prices.len()) {
            self.prev_ask_prices[i] = lob_state.ask_prices[i] as f64 / 1e9;
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.prev_mid_price = None;
        self.prev_spread = None;
        self.prev_relative_spread = None;
        self.prev_volume_ratio = None;
        self.prev_timestamp = None;
        self.prev_ask_prices.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lob_state() -> LobState {
        let mut state = LobState::new(10);

        // Set up 5 levels of bids and asks
        for i in 0..5 {
            // Asks: $100.01, $100.02, $100.03, $100.04, $100.05
            state.ask_prices[i] = 100_010_000_000 + i as i64 * 10_000_000;
            state.ask_sizes[i] = (100 + i * 10) as u32;

            // Bids: $100.00, $99.99, $99.98, $99.97, $99.96
            state.bid_prices[i] = 100_000_000_000 - i as i64 * 10_000_000;
            state.bid_sizes[i] = (100 + i * 10) as u32;
        }

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        state
    }

    #[test]
    fn test_fi2010_config_default() {
        let config = FI2010Config::default();
        assert_eq!(config.levels, 10);
        assert!(config.include_time_insensitive);
        assert!(config.include_time_sensitive);
        assert!(config.include_depth);
    }

    #[test]
    fn test_fi2010_feature_count() {
        let config = FI2010Config::default();
        let extractor = FI2010Extractor::new(config);
        assert_eq!(extractor.feature_count(), 80); // 20 + 20 + 40
    }

    #[test]
    fn test_time_insensitive_features() {
        let config = FI2010Config::default();
        let extractor = FI2010Extractor::new(config);
        let state = create_test_lob_state();

        let features = extractor.extract_time_insensitive(&state).unwrap();

        // Spread should be $0.01
        assert!((features[0] - 0.01).abs() < 1e-6);

        // Mid-price should be $100.005
        assert!((features[1] - 100.005).abs() < 1e-6);

        // Ask price differences should be $0.01 each
        for i in 0..4 {
            assert!(
                (features[2 + i] - 0.01).abs() < 1e-6,
                "Ask diff {} = {}, expected 0.01",
                i,
                features[2 + i]
            );
        }

        // Bid price differences should be $0.01 each
        for i in 0..4 {
            assert!(
                (features[11 + i] - 0.01).abs() < 1e-6,
                "Bid diff {} = {}, expected 0.01",
                i,
                features[11 + i]
            );
        }
    }

    #[test]
    fn test_time_sensitive_features() {
        let config = FI2010Config::default();
        let mut extractor = FI2010Extractor::new(config);
        let state = create_test_lob_state();

        // First extraction (no previous state)
        let features1 = extractor
            .extract_time_sensitive(&state, 1_000_000_000)
            .unwrap();

        // Derivatives should be 0 on first call
        assert_eq!(features1[0], 0.0); // mid_price_derivative
        assert_eq!(features1[1], 0.0); // spread_derivative

        // Update state
        extractor.update_state(&state, 1_000_000_000);

        // Create modified state
        let mut state2 = state.clone();
        state2.best_bid = Some(100_010_000_000); // Bid moved up by $0.01
        state2.bid_prices[0] = 100_010_000_000;

        // Second extraction (1 second later)
        let features2 = extractor
            .extract_time_sensitive(&state2, 2_000_000_000)
            .unwrap();

        // Mid-price derivative should be positive (price went up)
        assert!(
            features2[0] > 0.0,
            "Mid-price derivative should be positive"
        );
    }

    #[test]
    fn test_depth_features() {
        let config = FI2010Config::default();
        let extractor = FI2010Extractor::new(config);
        let state = create_test_lob_state();

        let features = extractor.extract_depth(&state);

        // Accumulated bid volumes: 100, 210, 330, 460, 600
        assert_eq!(features[0], 100.0);
        assert_eq!(features[1], 210.0);
        assert_eq!(features[2], 330.0);
        assert_eq!(features[3], 460.0);
        assert_eq!(features[4], 600.0);

        // Accumulated ask volumes should be the same
        assert_eq!(features[10], 100.0);
        assert_eq!(features[11], 210.0);

        // Accumulated bid price differences
        // Level 0: 0.0 (best bid - best bid)
        // Level 1: 0.01 (best bid - level 1 bid)
        // Level 2: 0.03 (accumulated)
        assert!((features[20] - 0.0).abs() < 1e-6);
        assert!((features[21] - 0.01).abs() < 1e-6);
        assert!((features[22] - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_full_extraction() {
        let config = FI2010Config::default();
        let mut extractor = FI2010Extractor::new(config);
        let state = create_test_lob_state();

        let features = extractor.extract(&state, 1_000_000_000).unwrap();

        assert_eq!(features.len(), 80);

        // All features should be finite
        for (i, &f) in features.iter().enumerate() {
            assert!(f.is_finite(), "Feature {i} is not finite: {f}");
        }
    }

    #[test]
    fn test_reset() {
        let config = FI2010Config::default();
        let mut extractor = FI2010Extractor::new(config);
        let state = create_test_lob_state();

        // Extract once to populate state
        let _ = extractor.extract(&state, 1_000_000_000).unwrap();

        // Reset
        extractor.reset();

        // After reset, derivatives should be 0 again
        let features = extractor
            .extract_time_sensitive(&state, 2_000_000_000)
            .unwrap();
        assert_eq!(features[0], 0.0);
    }

    #[test]
    fn test_partial_config() {
        let config = FI2010Config {
            levels: 10,
            include_time_insensitive: true,
            include_time_sensitive: false,
            include_depth: false,
            time_delta_ns: 1_000_000,
        };

        let mut extractor = FI2010Extractor::new(config);
        assert_eq!(extractor.feature_count(), 20);

        let state = create_test_lob_state();
        let features = extractor.extract(&state, 1_000_000_000).unwrap();
        assert_eq!(features.len(), 20);
    }
}
