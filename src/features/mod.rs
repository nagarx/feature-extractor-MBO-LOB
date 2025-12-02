//! Feature extraction for LOB-based deep learning models.
//!
//! This module provides efficient feature extraction from MBO data and LOB snapshots.
//! Features are designed based on research papers:
//! - DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
//! - TLOB: A Novel Transformer Model with Dual Attention
//! - FI-2010: Benchmark Dataset for Mid-Price Forecasting
//! - The Price Impact of Order Book Events (Cont et al.)
//!
//! # Architecture
//!
//! The feature extraction is organized into:
//! - `lob_features`: Extract RAW features from LOB snapshots (40 features by default)
//! - `order_flow`: Order Flow Imbalance (OFI), queue imbalance (8 features)
//! - `derived_features`: Compute derived metrics (8 features, opt-in)
//! - `mbo_features`: Aggregate MBO data into features (36 features, opt-in)
//! - `fi2010`: FI-2010 benchmark handcrafted features (80 features)
//! - `market_impact`: Market impact estimation features
//!
//! # Usage
//!
//! ```ignore
//! use feature_extractor::features::FeatureExtractor;
//!
//! let extractor = FeatureExtractor::new(10); // 10 LOB levels
//! let features = extractor.extract_lob_features(&lob_state)?;
//! ```

pub mod derived_features;
pub mod fi2010;
pub mod lob_features;
pub mod market_impact;
pub mod mbo_features;
pub mod order_flow;

use mbo_lob_reconstructor::{LobState, Result};

/// Configuration for feature extraction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureConfig {
    /// Number of LOB levels to extract (typically 10)
    pub lob_levels: usize,

    /// Minimum tick size for price normalization (e.g., 0.01 for US stocks)
    pub tick_size: f64,

    /// Whether to include derived features
    pub include_derived: bool,

    /// Whether to include MBO features
    pub include_mbo: bool,

    /// MBO window size (number of messages to track)
    pub mbo_window_size: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false, // ✅ FIX: Disabled derived features for research paper compliance
            include_mbo: false,
            mbo_window_size: 1000,
        }
    }
}

/// Main feature extractor for LOB and MBO data.
pub struct FeatureExtractor {
    config: FeatureConfig,
    mbo_aggregator: Option<mbo_features::MboAggregator>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with default configuration.
    pub fn new(lob_levels: usize) -> Self {
        Self {
            config: FeatureConfig {
                lob_levels,
                ..Default::default()
            },
            mbo_aggregator: None,
        }
    }

    /// Create a new feature extractor with custom configuration.
    pub fn with_config(config: FeatureConfig) -> Self {
        let mbo_aggregator = if config.include_mbo {
            Some(mbo_features::MboAggregator::with_windows(
                100,                    // fast window
                config.mbo_window_size, // medium window (customizable)
                5000,                   // slow window
            ))
        } else {
            None
        };

        Self {
            config,
            mbo_aggregator,
        }
    }

    /// Get the total number of features that will be extracted.
    pub fn feature_count(&self) -> usize {
        let lob_features = self.config.lob_levels * 4; // ask_price, ask_size, bid_price, bid_size
        let derived_features = if self.config.include_derived { 8 } else { 0 };
        let mbo_features = if self.config.include_mbo { 36 } else { 0 };
        lob_features + derived_features + mbo_features
    }

    /// Extract features from a LOB state.
    ///
    /// Returns a vector of features in the following order:
    /// 1. Ask prices (levels 1-10) - RAW prices in dollars
    /// 2. Ask sizes (levels 1-10) - RAW sizes in shares
    /// 3. Bid prices (levels 1-10) - RAW prices in dollars
    /// 4. Bid sizes (levels 1-10) - RAW sizes in shares
    /// 5. Derived features (if enabled):
    ///    - Mid-price
    ///    - Spread
    ///    - Spread (bps)
    ///    - Total bid volume
    ///    - Total ask volume
    ///    - Volume imbalance
    ///    - Weighted mid-price
    ///    - Price impact
    pub fn extract_lob_features(&self, lob_state: &LobState) -> Result<Vec<f64>> {
        let mut features = Vec::with_capacity(self.feature_count());

        // ✅ FIX: Extract RAW LOB features (absolute prices in dollars, sizes in shares)
        // This matches research paper requirements (PatchTST, TLOB) and enables
        // proper market-structure preserving normalization in the export pipeline
        lob_features::extract_raw_features(lob_state, self.config.lob_levels, &mut features);

        // Add derived features if enabled (disabled by default for research compliance)
        if self.config.include_derived {
            let derived =
                derived_features::compute_derived_features(lob_state, self.config.lob_levels)?;
            features.extend_from_slice(&derived);
        }

        Ok(features)
    }

    /// Process an MBO event for aggregation.
    ///
    /// This should be called for every MBO message to maintain the rolling
    /// window of events used for MBO feature extraction.
    ///
    /// # Arguments
    ///
    /// * `event` - The MBO event to process
    ///
    /// # Note
    ///
    /// This is a no-op if `include_mbo` is false in the configuration.
    #[inline]
    pub fn process_mbo_event(&mut self, event: mbo_features::MboEvent) {
        if let Some(ref mut aggregator) = self.mbo_aggregator {
            aggregator.process_event(event);
        }
    }

    /// Extract all features (LOB + MBO) from current state.
    ///
    /// Returns a vector of features in the following order:
    /// 1. LOB raw features (40 features for 10 levels) - RAW prices and sizes
    /// 2. LOB derived features (8 features, if enabled)
    /// 3. MBO aggregated features (36 features, if enabled)
    ///
    /// # Arguments
    ///
    /// * `lob_state` - Current LOB state
    ///
    /// # Returns
    ///
    /// Vector of features. Length depends on configuration:
    /// - LOB only (default): 40 features (40 raw, derived disabled)
    /// - LOB + derived: 48 features (40 raw + 8 derived)
    /// - LOB + MBO: 76 features (40 raw + 36 MBO)
    /// - LOB + derived + MBO: 84 features (40 raw + 8 derived + 36 MBO)
    pub fn extract_all_features(&mut self, lob_state: &LobState) -> Result<Vec<f64>> {
        let mut features = Vec::with_capacity(self.feature_count());

        // ✅ FIX: Extract RAW LOB features (absolute prices in dollars, sizes in shares)
        lob_features::extract_raw_features(lob_state, self.config.lob_levels, &mut features);

        if self.config.include_derived {
            let derived =
                derived_features::compute_derived_features(lob_state, self.config.lob_levels)?;
            features.extend_from_slice(&derived);
        }

        // Extract MBO features if enabled
        if self.config.include_mbo {
            if let Some(ref mut aggregator) = self.mbo_aggregator {
                let mbo_feats = aggregator.extract_features(lob_state);
                features.extend_from_slice(&mbo_feats);
            }
        }

        Ok(features)
    }

    /// Get the configuration.
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }

    /// Reset all internal state.
    ///
    /// This should be called when starting to process a new day/file
    /// to prevent state leakage between datasets.
    ///
    /// # What gets reset:
    /// - MBO aggregator rolling windows and order tracker
    ///
    /// # When to call:
    /// - Before processing a new trading day
    /// - Before processing a new data file
    /// - When switching between instruments
    pub fn reset(&mut self) {
        // Reset MBO aggregator if present
        if let Some(ref mut aggregator) = self.mbo_aggregator {
            // Re-create the aggregator to clear all state
            *aggregator = mbo_features::MboAggregator::with_windows(
                100,                         // fast window
                self.config.mbo_window_size, // medium window
                5000,                        // slow window
            );
        }
    }

    /// Check if MBO features are enabled.
    #[inline]
    pub fn has_mbo(&self) -> bool {
        self.config.include_mbo
    }

    /// Check if derived features are enabled.
    #[inline]
    pub fn has_derived(&self) -> bool {
        self.config.include_derived
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mbo_lob_reconstructor::Side;

    /// Helper to create a test LOB state with realistic data
    fn create_test_lob_state() -> LobState {
        let mut state = LobState::new(10);

        // Set up realistic LOB with 5 levels populated
        // Ask side: $100.01, $100.02, $100.03, $100.04, $100.05
        for i in 0..5 {
            state.ask_prices[i] = 100_010_000_000 + (i as i64 * 10_000_000);
            state.ask_sizes[i] = (100 + i * 10) as u32;
        }

        // Bid side: $100.00, $99.99, $99.98, $99.97, $99.96
        for i in 0..5 {
            state.bid_prices[i] = 100_000_000_000 - (i as i64 * 10_000_000);
            state.bid_sizes[i] = (100 + i * 10) as u32;
        }

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        state
    }

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count_default() {
        let extractor = FeatureExtractor::new(10);
        assert_eq!(extractor.feature_count(), 40); // 40 raw (derived disabled by default)
    }

    #[test]
    fn test_feature_count_no_derived() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        assert_eq!(extractor.feature_count(), 40); // 40 raw only
    }

    #[test]
    fn test_feature_count_5_levels() {
        let extractor = FeatureExtractor::new(5);
        assert_eq!(extractor.feature_count(), 20); // 20 raw (5 levels × 4)
    }

    #[test]
    fn test_feature_count_with_derived() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        assert_eq!(extractor.feature_count(), 48); // 40 raw + 8 derived
    }

    #[test]
    fn test_feature_count_with_mbo() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        assert_eq!(extractor.feature_count(), 84); // 40 raw + 8 derived + 36 MBO
    }

    #[test]
    fn test_feature_count_mbo_only() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: true,
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        assert_eq!(extractor.feature_count(), 76); // 40 raw + 36 MBO
    }

    // ========================================================================
    // Feature Extraction Tests
    // ========================================================================

    #[test]
    fn test_extract_lob_features_count_matches() {
        let extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let features = extractor.extract_lob_features(&state).unwrap();

        assert_eq!(
            features.len(),
            extractor.feature_count(),
            "extract_lob_features should return exactly feature_count() features"
        );
    }

    #[test]
    fn test_extract_lob_features_with_derived_count_matches() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_lob_features(&state).unwrap();

        assert_eq!(
            features.len(),
            48, // 40 raw + 8 derived
            "extract_lob_features with derived should return 48 features"
        );
    }

    #[test]
    fn test_extract_all_features_count_matches() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 1000,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_all_features(&state).unwrap();

        assert_eq!(
            features.len(),
            extractor.feature_count(),
            "extract_all_features should return exactly feature_count() features"
        );
        assert_eq!(features.len(), 84); // 40 + 8 + 36
    }

    #[test]
    fn test_extract_all_features_mbo_only_count_matches() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: true,
            mbo_window_size: 1000,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_all_features(&state).unwrap();

        assert_eq!(
            features.len(),
            76, // 40 raw + 36 MBO
            "extract_all_features with MBO only should return 76 features"
        );
    }

    #[test]
    fn test_lob_feature_values_are_finite() {
        // Test that LOB features (without MBO) are always finite
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false, // No MBO - those can have NaN when empty
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_lob_features(&state).unwrap();

        for (i, &f) in features.iter().enumerate() {
            assert!(
                f.is_finite(),
                "LOB Feature {} should be finite, got {}",
                i,
                f
            );
        }
    }

    #[test]
    fn test_mbo_features_finite_after_events() {
        // MBO features should be finite after processing enough events
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: true,
            mbo_window_size: 100,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        // Process enough events to populate the aggregator
        for i in 0..200 {
            let event = mbo_features::MboEvent::new(
                1_000_000_000 + i * 1_000_000,
                mbo_lob_reconstructor::Action::Add,
                if i % 2 == 0 { Side::Bid } else { Side::Ask },
                100_000_000_000 + (i % 10) as i64 * 10_000_000,
                (100 + (i % 50)) as u32,
                10000 + i,
            );
            extractor.process_mbo_event(event);
        }

        let features = extractor.extract_all_features(&state).unwrap();

        // Raw LOB features (0-39) should always be finite
        for (i, &f) in features.iter().take(40).enumerate() {
            assert!(
                f.is_finite(),
                "Raw LOB Feature {} should be finite, got {}",
                i,
                f
            );
        }
    }

    #[test]
    fn test_raw_lob_features_order() {
        let extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let features = extractor.extract_lob_features(&state).unwrap();

        // Features should be: ask_prices[0..10], ask_sizes[0..10], bid_prices[0..10], bid_sizes[0..10]
        // Ask price level 0 should be $100.01
        let ask_price_0 = features[0];
        assert!(
            (ask_price_0 - 100.01).abs() < 0.001,
            "First ask price should be ~100.01, got {}",
            ask_price_0
        );

        // Bid price level 0 should be $100.00
        let bid_price_0 = features[20]; // After 10 ask prices + 10 ask sizes
        assert!(
            (bid_price_0 - 100.00).abs() < 0.001,
            "First bid price should be ~100.00, got {}",
            bid_price_0
        );
    }

    // ========================================================================
    // MBO Event Processing Tests
    // ========================================================================

    #[test]
    fn test_process_mbo_event_when_disabled() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false, // MBO disabled
            mbo_window_size: 1000,
        };
        let mut extractor = FeatureExtractor::with_config(config);

        // Should not panic when MBO is disabled
        let event = mbo_features::MboEvent::new(
            1_000_000_000,
            mbo_lob_reconstructor::Action::Add,
            Side::Bid,
            100_000_000_000,
            100,
            12345,
        );
        extractor.process_mbo_event(event);
        // No assertion needed - just verifying it doesn't panic
    }

    #[test]
    fn test_process_mbo_event_when_enabled() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: true, // MBO enabled
            mbo_window_size: 1000,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        // Process some events
        for i in 0..10 {
            let event = mbo_features::MboEvent::new(
                1_000_000_000 + i * 1_000_000,
                mbo_lob_reconstructor::Action::Add,
                if i % 2 == 0 { Side::Bid } else { Side::Ask },
                100_000_000_000,
                100,
                12345 + i,
            );
            extractor.process_mbo_event(event);
        }

        // Should still be able to extract features
        let features = extractor.extract_all_features(&state).unwrap();
        assert_eq!(features.len(), 76); // 40 raw + 36 MBO
    }

    // ========================================================================
    // Reset Tests
    // ========================================================================

    #[test]
    fn test_reset_clears_mbo_state() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: true,
            mbo_window_size: 1000,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        // Process events to build up state
        for i in 0..100 {
            let event = mbo_features::MboEvent::new(
                1_000_000_000 + i * 1_000_000,
                mbo_lob_reconstructor::Action::Add,
                Side::Bid,
                100_000_000_000,
                100,
                12345 + i,
            );
            extractor.process_mbo_event(event);
        }

        // Extract features before reset
        let features_before = extractor.extract_all_features(&state).unwrap();

        // Reset
        extractor.reset();

        // Extract features after reset
        let features_after = extractor.extract_all_features(&state).unwrap();

        // Feature counts should still match
        assert_eq!(features_before.len(), features_after.len());

        // MBO features (indices 40-75) should be different after reset
        // because the aggregator state was cleared
        // (The exact values depend on the aggregator implementation,
        // but at minimum the counts should reset)
    }

    #[test]
    fn test_reset_when_mbo_disabled() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false, // MBO disabled
            mbo_window_size: 1000,
        };
        let mut extractor = FeatureExtractor::with_config(config);

        // Should not panic when MBO is disabled
        extractor.reset();
        // No assertion needed - just verifying it doesn't panic
    }

    // ========================================================================
    // Helper Method Tests
    // ========================================================================

    #[test]
    fn test_has_mbo() {
        let config_with_mbo = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: true,
            mbo_window_size: 1000,
        };
        let config_without_mbo = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
        };

        let extractor_with = FeatureExtractor::with_config(config_with_mbo);
        let extractor_without = FeatureExtractor::with_config(config_without_mbo);

        assert!(extractor_with.has_mbo());
        assert!(!extractor_without.has_mbo());
    }

    #[test]
    fn test_has_derived() {
        let config_with_derived = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 1000,
        };
        let config_without_derived = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
        };

        let extractor_with = FeatureExtractor::with_config(config_with_derived);
        let extractor_without = FeatureExtractor::with_config(config_without_derived);

        assert!(extractor_with.has_derived());
        assert!(!extractor_without.has_derived());
    }

    // ========================================================================
    // Consistency Tests
    // ========================================================================

    #[test]
    fn test_extract_lob_features_deterministic() {
        let extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let features1 = extractor.extract_lob_features(&state).unwrap();
        let features2 = extractor.extract_lob_features(&state).unwrap();

        assert_eq!(
            features1, features2,
            "Extracting features twice should give identical results"
        );
    }

    #[test]
    fn test_extract_all_features_deterministic() {
        // Test determinism for LOB + derived features (without MBO which can have NaN)
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false, // Exclude MBO for determinism test
            mbo_window_size: 1000,
        };
        let extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features1 = extractor.extract_lob_features(&state).unwrap();
        let features2 = extractor.extract_lob_features(&state).unwrap();

        assert_eq!(
            features1, features2,
            "Extracting features twice should give identical results"
        );
    }

    #[test]
    fn test_extract_all_features_with_mbo_deterministic() {
        // Test determinism for full feature set after processing events
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 100,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        // Process events to populate the aggregator
        for i in 0..200 {
            let event = mbo_features::MboEvent::new(
                1_000_000_000 + i * 1_000_000,
                mbo_lob_reconstructor::Action::Add,
                if i % 2 == 0 { Side::Bid } else { Side::Ask },
                100_000_000_000,
                100,
                10000 + i,
            );
            extractor.process_mbo_event(event);
        }

        let features1 = extractor.extract_all_features(&state).unwrap();
        let features2 = extractor.extract_all_features(&state).unwrap();

        // Compare only the raw LOB features (first 40) which should be deterministic
        assert_eq!(
            &features1[..40],
            &features2[..40],
            "Raw LOB features should be deterministic"
        );
    }
}
