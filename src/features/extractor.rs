//! Main feature extractor for LOB and MBO data.

use mbo_lob_reconstructor::{LobState, Result};

use super::config::FeatureConfig;
use super::derived_features;
use super::experimental;
use super::lob_features;
use super::mbo_features;
use super::signals;

/// Main feature extractor for LOB and MBO data.
pub struct FeatureExtractor {
    config: FeatureConfig,
    mbo_aggregator: Option<mbo_features::MboAggregator>,
    /// OFI computer for signal computation (when `include_signals` is true).
    ///
    /// Signals are "streaming" features that accumulate state across
    /// multiple LOB transitions, unlike the other "point-in-time" features.
    ofi_computer: Option<signals::OfiComputer>,
    /// Experimental feature extractor (opt-in).
    ///
    /// Experimental features are appended after standard features (index 98+).
    experimental_extractor: Option<experimental::ExperimentalExtractor>,
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
            ofi_computer: None,
            experimental_extractor: None,
        }
    }

    /// Create a new feature extractor with custom configuration.
    pub fn with_config(config: FeatureConfig) -> Self {
        let mbo_aggregator = if config.include_mbo {
            Some(Self::build_mbo_aggregator(&config))
        } else {
            None
        };

        let ofi_computer = if config.include_signals {
            Some(signals::OfiComputer::new())
        } else {
            None
        };

        let experimental_extractor = if config.experimental.enabled {
            Some(experimental::ExperimentalExtractor::new(
                config.experimental.clone(),
            ))
        } else {
            None
        };

        Self {
            config,
            mbo_aggregator,
            ofi_computer,
            experimental_extractor,
        }
    }

    /// Build an MBO aggregator with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Feature configuration to use for aggregator settings
    ///
    /// # Returns
    ///
    /// A configured `MboAggregator` with:
    /// - Window sizes: fast=100, medium=config.mbo_window_size, slow=5000
    /// - Tick size from config
    /// - Queue tracking if enabled in config
    fn build_mbo_aggregator(config: &FeatureConfig) -> mbo_features::MboAggregator {
        let aggregator = mbo_features::MboAggregator::with_windows(
            100,                    // fast window
            config.mbo_window_size, // medium window (customizable)
            5000,                   // slow window
        )
        .with_tick_size(config.tick_size);

        if config.include_queue_tracking {
            aggregator.with_queue_tracking()
        } else {
            aggregator
        }
    }

    /// Get the total number of features that will be extracted.
    pub fn feature_count(&self) -> usize {
        self.config.feature_count()
    }

    /// Check if signals are enabled.
    #[inline]
    pub fn has_signals(&self) -> bool {
        self.config.include_signals
    }

    /// Get the base feature count (excluding signals).
    ///
    /// This is the number of features produced by `extract_into()`.
    #[inline]
    pub fn base_feature_count(&self) -> usize {
        self.config.base_feature_count()
    }

    /// Get the signal feature count (0 if signals disabled).
    #[inline]
    pub fn signal_feature_count(&self) -> usize {
        self.config.signal_feature_count()
    }

    /// Check if OFI has sufficient warmup for signal computation.
    ///
    /// Returns `true` if at least `MIN_WARMUP_STATE_CHANGES` effective
    /// state changes have occurred since the last reset.
    #[inline]
    pub fn is_signals_warm(&self) -> bool {
        self.ofi_computer
            .as_ref()
            .map(|ofi| ofi.is_warm())
            .unwrap_or(false)
    }

    /// Update OFI state from a LOB snapshot.
    ///
    /// This should be called on **every LOB state transition** to maintain
    /// accurate OFI accumulation for signal computation.
    ///
    /// # Arguments
    ///
    /// * `lob` - The current LOB state after the transition
    ///
    /// # Note
    ///
    /// This is a no-op if `include_signals` is false.
    #[inline]
    pub fn update_ofi(&mut self, lob: &LobState) {
        if let Some(ref mut ofi) = self.ofi_computer {
            ofi.update(lob);
        }
    }

    /// Extract all features including signals.
    ///
    /// This method extracts:
    /// 1. Base features (LOB + derived + MBO) via `extract_into()`
    /// 2. Trading signals (14 features) using accumulated OFI state
    ///
    /// # Arguments
    ///
    /// * `lob` - Current LOB state
    /// * `ctx` - Signal context (timestamp, invalidity_delta)
    /// * `output` - Output buffer (will be cleared and filled)
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, `Err` if signals are disabled or extraction fails.
    ///
    /// # Contract
    ///
    /// The output buffer will contain exactly `feature_count()` features (98 for full config).
    pub fn extract_with_signals(
        &mut self,
        lob: &LobState,
        ctx: &signals::SignalContext,
        output: &mut Vec<f64>,
    ) -> Result<()> {
        if self.ofi_computer.is_none() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "extract_with_signals() called but signals are disabled. \
                 Enable signals in FeatureConfig with .with_signals(true)",
            ));
        }

        self.extract_into(lob, output)?;

        let ofi_computer = self.ofi_computer.as_mut().unwrap();
        let ofi_sample = ofi_computer.sample_and_reset(ctx.timestamp_ns as i64);

        let book_valid = signals::is_book_valid_from_lob(lob);

        let signal_vector = signals::compute_signals_with_book_valid(
            output,
            &ofi_sample,
            ctx.timestamp_ns as i64,
            ctx.invalidity_delta as u32,
            book_valid,
        );

        output.extend_from_slice(&signal_vector.to_vec());

        Ok(())
    }

    /// Extract features from a LOB state.
    ///
    /// Returns a vector of features in the following order:
    /// 1. Ask prices (levels 1-10) - RAW prices in dollars
    /// 2. Ask sizes (levels 1-10) - RAW sizes in shares
    /// 3. Bid prices (levels 1-10) - RAW prices in dollars
    /// 4. Bid sizes (levels 1-10) - RAW sizes in shares
    /// 5. Derived features (if enabled)
    pub fn extract_lob_features(&self, lob_state: &LobState) -> Result<Vec<f64>> {
        let mut features = Vec::with_capacity(self.feature_count());

        lob_features::extract_raw_features(lob_state, self.config.lob_levels, &mut features);

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
    /// # Note
    ///
    /// This is a no-op if `include_mbo` is false in the configuration.
    #[inline]
    pub fn process_mbo_event(&mut self, event: mbo_features::MboEvent) {
        if let Some(ref mut aggregator) = self.mbo_aggregator {
            aggregator.process_event(event.clone());
        }
        if let Some(ref mut exp) = self.experimental_extractor {
            exp.process_event(&event);
        }
    }

    /// Extract all features (LOB + MBO) from current state.
    ///
    /// # Performance Note
    ///
    /// This method allocates a new Vec on every call. For maximum performance
    /// in hot loops, use [`Self::extract_into`] with a pre-allocated buffer.
    pub fn extract_all_features(&mut self, lob_state: &LobState) -> Result<Vec<f64>> {
        let mut features = Vec::with_capacity(self.feature_count());
        self.extract_into(lob_state, &mut features)?;
        Ok(features)
    }

    /// Extract all features into a pre-allocated buffer (zero-allocation hot path).
    ///
    /// This is the **recommended method for maximum performance**. It writes features
    /// directly into the provided buffer without allocating new memory.
    ///
    /// # Feature Order
    ///
    /// 1. Ask prices (levels 1-10) - RAW prices in dollars
    /// 2. Ask sizes (levels 1-10) - RAW sizes in shares
    /// 3. Bid prices (levels 1-10) - RAW prices in dollars
    /// 4. Bid sizes (levels 1-10) - RAW sizes in shares
    /// 5. Derived features (8, if enabled)
    /// 6. MBO features (36, if enabled)
    #[inline]
    pub fn extract_into(&mut self, lob_state: &LobState, output: &mut Vec<f64>) -> Result<()> {
        output.clear();
        output.reserve(self.feature_count());

        lob_features::extract_raw_features(lob_state, self.config.lob_levels, output);

        if self.config.include_derived {
            let derived =
                derived_features::compute_derived_features(lob_state, self.config.lob_levels)?;
            output.extend_from_slice(&derived);
        }

        if self.config.include_mbo {
            if let Some(ref mut aggregator) = self.mbo_aggregator {
                let mbo_feats = aggregator.extract_features(lob_state);
                output.extend_from_slice(&mbo_feats);
            }
        }

        Ok(())
    }

    /// Extract all features and wrap in Arc for zero-copy sharing.
    ///
    /// Use this when you need to share features across multiple consumers
    /// (e.g., multi-scale sequence building) without deep copying.
    #[inline]
    pub fn extract_arc(&mut self, lob_state: &LobState) -> Result<std::sync::Arc<Vec<f64>>> {
        let mut buffer = Vec::with_capacity(self.feature_count());
        self.extract_into(lob_state, &mut buffer)?;
        Ok(std::sync::Arc::new(buffer))
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
    /// - OFI computer (cumulative OFI, warmup counter)
    ///
    /// # What is preserved:
    /// - Configuration (including tick_size)
    pub fn reset(&mut self) {
        if let Some(ref mut aggregator) = self.mbo_aggregator {
            *aggregator = Self::build_mbo_aggregator(&self.config);
        }

        if let Some(ref mut ofi) = self.ofi_computer {
            ofi.reset_on_clear();
        }

        if let Some(ref mut exp) = self.experimental_extractor {
            exp.reset();
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

    /// Check if experimental features are enabled.
    #[inline]
    pub fn has_experimental(&self) -> bool {
        self.config.experimental.enabled
    }

    /// Update experimental volatility features with a new mid-price sample.
    ///
    /// This is a no-op if experimental features are disabled or volatility group
    /// is not included.
    #[inline]
    pub fn update_experimental_price(&mut self, mid_price: f64, timestamp_ns: u64) {
        if let Some(ref mut exp) = self.experimental_extractor {
            exp.update_price(mid_price, timestamp_ns);
        }
    }

    /// Extract experimental features into the output buffer.
    #[inline]
    pub fn extract_experimental_into(&mut self, timestamp_ns: u64, output: &mut Vec<f64>) {
        if let Some(ref mut exp) = self.experimental_extractor {
            exp.extract_into(timestamp_ns, output);
        }
    }

    /// Get the experimental feature count (0 if disabled).
    #[inline]
    pub fn experimental_feature_count(&self) -> usize {
        self.config.experimental.feature_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mbo_lob_reconstructor::Side;

    fn create_test_lob_state() -> LobState {
        let mut state = LobState::new(10);

        for i in 0..5 {
            state.ask_prices[i] = 100_010_000_000 + (i as i64 * 10_000_000);
            state.ask_sizes[i] = (100 + i * 10) as u32;
        }

        for i in 0..5 {
            state.bid_prices[i] = 100_000_000_000 - (i as i64 * 10_000_000);
            state.bid_sizes[i] = (100 + i * 10) as u32;
        }

        state.best_bid = Some(100_000_000_000);
        state.best_ask = Some(100_010_000_000);

        state
    }

    // ========================================================================
    // Feature Count Tests (via extractor delegation to config)
    // ========================================================================

    #[test]
    fn test_feature_count_default() {
        let extractor = FeatureExtractor::new(10);
        assert_eq!(extractor.feature_count(), 40);
    }

    #[test]
    fn test_feature_count_with_derived() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        let extractor = FeatureExtractor::with_config(config);
        assert_eq!(extractor.feature_count(), 48);
    }

    #[test]
    fn test_feature_count_with_mbo() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            ..Default::default()
        };
        let extractor = FeatureExtractor::with_config(config);
        assert_eq!(extractor.feature_count(), 84);
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
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        let extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_lob_features(&state).unwrap();

        assert_eq!(features.len(), 48);
    }

    #[test]
    fn test_extract_all_features_count_matches() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_all_features(&state).unwrap();

        assert_eq!(
            features.len(),
            extractor.feature_count(),
            "extract_all_features should return exactly feature_count() features"
        );
        assert_eq!(features.len(), 84);
    }

    #[test]
    fn test_extract_all_features_mbo_only_count_matches() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let features = extractor.extract_all_features(&state).unwrap();

        assert_eq!(features.len(), 76);
    }

    #[test]
    fn test_lob_feature_values_are_finite() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
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
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            mbo_window_size: 100,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

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

        let ask_price_0 = features[0];
        assert!(
            (ask_price_0 - 100.01).abs() < 0.001,
            "First ask price should be ~100.01, got {}",
            ask_price_0
        );

        let bid_price_0 = features[20];
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
            include_derived: false,
            include_mbo: false,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);

        let event = mbo_features::MboEvent::new(
            1_000_000_000,
            mbo_lob_reconstructor::Action::Add,
            Side::Bid,
            100_000_000_000,
            100,
            12345,
        );
        extractor.process_mbo_event(event);
    }

    #[test]
    fn test_process_mbo_event_when_enabled() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

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

        let features = extractor.extract_all_features(&state).unwrap();
        assert_eq!(features.len(), 76);
    }

    // ========================================================================
    // Reset Tests
    // ========================================================================

    #[test]
    fn test_reset_clears_mbo_state() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

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

        let features_before = extractor.extract_all_features(&state).unwrap();

        extractor.reset();

        let features_after = extractor.extract_all_features(&state).unwrap();

        assert_eq!(features_before.len(), features_after.len());
    }

    #[test]
    fn test_reset_when_mbo_disabled() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);

        extractor.reset();
    }

    // ========================================================================
    // Helper Method Tests
    // ========================================================================

    #[test]
    fn test_has_mbo() {
        let config_with_mbo = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: true,
            ..Default::default()
        };
        let config_without_mbo = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: false,
            ..Default::default()
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
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        let config_without_derived = FeatureConfig {
            lob_levels: 10,
            include_derived: false,
            include_mbo: false,
            ..Default::default()
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
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
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
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 100,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

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

        assert_eq!(
            &features1[..40],
            &features2[..40],
            "Raw LOB features should be deterministic"
        );
    }

    // ========================================================================
    // Zero-Allocation API Tests
    // ========================================================================

    #[test]
    fn test_extract_into_basic() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();
        let mut buffer = Vec::new();

        extractor.extract_into(&state, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 40);
    }

    #[test]
    fn test_extract_into_reuses_buffer() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();
        let mut buffer = Vec::with_capacity(100);

        extractor.extract_into(&state, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 40);
        assert!(buffer.capacity() >= 100);

        extractor.extract_into(&state, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 40);
        assert!(buffer.capacity() >= 100);
    }

    #[test]
    fn test_extract_into_clears_previous_data() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();
        let mut buffer = vec![999.0; 200];

        extractor.extract_into(&state, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 40);
        assert_ne!(buffer[0], 999.0);
    }

    #[test]
    fn test_extract_into_matches_extract_all_features() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 100,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        for i in 0..50 {
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

        let vec_result = extractor.extract_all_features(&state).unwrap();

        let mut buffer = Vec::new();
        extractor.extract_into(&state, &mut buffer).unwrap();

        assert_eq!(
            vec_result.len(),
            buffer.len(),
            "Length mismatch: extract_all_features={}, extract_into={}",
            vec_result.len(),
            buffer.len()
        );

        for (i, (&expected, &actual)) in vec_result.iter().zip(buffer.iter()).enumerate() {
            assert_eq!(
                expected, actual,
                "Feature {} mismatch: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_extract_into_with_all_feature_types() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 100,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let mut buffer = Vec::new();
        extractor.extract_into(&state, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 84);
    }

    #[test]
    fn test_extract_arc_basic() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let arc_result = extractor.extract_arc(&state).unwrap();

        assert_eq!(arc_result.len(), 40);
        assert_eq!(std::sync::Arc::strong_count(&arc_result), 1);
    }

    #[test]
    fn test_extract_arc_sharing() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let arc1 = extractor.extract_arc(&state).unwrap();
        let arc2 = arc1.clone();
        let arc3 = arc1.clone();

        assert_eq!(std::sync::Arc::strong_count(&arc1), 3);
        assert_eq!(std::sync::Arc::strong_count(&arc2), 3);
        assert_eq!(std::sync::Arc::strong_count(&arc3), 3);

        assert_eq!(arc1[0], arc2[0]);
        assert_eq!(arc1[0], arc3[0]);

        drop(arc2);
        assert_eq!(std::sync::Arc::strong_count(&arc1), 2);
    }

    #[test]
    fn test_extract_arc_matches_extract_all_features() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 100,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let vec_result = extractor.extract_all_features(&state).unwrap();
        let arc_result = extractor.extract_arc(&state).unwrap();

        assert_eq!(vec_result.len(), arc_result.len());
        for (i, (&expected, &actual)) in vec_result.iter().zip(arc_result.iter()).enumerate() {
            assert_eq!(
                expected, actual,
                "Feature {} mismatch: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_extract_into_numerical_precision() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let mut buffer = Vec::new();
        extractor.extract_into(&state, &mut buffer).unwrap();

        for (i, &value) in buffer.iter().enumerate() {
            assert!(value.is_finite(), "Feature {} is not finite: {}", i, value);
        }

        let ask_price_0 = buffer[0];
        assert!(
            ask_price_0 > 99.0 && ask_price_0 < 101.0,
            "Ask price 0 out of range: {}",
            ask_price_0
        );

        let bid_price_0 = buffer[20];
        assert!(
            bid_price_0 > 99.0 && bid_price_0 < 101.0,
            "Bid price 0 out of range: {}",
            bid_price_0
        );
    }

    #[test]
    fn test_extract_into_multiple_calls_consistency() {
        let config = FeatureConfig {
            lob_levels: 10,
            include_derived: true,
            include_mbo: false,
            ..Default::default()
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let mut buffer = Vec::new();

        let mut results: Vec<Vec<f64>> = Vec::new();
        for _ in 0..10 {
            extractor.extract_into(&state, &mut buffer).unwrap();
            results.push(buffer.clone());
        }

        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(
                results[0], *result,
                "Extraction {} differs from first extraction",
                i
            );
        }
    }
}
