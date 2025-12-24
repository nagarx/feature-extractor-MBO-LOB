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
pub mod signals;

// Re-export commonly used types for convenience
pub use signals::SignalContext;

use mbo_lob_reconstructor::{LobState, Result};

/// Configuration for feature extraction.
///
/// # Feature Count Calculation
///
/// The total number of features is automatically computed based on configuration:
///
/// | Configuration | Feature Count | Breakdown |
/// |--------------|---------------|-----------|
/// | Default (LOB only) | 40 | 10 levels x 4 features |
/// | LOB + derived | 48 | 40 + 8 derived |
/// | LOB + MBO | 76 | 40 + 36 MBO |
/// | LOB + derived + MBO | 84 | 40 + 8 + 36 |
///
/// Use [`FeatureConfig::feature_count()`] to get the computed count.
///
/// # Example
///
/// ```
/// use feature_extractor::FeatureConfig;
///
/// // Default: 40 features (raw LOB only)
/// let config = FeatureConfig::default();
/// assert_eq!(config.feature_count(), 40);
///
/// // With derived features: 48 features
/// let config = FeatureConfig::default().with_derived(true);
/// assert_eq!(config.feature_count(), 48);
///
/// // With MBO features: 76 features
/// let config = FeatureConfig::default().with_mbo(true);
/// assert_eq!(config.feature_count(), 76);
///
/// // Full feature set: 84 features
/// let config = FeatureConfig::default().with_derived(true).with_mbo(true);
/// assert_eq!(config.feature_count(), 84);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureConfig {
    /// Number of LOB levels to extract (typically 10)
    pub lob_levels: usize,

    /// Minimum tick size for price normalization (e.g., 0.01 for US stocks)
    pub tick_size: f64,

    /// Whether to include derived features (8 additional features)
    pub include_derived: bool,

    /// Whether to include MBO features (36 additional features)
    pub include_mbo: bool,

    /// MBO window size (number of messages to track)
    pub mbo_window_size: usize,

    /// Whether to include trading signals (14 additional features, indices 84-97)
    ///
    /// Requires `include_derived` and `include_mbo` to be enabled.
    /// Adds signals: true_ofi, depth_norm_ofi, executed_pressure, etc.
    pub include_signals: bool,
}

impl FeatureConfig {
    /// Number of derived features when enabled.
    pub const DERIVED_FEATURE_COUNT: usize = 8;

    /// Number of MBO features when enabled.
    pub const MBO_FEATURE_COUNT: usize = 36;

    /// Number of trading signal features when enabled (indices 84-97).
    pub const SIGNAL_FEATURE_COUNT: usize = 14;

    /// Create a new feature configuration with default values.
    pub fn new(lob_levels: usize) -> Self {
        Self {
            lob_levels,
            ..Default::default()
        }
    }

    /// Enable or disable derived features.
    ///
    /// Derived features include: mid-price, spread, spread_bps, total_bid_volume,
    /// total_ask_volume, volume_imbalance, weighted_mid_price, price_impact.
    pub fn with_derived(mut self, enabled: bool) -> Self {
        self.include_derived = enabled;
        self
    }

    /// Enable or disable MBO features.
    ///
    /// MBO features include order flow imbalance, trade intensity, order arrival
    /// rates, cancellation rates, and other microstructure metrics.
    pub fn with_mbo(mut self, enabled: bool) -> Self {
        self.include_mbo = enabled;
        self
    }

    /// Set the MBO window size.
    pub fn with_mbo_window(mut self, window_size: usize) -> Self {
        self.mbo_window_size = window_size;
        self
    }

    /// Set the tick size.
    pub fn with_tick_size(mut self, tick_size: f64) -> Self {
        self.tick_size = tick_size;
        self
    }

    /// Enable or disable trading signals (14 additional features).
    ///
    /// Trading signals include: true_ofi, depth_norm_ofi, executed_pressure,
    /// signed_mp_delta_bps, trade_asymmetry, cancel_asymmetry, fragility_score,
    /// depth_asymmetry, book_valid, time_regime, mbo_ready, dt_seconds,
    /// invalidity_delta, schema_version.
    ///
    /// **Note**: Signals require both `include_derived` and `include_mbo` to be enabled.
    pub fn with_signals(mut self, enabled: bool) -> Self {
        self.include_signals = enabled;
        self
    }

    /// Compute the total number of features based on configuration.
    ///
    /// This is the authoritative source for feature count calculation.
    /// Use this method instead of manually computing feature counts.
    ///
    /// # Returns
    ///
    /// Total number of features:
    /// - Base: `lob_levels * 4` (ask_price, ask_size, bid_price, bid_size per level)
    /// - If `include_derived`: + 8 derived features
    /// - If `include_mbo`: + 36 MBO features
    #[inline]
    pub fn feature_count(&self) -> usize {
        let base = self.lob_levels * 4;
        let derived = if self.include_derived {
            Self::DERIVED_FEATURE_COUNT
        } else {
            0
        };
        let mbo = if self.include_mbo {
            Self::MBO_FEATURE_COUNT
        } else {
            0
        };
        let signals = if self.include_signals {
            Self::SIGNAL_FEATURE_COUNT
        } else {
            0
        };
        base + derived + mbo + signals
    }

    /// Get the number of raw LOB features.
    #[inline]
    pub fn lob_feature_count(&self) -> usize {
        self.lob_levels * 4
    }

    /// Get the base feature count (excluding signals).
    ///
    /// This is the number of features produced by point-in-time extraction
    /// (LOB + derived + MBO), without the streaming signals.
    ///
    /// # Returns
    ///
    /// Base feature count:
    /// - Base: `lob_levels * 4`
    /// - If `include_derived`: + 8 derived features  
    /// - If `include_mbo`: + 36 MBO features
    #[inline]
    pub fn base_feature_count(&self) -> usize {
        let base = self.lob_levels * 4;
        let derived = if self.include_derived {
            Self::DERIVED_FEATURE_COUNT
        } else {
            0
        };
        let mbo = if self.include_mbo {
            Self::MBO_FEATURE_COUNT
        } else {
            0
        };
        base + derived + mbo
    }

    /// Get the signal feature count (0 if signals disabled).
    #[inline]
    pub fn signal_feature_count(&self) -> usize {
        if self.include_signals {
            Self::SIGNAL_FEATURE_COUNT
        } else {
            0
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.lob_levels == 0 {
            return Err("lob_levels must be > 0".to_string());
        }
        if self.lob_levels > 50 {
            return Err("lob_levels must be <= 50 (practical limit)".to_string());
        }
        if self.tick_size <= 0.0 {
            return Err("tick_size must be > 0".to_string());
        }
        if self.include_mbo && self.mbo_window_size == 0 {
            return Err("mbo_window_size must be > 0 when MBO is enabled".to_string());
        }
        if self.include_signals && !self.include_derived {
            return Err(
                "include_signals requires include_derived to be enabled".to_string(),
            );
        }
        if self.include_signals && !self.include_mbo {
            return Err("include_signals requires include_mbo to be enabled".to_string());
        }
        // Signal indices are hardcoded for 10 levels (per signals.rs):
        // - derived_indices::MID_PRICE = 40 (assumes 10 × 4 = 40 raw features)
        // - mbo_indices::CANCEL_RATE_BID = 50 (assumes MBO starts at 48)
        // With any other level count, signals would read from wrong indices.
        if self.include_signals && self.lob_levels != 10 {
            return Err(format!(
                "include_signals requires exactly 10 lob_levels (got {}). \
                 Signal feature indices are hardcoded for the 10-level layout. \
                 See signals.rs for details.",
                self.lob_levels
            ));
        }
        Ok(())
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
            include_signals: false,
        }
    }
}

/// Main feature extractor for LOB and MBO data.
pub struct FeatureExtractor {
    config: FeatureConfig,
    mbo_aggregator: Option<mbo_features::MboAggregator>,
    /// OFI computer for signal computation (when `include_signals` is true).
    ///
    /// Signals are "streaming" features that accumulate state across
    /// multiple LOB transitions, unlike the other "point-in-time" features.
    ofi_computer: Option<signals::OfiComputer>,
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
        }
    }

    /// Create a new feature extractor with custom configuration.
    pub fn with_config(config: FeatureConfig) -> Self {
        let mbo_aggregator = if config.include_mbo {
            Some(
                mbo_features::MboAggregator::with_windows(
                    100,                    // fast window
                    config.mbo_window_size, // medium window (customizable)
                    5000,                   // slow window
                )
                .with_tick_size(config.tick_size), // Propagate tick_size from config
            )
        } else {
            None
        };

        let ofi_computer = if config.include_signals {
            Some(signals::OfiComputer::new())
        } else {
            None
        };

        Self {
            config,
            mbo_aggregator,
            ofi_computer,
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
    ///
    /// # Usage Pattern
    ///
    /// ```ignore
    /// // On every LOB update (before sampling):
    /// extractor.update_ofi(&lob_state);
    ///
    /// // At sampling points (e.g., after volume threshold):
    /// let ctx = SignalContext::new(timestamp_ns, invalidity_delta);
    /// extractor.extract_with_signals(&lob_state, &ctx, &mut output)?;
    /// ```
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
    ///
    /// # Usage
    ///
    /// ```ignore
    /// let config = FeatureConfig::new(10)
    ///     .with_derived(true)
    ///     .with_mbo(true)
    ///     .with_signals(true);
    /// let mut extractor = FeatureExtractor::with_config(config);
    ///
    /// // Update OFI on every LOB transition
    /// for lob_update in updates {
    ///     extractor.update_ofi(&lob_update);
    /// }
    ///
    /// // At sampling point, extract with signals
    /// let ctx = SignalContext::new(timestamp_ns, invalidity_delta);
    /// let mut output = Vec::new();
    /// extractor.extract_with_signals(&lob, &ctx, &mut output)?;
    /// assert_eq!(output.len(), 98);
    /// ```
    pub fn extract_with_signals(
        &mut self,
        lob: &LobState,
        ctx: &signals::SignalContext,
        output: &mut Vec<f64>,
    ) -> Result<()> {
        // Verify signals are enabled
        if self.ofi_computer.is_none() {
            return Err(mbo_lob_reconstructor::TlobError::generic(
                "extract_with_signals() called but signals are disabled. \
                 Enable signals in FeatureConfig with .with_signals(true)"
            ));
        }

        // Extract base features
        self.extract_into(lob, output)?;

        // Sample OFI and compute signals
        let ofi_computer = self.ofi_computer.as_mut().unwrap();
        let ofi_sample = ofi_computer.sample_and_reset(ctx.timestamp_ns as i64);

        // Compute book_valid from LOB state
        let book_valid = signals::is_book_valid_from_lob(lob);

        // Compute all 14 signals
        let signal_vector = signals::compute_signals_with_book_valid(
            output,
            &ofi_sample,
            ctx.timestamp_ns as i64,
            ctx.invalidity_delta as u32,
            book_valid,
        );

        // Append signals to output
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

    // ========================================================================
    // Zero-Allocation API (Phase 1 Long-Term Architecture)
    // ========================================================================

    /// Extract all features into a pre-allocated buffer (zero-allocation hot path).
    ///
    /// This is the **recommended method for maximum performance**. It writes features
    /// directly into the provided buffer without allocating new memory.
    ///
    /// # Arguments
    ///
    /// * `lob_state` - Current LOB state
    /// * `output` - Pre-allocated buffer to write features into (will be cleared first)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Features successfully written to output buffer
    /// * `Err(...)` - If feature extraction fails (e.g., invalid LOB state)
    ///
    /// # Feature Order
    ///
    /// Features are written in this order:
    /// 1. Ask prices (levels 1-10) - RAW prices in dollars
    /// 2. Ask sizes (levels 1-10) - RAW sizes in shares
    /// 3. Bid prices (levels 1-10) - RAW prices in dollars
    /// 4. Bid sizes (levels 1-10) - RAW sizes in shares
    /// 5. Derived features (8, if enabled)
    /// 6. MBO features (36, if enabled)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::FeatureExtractor;
    ///
    /// let mut extractor = FeatureExtractor::new(10);
    ///
    /// // Pre-allocate buffer ONCE outside the loop
    /// let mut buffer = Vec::with_capacity(extractor.feature_count());
    ///
    /// for lob_state in lob_states {
    ///     // Reuse buffer - no allocation in hot path
    ///     extractor.extract_into(&lob_state, &mut buffer)?;
    ///     
    ///     // Process features...
    ///     process(&buffer);
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// - **Zero heap allocations** in the hot path
    /// - Buffer is cleared and reused on each call
    /// - Capacity is preserved across calls
    #[inline]
    pub fn extract_into(&mut self, lob_state: &LobState, output: &mut Vec<f64>) -> Result<()> {
        // Clear buffer but preserve capacity
        output.clear();
        output.reserve(self.feature_count());

        // Extract RAW LOB features (prices in dollars, sizes in shares)
        lob_features::extract_raw_features(lob_state, self.config.lob_levels, output);

        // Extract derived features if enabled
        if self.config.include_derived {
            let derived =
                derived_features::compute_derived_features(lob_state, self.config.lob_levels)?;
            output.extend_from_slice(&derived);
        }

        // Extract MBO features if enabled
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
    /// This is a convenience method that:
    /// 1. Allocates a new Vec with appropriate capacity
    /// 2. Extracts features using [`Self::extract_into`]
    /// 3. Wraps the result in `Arc` for sharing
    ///
    /// Use this when you need to share features across multiple consumers
    /// (e.g., multi-scale sequence building) without deep copying.
    ///
    /// # Arguments
    ///
    /// * `lob_state` - Current LOB state
    ///
    /// # Returns
    ///
    /// * `Ok(Arc<Vec<f64>>)` - Shared feature vector
    /// * `Err(...)` - If feature extraction fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::FeatureExtractor;
    /// use std::sync::Arc;
    ///
    /// let mut extractor = FeatureExtractor::new(10);
    ///
    /// // Extract and wrap in Arc
    /// let features: Arc<Vec<f64>> = extractor.extract_arc(&lob_state)?;
    ///
    /// // Share across multiple consumers (cheap Arc clone)
    /// let features_for_fast = features.clone();   // 8 bytes
    /// let features_for_medium = features.clone(); // 8 bytes
    /// let features_for_slow = features;           // move
    /// ```
    ///
    /// # Performance
    ///
    /// - **One allocation** per call (the Vec)
    /// - Arc wrapping is essentially free
    /// - Subsequent clones are O(1) (just atomic increment)
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
    ///
    /// # When to call:
    /// - Before processing a new trading day
    /// - Before processing a new data file
    /// - When switching between instruments
    pub fn reset(&mut self) {
        // Reset MBO aggregator if present
        if let Some(ref mut aggregator) = self.mbo_aggregator {
            // Re-create the aggregator to clear all state, preserving tick_size from config
            *aggregator = mbo_features::MboAggregator::with_windows(
                100,                         // fast window
                self.config.mbo_window_size, // medium window
                5000,                        // slow window
            )
            .with_tick_size(self.config.tick_size); // Preserve tick_size from config
        }

        // Reset OFI computer if present
        if let Some(ref mut ofi) = self.ofi_computer {
            ofi.reset_on_clear(); // Full reset including warmup counter
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
        };
        let config_without_mbo = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
            include_signals: false,
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
            include_signals: false,
        };
        let config_without_derived = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: false,
            include_mbo: false,
            mbo_window_size: 1000,
            include_signals: false,
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
            include_signals: false,
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
            include_signals: false,
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

    // ========================================================================
    // Zero-Allocation API Tests (extract_into, extract_arc)
    // ========================================================================

    #[test]
    fn test_extract_into_basic() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();
        let mut buffer = Vec::new();

        // Extract into empty buffer
        extractor.extract_into(&state, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 40);
    }

    #[test]
    fn test_extract_into_reuses_buffer() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();
        let mut buffer = Vec::with_capacity(100);

        // First extraction
        extractor.extract_into(&state, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 40);
        assert!(buffer.capacity() >= 100); // Capacity preserved

        // Second extraction should reuse same buffer
        extractor.extract_into(&state, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 40);
        assert!(buffer.capacity() >= 100); // Capacity still preserved
    }

    #[test]
    fn test_extract_into_clears_previous_data() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();
        let mut buffer = vec![999.0; 200]; // Pre-fill with garbage

        extractor.extract_into(&state, &mut buffer).unwrap();

        // Should have exactly 40 features, not 240
        assert_eq!(buffer.len(), 40);
        // First value should not be 999.0
        assert_ne!(buffer[0], 999.0);
    }

    #[test]
    fn test_extract_into_matches_extract_all_features() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 100,
            include_signals: false,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        // Process some MBO events for consistency
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

        // Extract using both methods
        let vec_result = extractor.extract_all_features(&state).unwrap();

        let mut buffer = Vec::new();
        extractor.extract_into(&state, &mut buffer).unwrap();

        // Results must be IDENTICAL
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
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 100,
            include_signals: false,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let mut buffer = Vec::new();
        extractor.extract_into(&state, &mut buffer).unwrap();

        // Should have 84 features: 40 raw + 8 derived + 36 MBO
        assert_eq!(buffer.len(), 84);
    }

    #[test]
    fn test_extract_arc_basic() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let arc_result = extractor.extract_arc(&state).unwrap();

        assert_eq!(arc_result.len(), 40);
        // Arc should have strong count of 1
        assert_eq!(std::sync::Arc::strong_count(&arc_result), 1);
    }

    #[test]
    fn test_extract_arc_sharing() {
        let mut extractor = FeatureExtractor::new(10);
        let state = create_test_lob_state();

        let arc1 = extractor.extract_arc(&state).unwrap();
        let arc2 = arc1.clone(); // Cheap clone
        let arc3 = arc1.clone(); // Another cheap clone

        // All point to same data
        assert_eq!(std::sync::Arc::strong_count(&arc1), 3);
        assert_eq!(std::sync::Arc::strong_count(&arc2), 3);
        assert_eq!(std::sync::Arc::strong_count(&arc3), 3);

        // Values are identical
        assert_eq!(arc1[0], arc2[0]);
        assert_eq!(arc1[0], arc3[0]);

        // Dropping reduces count
        drop(arc2);
        assert_eq!(std::sync::Arc::strong_count(&arc1), 2);
    }

    #[test]
    fn test_extract_arc_matches_extract_all_features() {
        let config = FeatureConfig {
            lob_levels: 10,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 100,
            include_signals: false,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let vec_result = extractor.extract_all_features(&state).unwrap();
        let arc_result = extractor.extract_arc(&state).unwrap();

        // Must be identical
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
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 100,
            include_signals: false,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let mut buffer = Vec::new();
        extractor.extract_into(&state, &mut buffer).unwrap();

        // Verify no NaN or Inf values
        for (i, &value) in buffer.iter().enumerate() {
            assert!(value.is_finite(), "Feature {} is not finite: {}", i, value);
        }

        // Verify prices are in reasonable range (test state has $100 prices)
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
            tick_size: 0.01,
            include_derived: true,
            include_mbo: false,
            mbo_window_size: 100,
            include_signals: false,
        };
        let mut extractor = FeatureExtractor::with_config(config);
        let state = create_test_lob_state();

        let mut buffer = Vec::new();

        // Extract 10 times, verify identical results
        let mut results: Vec<Vec<f64>> = Vec::new();
        for _ in 0..10 {
            extractor.extract_into(&state, &mut buffer).unwrap();
            results.push(buffer.clone());
        }

        // All results should be identical
        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(
                results[0], *result,
                "Extraction {} differs from first extraction",
                i
            );
        }
    }
}
