# Feature Extractor TODO

> **Purpose**: Track implementation gaps, documentation inconsistencies, and future improvements.
> These items do NOT block current TLOB training but should be addressed for long-term maintainability.

---

## Priority Legend

| Priority | Description |
|----------|-------------|
| **P0** | Critical - Blocks current work |
| **P1** | High - Should fix soon |
| **P2** | Medium - Future improvement |
| **P3** | Low - Nice to have |

---

## 1. Not Yet Implemented

### 1.1 Trait-Based Sampling Architecture

**Status**: ✅ Implemented (2026-03)

**Location**: `src/preprocessing/sampling.rs` (trait + all implementations)

**What was done**:
- `Sampler` trait with unified `should_sample(&SamplingContext) -> bool` interface
- `TimeBasedSampler`: fixed-interval grid aligned to 09:30 ET market open
- `CompositeSampler`: OR/AND composition of child samplers
- Pipeline dispatch via `Box<dyn Sampler>` (replaces old `SamplerType` enum)
- `MlofiComputer`/`KolmOfComputer` `sample_and_reset()` for interval-scoped MLOFI/Kolm OF
- Config: `time_interval_ns`, `utc_offset_hours` fields on `SamplingConfig`
- Builder: `.time_sampling(interval_ns, utc_offset_hours)` method
- 18 new unit tests + 10 integration tests in `tests/sampling_integration_tests.rs`

---

### 1.2 median_order_lifetime() Feature (P3)

**Status**: ✅ Implemented

**Location**: `src/features/mbo_features/lifecycle_features.rs` (directory module)

**Implementation** (2026-01-25):
- Added `completed_lifetimes: VecDeque<f64>` bounded buffer (size 1000)
- When orders are cancelled or fully filled, their lifetime is recorded
- Median computed via sorting on extraction (O(n log n), n ≤ 1000)
- Memory cost: 8 KB (negligible)
- 9 comprehensive tests added covering:
  - No completed orders (returns 0.0)
  - Single order, odd count, even count
  - Cancelled vs filled orders
  - Mixed cancel and fill
  - Partial fills (only fully filled counts)
  - Buffer bounding behavior
  - Feature extraction index verification

---

### 1.3 FI-2010 Regression Features (P3)

**Status**: ⚠️ Not implemented

**Location**: `src/features/fi2010.rs`

**Issue**: The original FI-2010 benchmark paper defines 144 total handcrafted features:
- Time-Insensitive: 20 features ✅ Implemented
- Time-Sensitive: 20 features ✅ Implemented  
- Depth Features: 40 features ✅ Implemented
- **Regression Features: 24 features ❌ NOT implemented**

**Missing Features (24)**:
The regression features require fitting linear regressions on rolling windows of:
- Mid-price regression coefficients (slope, intercept, R²) × multiple windows
- Spread regression coefficients × multiple windows
- Volume regression coefficients × multiple windows

**Implementation Notes**:
- Would require a rolling window buffer for regression fitting
- Computationally expensive (least squares per extraction)
- May require `nalgebra` or similar linear algebra crate

**Impact on TLOB Training**: None - TLOB uses 40 raw LOB features, not FI-2010 features.

**Current State**: `fi2010.rs` implements 80 features (20+20+40), documentation updated accordingly.

---

## 2. Documentation Inconsistencies

### 2.1 FI-2010 Preset Feature Count (P1)

**Status**: ❌ Documentation does not match implementation

**Location**: `src/builder.rs:148-149` (docstring), `src/builder.rs:172-176` (implementation)

**Documentation Claims**:
```rust
/// - `Preset::FI2010`: 144 features (40 raw + 104 handcrafted)
```

**Actual Implementation**:
```rust
Preset::FI2010 => {
    builder.features = FeatureConfig::new(10).with_derived(true);  // Only 48 features!
    builder.window_size = 100;
    builder.stride = 1;
}
```

**Analysis**:
- The FI-2010 benchmark dataset has 144 handcrafted features
- Our `derived_features` module only provides 8 features
- The 80 FI-2010 handcrafted features are partially implemented in `src/features/fi2010.rs` but NOT
  integrated into the feature extractor pipeline

**Resolution Options**:
1. **Update Documentation** (recommended for now): Change docstring to reflect actual 48 features
2. **Implement Full FI-2010**: Integrate fi2010.rs features into FeatureConfig (significant work)

**Impact on TLOB Training**: None - TLOB doesn't use the FI-2010 preset

---

### 2.2 Missing Full Citations (P2)

**Status**: ⚠️ Some papers referenced without full citations

**Locations with Missing Citations**:

| Location | Reference | Missing |
|----------|-----------|---------|
| `ARCHITECTURE.md:164` | "MBO Paper" | Specific paper not cited |
| `CODEBASE.md:50-56` | Research compliance list | Missing full citation format |
| `README.md:687-695` | Research papers section | Missing years and venues |
| `docs/LABELING_STRATEGIES.md:473` | "TLOB: Zhang..." | Incorrectly attributes to DeepLOB |

**Recommended Full Citations**:

1. **DeepLOB**: Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional
   Neural Networks for Limit Order Books." IEEE Transactions on Signal Processing, 67(11).

2. **FI-2010**: Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018).
   "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods."
   Journal of Forecasting, 37(8).

3. **TLOB**: Berti, K., & Kasneci, G. (2025). "TLOB: A Novel Transformer Model with Dual Attention
   for Price Trend Prediction with Limit Order Book Data."

4. **BiN (Bilinear Normalization)**: Tran, D. T., et al. (2021). "Data Normalization for Bilinear
   Structures in High-Frequency Financial Time-series." ICPR 2020.

5. **OFI (Order Flow Imbalance)**: Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact
   of Order Book Events." Journal of Financial Econometrics, 12(1).

6. **Microprice**: Stoikov, S. (2018). "The Micro-Price: A High Frequency Estimator of Future Prices."
   Quantitative Finance, 18(12).

7. **Triple Barrier**: López de Prado, M. (2018). "Advances in Financial Machine Learning."
   Wiley. Chapter 3.

---

## 3. Code Quality Items

### 3.1 fi2010.rs Module Not Integrated (P2)

**Status**: ⚠️ Module exists but not used in pipeline

**Location**: `src/features/fi2010.rs`

**Issue**: The FI-2010 feature extraction module is implemented (80 features) but:
- Not exposed in `FeatureConfig`
- Not integrated into `FeatureExtractor`
- No `include_fi2010` flag exists

**Full FI-2010 Paper vs Our Implementation**:
- Original paper: 144 features (40 raw + 80 handcrafted + 24 regression)
- Our fi2010.rs: 80 features (20 time-insensitive + 20 time-sensitive + 40 depth)
- Missing: 24 regression features (see TODO 1.3)

**Impact**: Users expecting full FI-2010 feature set will get partial implementation.

---

### 3.2 Magnitude Export Integration (P2)

**Status**: ⚠️ API implemented, export_dataset integration pending

**Location**: `src/labeling/magnitude.rs`

**Current State**:
- ✅ `MagnitudeGenerator` fully implemented and tested
- ❌ NOT integrated into `export_dataset` CLI tool
- ❌ Cannot be configured via TOML

> **Note**: Triple Barrier IS fully integrated (Schema 2.4+). Use `strategy = "triple_barrier"` in TOML config.

**Tracking**: Documented in `docs/LABELING_STRATEGIES.md#integration-status`

---

### 3.3 MinMax and Bilinear Normalization Strategies (P2)

**Status**: ⚠️ Declared but using fallback implementations

**Location**: `src/export_aligned/normalization.rs`, `src/export/dataset_config.rs`

**Issue**: The `FeatureNormStrategy::MinMax` and `FeatureNormStrategy::Bilinear` variants are
declared in the enum and can be configured via TOML, but their implementations are incomplete:

| Strategy | Expected Behavior | Current Behavior |
|----------|-------------------|------------------|
| MinMax | `(x - min) / (max - min)` scaling to [0,1] | Falls back to Z-score |
| Bilinear | `(price - mid_price) / (k * tick_size)` | Uses `(value - mean) / (k * 0.01)` approximation |

**Implementation Requirements**:

1. **MinMax**:
   - Track min/max per feature during statistics collection (currently only mean/std)
   - Store min/max in `NormalizationParams` for denormalization
   - Handle edge case where min == max (avoid division by zero)

2. **Bilinear**:
   - Requires per-timestep mid_price (from feature index 40 or calculated from best bid/ask)
   - tick_size should come from `FeatureConfig.tick_size` (currently hardcoded as 0.01)
   - More complex because mid_price varies per timestep, not per feature

**Impact**: Low - no presets use these strategies, and users are warned via description text.

**Workarounds**:
- For MinMax needs: Use `ZScore` (similar scaling properties)
- For Bilinear needs: Use `NormalizationConfig::raw()` with TLOB model's BiN layer

**Resolution Date**: Documented 2026-01-25, pending full implementation.

---

## 4. Future Enhancements (P3)

### 4.1 Adaptive Sampling Based on Volatility

The `VolumeBasedSampler.set_threshold()` method exists but:
- No automatic volatility detection
- Threshold must be manually updated

**Potential Enhancement**: Auto-adjust sampling threshold based on realized volatility.

### 4.2 Weighted MLOFI

Referenced in `ARCHITECTURE.md:279`:
> "Weighted MLOFI (depth-aware OFI scalar per Xu et al.)"

Not yet implemented.

### 4.3 Additional Paper Presets

From `ARCHITECTURE.md:279`:
> "Additional paper presets (ViT-LOB)"

---

## 5. Testing Gaps

### 5.1 Reset Semantics Integration Tests

**Status**: ✅ Resolved

**Current State**: Multiple integration tests now verify reset semantics:
- `tests/state_leak_test.rs` (9 tests) -- verifies no state leakage across resets
- `tests/multiscale_reset_leak_test.rs` (4 tests) -- multi-scale window reset verification
- `tests/phase3_real_data_validation.rs::test_cross_day_state_isolation` -- processes day A + reset + day B == fresh pipeline on day B

---

## Resolution Tracking

| ID | Item | Status | Resolved Date |
|----|------|--------|---------------|
| 2.1 | FI-2010 preset docstring | ✅ Fixed | 2026-01-14 |
| 1.2 | median_order_lifetime | ✅ Implemented | 2026-01-25 |
| 3.3 | MinMax/Bilinear normalization | ⏳ Documented + fallback | 2026-01-25 |
| 1.1 | TimeBased sampling | ⏳ Documented | - |
| 2.2 | Missing citations | ⏳ Pending | - |
| 1.3 | FI-2010 regression features | ⏳ Documented | - |
| 3.1 | fi2010.rs integration | ⏳ Documented | - |
| 3.2 | Triple Barrier export integration | ✅ Integrated | 2026-02-15 |

---

*Last Updated: 2026-03-05*
