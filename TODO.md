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

### 1.1 TimeBased Sampling Strategy (P2)

**Status**: ⚠️ Declared but not implemented

**Location**: `src/config.rs:183` (SamplingStrategy enum), `src/pipeline.rs:815-821`

**Current Behavior**: Returns explicit error if used:
```rust
Err("TimeBased sampling strategy is not yet implemented. \
     Please use VolumeBased or EventBased sampling instead.")
```

**Documentation Impact**: The enum variant exists and appears in documentation, but users cannot
use it. Documentation should be updated to mark this as "planned" or "not implemented".

**Workaround**: Use `VolumeBased` (recommended) or `EventBased` sampling.

**Files to Update When Implementing**:
- `src/pipeline.rs` - Add TimeBasedSampler handling
- `src/preprocessing/sampling.rs` - Implement TimeBasedSampler struct
- `docs/USAGE_GUIDE.md` - Update sampling strategies section
- `CODEBASE.md` - Update sampling documentation

---

### 1.2 median_order_lifetime() Feature (P3)

**Status**: ⚠️ Placeholder implementation (always returns 0.0)

**Location**: `src/features/mbo_features.rs:1397-1400`

```rust
fn median_order_lifetime(&self) -> f64 {
    // Placeholder: track completed orders
    0.0
}
```

**Feature Index**: 79 (within MBO features block, absolute index in 98-feature mode)

**Impact**: MBO feature at index 79 is always 0.0, reducing feature set effectiveness.
This does NOT affect TLOB training (TLOB uses 40 features without MBO).

**Implementation Notes**:
- Requires tracking completed orders (currently only active orders are tracked)
- Would need a completed orders buffer with bounded size
- Median computation on every extract_features() call could be expensive

**Workaround**: Feature can be excluded from model training or treated as constant.

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

### 3.2 Triple Barrier and Magnitude Export Integration (P2)

**Status**: ⚠️ API implemented, export_dataset integration pending

**Location**: `src/labeling/triple_barrier.rs`, `src/labeling/magnitude.rs`

**Current State**:
- ✅ Labelers fully implemented and tested
- ❌ NOT integrated into `export_dataset` CLI tool
- ❌ Cannot be configured via TOML

**Tracking**: Documented in `docs/LABELING_STRATEGIES.md#integration-status`

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

**Current State**: Unit tests exist for individual component resets, but no integration test
verifies that `Pipeline::reset()` properly clears ALL state for multi-day processing.

**Recommended**: Add integration test that:
1. Processes Day 1
2. Calls `pipeline.reset()`
3. Processes Day 2
4. Verifies no state leakage

---

## Resolution Tracking

| ID | Item | Status | Resolved Date |
|----|------|--------|---------------|
| 2.1 | FI-2010 preset docstring | ✅ Fixed | 2026-01-14 |
| 1.1 | TimeBased sampling | ⏳ Documented | - |
| 2.2 | Missing citations | ⏳ Pending | - |
| 1.2 | median_order_lifetime | ⏳ Documented | - |
| 1.3 | FI-2010 regression features | ⏳ Documented | - |
| 3.1 | fi2010.rs integration | ⏳ Documented | - |

---

*Last Updated: 2026-01-14*
