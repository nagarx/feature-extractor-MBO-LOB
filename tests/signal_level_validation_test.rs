//! Test to expose and validate the signal layer's 10-level requirement.
//!
//! ISSUE: Signal indices are hardcoded for 10 LOB levels:
//! - derived_indices::MID_PRICE = 40 (assumes 10 levels × 4 = 40 raw features)
//! - mbo_indices::CANCEL_RATE_BID = 50 (assumes MBO starts at 48)
//!
//! If lob_levels ≠ 10, signals will read from wrong feature indices,
//! causing silent data corruption.
//!
//! This test validates that the config properly rejects non-10-level
//! configurations when signals are enabled.

use feature_extractor::features::FeatureConfig;

/// Test that signals are only allowed with 10 levels.
///
/// Signal indices are hardcoded for the 10-level layout:
/// - Raw LOB: 0-39 (10 × 4 = 40)
/// - Derived: 40-47 (8)
/// - MBO: 48-83 (36)
/// - Signals: 84-97 (14)
///
/// With 5 levels:
/// - Raw LOB: 0-19 (5 × 4 = 20)
/// - Derived: 20-27 (8)
/// - MBO: 28-63 (36)
/// - Signal indices 40, 50, 71 would point to wrong features!
#[test]
fn test_signals_require_10_levels() {
    // 10 levels with signals: should be valid
    let config_10 = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: true,
    };
    assert!(
        config_10.validate().is_ok(),
        "10 levels with signals should be valid"
    );

    // 5 levels with signals: should be REJECTED
    let config_5 = FeatureConfig {
        lob_levels: 5,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: true,
    };
    let result = config_5.validate();
    assert!(
        result.is_err(),
        "5 levels with signals should be rejected! Signal indices are hardcoded for 10 levels. \
         Got: {:?}",
        result
    );

    // Check error message mentions the constraint
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.to_lowercase().contains("10") || err_msg.to_lowercase().contains("level"),
        "Error message should mention the 10-level requirement. Got: {}",
        err_msg
    );
}

/// Test that 5 levels without signals is still valid.
#[test]
fn test_non_10_levels_without_signals_is_valid() {
    let config = FeatureConfig {
        lob_levels: 5,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: false, // No signals
    };
    assert!(
        config.validate().is_ok(),
        "5 levels without signals should be valid"
    );
}

/// Test that 20 levels with signals is also rejected.
#[test]
fn test_20_levels_with_signals_rejected() {
    let config = FeatureConfig {
        lob_levels: 20,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: true,
    };
    assert!(
        config.validate().is_err(),
        "20 levels with signals should be rejected"
    );
}

/// Test that exactly 10 levels is required for signals.
#[test]
fn test_exactly_10_levels_required_for_signals() {
    for levels in [1, 5, 9, 11, 15, 20, 50] {
        let config = FeatureConfig {
            lob_levels: levels,
            tick_size: 0.01,
            include_derived: true,
            include_mbo: true,
            mbo_window_size: 1000,
            include_signals: true,
        };
        assert!(
            config.validate().is_err(),
            "lob_levels={} with signals should be rejected (only 10 is valid)",
            levels
        );
    }

    // Only 10 should work
    let config_10 = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: true,
    };
    assert!(
        config_10.validate().is_ok(),
        "lob_levels=10 with signals should be valid"
    );
}

/// Document the index layout for 10 levels.
///
/// This test serves as documentation for the expected feature layout.
#[test]
fn test_feature_layout_documentation() {
    let config = FeatureConfig {
        lob_levels: 10,
        tick_size: 0.01,
        include_derived: true,
        include_mbo: true,
        mbo_window_size: 1000,
        include_signals: true,
    };

    // Verify expected layout
    let raw_lob = 10 * 4; // 40
    let derived = 8;
    let mbo = 36;
    let signals = 14;
    let total = raw_lob + derived + mbo + signals; // 98

    assert_eq!(
        config.feature_count(),
        total,
        "Expected {} features, got {}",
        total,
        config.feature_count()
    );

    // Document expected index ranges
    // Raw LOB: 0-39
    // Derived: 40-47
    // MBO: 48-83
    // Signals: 84-97
    assert_eq!(raw_lob, 40, "Raw LOB should be 40 features");
    assert_eq!(raw_lob + derived, 48, "Derived should end at 47 (48 total)");
    assert_eq!(raw_lob + derived + mbo, 84, "MBO should end at 83 (84 total)");
    assert_eq!(total, 98, "Total with signals should be 98");
}

