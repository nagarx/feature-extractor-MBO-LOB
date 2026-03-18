use super::*;

// ========================================================================
// ExportLabelConfig Tests
// ========================================================================

#[test]
fn test_label_config_default_is_single_horizon() {
    let config = ExportLabelConfig::default();
    assert!(
        !config.is_multi_horizon(),
        "Default should be single-horizon mode"
    );
    assert_eq!(config.horizon, 50);
    assert!(config.horizons.is_empty());
    assert_eq!(config.smoothing_window, 10);
    assert!((config.threshold - 0.0008).abs() < 1e-10);
}

#[test]
fn test_label_config_single_constructor() {
    let config = ExportLabelConfig::single(100, 20, 0.002);
    assert!(!config.is_multi_horizon());
    assert_eq!(config.horizon, 100);
    assert_eq!(config.smoothing_window, 20);
    assert!((config.threshold - 0.002).abs() < 1e-10);
}

#[test]
fn test_label_config_multi_constructor() {
    let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.001);
    assert!(config.is_multi_horizon());
    assert_eq!(config.horizons, vec![10, 20, 50]);
    assert_eq!(config.smoothing_window, 5);
    assert!((config.threshold - 0.001).abs() < 1e-10);
    // max_horizon should be set as fallback single horizon
    assert_eq!(config.horizon, 50);
}

#[test]
fn test_label_config_fi2010_preset() {
    let config = ExportLabelConfig::fi2010();
    assert!(config.is_multi_horizon());
    assert_eq!(config.horizons, vec![10, 20, 30, 50, 100]);
    assert_eq!(config.smoothing_window, 5);
    assert!((config.threshold - 0.002).abs() < 1e-10);
}

#[test]
fn test_label_config_deeplob_preset() {
    let config = ExportLabelConfig::deeplob();
    assert!(config.is_multi_horizon());
    assert_eq!(config.horizons, vec![10, 20, 50, 100]);
    assert_eq!(config.smoothing_window, 5);
}

#[test]
fn test_label_config_effective_horizons_single() {
    let config = ExportLabelConfig::single(200, 50, 0.0008);
    let effective = config.effective_horizons();
    assert_eq!(effective, vec![200]);
}

#[test]
fn test_label_config_effective_horizons_multi() {
    let config = ExportLabelConfig::multi(vec![10, 20, 100], 5, 0.002);
    let effective = config.effective_horizons();
    assert_eq!(effective, vec![10, 20, 100]);
}

#[test]
fn test_label_config_max_horizon_single() {
    let config = ExportLabelConfig::single(150, 30, 0.001);
    assert_eq!(config.max_horizon(), 150);
}

#[test]
fn test_label_config_max_horizon_multi() {
    let config = ExportLabelConfig::multi(vec![10, 50, 200, 100], 5, 0.002);
    assert_eq!(config.max_horizon(), 200);
}

#[test]
fn test_label_config_to_label_config_single() {
    let config = ExportLabelConfig::single(100, 20, 0.003);
    let label_config = config.to_label_config();
    assert_eq!(label_config.horizon, 100);
    assert_eq!(label_config.smoothing_window, 20);
    assert!((label_config.threshold - 0.003).abs() < 1e-10);
}

#[test]
fn test_label_config_to_label_config_multi_uses_first_horizon() {
    // When converting multi-horizon to single LabelConfig, use first horizon
    let config = ExportLabelConfig::multi(vec![10, 50, 100], 5, 0.002);
    let label_config = config.to_label_config();
    assert_eq!(
        label_config.horizon, 10,
        "Should use first horizon for single-horizon conversion"
    );
    assert_eq!(label_config.smoothing_window, 5);
}

#[test]
fn test_label_config_to_multi_horizon_config_single_returns_none() {
    let config = ExportLabelConfig::single(100, 20, 0.002);
    let multi_config = config.to_multi_horizon_config();
    assert!(
        multi_config.is_none(),
        "Single-horizon config should return None for multi-horizon conversion"
    );
}

#[test]
fn test_label_config_to_multi_horizon_config_multi_returns_some() {
    let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.002);
    let multi_config = config.to_multi_horizon_config();
    assert!(multi_config.is_some());

    let mc = multi_config.unwrap();
    assert_eq!(mc.horizons(), &[10, 20, 50]);
    assert_eq!(mc.smoothing_window, 5);
}

#[test]
fn test_label_config_validation_single_valid() {
    let config = ExportLabelConfig::single(100, 20, 0.002);
    assert!(config.validate().is_ok());
}

#[test]
fn test_label_config_validation_multi_valid() {
    let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.002);
    assert!(config.validate().is_ok());
}

#[test]
fn test_label_config_validation_single_zero_horizon() {
    let config = ExportLabelConfig::single(0, 5, 0.002);
    let result = config.validate();
    assert!(result.is_err());
    assert!(
        result.unwrap_err().contains("horizon must be > 0"),
        "Should reject zero horizon"
    );
}

#[test]
fn test_label_config_validation_multi_zero_horizon_in_array() {
    let config = ExportLabelConfig::multi(vec![10, 0, 50], 5, 0.002);
    let result = config.validate();
    assert!(result.is_err());
    assert!(
        result.unwrap_err().contains("All horizons must be > 0"),
        "Should reject zero horizon in array"
    );
}

#[test]
fn test_label_config_validation_zero_smoothing() {
    let config = ExportLabelConfig::single(100, 0, 0.002);
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("smoothing_window must be > 0"));
}

#[test]
fn test_label_config_validation_smoothing_greater_than_horizon() {
    let config = ExportLabelConfig::single(10, 20, 0.002);
    let result = config.validate();
    assert!(result.is_err());
    // Updated to match new error message format
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("smoothing_window")
            && err_msg.contains("should be <= minimum horizon")
    );
}

#[test]
fn test_label_config_validation_multi_smoothing_greater_than_min_horizon() {
    // smoothing=20, but min horizon is 10
    let config = ExportLabelConfig::multi(vec![10, 50, 100], 20, 0.002);
    let result = config.validate();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("smoothing_window (20) should be <= minimum horizon (10)"),
        "Should reject smoothing > min horizon"
    );
}

#[test]
fn test_label_config_validation_zero_threshold() {
    let config = ExportLabelConfig::single(100, 10, 0.0);
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("threshold must be > 0"));
}

#[test]
fn test_label_config_validation_negative_threshold() {
    let config = ExportLabelConfig::single(100, 10, -0.001);
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("threshold must be > 0"));
}

#[test]
fn test_label_config_validation_threshold_too_large() {
    let config = ExportLabelConfig::single(100, 10, 0.15);
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("threshold seems too large"));
}

#[test]
fn test_label_config_toml_serialization_single() {
    let config = ExportLabelConfig::single(100, 20, 0.0008);
    let toml_str = toml::to_string(&config).unwrap();

    // Should contain single horizon, not horizons array
    assert!(toml_str.contains("horizon = 100"));
    assert!(toml_str.contains("smoothing_window = 20"));
    assert!(
        !toml_str.contains("horizons"),
        "Empty horizons should be skipped"
    );
}

#[test]
fn test_label_config_toml_serialization_multi() {
    let config = ExportLabelConfig::multi(vec![10, 20, 50], 5, 0.002);
    let toml_str = toml::to_string(&config).unwrap();

    // Should contain horizons array
    assert!(toml_str.contains("horizons = [10, 20, 50]"));
    assert!(toml_str.contains("smoothing_window = 5"));
}

#[test]
fn test_label_config_toml_deserialization_single() {
    let toml_str = r#"
horizon = 200
smoothing_window = 50
threshold = 0.0008
"#;
    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
    assert!(!config.is_multi_horizon());
    assert_eq!(config.horizon, 200);
    assert_eq!(config.smoothing_window, 50);
    assert!((config.threshold - 0.0008).abs() < 1e-10);
}

#[test]
fn test_label_config_toml_deserialization_multi() {
    let toml_str = r#"
horizons = [10, 20, 50, 100, 200]
smoothing_window = 10
threshold = 0.0008
"#;
    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
    assert!(config.is_multi_horizon());
    assert_eq!(config.horizons, vec![10, 20, 50, 100, 200]);
    assert_eq!(config.smoothing_window, 10);
}

#[test]
fn test_label_config_toml_roundtrip_single() {
    let original = ExportLabelConfig::single(100, 20, 0.0015);
    let toml_str = toml::to_string(&original).unwrap();
    let loaded: ExportLabelConfig = toml::from_str(&toml_str).unwrap();

    assert_eq!(original.horizon, loaded.horizon);
    assert_eq!(original.smoothing_window, loaded.smoothing_window);
    assert!((original.threshold - loaded.threshold).abs() < 1e-10);
    assert_eq!(original.is_multi_horizon(), loaded.is_multi_horizon());
}

#[test]
fn test_label_config_toml_roundtrip_multi() {
    let original = ExportLabelConfig::multi(vec![10, 20, 50, 100], 5, 0.002);
    let toml_str = toml::to_string(&original).unwrap();
    let loaded: ExportLabelConfig = toml::from_str(&toml_str).unwrap();

    assert_eq!(original.horizons, loaded.horizons);
    assert_eq!(original.smoothing_window, loaded.smoothing_window);
    assert!((original.threshold - loaded.threshold).abs() < 1e-10);
    assert_eq!(original.is_multi_horizon(), loaded.is_multi_horizon());
}

#[test]
fn test_label_config_backward_compatibility() {
    // Old TOML format (single horizon only) should still work
    let old_format = r#"
horizon = 50
smoothing_window = 10
threshold = 0.0008
"#;
    let config: ExportLabelConfig = toml::from_str(old_format).unwrap();
    assert!(!config.is_multi_horizon());
    assert_eq!(config.horizon, 50);
    assert!(config.validate().is_ok());
}

// ========================================================================
// TlobDynamic Threshold Strategy Tests
// ========================================================================

#[test]
fn test_threshold_strategy_tlob_dynamic_creation() {
    let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
    assert!(matches!(
        strategy,
        ExportThresholdStrategy::TlobDynamic { fallback, divisor }
        if (fallback - 0.0008).abs() < 1e-10 && (divisor - 2.0).abs() < 1e-10
    ));

    // Test default divisor helper
    let strategy_default = ExportThresholdStrategy::tlob_dynamic_default(0.001);
    assert!(matches!(
        strategy_default,
        ExportThresholdStrategy::TlobDynamic { fallback, divisor }
        if (fallback - 0.001).abs() < 1e-10 && (divisor - 2.0).abs() < 1e-10
    ));
}

#[test]
fn test_threshold_strategy_tlob_dynamic_to_internal() {
    let export_strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.5);
    let internal = export_strategy.to_internal();

    assert!(matches!(
        internal,
        crate::labeling::ThresholdStrategy::TlobDynamic { fallback, divisor }
        if (fallback - 0.0008).abs() < 1e-10 && (divisor - 2.5).abs() < 1e-10
    ));
}

#[test]
fn test_threshold_strategy_tlob_dynamic_validation() {
    // Valid
    let valid = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
    assert!(valid.validate().is_ok());

    // Invalid fallback
    let invalid_fallback = ExportThresholdStrategy::tlob_dynamic(0.0, 2.0);
    assert!(invalid_fallback.validate().is_err());

    // Invalid divisor
    let invalid_divisor = ExportThresholdStrategy::tlob_dynamic(0.0008, 0.0);
    assert!(invalid_divisor.validate().is_err());
}

#[test]
fn test_threshold_strategy_tlob_dynamic_description() {
    let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
    let desc = strategy.description();
    assert!(desc.contains("TLOB Dynamic"));
    assert!(desc.contains("2"));
}

#[test]
fn test_threshold_strategy_tlob_dynamic_needs_global() {
    let fixed = ExportThresholdStrategy::fixed(0.002);
    let rolling = ExportThresholdStrategy::rolling_spread(100, 1.0, 0.002);
    let quantile = ExportThresholdStrategy::quantile(0.33, 5000, 0.002);
    let tlob_dynamic = ExportThresholdStrategy::tlob_dynamic_default(0.002);

    assert!(!fixed.needs_global_computation());
    assert!(!rolling.needs_global_computation());
    assert!(!quantile.needs_global_computation());
    assert!(
        tlob_dynamic.needs_global_computation(),
        "TlobDynamic should need global computation"
    );
}

#[test]
fn test_threshold_strategy_tlob_dynamic_toml_serialization() {
    let strategy = ExportThresholdStrategy::tlob_dynamic(0.0008, 2.0);
    let toml_str = toml::to_string(&strategy).expect("Should serialize TlobDynamic");

    // Verify it contains expected fields
    assert!(toml_str.contains("tlob_dynamic") || toml_str.contains("type"));
    assert!(toml_str.contains("fallback"));
    assert!(toml_str.contains("divisor"));
}

#[test]
fn test_threshold_strategy_tlob_dynamic_toml_deserialization() {
    let toml_str = r#"
        type = "tlob_dynamic"
        fallback = 0.0008
        divisor = 2.0
    "#;

    let strategy: ExportThresholdStrategy =
        toml::from_str(toml_str).expect("Should deserialize TlobDynamic");

    assert!(matches!(
        strategy,
        ExportThresholdStrategy::TlobDynamic { fallback, divisor }
        if (fallback - 0.0008).abs() < 1e-10 && (divisor - 2.0).abs() < 1e-10
    ));
}

#[test]
fn test_threshold_strategy_tlob_dynamic_toml_roundtrip() {
    let original = ExportThresholdStrategy::tlob_dynamic(0.0005, 3.0);
    let toml_str = toml::to_string(&original).expect("Should serialize");
    let loaded: ExportThresholdStrategy =
        toml::from_str(&toml_str).expect("Should deserialize");

    assert_eq!(original, loaded);
}

#[test]
fn test_label_config_with_tlob_dynamic_strategy() {
    let config = ExportLabelConfig::multi_with_strategy(
        vec![10, 20, 50],
        5,
        ExportThresholdStrategy::tlob_dynamic_default(0.0008),
    );

    assert!(config.is_multi_horizon());
    assert_eq!(config.horizons.len(), 3);

    // Check the threshold strategy was set correctly
    let thresh_strategy = config.effective_threshold_strategy();
    assert!(matches!(
        thresh_strategy,
        ExportThresholdStrategy::TlobDynamic { divisor, .. }
        if (divisor - 2.0).abs() < 1e-10
    ));
}

// ========================================================================
// Regression Config Tests
// ========================================================================

#[test]
fn test_to_regression_config_returns_some_for_regression_strategy() {
    let mut config = ExportLabelConfig::multi(vec![10, 60, 300], 10, 0.0008);
    config.strategy = LabelingStrategy::Regression;
    let reg = config.to_regression_config();
    assert!(reg.is_some(), "Regression strategy with horizons should produce a config");
    let reg = reg.unwrap();
    assert_eq!(reg.multi_horizon_config.horizons(), &[10, 60, 300]);
    assert_eq!(reg.return_type, RegressionReturnType::SmoothedReturn);
}

#[test]
fn test_to_regression_config_returns_none_for_classification() {
    let config = ExportLabelConfig::multi(vec![10, 60, 300], 10, 0.0008);
    assert_eq!(config.strategy, LabelingStrategy::Tlob);
    assert!(
        config.to_regression_config().is_none(),
        "TLOB strategy should not produce a regression config"
    );
}

#[test]
fn test_to_regression_config_single_horizon_regression() {
    let mut config = ExportLabelConfig::single(60, 10, 0.0008);
    config.strategy = LabelingStrategy::Regression;
    let reg = config.to_regression_config();
    assert!(
        reg.is_some(),
        "Single-horizon regression should work by synthesizing horizons"
    );
    let reg = reg.unwrap();
    assert_eq!(
        reg.multi_horizon_config.horizons(),
        &[60],
        "Should synthesize horizons from single horizon field"
    );
}

#[test]
fn test_to_regression_config_with_return_type() {
    let mut config = ExportLabelConfig::multi(vec![10, 60], 10, 0.0008);
    config.strategy = LabelingStrategy::Regression;
    config.return_type = Some(RegressionReturnType::PointReturn);
    let reg = config.to_regression_config().unwrap();
    assert_eq!(reg.return_type, RegressionReturnType::PointReturn);
}

#[test]
fn test_to_regression_config_default_return_type() {
    let mut config = ExportLabelConfig::multi(vec![10, 60], 10, 0.0008);
    config.strategy = LabelingStrategy::Regression;
    // return_type is None, should default to SmoothedReturn
    let reg = config.to_regression_config().unwrap();
    assert_eq!(reg.return_type, RegressionReturnType::SmoothedReturn);
}

#[test]
fn test_regression_validation_requires_smoothing() {
    let mut config = ExportLabelConfig::multi(vec![10, 60], 0, 0.0008);
    config.strategy = LabelingStrategy::Regression;
    let result = config.validate();
    assert!(
        result.is_err(),
        "Regression requires smoothing_window > 0"
    );
    assert!(result.unwrap_err().contains("smoothing_window"));
}

#[test]
fn test_regression_strategy_description() {
    let mut config = ExportLabelConfig::multi(vec![10, 60, 300], 10, 0.0008);
    config.strategy = LabelingStrategy::Regression;
    let desc = config.description();
    assert!(desc.contains("Regression"), "Description should mention Regression: {desc}");
    assert!(desc.contains("continuous bps"), "Description should mention continuous bps: {desc}");
}

#[test]
fn test_regression_return_type_serde_roundtrip() {
    let types = [
        RegressionReturnType::SmoothedReturn,
        RegressionReturnType::PointReturn,
        RegressionReturnType::PeakReturn,
        RegressionReturnType::MeanReturn,
        RegressionReturnType::DominantReturn,
    ];
    for rt in &types {
        let json = serde_json::to_string(rt).unwrap();
        let deserialized: RegressionReturnType = serde_json::from_str(&json).unwrap();
        assert_eq!(*rt, deserialized, "Serde roundtrip failed for {:?}", rt);
    }
}

#[test]
fn test_regression_toml_config_parsing() {
    let toml_str = r#"
        strategy = "regression"
        horizon = 60
        horizons = [10, 60, 300]
        smoothing_window = 10
        threshold = 0.0008
        return_type = "point_return"
    "#;
    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
    assert!(matches!(config.strategy, LabelingStrategy::Regression));
    assert_eq!(config.return_type, Some(RegressionReturnType::PointReturn));
    assert_eq!(config.horizons, vec![10, 60, 300]);
}

#[test]
fn test_regression_toml_config_default_return_type() {
    let toml_str = r#"
        strategy = "regression"
        horizons = [10, 60]
        smoothing_window = 10
        threshold = 0.0008
    "#;
    let config: ExportLabelConfig = toml::from_str(toml_str).unwrap();
    assert!(matches!(config.strategy, LabelingStrategy::Regression));
    assert_eq!(config.return_type, None, "Default should be None (resolves to SmoothedReturn)");
}

#[test]
fn test_regression_labeling_strategy_enum() {
    assert_eq!(
        LabelingStrategy::Regression.description(),
        "Regression (continuous bps returns)"
    );
    assert!(!LabelingStrategy::Regression.is_opportunity());
    assert!(!LabelingStrategy::Regression.is_triple_barrier());
}
