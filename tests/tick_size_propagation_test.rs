//! Test to verify tick_size propagation from SymbolConfig through to FeatureConfig.
//!
//! This test exposes the bug where tick_size configured in SymbolConfig is ignored
//! and a hardcoded value of 0.01 is used instead.

use feature_extractor::export::DatasetConfig;

#[test]
fn test_tick_size_propagates_from_symbol_config() {
    // Create a config with a custom tick_size (e.g., 0.001 for crypto or JPY pairs)
    let toml_content = r#"
[symbol]
name = "BTCUSD"
exchange = "BINANCE"
filename_pattern = "binance-{date}.mbo.dbn.zst"
tick_size = 0.001  # Custom tick_size for crypto

[data]
input_dir = "/tmp/data"
output_dir = "/tmp/output"

[dates]
start_date = "2025-01-01"
end_date = "2025-01-31"

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true
mbo_window_size = 1000

[sampling]
volume_threshold = 1000
time_interval_ns = 1000000

[sequence]
window_size = 100
stride = 10

[labels]
horizon_seconds = 10.0
threshold_bps = 1.0

[split]
train_pct = 0.7
val_pct = 0.15
test_pct = 0.15

[processing]
num_threads = 4
verbose = false
"#;

    // Parse the config
    let config: DatasetConfig = toml::from_str(toml_content).expect("Failed to parse TOML");
    
    // Verify the symbol config has the custom tick_size
    assert_eq!(
        config.symbol.tick_size, 0.001,
        "SymbolConfig should have tick_size = 0.001"
    );
    
    // Convert to PipelineConfig
    let pipeline_config = config.to_pipeline_config();
    
    // THE BUG: tick_size in FeatureConfig should be 0.001, not 0.01
    assert_eq!(
        pipeline_config.features.tick_size, 0.001,
        "BUG EXPOSED: FeatureConfig should use tick_size from SymbolConfig (0.001), \
         but got {} (likely hardcoded 0.01). \
         This means custom tick_size values are being ignored!",
        pipeline_config.features.tick_size
    );
}

#[test]
fn test_default_tick_size_is_001() {
    // Verify that the default tick_size is 0.01 (for US stocks)
    let toml_content = r#"
[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
# tick_size not specified, should default to 0.01

[data]
input_dir = "/tmp/data"
output_dir = "/tmp/output"

[dates]
start_date = "2025-01-01"
end_date = "2025-01-31"

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true
mbo_window_size = 1000

[sampling]
volume_threshold = 1000
time_interval_ns = 1000000

[sequence]
window_size = 100
stride = 10

[labels]
horizon_seconds = 10.0
threshold_bps = 1.0

[split]
train_pct = 0.7
val_pct = 0.15
test_pct = 0.15

[processing]
num_threads = 4
verbose = false
"#;

    let config: DatasetConfig = toml::from_str(toml_content).expect("Failed to parse TOML");
    
    // Default should be 0.01
    assert_eq!(
        config.symbol.tick_size, 0.01,
        "Default tick_size should be 0.01"
    );
    
    let pipeline_config = config.to_pipeline_config();
    
    // Should propagate the default
    assert_eq!(
        pipeline_config.features.tick_size, 0.01,
        "FeatureConfig should have default tick_size = 0.01"
    );
}

#[test]
fn test_tick_size_various_values() {
    // Test various tick_size values for different asset classes
    let test_cases = vec![
        (0.01, "US stocks (cents)"),
        (0.001, "Crypto (sub-cent)"),
        (0.0001, "Forex (pip)"),
        (1.0, "Futures (whole units)"),
        (0.05, "Some ETFs"),
    ];
    
    for (tick_size, description) in test_cases {
        let toml_content = format!(r#"
[symbol]
name = "TEST"
exchange = "TEST"
filename_pattern = "test-{{date}}.mbo.dbn.zst"
tick_size = {}

[data]
input_dir = "/tmp/data"
output_dir = "/tmp/output"

[dates]
start_date = "2025-01-01"
end_date = "2025-01-31"

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true
mbo_window_size = 1000

[sampling]
volume_threshold = 1000
time_interval_ns = 1000000

[sequence]
window_size = 100
stride = 10

[labels]
horizon_seconds = 10.0
threshold_bps = 1.0

[split]
train_pct = 0.7
val_pct = 0.15
test_pct = 0.15

[processing]
num_threads = 4
verbose = false
"#, tick_size);
        
        let config: DatasetConfig = toml::from_str(&toml_content)
            .expect(&format!("Failed to parse TOML for {}", description));
        
        let pipeline_config = config.to_pipeline_config();
        
        assert!(
            (pipeline_config.features.tick_size - tick_size).abs() < 1e-10,
            "For {}: Expected tick_size = {}, got {}. \
             tick_size is not being propagated from SymbolConfig!",
            description, tick_size, pipeline_config.features.tick_size
        );
    }
}

