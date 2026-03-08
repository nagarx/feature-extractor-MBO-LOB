//! Golden test: validates Rust constants in `contract.rs` match `pipeline_contract.toml`.
//!
//! This integration test reads `contracts/pipeline_contract.toml` at test time
//! and asserts every Rust-side constant matches the TOML definition.
//! If the TOML is not found (standalone build), the test is skipped.

use feature_extractor::contract;
use std::path::Path;

fn find_contract_toml() -> Option<String> {
    let candidates = [
        "../../contracts/pipeline_contract.toml",
        "../contracts/pipeline_contract.toml",
        "contracts/pipeline_contract.toml",
    ];
    for path in &candidates {
        if Path::new(path).exists() {
            return Some(std::fs::read_to_string(path).unwrap());
        }
    }
    None
}

fn parse_toml() -> Option<toml::Value> {
    let content = find_contract_toml()?;
    Some(content.parse::<toml::Value>().expect("Invalid TOML in pipeline_contract.toml"))
}

// =============================================================================
// Contract version
// =============================================================================

#[test]
fn schema_version_matches_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => {
            eprintln!("SKIP: pipeline_contract.toml not found, skipping contract validation");
            return;
        }
    };

    let toml_version = toml["contract"]["schema_version"]
        .as_str()
        .expect("Missing [contract].schema_version in TOML");

    let rust_version = format!("{}", contract::SCHEMA_VERSION);
    assert_eq!(
        rust_version, toml_version,
        "contract::SCHEMA_VERSION ({}) != TOML [contract].schema_version ({})",
        rust_version, toml_version
    );

    assert_eq!(
        contract::SCHEMA_VERSION_STR, toml_version,
        "contract::SCHEMA_VERSION_STR ({}) != TOML [contract].schema_version ({})",
        contract::SCHEMA_VERSION_STR, toml_version
    );
}

// =============================================================================
// Feature counts
// =============================================================================

#[test]
fn stable_feature_count_matches_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let toml_val = toml["features"]["stable_count"]
        .as_integer()
        .expect("Missing [features].stable_count") as usize;

    assert_eq!(
        contract::STABLE_FEATURE_COUNT, toml_val,
        "contract::STABLE_FEATURE_COUNT ({}) != TOML [features].stable_count ({})",
        contract::STABLE_FEATURE_COUNT, toml_val
    );
}

#[test]
fn experimental_feature_count_matches_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let toml_val = toml["features"]["experimental_count"]
        .as_integer()
        .expect("Missing [features].experimental_count") as usize;

    assert_eq!(
        contract::EXPERIMENTAL_FEATURE_COUNT, toml_val,
        "contract::EXPERIMENTAL_FEATURE_COUNT ({}) != TOML [features].experimental_count ({})",
        contract::EXPERIMENTAL_FEATURE_COUNT, toml_val
    );
}

#[test]
fn full_feature_count_matches_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let toml_val = toml["features"]["full_count"]
        .as_integer()
        .expect("Missing [features].full_count") as usize;

    assert_eq!(
        contract::FULL_FEATURE_COUNT, toml_val,
        "contract::FULL_FEATURE_COUNT ({}) != TOML [features].full_count ({})",
        contract::FULL_FEATURE_COUNT, toml_val
    );

    assert_eq!(
        contract::FULL_FEATURE_COUNT,
        contract::STABLE_FEATURE_COUNT + contract::EXPERIMENTAL_FEATURE_COUNT,
        "FULL_FEATURE_COUNT ({}) != STABLE ({}) + EXPERIMENTAL ({})",
        contract::FULL_FEATURE_COUNT,
        contract::STABLE_FEATURE_COUNT,
        contract::EXPERIMENTAL_FEATURE_COUNT,
    );
}

#[test]
fn lob_levels_matches_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let toml_val = toml["features"]["lob_levels"]
        .as_integer()
        .expect("Missing [features].lob_levels") as usize;

    assert_eq!(
        contract::LOB_LEVELS, toml_val,
        "contract::LOB_LEVELS ({}) != TOML [features].lob_levels ({})",
        contract::LOB_LEVELS, toml_val
    );
}

// =============================================================================
// Signal count
// =============================================================================

#[test]
fn signal_count_matches_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let signals_table = toml["features"]["signals"]
        .as_table()
        .expect("Missing [features.signals] table");

    assert_eq!(
        contract::SIGNAL_COUNT,
        signals_table.len(),
        "contract::SIGNAL_COUNT ({}) != number of entries in [features.signals] ({})",
        contract::SIGNAL_COUNT,
        signals_table.len()
    );
}

// =============================================================================
// Categorical indices
// =============================================================================

#[test]
fn categorical_indices_match_toml() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let toml_indices: Vec<usize> = toml["features"]["categorical"]["indices"]
        .as_array()
        .expect("Missing [features.categorical].indices")
        .iter()
        .map(|v| v.as_integer().unwrap() as usize)
        .collect();

    assert_eq!(
        contract::CATEGORICAL_INDICES, &toml_indices[..],
        "contract::CATEGORICAL_INDICES ({:?}) != TOML [features.categorical].indices ({:?})",
        contract::CATEGORICAL_INDICES, toml_indices
    );
}

// =============================================================================
// Key signal index spot-checks
// =============================================================================

#[test]
fn signal_index_spot_checks() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let signals = toml["features"]["signals"]
        .as_table()
        .expect("Missing [features.signals] table");

    let true_ofi = signals["true_ofi"].as_integer().unwrap() as usize;
    assert_eq!(true_ofi, 84, "TOML true_ofi index should be 84, got {}", true_ofi);

    let schema_version_idx = signals["schema_version"].as_integer().unwrap() as usize;
    assert_eq!(
        schema_version_idx, 97,
        "TOML schema_version signal index should be 97, got {}",
        schema_version_idx
    );
}

// =============================================================================
// Feature index spot-checks across different groups
// =============================================================================

#[test]
fn derived_feature_index_spot_checks() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let derived = &toml["features"]["derived"];
    let mid_price = derived["mid_price"].as_integer().unwrap() as usize;
    assert_eq!(mid_price, 40, "TOML mid_price should be at index 40, got {}", mid_price);

    let spread = derived["spread"].as_integer().unwrap() as usize;
    assert_eq!(spread, 41, "TOML spread should be at index 41, got {}", spread);
}

#[test]
fn mbo_feature_index_spot_checks() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let core = &toml["features"]["mbo"]["core"];
    let active_order_count = core["active_order_count"].as_integer().unwrap() as usize;
    assert_eq!(
        active_order_count, 83,
        "TOML active_order_count should be at index 83, got {}",
        active_order_count
    );

    let order_flow = &toml["features"]["mbo"]["order_flow"];
    let net_order_flow = order_flow["net_order_flow"].as_integer().unwrap() as usize;
    assert_eq!(
        net_order_flow, 54,
        "TOML net_order_flow should be at index 54, got {}",
        net_order_flow
    );
}

// =============================================================================
// LOB layout consistency
// =============================================================================

#[test]
fn lob_layout_consistency() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let lob = &toml["features"]["lob"];
    let ask_prices_start = lob["ask_prices"]["start"].as_integer().unwrap() as usize;
    let ask_prices_count = lob["ask_prices"]["count"].as_integer().unwrap() as usize;
    let ask_sizes_start = lob["ask_sizes"]["start"].as_integer().unwrap() as usize;

    assert_eq!(
        ask_prices_start + ask_prices_count,
        ask_sizes_start,
        "ask_prices end ({}) should equal ask_sizes start ({})",
        ask_prices_start + ask_prices_count,
        ask_sizes_start
    );

    assert_eq!(
        ask_prices_count, contract::LOB_LEVELS,
        "LOB group count ({}) should equal LOB_LEVELS ({})",
        ask_prices_count, contract::LOB_LEVELS
    );
}
