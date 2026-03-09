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
        "tests/fixtures/pipeline_contract.toml",
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
    Some(
        content
            .parse::<toml::Value>()
            .expect("Invalid TOML in pipeline_contract.toml"),
    )
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
        contract::SCHEMA_VERSION_STR,
        toml_version,
        "contract::SCHEMA_VERSION_STR ({}) != TOML [contract].schema_version ({})",
        contract::SCHEMA_VERSION_STR,
        toml_version
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
        contract::STABLE_FEATURE_COUNT,
        toml_val,
        "contract::STABLE_FEATURE_COUNT ({}) != TOML [features].stable_count ({})",
        contract::STABLE_FEATURE_COUNT,
        toml_val
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
        contract::EXPERIMENTAL_FEATURE_COUNT,
        toml_val,
        "contract::EXPERIMENTAL_FEATURE_COUNT ({}) != TOML [features].experimental_count ({})",
        contract::EXPERIMENTAL_FEATURE_COUNT,
        toml_val
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
        contract::FULL_FEATURE_COUNT,
        toml_val,
        "contract::FULL_FEATURE_COUNT ({}) != TOML [features].full_count ({})",
        contract::FULL_FEATURE_COUNT,
        toml_val
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
        contract::LOB_LEVELS,
        toml_val,
        "contract::LOB_LEVELS ({}) != TOML [features].lob_levels ({})",
        contract::LOB_LEVELS,
        toml_val
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
        contract::CATEGORICAL_INDICES,
        &toml_indices[..],
        "contract::CATEGORICAL_INDICES ({:?}) != TOML [features.categorical].indices ({:?})",
        contract::CATEGORICAL_INDICES,
        toml_indices
    );
}

// =============================================================================
// Helper: collect all integer-valued entries from a TOML table
// =============================================================================

fn collect_indices(table: &toml::value::Table, group: &str) -> Vec<(String, usize)> {
    table
        .iter()
        .filter_map(|(name, val)| {
            val.as_integer()
                .map(|idx| (format!("{group}.{name}"), idx as usize))
        })
        .collect()
}

fn collect_indices_recursive(table: &toml::value::Table, prefix: &str) -> Vec<(String, usize)> {
    let mut result = Vec::new();
    for (key, val) in table {
        let full_key = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match val {
            toml::Value::Integer(idx) => result.push((full_key, *idx as usize)),
            toml::Value::Table(sub) => result.extend(collect_indices_recursive(sub, &full_key)),
            _ => {}
        }
    }
    result
}

// =============================================================================
// Exhaustive signal index validation
// =============================================================================

#[test]
fn signal_indices_exhaustive() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let signals = toml["features"]["signals"]
        .as_table()
        .expect("Missing [features.signals] table");

    let entries = collect_indices(signals, "signals");
    assert_eq!(
        entries.len(),
        contract::SIGNAL_COUNT,
        "Signal entries in TOML ({}) != contract::SIGNAL_COUNT ({})",
        entries.len(),
        contract::SIGNAL_COUNT
    );

    for (name, idx) in &entries {
        assert!(
            (84..98).contains(idx),
            "Signal '{name}' has index {idx}, expected range [84, 97]"
        );
    }
}

// =============================================================================
// Exhaustive derived feature index validation
// =============================================================================

#[test]
fn derived_feature_indices_exhaustive() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let derived = toml["features"]["derived"]
        .as_table()
        .expect("Missing [features.derived] table");

    let entries = collect_indices(derived, "derived");
    assert_eq!(
        entries.len(),
        8,
        "Derived feature entries in TOML ({}) != expected 8",
        entries.len()
    );

    for (name, idx) in &entries {
        assert!(
            (40..48).contains(idx),
            "Derived feature '{name}' has index {idx}, expected range [40, 47]"
        );
    }
}

// =============================================================================
// Exhaustive MBO feature index validation
// =============================================================================

#[test]
fn mbo_feature_indices_exhaustive() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let mbo = toml["features"]["mbo"]
        .as_table()
        .expect("Missing [features.mbo] table");

    let entries = collect_indices_recursive(mbo, "mbo");
    assert_eq!(
        entries.len(),
        36,
        "MBO feature entries in TOML ({}) != expected 36. Entries: {:?}",
        entries.len(),
        entries.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
    );

    for (name, idx) in &entries {
        assert!(
            (48..84).contains(idx),
            "MBO feature '{name}' has index {idx}, expected range [48, 83]"
        );
    }
}

// =============================================================================
// Exhaustive experimental feature index validation
// =============================================================================

#[test]
fn experimental_feature_indices_exhaustive() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let experimental = toml["features"]["experimental"]
        .as_table()
        .expect("Missing [features.experimental] table");

    let entries = collect_indices_recursive(experimental, "experimental");
    assert_eq!(
        entries.len(),
        contract::EXPERIMENTAL_FEATURE_COUNT,
        "Experimental feature entries in TOML ({}) != contract::EXPERIMENTAL_FEATURE_COUNT ({}). Entries: {:?}",
        entries.len(),
        contract::EXPERIMENTAL_FEATURE_COUNT,
        entries.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
    );

    for (name, idx) in &entries {
        assert!(
            (98..116).contains(idx),
            "Experimental feature '{name}' has index {idx}, expected range [98, 115]"
        );
    }
}

// =============================================================================
// Cross-group uniqueness and coverage
// =============================================================================

#[test]
fn all_feature_indices_unique_and_complete() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let mut all_indices: std::collections::BTreeMap<usize, String> =
        std::collections::BTreeMap::new();

    let lob = toml["features"]["lob"]
        .as_table()
        .expect("Missing [features.lob]");
    for (group_name, group_val) in lob {
        let start = group_val["start"]
            .as_integer()
            .unwrap_or_else(|| panic!("Missing start in lob.{group_name}"))
            as usize;
        let count = group_val["count"]
            .as_integer()
            .unwrap_or_else(|| panic!("Missing count in lob.{group_name}"))
            as usize;
        for i in start..start + count {
            let name = format!("lob.{group_name}[{i}]");
            if let Some(existing) = all_indices.insert(i, name.clone()) {
                panic!("Duplicate index {i}: '{existing}' and '{name}'");
            }
        }
    }

    let derived = toml["features"]["derived"]
        .as_table()
        .expect("Missing [features.derived]");
    for (name, idx) in collect_indices(derived, "derived") {
        if let Some(existing) = all_indices.insert(idx, name.clone()) {
            panic!("Duplicate index {idx}: '{existing}' and '{name}'");
        }
    }

    let mbo = toml["features"]["mbo"]
        .as_table()
        .expect("Missing [features.mbo]");
    for (name, idx) in collect_indices_recursive(mbo, "mbo") {
        if let Some(existing) = all_indices.insert(idx, name.clone()) {
            panic!("Duplicate index {idx}: '{existing}' and '{name}'");
        }
    }

    let signals = toml["features"]["signals"]
        .as_table()
        .expect("Missing [features.signals]");
    for (name, idx) in collect_indices(signals, "signals") {
        if let Some(existing) = all_indices.insert(idx, name.clone()) {
            panic!("Duplicate index {idx}: '{existing}' and '{name}'");
        }
    }

    for i in 0..contract::STABLE_FEATURE_COUNT {
        assert!(
            all_indices.contains_key(&i),
            "Stable feature index {i} is not assigned to any feature in TOML"
        );
    }

    let experimental = toml["features"]["experimental"]
        .as_table()
        .expect("Missing [features.experimental]");
    for (name, idx) in collect_indices_recursive(experimental, "experimental") {
        if let Some(existing) = all_indices.insert(idx, name.clone()) {
            panic!("Duplicate index {idx}: '{existing}' and '{name}'");
        }
    }

    for i in 0..contract::FULL_FEATURE_COUNT {
        assert!(
            all_indices.contains_key(&i),
            "Feature index {i} is not assigned to any feature in TOML (full range 0..{})",
            contract::FULL_FEATURE_COUNT
        );
    }

    assert_eq!(
        all_indices.len(),
        contract::FULL_FEATURE_COUNT,
        "Total unique indices ({}) != FULL_FEATURE_COUNT ({})",
        all_indices.len(),
        contract::FULL_FEATURE_COUNT
    );

    println!(
        "Contract validation: all {} feature indices are unique and complete (0..{})",
        all_indices.len(),
        contract::FULL_FEATURE_COUNT
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
        ask_prices_count,
        contract::LOB_LEVELS,
        "LOB group count ({}) should equal LOB_LEVELS ({})",
        ask_prices_count,
        contract::LOB_LEVELS
    );
}

// =============================================================================
// Label contract validation
// =============================================================================

fn validate_label_strategy(
    toml: &toml::Value,
    strategy_key: &str,
    expected_encoding: &str,
    expected_values: &[i64],
    expected_class_names: &[(&str, &str)],
    expected_num_classes: usize,
    expected_shift: bool,
) {
    let label = &toml["labels"][strategy_key];

    let encoding = label["encoding"]
        .as_str()
        .unwrap_or_else(|| panic!("Missing [labels.{strategy_key}].encoding"));
    assert_eq!(
        encoding, expected_encoding,
        "[labels.{strategy_key}].encoding: expected '{expected_encoding}', got '{encoding}'"
    );

    let values: Vec<i64> = label["values"]
        .as_array()
        .unwrap_or_else(|| panic!("Missing [labels.{strategy_key}].values"))
        .iter()
        .map(|v| v.as_integer().unwrap())
        .collect();
    assert_eq!(
        values, expected_values,
        "[labels.{strategy_key}].values mismatch"
    );

    let class_names = label["class_names"]
        .as_table()
        .unwrap_or_else(|| panic!("Missing [labels.{strategy_key}].class_names"));
    for (key, expected_name) in expected_class_names {
        let actual = class_names[*key]
            .as_str()
            .unwrap_or_else(|| panic!("Missing [labels.{strategy_key}].class_names.{key}"));
        assert_eq!(
            actual, *expected_name,
            "[labels.{strategy_key}].class_names.{key}: expected '{expected_name}', got '{actual}'"
        );
    }

    let num_classes = label["num_classes"]
        .as_integer()
        .unwrap_or_else(|| panic!("Missing [labels.{strategy_key}].num_classes"))
        as usize;
    assert_eq!(
        num_classes, expected_num_classes,
        "[labels.{strategy_key}].num_classes mismatch"
    );

    let shift = label["shift_for_crossentropy"]
        .as_bool()
        .unwrap_or_else(|| panic!("Missing [labels.{strategy_key}].shift_for_crossentropy"));
    assert_eq!(
        shift, expected_shift,
        "[labels.{strategy_key}].shift_for_crossentropy mismatch"
    );
}

#[test]
fn label_contract_tlob() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    validate_label_strategy(
        &toml,
        "tlob",
        "signed",
        &[-1, 0, 1],
        &[("-1", "Down"), ("0", "Stable"), ("1", "Up")],
        3,
        true,
    );
}

#[test]
fn label_contract_triple_barrier() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    validate_label_strategy(
        &toml,
        "triple_barrier",
        "class_index",
        &[0, 1, 2],
        &[("0", "StopLoss"), ("1", "Timeout"), ("2", "ProfitTarget")],
        3,
        false,
    );
}

#[test]
fn label_contract_opportunity() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    validate_label_strategy(
        &toml,
        "opportunity",
        "signed",
        &[-1, 0, 1],
        &[("-1", "BigDown"), ("0", "NoOpportunity"), ("1", "BigUp")],
        3,
        true,
    );
}

// =============================================================================
// Normalization contract validation
// =============================================================================

#[test]
fn normalization_contract_categorical_indices() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let norm_cat: Vec<usize> = toml["normalization"]["categorical_indices"]
        .as_array()
        .expect("Missing [normalization].categorical_indices")
        .iter()
        .map(|v| v.as_integer().unwrap() as usize)
        .collect();

    let feature_cat: Vec<usize> = toml["features"]["categorical"]["indices"]
        .as_array()
        .expect("Missing [features.categorical].indices")
        .iter()
        .map(|v| v.as_integer().unwrap() as usize)
        .collect();

    for idx in &feature_cat {
        assert!(
            norm_cat.contains(idx),
            "Feature categorical index {idx} is not in [normalization].categorical_indices"
        );
    }

    let exp_cat: Vec<usize> = toml["features"]["categorical"]["experimental_indices"]
        .as_array()
        .expect("Missing [features.categorical].experimental_indices")
        .iter()
        .map(|v| v.as_integer().unwrap() as usize)
        .collect();

    for idx in &exp_cat {
        assert!(
            norm_cat.contains(idx),
            "Experimental categorical index {idx} is not in [normalization].categorical_indices"
        );
    }

    let combined_len = feature_cat.len() + exp_cat.len();
    assert_eq!(
        norm_cat.len(),
        combined_len,
        "[normalization].categorical_indices has {} entries, \
         but [features.categorical] has {} stable + {} experimental = {} total",
        norm_cat.len(),
        feature_cat.len(),
        exp_cat.len(),
        combined_len
    );
}

#[test]
fn normalization_non_normalizable_superset_of_categorical() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let non_norm: Vec<usize> = toml["normalization"]["non_normalizable_indices"]
        .as_array()
        .expect("Missing [normalization].non_normalizable_indices")
        .iter()
        .map(|v| v.as_integer().unwrap() as usize)
        .collect();

    let cat: Vec<usize> = toml["normalization"]["categorical_indices"]
        .as_array()
        .expect("Missing [normalization].categorical_indices")
        .iter()
        .map(|v| v.as_integer().unwrap() as usize)
        .collect();

    for idx in &cat {
        assert!(
            non_norm.contains(idx),
            "Categorical index {idx} must be in non_normalizable_indices (cannot normalize categorical features)"
        );
    }

    for idx in &non_norm {
        assert!(
            *idx < contract::FULL_FEATURE_COUNT,
            "Non-normalizable index {idx} exceeds FULL_FEATURE_COUNT ({})",
            contract::FULL_FEATURE_COUNT
        );
    }
}

// =============================================================================
// TOML structural fingerprint (drift detection)
// =============================================================================

#[test]
fn toml_structural_fingerprint() {
    let toml = match parse_toml() {
        Some(v) => v,
        None => return,
    };

    let expected_top_level_sections = [
        "contract",
        "features",
        "labels",
        "normalization",
        "export",
        "changelog",
    ];
    let root = toml.as_table().expect("TOML root should be a table");

    for section in &expected_top_level_sections {
        assert!(
            root.contains_key(*section),
            "TOML missing expected top-level section: [{section}]"
        );
    }

    let features = toml["features"]
        .as_table()
        .expect("Missing [features] table");
    let expected_feature_groups = [
        "lob_levels",
        "stable_count",
        "experimental_count",
        "full_count",
        "lob",
        "derived",
        "mbo",
        "signals",
        "experimental",
        "categorical",
        "unsigned",
        "safety_gates",
        "primary_signals",
        "asymmetry_signals",
        "sign_convention",
        "layout",
    ];
    for group in &expected_feature_groups {
        assert!(
            features.contains_key(*group),
            "TOML [features] missing expected sub-key: '{group}'"
        );
    }

    let labels = toml["labels"].as_table().expect("Missing [labels] table");
    let expected_label_strategies = ["tlob", "triple_barrier", "opportunity"];
    for strategy in &expected_label_strategies {
        assert!(
            labels.contains_key(*strategy),
            "TOML [labels] missing expected strategy: '{strategy}'"
        );
    }

    let mbo_groups = toml["features"]["mbo"]
        .as_table()
        .expect("Missing [features.mbo]");
    let expected_mbo_groups = [
        "order_flow",
        "size_distribution",
        "queue_depth",
        "institutional",
        "core",
    ];
    for group in &expected_mbo_groups {
        assert!(
            mbo_groups.contains_key(*group),
            "TOML [features.mbo] missing expected sub-group: '{group}'"
        );
    }

    let experimental_groups = toml["features"]["experimental"]
        .as_table()
        .expect("Missing [features.experimental]");
    let expected_exp_groups = ["institutional_v2", "volatility", "seasonality"];
    for group in &expected_exp_groups {
        assert!(
            experimental_groups.contains_key(*group),
            "TOML [features.experimental] missing expected sub-group: '{group}'"
        );
    }

    println!(
        "TOML structural fingerprint validated: {} top-level sections, \
         {} feature groups, {} label strategies, {} MBO sub-groups, {} experimental sub-groups",
        expected_top_level_sections.len(),
        expected_feature_groups.len(),
        expected_label_strategies.len(),
        expected_mbo_groups.len(),
        expected_exp_groups.len()
    );
}
