//! Pipeline contract constants -- single Rust-side source of truth.
//!
//! These constants MUST match `contracts/pipeline_contract.toml`.
//! Validated at `cargo test` time by `tests/contract_validation_test.rs`.
//!
//! NOTE: `schema::SCHEMA_VERSION` ("1.0.0") in `src/schema/mod.rs` is the
//! internal feature-definition schema version (for paper presets). It is a
//! separate system from the pipeline contract version defined here.

// =============================================================================
// Pipeline Contract
// =============================================================================

/// Schema version embedded in the feature vector at index 97.
/// Matches `[contract].schema_version` in `pipeline_contract.toml`.
///
/// Version history:
/// - 2.0: Initial signal layer implementation
/// - 2.1: Fixed sign convention for net_trade_flow (56) and net_cancel_flow (55)
/// - 2.2: Fixed MBO Core feature names (78-82), implemented median_order_lifetime
pub const SCHEMA_VERSION: f64 = 2.2;

/// Schema version as string for metadata JSON serialization.
pub const SCHEMA_VERSION_STR: &str = "2.2";

/// Number of stable (non-experimental) features.
pub const STABLE_FEATURE_COUNT: usize = 98;

/// Number of experimental features (18 original + 12 MLOFI + 20 Kolm OF).
pub const EXPERIMENTAL_FEATURE_COUNT: usize = 50;

/// Total feature count (stable + all experimental).
pub const FULL_FEATURE_COUNT: usize = 148;

/// Feature count without Kolm OF (stable + original experimental + MLOFI).
pub const LEGACY_FULL_FEATURE_COUNT: usize = 128;

/// Number of Kolm per-level OF features (Kolm, Turiel & Westray 2023).
pub const KOLM_OF_FEATURE_COUNT: usize = 20;

/// Number of LOB depth levels.
pub const LOB_LEVELS: usize = 10;

/// Number of trading signals (indices 84-97).
pub const SIGNAL_COUNT: usize = 14;

/// Feature indices that must NOT be normalized (categorical/flag features).
/// Matches `[features.categorical].indices` in `pipeline_contract.toml`.
pub const CATEGORICAL_INDICES: &[usize] = &[92, 93, 94, 97];

// =============================================================================
// Numerical Precision
// =============================================================================

/// Epsilon for floating-point comparison in assertions and equality checks.
/// Used when checking if two f64 values are "equal enough".
pub const FLOAT_CMP_EPS: f64 = 1e-10;

/// Epsilon added to denominators to prevent division by zero in ratio calculations.
/// Used in order flow ratios, size ratios, and similar bounded-output features.
pub const DIVISION_GUARD_EPS: f64 = 1e-8;
