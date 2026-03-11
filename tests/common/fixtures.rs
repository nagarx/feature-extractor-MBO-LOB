//! JSON fixture load/save/compare helpers.
//!
//! Extracted from the golden_snapshot pattern for reuse across
//! transformation tracing, golden snapshot, and export round-trip tests.

use serde::{de::DeserializeOwned, Serialize};
use std::path::{Path, PathBuf};

/// Load an existing fixture or generate it using the provided closure.
///
/// On first run (fixture file missing), calls `generator` to produce the data,
/// serializes it to JSON, and writes it to `path`. On subsequent runs, loads
/// and deserializes the existing fixture.
///
/// Returns `(data, was_generated)` -- `was_generated` is `true` if the fixture
/// was created in this call (caller should print a notice and optionally skip
/// comparison).
#[allow(dead_code)]
pub fn load_or_generate_fixture<T, F>(path: &Path, generator: F) -> (T, bool)
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> T,
{
    if path.exists() {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read fixture {}: {e}", path.display()));
        let data: T = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse fixture {}: {e}", path.display()));
        (data, false)
    } else {
        let data = generator();
        let json = serde_json::to_string_pretty(&data)
            .unwrap_or_else(|e| panic!("Failed to serialize fixture: {e}"));

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, &json)
            .unwrap_or_else(|e| panic!("Failed to write fixture {}: {e}", path.display()));

        eprintln!(
            "GENERATED new fixture: {}. Re-run to validate against it.",
            path.display()
        );
        (data, true)
    }
}

/// Get the absolute path to a fixture file under `tests/fixtures/`.
#[allow(dead_code)]
pub fn fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(filename)
}
