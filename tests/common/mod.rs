//! Shared test data paths and helpers for integration tests.
//!
//! All integration tests that access real NVIDIA MBO/MBP data should use these
//! constants and helpers rather than defining their own paths. This ensures a
//! single source of truth for data locations.
//!
//! Usage: add `mod common;` at the top of your integration test file, then use
//! `common::HOT_STORE_DIR`, `common::find_mbo_file("20250203")`, etc.

use std::path::{Path, PathBuf};

/// Decompressed MBO files (`.mbo.dbn`), preferred for speed.
pub const HOT_STORE_DIR: &str = "../data/hot_store";

/// Compressed MBO files (`.mbo.dbn.zst`).
pub const COMPRESSED_DIR: &str = "../data/NVDA_2025-02-03_to_2026-01-07";

/// MBP-10 ground truth files (`.mbp-10.dbn.zst`).
pub const MBP10_DIR: &str = "../data/NVDA_MBP10_2025-07";

/// Find a decompressed MBO file for a specific date in hot_store.
///
/// `date` is YYYYMMDD, e.g. "20250203".
/// Returns `None` if the file doesn't exist.
#[allow(dead_code)]
pub fn find_hot_store_file(date: &str) -> Option<PathBuf> {
    let path = PathBuf::from(HOT_STORE_DIR).join(format!("xnas-itch-{date}.mbo.dbn"));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Find a compressed MBO file for a specific date.
///
/// `date` is YYYYMMDD, e.g. "20250203".
/// Returns `None` if the file doesn't exist.
#[allow(dead_code)]
pub fn find_compressed_file(date: &str) -> Option<PathBuf> {
    let path = PathBuf::from(COMPRESSED_DIR).join(format!("xnas-itch-{date}.mbo.dbn.zst"));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Find an MBO file for a date, preferring hot_store (faster I/O).
/// Returns `None` if neither hot_store nor compressed file exists.
#[allow(dead_code)]
pub fn find_mbo_file(date: &str) -> Option<PathBuf> {
    find_hot_store_file(date).or_else(|| find_compressed_file(date))
}

/// Find an MBP-10 ground truth file for a specific date.
///
/// `date` is YYYYMMDD, e.g. "20250701".
#[allow(dead_code)]
pub fn find_mbp10_file(date: &str) -> Option<PathBuf> {
    let path = PathBuf::from(MBP10_DIR).join(format!("xnas-itch-{date}.mbp-10.dbn.zst"));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Find the first available `.mbo.dbn` file in hot_store.
/// Returns `None` if hot_store is empty or doesn't exist.
#[allow(dead_code)]
pub fn find_any_hot_store_file() -> Option<PathBuf> {
    let dir = Path::new(HOT_STORE_DIR);
    if !dir.is_dir() {
        return None;
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "dbn"))
        .collect();
    entries.sort();
    entries.into_iter().next()
}

/// Return `true` if real test data is available (hot_store has files).
#[allow(dead_code)]
pub fn has_test_data() -> bool {
    find_any_hot_store_file().is_some()
}

/// Macro to skip a test if real data is not available.
#[macro_export]
macro_rules! skip_if_no_data {
    () => {
        if !common::has_test_data() {
            eprintln!("Skipping test: no real data available in hot_store");
            return;
        }
    };
}
