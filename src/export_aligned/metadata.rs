//! Export metadata builders for provenance, normalization, and processing info.
//!
//! These helpers produce the standardized JSON blocks embedded in every
//! `{day}_metadata.json` file. All labeling strategies share the same structure.

use super::types::NormalizationParams;

/// Build provenance block for export metadata.
/// Captures git commit, crate version, and config hash.
pub(crate) fn build_provenance(config_hash: &str) -> serde_json::Value {
    serde_json::json!({
        "extractor_version": env!("CARGO_PKG_VERSION"),
        "git_commit": env!("GIT_COMMIT_HASH"),
        "git_dirty": env!("GIT_DIRTY") == "true",
        "config_hash": config_hash,
        "contract_version": crate::contract::SCHEMA_VERSION.to_string(),
        "export_timestamp_utc": chrono::Utc::now().to_rfc3339(),
    })
}

/// Build the standardized normalization metadata block.
/// All 4 label strategies produce an identical structure.
pub(crate) fn build_normalization_metadata(
    norm_params: &NormalizationParams,
    day_name: &str,
) -> serde_json::Value {
    serde_json::json!({
        "strategy": norm_params.strategy.to_string(),
        "applied": norm_params.normalization_applied,
        "levels": norm_params.levels,
        "sample_count": norm_params.sample_count,
        "feature_layout": &norm_params.feature_layout,
        "params_file": format!("{day_name}_normalization.json"),
    })
}

/// Build verification/processing block common to all strategies.
pub(crate) fn build_processing_metadata(
    messages_processed: usize,
    features_extracted: usize,
    sequences_generated: usize,
    sequences_aligned: usize,
    sequences_dropped: usize,
) -> serde_json::Value {
    let drop_rate = if sequences_generated > 0 {
        (sequences_dropped as f64 / sequences_generated as f64) * 100.0
    } else {
        0.0
    };
    serde_json::json!({
        "messages_processed": messages_processed,
        "features_extracted": features_extracted,
        "sequences_generated": sequences_generated,
        "sequences_aligned": sequences_aligned,
        "sequences_dropped": sequences_dropped,
        "drop_rate_percent": format!("{:.2}", drop_rate),
        "buffer_coverage_ok": features_extracted <= 50000,
    })
}
