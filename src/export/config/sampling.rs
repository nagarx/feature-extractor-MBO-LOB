//! Sampling strategy configuration for data export.

use crate::config::{SamplingConfig, SamplingStrategy};
use serde::{Deserialize, Serialize};

// ============================================================================
// Sampling Configuration (Export-specific wrapper)
// ============================================================================

/// Sampling strategy selection for export configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategyConfig {
    /// Sample every N shares traded
    VolumeBased,
    /// Sample every N events
    EventBased,
}

impl Default for SamplingStrategyConfig {
    fn default() -> Self {
        Self::EventBased
    }
}

/// Export-specific sampling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSamplingConfig {
    /// Sampling strategy
    #[serde(default)]
    pub strategy: SamplingStrategyConfig,

    /// Event count for event-based sampling
    #[serde(default = "default_event_count")]
    pub event_count: usize,

    /// Volume threshold for volume-based sampling
    #[serde(default = "default_volume_threshold")]
    pub volume_threshold: u64,
}

fn default_event_count() -> usize {
    1000
}

fn default_volume_threshold() -> u64 {
    1000
}

impl Default for ExportSamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategyConfig::EventBased,
            event_count: 1000,
            volume_threshold: 1000,
        }
    }
}

impl ExportSamplingConfig {
    /// Convert to internal SamplingConfig.
    pub fn to_sampling_config(&self) -> SamplingConfig {
        match self.strategy {
            SamplingStrategyConfig::EventBased => SamplingConfig {
                strategy: SamplingStrategy::EventBased,
                event_count: Some(self.event_count),
                volume_threshold: None,
                min_time_interval_ns: Some(1_000_000),
                adaptive: None,
                multiscale: None,
            },
            SamplingStrategyConfig::VolumeBased => SamplingConfig {
                strategy: SamplingStrategy::VolumeBased,
                event_count: None,
                volume_threshold: Some(self.volume_threshold),
                min_time_interval_ns: Some(1_000_000),
                adaptive: None,
                multiscale: None,
            },
        }
    }
}
