//! Sampling strategy configuration for data export.

use crate::config::{SamplingConfig, SamplingStrategy};
use serde::{Deserialize, Serialize};

// ============================================================================
// Sampling Configuration (Export-specific wrapper)
// ============================================================================

/// Sampling strategy selection for export configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategyConfig {
    /// Sample every N shares traded
    VolumeBased,
    /// Sample every N events
    #[default]
    EventBased,
    /// Sample at fixed time intervals aligned to market open
    TimeBased,
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

    /// Sampling interval in nanoseconds (for time-based sampling).
    /// Default: 5s = 5_000_000_000 ns.
    #[serde(default = "default_time_interval_ns")]
    pub time_interval_ns: u64,

    /// UTC offset for market open alignment (EST=-5, EDT=-4).
    #[serde(default = "default_utc_offset_hours")]
    pub utc_offset_hours: i32,
}

fn default_event_count() -> usize {
    1000
}

fn default_volume_threshold() -> u64 {
    1000
}

fn default_time_interval_ns() -> u64 {
    5_000_000_000 // 5 seconds
}

fn default_utc_offset_hours() -> i32 {
    -5 // EST
}

impl Default for ExportSamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategyConfig::EventBased,
            event_count: 1000,
            volume_threshold: 1000,
            time_interval_ns: default_time_interval_ns(),
            utc_offset_hours: default_utc_offset_hours(),
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
                time_interval_ns: None,
                utc_offset_hours: None,
                adaptive: None,
                multiscale: None,
            },
            SamplingStrategyConfig::VolumeBased => SamplingConfig {
                strategy: SamplingStrategy::VolumeBased,
                event_count: None,
                volume_threshold: Some(self.volume_threshold),
                min_time_interval_ns: Some(1_000_000),
                time_interval_ns: None,
                utc_offset_hours: None,
                adaptive: None,
                multiscale: None,
            },
            SamplingStrategyConfig::TimeBased => SamplingConfig {
                strategy: SamplingStrategy::TimeBased,
                event_count: None,
                volume_threshold: None,
                min_time_interval_ns: None,
                time_interval_ns: Some(self.time_interval_ns),
                utc_offset_hours: Some(self.utc_offset_hours),
                adaptive: None,
                multiscale: None,
            },
        }
    }
}
