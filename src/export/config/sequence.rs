//! Sequence building configuration for data export.

use crate::sequence_builder::SequenceConfig;
use serde::{Deserialize, Serialize};

// ============================================================================
// Sequence Configuration (Export-specific wrapper)
// ============================================================================

/// Export-specific sequence configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSequenceConfig {
    /// Window size (number of snapshots per sequence)
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// Stride (overlap between consecutive sequences)
    #[serde(default = "default_stride")]
    pub stride: usize,

    /// Maximum buffer size for sequence building
    #[serde(default = "default_buffer_size")]
    pub max_buffer_size: usize,
}

fn default_window_size() -> usize {
    100
}

fn default_stride() -> usize {
    10
}

fn default_buffer_size() -> usize {
    50000
}

impl Default for ExportSequenceConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            stride: 10,
            max_buffer_size: 50000,
        }
    }
}

impl ExportSequenceConfig {
    /// Convert to internal SequenceConfig.
    pub fn to_sequence_config(&self, feature_count: usize) -> SequenceConfig {
        SequenceConfig {
            window_size: self.window_size,
            stride: self.stride,
            feature_count,
            max_buffer_size: self.max_buffer_size,
        }
    }

    /// Validate the sequence configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("window_size must be > 0".to_string());
        }
        if self.stride == 0 {
            return Err("stride must be > 0".to_string());
        }
        if self.stride > self.window_size {
            return Err("stride should typically be <= window_size".to_string());
        }
        if self.max_buffer_size < self.window_size {
            return Err("max_buffer_size must be >= window_size".to_string());
        }
        Ok(())
    }
}
