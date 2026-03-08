//! Processing, split, and experiment metadata configuration.

use serde::{Deserialize, Serialize};

// ============================================================================
// Processing Configuration
// ============================================================================

/// Configuration for parallel processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Number of threads for parallel processing.
    ///
    /// Defaults to number of CPU cores - 2 if not specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,

    /// Error handling mode.
    ///
    /// - `"fail_fast"`: Stop on first error
    /// - `"collect_errors"`: Continue processing, collect all errors
    #[serde(default = "default_error_mode")]
    pub error_mode: String,

    /// Enable verbose progress reporting.
    #[serde(default)]
    pub verbose: bool,
}

fn default_error_mode() -> String {
    "collect_errors".to_string()
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            threads: None,
            error_mode: "collect_errors".to_string(),
            verbose: false,
        }
    }
}

impl ProcessingConfig {
    /// Validate the processing configuration.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(threads) = self.threads {
            if threads == 0 {
                return Err("threads must be > 0".to_string());
            }
        }

        if self.error_mode != "fail_fast" && self.error_mode != "collect_errors" {
            return Err(format!(
                "error_mode must be 'fail_fast' or 'collect_errors', got '{}'",
                self.error_mode
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Split Configuration
// ============================================================================

/// Configuration for train/validation/test splits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    /// Proportion of days for training (0.0 to 1.0)
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,

    /// Proportion of days for validation (0.0 to 1.0)
    #[serde(default = "default_val_ratio")]
    pub val_ratio: f64,

    /// Proportion of days for testing (0.0 to 1.0)
    #[serde(default = "default_test_ratio")]
    pub test_ratio: f64,
}

fn default_train_ratio() -> f64 {
    0.7
}

fn default_val_ratio() -> f64 {
    0.15
}

fn default_test_ratio() -> f64 {
    0.15
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            val_ratio: 0.15,
            test_ratio: 0.15,
        }
    }
}

impl SplitConfig {
    /// Validate the split configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.train_ratio < 0.0 || self.train_ratio > 1.0 {
            return Err("train_ratio must be between 0 and 1".to_string());
        }
        if self.val_ratio < 0.0 || self.val_ratio > 1.0 {
            return Err("val_ratio must be between 0 and 1".to_string());
        }
        if self.test_ratio < 0.0 || self.test_ratio > 1.0 {
            return Err("test_ratio must be between 0 and 1".to_string());
        }

        let total = self.train_ratio + self.val_ratio + self.test_ratio;
        if (total - 1.0).abs() > 0.001 {
            return Err(format!(
                "Split ratios must sum to 1.0, got {} + {} + {} = {}",
                self.train_ratio, self.val_ratio, self.test_ratio, total
            ));
        }

        Ok(())
    }

    /// Split a list of days according to the ratios.
    ///
    /// Days are kept in order (chronological split).
    pub fn split_days<'a>(&self, days: &'a [String]) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
        let total = days.len();
        let train_end = (total as f64 * self.train_ratio).round() as usize;
        let val_end = train_end + (total as f64 * self.val_ratio).round() as usize;

        let train: Vec<&str> = days[..train_end].iter().map(|s| s.as_str()).collect();
        let val: Vec<&str> = days[train_end..val_end]
            .iter()
            .map(|s| s.as_str())
            .collect();
        let test: Vec<&str> = days[val_end..].iter().map(|s| s.as_str()).collect();

        (train, val, test)
    }
}

// ============================================================================
// Experiment Metadata
// ============================================================================

/// Metadata for experiment tracking and reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentInfo {
    /// Experiment name
    pub name: String,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Version
    #[serde(default = "default_version")]
    pub version: String,

    /// Tags for categorization
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

impl Default for ExperimentInfo {
    fn default() -> Self {
        Self {
            name: "Unnamed Experiment".to_string(),
            description: None,
            version: "1.0.0".to_string(),
            tags: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_config_validation() {
        let valid = SplitConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = SplitConfig {
            train_ratio: 0.5,
            val_ratio: 0.5,
            test_ratio: 0.5, // Sum > 1
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_split_days() {
        let split = SplitConfig {
            train_ratio: 0.6,
            val_ratio: 0.2,
            test_ratio: 0.2,
        };

        let days: Vec<String> = (1..=10).map(|i| format!("2025-02-{:02}", i)).collect();
        let (train, val, test) = split.split_days(&days);

        assert_eq!(train.len(), 6);
        assert_eq!(val.len(), 2);
        assert_eq!(test.len(), 2);
    }
}
