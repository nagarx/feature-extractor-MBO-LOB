//! Symbol, data path, and date range configuration.
//!
//! These types define WHERE data comes from and WHICH dates to process.
//! They are symbol-agnostic and can be used for any instrument.

use chrono::Datelike;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ============================================================================
// Symbol Configuration
// ============================================================================

/// Configuration for a trading symbol/instrument.
///
/// Provides symbol-specific parameters like exchange, naming conventions,
/// and file path patterns. This decouples the pipeline from any specific symbol.
///
/// # Example
///
/// ```ignore
/// let nvda = SymbolConfig {
///     name: "NVDA".to_string(),
///     exchange: "XNAS".to_string(),
///     filename_pattern: "xnas-itch-{date}.mbo.dbn.zst".to_string(),
///     tick_size: 0.01,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolConfig {
    /// Symbol name (e.g., "NVDA", "AAPL", "GOOGL")
    pub name: String,

    /// Exchange code (e.g., "XNAS" for NASDAQ)
    pub exchange: String,

    /// Filename pattern for data files.
    ///
    /// Use `{date}` as placeholder for YYYYMMDD date format.
    /// Examples:
    /// - `"xnas-itch-{date}.mbo.dbn.zst"` → `xnas-itch-20250203.mbo.dbn.zst`
    /// - `"{symbol}_{date}.dbn.zst"` → `NVDA_20250203.dbn.zst`
    pub filename_pattern: String,

    /// Minimum tick size for price normalization (e.g., 0.01 for US stocks)
    #[serde(default = "default_tick_size")]
    pub tick_size: f64,
}

fn default_tick_size() -> f64 {
    0.01
}

impl SymbolConfig {
    /// Create a new symbol configuration.
    pub fn new(name: &str, exchange: &str, filename_pattern: &str) -> Self {
        Self {
            name: name.to_string(),
            exchange: exchange.to_string(),
            filename_pattern: filename_pattern.to_string(),
            tick_size: 0.01,
        }
    }

    /// Create NASDAQ symbol configuration with standard naming.
    ///
    /// Uses pattern: `xnas-itch-{date}.mbo.dbn.zst`
    pub fn nasdaq(name: &str) -> Self {
        Self {
            name: name.to_string(),
            exchange: "XNAS".to_string(),
            filename_pattern: "xnas-itch-{date}.mbo.dbn.zst".to_string(),
            tick_size: 0.01,
        }
    }

    /// Generate filename for a specific date.
    ///
    /// # Arguments
    ///
    /// * `date` - Date in YYYY-MM-DD format (e.g., "2025-02-03")
    ///
    /// # Returns
    ///
    /// Filename with date substituted (e.g., "xnas-itch-20250203.mbo.dbn.zst")
    pub fn filename_for_date(&self, date: &str) -> String {
        // Convert YYYY-MM-DD to YYYYMMDD
        let date_compact = date.replace('-', "");
        self.filename_pattern
            .replace("{date}", &date_compact)
            .replace("{symbol}", &self.name)
    }

    /// Validate the symbol configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Symbol name cannot be empty".to_string());
        }
        if self.exchange.is_empty() {
            return Err("Exchange cannot be empty".to_string());
        }
        if !self.filename_pattern.contains("{date}") {
            return Err("Filename pattern must contain {date} placeholder".to_string());
        }
        if self.tick_size <= 0.0 {
            return Err("Tick size must be positive".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// Data Path Configuration
// ============================================================================

/// Configuration for input and output data paths.
///
/// Supports both compressed (.dbn.zst) and decompressed (.dbn) files,
/// with optional hot store for faster processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPathConfig {
    /// Directory containing raw MBO data files.
    pub input_dir: PathBuf,

    /// Optional hot store directory with decompressed files.
    ///
    /// When set, the processor prefers decompressed files for ~30% faster processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hot_store_dir: Option<PathBuf>,

    /// Output directory for exported datasets.
    pub output_dir: PathBuf,
}

impl DataPathConfig {
    /// Create a new data path configuration.
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(input_dir: P1, output_dir: P2) -> Self {
        Self {
            input_dir: input_dir.as_ref().to_path_buf(),
            hot_store_dir: None,
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }

    /// Set hot store directory for faster processing.
    pub fn with_hot_store<P: AsRef<Path>>(mut self, hot_store_dir: P) -> Self {
        self.hot_store_dir = Some(hot_store_dir.as_ref().to_path_buf());
        self
    }

    /// Validate paths exist (or can be created for output).
    pub fn validate(&self) -> Result<(), String> {
        if !self.input_dir.exists() {
            return Err(format!(
                "Input directory does not exist: {}",
                self.input_dir.display()
            ));
        }

        if let Some(ref hot_store) = self.hot_store_dir {
            if !hot_store.exists() {
                return Err(format!(
                    "Hot store directory does not exist: {}",
                    hot_store.display()
                ));
            }
        }

        // Output directory will be created if needed
        Ok(())
    }

    /// Validate paths exist (lenient - allows creation of output dir).
    pub fn validate_lenient(&self) -> Result<(), String> {
        if !self.input_dir.exists() {
            return Err(format!(
                "Input directory does not exist: {}",
                self.input_dir.display()
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Date Range Configuration
// ============================================================================

/// Configuration for date range selection.
///
/// Supports both explicit date lists and date ranges.
/// For ranges, weekends are automatically excluded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DateRangeConfig {
    /// Start date (inclusive), format: YYYY-MM-DD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_date: Option<String>,

    /// End date (inclusive), format: YYYY-MM-DD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_date: Option<String>,

    /// Explicit list of dates (overrides start/end if provided).
    ///
    /// Format: YYYY-MM-DD for each date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_dates: Option<Vec<String>>,

    /// Dates to exclude (e.g., holidays).
    ///
    /// Format: YYYY-MM-DD for each date.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude_dates: Vec<String>,
}

impl DateRangeConfig {
    /// Create from explicit list of dates.
    pub fn from_dates(dates: Vec<String>) -> Self {
        Self {
            start_date: None,
            end_date: None,
            explicit_dates: Some(dates),
            exclude_dates: Vec::new(),
        }
    }

    /// Create from date range.
    pub fn from_range(start: &str, end: &str) -> Self {
        Self {
            start_date: Some(start.to_string()),
            end_date: Some(end.to_string()),
            explicit_dates: None,
            exclude_dates: Vec::new(),
        }
    }

    /// Add dates to exclude.
    pub fn with_exclusions(mut self, dates: Vec<String>) -> Self {
        self.exclude_dates = dates;
        self
    }

    /// Get the list of dates to process.
    ///
    /// If `explicit_dates` is provided, returns those.
    /// Otherwise, generates dates from `start_date` to `end_date`,
    /// excluding weekends and `exclude_dates`.
    pub fn get_dates(&self) -> Result<Vec<String>, String> {
        if let Some(ref explicit) = self.explicit_dates {
            // Filter out excluded dates
            let dates: Vec<String> = explicit
                .iter()
                .filter(|d| !self.exclude_dates.contains(d))
                .cloned()
                .collect();
            return Ok(dates);
        }

        // Generate from range
        let start = self
            .start_date
            .as_ref()
            .ok_or("Either explicit_dates or start_date must be provided")?;
        let end = self
            .end_date
            .as_ref()
            .ok_or("Either explicit_dates or end_date must be provided")?;

        // Parse dates
        let start_date = chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
            .map_err(|e| format!("Invalid start_date '{}': {}", start, e))?;
        let end_date = chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
            .map_err(|e| format!("Invalid end_date '{}': {}", end, e))?;

        if start_date > end_date {
            return Err(format!(
                "start_date ({}) must be before end_date ({})",
                start, end
            ));
        }

        // Generate all weekdays in range
        let mut dates = Vec::new();
        let mut current = start_date;

        while current <= end_date {
            // Skip weekends (Saturday = 6, Sunday = 0 in chrono)
            let weekday = current.weekday();
            if weekday != chrono::Weekday::Sat && weekday != chrono::Weekday::Sun {
                let date_str = current.format("%Y-%m-%d").to_string();
                if !self.exclude_dates.contains(&date_str) {
                    dates.push(date_str);
                }
            }
            current = current.succ_opt().unwrap_or(current);
        }

        Ok(dates)
    }

    /// Validate the date configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.explicit_dates.is_none() && (self.start_date.is_none() || self.end_date.is_none()) {
            return Err(
                "Either explicit_dates or both start_date and end_date must be provided"
                    .to_string(),
            );
        }

        // Validate date formats
        if let Some(ref start) = self.start_date {
            chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
                .map_err(|e| format!("Invalid start_date '{}': {}", start, e))?;
        }

        if let Some(ref end) = self.end_date {
            chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
                .map_err(|e| format!("Invalid end_date '{}': {}", end, e))?;
        }

        for date in &self.exclude_dates {
            chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
                .map_err(|e| format!("Invalid exclude_date '{}': {}", date, e))?;
        }

        if let Some(ref explicit) = self.explicit_dates {
            for date in explicit {
                chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
                    .map_err(|e| format!("Invalid explicit date '{}': {}", date, e))?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_config_filename_generation() {
        let symbol = SymbolConfig::nasdaq("NVDA");
        assert_eq!(
            symbol.filename_for_date("2025-02-03"),
            "xnas-itch-20250203.mbo.dbn.zst"
        );
    }

    #[test]
    fn test_date_range_generation() {
        let dates = DateRangeConfig::from_range("2025-02-03", "2025-02-07");
        let result = dates.get_dates().unwrap();
        // Feb 3-7, 2025: Mon, Tue, Wed, Thu, Fri (all weekdays)
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], "2025-02-03");
        assert_eq!(result[4], "2025-02-07");
    }

    #[test]
    fn test_date_range_skips_weekends() {
        let dates = DateRangeConfig::from_range("2025-02-07", "2025-02-10");
        let result = dates.get_dates().unwrap();
        // Feb 7 (Fri) and Feb 10 (Mon) - skips Sat/Sun
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "2025-02-07");
        assert_eq!(result[1], "2025-02-10");
    }
}
