//! Feature Schema Module
//!
//! Provides versioned, typed feature definitions for reproducible ML experiments.
//! Based on analysis of 17 research papers including DeepLOB, TLOB, FI-2010, TransLOB.
//!
//! # Design Philosophy
//!
//! - **Versioned**: Schema versions enable reproducibility across experiments
//! - **Typed**: Feature categories and indices are compile-time checked
//! - **Paper-Aligned**: Presets match published research configurations
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::schema::{Preset, FeatureSchema};
//!
//! // Use a paper-aligned preset
//! let schema = FeatureSchema::from_preset(Preset::DeepLOB);
//! assert_eq!(schema.total_count(), 40);
//!
//! // Access feature metadata
//! let bid_price_1 = schema.get_feature("bid_price_1").unwrap();
//! assert_eq!(bid_price_1.category, FeatureCategory::RawLOB);
//! ```

mod feature_def;
mod presets;

pub use feature_def::{FeatureCategory, FeatureDef, FeatureSchema};
pub use presets::{Preset, PresetConfig};

/// Current schema version
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Default number of LOB levels
pub const DEFAULT_LEVELS: usize = 10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version() {
        assert!(!SCHEMA_VERSION.is_empty());
    }

    #[test]
    fn test_preset_deeplob() {
        let schema = FeatureSchema::from_preset(Preset::DeepLOB);
        assert_eq!(schema.total_count(), 40); // 10 levels × 4 features
    }

    #[test]
    fn test_preset_tlob() {
        let schema = FeatureSchema::from_preset(Preset::TLOB);
        assert_eq!(schema.total_count(), 40); // Same as DeepLOB for raw features
    }

    #[test]
    fn test_preset_fi2010() {
        let schema = FeatureSchema::from_preset(Preset::FI2010);
        // FI-2010 core features: 40 raw + 20 time-insensitive + 20 time-sensitive + 40 depth = 120
        assert_eq!(schema.total_count(), 120);
    }

    #[test]
    fn test_feature_lookup() {
        let schema = FeatureSchema::from_preset(Preset::DeepLOB);

        // Check first ask price
        let feat = schema.get_feature("ask_price_1").unwrap();
        assert_eq!(feat.index, 0);
        assert_eq!(feat.category, FeatureCategory::RawLOB);

        // Check first bid price
        let feat = schema.get_feature("bid_price_1").unwrap();
        assert_eq!(feat.index, 20); // After 10 ask prices + 10 ask sizes
    }

    #[test]
    fn test_feature_category_slice() {
        let schema = FeatureSchema::from_preset(Preset::DeepLOB);
        let raw_lob = schema.features_by_category(FeatureCategory::RawLOB);
        assert_eq!(raw_lob.len(), 40);
    }
}
