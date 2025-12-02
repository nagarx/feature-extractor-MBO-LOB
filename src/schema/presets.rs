//! Paper-aligned preset configurations.
//!
//! Each preset matches a published research paper's feature configuration,
//! enabling easy reproduction of results.

use super::feature_def::{FeatureSchema, FeatureSchemaBuilder};
use serde::{Deserialize, Serialize};

/// Paper-aligned feature presets.
///
/// Each preset corresponds to a specific research paper's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Preset {
    /// DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
    ///
    /// - Features: 40 raw LOB (10 levels × 4)
    /// - Normalization: Z-score (per previous day)
    /// - Sequence length: 100
    /// - Horizons: k ∈ {10, 20, 50}
    #[default]
    DeepLOB,

    /// TLOB: Transformer Model with Dual Attention
    ///
    /// - Features: 40 raw LOB
    /// - Normalization: Bilinear (BiN-CTABL)
    /// - Sequence length: 100
    /// - Horizons: k ∈ {10, 20, 50, 100}
    TLOB,

    /// FI-2010: Benchmark Dataset for Mid-Price Forecasting
    ///
    /// - Features: 144 total (40 raw + 104 handcrafted)
    /// - Categories: Basic, Time-Insensitive, Time-Sensitive, Depth
    /// - Normalization: Z-score
    FI2010,

    /// TransLOB: Transformers for Limit Order Books
    ///
    /// - Features: 40 raw LOB
    /// - Normalization: Z-score
    /// - Multi-horizon support
    TransLOB,

    /// LiT: Limit Order Book Transformer
    ///
    /// - Features: 40 raw LOB (patched)
    /// - Patch size: (H=40, W=4)
    /// - Levels: 20
    LiT,

    /// Minimal: Just raw LOB features
    ///
    /// - Features: 40 raw LOB
    /// - No derived features
    /// - Good for quick experiments
    Minimal,

    /// Full: All available features
    ///
    /// - Features: Raw LOB + Order Flow + FI-2010 + Derived
    /// - Maximum feature set
    Full,
}

impl Preset {
    /// Build a feature schema from this preset.
    pub fn build_schema(self) -> FeatureSchema {
        match self {
            Preset::DeepLOB => self.build_deeplob(),
            Preset::TLOB => self.build_tlob(),
            Preset::FI2010 => self.build_fi2010(),
            Preset::TransLOB => self.build_translob(),
            Preset::LiT => self.build_lit(),
            Preset::Minimal => self.build_minimal(),
            Preset::Full => self.build_full(),
        }
    }

    /// Get the preset configuration.
    pub fn config(self) -> PresetConfig {
        match self {
            Preset::DeepLOB => PresetConfig {
                name: "DeepLOB",
                paper: "Zhang et al. (2019)",
                levels: 10,
                sequence_length: 100,
                horizons: vec![10, 20, 50],
                normalization: "zscore",
                feature_count: 40,
            },
            Preset::TLOB => PresetConfig {
                name: "TLOB",
                paper: "TLOB Paper (2023)",
                levels: 10,
                sequence_length: 100,
                horizons: vec![10, 20, 50, 100],
                normalization: "bilinear",
                feature_count: 40,
            },
            Preset::FI2010 => PresetConfig {
                name: "FI-2010",
                paper: "Ntakaris et al. (2018)",
                levels: 10,
                sequence_length: 100,
                horizons: vec![10, 20, 30, 50, 100],
                normalization: "zscore",
                // 40 raw + 20 time-insensitive + 20 time-sensitive + 40 depth = 120
                // Note: Full FI-2010 has 144 features but we implement 120 core features
                feature_count: 120,
            },
            Preset::TransLOB => PresetConfig {
                name: "TransLOB",
                paper: "TransLOB Paper",
                levels: 10,
                sequence_length: 100,
                horizons: vec![10, 20, 50],
                normalization: "zscore",
                feature_count: 40,
            },
            Preset::LiT => PresetConfig {
                name: "LiT",
                paper: "LiT Paper",
                levels: 20,
                sequence_length: 100,
                horizons: vec![10, 20, 50],
                normalization: "zscore",
                feature_count: 80, // 20 levels × 4
            },
            Preset::Minimal => PresetConfig {
                name: "Minimal",
                paper: "N/A",
                levels: 10,
                sequence_length: 100,
                horizons: vec![10],
                normalization: "zscore",
                feature_count: 40,
            },
            Preset::Full => PresetConfig {
                name: "Full",
                paper: "N/A",
                levels: 10,
                sequence_length: 100,
                horizons: vec![10, 20, 50, 100],
                normalization: "zscore",
                feature_count: 40 + 8 + 20 + 20 + 40 + 8, // 136
            },
        }
    }

    fn build_deeplob(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(10).with_raw_lob().build()
    }

    fn build_tlob(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(10).with_raw_lob().build()
    }

    fn build_fi2010(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(10)
            .with_raw_lob()
            .with_fi2010_time_insensitive()
            .with_fi2010_time_sensitive()
            .with_fi2010_depth()
            .build()
    }

    fn build_translob(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(10).with_raw_lob().build()
    }

    fn build_lit(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(20).with_raw_lob().build()
    }

    fn build_minimal(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(10).with_raw_lob().build()
    }

    fn build_full(self) -> FeatureSchema {
        FeatureSchemaBuilder::new(10)
            .with_raw_lob()
            .with_order_flow()
            .with_fi2010_time_insensitive()
            .with_fi2010_time_sensitive()
            .with_fi2010_depth()
            .with_derived()
            .build()
    }
}

/// Configuration details for a preset.
#[derive(Debug, Clone)]
pub struct PresetConfig {
    /// Preset name
    pub name: &'static str,

    /// Source paper reference
    pub paper: &'static str,

    /// Number of LOB levels
    pub levels: usize,

    /// Recommended sequence length
    pub sequence_length: usize,

    /// Recommended prediction horizons
    pub horizons: Vec<usize>,

    /// Recommended normalization strategy
    pub normalization: &'static str,

    /// Total feature count
    pub feature_count: usize,
}

impl PresetConfig {
    /// Get the horizons as a slice.
    pub fn horizons(&self) -> &[usize] {
        &self.horizons
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::feature_def::FeatureCategory;

    #[test]
    fn test_preset_deeplob() {
        let schema = Preset::DeepLOB.build_schema();
        let config = Preset::DeepLOB.config();

        assert_eq!(schema.total_count(), 40);
        assert_eq!(config.levels, 10);
        assert_eq!(config.sequence_length, 100);
        assert_eq!(config.normalization, "zscore");
    }

    #[test]
    fn test_preset_fi2010() {
        let schema = Preset::FI2010.build_schema();
        let config = Preset::FI2010.config();

        // FI-2010 core features: 40 raw + 20 time-insensitive + 20 time-sensitive + 40 depth = 120
        assert_eq!(schema.total_count(), 120);
        assert_eq!(config.horizons, vec![10, 20, 30, 50, 100]);
    }

    #[test]
    fn test_preset_lit() {
        let schema = Preset::LiT.build_schema();
        let config = Preset::LiT.config();

        assert_eq!(schema.total_count(), 80); // 20 levels × 4
        assert_eq!(config.levels, 20);
    }

    #[test]
    fn test_preset_full() {
        let schema = Preset::Full.build_schema();

        // Should have all feature categories
        let raw_lob = schema.features_by_category(FeatureCategory::RawLOB);
        let order_flow = schema.features_by_category(FeatureCategory::OrderFlow);
        let derived = schema.features_by_category(FeatureCategory::Derived);

        assert_eq!(raw_lob.len(), 40);
        assert_eq!(order_flow.len(), 8);
        assert_eq!(derived.len(), 8);
    }

    #[test]
    fn test_preset_default() {
        assert_eq!(Preset::default(), Preset::DeepLOB);
    }

    #[test]
    fn test_all_presets_build() {
        let presets = [
            Preset::DeepLOB,
            Preset::TLOB,
            Preset::FI2010,
            Preset::TransLOB,
            Preset::LiT,
            Preset::Minimal,
            Preset::Full,
        ];

        for preset in presets {
            let schema = preset.build_schema();
            let config = preset.config();

            assert!(schema.total_count() > 0);
            assert_eq!(schema.total_count(), config.feature_count);
        }
    }
}
