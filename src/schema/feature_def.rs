//! Feature definitions and schema types.
//!
//! This module defines the core types for feature metadata:
//! - `FeatureCategory`: Enum of feature types (RawLOB, OrderFlow, FI2010, etc.)
//! - `FeatureDef`: Metadata for a single feature
//! - `FeatureSchema`: Collection of feature definitions

use super::presets::Preset;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature category based on research papers.
///
/// Each category corresponds to a distinct type of market microstructure feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureCategory {
    /// Raw LOB features: prices and volumes at each level
    /// Source: All papers (DeepLOB, TLOB, FI-2010, etc.)
    RawLOB,

    /// Order flow features: OFI, queue imbalance, trade flow
    /// Source: "The Price Impact of Order Book Events" (Cont et al.)
    OrderFlow,

    /// FI-2010 time-insensitive features: spread, mid-price, diffs
    /// Source: "Benchmark Dataset for Mid-Price Forecasting"
    FI2010TimeInsensitive,

    /// FI-2010 time-sensitive features: derivatives, intensity
    /// Source: "Benchmark Dataset for Mid-Price Forecasting"
    FI2010TimeSensitive,

    /// FI-2010 depth features: accumulated differences
    /// Source: "Benchmark Dataset for Mid-Price Forecasting"
    FI2010Depth,

    /// Derived analytics: microprice, VWAP, imbalance
    /// Source: TLOB, DeepLOB papers
    Derived,

    /// MBO-specific features: order lifecycle, institutional patterns
    /// Source: "Deep Learning for Market by Order Data"
    MBO,
}

impl FeatureCategory {
    /// Get all categories in standard order.
    pub fn all() -> &'static [FeatureCategory] {
        &[
            FeatureCategory::RawLOB,
            FeatureCategory::OrderFlow,
            FeatureCategory::FI2010TimeInsensitive,
            FeatureCategory::FI2010TimeSensitive,
            FeatureCategory::FI2010Depth,
            FeatureCategory::Derived,
            FeatureCategory::MBO,
        ]
    }

    /// Get the display name for this category.
    pub fn name(&self) -> &'static str {
        match self {
            FeatureCategory::RawLOB => "Raw LOB",
            FeatureCategory::OrderFlow => "Order Flow",
            FeatureCategory::FI2010TimeInsensitive => "FI-2010 Time-Insensitive",
            FeatureCategory::FI2010TimeSensitive => "FI-2010 Time-Sensitive",
            FeatureCategory::FI2010Depth => "FI-2010 Depth",
            FeatureCategory::Derived => "Derived",
            FeatureCategory::MBO => "MBO",
        }
    }
}

/// Definition of a single feature.
///
/// Contains metadata about a feature including its name, index, category,
/// and optional reference to the source paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDef {
    /// Unique feature name (e.g., "ask_price_1", "ofi", "spread")
    pub name: String,

    /// Index in the feature vector
    pub index: usize,

    /// Feature category
    pub category: FeatureCategory,

    /// Human-readable description
    pub description: String,

    /// Reference to source paper (optional)
    pub paper_ref: Option<String>,

    /// LOB level (1-indexed, None for non-level features)
    pub level: Option<usize>,
}

impl FeatureDef {
    /// Create a new feature definition.
    pub fn new(
        name: impl Into<String>,
        index: usize,
        category: FeatureCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            index,
            category,
            description: description.into(),
            paper_ref: None,
            level: None,
        }
    }

    /// Set the paper reference.
    pub fn with_paper_ref(mut self, paper: impl Into<String>) -> Self {
        self.paper_ref = Some(paper.into());
        self
    }

    /// Set the LOB level.
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }
}

/// Feature schema containing all feature definitions.
///
/// The schema is versioned and can be created from presets or custom configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSchema {
    /// Schema version
    pub version: String,

    /// All feature definitions
    features: Vec<FeatureDef>,

    /// Name-to-index lookup
    #[serde(skip)]
    name_index: HashMap<String, usize>,

    /// Category-to-indices lookup
    #[serde(skip)]
    category_indices: HashMap<FeatureCategory, Vec<usize>>,

    /// Number of LOB levels
    pub levels: usize,
}

impl FeatureSchema {
    /// Create a new empty schema.
    pub fn new(version: impl Into<String>, levels: usize) -> Self {
        Self {
            version: version.into(),
            features: Vec::new(),
            name_index: HashMap::new(),
            category_indices: HashMap::new(),
            levels,
        }
    }

    /// Create a schema from a preset.
    pub fn from_preset(preset: Preset) -> Self {
        preset.build_schema()
    }

    /// Add a feature to the schema.
    pub fn add_feature(&mut self, feature: FeatureDef) {
        let index = feature.index;
        let name = feature.name.clone();
        let category = feature.category;

        self.features.push(feature);
        self.name_index.insert(name, index);
        self.category_indices
            .entry(category)
            .or_default()
            .push(index);
    }

    /// Get the total number of features.
    pub fn total_count(&self) -> usize {
        self.features.len()
    }

    /// Get a feature by name.
    pub fn get_feature(&self, name: &str) -> Option<&FeatureDef> {
        self.name_index
            .get(name)
            .and_then(|&idx| self.features.iter().find(|f| f.index == idx))
    }

    /// Get a feature by index.
    pub fn get_feature_by_index(&self, index: usize) -> Option<&FeatureDef> {
        self.features.iter().find(|f| f.index == index)
    }

    /// Get all features in a category.
    pub fn features_by_category(&self, category: FeatureCategory) -> Vec<&FeatureDef> {
        self.features
            .iter()
            .filter(|f| f.category == category)
            .collect()
    }

    /// Get feature indices for a category.
    pub fn indices_by_category(&self, category: FeatureCategory) -> &[usize] {
        self.category_indices
            .get(&category)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all feature definitions.
    pub fn all_features(&self) -> &[FeatureDef] {
        &self.features
    }

    /// Check if the schema contains a feature.
    pub fn contains(&self, name: &str) -> bool {
        self.name_index.contains_key(name)
    }

    /// Get feature names in order.
    pub fn feature_names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.features.iter().map(|f| f.name.as_str()).collect();
        names.sort_by_key(|n| self.name_index.get(*n).unwrap_or(&usize::MAX));
        names
    }

    /// Rebuild internal indices (call after deserialization).
    pub fn rebuild_indices(&mut self) {
        self.name_index.clear();
        self.category_indices.clear();

        for feature in &self.features {
            self.name_index.insert(feature.name.clone(), feature.index);
            self.category_indices
                .entry(feature.category)
                .or_default()
                .push(feature.index);
        }
    }
}

/// Builder for creating custom feature schemas.
pub struct FeatureSchemaBuilder {
    schema: FeatureSchema,
    next_index: usize,
}

impl FeatureSchemaBuilder {
    /// Create a new schema builder.
    pub fn new(levels: usize) -> Self {
        Self {
            schema: FeatureSchema::new(super::SCHEMA_VERSION, levels),
            next_index: 0,
        }
    }

    /// Add raw LOB features (prices and volumes at each level).
    pub fn with_raw_lob(mut self) -> Self {
        let levels = self.schema.levels;

        // Ask prices
        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("ask_price_{level}"),
                self.next_index,
                FeatureCategory::RawLOB,
                format!("Ask price at level {level}"),
            )
            .with_level(level)
            .with_paper_ref("DeepLOB, TLOB");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        // Ask sizes
        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("ask_size_{level}"),
                self.next_index,
                FeatureCategory::RawLOB,
                format!("Ask size at level {level}"),
            )
            .with_level(level)
            .with_paper_ref("DeepLOB, TLOB");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        // Bid prices
        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("bid_price_{level}"),
                self.next_index,
                FeatureCategory::RawLOB,
                format!("Bid price at level {level}"),
            )
            .with_level(level)
            .with_paper_ref("DeepLOB, TLOB");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        // Bid sizes
        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("bid_size_{level}"),
                self.next_index,
                FeatureCategory::RawLOB,
                format!("Bid size at level {level}"),
            )
            .with_level(level)
            .with_paper_ref("DeepLOB, TLOB");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        self
    }

    /// Add order flow features (OFI, queue imbalance).
    pub fn with_order_flow(mut self) -> Self {
        let order_flow_features = [
            ("ofi", "Order Flow Imbalance"),
            ("ofi_bid", "Order Flow Imbalance (bid side)"),
            ("ofi_ask", "Order Flow Imbalance (ask side)"),
            (
                "queue_imbalance",
                "Queue Imbalance: (n_bid - n_ask) / (n_bid + n_ask)",
            ),
            ("trade_imbalance", "Trade Imbalance: signed trade volume"),
            (
                "depth_imbalance",
                "Depth Imbalance: (bid_vol - ask_vol) / total",
            ),
            ("arrival_rate_bid", "Order arrival rate on bid side"),
            ("arrival_rate_ask", "Order arrival rate on ask side"),
        ];

        for (name, desc) in order_flow_features {
            let feat = FeatureDef::new(name, self.next_index, FeatureCategory::OrderFlow, desc)
                .with_paper_ref("Cont et al. (2014)");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        self
    }

    /// Add FI-2010 time-insensitive features.
    pub fn with_fi2010_time_insensitive(mut self) -> Self {
        let levels = self.schema.levels;

        // Spread and mid-price
        let basic_features = [
            ("spread", "Best ask - best bid"),
            ("mid_price", "(best ask + best bid) / 2"),
        ];

        for (name, desc) in basic_features {
            let feat = FeatureDef::new(
                name,
                self.next_index,
                FeatureCategory::FI2010TimeInsensitive,
                desc,
            )
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        // Price differences between levels
        for level in 1..levels {
            let feat = FeatureDef::new(
                format!("ask_price_diff_{level}"),
                self.next_index,
                FeatureCategory::FI2010TimeInsensitive,
                format!("Ask price level {level} - level {}", level + 1),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        for level in 1..levels {
            let feat = FeatureDef::new(
                format!("bid_price_diff_{level}"),
                self.next_index,
                FeatureCategory::FI2010TimeInsensitive,
                format!("Bid price level {level} - level {}", level + 1),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        self
    }

    /// Add FI-2010 time-sensitive features.
    pub fn with_fi2010_time_sensitive(mut self) -> Self {
        let time_sensitive_features = [
            ("mid_price_derivative", "d(mid_price)/dt"),
            ("spread_derivative", "d(spread)/dt"),
            ("price_intensity", "Price change intensity"),
            ("volume_intensity", "Volume change intensity"),
            ("bid_intensity", "Bid side intensity"),
            ("ask_intensity", "Ask side intensity"),
            ("relative_spread", "spread / mid_price"),
            ("relative_spread_derivative", "d(relative_spread)/dt"),
            ("bid_ask_volume_ratio", "bid_volume / ask_volume"),
            ("volume_ratio_derivative", "d(volume_ratio)/dt"),
        ];

        for (name, desc) in time_sensitive_features {
            let feat = FeatureDef::new(
                name,
                self.next_index,
                FeatureCategory::FI2010TimeSensitive,
                desc,
            )
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        // Add derivatives for each level
        for level in 1..=self.schema.levels {
            let feat = FeatureDef::new(
                format!("ask_price_derivative_{level}"),
                self.next_index,
                FeatureCategory::FI2010TimeSensitive,
                format!("d(ask_price_{level})/dt"),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        self
    }

    /// Add FI-2010 depth features.
    pub fn with_fi2010_depth(mut self) -> Self {
        let levels = self.schema.levels;

        // Accumulated volume differences
        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("acc_bid_volume_{level}"),
                self.next_index,
                FeatureCategory::FI2010Depth,
                format!("Accumulated bid volume up to level {level}"),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("acc_ask_volume_{level}"),
                self.next_index,
                FeatureCategory::FI2010Depth,
                format!("Accumulated ask volume up to level {level}"),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        // Accumulated price differences
        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("acc_bid_price_diff_{level}"),
                self.next_index,
                FeatureCategory::FI2010Depth,
                format!("Accumulated bid price difference to level {level}"),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        for level in 1..=levels {
            let feat = FeatureDef::new(
                format!("acc_ask_price_diff_{level}"),
                self.next_index,
                FeatureCategory::FI2010Depth,
                format!("Accumulated ask price difference to level {level}"),
            )
            .with_level(level)
            .with_paper_ref("FI-2010 Benchmark");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        self
    }

    /// Add derived features (microprice, VWAP, etc.).
    pub fn with_derived(mut self) -> Self {
        let derived_features = [
            ("microprice", "Volume-weighted mid-price"),
            ("vwap_bid", "Volume-weighted average bid price"),
            ("vwap_ask", "Volume-weighted average ask price"),
            ("spread_bps", "Spread in basis points"),
            ("total_bid_volume", "Total volume on bid side"),
            ("total_ask_volume", "Total volume on ask side"),
            ("volume_imbalance", "(bid_vol - ask_vol) / total_vol"),
            ("price_impact", "Estimated price impact"),
        ];

        for (name, desc) in derived_features {
            let feat = FeatureDef::new(name, self.next_index, FeatureCategory::Derived, desc)
                .with_paper_ref("TLOB, DeepLOB");
            self.schema.add_feature(feat);
            self.next_index += 1;
        }

        self
    }

    /// Build the final schema.
    pub fn build(mut self) -> FeatureSchema {
        self.schema.rebuild_indices();
        self.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_category_all() {
        let categories = FeatureCategory::all();
        assert!(categories.len() >= 5);
    }

    #[test]
    fn test_feature_def_builder() {
        let feat = FeatureDef::new("test", 0, FeatureCategory::RawLOB, "Test feature")
            .with_paper_ref("Test Paper")
            .with_level(1);

        assert_eq!(feat.name, "test");
        assert_eq!(feat.paper_ref, Some("Test Paper".to_string()));
        assert_eq!(feat.level, Some(1));
    }

    #[test]
    fn test_schema_builder_raw_lob() {
        let schema = FeatureSchemaBuilder::new(10).with_raw_lob().build();

        assert_eq!(schema.total_count(), 40); // 10 levels × 4 features
        assert!(schema.contains("ask_price_1"));
        assert!(schema.contains("bid_size_10"));
    }

    #[test]
    fn test_schema_builder_order_flow() {
        let schema = FeatureSchemaBuilder::new(10)
            .with_raw_lob()
            .with_order_flow()
            .build();

        assert_eq!(schema.total_count(), 48); // 40 raw + 8 order flow
        assert!(schema.contains("ofi"));
        assert!(schema.contains("queue_imbalance"));
    }

    #[test]
    fn test_schema_feature_lookup() {
        let schema = FeatureSchemaBuilder::new(10).with_raw_lob().build();

        let feat = schema.get_feature("ask_price_5").unwrap();
        assert_eq!(feat.index, 4); // 0-indexed, 5th ask price
        assert_eq!(feat.level, Some(5));
    }

    #[test]
    fn test_schema_category_indices() {
        let schema = FeatureSchemaBuilder::new(10)
            .with_raw_lob()
            .with_order_flow()
            .build();

        let raw_lob = schema.features_by_category(FeatureCategory::RawLOB);
        assert_eq!(raw_lob.len(), 40);

        let order_flow = schema.features_by_category(FeatureCategory::OrderFlow);
        assert_eq!(order_flow.len(), 8);
    }
}
