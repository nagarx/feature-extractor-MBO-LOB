//! Tests for aligned export module

use feature_extractor::{AlignedBatchExporter, LabelConfig, PipelineOutput, Sequence};
use ndarray::{Array1, Array3};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use tempfile::TempDir;

/// Create mock pipeline output for testing with realistic LOB data
///
/// Creates feature vectors with proper LOB structure:
/// - Indices 0-9: Ask prices (ascending from mid + spread/2)
/// - Indices 10-19: Ask sizes
/// - Indices 20-29: Bid prices (descending from mid - spread/2)
/// - Indices 30-39: Bid sizes
/// - Indices 40-47: Derived features (if 48 features)
fn create_mock_output(n_sequences: usize, window: usize, features: usize) -> PipelineOutput {
    let mut sequences = Vec::new();
    let mut mid_prices = Vec::new();

    let base_mid_price = 100.0;
    let spread = 0.01; // 1 cent spread
    let tick_size = 0.01;

    // Generate synthetic sequences and mid-prices
    for seq_idx in 0..n_sequences {
        let mut seq_features = Vec::new();

        for t in 0..window {
            // Create realistic LOB feature vector
            let mid_price = base_mid_price + (seq_idx + t) as f64 * 0.001;
            let best_ask = mid_price + spread / 2.0;
            let best_bid = mid_price - spread / 2.0;

            let mut timestep = vec![0.0; features];

            // Ask prices (indices 0-9): ascending from best ask
            for level in 0..10.min(features) {
                timestep[level] = best_ask + (level as f64) * tick_size;
            }

            // Ask sizes (indices 10-19): decreasing with level
            for level in 0..10.min(features.saturating_sub(10)) {
                timestep[10 + level] = (1000 - level * 50) as f64;
            }

            // Bid prices (indices 20-29): descending from best bid
            for level in 0..10.min(features.saturating_sub(20)) {
                timestep[20 + level] = best_bid - (level as f64) * tick_size;
            }

            // Bid sizes (indices 30-39): decreasing with level
            for level in 0..10.min(features.saturating_sub(30)) {
                timestep[30 + level] = (1000 - level * 50) as f64;
            }

            // Derived features (indices 40-47): if present
            if features > 40 {
                timestep[40] = mid_price; // mid-price
                if features > 41 {
                    timestep[41] = spread; // spread
                }
                if features > 42 {
                    timestep[42] = spread / mid_price * 10000.0; // spread_bps
                }
                // Fill remaining with reasonable values
                for f in 43..features {
                    timestep[f] = 500.0 + (f as f64) * 10.0;
                }
            }

            seq_features.push(timestep);

            // Also create a mid-price for each timestep
            mid_prices.push(mid_price);
        }

        sequences.push(Sequence {
            features: seq_features,
            start_timestamp: seq_idx as u64 * 1_000_000_000,
            end_timestamp: (seq_idx + 1) as u64 * 1_000_000_000,
            duration_ns: 1_000_000_000,
            length: window,
        });
    }

    PipelineOutput {
        sequences,
        mid_prices,
        messages_processed: n_sequences * window,
        features_extracted: n_sequences * window,
        sequences_generated: n_sequences,
        stride: 10, // Default stride for tests
        window_size: window,
        multiscale_sequences: None,
        adaptive_stats: None,
    }
}

#[test]
fn test_aligned_export_basic() {
    // Create mock data with 40 raw LOB features (default)
    let output = create_mock_output(100, 100, 40);

    // Create exporter
    let temp_dir = TempDir::new().unwrap();
    let label_config = LabelConfig {
        horizon: 20, // Small horizon for testing
        smoothing_window: 5,
        threshold: 0.001,
    };

    let exporter = AlignedBatchExporter::new(
        temp_dir.path(),
        label_config,
        100, // window_size
        10,  // stride
    );

    // Export
    let result = exporter.export_day("test", &output).unwrap();

    // Verify
    assert!(result.n_sequences > 0, "Should have exported sequences");
    assert_eq!(result.seq_shape, (100, 40), "Shape should be (100, 40)");

    // Check files exist
    assert!(temp_dir.path().join("test_sequences.npy").exists());
    assert!(temp_dir.path().join("test_labels.npy").exists());
    assert!(temp_dir.path().join("test_metadata.json").exists());
}

#[test]
fn test_aligned_export_lengths_match() {
    let output = create_mock_output(50, 100, 40);

    let temp_dir = TempDir::new().unwrap();
    let label_config = LabelConfig {
        horizon: 20,
        smoothing_window: 5,
        threshold: 0.001,
    };

    let exporter = AlignedBatchExporter::new(temp_dir.path(), label_config, 100, 10);

    let result = exporter.export_day("test", &output).unwrap();

    // Load back
    let seq_file = File::open(temp_dir.path().join("test_sequences.npy")).unwrap();
    let sequences: Array3<f32> = Array3::read_npy(seq_file).unwrap();

    let label_file = File::open(temp_dir.path().join("test_labels.npy")).unwrap();
    let labels: Array1<i8> = Array1::read_npy(label_file).unwrap();

    // Critical check: lengths must match!
    assert_eq!(
        sequences.shape()[0],
        labels.len(),
        "Sequences and labels must have same length"
    );

    // Also check reported vs actual
    assert_eq!(sequences.shape()[0], result.n_sequences);
}

#[test]
fn test_aligned_export_shapes() {
    let output = create_mock_output(30, 100, 40);

    let temp_dir = TempDir::new().unwrap();
    let label_config = LabelConfig {
        horizon: 20,
        smoothing_window: 5,
        threshold: 0.001,
    };

    let exporter = AlignedBatchExporter::new(temp_dir.path(), label_config, 100, 10);

    exporter.export_day("test", &output).unwrap();

    // Load back
    let seq_file = File::open(temp_dir.path().join("test_sequences.npy")).unwrap();
    let sequences: Array3<f32> = Array3::read_npy(seq_file).unwrap();

    let label_file = File::open(temp_dir.path().join("test_labels.npy")).unwrap();
    let labels: Array1<i8> = Array1::read_npy(label_file).unwrap();

    // Check shapes
    assert_eq!(sequences.ndim(), 3, "Sequences should be 3D");
    assert_eq!(sequences.shape()[1], 100, "Window size should be 100");
    assert_eq!(sequences.shape()[2], 40, "Features should be 40 (raw LOB)");

    assert_eq!(labels.ndim(), 1, "Labels should be 1D");
}

#[test]
fn test_aligned_export_label_values() {
    let output = create_mock_output(20, 100, 40);

    let temp_dir = TempDir::new().unwrap();
    let label_config = LabelConfig {
        horizon: 20,
        smoothing_window: 5,
        threshold: 0.001,
    };

    let exporter = AlignedBatchExporter::new(temp_dir.path(), label_config, 100, 10);

    exporter.export_day("test", &output).unwrap();

    // Load labels
    let label_file = File::open(temp_dir.path().join("test_labels.npy")).unwrap();
    let labels: Array1<i8> = Array1::read_npy(label_file).unwrap();

    // All labels should be in {-1, 0, 1}
    for &label in labels.iter() {
        assert!(
            (-1..=1).contains(&label),
            "Label {label} not in valid range {{-1, 0, 1}}"
        );
    }
}

#[test]
fn test_aligned_export_no_nan_inf() {
    let output = create_mock_output(20, 100, 40);

    let temp_dir = TempDir::new().unwrap();
    let label_config = LabelConfig {
        horizon: 20,
        smoothing_window: 5,
        threshold: 0.001,
    };

    let exporter = AlignedBatchExporter::new(temp_dir.path(), label_config, 100, 10);

    exporter.export_day("test", &output).unwrap();

    // Load sequences
    let seq_file = File::open(temp_dir.path().join("test_sequences.npy")).unwrap();
    let sequences: Array3<f32> = Array3::read_npy(seq_file).unwrap();

    // Check all values are finite
    for &value in sequences.iter() {
        assert!(value.is_finite(), "Found non-finite value: {value}");
    }
}
