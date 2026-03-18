//! NPY file writing for sequences, labels, and tensor-formatted exports.

use super::AlignedBatchExporter;
use crate::export::tensor_format::TensorFormat;
use mbo_lob_reconstructor::Result;
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::WriteNpyExt;
use std::fs::File;
use std::path::Path;

impl AlignedBatchExporter {
    /// Export sequences with optional tensor formatting.
    pub(super) fn export_sequences_with_format(
        &self,
        sequences: &[Vec<Vec<f64>>],
        shape: (usize, usize),
        path: &Path,
    ) -> Result<()> {
        match &self.tensor_format {
            None | Some(TensorFormat::Flat) => self.export_sequences(sequences, shape, path),
            Some(TensorFormat::DeepLOB { levels }) => {
                self.export_sequences_deeplob(sequences, *levels, path)
            }
            Some(TensorFormat::HLOB { levels }) => {
                self.export_sequences_hlob(sequences, *levels, path)
            }
            Some(TensorFormat::Image {
                channels,
                height,
                width,
            }) => self.export_sequences_image(sequences, *channels, *height, *width, path),
        }
    }

    /// Export sequences as 3D numpy array: [n_sequences, window_size, n_features]
    pub(super) fn export_sequences(
        &self,
        sequences: &[Vec<Vec<f64>>],
        shape: (usize, usize),
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let (window, features) = shape;

        let flat: Vec<f32> = sequences
            .iter()
            .flat_map(|seq| seq.iter())
            .flat_map(|timestep| timestep.iter().copied().map(|x| x as f32))
            .collect();

        let array = Array3::from_shape_vec((n_seq, window, features), flat)
            .map_err(|e| format!("Failed to create 3D array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write sequences: {e}"))?;

        println!(
            "  ✅ Sequences: {} [{} × {} × {}]",
            path.display(),
            n_seq,
            window,
            features
        );

        Ok(())
    }

    /// Export labels as 1D numpy array: [n_sequences]
    pub(super) fn export_labels(&self, labels: &[i8], path: &Path) -> Result<()> {
        let array = Array1::from_vec(labels.to_vec());

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write labels: {e}"))?;

        let up = labels.iter().filter(|&&l| l == 1).count();
        let down = labels.iter().filter(|&&l| l == -1).count();
        let stable = labels.iter().filter(|&&l| l == 0).count();

        println!("  ✅ Labels: {} [{} samples]", path.display(), labels.len());
        println!(
            "     Distribution: Up={} ({:.1}%), Down={} ({:.1}%), Stable={} ({:.1}%)",
            up,
            100.0 * up as f64 / labels.len() as f64,
            down,
            100.0 * down as f64 / labels.len() as f64,
            stable,
            100.0 * stable as f64 / labels.len() as f64
        );

        Ok(())
    }

    /// Export multi-horizon labels as 2D numpy array: [n_sequences, num_horizons]
    pub(super) fn export_multi_horizon_labels(
        &self,
        label_matrix: &[Vec<i8>],
        path: &Path,
    ) -> Result<()> {
        let n_seq = label_matrix.len();
        let n_horizons = label_matrix.first().map(|r| r.len()).unwrap_or(0);

        let flat: Vec<i8> = label_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let array = Array2::from_shape_vec((n_seq, n_horizons), flat)
            .map_err(|e| format!("Failed to create 2D label array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write multi-horizon labels: {e}"))?;

        println!(
            "  ✅ Multi-horizon labels: {} [{} × {}]",
            path.display(),
            n_seq,
            n_horizons
        );

        Ok(())
    }

    /// Export regression labels as 2D float64 numpy array: [n_sequences, num_horizons]
    ///
    /// Values are continuous forward returns in bps (basis points).
    pub(super) fn export_regression_labels(
        &self,
        label_matrix: &[Vec<f64>],
        path: &Path,
    ) -> Result<()> {
        let n_seq = label_matrix.len();
        let n_horizons = label_matrix.first().map(|r| r.len()).unwrap_or(0);

        let flat: Vec<f64> = label_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let array = Array2::from_shape_vec((n_seq, n_horizons), flat)
            .map_err(|e| format!("Failed to create 2D regression label array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write regression labels: {e}"))?;

        let all_vals: Vec<f64> = label_matrix.iter().flat_map(|r| r.iter().copied()).collect();
        let mean = all_vals.iter().sum::<f64>() / all_vals.len().max(1) as f64;
        let abs_mean = all_vals.iter().map(|v| v.abs()).sum::<f64>() / all_vals.len().max(1) as f64;

        println!(
            "  ✅ Regression labels: {} [{} × {}] (mean={:.2} bps, |mean|={:.2} bps)",
            path.display(),
            n_seq,
            n_horizons,
            mean,
            abs_mean,
        );

        Ok(())
    }

    /// Export forward mid-price trajectories as 2D float64 numpy array.
    ///
    /// Shape: `[n_sequences, n_columns]` where n_columns = smoothing_window + max_horizon + 1.
    /// Values are raw USD mid_prices (NOT normalized).
    ///
    /// Column layout (with smoothing_window_offset = k):
    /// - Column 0: mid_price at t-k (k events before sequence end)
    /// - Column k: mid_price at t (base price at sequence end / prediction point)
    /// - Column k+h: mid_price at t+h (h events forward)
    ///
    /// This enables Python-side computation of any label type from the same aligned samples.
    pub(super) fn export_forward_prices(
        &self,
        trajectories: &[Vec<f64>],
        path: &Path,
    ) -> Result<()> {
        let n_seq = trajectories.len();
        let n_cols = trajectories.first().map(|r| r.len()).unwrap_or(0);

        if n_seq == 0 || n_cols == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Empty forward price trajectories",
            )
            .into());
        }

        let flat: Vec<f64> = trajectories
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let array = Array2::from_shape_vec((n_seq, n_cols), flat)
            .map_err(|e| format!("Failed to create forward prices array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write forward prices: {e}"))?;

        // Compute basic statistics for logging
        let base_col = if n_cols > 0 {
            trajectories.iter().map(|r| r[0]).sum::<f64>() / n_seq as f64
        } else {
            0.0
        };

        println!(
            "  ✅ Forward prices: {} [{} × {}] (mean_base=${:.2})",
            path.display(),
            n_seq,
            n_cols,
            base_col,
        );

        Ok(())
    }

    /// Export sequences in DeepLOB format: (N, T, 4, L)
    ///
    /// Channels: [ask_prices, ask_volumes, bid_prices, bid_volumes]
    pub(super) fn export_sequences_deeplob(
        &self,
        sequences: &[Vec<Vec<f64>>],
        levels: usize,
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let n_timesteps = sequences.first().map(|s| s.len()).unwrap_or(0);

        let mut data: Vec<f32> = Vec::with_capacity(n_seq * n_timesteps * 4 * levels);

        for seq in sequences {
            for timestep in seq {
                for l in 0..levels {
                    data.push(timestep.get(l).copied().unwrap_or(0.0) as f32);
                }
                for l in 0..levels {
                    data.push(timestep.get(levels + l).copied().unwrap_or(0.0) as f32);
                }
                for l in 0..levels {
                    data.push(timestep.get(2 * levels + l).copied().unwrap_or(0.0) as f32);
                }
                for l in 0..levels {
                    data.push(timestep.get(3 * levels + l).copied().unwrap_or(0.0) as f32);
                }
            }
        }

        let array = Array4::from_shape_vec((n_seq, n_timesteps, 4, levels), data)
            .map_err(|e| format!("Failed to create DeepLOB array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write DeepLOB sequences: {e}"))?;

        println!(
            "  ✅ DeepLOB sequences: {} [{} × {} × 4 × {}]",
            path.display(),
            n_seq,
            n_timesteps,
            levels
        );

        Ok(())
    }

    /// Export sequences in HLOB format: (N, T, L, 4)
    ///
    /// Per-level features: [ask_price, ask_volume, bid_price, bid_volume]
    pub(super) fn export_sequences_hlob(
        &self,
        sequences: &[Vec<Vec<f64>>],
        levels: usize,
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let n_timesteps = sequences.first().map(|s| s.len()).unwrap_or(0);

        let mut data: Vec<f32> = Vec::with_capacity(n_seq * n_timesteps * levels * 4);

        for seq in sequences {
            for timestep in seq {
                for l in 0..levels {
                    data.push(timestep.get(l).copied().unwrap_or(0.0) as f32);
                    data.push(timestep.get(levels + l).copied().unwrap_or(0.0) as f32);
                    data.push(timestep.get(2 * levels + l).copied().unwrap_or(0.0) as f32);
                    data.push(timestep.get(3 * levels + l).copied().unwrap_or(0.0) as f32);
                }
            }
        }

        let array = Array4::from_shape_vec((n_seq, n_timesteps, levels, 4), data)
            .map_err(|e| format!("Failed to create HLOB array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write HLOB sequences: {e}"))?;

        println!(
            "  ✅ HLOB sequences: {} [{} × {} × {} × 4]",
            path.display(),
            n_seq,
            n_timesteps,
            levels
        );

        Ok(())
    }

    /// Export sequences in Image format: (N, T, C, H, W)
    pub(super) fn export_sequences_image(
        &self,
        sequences: &[Vec<Vec<f64>>],
        channels: usize,
        height: usize,
        width: usize,
        path: &Path,
    ) -> Result<()> {
        let n_seq = sequences.len();
        let n_timesteps = sequences.first().map(|s| s.len()).unwrap_or(0);

        let expected_features = channels * height * width;

        let mut data: Vec<f32> = Vec::with_capacity(n_seq * n_timesteps * expected_features);

        for seq in sequences {
            for timestep in seq {
                for i in 0..expected_features {
                    data.push(timestep.get(i).copied().unwrap_or(0.0) as f32);
                }
            }
        }

        let shape = ndarray::IxDyn(&[n_seq, n_timesteps, channels, height, width]);
        let array = ndarray::ArrayD::from_shape_vec(shape, data)
            .map_err(|e| format!("Failed to create Image array: {e}"))?;

        let mut file = File::create(path)?;
        array
            .write_npy(&mut file)
            .map_err(|e| format!("Failed to write Image sequences: {e}"))?;

        println!(
            "  ✅ Image sequences: {} [{} × {} × {} × {} × {}]",
            path.display(),
            n_seq,
            n_timesteps,
            channels,
            height,
            width
        );

        Ok(())
    }
}
