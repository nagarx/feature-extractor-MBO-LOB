//! Triple Barrier labeling strategy.
//!
//! López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3.

use crate::export_aligned::types::{LabelEncoding, LabelingResult};
use crate::export_aligned::AlignedBatchExporter;
use crate::labeling::{BarrierLabel, TripleBarrierConfig, TripleBarrierLabeler};
use crate::pipeline::PipelineOutput;
use mbo_lob_reconstructor::Result;
use std::collections::HashMap;

impl AlignedBatchExporter {
    /// Triple Barrier labeling strategy with optional volatility scaling.
    pub(crate) fn labeling_triple_barrier(
        &self,
        output: &PipelineOutput,
        configs: &[TripleBarrierConfig],
        horizons: &[usize],
    ) -> Result<LabelingResult> {
        let is_multi_horizon = configs.len() > 1;

        // Volatility-adaptive barrier scaling (Schema 3.3+)
        let (effective_configs, vol_metadata) = if let Some((ref_vol, floor, cap)) =
            self.volatility_scaling
        {
            let daily_vol = Self::compute_daily_volatility(&output.mid_prices);
            let raw_scale = if ref_vol > 0.0 && daily_vol > 0.0 {
                daily_vol / ref_vol
            } else {
                1.0
            };
            let scale = raw_scale.clamp(floor, cap);

            println!(
                "  📊 Volatility scaling: daily_vol={:.6} ({:.2} bps), ref={:.6} ({:.2} bps), scale={:.3}x{}",
                daily_vol,
                daily_vol * 10000.0,
                ref_vol,
                ref_vol * 10000.0,
                scale,
                if raw_scale < floor { " [FLOOR]" } else if raw_scale > cap { " [CAP]" } else { "" }
            );

            let scaled: Vec<TripleBarrierConfig> = configs
                .iter()
                .map(|cfg| {
                    let mut scaled_cfg = cfg.clone();
                    scaled_cfg.profit_target_pct *= scale;
                    scaled_cfg.stop_loss_pct *= scale;
                    scaled_cfg
                })
                .collect();

            let meta = Some(serde_json::json!({
                "enabled": true,
                "daily_volatility": daily_vol,
                "daily_volatility_bps": daily_vol * 10000.0,
                "reference_volatility": ref_vol,
                "reference_volatility_bps": ref_vol * 10000.0,
                "raw_scaling_factor": raw_scale,
                "clamped_scaling_factor": scale,
                "volatility_floor": floor,
                "volatility_cap": cap,
                "was_clamped": raw_scale < floor || raw_scale > cap,
            }));

            (scaled, meta)
        } else {
            (configs.to_vec(), None)
        };

        println!(
            "  📊 Generating TRIPLE BARRIER labels for {} horizon(s): {:?}",
            horizons.len(),
            horizons
        );
        if !effective_configs.is_empty() {
            let first_config = &effective_configs[0];
            println!(
                "    Profit target: {:.4}% ({:.1} bps)",
                first_config.profit_target_pct * 100.0,
                first_config.profit_target_pct * 10000.0
            );
            println!(
                "    Stop-loss: {:.4}% ({:.1} bps)",
                first_config.stop_loss_pct * 100.0,
                first_config.stop_loss_pct * 10000.0
            );
            println!(
                "    Risk/Reward ratio: {:.2}:1",
                first_config.risk_reward_ratio()
            );
            println!(
                "    Break-even win rate: {:.1}%",
                first_config.breakeven_win_rate() * 100.0
            );
        }

        let mut all_labels: Vec<Vec<(usize, BarrierLabel, usize, f64)>> = Vec::new();

        for (config, &horizon) in effective_configs.iter().zip(horizons.iter()) {
            let mut labeler = TripleBarrierLabeler::new(config.clone());
            labeler.add_prices(&output.mid_prices);
            let labels = labeler.generate_labels()?;
            let stats = labeler.compute_stats(&labels);

            let pt_rate = stats.profit_target_count as f64 / stats.total.max(1) as f64;
            let sl_rate = stats.stop_loss_count as f64 / stats.total.max(1) as f64;
            let to_rate = stats.timeout_count as f64 / stats.total.max(1) as f64;

            println!(
                "    Horizon {}: {} labels | PT={:.1}%, SL={:.1}%, TO={:.1}% | Win rate={:.1}%",
                horizon,
                labels.len(),
                pt_rate * 100.0,
                sl_rate * 100.0,
                to_rate * 100.0,
                stats.win_rate() * 100.0
            );

            self.validate_class_balance(
                horizon,
                config.profit_target_pct,
                config.stop_loss_pct,
                pt_rate,
                sl_rate,
                to_rate,
            );

            all_labels.push(labels);
        }

        let (label_indices, label_matrix, combined_dist) =
            Self::build_triple_barrier_label_matrix(&all_labels)?;

        println!(
            "    Valid aligned indices: {} (intersection of all horizons)",
            label_indices.len()
        );

        let first_config = &effective_configs[0];

        let per_horizon_barriers: Vec<serde_json::Value> = effective_configs
            .iter()
            .zip(horizons.iter())
            .map(|(cfg, &h)| {
                serde_json::json!({
                    "horizon": h,
                    "profit_target_pct": cfg.profit_target_pct,
                    "profit_target_bps": cfg.profit_target_pct * 10000.0,
                    "stop_loss_pct": cfg.stop_loss_pct,
                    "stop_loss_bps": cfg.stop_loss_pct * 10000.0,
                    "risk_reward_ratio": cfg.risk_reward_ratio(),
                })
            })
            .collect();

        let base_barriers: Vec<serde_json::Value> = configs
            .iter()
            .zip(horizons.iter())
            .map(|(cfg, &h)| {
                serde_json::json!({
                    "horizon": h,
                    "profit_target_pct": cfg.profit_target_pct,
                    "profit_target_bps": cfg.profit_target_pct * 10000.0,
                    "stop_loss_pct": cfg.stop_loss_pct,
                    "stop_loss_bps": cfg.stop_loss_pct * 10000.0,
                })
            })
            .collect();

        let strategy_metadata = serde_json::json!({
            "strategy": "triple_barrier",
            "horizons": horizons,
            "profit_target_pct": first_config.profit_target_pct,
            "profit_target_bps": first_config.profit_target_pct * 10000.0,
            "stop_loss_pct": first_config.stop_loss_pct,
            "stop_loss_bps": first_config.stop_loss_pct * 10000.0,
            "risk_reward_ratio": first_config.risk_reward_ratio(),
            "breakeven_win_rate": first_config.breakeven_win_rate(),
            "timeout_strategy": format!("{:?}", first_config.timeout_strategy),
            "min_holding_period": first_config.min_holding_period,
            "per_horizon_barriers": per_horizon_barriers,
            "base_barriers": base_barriers,
            "volatility_scaling": vol_metadata,
            "label_encoding": {
                "format": "class_index_int8",
                "values": {
                    "0": "StopLoss",
                    "1": "Timeout",
                    "2": "ProfitTarget",
                },
                "note": "Ready for PyTorch CrossEntropyLoss (class indices 0, 1, 2)",
            },
            "research_reference": "López de Prado, M. (2018). Advances in Financial Machine Learning, Chapter 3."
        });

        let horizons_config = serde_json::json!({
            "labeling_strategy": "triple_barrier",
            "horizons": horizons,
            "profit_target_pct": first_config.profit_target_pct,
            "stop_loss_pct": first_config.stop_loss_pct,
            "risk_reward_ratio": first_config.risk_reward_ratio(),
            "timeout_strategy": format!("{:?}", first_config.timeout_strategy),
            "per_horizon": effective_configs.iter().zip(horizons.iter()).map(|(cfg, &h)| {
                serde_json::json!({
                    "horizon": h,
                    "profit_target_bps": cfg.profit_target_pct * 10000.0,
                    "stop_loss_bps": cfg.stop_loss_pct * 10000.0,
                    "risk_reward_ratio": cfg.risk_reward_ratio(),
                })
            }).collect::<Vec<_>>(),
            "volatility_scaling": vol_metadata,
        });

        Ok(LabelingResult {
            label_indices,
            label_matrix,
            encoding: LabelEncoding::TripleBarrierClassIndex,
            distribution: combined_dist,
            strategy_name: "triple_barrier",
            strategy_metadata,
            is_multi_horizon,
            horizons_config: Some(horizons_config),
        })
    }

    /// Compute realized volatility from a price series.
    ///
    /// RV = std(ln(P_t / P_{t-1})), Andersen & Bollerslev (1998).
    fn compute_daily_volatility(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let log_returns: Vec<f64> = prices
            .windows(2)
            .filter(|w| w[0] > 0.0 && w[1] > 0.0)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if log_returns.len() < 2 {
            return 0.0;
        }

        let n = log_returns.len() as f64;
        let mean = log_returns.iter().sum::<f64>() / n;
        let var = log_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    }

    /// Build label matrix for Triple Barrier labeling.
    fn build_triple_barrier_label_matrix(
        all_labels: &[Vec<(usize, BarrierLabel, usize, f64)>],
    ) -> Result<(Vec<usize>, Vec<Vec<i8>>, HashMap<String, usize>)> {
        use std::collections::BTreeSet;

        let mut valid_indices: Option<BTreeSet<usize>> = None;

        for horizon_labels in all_labels {
            let indices: BTreeSet<usize> =
                horizon_labels.iter().map(|(idx, _, _, _)| *idx).collect();
            valid_indices = match valid_indices {
                None => Some(indices),
                Some(prev) => Some(prev.intersection(&indices).cloned().collect()),
            };
        }

        let valid_indices: Vec<usize> = valid_indices.unwrap_or_default().into_iter().collect();

        if valid_indices.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No valid indices found across all horizons",
            )
            .into());
        }

        let mut horizon_lookups: Vec<HashMap<usize, BarrierLabel>> = Vec::new();
        for horizon_labels in all_labels {
            let lookup: HashMap<usize, BarrierLabel> = horizon_labels
                .iter()
                .map(|(idx, label, _, _)| (*idx, *label))
                .collect();
            horizon_lookups.push(lookup);
        }

        let mut label_matrix = Vec::with_capacity(valid_indices.len());
        let mut dist = HashMap::new();
        dist.insert("StopLoss".to_string(), 0);
        dist.insert("Timeout".to_string(), 0);
        dist.insert("ProfitTarget".to_string(), 0);

        for &idx in &valid_indices {
            let mut row = Vec::with_capacity(all_labels.len());
            for lookup in &horizon_lookups {
                let label = lookup.get(&idx).unwrap_or(&BarrierLabel::Timeout);
                row.push(label.as_class_index() as i8);
                *dist.entry(label.name().to_string()).or_insert(0) += 1;
            }
            label_matrix.push(row);
        }

        Ok((valid_indices, label_matrix, dist))
    }
}
