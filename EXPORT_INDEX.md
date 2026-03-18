# Export Index

**Living ledger of all dataset exports.** Updated after every export run.

**Current best classification export:** `nvda_xnas_128feat` (233 days, 128 features, TLOB labels, XNAS ITCH)
**Current best regression export:** `nvda_xnas_128feat_regression` (233 days, 128 features, smoothed-return H10/H60/H300)

**How to read this index:**
- **CURRENT** = actively used in experiments/backtests. Do not delete.
- **ARCHIVED** = experiment completed or failed. Keep for reference. Safe to delete if disk is needed.
- **ORPHAN** = no manifest, no config, no known purpose. Candidate for deletion.

---

## Summary

| Export | Status | Features | Exchange | Labels | Days | Used By |
|--------|--------|----------|----------|--------|------|---------|
| [nvda_xnas_128feat](#nvda_xnas_128feat--current) | CURRENT | 128 | XNAS | TLOB classification | 233 | HMHP 128-feat, XGBoost baselines |
| [nvda_xnas_128feat_regression](#nvda_xnas_128feat_regression--current) | CURRENT | 128 | XNAS | Regression (smoothed) | 233 | TLOB REG-01, HMHP-R, Ridge, GradBoost, R4/R5 backtests |
| [nvda_xnas_128feat_regression_fwd_prices](#nvda_xnas_128feat_regression_fwd_prices--current) | CURRENT | 128 | XNAS | Regression + forward prices | 35 | P0 label-execution validation |
| [nvda_arcx_128feat](#nvda_arcx_128feat--current) | CURRENT | 128 | ARCX | TLOB classification | 233 | HMHP 128-feat ARCX, XGBoost ARCX |
| [nvda_xnas_128feat_regression_pointreturn](#nvda_xnas_128feat_regression_pointreturn--archived) | ARCHIVED | 128 | XNAS | Regression (point-return) | 233 | E2 (IC gate FAILED) |
| [nvda_xnas_128feat_regression_finegrained](#nvda_xnas_128feat_regression_finegrained--archived) | ARCHIVED | 128 | XNAS | Regression (fine-grained) | 233 | E2 derived (IC gate FAILED) |
| [nvda_xnas_128feat_opportunity](#nvda_xnas_128feat_opportunity--archived) | ARCHIVED | 128 | XNAS | Opportunity | 233 | Never trained (wrong objective) |
| [nvda_xnas_128feat_profit8bps](#nvda_xnas_128feat_profit8bps--archived) | ARCHIVED | 128 | XNAS | TLOB (profit-threshold) | 233 | HMHP profit8bps experiment |
| [nvda_xnas_kolm_of_regression](#nvda_xnas_kolm_of_regression--archived) | ARCHIVED | 136 | XNAS | Regression (point-return) | 233 | Kolm OF experiment (IC=0.0001, FAILED) |
| [e3_ic_gate](#e3_ic_gate--archived) | ARCHIVED | 98 | ARCX | Regression (point-return) | 35 | E3 ARCX fine-grained (IC gate FAILED) |
| [arcx_pillar](#arcx_pillar--orphan) | ORPHAN | ? | ARCX | ? | ? | None |
| [raw_lob_full](#raw_lob_full--orphan) | ORPHAN | ? | ? | ? | ? | None |

---

## Current Exports

### nvda_xnas_128feat -- CURRENT

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_full.toml` |
| **Output** | `data/exports/nvda_xnas_128feat/` |
| **Exchange** | XNAS (Nasdaq ITCH) |
| **Features** | 128 (LOB 40 + Derived 8 + MBO 36 + Signals 14 + Experimental 30: inst_v2, volatility, seasonality, mlofi) |
| **Sampling** | event_based, event_count=1000 |
| **Labels** | TLOB classification, horizons=[10, 20, 50, 60, 100, 200, 300], threshold=0.0008 |
| **Normalization** | market_structure_zscore, per-day |
| **Dates** | 2025-02-03 to 2026-01-06 (233 trading days) |
| **Split** | Train: 163 days, Val: 35 days, Test: 35 days |
| **Sequence** | window=100, stride=10 |
| **Schema** | 2.2 |
| **Exported** | 2026-03-12 |
| **Git** | d1cf6853 |

**Used by**: HMHP 128-feat XNAS (59.62% test acc), HMHP 128-feat ARCX, HMHP 40-feat (95.50% DA), XGBoost baselines
**Issues**: None known. Primary classification dataset.

---

### nvda_xnas_128feat_regression -- CURRENT

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_regression.toml` |
| **Output** | `data/exports/nvda_xnas_128feat_regression/` |
| **Exchange** | XNAS (Nasdaq ITCH) |
| **Features** | 128 (same as above) |
| **Sampling** | event_based, event_count=1000 |
| **Labels** | Regression (smoothed-return), horizons=[10, 60, 300], smoothing_window=10 |
| **Normalization** | market_structure_zscore, per-day |
| **Dates** | 2025-02-03 to 2026-01-06 (233 trading days) |
| **Split** | Train: 163 days / 162,999 seq, Val: 35 days / 52,885 seq, Test: 35 days / 50,724 seq |
| **Sequence** | window=100, stride=10 |
| **Schema** | 2.2 |
| **Exported** | 2026-03-15 |
| **Git** | d1cf6853 |

**Used by**: TLOB REG-01 (R2=0.464, IC=0.677), TLOB T=20 ablation, HMHP-R multi-horizon, TemporalRidge (R2=0.324, IC=0.616), TemporalGradBoost (R2=0.397), Backtest R4 (all thresholds), Backtest R5 (hybrid), F1 OFI persistence analysis
**Issues**: Labels are smoothed-average returns. Model R2=0.464 but execution r=0.013 (E1 finding). OFI ACF=0.021 in exported data (F1 finding — event-based sampling destroys persistence).

---

### nvda_xnas_128feat_regression_fwd_prices -- CURRENT

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_regression_fwd_prices.toml` |
| **Output** | `data/exports/nvda_xnas_128feat_regression_fwd_prices/` |
| **Exchange** | XNAS |
| **Features** | 128 + forward_prices.npy (N, k+H+1) float64 USD |
| **Sampling** | event_based, event_count=1000 |
| **Labels** | Regression, horizons=[10, 60, 300] + forward mid-price trajectories |
| **Normalization** | market_structure_zscore, per-day |
| **Dates** | Test split only: 2025-11-14 to 2026-01-06 (35 days) |
| **Split** | Train only (35 test days exported as "train" split for analysis) |
| **Sequence** | window=100, stride=10 |
| **Schema** | 2.2 |
| **Exported** | 2026-03-17 |
| **Git** | d1cf6853 |

**Used by**: P0 label-execution validation (r=0.642, win_rate=69.3%)
**Issues**: None. Special-purpose export for label alignment analysis. Do not use for training.

---

### nvda_arcx_128feat -- CURRENT

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_arcx_128feat_full.toml` |
| **Output** | `data/exports/nvda_arcx_128feat/` |
| **Exchange** | ARCX (NYSE Arca PILLAR) |
| **Features** | 128 |
| **Sampling** | event_based, event_count=1000 |
| **Labels** | TLOB classification, horizons=[10, 20, 50, 60, 100, 200, 300] |
| **Normalization** | market_structure_zscore, per-day |
| **Dates** | 2025-02-03 to 2026-01-06 (233 trading days) |
| **Split** | Train: 163 days, Val: 35 days, Test: 35 days |
| **Sequence** | window=100, stride=10 |
| **Schema** | 2.2 |
| **Exported** | 2026-03-12 |
| **Git** | d1cf6853 |

**Used by**: HMHP 128-feat ARCX (58.79% test acc, 97.21% DA at high conviction)
**Issues**: None. ARCX has thinner book (7,953 shares avg depth vs 18,189 XNAS) → sharper signal per unit.

---

## Archived Exports

### nvda_xnas_128feat_regression_pointreturn -- ARCHIVED

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_regression_pointreturn.toml` |
| **Output** | `data/exports/nvda_xnas_128feat_regression_pointreturn/` |
| **Exchange** | XNAS |
| **Features** | 128 |
| **Sampling** | event_based, event_count=1000 |
| **Labels** | Regression (point_return), horizons=[10, 60, 300] |
| **Dates** | 233 trading days |
| **Split** | Train: 163, Val: 35, Test: 35 |
| **Exported** | 2026-03-15 |

**Used by**: E2 (IC gate FAILED — 0/128 features have IC > 0.05 for point-return labels)
**Issues**: **CONCLUSIVE FAILURE.** OFI features are contemporaneous — they describe what the return IS, not what it WILL BE. No model can predict point returns from LOB/MBO features at H10. See EXPERIMENT_INDEX E2.

---

### nvda_xnas_128feat_regression_finegrained -- ARCHIVED

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_regression_finegrained.toml` |
| **Output** | `data/exports/nvda_xnas_128feat_regression_finegrained/` |
| **Exchange** | XNAS |
| **Features** | 128 |
| **Sampling** | event_based, event_count=200 (5x finer than standard) |
| **Labels** | Regression (smoothed-return), horizons=[1, 2, 5, 10, 20] (fine-grained) |
| **Dates** | 233 trading days |
| **Split** | Train: 163, Val: 35, Test: 35 |
| **Sequence** | window=100, stride=5 |
| **Exported** | 2026-03-16 |

**Used by**: E2 derived experiment (IC gate failed)
**Issues**: Fine-grained sampling (event_count=200) does not improve point-return IC.

---

### nvda_xnas_128feat_opportunity -- ARCHIVED

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_opportunity.toml` |
| **Output** | `data/exports/nvda_xnas_128feat_opportunity/` |
| **Exchange** | XNAS |
| **Features** | 128 |
| **Labels** | Opportunity (peak-return detection), horizons=[10, 20, 50, 60, 100, 200, 300] |
| **Dates** | 233 trading days |
| **Exported** | 2026-03-13 |

**Used by**: Never trained. Wrong objective (magnitude detection instead of readability detection).
**Issues**: Opportunity labeling cancelled per strategic framework: "A 10 bps move at 97% confidence > 50 bps at 60%."

---

### nvda_xnas_128feat_profit8bps -- ARCHIVED

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_128feat_profit_threshold.toml` |
| **Output** | `data/exports/nvda_xnas_128feat_profit8bps/` |
| **Exchange** | XNAS |
| **Features** | 128 |
| **Labels** | TLOB (profit-threshold 8 bps), horizons=[10, 60, 300] |
| **Dates** | 233 trading days |
| **Exported** | 2026-03-14 |

**Used by**: HMHP profit8bps experiment (completed, not indexed in EXPERIMENT_INDEX)
**Issues**: Profit-threshold labeling does not improve backtest results over standard TLOB.

---

### nvda_xnas_kolm_of_regression -- ARCHIVED

| Field | Value |
|-------|-------|
| **Config** | `configs/nvda_xnas_kolm_of_regression.toml` |
| **Output** | `data/exports/nvda_xnas_kolm_of_regression/` |
| **Exchange** | XNAS |
| **Features** | 136 (98 + inst_v2(8) + vol(6) + season(4) + kolm_of(20)) |
| **Sampling** | event_based, event_count=100 |
| **Labels** | Regression (point-return), horizons=[1, 2, 3, 5] |
| **Dates** | 233 trading days |
| **Split** | Train: 163, Val: 35, Test: 35 |
| **Sequence** | window=20, stride=10 |
| **Exported** | 2026-03-16 |

**Used by**: Kolm OF experiment (EXPERIMENT_INDEX, 2026-03-17)
**Issues**: **CONCLUSIVE FAILURE.** Kolm per-level OF (20-dim) has IC=0.0001 for point-returns. Cumulative-per-window OF destroys the per-event temporal dynamics that Kolm's LSTM exploits. Scalar OFI works because depth normalization creates mean-reverting signal.

---

### e3_ic_gate -- ARCHIVED

| Field | Value |
|-------|-------|
| **Config** | `configs/e3_ic_gate.toml` |
| **Output** | `data/exports/e3_ic_gate/` |
| **Exchange** | ARCX (NYSE Arca PILLAR) |
| **Features** | 98 (no experimental) |
| **Sampling** | event_based, event_count=100 |
| **Labels** | Regression (point-return), horizons=[10, 60, 300] |
| **Dates** | Test split only: 35 days |
| **Sequence** | window=20, stride=5 |
| **Exported** | 2026-03-17 |

**Used by**: E3 ARCX fine-grained (EXPERIMENT_INDEX, 2026-03-17)
**Issues**: **CONCLUSIVE FAILURE.** 0/93 features have IC > 0.05 for point-return at any horizon. ARCX + fine-grained = slightly better IC (0.035 vs 0.025 XNAS) but still far below threshold. Eliminates ARCX + fine-grained hypothesis.

---

## Orphan Exports

### arcx_pillar -- ORPHAN

No `dataset_manifest.json`. No matching config. Contains only 2 files. **Candidate for deletion.**

### raw_lob_full -- ORPHAN

No `dataset_manifest.json`. No matching config. Contains ~703 files (raw LOB snapshots). **Candidate for deletion** — superseded by properly exported datasets.

---

## Config → Export Mapping

| Config | Export Directory | Status |
|--------|-----------------|--------|
| `nvda_xnas_128feat_full.toml` | `nvda_xnas_128feat` | CURRENT |
| `nvda_xnas_128feat_regression.toml` | `nvda_xnas_128feat_regression` | CURRENT |
| `nvda_xnas_128feat_regression_fwd_prices.toml` | `nvda_xnas_128feat_regression_fwd_prices` | CURRENT |
| `nvda_arcx_128feat_full.toml` | `nvda_arcx_128feat` | CURRENT |
| `nvda_xnas_128feat_regression_pointreturn.toml` | `nvda_xnas_128feat_regression_pointreturn` | ARCHIVED |
| `nvda_xnas_128feat_regression_finegrained.toml` | `nvda_xnas_128feat_regression_finegrained` | ARCHIVED |
| `nvda_xnas_128feat_opportunity.toml` | `nvda_xnas_128feat_opportunity` | ARCHIVED |
| `nvda_xnas_128feat_profit_threshold.toml` | `nvda_xnas_128feat_profit8bps` | ARCHIVED |
| `nvda_xnas_kolm_of_regression.toml` | `nvda_xnas_kolm_of_regression` | ARCHIVED |
| `e3_ic_gate.toml` | `e3_ic_gate` | ARCHIVED |

**Unmapped configs** (22 configs have no active export on disk — historical or unused):
`nvda_98feat.toml`, `nvda_98feat_v2.toml`, `nvda_98feat_full.toml`, `nvda_84feat_baseline.toml`, `nvda_116feat_full_analysis.toml`, `nvda_11month_complete.toml`, `nvda_11month_triple_barrier.toml`, `nvda_11month_triple_barrier_calibrated.toml`, `nvda_11month_triple_barrier_volscaled.toml`, `nvda_arcx_98feat.toml`, `nvda_arcx_98feat_multi_horizon.toml`, `nvda_arcx_128feat_regression_finegrain.toml`, `nvda_balanced.toml`, `nvda_bigmove_detection.toml`, `nvda_extended_multi_horizon.toml`, `nvda_multi_horizon.toml`, `nvda_spread_adaptive.toml`, `nvda_tlob_raw_v2.toml`, `nvda_tlob_repo_v1.toml`, `nvda_tlob_repo_v2.toml`, `nvda_triple_barrier.toml`, `template_multi_symbol.toml`

---

## Key Findings Across Exports

1. **All 128-feature exports use event_count=1000** — this destroys OFI persistence (ACF=0.021). Future exports should use `time_based` sampling.
2. **Point-return labels have zero signal** — confirmed across XNAS (E2), ARCX (E3), and fine-grained (E2 derived). Do not create more point-return exports.
3. **Smoothed-return labels have strong signal** (IC=0.677) but weak execution correlation (r=0.013 prediction-to-price). The label-execution gap is the primary bottleneck.
4. **ARCX has sharper per-unit signal** (OFI r=0.688 vs XNAS 0.577) but same fundamental limitations.
5. **Kolm 20-dim OF adds zero value** in cumulative-per-window architecture (IC=0.0001).

---

*Last updated: 2026-03-18. This index should be updated after every export run.*
