#!/usr/bin/env python3
"""
Triple Barrier Threshold Calibration Tool.

This tool analyzes actual price movement in exported data to recommend
Triple Barrier thresholds that will produce balanced class distributions.

The fundamental insight is that barrier thresholds must be calibrated to
the actual price dynamics of the specific asset and time horizons being used.
Arbitrary thresholds often lead to severe class imbalance (mostly Timeout).

Supports two modes:
  1. Global calibration (default): Single set of barriers for all days.
  2. Per-day volatility analysis (--per-day): Computes per-day realized
     volatility and recommends volatility-scaled barriers so each day
     produces balanced class distributions regardless of market regime.

Design Principles (RULE.md):
- Data-driven: All recommendations based on actual price statistics
- Configurable: Target class distributions can be specified
- Reproducible: Same data produces same recommendations
- Documented: Clear explanation of methodology

Usage:
    # Analyze existing export and get recommendations
    python tools/calibrate_triple_barrier.py \\
        --data-dir ../data/exports/nvda_11month_triple_barrier \\
        --horizons 50 100 200
    
    # Specify target class balance
    python tools/calibrate_triple_barrier.py \\
        --data-dir ../data/exports/nvda_11month_triple_barrier \\
        --horizons 50 100 200 \\
        --target-pt-rate 0.25 \\
        --target-sl-rate 0.25

    # Per-day volatility analysis for dynamic barrier scaling
    python tools/calibrate_triple_barrier.py \\
        --data-dir ../data/exports/nvda_11month_triple_barrier \\
        --horizons 50 100 200 \\
        --per-day \\
        --output volatility_calibration.json

Reference:
    López de Prado (2018), "Advances in Financial Machine Learning", Ch. 3
    - Triple Barrier method for labeling trading outcomes
    - Recommends calibrating barriers to market volatility

Author: HFT Pipeline Team
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Feature indices in 98-feature schema
MID_PRICE_IDX = 40  # Mid-price is derived feature at index 40

# Default target class rates (sum should be 1.0)
DEFAULT_TARGET_PT_RATE = 0.30  # ProfitTarget
DEFAULT_TARGET_SL_RATE = 0.30  # StopLoss
DEFAULT_TARGET_TO_RATE = 0.40  # Timeout (implicit: 1 - PT - SL)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PriceMovementStats:
    """Statistics for price movement over a specific horizon."""
    
    horizon: int
    """Horizon in ticks."""
    
    num_samples: int
    """Number of return observations."""
    
    mean_abs_return_bps: float
    """Mean absolute return in basis points."""
    
    std_return_bps: float
    """Standard deviation of returns in basis points."""
    
    percentiles: Dict[int, float] = field(default_factory=dict)
    """Percentiles of absolute returns (e.g., {10: 5.2, 25: 12.1, ...})."""
    
    def get_threshold_for_rate(self, target_rate: float) -> float:
        """
        Get the threshold (in bps) that would capture approximately target_rate
        of movements as "hits" (either PT or SL).
        
        For example, if target_rate=0.30, this returns the threshold where
        ~30% of absolute returns exceed that threshold.
        
        Args:
            target_rate: Target rate of barrier hits (0 to 1)
            
        Returns:
            Threshold in basis points
        """
        # target_rate=0.30 means we want 30% to exceed the threshold
        # This corresponds to the (100 - 30) = 70th percentile
        percentile = int((1 - target_rate) * 100)
        
        # Interpolate if exact percentile not available
        available = sorted(self.percentiles.keys())
        
        if percentile in self.percentiles:
            return self.percentiles[percentile]
        
        # Linear interpolation
        lower = max(p for p in available if p <= percentile)
        upper = min(p for p in available if p >= percentile)
        
        if lower == upper:
            return self.percentiles[lower]
        
        # Interpolate
        ratio = (percentile - lower) / (upper - lower)
        return self.percentiles[lower] + ratio * (self.percentiles[upper] - self.percentiles[lower])


@dataclass
class BarrierRecommendation:
    """Recommended barrier thresholds for a specific horizon."""
    
    horizon: int
    """Horizon in ticks."""
    
    profit_target_bps: float
    """Recommended profit target in basis points."""
    
    stop_loss_bps: float
    """Recommended stop loss in basis points."""
    
    risk_reward_ratio: float
    """Profit target / stop loss ratio."""
    
    expected_pt_rate: float
    """Expected ProfitTarget class rate with these thresholds."""
    
    expected_sl_rate: float
    """Expected StopLoss class rate with these thresholds."""
    
    expected_to_rate: float
    """Expected Timeout class rate with these thresholds."""
    
    def to_dict(self) -> Dict:
        return {
            'horizon': self.horizon,
            'profit_target_bps': round(self.profit_target_bps, 2),
            'stop_loss_bps': round(self.stop_loss_bps, 2),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'expected_class_rates': {
                'profit_target': round(self.expected_pt_rate, 3),
                'stop_loss': round(self.expected_sl_rate, 3),
                'timeout': round(self.expected_to_rate, 3),
            }
        }


@dataclass 
class CalibrationResult:
    """Complete calibration result with recommendations for all horizons."""
    
    data_source: str
    """Path to the data used for calibration."""
    
    num_days_analyzed: int
    """Number of trading days analyzed."""
    
    num_samples: int
    """Total number of price observations."""
    
    price_range: Tuple[float, float]
    """Min and max prices observed."""
    
    mean_price: float
    """Mean price across observations."""
    
    movement_stats: Dict[int, PriceMovementStats]
    """Price movement statistics by horizon."""
    
    recommendations: Dict[int, BarrierRecommendation]
    """Barrier recommendations by horizon."""
    
    target_pt_rate: float
    """Target ProfitTarget rate used for recommendations."""
    
    target_sl_rate: float
    """Target StopLoss rate used for recommendations."""
    
    def print_report(self) -> None:
        """Print human-readable calibration report."""
        print()
        print("=" * 70)
        print("TRIPLE BARRIER CALIBRATION REPORT")
        print("=" * 70)
        print()
        print(f"Data source: {self.data_source}")
        print(f"Days analyzed: {self.num_days_analyzed}")
        print(f"Price observations: {self.num_samples:,}")
        print(f"Price range: ${self.price_range[0]:.2f} - ${self.price_range[1]:.2f}")
        print(f"Mean price: ${self.mean_price:.2f}")
        print()
        print(f"Target class distribution:")
        print(f"  ProfitTarget: {self.target_pt_rate * 100:.0f}%")
        print(f"  StopLoss:     {self.target_sl_rate * 100:.0f}%")
        print(f"  Timeout:      {(1 - self.target_pt_rate - self.target_sl_rate) * 100:.0f}%")
        print()
        
        # Movement statistics
        print("-" * 70)
        print("PRICE MOVEMENT STATISTICS")
        print("-" * 70)
        print()
        print(f"{'Horizon':<10} {'Mean |Δ|':<12} {'Std Δ':<12} {'P25':<10} {'P50':<10} {'P75':<10}")
        print(f"{'(ticks)':<10} {'(bps)':<12} {'(bps)':<12} {'(bps)':<10} {'(bps)':<10} {'(bps)':<10}")
        print("-" * 70)
        
        for horizon in sorted(self.movement_stats.keys()):
            stats = self.movement_stats[horizon]
            print(f"{horizon:<10} {stats.mean_abs_return_bps:<12.2f} {stats.std_return_bps:<12.2f} "
                  f"{stats.percentiles.get(25, 0):<10.2f} {stats.percentiles.get(50, 0):<10.2f} "
                  f"{stats.percentiles.get(75, 0):<10.2f}")
        print()
        
        # Recommendations
        print("-" * 70)
        print("RECOMMENDED BARRIER THRESHOLDS")
        print("-" * 70)
        print()
        print(f"{'Horizon':<10} {'PT (bps)':<12} {'SL (bps)':<12} {'R:R':<8} {'Exp PT%':<10} {'Exp SL%':<10} {'Exp TO%':<10}")
        print("-" * 70)
        
        for horizon in sorted(self.recommendations.keys()):
            rec = self.recommendations[horizon]
            print(f"{horizon:<10} {rec.profit_target_bps:<12.2f} {rec.stop_loss_bps:<12.2f} "
                  f"{rec.risk_reward_ratio:<8.2f} {rec.expected_pt_rate * 100:<10.1f} "
                  f"{rec.expected_sl_rate * 100:<10.1f} {rec.expected_to_rate * 100:<10.1f}")
        print()
        
        # Config snippet
        print("-" * 70)
        print("TOML CONFIG SNIPPET")
        print("-" * 70)
        print()
        
        # Use first horizon's recommendation for global settings
        first_h = min(self.recommendations.keys())
        rec = self.recommendations[first_h]
        
        print("[labeling]")
        print('strategy = "TripleBarrier"')
        print(f"profit_target_pct = {rec.profit_target_bps / 10000:.6f}  # {rec.profit_target_bps:.1f} bps")
        print(f"stop_loss_pct = {rec.stop_loss_bps / 10000:.6f}  # {rec.stop_loss_bps:.1f} bps")
        print(f"max_horizons = {list(self.recommendations.keys())}")
        print('timeout_strategy = "LabelAsTimeout"')
        print()
        
        # Per-horizon recommendations as comments
        print("# Per-horizon recommendations (if using horizon-specific barriers):")
        for horizon in sorted(self.recommendations.keys()):
            rec = self.recommendations[horizon]
            print(f"# H{horizon}: PT={rec.profit_target_bps:.1f}bps, SL={rec.stop_loss_bps:.1f}bps "
                  f"(expect {rec.expected_pt_rate*100:.0f}%/{rec.expected_sl_rate*100:.0f}%/{rec.expected_to_rate*100:.0f}%)")
    
    def to_json(self, path: Path) -> None:
        """Save calibration result to JSON."""
        data = {
            'data_source': self.data_source,
            'num_days_analyzed': self.num_days_analyzed,
            'num_samples': self.num_samples,
            'price_range': list(self.price_range),
            'mean_price': self.mean_price,
            'target_class_rates': {
                'profit_target': self.target_pt_rate,
                'stop_loss': self.target_sl_rate,
                'timeout': 1 - self.target_pt_rate - self.target_sl_rate,
            },
            'movement_stats': {
                str(h): {
                    'horizon': stats.horizon,
                    'num_samples': stats.num_samples,
                    'mean_abs_return_bps': round(stats.mean_abs_return_bps, 2),
                    'std_return_bps': round(stats.std_return_bps, 2),
                    'percentiles': {str(k): round(v, 2) for k, v in stats.percentiles.items()},
                }
                for h, stats in self.movement_stats.items()
            },
            'recommendations': {
                str(h): rec.to_dict()
                for h, rec in self.recommendations.items()
            },
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration result to {path}")


# =============================================================================
# Per-Day Volatility Data Classes
# =============================================================================

@dataclass
class DayVolatilityStats:
    """Volatility statistics for a single trading day."""

    date: str
    """Trading date (YYYYMMDD)."""

    realized_vol: float
    """Realized volatility: std(log_returns) over the day."""

    num_prices: int
    """Number of mid-price observations for the day."""

    scaling_factor: float
    """vol_day / vol_reference, clamped to [floor, cap]."""

    split: str
    """Which data split this day belongs to (train/val/test)."""

    scaled_barriers: Dict[int, Dict[str, float]]
    """Per-horizon scaled barriers: {horizon: {pt_pct, sl_pct, pt_bps, sl_bps}}."""


@dataclass
class VolatilityCalibrationResult:
    """Complete per-day volatility calibration result."""

    data_source: str
    """Path to the data used for calibration."""

    num_days: int
    """Total number of trading days analyzed."""

    reference_volatility: float
    """Median daily realized volatility across training days."""

    volatility_floor: float
    """Minimum scaling factor applied."""

    volatility_cap: float
    """Maximum scaling factor applied."""

    base_barriers: Dict[int, Dict[str, float]]
    """Base (unscaled) barriers per horizon: {horizon: {pt_pct, sl_pct}}."""

    day_stats: List[DayVolatilityStats]
    """Per-day volatility statistics."""

    horizons: List[int]
    """Horizons analyzed."""

    def print_report(self) -> None:
        """Print human-readable per-day volatility report."""
        print()
        print("=" * 80)
        print("PER-DAY VOLATILITY CALIBRATION REPORT")
        print("=" * 80)
        print()
        print(f"Data source: {self.data_source}")
        print(f"Days analyzed: {self.num_days}")
        print(f"Reference volatility (median of train): {self.reference_volatility:.8f}")
        print(f"  = {self.reference_volatility * 10000:.4f} bps per tick")
        print(f"Scaling floor: {self.volatility_floor:.2f}x")
        print(f"Scaling cap:   {self.volatility_cap:.2f}x")
        print()

        print("-" * 80)
        print("BASE BARRIERS (before scaling)")
        print("-" * 80)
        for h in self.horizons:
            b = self.base_barriers[h]
            print(f"  H{h}: PT={b['pt_bps']:.1f} bps, SL={b['sl_bps']:.1f} bps")
        print()

        vols = [d.realized_vol for d in self.day_stats]
        scales = [d.scaling_factor for d in self.day_stats]
        print("-" * 80)
        print("VOLATILITY DISTRIBUTION")
        print("-" * 80)
        print(f"  Min:    {min(vols):.8f} ({min(vols)*10000:.4f} bps)")
        print(f"  P25:    {np.percentile(vols, 25):.8f} ({np.percentile(vols, 25)*10000:.4f} bps)")
        print(f"  Median: {np.median(vols):.8f} ({np.median(vols)*10000:.4f} bps)")
        print(f"  P75:    {np.percentile(vols, 75):.8f} ({np.percentile(vols, 75)*10000:.4f} bps)")
        print(f"  Max:    {max(vols):.8f} ({max(vols)*10000:.4f} bps)")
        print()
        print(f"  Scaling factor range: {min(scales):.3f}x - {max(scales):.3f}x")
        print(f"  Days at floor ({self.volatility_floor:.2f}x): "
              f"{sum(1 for s in scales if s <= self.volatility_floor)}")
        print(f"  Days at cap ({self.volatility_cap:.2f}x):   "
              f"{sum(1 for s in scales if s >= self.volatility_cap)}")
        print()

        for split in ["train", "val", "test"]:
            split_days = [d for d in self.day_stats if d.split == split]
            if not split_days:
                continue
            split_vols = [d.realized_vol for d in split_days]
            split_scales = [d.scaling_factor for d in split_days]
            print(f"  {split:5s} ({len(split_days):3d} days): "
                  f"vol={np.median(split_vols):.8f} (median), "
                  f"scale={np.median(split_scales):.3f}x (median)")

        print()
        print("-" * 80)
        print("PER-DAY DETAILS (sample)")
        print("-" * 80)
        print(f"{'Date':<12} {'Split':<6} {'Vol (bps)':<12} {'Scale':<8} ", end="")
        for h in self.horizons:
            print(f"{'H'+str(h)+' PT':<10} {'H'+str(h)+' SL':<10} ", end="")
        print()
        print("-" * 80)

        sample = self.day_stats[::max(1, len(self.day_stats) // 20)]
        for d in sample:
            print(f"{d.date:<12} {d.split:<6} {d.realized_vol*10000:<12.4f} "
                  f"{d.scaling_factor:<8.3f} ", end="")
            for h in self.horizons:
                b = d.scaled_barriers[h]
                print(f"{b['pt_bps']:<10.1f} {b['sl_bps']:<10.1f} ", end="")
            print()

        print()
        print("-" * 80)
        print("TOML CONFIG SNIPPET (for volatility-scaled export)")
        print("-" * 80)
        print()
        print("[labels]")
        print('strategy = "triple_barrier"')
        print(f"max_horizons = {self.horizons}")
        pts = [float(self.base_barriers[h]['pt_pct']) for h in self.horizons]
        sls = [float(self.base_barriers[h]['sl_pct']) for h in self.horizons]
        print(f"profit_targets = {[round(p, 6) for p in pts]}")
        print(f"stop_losses = {[round(s, 6) for s in sls]}")
        print(f"volatility_scaling = true")
        print(f"volatility_reference = {self.reference_volatility:.10f}")
        print(f"volatility_floor = {self.volatility_floor}")
        print(f"volatility_cap = {self.volatility_cap}")
        print('timeout_strategy = "label_as_timeout"')
        print()

    def to_json(self, path: Path) -> None:
        """Save per-day volatility calibration result to JSON."""
        data = {
            'data_source': self.data_source,
            'num_days': self.num_days,
            'reference_volatility': self.reference_volatility,
            'volatility_floor': self.volatility_floor,
            'volatility_cap': self.volatility_cap,
            'horizons': self.horizons,
            'base_barriers': {
                str(h): {
                    'profit_target_pct': b['pt_pct'],
                    'stop_loss_pct': b['sl_pct'],
                    'profit_target_bps': b['pt_bps'],
                    'stop_loss_bps': b['sl_bps'],
                }
                for h, b in self.base_barriers.items()
            },
            'volatility_stats': {
                'min': float(min(d.realized_vol for d in self.day_stats)),
                'p25': float(np.percentile([d.realized_vol for d in self.day_stats], 25)),
                'median': float(np.median([d.realized_vol for d in self.day_stats])),
                'p75': float(np.percentile([d.realized_vol for d in self.day_stats], 75)),
                'max': float(max(d.realized_vol for d in self.day_stats)),
            },
            'per_day': [
                {
                    'date': d.date,
                    'split': d.split,
                    'realized_volatility': d.realized_vol,
                    'num_prices': d.num_prices,
                    'scaling_factor': d.scaling_factor,
                    'scaled_barriers': {
                        str(h): {
                            'profit_target_pct': b['pt_pct'],
                            'stop_loss_pct': b['sl_pct'],
                            'profit_target_bps': b['pt_bps'],
                            'stop_loss_bps': b['sl_bps'],
                        }
                        for h, b in d.scaled_barriers.items()
                    },
                }
                for d in self.day_stats
            ],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved per-day volatility calibration to {path}")


# =============================================================================
# Analysis Functions
# =============================================================================

def load_price_data(data_dir: Path, max_days: int = 50) -> Tuple[np.ndarray, int]:
    """
    Load mid-price data from exported sequences.
    
    Args:
        data_dir: Path to export directory (should contain train/ subdirectory)
        max_days: Maximum number of days to load (for speed)
        
    Returns:
        Tuple of (price array, number of days loaded)
    """
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    seq_files = sorted(train_dir.glob("*_sequences.npy"))[:max_days]
    
    if not seq_files:
        raise FileNotFoundError(f"No sequence files found in {train_dir}")
    
    all_prices = []
    for seq_file in seq_files:
        seqs = np.load(seq_file)  # [N, T, F]
        # Extract mid-price at the end of each sequence
        end_prices = seqs[:, -1, MID_PRICE_IDX]
        all_prices.extend(end_prices.tolist())
    
    return np.array(all_prices), len(seq_files)


def compute_movement_stats(
    prices: np.ndarray,
    horizons: List[int],
) -> Dict[int, PriceMovementStats]:
    """
    Compute price movement statistics for given horizons.
    
    Args:
        prices: Array of prices (sequential observations)
        horizons: List of horizons (tick offsets) to analyze
        
    Returns:
        Dict mapping horizon to PriceMovementStats
    """
    stats = {}
    
    for horizon in horizons:
        if horizon >= len(prices):
            continue
        
        # Compute returns over H ticks
        returns_pct = (prices[horizon:] - prices[:-horizon]) / prices[:-horizon] * 100
        returns_bps = returns_pct * 100  # Convert to basis points
        
        abs_returns_bps = np.abs(returns_bps)
        
        # Compute percentiles
        percentile_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
                           55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        percentiles = {
            p: float(np.percentile(abs_returns_bps, p))
            for p in percentile_values
        }
        
        stats[horizon] = PriceMovementStats(
            horizon=horizon,
            num_samples=len(returns_bps),
            mean_abs_return_bps=float(abs_returns_bps.mean()),
            std_return_bps=float(returns_bps.std()),
            percentiles=percentiles,
        )
    
    return stats


def compute_barrier_recommendations(
    movement_stats: Dict[int, PriceMovementStats],
    target_pt_rate: float,
    target_sl_rate: float,
    risk_reward_ratio: float = 1.5,
) -> Dict[int, BarrierRecommendation]:
    """
    Compute barrier threshold recommendations.
    
    The key insight is:
    - For a symmetric barrier, the rate of hitting either barrier is
      approximately 2 * (rate of exceeding the threshold on one side)
    - We want PT_rate + SL_rate ≈ target_decisive_rate
    - So each barrier should be set at the percentile corresponding to
      capturing (target_rate / 2) on each side
    
    With asymmetric barriers (R:R ratio != 1):
    - PT barrier is set higher, SL barrier is set lower
    - This makes PT harder to hit than SL when R:R > 1
    
    Args:
        movement_stats: Price movement statistics by horizon
        target_pt_rate: Target ProfitTarget rate (e.g., 0.30)
        target_sl_rate: Target StopLoss rate (e.g., 0.30)
        risk_reward_ratio: PT / SL ratio (e.g., 1.5 means PT is 1.5x SL)
        
    Returns:
        Dict mapping horizon to BarrierRecommendation
    """
    recommendations = {}
    
    # Combined decisive rate (PT + SL)
    total_decisive = target_pt_rate + target_sl_rate
    
    for horizon, stats in movement_stats.items():
        # For symmetric barriers, the "hit" rate is approximately the
        # rate of exceeding the threshold (in either direction)
        # Since we have absolute returns, the threshold that gives
        # X% "decisive" outcomes is the (100-X)th percentile
        
        # However, we want asymmetric barriers with R:R ratio
        # A simple approach: find the symmetric threshold first,
        # then scale PT and SL to achieve the desired ratio
        
        # Get the threshold for total decisive rate
        base_threshold = stats.get_threshold_for_rate(total_decisive)
        
        # Scale by R:R ratio
        # If R:R = 1.5, PT should be ~1.22x base, SL should be ~0.82x base
        # (so that PT/SL = 1.5)
        scale_factor = np.sqrt(risk_reward_ratio)
        
        pt_threshold = base_threshold * scale_factor
        sl_threshold = base_threshold / scale_factor
        
        # Estimate expected class rates (simplified model)
        # This is approximate; actual rates depend on price path dynamics
        expected_pt = target_pt_rate * (1 / scale_factor)  # PT harder to hit
        expected_sl = target_sl_rate * scale_factor  # SL easier to hit
        expected_to = 1 - expected_pt - expected_sl
        
        # Clamp to valid ranges
        expected_pt = max(0.05, min(0.45, expected_pt))
        expected_sl = max(0.05, min(0.45, expected_sl))
        expected_to = 1 - expected_pt - expected_sl
        
        recommendations[horizon] = BarrierRecommendation(
            horizon=horizon,
            profit_target_bps=pt_threshold,
            stop_loss_bps=sl_threshold,
            risk_reward_ratio=risk_reward_ratio,
            expected_pt_rate=expected_pt,
            expected_sl_rate=expected_sl,
            expected_to_rate=expected_to,
        )
    
    return recommendations


def calibrate(
    data_dir: Path,
    horizons: List[int],
    target_pt_rate: float = DEFAULT_TARGET_PT_RATE,
    target_sl_rate: float = DEFAULT_TARGET_SL_RATE,
    risk_reward_ratio: float = 1.5,
    max_days: int = 50,
) -> CalibrationResult:
    """
    Run complete barrier calibration.
    
    Args:
        data_dir: Path to export directory
        horizons: List of horizons to calibrate for
        target_pt_rate: Target ProfitTarget rate
        target_sl_rate: Target StopLoss rate
        risk_reward_ratio: Desired PT / SL ratio
        max_days: Maximum days to analyze
        
    Returns:
        CalibrationResult with recommendations
    """
    print(f"Loading price data from {data_dir}...")
    prices, num_days = load_price_data(data_dir, max_days)
    print(f"Loaded {len(prices):,} price observations from {num_days} days")
    
    print("Computing price movement statistics...")
    movement_stats = compute_movement_stats(prices, horizons)
    
    print("Computing barrier recommendations...")
    recommendations = compute_barrier_recommendations(
        movement_stats,
        target_pt_rate,
        target_sl_rate,
        risk_reward_ratio,
    )
    
    return CalibrationResult(
        data_source=str(data_dir),
        num_days_analyzed=num_days,
        num_samples=len(prices),
        price_range=(float(prices.min()), float(prices.max())),
        mean_price=float(prices.mean()),
        movement_stats=movement_stats,
        recommendations=recommendations,
        target_pt_rate=target_pt_rate,
        target_sl_rate=target_sl_rate,
    )


# =============================================================================
# Per-Day Volatility Analysis
# =============================================================================

def compute_realized_volatility(prices: np.ndarray) -> float:
    """
    Compute realized volatility as std(log_returns).

    This is the standard realized volatility estimator:
        RV = std(ln(P_t / P_{t-1}))

    Args:
        prices: Array of mid-prices (must have at least 2 elements)

    Returns:
        Realized volatility (dimensionless, per-tick)

    Reference:
        Andersen, T. G. & Bollerslev, T. (1998).
        "Answering the Skeptics: Yes, Standard Volatility Models Do Provide
        Accurate Forecasts", International Economic Review.
    """
    if len(prices) < 2:
        return 0.0

    valid_mask = (prices[:-1] > 0) & (prices[1:] > 0)
    if not np.any(valid_mask):
        return 0.0

    log_returns = np.log(prices[1:][valid_mask] / prices[:-1][valid_mask])
    if len(log_returns) < 2:
        return 0.0

    return float(np.std(log_returns, ddof=1))


def load_day_prices(seq_file: Path) -> np.ndarray:
    """
    Load all mid-prices from a single day's sequence file.

    Extracts the mid-price time series from the sequence tensor by taking
    the mid-price feature across all timesteps of all sequences.

    Args:
        seq_file: Path to {date}_sequences.npy

    Returns:
        1D array of mid-prices for the day
    """
    seqs = np.load(seq_file, mmap_mode='r')  # [N, T, F]
    # Use mid-price from the last timestep of each sequence for a
    # non-overlapping price series that respects the stride
    prices = seqs[:, -1, MID_PRICE_IDX].copy()
    return prices


def calibrate_per_day(
    data_dir: Path,
    horizons: List[int],
    target_pt_rate: float = DEFAULT_TARGET_PT_RATE,
    target_sl_rate: float = DEFAULT_TARGET_SL_RATE,
    risk_reward_ratio: float = 1.5,
    volatility_floor: float = 0.3,
    volatility_cap: float = 3.0,
) -> VolatilityCalibrationResult:
    """
    Run per-day volatility analysis and compute dynamic barrier recommendations.

    Workflow:
      1. Load each day's prices and compute realized volatility
      2. Determine reference volatility (median of training days)
      3. Compute base barriers from the reference volatility regime
      4. For each day, compute scaling_factor = vol_day / vol_reference
      5. Scale base barriers by the clamped scaling factor

    Args:
        data_dir: Path to export directory
        horizons: Horizons to calibrate
        target_pt_rate: Target ProfitTarget rate
        target_sl_rate: Target StopLoss rate
        risk_reward_ratio: Desired PT / SL ratio
        volatility_floor: Minimum scaling factor
        volatility_cap: Maximum scaling factor

    Returns:
        VolatilityCalibrationResult with per-day statistics
    """
    print(f"Loading per-day price data from {data_dir}...")

    day_data: List[Tuple[str, str, np.ndarray]] = []  # (date, split, prices)

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        seq_files = sorted(split_dir.glob("*_sequences.npy"))
        for seq_file in seq_files:
            date = seq_file.stem.replace("_sequences", "")
            prices = load_day_prices(seq_file)
            if len(prices) >= 10:
                day_data.append((date, split, prices))

    if not day_data:
        raise FileNotFoundError(f"No sequence files found in {data_dir}")

    print(f"Loaded {len(day_data)} days across all splits")

    # Step 1: Compute per-day realized volatility
    day_vols: List[Tuple[str, str, float, int]] = []
    for date, split, prices in day_data:
        vol = compute_realized_volatility(prices)
        day_vols.append((date, split, vol, len(prices)))

    # Step 2: Reference volatility = median of training days
    train_vols = [v for _, s, v, _ in day_vols if s == "train" and v > 0]
    if not train_vols:
        raise ValueError("No valid training days found for reference volatility")

    reference_vol = float(np.median(train_vols))
    print(f"Reference volatility (median of {len(train_vols)} train days): "
          f"{reference_vol:.8f} ({reference_vol * 10000:.4f} bps/tick)")

    # Step 3: Compute base barriers from reference volatility regime
    # Collect prices from days near the reference volatility
    ref_prices = []
    for date, split, prices in day_data:
        if split != "train":
            continue
        vol = compute_realized_volatility(prices)
        if vol > 0 and 0.7 * reference_vol <= vol <= 1.3 * reference_vol:
            ref_prices.extend(prices.tolist())

    if len(ref_prices) < 100:
        # Fall back to all training prices
        ref_prices = []
        for date, split, prices in day_data:
            if split == "train":
                ref_prices.extend(prices.tolist())

    ref_prices_arr = np.array(ref_prices)
    print(f"Computing base barriers from {len(ref_prices_arr):,} reference prices...")

    movement_stats = compute_movement_stats(ref_prices_arr, horizons)
    base_recs = compute_barrier_recommendations(
        movement_stats, target_pt_rate, target_sl_rate, risk_reward_ratio
    )

    base_barriers: Dict[int, Dict[str, float]] = {}
    for h, rec in base_recs.items():
        base_barriers[h] = {
            'pt_pct': rec.profit_target_bps / 10000.0,
            'sl_pct': rec.stop_loss_bps / 10000.0,
            'pt_bps': rec.profit_target_bps,
            'sl_bps': rec.stop_loss_bps,
        }

    # Step 4: Compute per-day scaling factors and scaled barriers
    day_stats: List[DayVolatilityStats] = []
    for date, split, vol, n_prices in day_vols:
        if vol <= 0 or reference_vol <= 0:
            raw_scale = 1.0
        else:
            raw_scale = vol / reference_vol

        scale = max(volatility_floor, min(volatility_cap, raw_scale))

        scaled_barriers: Dict[int, Dict[str, float]] = {}
        for h in horizons:
            base = base_barriers[h]
            scaled_pt = base['pt_pct'] * scale
            scaled_sl = base['sl_pct'] * scale
            scaled_barriers[h] = {
                'pt_pct': scaled_pt,
                'sl_pct': scaled_sl,
                'pt_bps': scaled_pt * 10000.0,
                'sl_bps': scaled_sl * 10000.0,
            }

        day_stats.append(DayVolatilityStats(
            date=date,
            realized_vol=vol,
            num_prices=n_prices,
            scaling_factor=scale,
            split=split,
            scaled_barriers=scaled_barriers,
        ))

    return VolatilityCalibrationResult(
        data_source=str(data_dir),
        num_days=len(day_stats),
        reference_volatility=reference_vol,
        volatility_floor=volatility_floor,
        volatility_cap=volatility_cap,
        base_barriers=base_barriers,
        day_stats=day_stats,
        horizons=horizons,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate Triple Barrier thresholds based on price movement data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic calibration
    python tools/calibrate_triple_barrier.py \\
        --data-dir ../data/exports/nvda_11month_triple_barrier \\
        --horizons 50 100 200

    # Custom target class distribution
    python tools/calibrate_triple_barrier.py \\
        --data-dir ../data/exports/nvda_11month_triple_barrier \\
        --horizons 50 100 200 \\
        --target-pt-rate 0.25 \\
        --target-sl-rate 0.25 \\
        --risk-reward 2.0
        
    # Save results to JSON
    python tools/calibrate_triple_barrier.py \\
        --data-dir ../data/exports/nvda_11month_triple_barrier \\
        --horizons 50 100 200 \\
        --output calibration_result.json
        """,
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=True,
        help="Path to export directory containing train/ subdirectory",
    )
    
    parser.add_argument(
        "--horizons", "-H",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Horizons (in ticks) to calibrate (default: 50 100 200)",
    )
    
    parser.add_argument(
        "--target-pt-rate",
        type=float,
        default=DEFAULT_TARGET_PT_RATE,
        help=f"Target ProfitTarget rate (default: {DEFAULT_TARGET_PT_RATE})",
    )
    
    parser.add_argument(
        "--target-sl-rate",
        type=float,
        default=DEFAULT_TARGET_SL_RATE,
        help=f"Target StopLoss rate (default: {DEFAULT_TARGET_SL_RATE})",
    )
    
    parser.add_argument(
        "--risk-reward", "-r",
        type=float,
        default=1.5,
        help="Risk-reward ratio (PT / SL). Default: 1.5",
    )
    
    parser.add_argument(
        "--max-days",
        type=int,
        default=50,
        help="Maximum days to analyze (default: 50)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for calibration results",
    )

    parser.add_argument(
        "--per-day",
        action="store_true",
        default=False,
        help="Enable per-day volatility analysis mode. Computes realized "
             "volatility for each day and recommends volatility-scaled barriers.",
    )

    parser.add_argument(
        "--volatility-floor",
        type=float,
        default=0.3,
        help="Minimum scaling factor for volatility scaling (default: 0.3)",
    )

    parser.add_argument(
        "--volatility-cap",
        type=float,
        default=3.0,
        help="Maximum scaling factor for volatility scaling (default: 3.0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Validate target rates
    if args.target_pt_rate + args.target_sl_rate >= 1.0:
        print("ERROR: target_pt_rate + target_sl_rate must be < 1.0")
        sys.exit(1)

    if args.per_day:
        # Per-day volatility analysis mode
        result = calibrate_per_day(
            data_dir=data_dir,
            horizons=args.horizons,
            target_pt_rate=args.target_pt_rate,
            target_sl_rate=args.target_sl_rate,
            risk_reward_ratio=args.risk_reward,
            volatility_floor=args.volatility_floor,
            volatility_cap=args.volatility_cap,
        )
        result.print_report()
        if args.output:
            result.to_json(Path(args.output))
    else:
        # Global calibration mode (original behavior)
        result = calibrate(
            data_dir=data_dir,
            horizons=args.horizons,
            target_pt_rate=args.target_pt_rate,
            target_sl_rate=args.target_sl_rate,
            risk_reward_ratio=args.risk_reward,
            max_days=args.max_days,
        )
        result.print_report()
        if args.output:
            result.to_json(Path(args.output))


if __name__ == "__main__":
    main()
