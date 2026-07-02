"""Signal construction: EOV, SLCS, VRP/Skew/Term-structure triad."""
from __future__ import annotations
import numpy as np
import pandas as pd


def pct_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]; NaN-safe."""
    return series.rank(pct=True, na_option="keep")


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score; robust to zero-variance (returns 0s)."""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute_eov(call_volume: pd.Series, call_open_interest: pd.Series,
                 call_volume_20d_avg: pd.Series) -> pd.Series:
    """Excess Option Volume factor (Barclays Fig. 26-29 replication).

    EOV = 0.5 * [pctrank(CallVol/OI) + pctrank(CallVol/CallVol_20davg)]
    """
    if (call_open_interest <= 0).any():
        call_open_interest = call_open_interest.clip(lower=1.0)
    if (call_volume_20d_avg <= 0).any():
        call_volume_20d_avg = call_volume_20d_avg.clip(lower=1.0)
    ratio_oi = call_volume / call_open_interest
    ratio_avg = call_volume / call_volume_20d_avg
    return 0.5 * (pct_rank(ratio_oi) + pct_rank(ratio_avg))


def compute_slcs(small_call_vol: pd.Series, total_call_vol: pd.Series,
                  small_put_vol: pd.Series, total_put_vol: pd.Series) -> pd.Series:
    """Small-Lot Call Skew: retail proxy per Barclays Fig. 12-16."""
    total_call_vol = total_call_vol.replace(0, np.nan)
    total_put_vol = total_put_vol.replace(0, np.nan)
    call_frac = (small_call_vol / total_call_vol).fillna(0.0)
    put_frac = (small_put_vol / total_put_vol).fillna(0.0)
    return call_frac - put_frac


def bipower_variation(intraday_returns: pd.Series) -> float:
    """Jump-robust realized volatility estimator (Barndorff-Nielsen & Shephard 2004).

    BV = (pi/2) * sum(|r_t| * |r_{t-1}|), annualized sqrt.
    intraday_returns: a single day's/window's return series (log returns).
    """
    r = intraday_returns.dropna().values
    if len(r) < 2:
        return float("nan")
    abs_r = np.abs(r)
    bv = (np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1])
    return float(np.sqrt(bv))


def compute_vrp(implied_vol_1m: pd.Series, realized_vol_forecast: pd.Series) -> pd.Series:
    """VRP = IV(1M) / realized_vol_forecast(1M). >1 => vol looks "rich" (short candidate)."""
    denom = realized_vol_forecast.replace(0, np.nan)
    return (implied_vol_1m / denom).fillna(0.0)


def compute_combined_alpha_score(eov: pd.Series, slcs: pd.Series,
                                  liquidity_mask: pd.Series,
                                  slcs_weight: float = 0.5) -> pd.Series:
    """Score^Alpha = z(EOV)*liquidity_mask + w*z(SLCS)."""
    z_eov = zscore(eov)
    z_slcs = zscore(slcs)
    score = z_eov.where(liquidity_mask.astype(bool), 0.0) + slcs_weight * z_slcs
    return score


def compute_carry_score(vrp: pd.Series, skew_pctile: pd.Series,
                         gamma_squeeze_risk: pd.Series, lam: float = 0.75) -> pd.Series:
    """Score^Carry = z(VRP) - 0.5*z(SkewPctile) - lam * GammaSqueezeRisk."""
    return zscore(vrp) - 0.5 * zscore(skew_pctile) - lam * gamma_squeeze_risk
