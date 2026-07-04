"""Signal construction: EOV, SLCS, VRP/Skew/Term-structure triad."""

from __future__ import annotations
import numpy as np
import pandas as pd


def pct_rank(series: pd.Series) -> pd.Series:
    """
    Cross-sectional percentile rank in [0, 1]; NaN-safe.

    Args:
        series (pd.Series): Input series to rank.

    Returns:
        pd.Series: Series of percentile ranks in [0, 1], preserving NaNs.
    """
    return series.rank(pct=True, na_option="keep")


def zscore(series: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score; robust to zero-variance (returns 0s).

    Args:
        series (pd.Series): Input series to z-score.

    Returns:
        pd.Series: Series of z-scores, preserving NaNs.
    """
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute_eov(
    call_volume: pd.Series,
    call_open_interest: pd.Series,
    call_volume_20d_avg: pd.Series,
) -> pd.Series:
    """
    Excess Option Volume factor (Barclays Fig. 26-29 replication).

    EOV = 0.5 * [pctrank(CallVol/OI) + pctrank(CallVol/CallVol_20davg)]

    Args:
        call_volume (pd.Series): Series of call option volumes.
        call_open_interest (pd.Series): Series of call option open interests.
        call_volume_20d_avg (pd.Series): Series of 20-day average call option
                                         volumes.

    Returns:
        pd.Series: Series of Excess Option Volume (EOV) values.
    """
    if (call_open_interest <= 0).any():
        call_open_interest = call_open_interest.clip(lower=1.0)
    if (call_volume_20d_avg <= 0).any():
        call_volume_20d_avg = call_volume_20d_avg.clip(lower=1.0)
    ratio_oi = call_volume / call_open_interest
    ratio_avg = call_volume / call_volume_20d_avg
    return 0.5 * (pct_rank(ratio_oi) + pct_rank(ratio_avg))


def compute_slcs(
    small_call_vol: pd.Series,
    total_call_vol: pd.Series,
    small_put_vol: pd.Series,
    total_put_vol: pd.Series,
) -> pd.Series:
    """
    Small-Lot Call Skew (SLCS) factor (Barclays Fig. 12-16 replication).

    SLCS = pctrank(SmallCallVol/TotalCallVol - SmallPutVol/TotalPutVol)

    Args:
        small_call_vol (pd.Series): Series of small-lot call option volumes.
        total_call_vol (pd.Series): Series of total call option volumes.
        small_put_vol (pd.Series): Series of small-lot put option volumes.
        total_put_vol (pd.Series): Series of total put option volumes.

    Returns:
        pd.Series: Series of Small-Lot Call Skew (SLCS) values.
    """
    total_call_vol = total_call_vol.replace(0, np.nan)
    total_put_vol = total_put_vol.replace(0, np.nan)
    call_frac = (small_call_vol / total_call_vol).fillna(0.0)
    put_frac = (small_put_vol / total_put_vol).fillna(0.0)
    return call_frac - put_frac


def bipower_variation(intraday_returns: pd.Series) -> float:
    """
    Jump-robust realized volatility estimator (Barndorff-Nielsen & Shephard 2004).

    BV = (pi/2) * sum(|r_t| * |r_{t-1}|), annualized sqrt.

    Args:
        intraday_returns (pd.Series): A single day's/window's return series (log returns).

    Returns:
        float: The estimated jump-robust realized volatility.
    """
    r = intraday_returns.dropna().values
    if len(r) < 2:
        return float("nan")
    abs_r = np.abs(r)
    bv = (np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1])
    return float(np.sqrt(bv))


def compute_vrp(
    implied_vol_1m: pd.Series, realized_vol_forecast: pd.Series
) -> pd.Series:
    """
    Volatility Risk Premium (VRP) factor (Barclays Fig. 17-19 replication).

    VRP = IV(1M) / realized_vol_forecast(1M). >1 => vol looks "rich" (short candidate).

    Args:
        implied_vol_1m (pd.Series): Series of 1-month implied volatilities.
        realized_vol_forecast (pd.Series): Series of 1-month realized volatility forecasts.

    Returns:
        pd.Series: Series of Volatility Risk Premium (VRP) values.
    """
    denom = realized_vol_forecast.replace(0, np.nan)
    return (implied_vol_1m / denom).fillna(0.0)


def compute_combined_alpha_score(
    eov: pd.Series, slcs: pd.Series, liquidity_mask: pd.Series, slcs_weight: float = 0.5
) -> pd.Series:
    """
    Alpha score combining EOV and SLCS, with liquidity mask applied to EOV.

    Score^Alpha = z(EOV)*liquidity_mask + w*z(SLCS).

    Args:
        eov (pd.Series): Series of Excess Option Volume (EOV) values.
        slcs (pd.Series): Series of Small-Lot Call Skew (SLCS) values.
        liquidity_mask (pd.Series): Boolean series indicating liquid dates.
        slcs_weight (float): Weight for SLCS in the combined score. Default is 0.5.

    Returns:
        pd.Series: Series of combined alpha scores.
    """
    z_eov = zscore(eov)
    z_slcs = zscore(slcs)
    score = z_eov.where(liquidity_mask.astype(bool), 0.0) + slcs_weight * z_slcs
    return score


def compute_carry_score(
    vrp: pd.Series,
    skew_pctile: pd.Series,
    gamma_squeeze_risk: pd.Series,
    lam: float = 0.75,
) -> pd.Series:
    """
    Carry score combining VRP, Skew Percentile, and Gamma Squeeze Risk.

    Score^Carry = z(VRP) - 0.5*z(SkewPctile) - lam * GammaSqueezeRisk.

    Args:
        vrp (pd.Series): Series of Volatility Risk Premium (VRP) values.
        skew_pctile (pd.Series): Series of skew percentiles.
        gamma_squeeze_risk (pd.Series): Series of gamma squeeze risk values.
        lam (float): Weight for gamma squeeze risk in the carry score. Default is 0.75.

    Returns:
        pd.Series: Series of carry scores.
    """
    return zscore(vrp) - 0.5 * zscore(skew_pctile) - lam * gamma_squeeze_risk
