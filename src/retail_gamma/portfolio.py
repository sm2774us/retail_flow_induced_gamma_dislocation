"""Portfolio construction: vol targeting, fractional Kelly sizing, drawdown governor."""

from __future__ import annotations
import numpy as np
import pandas as pd


def ewma_vol(returns: pd.Series, span: int = 20, annualize: bool = True) -> pd.Series:
    """
    EWMA realized volatility estimate.

    Args:
        returns (pd.Series): Series of returns.
        span (int): Span for EWMA calculation. Default is 20.
        annualize (bool): Whether to annualize the volatility. Default is True.

    Returns:
        pd.Series: Series of EWMA volatility estimates.
    """
    var = returns.pow(2).ewm(span=span, adjust=False).mean()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def vol_target_scalar(
    current_vol: float, target_vol: float = 0.10, max_leverage: float = 3.0
) -> float:
    """
    Scale factor to apply to weights to hit target annualized vol, capped.

    Args:
        current_vol (float): Current annualized volatility.
        target_vol (float): Target annualized volatility. Default is 0.10.
        max_leverage (float): Maximum leverage cap. Default is 3.0.

    Returns:
        float: Scale factor to apply to weights, capped at max_leverage.
    """
    if current_vol is None or current_vol <= 0 or np.isnan(current_vol):
        return 1.0
    scalar = target_vol / current_vol
    return float(np.clip(scalar, 0.0, max_leverage))


def fractional_kelly_weights(
    expected_returns: pd.Series,
    variances: pd.Series,
    kelly_fraction: float = 0.5,
    w_max: float = 0.02,
) -> pd.Series:
    """
    Weights based on fractional Kelly criterion, capped at +/- w_max per name.

    w_i = kelly_fraction * mu_i / sigma_i^2, capped at +/- w_max per name.

    Args:
        expected_returns (pd.Series): Series of expected returns (mu_i).
        variances (pd.Series): Series of variances (sigma_i^2).
        kelly_fraction (float): Fraction of Kelly to use. Default is 0.5.
        w_max (float): Maximum absolute weight per name. Default is 0.02.

    Returns:
        pd.Series: Series of weights based on fractional Kelly criterion, capped at +/- w_max.

    Raises:
        ValueError: If kelly_fraction is not in (0, 1].
    """
    if not (0.0 < kelly_fraction <= 1.0):
        raise ValueError("kelly_fraction must be in (0, 1]")
    var = variances.replace(0, np.nan)
    raw = kelly_fraction * (expected_returns / var)
    raw = raw.fillna(0.0)
    return raw.clip(lower=-w_max, upper=w_max)


def drawdown_governor(
    cumulative_pnl: pd.Series, dd_cut_50: float = -0.05, dd_cut_100: float = -0.08
) -> pd.Series:
    """
    Return a per-date exposure multiplier in {1.0, 0.5, 0.0} based on trailing 20d drawdown.

    Args:
        cumulative_pnl (pd.Series): Series of cumulative PnL values.
        dd_cut_50 (float): Trailing drawdown threshold to cut gross exposure to 50%. Default is -0.05.
        dd_cut_100 (float): Trailing drawdown threshold to cut new risk to 0%. Default is -0.08.

    Returns:
        pd.Series: Series of exposure multipliers based on trailing drawdown.
    """
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    # normalize drawdown by running_max where running_max > 0, else absolute
    denom = running_max.replace(0, np.nan).abs()
    dd_pct = (drawdown / denom).fillna(drawdown)

    multiplier = pd.Series(1.0, index=cumulative_pnl.index)
    multiplier[dd_pct <= dd_cut_50] = 0.5
    multiplier[dd_pct <= dd_cut_100] = 0.0
    return multiplier


def gamma_squeeze_circuit_breaker(
    intraday_move_atr_units: pd.Series,
    slcs_zscore: pd.Series,
    atr_threshold: float = 3.0,
    slcs_zscore_threshold: float = 1.2816,
) -> pd.Series:
    """
    Boolean mask: True => flatten short-vol position immediately.

    Triggers when |move| > atr_threshold ATRs AND slcs_zscore exceeds the
    90th-percentile z-score threshold (default corresponds to ~90th pct of normal dist).

    Args:
        intraday_move_atr_units (pd.Series): Series of intraday moves in ATR units.
        slcs_zscore (pd.Series): Series of SLCS z-scores.
        atr_threshold (float): ATR threshold for triggering. Default is 3.0.
        slcs_zscore_threshold (float): SLCS z-score threshold for triggering. Default is
            1.2816 (90th percentile of standard normal distribution).

    Returns:
        pd.Series: Boolean series indicating whether to flatten short-vol position.
    """
    move_trigger = intraday_move_atr_units.abs() >= atr_threshold
    flow_trigger = slcs_zscore >= slcs_zscore_threshold
    return (move_trigger & flow_trigger).fillna(False)


def ledoit_wolf_shrink_cov(returns: pd.DataFrame, shrink: float = 0.2) -> pd.DataFrame:
    """
    Simple Ledoit-Wolf-style shrinkage: blend sample covariance with a scaled identity target.

    Args:
        returns (pd.DataFrame): DataFrame of returns.
        shrink (float): Shrinkage intensity in [0, 1]. Default is 0.2.

    Returns:
        pd.DataFrame: Shrunk covariance matrix.

    Raises:
        ValueError: If shrink is not in [0, 1].
    """
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink must be in [0, 1]")
    sample_cov = returns.cov()
    avg_var = np.diag(sample_cov.values).mean() if sample_cov.shape[0] > 0 else 0.0
    target = np.eye(sample_cov.shape[0]) * avg_var
    shrunk = (1 - shrink) * sample_cov.values + shrink * target
    return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)
