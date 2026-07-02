"""Vectorized point-in-time backtest engine with costs, borrow, and turnover penalty."""
from __future__ import annotations
import numpy as np
import pandas as pd


def market_impact_cost(trade_notional: pd.Series, adv_notional: pd.Series,
                        daily_vol: pd.Series, eta: float = 0.5) -> pd.Series:
    """Square-root market impact model: cost = eta * sigma * sqrt(Q/ADV) * |trade_notional|."""
    adv_notional = adv_notional.replace(0, np.nan)
    participation = (trade_notional.abs() / adv_notional).clip(lower=0.0)
    cost_bps = eta * daily_vol * np.sqrt(participation)
    return (cost_bps * trade_notional.abs()).fillna(0.0)


def apply_borrow_filter(target_weights: pd.Series, borrow_fee_bps: pd.Series,
                         max_borrow_bps: float = 300.0) -> pd.Series:
    """Zero-out short positions in names whose borrow fee exceeds threshold."""
    w = target_weights.copy()
    blocked = (w < 0) & (borrow_fee_bps > max_borrow_bps)
    w[blocked] = 0.0
    return w


def turnover_penalized_rebalance(target_weights: pd.Series, prev_weights: pd.Series,
                                  kappa: float = 0.0) -> pd.Series:
    """Shrink weight changes toward prev_weights by penalty kappa in [0,1).

    new_w = prev_w + (1 - kappa) * (target_w - prev_w)
    kappa=0 -> full rebalance to target; kappa->1 -> no rebalance (freeze).
    """
    if not (0.0 <= kappa < 1.0):
        raise ValueError("kappa must be in [0, 1)")
    aligned_target, aligned_prev = target_weights.align(prev_weights, fill_value=0.0)
    return aligned_prev + (1.0 - kappa) * (aligned_target - aligned_prev)


def daily_pnl(weights: pd.Series, fwd_returns: pd.Series, gross_notional: float = 1.0) -> float:
    """Simple daily P&L in return units: sum(w_i * r_i) scaled by gross_notional (weights sum to target gross)."""
    w, r = weights.align(fwd_returns, fill_value=0.0)
    return float((w * r).sum() * gross_notional)


def run_backtest(scores: pd.DataFrame, fwd_returns: pd.DataFrame,
                  adv_notional: pd.DataFrame, daily_vol: pd.DataFrame,
                  borrow_fee_bps: pd.DataFrame | None = None,
                  quintile: float = 0.2, kappa: float = 0.1,
                  eta: float = 0.5, commission_bps: float = 5.0,
                  max_borrow_bps: float = 300.0) -> pd.DataFrame:
    """Run a dollar-neutral quintile long/short backtest across dates.

    scores, fwd_returns, adv_notional, daily_vol: DataFrame[date x ticker].
    Returns a DataFrame indexed by date with columns: gross_pnl, costs, net_pnl, turnover.
    """
    dates = scores.index
    tickers = scores.columns
    prev_w = pd.Series(0.0, index=tickers)
    records = []

    for dt in dates:
        row = scores.loc[dt].dropna()
        if row.empty:
            records.append({"date": dt, "gross_pnl": 0.0, "costs": 0.0,
                             "net_pnl": 0.0, "turnover": 0.0})
            continue
        n = len(row)
        n_q = max(1, int(np.floor(n * quintile)))
        longs = row.sort_values(ascending=False).index[:n_q]
        shorts = row.sort_values(ascending=True).index[:n_q]

        target_w = pd.Series(0.0, index=tickers)
        if n_q > 0:
            target_w.loc[longs] = 1.0 / n_q
            target_w.loc[shorts] = -1.0 / n_q

        if borrow_fee_bps is not None and dt in borrow_fee_bps.index:
            target_w = apply_borrow_filter(target_w, borrow_fee_bps.loc[dt].reindex(tickers).fillna(0.0),
                                            max_borrow_bps=max_borrow_bps)

        new_w = turnover_penalized_rebalance(target_w, prev_w, kappa=kappa)
        trade = (new_w - prev_w)
        turnover = trade.abs().sum()

        fwd_r = fwd_returns.loc[dt].reindex(tickers) if dt in fwd_returns.index else pd.Series(0.0, index=tickers)
        gross_pnl = daily_pnl(new_w, fwd_r)

        adv = adv_notional.loc[dt].reindex(tickers) if dt in adv_notional.index else pd.Series(np.nan, index=tickers)
        vol = daily_vol.loc[dt].reindex(tickers) if dt in daily_vol.index else pd.Series(0.02, index=tickers)
        impact = market_impact_cost(trade, adv, vol, eta=eta).sum()
        commission = trade.abs().sum() * (commission_bps / 1e4)
        costs = impact + commission

        records.append({"date": dt, "gross_pnl": gross_pnl, "costs": costs,
                         "net_pnl": gross_pnl - costs, "turnover": turnover})
        prev_w = new_w

    return pd.DataFrame.from_records(records).set_index("date")


def information_coefficient(scores: pd.Series, fwd_returns: pd.Series) -> float:
    """Spearman rank IC between scores and next-period returns for a single cross-section."""
    s, r = scores.align(fwd_returns, join="inner")
    s = s.dropna()
    r = r.reindex(s.index).dropna()
    common = s.index.intersection(r.index)
    if len(common) < 3:
        return float("nan")
    return float(s.loc[common].corr(r.loc[common], method="spearman"))
