"""Vectorized point-in-time backtest engine with costs, borrow, and turnover penalty.

Syntax:
    from retail_gamma.backtest import run_backtest
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def market_impact_cost(
    trade_notional: pd.Series,
    adv_notional: pd.Series,
    daily_vol: pd.Series,
    eta: float = 0.5,
) -> pd.Series:
    """
    The square-root market impact model, which estimates the cost of trading based on the size of the trade relative to the average daily volume (ADV) and the volatility of the asset. The cost is calculated as:

    cost = eta * sigma * sqrt(Q/ADV) * |trade_notional|

    Args:
        trade_notional (pd.Series): The notional value of the trades being executed, represented
                                    as a pandas Series where the index corresponds to the asset
                                    identifiers (e.g., tickers) and the values represent the
                                    notional amounts of the trades.
        adv_notional (pd.Series): The average daily volume (ADV) of the assets being traded,
                                  represented as a pandas Series where the index corresponds to the
                                  asset identifiers and the values represent the ADV in notional
                                  terms.
        daily_vol (pd.Series): The daily volatility of the assets being traded, represented as a
                               pandas Series where the index corresponds to the asset identifiers
                               and the values represent the daily volatility.
        eta (float, optional): A scaling factor that adjusts the impact of volatility on the cost.
                               It is a constant that can be tuned based on empirical observations
                               or market conditions. Defaults to 0.5.

    Returns:
        pd.Series: A pandas Series representing the estimated market impact costs for each asset,
                   where the index corresponds to the asset identifiers and the values represent
                   the calculated costs based on the square-root market impact model. The costs are
                   calculated in the same units as the trade notional, and any missing values in
                   the input Series are handled by replacing them with zeros in the output Series.
    """
    adv_notional = adv_notional.replace(0, np.nan)
    participation = (trade_notional.abs() / adv_notional).clip(lower=0.0)
    cost_bps = eta * daily_vol * np.sqrt(participation)
    return (cost_bps * trade_notional.abs()).fillna(0.0)


def apply_borrow_filter(
    target_weights: pd.Series, borrow_fee_bps: pd.Series, max_borrow_bps: float = 300.0
) -> pd.Series:
    """
    Zero-out short positions in names whose borrow fee exceeds threshold.

    Args:
        target_weights (pd.Series): A pandas Series representing the target weights for each asset,
                                    where the index corresponds to the asset identifiers (e.g.,
                                    tickers) and the values represent the desired weights for each
                                    asset.
        borrow_fee_bps (pd.Series): A pandas Series representing the borrow fees for each asset,
                                    where the index corresponds to the asset identifiers and the
                                    values represent the borrow fees in basis points (bps).
        max_borrow_bps (float, optional): A threshold value for the maximum allowable borrow fee in
                                          basis points (bps). Assets with borrow fees exceeding
                                          this threshold will have their target weights set to
                                          zero. Defaults to 300.0 bps.

    Returns:
        pd.Series: A pandas Series representing the adjusted target weights for each asset, where
                   the index corresponds to the asset identifiers and the values represent the
                   modified weights after applying the borrow fee filter. Assets with borrow fees
                   exceeding the specified threshold will have their weights set to zero, while
                   other assets will retain their original target weights. The output Series will
                   have the same index as the input target_weights Series.
    """
    w = target_weights.copy()
    blocked = (w < 0) & (borrow_fee_bps > max_borrow_bps)
    w[blocked] = 0.0
    return w


def turnover_penalized_rebalance(
    target_weights: pd.Series, prev_weights: pd.Series, kappa: float = 0.0
) -> pd.Series:
    """
    Shrink weight changes toward prev_weights by penalty kappa in [0,1).

    new_w = prev_w + (1 - kappa) * (target_w - prev_w)
    kappa=0 -> full rebalance to target; kappa->1 -> no rebalance (freeze).

    Args:
        target_weights (pd.Series): A pandas Series representing the target weights for each asset,
                                    where the index corresponds to the asset identifiers (e.g.,
                                    tickers) and the values represent the desired weights for each
                                    asset.
        prev_weights (pd.Series): A pandas Series representing the previous weights for each asset,
                                  where the index corresponds to the asset identifiers and the
                                  values represent the weights from the previous period.
        kappa (float, optional): A penalty factor in the range [0, 1) that controls the degree of
                                 shrinkage applied to weight changes. A value of 0 indicates full
                                 rebalance to target weights, while a value approaching 1 indicates
                                 no rebalance (freeze). Defaults to 0.0.

    Returns:
        pd.Series: A pandas Series representing the adjusted weights for each asset after applying
                   the turnover penalty. The index corresponds to the asset identifiers, and the
                   values represent the modified weights that are shrunk toward the previous weights
                   based on the specified penalty factor kappa. The output Series will have the same
                   index as the input target_weights Series.
    """
    if not (0.0 <= kappa < 1.0):
        raise ValueError("kappa must be in [0, 1)")
    aligned_target, aligned_prev = target_weights.align(prev_weights, fill_value=0.0)
    return aligned_prev + (1.0 - kappa) * (aligned_target - aligned_prev)


def daily_pnl(
    weights: pd.Series, fwd_returns: pd.Series, gross_notional: float = 1.0
) -> float:
    """
    Simple daily P&L in return units: sum(w_i * r_i) scaled by gross_notional (weights sum to target gross).

    Args:
        weights (pd.Series): A pandas Series representing the weights of each asset in the
                             portfolio, where the index corresponds to the asset identifiers (e.g.,
                             tickers) and the values represent the weights assigned to each asset.
        fwd_returns (pd.Series): A pandas Series representing the forward returns of each asset,
                                 where the index corresponds to the asset identifiers and the
                                 values represent the returns for the next period.
        gross_notional (float, optional): A scaling factor that represents the total notional value
                                          of the portfolio. The daily P&L is calculated as the
                                          weighted sum of the forward returns, scaled by this gross
                                          notional value. Defaults to 1.0.0.

    Returns:
        float: The calculated daily profit and loss (P&L) in return units, which is the sum of the
               product of weights and forward returns, scaled by the gross notional value.
    """
    w, r = weights.align(fwd_returns, fill_value=0.0)
    return float((w * r).sum() * gross_notional)


def run_backtest(
    scores: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    adv_notional: pd.DataFrame,
    daily_vol: pd.DataFrame,
    borrow_fee_bps: pd.DataFrame | None = None,
    quintile: float = 0.2,
    kappa: float = 0.1,
    eta: float = 0.5,
    commission_bps: float = 5.0,
    max_borrow_bps: float = 300.0,
) -> pd.DataFrame:
    """
    Run a dollar-neutral quintile long/short backtest across dates.

    Args:
        scores (pd.DataFrame): A DataFrame of scores for each asset, indexed by date
                               and with columns corresponding to asset identifiers (e.g.,
                               tickers). The scores are used to determine the long and short
                               positions in the portfolio.
        fwd_returns (pd.DataFrame): A DataFrame of forward returns for each asset,
                                    indexed by date and with columns corresponding to asset
                                    identifiers. The forward returns represent the returns for the
                                    next period and are used to calculate the profit and loss (P&L)
                                    of the portfolio.
        adv_notional (pd.DataFrame): A DataFrame of average daily volume (ADV) in notional terms
                                     for each asset, indexed by date and with columns corresponding
                                     to asset identifiers. The ADV is used to estimate market
                                     impact costs.
        daily_vol (pd.DataFrame): A DataFrame of daily volatility for each asset, indexed by date
                                  and with columns corresponding to asset identifiers. The daily
                                  volatility is used in the market impact cost calculation.
        borrow_fee_bps (pd.DataFrame | None, optional): A DataFrame of borrow fees in basis points
                                                        (bps) for each asset, indexed by date and
                                                        with columns corresponding to asset
                                                        identifiers. If provided, assets with
                                                        borrow fees exceeding the specified
                                                        threshold will have their target weights
                                                        set to zero. Defaults to None.
        quintile (float, optional): A float representing the fraction of assets to be included in
                                    the long and short positions. For example, a value of 0.2
                                    indicates that the top 20% of assets will be longed and the
                                    bottom 20% will be shorted. Defaults to 0.2.
        kappa (float, optional): A float in the range [0, 1) that controls the degree of shrinkage
                                 applied to weight changes during rebalancing. A value of 0
                                 indicates full rebalance to target weights, while a value
                                 approaching 1 indicates no rebalance (freeze). Defaults to 0.1.
        eta (float, optional): A scaling factor that adjusts the impact of volatility on the market
                               impact cost. It is used in the square-root market impact model.
                               Defaults to 0.5.
        commission_bps (float, optional): A float representing the commission cost in basis points
                                          (bps) for each trade. The commission cost is calculated
                                          based on the absolute value of the trades executed.
                                          Defaults to 5.0 bps.
        max_borrow_bps (float, optional): A float representing the maximum allowable borrow fee in
                                          basis points (bps). Assets with borrow fees exceeding
                                          this threshold will have their target weights set to
                                          zero. Defaults to 300.0 bps.

    Returns:
        pd.DataFrame: A DataFrame indexed by date with columns: gross_pnl, costs, net_pnl,
                      turnover. Each row represents the results of the backtest for a specific
                      date, including the gross profit and loss (P&L), costs incurred (including
                      market impact and commission), net P&L after costs, and the turnover of the
                      portfolio for that date. The DataFrame provides a summary of the backtest
                      performance across the specified dates, allowing for analysis of the
                      strategy's effectiveness and risk-adjusted returns over time.
    """
    dates = scores.index
    tickers = scores.columns
    prev_w = pd.Series(0.0, index=tickers)
    records = []

    for dt in dates:
        row = scores.loc[dt].dropna()
        if row.empty:
            records.append(
                {
                    "date": dt,
                    "gross_pnl": 0.0,
                    "costs": 0.0,
                    "net_pnl": 0.0,
                    "turnover": 0.0,
                }
            )
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
            target_w = apply_borrow_filter(
                target_w,
                borrow_fee_bps.loc[dt].reindex(tickers).fillna(0.0),
                max_borrow_bps=max_borrow_bps,
            )

        new_w = turnover_penalized_rebalance(target_w, prev_w, kappa=kappa)
        trade = new_w - prev_w
        turnover = trade.abs().sum()

        fwd_r = (
            fwd_returns.loc[dt].reindex(tickers)
            if dt in fwd_returns.index
            else pd.Series(0.0, index=tickers)
        )
        gross_pnl = daily_pnl(new_w, fwd_r)

        adv = (
            adv_notional.loc[dt].reindex(tickers)
            if dt in adv_notional.index
            else pd.Series(np.nan, index=tickers)
        )
        vol = (
            daily_vol.loc[dt].reindex(tickers)
            if dt in daily_vol.index
            else pd.Series(0.02, index=tickers)
        )
        impact = market_impact_cost(trade, adv, vol, eta=eta).sum()
        commission = trade.abs().sum() * (commission_bps / 1e4)
        costs = impact + commission

        records.append(
            {
                "date": dt,
                "gross_pnl": gross_pnl,
                "costs": costs,
                "net_pnl": gross_pnl - costs,
                "turnover": turnover,
            }
        )
        prev_w = new_w

    return pd.DataFrame.from_records(records).set_index("date")


def information_coefficient(scores: pd.Series, fwd_returns: pd.Series) -> float:
    """
    Spearman rank IC between scores and next-period returns for a single cross-section.

    Args:
        scores (pd.Series): A pandas Series representing the scores for each asset in a single
                            cross-section, where the index corresponds to the asset identifiers
                            (e.g., tickers) and the values represent the scores assigned to each
                            asset.
        fwd_returns (pd.Series): A pandas Series representing the forward returns for each asset in
                                 the same cross-section, where the index corresponds to the asset
                                 identifiers and the values represent the returns for the next
                                 period.

    Returns:
        float: The Spearman rank IC between the scores and forward returns.
    """
    s, r = scores.align(fwd_returns, join="inner")
    s = s.dropna()
    r = r.reindex(s.index).dropna()
    common = s.index.intersection(r.index)
    if len(common) < 3:
        return float("nan")
    return float(s.loc[common].corr(r.loc[common], method="spearman"))
