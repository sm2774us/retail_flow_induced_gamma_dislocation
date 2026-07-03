"""State-of-the-art Plotly visualization layer for the RFGD strategy.

Every chart function returns a `plotly.graph_objects.Figure` AND writes both
a standalone interactive `.html` (for the trading desk to open independently,
no Python/Jupyter required) and a static `.png` (for the tex dissertation /
slide decks / email) to the given output directory. This mirrors the
dual-output convention used by internal research-viz libraries at systematic
funds (interactive for research iteration, static for compliance/archival).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

TEMPLATE = "plotly_white"
COLORWAY = ["#0B5FFF", "#FF5A5F", "#00B67A", "#F5A623", "#8E44AD", "#1ABC9C"]


def _persist(fig: go.Figure, name: str, output_dir: str | Path,
             width: int = 1200, height: int = 650) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{name}.html"
    png_path = out_dir / f"{name}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=2)
    except Exception as e:  # pragma: no cover - kaleido availability dependent
        png_path = None
    return {"html": str(html_path), "png": str(png_path) if png_path else None}


def _base_layout(fig: go.Figure, title: str, yaxis_title: str = "", xaxis_title: str = "Date") -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        colorway=COLORWAY,
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=80, b=50),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


def plot_small_lot_share_regime(small_lot_share: pd.Series, regime_dates: dict[str, str],
                                 output_dir: str | Path, name: str = "small_lot_share_regime") -> tuple[go.Figure, dict]:
    """Small-lot call volume share over time with regime-shift annotations."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=small_lot_share.index, y=small_lot_share.values, mode="lines",
                              line=dict(width=2.5, color=COLORWAY[0]),
                              name="Small-lot call volume share (10d MA)",
                              fill="tozeroy", fillcolor="rgba(11,95,255,0.08)"))
    for label, dt in regime_dates.items():
        fig.add_vline(x=dt, line_dash="dash", line_color=COLORWAY[1], opacity=0.7)
        fig.add_annotation(x=dt, y=1.0, yref="paper", text=label, showarrow=False,
                            yshift=10, font=dict(size=11, color=COLORWAY[1]))
    fig = _base_layout(fig, "Retail Proxy: Small-Lot Call Volume Share Through Time",
                        yaxis_title="Share of total call volume")
    fig.update_yaxes(tickformat=".0%")
    return fig, _persist(fig, name, output_dir)


def plot_rolling_ic_regime(rolling_ic: pd.Series, regime_dates: dict[str, str],
                            output_dir: str | Path, name: str = "rolling_ic_regime") -> tuple[go.Figure, dict]:
    """Rolling information coefficient with a zero line and regime annotations."""
    colors = np.where(rolling_ic.values >= 0, COLORWAY[2], COLORWAY[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic.values, mode="lines",
                              line=dict(width=2, color=COLORWAY[0]), name="Rolling IC (EOV vs fwd return)"))
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)")
    for label, dt in regime_dates.items():
        fig.add_vline(x=dt, line_dash="dash", line_color=COLORWAY[1], opacity=0.7)
        fig.add_annotation(x=dt, y=1.0, yref="paper", text=label, showarrow=False,
                            yshift=10, font=dict(size=11, color=COLORWAY[1]))
    fig = _base_layout(fig, "Signal Regime Shift: Rolling Cross-Sectional Information Coefficient",
                        yaxis_title="Information Coefficient (Spearman)")
    return fig, _persist(fig, name, output_dir)


def plot_equity_curve(strategy_returns: pd.Series, benchmark_returns: pd.Series | None,
                       output_dir: str | Path, name: str = "equity_curve",
                       drawdown: bool = True) -> tuple[go.Figure, dict]:
    """Cumulative P&L equity curve, optionally with a drawdown subplot and benchmark overlay."""
    cum = strategy_returns.cumsum()
    rows = 2 if drawdown else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.72, 0.28] if drawdown else [1.0],
                         subplot_titles=("Cumulative Net P&L", "Drawdown") if drawdown else None)
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="RFGD strategy", mode="lines",
                              line=dict(width=2.5, color=COLORWAY[0])), row=1, col=1)
    if benchmark_returns is not None:
        bcum = benchmark_returns.reindex(strategy_returns.index).fillna(0).cumsum()
        fig.add_trace(go.Scatter(x=bcum.index, y=bcum.values, name="Benchmark", mode="lines",
                                  line=dict(width=1.5, dash="dot", color=COLORWAY[3])), row=1, col=1)
    if drawdown:
        running_max = cum.cummax()
        dd = cum - running_max
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines",
                                  line=dict(width=1.5, color=COLORWAY[1]),
                                  fill="tozeroy", fillcolor="rgba(255,90,95,0.15)"), row=2, col=1)
    fig = _base_layout(fig, "RFGD Strategy: Cumulative Performance & Drawdown", yaxis_title="Cumulative return")
    fig.update_layout(height=700)
    return fig, _persist(fig, name, output_dir, height=700)


def plot_signal_decay_ribbon(rolling_ic_by_period: pd.DataFrame, output_dir: str | Path,
                              name: str = "signal_decay_ribbon") -> tuple[go.Figure, dict]:
    """Multi-year rolling-IC ribbon chart -- one line per calendar-year regime, x-axis = trading day
    of year -- lets a PM see at a glance whether the signal is decaying year over year."""
    fig = go.Figure()
    years = sorted(rolling_ic_by_period["year"].unique())
    palette = COLORWAY * (1 + len(years) // len(COLORWAY))
    for i, yr in enumerate(years):
        sub = rolling_ic_by_period[rolling_ic_by_period["year"] == yr]
        fig.add_trace(go.Scatter(x=sub["day_of_year"], y=sub["rolling_ic"], mode="lines",
                                  name=str(yr), line=dict(width=2, color=palette[i])))
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)")
    fig = _base_layout(fig, "Year-over-Year Signal Decay Ribbon (Rolling IC)",
                        yaxis_title="Rolling IC", xaxis_title="Trading day of year")
    return fig, _persist(fig, name, output_dir)


def plot_risk_dashboard(kpi_history: pd.DataFrame, output_dir: str | Path,
                         name: str = "risk_dashboard") -> tuple[go.Figure, dict]:
    """4-panel institutional risk dashboard: Sharpe, drawdown, turnover, gross exposure.

    NOTE on the turnover panel: with long daily histories (multi-year, thousands of
    trading days) a `go.Bar` trace degenerates into sub-pixel-width bars that render
    as an effectively blank panel in static (rasterized) PNG export, even though the
    underlying data is non-empty -- each bar's rendered width falls below one pixel
    and anti-aliasing washes the color out. We therefore render turnover as a filled
    area line (matching the drawdown panel's proven-visible treatment) plus a light
    trailing rolling-mean overlay, which remains legible at any history length.
    """
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Rolling Sharpe (63d)", "Drawdown", "Daily Turnover", "Gross Exposure Utilization"))

    fig.add_trace(go.Scatter(x=kpi_history.index, y=kpi_history["rolling_sharpe"],
                              line=dict(color=COLORWAY[0], width=2), name="Sharpe"), row=1, col=1)
    fig.add_trace(go.Scatter(x=kpi_history.index, y=kpi_history["drawdown"], fill="tozeroy",
                              line=dict(color=COLORWAY[1], width=1.5), name="Drawdown",
                              fillcolor="rgba(255,90,95,0.15)"), row=1, col=2)

    turnover = kpi_history["turnover"]
    fig.add_trace(go.Scatter(x=turnover.index, y=turnover.values, mode="lines",
                              line=dict(color=COLORWAY[2], width=1.2), name="Turnover",
                              fill="tozeroy", fillcolor="rgba(0,182,122,0.25)"), row=2, col=1)
    turnover_ma = turnover.rolling(20, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=turnover_ma.index, y=turnover_ma.values, mode="lines",
                              line=dict(color=COLORWAY[2], width=2, dash="solid"),
                              name="Turnover (20d MA)"), row=2, col=1)

    fig.add_trace(go.Scatter(x=kpi_history.index, y=kpi_history["gross_exposure"],
                              line=dict(color=COLORWAY[4], width=2), name="Gross Exp"), row=2, col=2)

    fig.update_layout(template=TEMPLATE, colorway=COLORWAY, showlegend=False, height=650,
                       title=dict(text="RFGD Live Risk Dashboard", x=0.02, font=dict(size=18)),
                       margin=dict(l=50, r=30, t=90, b=40),
                       font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12))
    turnover_max = float(turnover.max()) if len(turnover) else 1.0
    fig.update_yaxes(range=[0, max(turnover_max * 1.15, 1e-6)], row=2, col=1)
    return fig, _persist(fig, name, output_dir, height=650)


def plot_cpcv_fold_diagnostics(fold_results: pd.DataFrame, output_dir: str | Path,
                                name: str = "cpcv_fold_diagnostics") -> tuple[go.Figure, dict]:
    """Box/strip plot of out-of-sample Sharpe across CPCV folds -- the key validation
    chart demonstrating the strategy is not overfit to a single backtest path."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=fold_results["oos_sharpe"], name="OOS Sharpe (CPCV folds)",
                          boxpoints="all", jitter=0.4, pointpos=-1.8,
                          marker_color=COLORWAY[0], line_color=COLORWAY[0]))
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)")
    fig = _base_layout(fig, "Combinatorial Purged Cross-Validation: Out-of-Sample Sharpe Distribution",
                        yaxis_title="Out-of-sample Sharpe ratio", xaxis_title="")
    return fig, _persist(fig, name, output_dir)


def plot_pbo_histogram(rank_logits: np.ndarray, pbo_estimate: float, output_dir: str | Path,
                        name: str = "pbo_histogram") -> tuple[go.Figure, dict]:
    """Probability-of-Backtest-Overfitting histogram (Bailey-Borwein-Lopez de Prado)."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rank_logits, nbinsx=30, marker_color=COLORWAY[0],
                                name="Rank logit distribution"))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORWAY[1],
                  annotation_text=f"PBO = {pbo_estimate:.1%}")
    fig = _base_layout(fig, "Probability of Backtest Overfitting (CSCV)",
                        yaxis_title="Frequency", xaxis_title="Logit of relative rank")
    return fig, _persist(fig, name, output_dir)
