"""Live monitoring: rolling IC/IR, signal decay alerts, KPI artifact generation, Slack hook."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request


def rolling_ic(ic_series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling mean IC over `window` observations."""
    return ic_series.rolling(window, min_periods=max(5, window // 4)).mean()


def rolling_ir(daily_returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling information ratio (annualized) of a daily net-return series."""
    mean = daily_returns.rolling(window, min_periods=max(5, window // 4)).mean()
    std = daily_returns.rolling(window, min_periods=max(5, window // 4)).std(ddof=0)
    return (mean / std.replace(0, np.nan)) * np.sqrt(252)


def signal_decay_alert(rolling_ic_series: pd.Series, lookback_years: int = 3,
                        n_consecutive: int = 10) -> bool:
    """True if the most recent `n_consecutive` rolling-IC obs are all below
    (historical_mean - 1 std) over the trailing `lookback_years`-scaled window."""
    hist = rolling_ic_series.dropna()
    if len(hist) < n_consecutive + 5:
        return False
    threshold = hist.mean() - hist.std(ddof=0)
    recent = hist.iloc[-n_consecutive:]
    return bool((recent < threshold).all())


def build_kpi_report(backtest_df: pd.DataFrame, ic_series: pd.Series | None = None) -> dict:
    """Assemble a JSON-serializable KPI dict from a backtest results DataFrame.

    backtest_df expects columns: gross_pnl, costs, net_pnl, turnover (indexed by date).
    """
    if backtest_df.empty:
        return {"status": "no_data", "n_days": 0}

    net = backtest_df["net_pnl"]
    cum = net.cumsum()
    ann_return = net.mean() * 252
    ann_vol = net.std(ddof=0) * np.sqrt(252)
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    running_max = cum.cummax()
    max_dd = float((cum - running_max).min())
    avg_turnover = float(backtest_df["turnover"].mean())
    total_costs = float(backtest_df["costs"].sum())

    report = {
        "n_days": int(len(backtest_df)),
        "annualized_return": float(ann_return),
        "annualized_vol": float(ann_vol),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_daily_turnover": avg_turnover,
        "total_costs": total_costs,
        "final_cum_pnl": float(cum.iloc[-1]) if len(cum) else 0.0,
    }
    if ic_series is not None and not ic_series.empty:
        r_ic = rolling_ic(ic_series)
        report["mean_ic"] = float(ic_series.mean())
        report["latest_rolling_ic"] = float(r_ic.dropna().iloc[-1]) if r_ic.dropna().shape[0] else None
        report["signal_decay_alert"] = signal_decay_alert(r_ic)
    return report


def write_kpi_artifact(report: dict, output_dir: str | Path = "outputs") -> Path:
    """Write KPI report to outputs/kpi_report.json; returns the path written."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "kpi_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def notify_slack_if_configured(message: str, webhook_url: str | None = None) -> bool:
    """POST a message to a Slack webhook if configured via arg or SLACK_WEBHOOK_URL env var.

    Returns True if a notification was sent, False if skipped (no webhook configured)
    or on failure (failure is swallowed -- monitoring must never crash the pipeline).
    """
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def run_monitoring_cycle(backtest_df: pd.DataFrame, ic_series: pd.Series | None = None,
                          output_dir: str | Path = "outputs",
                          webhook_url: str | None = None) -> dict:
    """Full monitoring cycle: build KPI report, write artifact, conditionally Slack-notify."""
    report = build_kpi_report(backtest_df, ic_series)
    write_kpi_artifact(report, output_dir)
    if report.get("signal_decay_alert"):
        msg = (f"[RFGD ALERT] Signal decay detected: latest rolling IC="
               f"{report.get('latest_rolling_ic')} below historical threshold.")
        sent = notify_slack_if_configured(msg, webhook_url)
        report["slack_notified"] = sent
    else:
        report["slack_notified"] = False
    return report
