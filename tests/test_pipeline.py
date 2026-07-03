"""Centralized 100%-path unit test suite for the RFGD pipeline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from retail_gamma import signals, backtest, portfolio, monitoring


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tickers():
    return ["AAPL", "TSLA", "AMZN", "MSFT", "GOOGL"]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=30, freq="B")


@pytest.fixture
def panel(tickers, dates, rng):
    """Synthetic panel data for scores/returns/adv/vol."""
    n_t, n_d = len(tickers), len(dates)
    scores = pd.DataFrame(rng.normal(size=(n_d, n_t)), index=dates, columns=tickers)
    fwd_returns = pd.DataFrame(rng.normal(scale=0.01, size=(n_d, n_t)), index=dates, columns=tickers)
    adv = pd.DataFrame(rng.uniform(1e7, 1e8, size=(n_d, n_t)), index=dates, columns=tickers)
    vol = pd.DataFrame(rng.uniform(0.01, 0.03, size=(n_d, n_t)), index=dates, columns=tickers)
    borrow = pd.DataFrame(rng.uniform(0, 500, size=(n_d, n_t)), index=dates, columns=tickers)
    return {"scores": scores, "fwd_returns": fwd_returns, "adv": adv, "vol": vol, "borrow": borrow}


# ---------------------------------------------------------------------------
# signals.py
# ---------------------------------------------------------------------------
def test_pct_rank_basic():
    s = pd.Series([3, 1, 2])
    r = signals.pct_rank(s)
    assert r.loc[1] == pytest.approx(1 / 3)
    assert r.loc[0] == 1.0


def test_pct_rank_with_nan():
    s = pd.Series([1.0, np.nan, 2.0])
    r = signals.pct_rank(s)
    assert np.isnan(r.iloc[1])


def test_zscore_normal():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    z = signals.zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)


def test_zscore_zero_variance():
    s = pd.Series([5.0, 5.0, 5.0])
    z = signals.zscore(s)
    assert (z == 0.0).all()


def test_compute_eov_basic():
    call_vol = pd.Series([100.0, 200.0], index=["A", "B"])
    oi = pd.Series([50.0, 400.0], index=["A", "B"])
    avg20 = pd.Series([80.0, 150.0], index=["A", "B"])
    eov = signals.compute_eov(call_vol, oi, avg20)
    assert set(eov.index) == {"A", "B"}
    assert eov.between(0, 1).all()


def test_compute_eov_handles_zero_oi_and_avg():
    call_vol = pd.Series([10.0, 20.0])
    oi = pd.Series([0.0, 5.0])
    avg20 = pd.Series([0.0, 10.0])
    eov = signals.compute_eov(call_vol, oi, avg20)
    assert eov.notna().all()


def test_compute_slcs_basic():
    small_c = pd.Series([10.0])
    total_c = pd.Series([100.0])
    small_p = pd.Series([5.0])
    total_p = pd.Series([50.0])
    out = signals.compute_slcs(small_c, total_c, small_p, total_p)
    assert out.iloc[0] == pytest.approx(0.1 - 0.1)


def test_compute_slcs_zero_denominator():
    small_c = pd.Series([10.0])
    total_c = pd.Series([0.0])
    small_p = pd.Series([5.0])
    total_p = pd.Series([0.0])
    out = signals.compute_slcs(small_c, total_c, small_p, total_p)
    assert out.iloc[0] == 0.0


def test_bipower_variation_normal():
    r = pd.Series(np.random.default_rng(0).normal(scale=0.01, size=50))
    bv = signals.bipower_variation(r)
    assert bv > 0


def test_bipower_variation_insufficient_data():
    r = pd.Series([0.01])
    bv = signals.bipower_variation(r)
    assert np.isnan(bv)


def test_compute_vrp():
    iv = pd.Series([0.4, 0.3])
    rv = pd.Series([0.2, 0.0])
    vrp = signals.compute_vrp(iv, rv)
    assert vrp.iloc[0] == pytest.approx(2.0)
    assert vrp.iloc[1] == 0.0  # zero denominator -> filled 0


def test_combined_alpha_score():
    eov = pd.Series([0.9, 0.1], index=["A", "B"])
    slcs = pd.Series([0.2, -0.2], index=["A", "B"])
    mask = pd.Series([True, False], index=["A", "B"])
    score = signals.compute_combined_alpha_score(eov, slcs, mask)
    assert score.loc["B"] == pytest.approx(0.5 * signals.zscore(slcs).loc["B"])


def test_carry_score():
    vrp = pd.Series([1.2, 0.8])
    skew = pd.Series([0.1, 0.3])
    risk = pd.Series([0.0, 1.0])
    score = signals.compute_carry_score(vrp, skew, risk)
    assert len(score) == 2


# ---------------------------------------------------------------------------
# backtest.py
# ---------------------------------------------------------------------------
def test_market_impact_cost_basic():
    trade = pd.Series([1e6, -2e6])
    adv = pd.Series([1e8, 1e8])
    vol = pd.Series([0.02, 0.02])
    cost = backtest.market_impact_cost(trade, adv, vol)
    assert (cost >= 0).all()


def test_market_impact_cost_zero_adv():
    trade = pd.Series([1e6])
    adv = pd.Series([0.0])
    vol = pd.Series([0.02])
    cost = backtest.market_impact_cost(trade, adv, vol)
    assert cost.iloc[0] == 0.0


def test_apply_borrow_filter_blocks_expensive_shorts():
    w = pd.Series([-0.1, 0.1, -0.05], index=["A", "B", "C"])
    fee = pd.Series([500.0, 500.0, 100.0], index=["A", "B", "C"])
    out = backtest.apply_borrow_filter(w, fee, max_borrow_bps=300.0)
    assert out.loc["A"] == 0.0
    assert out.loc["B"] == 0.1
    assert out.loc["C"] == -0.05


def test_turnover_penalized_rebalance_full():
    target = pd.Series([0.1, -0.1], index=["A", "B"])
    prev = pd.Series([0.0, 0.0], index=["A", "B"])
    out = backtest.turnover_penalized_rebalance(target, prev, kappa=0.0)
    pd.testing.assert_series_equal(out, target)


def test_turnover_penalized_rebalance_invalid_kappa():
    target = pd.Series([0.1])
    prev = pd.Series([0.0])
    with pytest.raises(ValueError):
        backtest.turnover_penalized_rebalance(target, prev, kappa=1.0)


def test_daily_pnl():
    w = pd.Series([0.5, -0.5], index=["A", "B"])
    r = pd.Series([0.02, -0.01], index=["A", "B"])
    pnl = backtest.daily_pnl(w, r)
    assert pnl == pytest.approx(0.5 * 0.02 + (-0.5) * (-0.01))


def test_run_backtest_end_to_end(panel):
    result = backtest.run_backtest(
        panel["scores"], panel["fwd_returns"], panel["adv"], panel["vol"],
        borrow_fee_bps=panel["borrow"], quintile=0.4, kappa=0.1,
    )
    assert not result.empty
    assert {"gross_pnl", "costs", "net_pnl", "turnover"}.issubset(result.columns)
    assert (result["turnover"] >= 0).all()


def test_run_backtest_empty_row_date():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    scores = pd.DataFrame({"A": [np.nan, 0.5]}, index=dates)
    fwd = pd.DataFrame({"A": [0.01, 0.02]}, index=dates)
    adv = pd.DataFrame({"A": [1e8, 1e8]}, index=dates)
    vol = pd.DataFrame({"A": [0.02, 0.02]}, index=dates)
    result = backtest.run_backtest(scores, fwd, adv, vol)
    assert result.loc[dates[0], "net_pnl"] == 0.0


def test_information_coefficient_normal():
    s = pd.Series([1, 2, 3, 4], index=list("abcd"))
    r = pd.Series([0.1, 0.2, 0.3, 0.4], index=list("abcd"))
    ic = backtest.information_coefficient(s, r)
    assert ic == pytest.approx(1.0)


def test_information_coefficient_insufficient_data():
    s = pd.Series([1], index=["a"])
    r = pd.Series([1], index=["a"])
    ic = backtest.information_coefficient(s, r)
    assert np.isnan(ic)


# ---------------------------------------------------------------------------
# portfolio.py
# ---------------------------------------------------------------------------
def test_ewma_vol():
    r = pd.Series(np.random.default_rng(1).normal(scale=0.01, size=100))
    v = portfolio.ewma_vol(r)
    assert (v.dropna() >= 0).all()


def test_vol_target_scalar_normal():
    s = portfolio.vol_target_scalar(0.20, target_vol=0.10, max_leverage=3.0)
    assert s == pytest.approx(0.5)


def test_vol_target_scalar_zero_or_nan():
    assert portfolio.vol_target_scalar(0.0) == 1.0
    assert portfolio.vol_target_scalar(np.nan) == 1.0
    assert portfolio.vol_target_scalar(None) == 1.0


def test_vol_target_scalar_capped():
    s = portfolio.vol_target_scalar(0.01, target_vol=0.10, max_leverage=3.0)
    assert s == 3.0


def test_fractional_kelly_weights_basic():
    mu = pd.Series([0.02, -0.01], index=["A", "B"])
    var = pd.Series([0.01, 0.01], index=["A", "B"])
    w = portfolio.fractional_kelly_weights(mu, var, kelly_fraction=0.5, w_max=0.02)
    assert w.loc["A"] == 0.02  # capped
    assert w.loc["B"] == -0.02  # capped (symmetric)


def test_fractional_kelly_weights_invalid_fraction():
    mu = pd.Series([0.01])
    var = pd.Series([0.01])
    with pytest.raises(ValueError):
        portfolio.fractional_kelly_weights(mu, var, kelly_fraction=0.0)


def test_fractional_kelly_weights_zero_variance():
    mu = pd.Series([0.01])
    var = pd.Series([0.0])
    w = portfolio.fractional_kelly_weights(mu, var)
    assert w.iloc[0] == 0.0


def test_drawdown_governor_thresholds():
    cum = pd.Series([0.0, 0.10, 0.05, -0.02, -0.05])
    mult = portfolio.drawdown_governor(cum, dd_cut_50=-0.10, dd_cut_100=-0.20)
    assert mult.isin([0.0, 0.5, 1.0]).all()


def test_drawdown_governor_full_cut():
    cum = pd.Series([1.0, 1.0, 0.0])  # -100% drawdown from peak
    mult = portfolio.drawdown_governor(cum, dd_cut_50=-0.05, dd_cut_100=-0.08)
    assert mult.iloc[-1] == 0.0


def test_gamma_squeeze_circuit_breaker():
    move = pd.Series([1.0, 4.0, 5.0])
    slcs_z = pd.Series([0.5, 0.5, 2.0])
    trig = portfolio.gamma_squeeze_circuit_breaker(move, slcs_z, atr_threshold=3.0,
                                                     slcs_zscore_threshold=1.28)
    assert trig.tolist() == [False, False, True]


def test_ledoit_wolf_shrink_cov():
    df = pd.DataFrame(np.random.default_rng(2).normal(size=(50, 3)), columns=list("XYZ"))
    cov = portfolio.ledoit_wolf_shrink_cov(df, shrink=0.5)
    assert cov.shape == (3, 3)
    assert np.allclose(cov.values, cov.values.T, atol=1e-8)


def test_ledoit_wolf_shrink_cov_invalid_shrink():
    df = pd.DataFrame(np.random.default_rng(3).normal(size=(10, 2)))
    with pytest.raises(ValueError):
        portfolio.ledoit_wolf_shrink_cov(df, shrink=1.5)


# ---------------------------------------------------------------------------
# monitoring.py
# ---------------------------------------------------------------------------
def test_rolling_ic_and_ir():
    ic = pd.Series(np.random.default_rng(4).normal(scale=0.05, size=100))
    r_ic = monitoring.rolling_ic(ic, window=20)
    assert r_ic.dropna().shape[0] > 0

    rets = pd.Series(np.random.default_rng(5).normal(scale=0.01, size=100))
    r_ir = monitoring.rolling_ir(rets, window=20)
    assert r_ir.dropna().shape[0] > 0


def test_signal_decay_alert_triggers():
    # historical noise then a sharp sustained drop
    hist = np.random.default_rng(6).normal(loc=0.05, scale=0.01, size=100)
    decayed = np.full(10, -0.5)
    s = pd.Series(np.concatenate([hist, decayed]))
    assert monitoring.signal_decay_alert(s, n_consecutive=10) is True


def test_signal_decay_alert_insufficient_data():
    s = pd.Series([0.1, 0.2])
    assert monitoring.signal_decay_alert(s) is False


def test_build_kpi_report_empty():
    report = monitoring.build_kpi_report(pd.DataFrame())
    assert report["status"] == "no_data"


def test_build_kpi_report_normal():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame({
        "gross_pnl": np.random.default_rng(7).normal(scale=0.01, size=50),
        "costs": np.abs(np.random.default_rng(8).normal(scale=0.0005, size=50)),
        "turnover": np.random.default_rng(9).uniform(0, 0.5, size=50),
    }, index=dates)
    df["net_pnl"] = df["gross_pnl"] - df["costs"]
    report = monitoring.build_kpi_report(df)
    assert "sharpe_ratio" in report
    assert report["n_days"] == 50


def test_build_kpi_report_with_ic():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "gross_pnl": np.zeros(30), "costs": np.zeros(30), "turnover": np.zeros(30),
    }, index=dates)
    df["net_pnl"] = 0.0
    ic = pd.Series(np.random.default_rng(10).normal(scale=0.02, size=30))
    report = monitoring.build_kpi_report(df, ic_series=ic)
    assert "mean_ic" in report
    assert "signal_decay_alert" in report


def test_write_kpi_artifact(tmp_path):
    report = {"a": 1}
    path = monitoring.write_kpi_artifact(report, output_dir=tmp_path)
    assert path.exists()
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == {"a": 1}


def test_notify_slack_no_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    sent = monitoring.notify_slack_if_configured("test message")
    assert sent is False


def test_notify_slack_with_bad_webhook():
    # Invalid URL scheme should be swallowed and return False, never raise.
    sent = monitoring.notify_slack_if_configured("test", webhook_url="http://invalid.invalid/hook")
    assert sent is False


def test_run_monitoring_cycle_no_alert(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "gross_pnl": np.full(30, 0.001), "costs": np.zeros(30), "turnover": np.zeros(30),
    }, index=dates)
    df["net_pnl"] = df["gross_pnl"]
    report = monitoring.run_monitoring_cycle(df, output_dir=tmp_path)
    assert report["slack_notified"] is False
    assert (tmp_path / "kpi_report.json").exists()


def test_run_monitoring_cycle_with_alert(tmp_path, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "gross_pnl": np.zeros(100), "costs": np.zeros(100), "turnover": np.zeros(100),
    }, index=dates)
    df["net_pnl"] = 0.0
    hist = np.random.default_rng(11).normal(loc=0.05, scale=0.01, size=90)
    decayed = np.full(10, -0.5)
    ic = pd.Series(np.concatenate([hist, decayed]))
    report = monitoring.run_monitoring_cycle(df, ic_series=ic, output_dir=tmp_path)
    assert "signal_decay_alert" in report
    assert report["slack_notified"] is False  # no webhook configured in test env


# ---------------------------------------------------------------------------
# data_adapter.py
# ---------------------------------------------------------------------------
from retail_gamma import data_adapter as da


def test_synthetic_adapter_equity_ohlcv():
    adapter = da.SyntheticDataAdapter(seed=1)
    df = adapter.get_equity_ohlcv(["AAPL", "TSLA"], "2024-01-01", "2024-02-01")
    assert set(df["ticker"].unique()) == {"AAPL", "TSLA"}
    assert {"open", "high", "low", "close", "volume", "date"}.issubset(df.columns)
    assert (df["close"] > 0).all()


def test_synthetic_adapter_option_flow_proxy():
    adapter = da.SyntheticDataAdapter(seed=2)
    df = adapter.get_option_flow_proxy(["AAPL"], "2019-01-01", "2020-01-01")
    assert {"call_volume", "call_oi", "small_lot_call_volume",
            "put_volume", "put_oi", "small_lot_put_volume"}.issubset(df.columns)
    assert (df["small_lot_call_volume"] <= df["call_volume"] + 1e-6).all()


def test_synthetic_adapter_regime_shift_present():
    adapter = da.SyntheticDataAdapter(seed=3)
    df = adapter.get_option_flow_proxy(["AAPL"], "2019-06-01", "2020-01-01")
    pre = df[df["date"] < pd.Timestamp("2019-10-02")]["_signal_strength"].mean()
    post = df[df["date"] >= pd.Timestamp("2019-10-02")]["_signal_strength"].mean()
    assert post > pre


def test_yfinance_adapter_import_error_or_construct(monkeypatch):
    # Either yfinance is installed (construct succeeds) or ImportError raised cleanly.
    try:
        adapter = da.YFinanceAdapter()
        assert hasattr(adapter, "get_equity_ohlcv")
    except ImportError:
        pass


def test_yfinance_adapter_option_flow_not_implemented():
    try:
        adapter = da.YFinanceAdapter()
    except ImportError:
        pytest.skip("yfinance not installed in this environment")
    with pytest.raises(NotImplementedError):
        adapter.get_option_flow_proxy(["AAPL"], "2024-01-01", "2024-01-05")


def test_occ_adapter_stubs_raise():
    adapter = da.OCCPublicDataAdapter()
    with pytest.raises(NotImplementedError):
        adapter.get_equity_ohlcv(["AAPL"], "2024-01-01", "2024-01-05")
    with pytest.raises(NotImplementedError):
        adapter.get_option_flow_proxy(["AAPL"], "2024-01-01", "2024-01-05")


def test_get_default_adapter_falls_back_to_synthetic():
    adapter = da.get_default_adapter(prefer_live=False)
    assert isinstance(adapter, da.SyntheticDataAdapter)


def test_get_default_adapter_prefer_live_no_network():
    # In this sandboxed test environment live network egress to Yahoo is blocked,
    # so this must gracefully fall back rather than raising.
    adapter = da.get_default_adapter(prefer_live=True)
    assert adapter is not None


def test_get_default_flow_adapter_returns_synthetic_and_works():
    flow_adapter = da.get_default_flow_adapter(seed=99)
    assert isinstance(flow_adapter, da.SyntheticDataAdapter)
    df = flow_adapter.get_option_flow_proxy(["AAPL"], "2024-01-01", "2024-02-01")
    assert not df.empty


def test_yfinance_adapter_flow_never_called_via_default_flow_adapter():
    # Regression guard: get_default_flow_adapter must never return an adapter
    # whose get_option_flow_proxy raises NotImplementedError.
    flow_adapter = da.get_default_flow_adapter()
    result = flow_adapter.get_option_flow_proxy(["MSFT"], "2024-01-01", "2024-01-10")
    assert not result.empty


def test_market_data_adapter_protocol_conformance():
    adapter = da.SyntheticDataAdapter()
    assert isinstance(adapter, da.MarketDataAdapter)


# ---------------------------------------------------------------------------
# visualization.py
# ---------------------------------------------------------------------------
from retail_gamma import visualization as viz


def test_plot_small_lot_share_regime(tmp_path):
    idx = pd.date_range("2019-01-01", periods=50, freq="B")
    s = pd.Series(np.linspace(0.3, 0.45, 50), index=idx)
    fig, paths = viz.plot_small_lot_share_regime(s, {"Zero-commission": "2019-10-02"}, tmp_path)
    assert paths["html"] is not None
    assert Path(paths["html"]).exists()


def test_plot_rolling_ic_regime(tmp_path):
    idx = pd.date_range("2019-01-01", periods=50, freq="B")
    s = pd.Series(np.random.default_rng(0).normal(0.02, 0.05, 50), index=idx)
    fig, paths = viz.plot_rolling_ic_regime(s, {"Regime": "2019-10-02"}, tmp_path)
    assert Path(paths["html"]).exists()


def test_plot_equity_curve_with_benchmark(tmp_path):
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    strat = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 60), index=idx)
    bench = pd.Series(np.random.default_rng(2).normal(0.0002, 0.008, 60), index=idx)
    fig, paths = viz.plot_equity_curve(strat, bench, tmp_path)
    assert Path(paths["html"]).exists()


def test_plot_equity_curve_no_benchmark_no_drawdown(tmp_path):
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    strat = pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, 30), index=idx)
    fig, paths = viz.plot_equity_curve(strat, None, tmp_path, drawdown=False, name="eq_no_dd")
    assert Path(paths["html"]).exists()


def test_plot_signal_decay_ribbon(tmp_path):
    df = pd.DataFrame({
        "year": [2019]*10 + [2020]*10,
        "day_of_year": list(range(10)) * 2,
        "rolling_ic": np.random.default_rng(4).normal(0, 0.05, 20),
    })
    fig, paths = viz.plot_signal_decay_ribbon(df, tmp_path)
    assert Path(paths["html"]).exists()


def test_plot_risk_dashboard(tmp_path):
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "rolling_sharpe": np.random.default_rng(5).normal(1, 0.3, 40),
        "drawdown": -np.abs(np.random.default_rng(6).normal(0, 0.02, 40)),
        "turnover": np.random.default_rng(7).uniform(0, 0.3, 40),
        "gross_exposure": np.random.default_rng(8).uniform(0.8, 1.2, 40),
    }, index=idx)
    fig, paths = viz.plot_risk_dashboard(df, tmp_path)
    assert Path(paths["html"]).exists()
    # Regression guard: the turnover panel must contain real, non-degenerate data
    # (previously rendered via go.Bar, which collapses to an invisible panel over
    # long daily histories -- now a filled-area line, verified here by trace type
    # and by the presence of nonzero y-values).
    turnover_traces = [t for t in fig.data if t.name in ("Turnover", "Turnover (20d MA)")]
    assert len(turnover_traces) == 2
    for t in turnover_traces:
        assert t.type == "scatter"
        assert np.nanmax(np.abs(np.asarray(t.y, dtype=float))) > 0


def test_plot_risk_dashboard_large_history_turnover_visible(tmp_path):
    # Simulate a multi-year daily history (the regime that broke go.Bar rendering)
    # and assert the turnover trace still carries the full, non-empty data series.
    idx = pd.date_range("2019-10-02", "2026-06-30", freq="B")
    n = len(idx)
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "rolling_sharpe": rng.normal(1, 0.5, n),
        "drawdown": -np.abs(rng.normal(0, 0.05, n)),
        "turnover": rng.uniform(1.5, 3.0, n),
        "gross_exposure": rng.uniform(0.5, 1.5, n),
    }, index=idx)
    fig, paths = viz.plot_risk_dashboard(df, tmp_path, name="risk_dashboard_large")
    turnover_trace = next(t for t in fig.data if t.name == "Turnover")
    y = np.asarray(turnover_trace.y, dtype=float)
    assert len(y) == n
    assert np.isclose(np.nanmean(y), df["turnover"].mean(), atol=1e-9)
    assert Path(paths["html"]).exists()


def test_plot_cpcv_fold_diagnostics(tmp_path):
    df = pd.DataFrame({"oos_sharpe": np.random.default_rng(9).normal(1, 0.5, 15)})
    fig, paths = viz.plot_cpcv_fold_diagnostics(df, tmp_path)
    assert Path(paths["html"]).exists()


def test_plot_pbo_histogram(tmp_path):
    logits = np.random.default_rng(10).normal(0, 1, 200)
    fig, paths = viz.plot_pbo_histogram(logits, 0.35, tmp_path)
    assert Path(paths["html"]).exists()


# ---------------------------------------------------------------------------
# validation.py (CPCV, purging/embargo, PBO, Deflated Sharpe)
# ---------------------------------------------------------------------------
from retail_gamma import validation as val


def test_combinatorial_purged_cv_basic():
    splits = list(val.combinatorial_purged_cv(n_obs=120, n_groups=6, n_test_groups=2,
                                               label_horizon=3, embargo_frac=0.01))
    assert len(splits) > 0
    for train_idx, test_idx, combo in splits:
        assert len(test_idx) > 0
        # purging: no overlap between train and test
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_combinatorial_purged_cv_invalid_groups():
    with pytest.raises(ValueError):
        list(val.combinatorial_purged_cv(n_obs=100, n_groups=2, n_test_groups=2))


def test_combinatorial_purged_cv_insufficient_obs():
    with pytest.raises(ValueError):
        list(val.combinatorial_purged_cv(n_obs=3, n_groups=6, n_test_groups=2))


def test_run_cpcv_backtest_basic():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    returns = pd.Series(np.random.default_rng(11).normal(0.0005, 0.01, 120), index=idx)
    result = val.run_cpcv_backtest(returns, n_groups=6, n_test_groups=2, label_horizon=3)
    assert not result.empty
    assert "oos_sharpe" in result.columns


def test_run_cpcv_backtest_insufficient_obs():
    returns = pd.Series(np.random.default_rng(12).normal(0, 0.01, 5))
    with pytest.raises(ValueError):
        val.run_cpcv_backtest(returns)


def test_probability_of_backtest_overfitting_basic():
    sharpes = pd.Series(np.random.default_rng(13).normal(1.0, 0.4, 30))
    pbo, logits = val.probability_of_backtest_overfitting(sharpes, n_trials_simulated=100)
    assert 0.0 <= pbo <= 1.0
    assert len(logits) > 0


def test_probability_of_backtest_overfitting_insufficient_data():
    sharpes = pd.Series([1.0, 2.0])
    pbo, logits = val.probability_of_backtest_overfitting(sharpes)
    assert np.isnan(pbo)
    assert len(logits) == 0


def test_deflated_sharpe_ratio_basic():
    dsr = val.deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=10, n_obs=500)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_ratio_single_trial():
    dsr = val.deflated_sharpe_ratio(observed_sharpe=1.0, n_trials=1, n_obs=250)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_ratio_insufficient_obs():
    dsr = val.deflated_sharpe_ratio(observed_sharpe=1.0, n_trials=5, n_obs=1)
    assert np.isnan(dsr)
