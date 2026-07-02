"""Centralized 100%-path unit test suite for the RFGD pipeline."""
import json
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
