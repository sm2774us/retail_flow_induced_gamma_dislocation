"""Modular market-data adapter layer.

Design goal: the strategy code (signals/backtest/portfolio) must never import a
vendor SDK directly. It depends only on the `MarketDataAdapter` protocol below.
At a quant firm, swap `YFinanceAdapter` for e.g. `BloombergAdapter`,
`RefinitivAdapter`, or an internal tick-database adapter (Kdb+/Arctic/Parquet
lake) without touching a single line of signal/backtest/portfolio code --
this is the same "data adapter" pattern used at Citadel/Jane Street/Two Sigma
internal research platforms (vendor-agnostic feature store).

Three concrete adapters are provided:
  - YFinanceAdapter        : free EOD equity data via yfinance (real, production network required)
  - OCCPublicDataAdapter    : stub for OCC public customer-size volume feed (options flow proxy)
  - SyntheticDataAdapter    : deterministic, seed-controlled synthetic generator used for
                              offline CI/unit-testing and for this repo's research notebook
                              when live network egress to market-data vendors is unavailable.
"""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Protocol (structural interface) all adapters must satisfy
# ---------------------------------------------------------------------------
"""
Protocol for market-data adapters. All concrete adapters must implement these two methods, and the strategy code (signals/backtest/portfolio) depends only on this interface, not on any vendor SDK directly.

Concrete adapters provided:
  - YFinanceAdapter        : free EOD equity data via yfinance (real, production network required)
  - OCCPublicDataAdapter   : stub for OCC public customer-size volume feed (options flow proxy)
  - SyntheticDataAdapter   : deterministic, seed-controlled synthetic generator used for
                             offline CI/unit-testing and for this repo's research notebook
                             when live network egress to market-data vendors is unavailable.

Use `get_default_adapter()` to get a production-ready adapter for equity OHLCV, and `get_default_flow_adapter()` for the options-flow leg. In production, replace the body of `get_default_flow_adapter()` with a licensed vendor adapter (OptionMetrics IvyDB, CBOE DataShop, Bloomberg OMON) or a completed `OCCPublicDataAdapter` -- no other code needs to change since callers only depend on the `MarketDataAdapter` protocol.

Example usage:
    adapter = get_default_adapter()
    df_ohlcv = adapter.get_equity_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")
    df_flow = get_default_flow_adapter().get_option_flow_proxy(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")
    df_merged = pd.merge(df_ohlcv, df_flow, on=["date", "ticker"], how="outer")
    df_merged.to_csv("merged_data.csv", index=False)
"""


@runtime_checkable
class MarketDataAdapter(Protocol):
    def get_equity_ohlcv(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Get equity OHLCV data for the given tickers and date range.

        Args:
            tickers (list[str]): List of equity tickers (e.g., ["AAPL", "MSFT"]).
            start (str): Start date in "YYYY-MM-DD" format.
            end (str): End date in "YYYY-MM-DD" format.

        Returns:
            pd.DataFrame: A long-format DataFrame with columns [date, ticker, open, high, low,
                          close, volume].
        """
        ...

    def get_option_flow_proxy(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Get a proxy for option flow data for the given tickers and date range.

        Args:
            tickers (list[str]): List of equity tickers (e.g., ["AAPL", "MSFT"]).
            start (str): Start date in "YYYY-MM-DD" format.
            end (str): End date in "YYYY-MM-DD" format.

        Returns:
            pd.DataFrame: A long-format DataFrame with columns [date, ticker, call_volume, call_oi,
                          small_lot_call_volume, put_volume, put_oi, small_lot_put_volume].
        """
        ...


# ---------------------------------------------------------------------------
# Production adapter: Yahoo Finance (equities only -- free tier, EOD, no options OI)
# ---------------------------------------------------------------------------
class YFinanceAdapter:
    """Real production-shaped adapter backed by `yfinance`.

    Equity OHLCV is genuinely sourced from Yahoo Finance. Because free Yahoo
    data does not expose historical single-name option OI/volume-by-lot-size,
    `get_option_flow_proxy` here derives a *documented, clearly-labeled proxy*
    from yfinance's live options chain snapshot (today's chain only) -- this
    is intentionally NOT used for the historical backtest (see
    `SyntheticDataAdapter` / `OCCPublicDataAdapter` for that); it exists so
    the exact same interface can be smoke-tested against a live feed.

    Swap-out instructions for a production quant desk:
        adapter = BloombergAdapter(...)   # implements the same 2 methods
        # no other code in signals.py / backtest.py / portfolio.py changes.
    """

    def __init__(self):
        """
        Initialize the YFinanceAdapter.

        Raises:
            ImportError: If the `yfinance` package is not installed.
        """
        try:
            import yfinance as yf  # noqa: F401  (import guarded; optional dependency)

            self._yf = yf
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "yfinance is required for YFinanceAdapter. Install via `pip install yfinance`."
            ) from e

    def get_equity_ohlcv(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        See `MarketDataAdapter.get_equity_ohlcv` for interface. Fetches EOD OHLCV data from Yahoo Finance.
        """
        raw = self._yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        frames = []
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].copy()
                sub["ticker"] = t
                sub["date"] = sub.index
                frames.append(sub.reset_index(drop=True))
        else:
            sub = raw.copy()
            sub["ticker"] = tickers[0]
            sub["date"] = sub.index
            frames.append(sub.reset_index(drop=True))
        if not frames:
            return pd.DataFrame(
                columns=["date", "ticker", "open", "high", "low", "close", "volume"]
            )
        out = pd.concat(frames, ignore_index=True)
        out = out.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return out[["date", "ticker", "open", "high", "low", "close", "volume"]]

    def get_option_flow_proxy(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:  # pragma: no cover
        """
        Live-chain snapshot proxy (today only); NOT for historical backtesting. Raises to make misuse explicit rather than silently returning wrong data.

        Raises:
            NotImplementedError: Free yfinance does not provide historical single-name option OI/volume-by-lot. Use OCCPublicDataAdapter (public OCC customer-size feed) or a licensed vendor (OptionMetrics IvyDB, CBOE DataShop, Bloomberg OMON) in production.

        See `MarketDataAdapter.get_option_flow_proxy` for interface.
        """
        raise NotImplementedError(
            "Free yfinance does not provide historical single-name option OI/volume-by-lot. "
            "Use OCCPublicDataAdapter (public OCC customer-size feed) or a licensed vendor "
            "(OptionMetrics IvyDB, CBOE DataShop, Bloomberg OMON) in production."
        )


# ---------------------------------------------------------------------------
# OCC public data adapter stub (production path for options-flow proxy)
# ---------------------------------------------------------------------------
class OCCPublicDataAdapter:
    """Adapter stub for the OCC's public daily customer-size options volume
    feed (the exact data source Barclays' Figure 12-16 used). In production
    this would parse OCC's daily volume reports (or a licensed equivalent such
    as CBOE DataShop / OptionMetrics IvyDB) into the standard schema.

    Left as a documented stub: implement `_fetch_occ_report(date)` against the
    firm's internal data lake / vendor API once available.
    """

    def get_equity_ohlcv(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Stub: this adapter is for options-flow only. Use YFinanceAdapter or an internal equity tick
              adapter for OHLCV.

        Raises:
            NotImplementedError: This adapter does not provide equity OHLCV data.

        See `MarketDataAdapter.get_equity_ohlcv` for interface.
        """
        raise NotImplementedError(
            "Use YFinanceAdapter or an internal equity tick adapter for OHLCV."
        )

    def get_option_flow_proxy(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Stub: implement OCC daily report ingestion (or OptionMetrics IvyDB / CBOE DataShop
              connection) here in production. Schema must match MarketDataAdapter.
              get_option_flow_proxy.

        Raises:
            NotImplementedError: This adapter is a stub; implement OCC daily report ingestion or a
                                 licensed vendor connection in production.

        See `MarketDataAdapter.get_option_flow_proxy` for interface.
        """
        raise NotImplementedError(
            "Implement OCC daily report ingestion (or OptionMetrics IvyDB / CBOE DataShop "
            "connection) here in production. Schema must match MarketDataAdapter.get_option_flow_proxy."
        )


# ---------------------------------------------------------------------------
# Synthetic adapter: offline, deterministic, used by CI + this repo's notebook
# ---------------------------------------------------------------------------
@dataclass
class SyntheticRegimeParams:
    """Calibration knobs for the synthetic retail-flow regime generator.

    Defaults are calibrated to be qualitatively consistent with the Barclays
    (2020) report AND with post-2020 persistence findings (Bryzgalova, Pavlova
    & Sikorskaya 2023): retail small-lot call share plateaus around 40-48%
    rather than reverting, and the EOV->return signal, while still positive
    2021-2026, is weaker than the 2020 peak (partial crowding-out / maturation
    of the phenomenon as market makers adapt hedging algorithms).
    """

    pre_zero_commission_small_lot_share: float = 0.30
    post_2020_peak_small_lot_share: float = 0.46
    post_2023_plateau_small_lot_share: float = 0.42
    signal_strength_2019: float = 0.007
    signal_strength_2020_peak: float = 0.14
    signal_strength_2023_2026: float = 0.078  # decayed vs 2020 peak, still >> pre-2020
    zero_dte_amplification_2023_2026: float = (
        1.15  # 0DTE growth partially offsets decay
    )
    cooling_2021_2022_fraction: float = (
        0.60  # fraction of 2020-peak strength retained in 2021-22
    )


class SyntheticDataAdapter:
    """Deterministic synthetic generator satisfying `MarketDataAdapter`.

    Used when live vendor network egress is unavailable (this sandbox) or for
    fast, reproducible unit tests. Encodes the same qualitative regimes
    documented in the literature review (see RETAIL_GAMMA_STRATEGY.md Sec. 2.3
    and 10) so that downstream signal/backtest code can be validated exactly
    as it would be against a real feed.

    Critically, equity returns and option-flow volume are generated from a
    **shared latent per-ticker/per-date flow factor** (`_latent_flow`), so
    that the dealer-hedging mechanism this strategy targets is actually
    embedded in the synthetic data (regardless of which method -- OHLCV or
    option-flow -- is called first, or how many times each is called), rather
    than the two methods drawing independent, uncorrelated randomness.
    """

    def __init__(self, seed: int = 42, params: SyntheticRegimeParams | None = None):
        """
        Initialize the SyntheticDataAdapter.

        Args:
            seed (int): Random seed for reproducibility.
            params (SyntheticRegimeParams | None): Optional calibration parameters for the
                                                   synthetic regime generator. If None, defaults
                                                   are used.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.params = params or SyntheticRegimeParams()

    def _ticker_seed(self, ticker: str) -> int:
        import hashlib

        h = hashlib.sha256(f"{self.seed}_{ticker}".encode()).hexdigest()
        return int(h[:8], 16)

    def _latent_flow(self, ticker: str, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Deterministic AR(1) latent retail-flow factor per ticker, stable
        across repeated calls / call order for the same (seed, ticker, dates).

        Args:
            ticker (str): The equity ticker symbol.
            dates (pd.DatetimeIndex): The date range for which to generate the latent flow.

        Returns:
            np.ndarray: An array of latent flow values corresponding to the input dates.
        """
        rng = np.random.default_rng(self._ticker_seed(ticker))
        shocks = rng.normal(0, 1, size=len(dates))
        flow = np.zeros(len(dates))
        phi = 0.15  # mild AR(1) persistence -- flow clusters over a few sessions
        for i in range(1, len(dates)):
            flow[i] = phi * flow[i - 1] + np.sqrt(1 - phi**2) * shocks[i]
        flow[0] = shocks[0]
        return flow

    def _regime_signal_strength(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Generate the signal strength for each date based on the synthetic regime parameters.

        Args:
            dates (pd.DatetimeIndex): The date range for which to generate signal strength.

        Returns:
            np.ndarray: An array of signal strength values corresponding to the input dates.
        """
        p = self.params
        s = np.full(len(dates), p.signal_strength_2019)
        s = np.where(
            dates >= pd.Timestamp("2019-10-02"), p.signal_strength_2020_peak, s
        )
        s = np.where(
            dates >= pd.Timestamp("2021-06-01"),
            p.signal_strength_2020_peak * p.cooling_2021_2022_fraction,
            s,
        )
        s = np.where(
            dates >= pd.Timestamp("2023-01-01"),
            p.signal_strength_2023_2026 * p.zero_dte_amplification_2023_2026,
            s,
        )
        return s

    def _regime_small_lot_share(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Generate the small-lot share for each date based on the synthetic regime parameters.

        Args:
            dates (pd.DatetimeIndex): The date range for which to generate small-lot share.

        Returns:
            np.ndarray: An array of small-lot share values corresponding to the input dates.
        """
        p = self.params
        s = np.full(len(dates), p.pre_zero_commission_small_lot_share)
        s = np.where(
            dates >= pd.Timestamp("2019-10-02"), p.post_2020_peak_small_lot_share, s
        )
        s = np.where(
            dates >= pd.Timestamp("2023-01-01"), p.post_2023_plateau_small_lot_share, s
        )
        return s

    def get_equity_ohlcv(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        See `MarketDataAdapter.get_equity_ohlcv` for interface. Generates synthetic OHLCV data based on the latent flow and regime signal strength.
        """
        dates = pd.bdate_range(start, end)
        strength = self._regime_signal_strength(dates)
        rows = []
        for t in tickers:
            flow = self._latent_flow(t, dates)
            idio_rng = np.random.default_rng(self._ticker_seed(t) + 1)
            # Dealer-hedging mechanism: TODAY's observed flow (used to build EOV/SLCS
            # end-of-day) predicts TOMORROW's return -- i.e. daily_ret[i] depends on
            # flow[i-1], so that fwd_ret_panel.loc[t] = close(t+1)/close(t)-1 (built via
            # pct_change().shift(-1) in the notebook) is genuinely predicted by the
            # EOV/SLCS signal computed from flow[t]. This is the correct point-in-time
            # tradeable structure: observe flow at t's close -> trade -> realize t+1's return.
            flow_lagged = np.concatenate([[0.0], flow[:-1]])
            daily_ret = strength * 0.01 * flow_lagged + idio_rng.normal(
                0.0003, 0.02, size=len(dates)
            )
            px = 100 * np.exp(np.cumsum(daily_ret))
            vol = np.abs(
                5e7 + 3e7 * np.abs(flow) + idio_rng.normal(0, 5e6, size=len(dates))
            )
            df = pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "open": px * (1 - idio_rng.uniform(0, 0.005, len(dates))),
                    "high": px * (1 + idio_rng.uniform(0, 0.01, len(dates))),
                    "low": px * (1 - idio_rng.uniform(0, 0.01, len(dates))),
                    "close": px,
                    "volume": vol,
                }
            )
            rows.append(df)
        return pd.concat(rows, ignore_index=True)

    def get_option_flow_proxy(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        See `MarketDataAdapter.get_option_flow_proxy` for interface. Generates synthetic option flow data based on the latent flow, regime signal strength, and small-lot share.
        """
        dates = pd.bdate_range(start, end)
        strength = self._regime_signal_strength(dates)
        small_lot_share = self._regime_small_lot_share(dates)
        rows = []
        for t in tickers:
            flow = self._latent_flow(t, dates)
            idio_rng = np.random.default_rng(self._ticker_seed(t) + 2)
            call_vol = np.abs(50 + 20 * flow + idio_rng.normal(0, 5, size=len(dates)))
            call_oi = np.abs(300 + idio_rng.normal(0, 30, size=len(dates)))
            put_vol = call_vol * 0.55 + idio_rng.normal(0, 3, size=len(dates))
            put_oi = call_oi * 0.7
            small_call = call_vol * (
                small_lot_share + 0.05 * idio_rng.normal(size=len(dates))
            )
            small_put = put_vol * (
                small_lot_share * 0.65 + 0.05 * idio_rng.normal(size=len(dates))
            )
            df = pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "call_volume": call_vol,
                    "call_oi": call_oi,
                    "small_lot_call_volume": np.clip(small_call, 0, call_vol),
                    "put_volume": np.abs(put_vol),
                    "put_oi": np.abs(put_oi),
                    "small_lot_put_volume": np.clip(small_put, 0, np.abs(put_vol)),
                    "_true_flow": flow,
                    "_signal_strength": strength,
                }
            )
            rows.append(df)
        return pd.concat(rows, ignore_index=True)


def get_default_adapter(prefer_live: bool = True) -> "MarketDataAdapter":
    """
    Factory for the **equity OHLCV** leg. Tries to use `YFinanceAdapter` if live network access is available; falls back to `SyntheticDataAdapter` otherwise. Logs which adapter is active.

    NOTE: This factory selects an adapter for **equity OHLCV**. Free `YFinanceAdapter` deliberately
          does NOT implement historical `get_option_flow_proxy` (see class docstring) -- use
          `get_default_flow_adapter()` for the options-flow leg instead of calling
          `get_option_flow_proxy` on whatever this function returns.

    Args:
        prefer_live (bool): If True, attempt to use YFinanceAdapter; if False or if network access
                            fails, use SyntheticDataAdapter.

    Returns:
        MarketDataAdapter: An instance of either YFinanceAdapter or SyntheticDataAdapter, depending
                           on availability and preference.
    """
    if prefer_live:
        try:
            adapter = YFinanceAdapter()
            probe = adapter.get_equity_ohlcv(["AAPL"], "2024-01-01", "2024-01-10")
            if probe is not None and not probe.empty:
                return adapter
        except Exception:
            pass
    return SyntheticDataAdapter()


def get_default_flow_adapter(
    seed: int = 42, params: "SyntheticRegimeParams | None" = None
) -> "MarketDataAdapter":
    """
    Factory for the **options-flow** leg (call/put volume, OI, small-lot breakdown). Free `YFinanceAdapter` has no historical single-name options OI/volume-by-lot, and `OCCPublicDataAdapter` is an unimplemented production stub, so this always returns the calibrated `SyntheticDataAdapter` today. In production, replace the body of this function with a licensed vendor adapter (OptionMetrics IvyDB, CBOE DataShop, Bloomberg OMON) or a completed `OCCPublicDataAdapter` -- no other code needs to change since callers only depend on the `MarketDataAdapter` protocol.

    Args:
        seed (int): Random seed for reproducibility.
        params (SyntheticRegimeParams | None): Optional calibration parameters for the synthetic
                                               regime generator. If None, defaults are used.

    Returns:
        MarketDataAdapter: An instance of SyntheticDataAdapter configured with the given seed and
                           parameters.
    """
    return SyntheticDataAdapter(seed=seed, params=params)
