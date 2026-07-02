# Retail Flow-Induced Gamma Dislocation (RFGD)

Systematic single-stock strategy operationalizing Barclays' *"Impact of Retail Options Trading"* (14 Sep 2020) research: retail-driven, short-dated, call-skewed option flow forces dealer delta-hedging that is large enough (~30-40% of underlying volume) to move stocks and distort the vol surface (flat skew, muted VRP on high-flow names).

See `RETAIL_GAMMA_STRATEGY.md` for the full research write-up (mechanism, academic grounding, signal research, backtest methodology, portfolio construction, sizing, live monitoring).

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest tests/ -q          # 46 tests, 98% coverage
```

## Structure
- `src/retail_gamma/signals.py` — EOV, SLCS, VRP/skew/bipower-variation signal construction
- `src/retail_gamma/backtest.py` — point-in-time backtest engine, costs, borrow filter
- `src/retail_gamma/portfolio.py` — vol targeting, fractional-Kelly sizing, drawdown governor, gamma-squeeze circuit breaker
- `src/retail_gamma/monitoring.py` — rolling IC/IR, signal-decay alerting, KPI artifact + Slack hook
- `tests/test_pipeline.py` — centralized pytest suite covering every code path
- `.github/workflows/ci.yml` — nightly CI: tests + KPI/signal-decay artifact generation (+ optional Slack)
- `notebooks/RFGD_Research.ipynb` — full research-to-deployment walkthrough
- `tex/RFGD_dissertation.tex` (+ compiled `.pdf`) — PhD-dissertation-style formal writeup

## Disclosures
Research/education artifact replicating a sell-side research idea; not investment advice. See §8 of the markdown write-up for risk disclosures.
