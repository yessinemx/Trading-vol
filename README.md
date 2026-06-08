# FX Option Backtesting Engine

A modular research-grade backtester for European FX options on **EUR/NOK**,
built around the Garman–Kohlhagen two-rate Black–Scholes model with daily
delta hedging, Greeks P&L attribution and a signal/sizing framework.

<p align="left">
  <img alt="Python"  src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Poetry"  src="https://img.shields.io/badge/build-poetry-60a5fa">
  <img alt="Tests"   src="https://img.shields.io/badge/tests-19%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---


- [FX Option Backtesting Engine](#fx-option-backtesting-engine)
  - [Highlights](#highlights)
  - [Architecture](#architecture)
  - [Project layout](#project-layout)
  - [Installation](#installation)
    - [With Poetry](#with-poetry)
  - [Market data](#market-data)
  - [Quick start](#quick-start)
  - [Strategies, signals and sizing](#strategies-signals-and-sizing)
    - [Strategies](#strategies)
    - [Signals](#signals)
    - [Sizing](#sizing)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Limitations and roadmap](#limitations-and-roadmap)
  - [Author](#author)

---

## Highlights

- **Pricing engine.** Closed-form Garman–Kohlhagen with full first-order Greeks
  (Δ, Γ, ν, Θ, ρ_d, ρ_f), available in both a scalar form and a NumPy-vectorised
  variant (`scipy.special.ndtr`) for daily mark-to-market over the full book.
- **Volatility surface.** Cubic spline in strike, linear interpolation in time,
  with **linear extrapolation** beyond the quoted boundaries.
- **Rate and forward curves.** Linear interpolation and extrapolation with
  deduplicated tenors.
- **Strategies.** Long/short *Straddle*, *Call Ratio Spread*, *Calendar Spread*,
  all derived from a common `Strategy` base with weekly / monthly / quarterly
  roll scheduling.
- **Daily delta hedging.** A dedicated spot hedge book rebalances the portfolio
  delta on every business day; hedge P&L and option P&L are reported separately.
- **Greeks P&L attribution.** Second-order Taylor decomposition (Δ, Γ, ν, Θ)
  computed *position-by-position over continuing legs* so roll premia don't
  pollute the breakdown.
- **Performance analytics.** Sharpe, Sortino, max drawdown, historical VaR/CVaR
  at 95 %, win rate, annualised volatility.
- **Signal & sizing layer.** Pluggable entry filters (momentum SMA, realised
  volatility threshold) and per-strategy sizing hooks.
- **End-to-end notebook.** [`notebooks/backtester.ipynb`](notebooks/backtester.ipynb)
  walks through loading, pricing, strategy comparison, signal gating and Greek
  attribution on EUR/NOK 2021–2024 data.

---

## Architecture

![Architecture](docs/architecture.png)

The data layer ingests stacked Parquet files and pivots them once per load
into date-keyed views. Pre-built `VolSurface` objects are cached per date,
so the daily mark-to-market loop is a pure vectorised pass through the
pricing engine. Roll schedulers, contract specs and sizing produce the
order flow; the portfolio holds option legs alongside a spot hedge book;
the analytics layer turns the daily series into metrics and a Greek
decomposition.

See [`docs/USER_GUIDE.pdf`](docs/USER_GUIDE.pdf) for the full methodology
write-up (pricing formulas, Greek closed forms, metric definitions and the
quantitative deep dive).

---

## Project layout

```
src/
├── data/           # Parquet ingestion + stacked → date-keyed pivot
├── curves/         # Rate & forward curve interpolators (linear)
├── volatility/     # Vol surface (cubic spline in strike + linear extrapolation)
├── pricing/        # Garman–Kohlhagen pricer (scalar + vectorised) and Greeks
├── strategies/     # Straddle, CallRatioSpread, CalendarSpread, signals, sizing
├── portfolio/      # OptionPosition, Portfolio with spot delta-hedge book
├── analytics/      # Performance metrics & Greeks P&L attribution
└── engine.py       # Backtester orchestrator (daily loop, hedge, attribution)
notebooks/
└── backtester.ipynb   # End-to-end walkthrough on EUR/NOK
tests/                 # Unit tests: pricing parity, interpolation, analytics
data/                  # Parquet market data (gitignored)
outputs/               # Generated CSV / PNG artefacts
docs/
├── architecture.png   # Component diagram
├── USER_GUIDE.tex     # LaTeX source of the user guide
└── USER_GUIDE.pdf     # Compiled methodology + deep-dive 
```

---

## Installation

Requires **Python ≥ 3.11**.

### With Poetry 

```bash
poetry install
poetry shell
```

## Market data

Four Parquet datasets must be placed in the `data/` directory.

| File pattern                  | Content                                        |
| ----------------------------- | ---------------------------------------------- |
| `*spot*.parquet`              | Daily spot FX rates (NOK per EUR)              |
| `*forwardcurve*.parquet`      | Forward points by tenor (pips, scale 1/10 000) |
| `*depocurve*.parquet`         | Domestic (NOK) and foreign (EUR) deposit rates |
| `*volatilitysurface*.parquet` | Implied vols by strike (or delta) and tenor    |

Convention: prices, P&L and Greeks are expressed in the **domestic currency
(NOK)** consistently with FX market practice.

---

## Quick start

```python
from src.engine import Backtester
from src.strategies.straddle import Straddle

bt = Backtester("data")
bt.load_data()

strategy = Straddle(tenor_days=30, direction=+1)   # +1 long, -1 short
results  = bt.run(
    strategy,
    start="2021-01-01",
    end="2024-12-31",
    notional=1_000_000,
    roll_freq="W-FRI",
)

print(results["metrics"])             
results["daily_pnl"].cumsum().plot()
```

The `results` dictionary exposes:

| Key           | Type           | Content                                           |
| ------------- | -------------- | ------------------------------------------------- |
| `metrics`     | `dict`         | Sharpe, Sortino, VaR_95, CVaR_95, MaxDD, win rate |
| `daily_pnl`   | `pd.Series`    | Total daily P&L in NOK (option + hedge)           |
| `components`  | `pd.DataFrame` | Option P&L and hedge P&L, split                   |
| `greeks`      | `pd.DataFrame` | Daily portfolio Δ / Γ / ν / Θ                     |
| `attribution` | `pd.DataFrame` | Δ-, Γ-, ν-, Θ-, residual-P&L per day              |
| `portfolio`   | `Portfolio`    | Final portfolio object (positions + hedge book)   |

---

## Strategies, signals and sizing

### Strategies

| Class             | Construction                           | Description                                   |
| ----------------- | -------------------------------------- | --------------------------------------------- |
| `Straddle`        | `Straddle(tenor_days, direction=±1)`   | ATM call + put: pure realised-vol bet         |
| `CallRatioSpread` | `CallRatioSpread(tenor_days, otm_pct)` | Long 1 ATM call, short 2 OTM calls            |
| `CalendarSpread`  | `CalendarSpread(near_days, far_days)`  | Long far-dated, short near-dated, same strike |

### Signals

```python
from src.strategies.signals import momentum_signal, realised_vol_signal
from src.strategies.straddle import Straddle

gated = Straddle(
    tenor_days=30,
    direction=+1,
    signal_fn=lambda hist: momentum_signal(hist, lookback=20),
)
```

`momentum_signal(history, lookback)` — *trade* when spot ≥ its SMA over the
look-back window.
`realised_vol_signal(history, lookback, threshold)` — *trade* when the annualised
realised vol over the window exceeds `threshold`.

Filters compose by logical conjunction; on a skip day the strategy holds cash
and the daily loop just marks-to-market the surviving book.

### Sizing

Each strategy receives a `notional` and internally sizes legs by contract spec.
Custom per-strategy sizing can be wired in by overriding the `size_legs` hook
of `Strategy`.

---

## Testing

```bash
poetry run pytest      
```

Coverage includes:
- Put–call parity and Greek-sign invariants for the GK pricer
- Linear and cubic interpolation, plus boundary extrapolation behaviour
- Performance metric formulas (Sharpe, Sortino, MaxDD, VaR, CVaR)
- Synthetic fixtures (`tests/conftest.py`) for `VolSurface`, curves, spot

---

## Documentation

- [**User Guide (PDF)**](docs/USER_GUIDE.pdf) — academic write-up of the model,
  Greek closed forms, performance metric definitions and the quantitative deep
  dive on the long straddle (with and without the momentum filter).
- [**LaTeX source**](docs/USER_GUIDE.tex) — recompile with:
  ```bash
  pdflatex -interaction=nonstopmode -output-directory docs docs/USER_GUIDE.tex
  ```
- [**End-to-end notebook**](notebooks/backtester.ipynb) — every result of the
  user guide is reproducible from this notebook.

---

## Limitations and roadmap

**Modelling assumptions.** Mid-price execution (no bid/ask), discrete daily
hedging (gamma slippage between rebalances), Taylor attribution truncated at
second order (vanna and volga absorbed into the residual term).

**Planned extensions.**
1. Bid/ask spread and hedge transaction cost as a haircut on daily P&L
2. Vanna/volga adjustment for the smile
3. Configurable intraday rehedging frequency
4. Additional signal templates (macro regime, risk-on / risk-off filters)

---

## Author

**Yassine Mannai** — [yassine.mannai@dauphine.eu](mailto:yassine.mannai@dauphine.eu)
