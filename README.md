# FX Option Backtesting Engine

A modular backtesting framework for Foreign Exchange options using the Garman-Kohlhagen model.

## Features

- **Garman-Kohlhagen pricing** with full Greeks (Delta, Gamma, Vega, Theta, Rho)
- **Volatility surface** interpolation: cubic spline in strike, linear in time
- **Rate & forward curve** interpolation: linear with flat extrapolation
- **Multi-leg strategies**: Straddle, Call Ratio Spread, Calendar Spread
- **Daily delta hedging** and roll scheduling
- **Performance analytics**: Sharpe, Sortino, Max Drawdown, VaR, CVaR
- **Greeks P&L attribution**: decompose daily P&L into risk factor contributions

## Project Structure

```
fx_backtester/
├── data/           # Market data loading (Parquet ingestion)
├── curves/         # Rate & forward curve interpolators
├── volatility/     # Vol surface (cubic spline)
├── pricing/        # Garman-Kohlhagen model
├── strategies/     # Strategy definitions (Straddle, RatioSpread, Calendar)
├── portfolio/      # Position and cash management
├── analytics/      # Performance metrics & Greeks P&L
└── engine.py       # Main backtester orchestrator
notebooks/          # Jupyter notebooks
tests/              # Unit tests
data/               # Parquet market data (not committed)
```

## Installation

```bash
poetry install
```

## Quick Start

```python
from fx_backtester.engine import Backtester
from fx_backtester.strategies.straddle import Straddle

bt = Backtester("data/")
bt.load_data()

strategy = Straddle(tenor_days=30, direction=1)
results = bt.run(strategy, start="2021-01-01", end="2023-12-31", notional=1_000_000)

print(results["metrics"])
```

## Market Data

Four Parquet datasets (place in `data/`):

| File | Description |
|---|---|
| `*spot*.parquet` | Daily spot FX rates |
| `*forwardcurve*.parquet` | Forward points by tenor |
| `*depocurve*.parquet` | Interest rates (domestic & foreign) |
| `*volatilitysurface*.parquet` | Implied vols by strike/tenor |

## Strategies

| Strategy | Description |
|---|---|
| `Straddle` | Long ATM call + put — pure volatility bet |
| `CallRatioSpread` | Long 1 ATM call, short 2 OTM calls |
| `CalendarSpread` | Long far-dated, short near-dated option |

## Architecture

See the system diagram in `docs/architecture.png` for the full component overview.

---

*Implements the Garman-Kohlhagen model: C = S·e^(−rf·T)·N(d1) − K·e^(−rd·T)·N(d2)*
