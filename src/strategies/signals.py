"""Signal functions for conditioning strategy entries on market state

Each signal function maps the spot price history up to a given date
onto a boolean (trade / skip). Consumed by Strategy.signal and
composable with any strategy implementation
"""

from __future__ import annotations

import pandas as pd


def momentum_signal(history: pd.Series, lookback: int = 20) -> bool:
    """
    Trend filter: trade only when the latest spot exceeds its
    rolling mean over the lookback window (positive momentum)
    """
    if len(history) < lookback:
        return True
    sma = history.iloc[-lookback:].mean()
    return bool(history.iloc[-1] >= sma)


def realised_vol_signal(history: pd.Series, lookback: int = 20, threshold: float = 0.0) -> bool:
    """
    Volatility filter: trade only when the annualised realised volatility
    over the lookback window exceeds the threshold (decimal)
    Suited to long-gamma strategies that require sufficient spot movement
    """
    if len(history) < lookback + 1:
        return True
    rets = history.iloc[-(lookback + 1):].pct_change().dropna()
    realised = float(rets.std() * (252 ** 0.5))
    return realised >= threshold
