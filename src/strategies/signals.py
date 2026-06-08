"""Basic signal components to trigger strategy entries/exits.

These are intentionally lightweight: a signal maps the spot history up to a
given date onto a boolean (trade / don't trade). They are consumed by
``Strategy.signal`` and can be combined with any strategy.
"""

from __future__ import annotations

import pandas as pd


def momentum_signal(history: pd.Series, lookback: int = 20) -> bool:
    """
    Simple trend filter: trade only when the latest spot is above its
    rolling mean over ``lookback`` business days (positive momentum).
    """
    if len(history) < lookback:
        return True
    sma = history.iloc[-lookback:].mean()
    return bool(history.iloc[-1] >= sma)


def realised_vol_signal(history: pd.Series, lookback: int = 20, threshold: float = 0.0) -> bool:
    """
    Volatility filter: trade only when annualised realised vol over the
    lookback window exceeds ``threshold`` (decimal). Useful for long-gamma
    strategies that need movement to pay off.
    """
    if len(history) < lookback + 1:
        return True
    rets = history.iloc[-(lookback + 1):].pct_change().dropna()
    realised = float(rets.std() * (252 ** 0.5))
    return realised >= threshold
