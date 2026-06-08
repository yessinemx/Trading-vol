"""Performance metrics for options strategies"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / downside


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    rolling_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - rolling_max
    return float(drawdown.min())


def var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level"""
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional Value-at-Risk (Expected Shortfall)"""
    v = var(returns, confidence)
    return float(returns[returns <= v].mean())


def compute_metrics(daily_pnl: pd.Series, rf: float = 0.0) -> dict:
    cumulative = daily_pnl.cumsum()
    return {
        "total_pnl": cumulative.iloc[-1],
        "sharpe": sharpe_ratio(daily_pnl, rf),
        "sortino": sortino_ratio(daily_pnl, rf),
        "max_drawdown": max_drawdown(cumulative),
        "var_95": var(daily_pnl),
        "cvar_95": cvar(daily_pnl),
        "win_rate": (daily_pnl > 0).mean(),
        "avg_daily_pnl": daily_pnl.mean(),
        "volatility_ann": daily_pnl.std() * np.sqrt(252),
    }
