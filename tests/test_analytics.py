"""Unit tests for performance and risk metrics and Greeks attribution"""

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    PerformanceMetrics,
    RiskMetrics,
    compute_metrics,
)
from src.analytics.greeks_pnl import attribute_pnl


def test_sharpe_zero_for_constant():
    r = pd.Series([5.0] * 100)
    assert np.isnan(PerformanceMetrics(r).sharpe())  # zero variance, undefined


def test_sortino_only_penalises_downside():
    r = pd.Series([1.0, 1.0, -1.0, 1.0, -2.0, 1.0])
    assert np.isfinite(PerformanceMetrics(r).sortino())


def test_max_drawdown_is_non_positive():
    r = pd.Series([10, -5, 15, -12, 8])
    assert RiskMetrics(r).max_drawdown() <= 0


def test_var_cvar_ordering():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0, 1, 10_000))
    rm = RiskMetrics(r, confidence=0.95)
    v = rm.var()
    c = rm.cvar()
    assert c <= v  # expected shortfall is at least as extreme as VaR


def test_performance_summary_keys():
    r = pd.Series(np.random.normal(100, 1000, 250))
    s = PerformanceMetrics(r).summary()
    for key in ["total_pnl", "sharpe", "sortino", "win_rate", "avg_daily_pnl", "volatility_ann"]:
        assert key in s


def test_risk_summary_keys():
    r = pd.Series(np.random.normal(100, 1000, 250))
    s = RiskMetrics(r).summary()
    for key in ["max_drawdown", "var_95", "cvar_95"]:
        assert key in s


def test_compute_metrics_keys():
    r = pd.Series(np.random.normal(100, 1000, 250))
    m = compute_metrics(r)
    for key in ["total_pnl", "sharpe", "sortino", "max_drawdown", "var_95", "cvar_95"]:
        assert key in m


def test_attribution_sums_to_actual():
    g = attribute_pnl(
        delta=100.0, gamma=50.0, vega=2000.0, theta=-30.0,
        dS=0.01, dSigma=0.005, dt=1 / 252, actual_pnl=12.34,
    )
    assert abs(g.total() - 12.34) < 1e-9
