"""Unit tests for performance analytics and Greeks attribution."""

import numpy as np
import pandas as pd

from src.analytics.performance import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    var,
    cvar,
    compute_metrics,
)
from src.analytics.greeks_pnl import attribute_pnl


def test_sharpe_zero_for_constant():
    r = pd.Series([5.0] * 100)
    assert np.isnan(sharpe_ratio(r))  # zero variance → undefined


def test_sortino_only_penalises_downside():
    r = pd.Series([1.0, 1.0, -1.0, 1.0, -2.0, 1.0])
    assert np.isfinite(sortino_ratio(r))


def test_max_drawdown_is_negative():
    cum = pd.Series([0, 10, 5, 20, 8]).cumsum()
    assert max_drawdown(cum) <= 0


def test_var_cvar_ordering():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0, 1, 10_000))
    v = var(r, 0.95)
    c = cvar(r, 0.95)
    assert c <= v  # expected shortfall is at least as extreme as VaR


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
