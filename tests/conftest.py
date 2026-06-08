"""Shared fixtures: small synthetic market data for offline tests."""

import numpy as np
import pandas as pd
import pytest

from src.curves.interpolator import CurveInterpolator
from src.volatility.surface import VolSurface

DATE = pd.Timestamp("2024-01-02")


@pytest.fixture
def small_surface():
    # 2 maturities, 3 strikes, mild smile
    rows = []
    for mat_days, base_vol in [(30, 0.10), (90, 0.12)]:
        maturity = DATE + pd.Timedelta(days=mat_days)
        for strike, bump in [(9.0, 0.03), (11.0, 0.00), (13.0, 0.03)]:
            rows.append({
                "DATE": DATE,
                "MATURITY": maturity,
                "MID_STRIKE": strike,
                "MID_PRICE": base_vol + bump,
            })
    return VolSurface(pd.DataFrame(rows))


@pytest.fixture
def flat_curve():
    tenors = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0])
    rates = np.array([0.030, 0.032, 0.034, 0.036, 0.038, 0.040])
    return CurveInterpolator(tenors, rates)


@pytest.fixture
def spot():
    return 11.50


@pytest.fixture
def synthetic_spot_series():
    # 250 business days of GBM-ish spot around 11.50
    rng = np.random.default_rng(42)
    n = 250
    dates = pd.bdate_range(start="2023-01-03", periods=n)
    log_returns = rng.normal(0, 0.005, size=n)
    prices = 11.50 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=dates, name="MID_PRICE")
