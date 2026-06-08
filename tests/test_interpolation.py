"""Unit tests for curve and vol-surface interpolation."""

import numpy as np
import pandas as pd

from src.curves.interpolator import CurveInterpolator
from src.volatility.surface import VolSurface


def test_curve_linear_interpolation():
    curve = CurveInterpolator(np.array([0.25, 1.0]), np.array([0.02, 0.04]))
    # Midpoint in time → midpoint in rate.
    assert abs(curve(0.625) - 0.03) < 1e-12


def test_curve_linear_extrapolation():
    curve = CurveInterpolator(np.array([1.0, 2.0]), np.array([0.02, 0.03]))
    # Slope is 0.01/yr → extrapolate to t=3 gives 0.04.
    assert abs(curve(3.0) - 0.04) < 1e-12


def test_curve_handles_duplicate_tenors():
    curve = CurveInterpolator(
        np.array([0.0082, 0.0082, 1.0]), np.array([0.039, 0.040, 0.036])
    )
    val = curve(0.5)
    assert np.isfinite(val)


def _toy_surface() -> pd.DataFrame:
    date = pd.Timestamp("2024-01-02")
    rows = []
    for mat_days, base in [(30, 0.10), (90, 0.12)]:
        maturity = date + pd.Timedelta(days=mat_days)
        for k, bump in [(0.9, 0.03), (1.0, 0.0), (1.1, 0.03)]:
            rows.append(
                {"DATE": date, "MATURITY": maturity, "MID_STRIKE": k, "MID_PRICE": base + bump}
            )
    return pd.DataFrame(rows)


def test_vol_surface_atm_interpolation():
    vs = VolSurface(_toy_surface())
    # ATM (k=1.0) at the 30d tenor (~0.082y) should be the quoted 0.10.
    v = vs.get_vol(1.0, 30 / 365.25)
    assert abs(v - 0.10) < 5e-3


def test_vol_surface_positive_extrapolation():
    vs = VolSurface(_toy_surface())
    # Far OTM strike → linear extrapolation must stay positive.
    v = vs.get_vol(2.0, 30 / 365.25)
    assert v > 0
