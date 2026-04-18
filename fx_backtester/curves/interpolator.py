"""Linear interpolation/extrapolation for forward and interest rate curves."""

import numpy as np
import pandas as pd


class CurveInterpolator:
    """
    Linear interpolation and flat extrapolation for rate/forward curves.
    Input: maturities as year fractions, rates as decimals.
    """

    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        order = np.argsort(tenors)
        self._tenors = tenors[order]
        self._rates = rates[order]

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.interp(t, self._tenors, self._rates)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        date: pd.Timestamp,
        value_col: str = "MID_PRICE",
    ) -> "CurveInterpolator":
        """Build interpolator for a single date from a stacked DataFrame."""
        subset = df[df["DATE"] == date].copy()
        subset["t"] = (subset["MATURITY"] - date).dt.days / 365.25
        subset = subset.dropna(subset=["t", value_col])
        subset = subset.sort_values("t")
        return cls(subset["t"].values, subset[value_col].values)
