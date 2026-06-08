"""Linear interpolation and extrapolation for forward and interest rate term structures"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class CurveInterpolator:
    """
    Linear interpolation and extrapolation for rate and forward curves

    Conventions: forward and interest-rate curves use linear interpolation
    and linear extrapolation beyond the quoted tenors

    Inputs: maturities as year fractions; rates as decimals
    """

    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        tenors = np.asarray(tenors, dtype=float)
        rates = np.asarray(rates, dtype=float)

        # drop non-finite points and collapse duplicate tenors by averaging;
        # duplicate tenors produce an infinite extrapolation slope in interp1d
        mask = np.isfinite(tenors) & np.isfinite(rates)
        tenors, rates = tenors[mask], rates[mask]
        if tenors.size:
            uniq, inv = np.unique(tenors, return_inverse=True)
            if uniq.size != tenors.size:
                agg = np.zeros_like(uniq)
                np.add.at(agg, inv, rates)
                counts = np.bincount(inv)
                rates = agg / counts
                tenors = uniq
            else:
                order = np.argsort(tenors)
                tenors, rates = tenors[order], rates[order]

        self._tenors = tenors
        self._rates = rates

        if self._tenors.size == 0:
            raise ValueError("Cannot build a curve with no points.")
        if self._tenors.size == 1:
            # single-point curve: return a flat constant (no slope to extrapolate)
            self._fn = lambda t: np.full_like(np.asarray(t, dtype=float), self._rates[0])
        else:
            self._fn = interp1d(
                self._tenors,
                self._rates,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        # linear extrapolation beyond the quoted tenors
        # duplicate tenors are collapsed in __init__, so the slope is always finite;
        # fall back to the last quoted rate only if a degenerate query returns non-finite
        result = self._fn(t)
        result = np.where(np.isfinite(result), result, self._rates[-1])
        return float(result) if np.isscalar(t) else np.asarray(result)


    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        date: pd.Timestamp,
        value_col: str = "MID_PRICE",
    ) -> "CurveInterpolator":
        """Build a CurveInterpolator for a single date from a stacked DataFrame"""
        subset = df[df["DATE"] == date].copy()
        subset["t"] = (subset["MATURITY"] - date).dt.days / 365.25
        subset = subset.dropna(subset=["t", value_col])
        subset = subset.sort_values("t")
        return cls(subset["t"].values, subset[value_col].values)
