"""Volatility surface construction and interpolation

Conventions:
    - Strike dimension: cubic-spline interpolation, linear extrapolation
    - Time dimension: linear interpolation and extrapolation across quoted tenors
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


class _StrikeSmile:
    """Single-maturity smile: cubic-spline interpolation, linear extrapolation"""

    def __init__(self, strikes: np.ndarray, vols: np.ndarray):
        self._k_min = float(strikes[0])
        self._k_max = float(strikes[-1])
        self._v_min = float(vols[0])
        self._v_max = float(vols[-1])
        # natural cubic spline for interior interpolation (no extrapolation delegated to spline)
        self._spline = CubicSpline(strikes, vols, extrapolate=False)
        # boundary first-derivative slopes for linear extrapolation beyond the quoted strike range
        self._slope_low = float(self._spline(self._k_min, 1))
        self._slope_high = float(self._spline(self._k_max, 1))

    def __call__(self, strike: float) -> float:
        if strike < self._k_min:
            v = self._v_min + self._slope_low * (strike - self._k_min)
        elif strike > self._k_max:
            v = self._v_max + self._slope_high * (strike - self._k_max)
        else:
            v = float(self._spline(strike))
        # implied vol is floored at 1e-4 to prevent non-positive values after linear extrapolation
        return max(v, 1e-4)


class VolSurface:
    """
    Volatility surface for a single observation date

    Strike interpolation: cubic spline
    Strike extrapolation: linear extension via boundary derivative
    Time interpolation and extrapolation: linear
    """

    def __init__(self, df_date: pd.DataFrame):
        """
        df_date: subset of the vol surface DataFrame for a single DATE,
                 with columns [DATE, MATURITY, MID_STRIKE, MID_PRICE]
        """
        self._smiles: dict[float, _StrikeSmile] = {}
        self._tenors: list[float] = []

        ref_date = df_date["DATE"].iloc[0]
        for maturity, grp in df_date.groupby("MATURITY"):
            t = (maturity - ref_date).days / 365.25
            if t <= 0:
                continue
            grp = grp.sort_values("MID_STRIKE").dropna(subset=["MID_STRIKE", "MID_PRICE"])
            grp = grp.drop_duplicates(subset="MID_STRIKE")
            if len(grp) < 3:
                continue
            strikes = grp["MID_STRIKE"].values.astype(float)
            vols = grp["MID_PRICE"].values.astype(float)
            # skip maturities that map to the same year-fraction to avoid duplicate tenor keys
            if t in self._smiles:
                continue
            self._smiles[t] = _StrikeSmile(strikes, vols)
            self._tenors.append(t)

        self._tenors = sorted(self._tenors)

    @property
    def is_empty(self) -> bool:
        return len(self._tenors) == 0

    def get_vol(self, strike: float, t: float) -> float:
        """Return implied volatility for a given strike and time to maturity in years"""
        if not self._tenors:
            raise ValueError("Empty vol surface.")

        tenors = np.array(self._tenors)
        vols_at_strike = np.array([self._smiles[t_](strike) for t_ in tenors])

        if tenors.size == 1:
            return float(vols_at_strike[0])

        # linear interpolation in the time dimension using np.interp (no object allocation);
        # handle linear extrapolation beyond the quoted tenor range manually
        if t <= tenors[0]:
            slope = (vols_at_strike[1] - vols_at_strike[0]) / (tenors[1] - tenors[0])
            return float(vols_at_strike[0] + slope * (t - tenors[0]))
        if t >= tenors[-1]:
            slope = (vols_at_strike[-1] - vols_at_strike[-2]) / (tenors[-1] - tenors[-2])
            return float(vols_at_strike[-1] + slope * (t - tenors[-1]))
        return float(np.interp(t, tenors, vols_at_strike))

