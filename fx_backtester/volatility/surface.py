"""Volatility surface interpolation: cubic spline in strike, linear in time."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


class VolSurface:
    """
    Builds a vol surface for a single date.
    Strike interpolation: cubic spline.
    Time interpolation: linear between adjacent maturities.
    Extrapolation: linear (flat tangent extension).
    """

    def __init__(self, df_date: pd.DataFrame):
        """
        df_date: subset of vol surface DataFrame for one DATE,
                 with columns [MATURITY, MID_STRIKE, MID_PRICE].
        """
        self._splines: dict[float, CubicSpline] = {}
        self._tenors: list[float] = []

        ref_date = df_date["DATE"].iloc[0]
        for maturity, grp in df_date.groupby("MATURITY"):
            t = (maturity - ref_date).days / 365.25
            grp = grp.sort_values("MID_STRIKE").dropna(subset=["MID_STRIKE", "MID_PRICE"])
            if len(grp) < 3:
                continue
            strikes = grp["MID_STRIKE"].values.astype(float)
            vols = grp["MID_PRICE"].values.astype(float)
            # extrapolate=True uses linear extension beyond boundaries
            self._splines[t] = CubicSpline(strikes, vols, extrapolate=True)
            self._tenors.append(t)

        self._tenors = sorted(self._tenors)

    def get_vol(self, strike: float, t: float) -> float:
        """Return implied vol for a given strike and time-to-maturity (years)."""
        if not self._tenors:
            raise ValueError("Empty vol surface.")

        tenors = np.array(self._tenors)
        vols_at_strike = np.array([float(self._splines[t_](strike)) for t_ in tenors])
        return float(np.interp(t, tenors, vols_at_strike))
