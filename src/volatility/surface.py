"""Vol surface: cubic spline in strike, linear in time"""

import numpy as np
from scipy.interpolate import CubicSpline


class _StrikeSmile:
    # single-maturity smile; cubic in the quoted range, linear extrap outside
    def __init__(self, strikes, vols):
        self._k_min = float(strikes[0])
        self._k_max = float(strikes[-1])
        self._v_min = float(vols[0])
        self._v_max = float(vols[-1])
        self._spline = CubicSpline(strikes, vols, extrapolate=False)
        # boundary slopes for linear extrapolation
        self._slope_low = float(self._spline(self._k_min, 1))
        self._slope_high = float(self._spline(self._k_max, 1))

    def __call__(self, strike):
        if strike < self._k_min:
            v = self._v_min + self._slope_low * (strike - self._k_min)
        elif strike > self._k_max:
            v = self._v_max + self._slope_high * (strike - self._k_max)
        else:
            v = float(self._spline(strike))
        return max(v, 1e-4)


class VolSurface:
    def __init__(self, df_date):
        # df_date: rows of the surface for a single observation date
        self._smiles = {}
        self._tenors = []

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
            if t in self._smiles:
                # two maturities round to the same year-fraction; keep the first
                continue
            self._smiles[t] = _StrikeSmile(strikes, vols)
            self._tenors.append(t)

        self._tenors = sorted(self._tenors)

    @property
    def is_empty(self):
        return len(self._tenors) == 0

    def get_vol(self, strike, t):
        if not self._tenors:
            raise ValueError("Empty vol surface.")

        tenors = np.array(self._tenors)
        vols_at_strike = np.array([self._smiles[t_](strike) for t_ in tenors])

        if tenors.size == 1:
            return float(vols_at_strike[0])

        # linear in time, manual extrapolation on the edges
        if t <= tenors[0]:
            slope = (vols_at_strike[1] - vols_at_strike[0]) / (tenors[1] - tenors[0])
            return float(vols_at_strike[0] + slope * (t - tenors[0]))
        if t >= tenors[-1]:
            slope = (vols_at_strike[-1] - vols_at_strike[-2]) / (tenors[-1] - tenors[-2])
            return float(vols_at_strike[-1] + slope * (t - tenors[-1]))
        return float(np.interp(t, tenors, vols_at_strike))

