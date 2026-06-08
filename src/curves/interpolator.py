"""Linear interp/extrap for rate and forward curves"""

import numpy as np
from scipy.interpolate import interp1d


class CurveInterpolator:
    def __init__(self, tenors, rates):
        tenors = np.asarray(tenors, dtype=float)
        rates = np.asarray(rates, dtype=float)

        # drop NaNs and average duplicate tenors
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
            # single point -> flat curve
            self._fn = lambda t: np.full_like(np.asarray(t, dtype=float), self._rates[0])
        else:
            self._fn = interp1d(
                self._tenors,
                self._rates,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

    def __call__(self, t):
        result = self._fn(t)
        # if extrapolation gave NaN, fall back to last quote
        result = np.where(np.isfinite(result), result, self._rates[-1])
        return float(result) if np.isscalar(t) else np.asarray(result)

    @classmethod
    def from_dataframe(cls, df, date, value_col="MID_PRICE"):
        subset = df[df["DATE"] == date].copy()
        subset["t"] = (subset["MATURITY"] - date).dt.days / 365.25
        subset = subset.dropna(subset=["t", value_col])
        subset = subset.sort_values("t")
        return cls(subset["t"].values, subset[value_col].values)
