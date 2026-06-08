"""Greeks-based P&L attribution: Delta, Gamma, Vega, Theta."""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class GreeksPnL:
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    residual: float

    def total(self) -> float:
        return self.delta_pnl + self.gamma_pnl + self.vega_pnl + self.theta_pnl + self.residual


def attribute_pnl(
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    dS: float,
    dSigma: float,
    dt: float = 1 / 252,
    actual_pnl: float | None = None,
) -> GreeksPnL:
    """
    Second-order Taylor P&L decomposition of an option book.

        dP ≈ Δ·dS + ½·Γ·dS² + ν·dσ + Θ·dt + residual

    Unit conventions (matching the pricing engine):
    ----------
    delta, gamma : notional-weighted, per 1.0 spot unit.
    vega         : notional-weighted, expressed *per 1% vol move*.
    theta        : notional-weighted, expressed *per calendar day*.
    dS           : spot move (price units).
    dSigma       : vol move in decimal (e.g. 0.01 == 1 vol point).
    dt           : time elapsed in years.
    actual_pnl   : realized option P&L, used to back out the residual.
    """
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS**2
    vega_pnl = vega * (dSigma * 100.0)   # vega is per 1% move → scale decimal move
    theta_pnl = theta * (dt * 365.0)     # theta is per calendar day → scale years

    attributed = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = (actual_pnl - attributed) if actual_pnl is not None else 0.0

    return GreeksPnL(delta_pnl, gamma_pnl, vega_pnl, theta_pnl, residual)


def attribution_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Convenience: cumulative Greek P&L contributions over time."""
    cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "residual_pnl"]
    present = [c for c in cols if c in records.columns]
    return records[present].cumsum()

