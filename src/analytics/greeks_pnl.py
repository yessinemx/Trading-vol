"""Greek-based P&L attribution: delta, gamma, vega and theta components"""

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
    Second-order Taylor decomposition of option book P&L

    Approximation: dP ~ Delta*dS + (1/2)*Gamma*dS^2 + vega*dSigma + theta*dt + residual

    Unit conventions (matching the pricing engine):
    ----------
    delta, gamma: notional-weighted, per unit spot move
    vega: notional-weighted, per 1% vol move
    theta: notional-weighted, per calendar day
    dS: spot move in price units
    dSigma: vol move in decimal 
    dt: time elapsed in years
    actual_pnl: realised option P&L used to compute the residual
    """
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS**2
    vega_pnl = vega * (dSigma * 100.0)   # vega per 1% move; convert decimal vol move
    theta_pnl = theta * (dt * 365.0)     # theta per calendar day; convert year fraction to days

    attributed = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = (actual_pnl - attributed) if actual_pnl is not None else 0.0

    return GreeksPnL(delta_pnl, gamma_pnl, vega_pnl, theta_pnl, residual)


def attribution_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Cumulative Greek P&L contributions over time"""
    cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "residual_pnl"]
    present = [c for c in cols if c in records.columns]
    return records[present].cumsum()

