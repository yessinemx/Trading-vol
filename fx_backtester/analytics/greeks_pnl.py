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
    dt: float = 1 / 365,
    actual_pnl: float | None = None,
) -> GreeksPnL:
    """
    First/second order Greeks P&L decomposition.

    Parameters
    ----------
    delta, gamma, vega, theta : Greeks at start of period
    dS      : spot move
    dSigma  : vol move (decimal)
    dt      : time elapsed in years
    actual_pnl : realized P&L to compute residual
    """
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS**2
    vega_pnl = vega * dSigma * 100   # vega is per 1% move
    theta_pnl = theta * dt * 365      # theta is per calendar day

    attributed = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = (actual_pnl - attributed) if actual_pnl is not None else 0.0

    return GreeksPnL(delta_pnl, gamma_pnl, vega_pnl, theta_pnl, residual)
