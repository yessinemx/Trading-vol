"""Greek P&L attribution (delta/gamma/vega/theta)"""

from dataclasses import dataclass


@dataclass
class GreeksPnL:
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    residual: float

    def total(self):
        return self.delta_pnl + self.gamma_pnl + self.vega_pnl + self.theta_pnl + self.residual


def attribute_pnl(delta, gamma, vega, theta, dS, dSigma, dt=1 / 252, actual_pnl=None):
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS ** 2
    vega_pnl = vega * (dSigma * 100.0)
    theta_pnl = theta * (dt * 365.0)

    attributed = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = (actual_pnl - attributed) if actual_pnl is not None else 0.0
    return GreeksPnL(delta_pnl, gamma_pnl, vega_pnl, theta_pnl, residual)


def attribution_frame(records):
    cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "residual_pnl"]
    present = [c for c in cols if c in records.columns]
    return records[present].cumsum()

