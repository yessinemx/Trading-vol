"""Garman-Kohlhagen pricing model for FX options."""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class GKResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho_d: float
    rho_f: float


def garman_kohlhagen(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    option_type: str = "call",
) -> GKResult:
    """
    Price an FX option using the Garman-Kohlhagen model.

    Parameters
    ----------
    S       : spot price (domestic per foreign)
    K       : strike
    T       : time to maturity in years
    r_d     : domestic risk-free rate (continuous, decimal)
    r_f     : foreign risk-free rate (continuous, decimal)
    sigma   : implied volatility (decimal)
    option_type : 'call' or 'put'

    Returns
    -------
    GKResult with price and all first-order Greeks.
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return GKResult(intrinsic, float(S > K) if option_type == "call" else float(S < K), 0, 0, 0, 0, 0)

    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_neg_d1 = norm.cdf(-d1)
    n_neg_d2 = norm.cdf(-d2)
    npd1 = norm.pdf(d1)

    disc_d = np.exp(-r_d * T)
    disc_f = np.exp(-r_f * T)

    if option_type == "call":
        price = S * disc_f * nd1 - K * disc_d * nd2
        delta = disc_f * nd1
        rho_d = K * T * disc_d * nd2
        rho_f = -S * T * disc_f * nd1
    else:
        price = K * disc_d * n_neg_d2 - S * disc_f * n_neg_d1
        delta = -disc_f * n_neg_d1
        rho_d = -K * T * disc_d * n_neg_d2
        rho_f = S * T * disc_f * n_neg_d1

    gamma = disc_f * npd1 / (S * sigma * np.sqrt(T))
    vega = S * disc_f * npd1 * np.sqrt(T)
    theta = (
        -S * disc_f * npd1 * sigma / (2 * np.sqrt(T))
        - r_d * K * disc_d * (nd2 if option_type == "call" else -n_neg_d2)
        + r_f * S * disc_f * (nd1 if option_type == "call" else -n_neg_d1)
    )

    return GKResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega / 100,  # per 1% vol move
        theta=theta / 365,  # per calendar day
        rho_d=rho_d / 100,
        rho_f=rho_f / 100,
    )
