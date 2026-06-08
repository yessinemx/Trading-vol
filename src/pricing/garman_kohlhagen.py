"""Garman-Kohlhagen FX option pricer (scalar + vectorised)"""

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
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


def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type="call"):
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return GKResult(intrinsic, float(S > K) if option_type == "call" else float(S < K), 0, 0, 0, 0, 0)

    sigma = max(float(sigma), 1e-6)
    if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(sigma) and S > 0 and K > 0):
        return GKResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
        vega=vega / 100,    # per 1% vol move
        theta=theta / 365,  # per calendar day
        rho_d=rho_d / 100,
        rho_f=rho_f / 100,
    )


def garman_kohlhagen_vec(S, K, T, r_d, r_f, sigma, is_call):

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r_d = np.asarray(r_d, dtype=float)
    r_f = np.asarray(r_f, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    is_call = np.asarray(is_call, dtype=bool)

    n = S.shape[0] if S.ndim else 1
    price = np.zeros(n, dtype=float)
    delta = np.zeros(n, dtype=float)
    gamma = np.zeros(n, dtype=float)
    vega = np.zeros(n, dtype=float)
    theta = np.zeros(n, dtype=float)

    # expired legs -> settle at intrinsic
    expired = T <= 0
    if expired.any():
        dS = S[expired] - K[expired]
        price[expired] = np.where(is_call[expired], np.maximum(dS, 0.0), np.maximum(-dS, 0.0))
        delta[expired] = np.where(is_call[expired],
                                  (dS > 0).astype(float),
                                  -(dS < 0).astype(float))

    # live legs only
    valid = ~expired & np.isfinite(S) & np.isfinite(K) & np.isfinite(sigma) & (S > 0) & (K > 0)
    if valid.any():
        Sv, Kv, Tv = S[valid], K[valid], T[valid]
        rdv, rfv, sv = r_d[valid], r_f[valid], sigma[valid]
        cv = is_call[valid]

        sqrtT = np.sqrt(Tv)
        d1 = (np.log(Sv / Kv) + (rdv - rfv + 0.5 * sv ** 2) * Tv) / (sv * sqrtT)
        d2 = d1 - sv * sqrtT

        nd1 = ndtr(d1);   nd2 = ndtr(d2)
        nd1n = ndtr(-d1); nd2n = ndtr(-d2)
        npd1 = np.exp(-0.5 * d1 ** 2) * (1.0 / np.sqrt(2 * np.pi))

        disc_d = np.exp(-rdv * Tv)
        disc_f = np.exp(-rfv * Tv)

        price[valid] = np.where(cv, Sv * disc_f * nd1 - Kv * disc_d * nd2,
                                    Kv * disc_d * nd2n - Sv * disc_f * nd1n)
        delta[valid] = np.where(cv, disc_f * nd1, -disc_f * nd1n)
        gamma[valid] = disc_f * npd1 / (Sv * sv * sqrtT)
        vega[valid]  = Sv * disc_f * npd1 * sqrtT / 100.0  # per 1% vol move

        theta_call = (-Sv * disc_f * npd1 * sv / (2 * sqrtT)
                      - rdv * Kv * disc_d * nd2
                      + rfv * Sv * disc_f * nd1)
        theta_put  = (-Sv * disc_f * npd1 * sv / (2 * sqrtT)
                      + rdv * Kv * disc_d * nd2n
                      - rfv * Sv * disc_f * nd1n)
        theta[valid] = np.where(cv, theta_call, theta_put) / 365.0

    return price, delta, gamma, vega, theta

