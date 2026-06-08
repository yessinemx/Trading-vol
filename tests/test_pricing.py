"""Tests: Garman-Kohlhagen scalar pricer."""

import numpy as np

from src.pricing.garman_kohlhagen import garman_kohlhagen


def test_put_call_parity():
    # C - P = S e^{-rf T} - K e^{-rd T}
    S, K, T, rd, rf, sigma = 1.10, 1.12, 0.5, 0.03, 0.01, 0.10
    call = garman_kohlhagen(S, K, T, rd, rf, sigma, "call").price
    put = garman_kohlhagen(S, K, T, rd, rf, sigma, "put").price
    lhs = call - put
    rhs = S * np.exp(-rf * T) - K * np.exp(-rd * T)
    assert abs(lhs - rhs) < 1e-9


def test_call_delta_bounds():
    res = garman_kohlhagen(1.10, 1.10, 0.25, 0.02, 0.01, 0.12, "call")
    assert 0.0 <= res.delta <= 1.0


def test_put_delta_bounds():
    res = garman_kohlhagen(1.10, 1.10, 0.25, 0.02, 0.01, 0.12, "put")
    assert -1.0 <= res.delta <= 0.0


def test_gamma_vega_positive():
    res = garman_kohlhagen(1.10, 1.10, 0.25, 0.02, 0.01, 0.12, "call")
    assert res.gamma > 0
    assert res.vega > 0


def test_expiry_intrinsic():
    itm = garman_kohlhagen(1.20, 1.10, 0.0, 0.02, 0.01, 0.12, "call")
    assert abs(itm.price - 0.10) < 1e-12
    otm = garman_kohlhagen(1.00, 1.10, 0.0, 0.02, 0.01, 0.12, "call")
    assert otm.price == 0.0


def test_degenerate_inputs_are_finite():
    # zero vol shouldn't blow up
    res = garman_kohlhagen(1.10, 1.10, 0.25, 0.02, 0.01, 0.0, "call")
    assert np.isfinite(res.price)

