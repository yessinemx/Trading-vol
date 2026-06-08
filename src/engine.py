"""Backtester: daily loop, pricing, hedging, attribution"""

import pandas as pd
import numpy as np
from scipy.optimize import brentq

from .data.loader import MarketDataLoader
from .curves.interpolator import CurveInterpolator
from .volatility.surface import VolSurface
from .pricing.garman_kohlhagen import garman_kohlhagen, garman_kohlhagen_vec
from .strategies.base import Strategy, Leg
from .portfolio.position import Portfolio, OptionPosition
from .analytics.metrics import compute_metrics
from .analytics.greeks_pnl import attribute_pnl


class Backtester:
    """FX option backtester with daily delta hedging"""

    def __init__(self, data_dir):
        self.loader = MarketDataLoader(data_dir)
        self._spot = None
        self._fwd = None
        self._rates = None
        self._vol = None
        self._fwd_by_date = {}
        self._rates_by_date = {}
        self._vol_by_date = {}
        self._vol_surfaces = {}

    def load_data(self):
        self._spot = self.loader.load_spot()
        self._fwd = self.loader.load_forward_curve()
        self._rates = self.loader.load_interest_rates()
        self._vol = self.loader.load_vol_surface()
        self._fwd_by_date = self.loader.pivot_by_date(self._fwd)
        self._rates_by_date = self.loader.pivot_by_date(self._rates)
        self._vol_by_date = self.loader.pivot_by_date(self._vol)
        # build all surfaces once 
        self._vol_surfaces = {
            date: VolSurface(sub) for date, sub in self._vol_by_date.items()
        }

    @staticmethod
    def _strike_for_leg(leg, S, fwd_outright, rd, rf, t, vol_surface):
        # ATM leg -> strike = forward
        if abs(leg.strike_delta - 0.5) < 1e-9:
            return fwd_outright

        target = leg.strike_delta
        lo, hi = fwd_outright * 0.5, fwd_outright * 2.0

        def delta_err(k):
            vol = vol_surface.get_vol(k, t)
            res = garman_kohlhagen(S, k, t, rd, rf, vol, leg.option_type)
            return abs(res.delta) - target

        try:
            return float(brentq(delta_err, lo, hi, xtol=1e-6, maxiter=50))
        except ValueError:
            # bracket didn't straddle the root, fall back to a grid
            grid = np.linspace(lo, hi, 201)
            best_k, best_err = fwd_outright, 1e9
            for k in grid:
                err = abs(delta_err(k))
                if err < best_err:
                    best_err, best_k = err, k
            return float(best_k)

    def run(self, strategy, start, end, notional=1_000_000, roll_freq="W-FRI"):
        assert self._spot is not None, "call load_data() first"

        dates = pd.date_range(start=start, end=end, freq="B")
        roll_dates = set(
            strategy.roll_dates(pd.Timestamp(start), pd.Timestamp(end), roll_freq)
        )
        do_hedge = getattr(strategy, "delta_hedge", True)

        portfolio = Portfolio()
        open_positions = []
        records = []

        prev_mtm = 0.0
        prev_S = None
        prev_delta = 0.0
        prev_snap = {}
        mtm_new_today = 0.0

        for date in dates:
            if date not in self._spot.index:
                continue

            S = float(self._spot.loc[date, "MID_PRICE"])
            vol_surface = self._vol_surfaces.get(date)
            rates_date = self._rates_by_date.get(date)
            if vol_surface is None or vol_surface.is_empty or rates_date is None or rates_date.empty:
                continue
            rd_interp = CurveInterpolator.from_dataframe(
                rates_date[rates_date["curve_id"] == 1], date
            )
            rf_interp = CurveInterpolator.from_dataframe(
                rates_date[rates_date["curve_id"] == 2], date
            )

            # settle expired legs at intrinsic
            still_open = []
            for pos in open_positions:
                if date >= pos.expiry:
                    intrinsic = (
                        max(S - pos.strike, 0.0) if pos.option_type == "call"
                        else max(pos.strike - S, 0.0)
                    )
                    portfolio.close_position(pos, intrinsic)
                else:
                    still_open.append(pos)
            open_positions = still_open

            # open new legs on roll dates 
            mtm_new_today = 0.0
            if date in roll_dates:
                history = self._spot.loc[:date, "MID_PRICE"]
                if strategy.signal(date, S, history):
                    fwd_df = self._fwd_by_date.get(date)
                    for leg in strategy.legs:
                        t = leg.tenor_days / 365.25
                        rd = float(rd_interp(t))
                        rf = float(rf_interp(t))
                        if fwd_df is not None and not fwd_df.empty:
                            fwd_interp = CurveInterpolator.from_dataframe(fwd_df, date)
                            fwd_outright = S + float(fwd_interp(t))
                        else:
                            fwd_outright = S * np.exp((rd - rf) * t)

                        strike = self._strike_for_leg(
                            leg, S, fwd_outright, rd, rf, t, vol_surface
                        )
                        vol = vol_surface.get_vol(strike, t)
                        res = garman_kohlhagen(S, strike, t, rd, rf, vol, leg.option_type)

                        leg_notional = strategy.size(leg, notional, S)
                        premium = res.price * leg_notional * leg.direction
                        expiry = date + pd.Timedelta(days=leg.tenor_days)

                        pos = OptionPosition(
                            open_date=date,
                            expiry=expiry,
                            option_type=leg.option_type,
                            strike=strike,
                            notional=leg_notional,
                            direction=leg.direction,
                            premium_paid=premium,
                            vol_open=vol,
                        )
                        mtm_new_today += res.price * leg_notional * leg.direction
                        portfolio.open_position(pos)
                        open_positions.append(pos)

            # MTM + aggregated greeks 
            mtm = 0.0
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
            snap_today = {}
            attr_delta = attr_gamma = attr_vega = attr_theta = 0.0
            carry_pnl = 0.0  # MTM change of legs alive yesterday too

            dS = (S - prev_S) if prev_S is not None else 0.0

            if open_positions:
                n_pos = len(open_positions)
                T_arr     = np.array([max((pos.expiry - date).days / 365.25, 1e-6) for pos in open_positions])
                K_arr     = np.array([pos.strike for pos in open_positions])
                vol_arr   = np.array([vol_surface.get_vol(pos.strike, t) for pos, t in zip(open_positions, T_arr)])
                rd_arr    = np.array([float(rd_interp(t)) for t in T_arr])
                rf_arr    = np.array([float(rf_interp(t)) for t in T_arr])
                is_call   = np.array([pos.option_type == "call" for pos in open_positions])
                signed_ns = np.array([pos.direction * pos.notional for pos in open_positions])

                prices, deltas, gammas, vegas, thetas = garman_kohlhagen_vec(
                    np.full(n_pos, S), K_arr, T_arr, rd_arr, rf_arr, vol_arr, is_call
                )

                mtm               = float((signed_ns * prices).sum())
                greeks["delta"]   = float((signed_ns * deltas).sum())
                greeks["gamma"]   = float((signed_ns * gammas).sum())
                greeks["vega"]    = float((signed_ns * vegas).sum())
                greeks["theta"]   = float((signed_ns * thetas).sum())

                for i, pos in enumerate(open_positions):
                    key = id(pos)
                    signed_n = signed_ns[i]
                    snap_today[key] = {
                        "price": prices[i], "delta": deltas[i], "gamma": gammas[i],
                        "vega": vegas[i], "theta": thetas[i],
                        "vol": vol_arr[i], "signed_n": signed_n,
                    }
                    # only attribute legs that existed yesterday
                    prev = prev_snap.get(key)
                    if prev is not None and prev_S is not None:
                        d_sigma = vol_arr[i] - prev["vol"]
                        g = attribute_pnl(
                            delta=prev["signed_n"] * prev["delta"],
                            gamma=prev["signed_n"] * prev["gamma"],
                            vega=abs(prev["signed_n"] * prev["vega"]),
                            theta=prev["signed_n"] * prev["theta"],
                            dS=dS,
                            dSigma=d_sigma,
                            dt=1 / 252,
                        )
                        attr_delta += g.delta_pnl
                        attr_gamma += g.gamma_pnl
                        attr_vega  += g.vega_pnl
                        attr_theta += g.theta_pnl
                        carry_pnl  += signed_n * (prices[i] - prev["price"])

            port_delta_units = greeks["delta"]

            # daily delta hedge
            hedge_pnl = 0.0
            if do_hedge:
                target_hedge = -port_delta_units
                hedge_pnl = portfolio.rebalance_hedge(target_hedge, S)
            elif prev_S is not None:
                hedge_pnl = portfolio.hedge_mtm(S) - portfolio.hedge_mtm(prev_S)

            # strip MTM of freshly-opened legs 
            option_pnl = mtm - prev_mtm - mtm_new_today
            day_pnl = option_pnl + hedge_pnl

            residual = carry_pnl - (attr_delta + attr_gamma + attr_vega + attr_theta)
            # report delta + hedge as a single net-delta line
            hedged_delta_pnl = attr_delta + hedge_pnl

            records.append({
                "date": date,
                "pnl": day_pnl,
                "option_pnl": option_pnl,
                "hedge_pnl": hedge_pnl,
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "vega": greeks["vega"],
                "theta": greeks["theta"],
                "delta_pnl": hedged_delta_pnl,
                "gamma_pnl": attr_gamma,
                "vega_pnl": attr_vega,
                "theta_pnl": attr_theta,
                "residual_pnl": residual,
            })

            prev_mtm = mtm
            prev_S = S
            prev_delta = port_delta_units
            prev_snap = snap_today

        df = pd.DataFrame(records).set_index("date")
        metrics = compute_metrics(df["pnl"])

        return {
            "metrics": metrics,
            "daily_pnl": df["pnl"],
            "greeks": df[["delta", "gamma", "vega", "theta"]],
            "attribution": df[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "residual_pnl"]],
            "components": df[["option_pnl", "hedge_pnl"]],
            "portfolio": portfolio,
        }

