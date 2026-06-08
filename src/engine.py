"""Main backtesting engine orchestrating data loading, pricing, strategies and portfolio management"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from .data.loader import MarketDataLoader
from .curves.interpolator import CurveInterpolator
from .volatility.surface import VolSurface
from scipy.optimize import brentq

from .pricing.garman_kohlhagen import garman_kohlhagen, garman_kohlhagen_vec
from .strategies.base import Strategy, Leg
from .portfolio.position import Portfolio, OptionPosition
from .analytics.metrics import compute_metrics
from .analytics.greeks_pnl import attribute_pnl


class Backtester:
    """
    End-to-end FX option backtesting engine with daily delta hedging

    Usage
    -----
    bt = Backtester("data/")
    bt.load_data()
    results = bt.run(strategy, start="2020-01-01", end="2022-12-31", notional=1_000_000)
    """

    def __init__(self, data_dir: str | Path):
        self.loader = MarketDataLoader(data_dir)
        self._spot: pd.DataFrame | None = None
        self._fwd: pd.DataFrame | None = None
        self._rates: pd.DataFrame | None = None
        self._vol: pd.DataFrame | None = None
        # date-keyed sub-frame views for fast day-by-day access
        self._fwd_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        self._rates_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        self._vol_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        # pre-built VolSurface objects keyed by date (constructed once in load_data)
        self._vol_surfaces: dict[pd.Timestamp, VolSurface] = {}

    def load_data(self) -> None:
        self._spot = self.loader.load_spot()
        self._fwd = self.loader.load_forward_curve()
        self._rates = self.loader.load_interest_rates()
        self._vol = self.loader.load_vol_surface()
        # pivot stacked data once for efficient day-by-day access
        self._fwd_by_date = self.loader.pivot_by_date(self._fwd)
        self._rates_by_date = self.loader.pivot_by_date(self._rates)
        self._vol_by_date = self.loader.pivot_by_date(self._vol)
        # pre-build all VolSurface objects once; eliminates repeated CubicSpline
        # fitting on every simulation day, which is the dominant computational cost
        self._vol_surfaces = {
            date: VolSurface(sub)
            for date, sub in self._vol_by_date.items()
        }

    # -- Helpers -------------------------------------------------------------
    @staticmethod
    def _strike_for_leg(
        leg: Leg, S: float, fwd_outright: float, rd: float, rf: float, t: float,
        vol_surface: VolSurface,
    ) -> float:
        """
        Resolve a leg's strike from its target delta

        For ATM legs (delta 0.5) the strike equals the forward outright
        For non-ATM legs, uses Brent's method on |delta(K)| - target over
        [fwd*0.5, fwd*2.0]; falls back to a 201-point grid if the bracket
        fails to straddle the root
        """
        if abs(leg.strike_delta - 0.5) < 1e-9:
            return fwd_outright

        target = leg.strike_delta
        lo, hi = fwd_outright * 0.5, fwd_outright * 2.0

        def delta_err(k: float) -> float:
            vol = vol_surface.get_vol(k, t)
            res = garman_kohlhagen(S, k, t, rd, rf, vol, leg.option_type)
            return abs(res.delta) - target

        try:
            return float(brentq(delta_err, lo, hi, xtol=1e-6, maxiter=50))
        except ValueError:
            # fallback: grid search if bracket does not straddle the root
            grid = np.linspace(lo, hi, 201)
            best_k, best_err = fwd_outright, 1e9
            for k in grid:
                err = abs(delta_err(k))
                if err < best_err:
                    best_err, best_k = err, k
            return float(best_k)

    def run(
        self,
        strategy: Strategy,
        start: str,
        end: str,
        notional: float = 1_000_000,
        roll_freq: str = "W-FRI",
    ) -> dict:
        """
        Run the backtest for the given strategy with daily delta hedging

        Returns a dict with keys:
          - 'metrics': performance summary dict
          - 'daily_pnl': pd.Series of total daily P&L (option + hedge)
          - 'greeks': pd.DataFrame of daily portfolio Greeks
          - 'attribution': pd.DataFrame of Greek P&L decomposition
          - 'portfolio': final Portfolio object
        """
        assert self._spot is not None, "Call load_data() first."

        dates = pd.date_range(start=start, end=end, freq="B")
        roll_dates = set(
            strategy.roll_dates(pd.Timestamp(start), pd.Timestamp(end), roll_freq)
        )
        do_hedge = getattr(strategy, "delta_hedge", True)

        portfolio = Portfolio()
        open_positions: list[OptionPosition] = []
        records: list[dict] = []

        prev_mtm = 0.0          # option-only MTM carried forward for daily differencing
        prev_S = None           # previous spot used for hedge rebalancing and attribution
        prev_delta = 0.0        # previous portfolio delta in foreign-currency units
        prev_snap: dict[int, dict] = {}  # per-position Greek snapshot for attribution
        mtm_new_today = 0.0     # MTM of legs opened today; excluded from option_pnl

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

            # -- settle expired positions at intrinsic value -----------------
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

            # -- open new legs on roll dates (entry gated by signal) ---------
            mtm_new_today = 0.0
            if date in roll_dates:
                history = self._spot.loc[:date, "MID_PRICE"]
                if strategy.signal(date, S, history):
                    fwd_df = self._fwd_by_date.get(date)
                    for leg in strategy.legs:
                        t = leg.tenor_days / 365.25
                        rd = float(rd_interp(t))
                        rf = float(rf_interp(t))
                        # forward outright = spot + forward points (in spot price units)
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

            # -- mark-to-market option book and aggregate Greeks -------------
            mtm = 0.0
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
            # per-position Greek snapshot stored for next-day attribution
            snap_today: dict[int, dict] = {}
            # Greek P&L accumulators over positions alive both yesterday and today
            attr_delta = attr_gamma = attr_vega = attr_theta = 0.0
            carry_pnl = 0.0  # MTM change of continuing positions; excludes premium flows

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

                # per-position snapshots and attribution; pricing is already computed
                for i, pos in enumerate(open_positions):
                    key = id(pos)
                    signed_n = signed_ns[i]
                    snap_today[key] = {
                        "price": prices[i], "delta": deltas[i], "gamma": gammas[i],
                        "vega": vegas[i], "theta": thetas[i],
                        "vol": vol_arr[i], "signed_n": signed_n,
                    }
                    # attribute only positions that existed on the previous day
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

            # portfolio delta in foreign-currency units (delta per unit notional)
            port_delta_units = greeks["delta"]

            # -- daily delta hedge -------------------------------------------
            hedge_pnl = 0.0
            if do_hedge:
                # hold -delta units of the foreign currency to neutralise option delta
                target_hedge = -port_delta_units
                hedge_pnl = portfolio.rebalance_hedge(target_hedge, S)
            elif prev_S is not None:
                hedge_pnl = portfolio.hedge_mtm(S) - portfolio.hedge_mtm(prev_S)

            # -- daily P&L: option MTM change plus hedge P&L -----------------
            # subtract MTM of legs opened today to avoid counting their fair-value
            # as a spurious gain on the roll day (premium cash flow is already booked)
            option_pnl = mtm - prev_mtm - mtm_new_today
            day_pnl = option_pnl + hedge_pnl

            # -- Greek P&L attribution (delta/gamma/vega/theta + residual) ---
            residual = carry_pnl - (attr_delta + attr_gamma + attr_vega + attr_theta)
            # the delta attribution is combined with hedge P&L to report the net hedged delta contribution
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

