"""Main backtesting engine: wires data, pricing, strategies, and portfolio."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from .data.loader import MarketDataLoader
from .curves.interpolator import CurveInterpolator
from .volatility.surface import VolSurface
from .pricing.garman_kohlhagen import garman_kohlhagen
from .strategies.base import Strategy, Leg
from .portfolio.position import Portfolio, OptionPosition
from .analytics.performance import compute_metrics
from .analytics.greeks_pnl import attribute_pnl


class Backtester:
    """
    End-to-end FX option backtesting engine.

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

    def load_data(self) -> None:
        self._spot = self.loader.load_spot()
        self._fwd = self.loader.load_forward_curve()
        self._rates = self.loader.load_interest_rates()
        self._vol = self.loader.load_vol_surface()

    def run(
        self,
        strategy: Strategy,
        start: str,
        end: str,
        notional: float = 1_000_000,
        roll_freq: str = "W-FRI",
    ) -> dict:
        """
        Run the backtest for the given strategy.

        Returns a dict with:
          - 'metrics': performance summary
          - 'daily_pnl': pd.Series of daily P&L
          - 'greeks': pd.DataFrame of daily Greeks attribution
        """
        assert self._spot is not None, "Call load_data() first."

        dates = pd.date_range(start=start, end=end, freq="B")
        roll_dates = set(strategy.roll_dates(pd.Timestamp(start), pd.Timestamp(end), roll_freq))

        portfolio = Portfolio()
        daily_pnl: list[dict] = []
        open_positions: list[dict] = []

        prev_mtm = 0.0

        for date in dates:
            if date not in self._spot.index:
                continue

            S = float(self._spot.loc[date, "MID_PRICE"])
            vol_df_date = self._vol[self._vol["DATE"] == date]
            rates_date = self._rates[self._rates["DATE"] == date]

            if vol_df_date.empty or rates_date.empty:
                continue

            vol_surface = VolSurface(vol_df_date)

            # Split domestic/foreign curves (curve_id 1=domestic, 2=foreign)
            rd_interp = CurveInterpolator.from_dataframe(
                rates_date[rates_date["curve_id"] == 1], date
            )
            rf_interp = CurveInterpolator.from_dataframe(
                rates_date[rates_date["curve_id"] == 2], date
            )

            # Close expired positions
            still_open = []
            for pos_info in open_positions:
                pos: OptionPosition = pos_info["position"]
                if date >= pos.expiry:
                    t = max((pos.expiry - date).days / 365.25, 0)
                    vol = vol_surface.get_vol(pos.strike, t)
                    rd = float(rd_interp(t))
                    rf = float(rf_interp(t))
                    result = garman_kohlhagen(S, pos.strike, t, rd, rf, vol, pos.option_type)
                    portfolio.cash += pos.direction * result.price * notional
                else:
                    still_open.append(pos_info)
            open_positions = still_open

            # Open new positions on roll dates
            if date in roll_dates:
                for leg in strategy.legs:
                    t = leg.tenor_days / 365.25
                    rd = float(rd_interp(t))
                    rf = float(rf_interp(t))
                    # Approximate ATM strike via forward
                    fwd_df = self._fwd[self._fwd["DATE"] == date]
                    if not fwd_df.empty:
                        fwd_interp = CurveInterpolator.from_dataframe(fwd_df, date)
                        fwd_pts = float(fwd_interp(t))
                        # Forward points: add to spot (assume points are in price units)
                        strike = S + fwd_pts
                    else:
                        strike = S * np.exp((rd - rf) * t)

                    vol = vol_surface.get_vol(strike, t)
                    result = garman_kohlhagen(S, strike, t, rd, rf, vol, leg.option_type)
                    premium = result.price * notional * leg.ratio
                    expiry = date + pd.Timedelta(days=leg.tenor_days)

                    pos = OptionPosition(
                        open_date=date,
                        expiry=expiry,
                        option_type=leg.option_type,
                        strike=strike,
                        notional=notional * leg.ratio,
                        direction=leg.direction,
                        premium_paid=premium * leg.direction,
                    )
                    portfolio.open_position(pos)
                    open_positions.append({"position": pos, "open_greeks": result})

            # Mark-to-market
            mtm = 0.0
            greeks_today = {"date": date, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
            for pos_info in open_positions:
                pos = pos_info["position"]
                t = max((pos.expiry - date).days / 365.25, 1e-6)
                vol = vol_surface.get_vol(pos.strike, t)
                rd = float(rd_interp(t))
                rf = float(rf_interp(t))
                result = garman_kohlhagen(S, pos.strike, t, rd, rf, vol, pos.option_type)
                mtm += pos.direction * result.price * pos.notional
                greeks_today["delta"] += pos.direction * result.delta * pos.notional
                greeks_today["gamma"] += pos.direction * result.gamma * pos.notional
                greeks_today["vega"] += pos.direction * result.vega * pos.notional
                greeks_today["theta"] += pos.direction * result.theta * pos.notional

            day_pnl = mtm - prev_mtm
            prev_mtm = mtm

            daily_pnl.append({
                "date": date,
                "pnl": day_pnl,
                **{k: v for k, v in greeks_today.items() if k != "date"},
            })

        pnl_df = pd.DataFrame(daily_pnl).set_index("date")
        metrics = compute_metrics(pnl_df["pnl"])

        return {
            "metrics": metrics,
            "daily_pnl": pnl_df["pnl"],
            "greeks": pnl_df[["delta", "gamma", "vega", "theta"]],
            "portfolio": portfolio,
        }
