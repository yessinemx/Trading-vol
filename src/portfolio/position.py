"""Option position and portfolio cash management"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class OptionPosition:
    """Single open option leg"""
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    option_type: str          # 'call' or 'put'
    strike: float
    notional: float           # in domestic currency
    direction: int            # +1 long, -1 short
    premium_paid: float       # positive = cash out
    vol_open: float = 0.0     # implied vol used at inception 


@dataclass
class Portfolio:
    """
    Tracks open option positions, the spot delta hedge and cash balances

    The delta hedge is held as a foreign-currency spot position
    (hedge_units) carried at cost; its P&L is realised through spot
    moves and booked into cash whenever the hedge is rebalanced
    """
    cash: float = 0.0
    positions: list[OptionPosition] = field(default_factory=list)
    pnl_history: list[dict] = field(default_factory=list)

    # Delta hedge state
    hedge_units: float = 0.0   # signed foreign-ccy units held to hedge delta
    hedge_cost: float = 0.0    # spot level at which current hedge was struck
    _hedge_init: bool = False  # whether the hedge has been struck at least once

    def open_position(self, pos: OptionPosition) -> None:
        self.cash -= pos.premium_paid
        self.positions.append(pos)

    def close_position(self, pos: OptionPosition, settle_value: float) -> None:
        """Book the signed option settlement into cash and remove the leg"""
        self.cash += pos.direction * settle_value * abs(pos.notional)
        if pos in self.positions:
            self.positions.remove(pos)

    def rebalance_hedge(self, target_units: float, spot: float) -> float:
        """
        Adjust the spot hedge to target_units foreign-currency units at the given spot

        Returns the hedge P&L realised since the last rebalance (domestic currency),
        which is also booked into cash
        """
        if not self._hedge_init:
            self.hedge_units = target_units
            self.hedge_cost = spot
            self._hedge_init = True
            return 0.0
        # mark-to-market P&L on the existing hedge position
        hedge_pnl = self.hedge_units * (spot - self.hedge_cost)
        self.cash += hedge_pnl
        # roll hedge to the new target at the current spot level
        self.hedge_units = target_units
        self.hedge_cost = spot
        return hedge_pnl

    def hedge_mtm(self, spot: float) -> float:
        """Unrealised hedge P&L at the given spot (domestic currency)"""
        if not self._hedge_init:
            return 0.0
        return self.hedge_units * (spot - self.hedge_cost)

    def record_daily(self, date: pd.Timestamp, mtm: float) -> None:
        self.pnl_history.append({"date": date, "mtm": mtm, "cash": self.cash})

    def pnl_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.pnl_history).set_index("date")

