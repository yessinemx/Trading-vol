"""Option position + portfolio book-keeping"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class OptionPosition:
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    option_type: str         # 'call' or 'put'
    strike: float
    notional: float          # domestic currency
    direction: int           # +1 long, -1 short
    premium_paid: float      # >0 = cash out
    vol_open: float = 0.0    # implied vol at inception


@dataclass
class Portfolio:
    cash: float = 0.0
    positions: list = field(default_factory=list)
    pnl_history: list = field(default_factory=list)

    # delta hedge held as a spot position in the foreign ccy, marked to spot
    hedge_units: float = 0.0
    hedge_cost: float = 0.0
    _hedge_init: bool = False

    def open_position(self, pos):
        self.cash -= pos.premium_paid
        self.positions.append(pos)

    def close_position(self, pos, settle_value):
        self.cash += pos.direction * settle_value * abs(pos.notional)
        if pos in self.positions:
            self.positions.remove(pos)

    def rebalance_hedge(self, target_units, spot):
        # roll hedge to target_units, realise P&L on the previous leg
        if not self._hedge_init:
            self.hedge_units = target_units
            self.hedge_cost = spot
            self._hedge_init = True
            return 0.0
        hedge_pnl = self.hedge_units * (spot - self.hedge_cost)
        self.cash += hedge_pnl
        self.hedge_units = target_units
        self.hedge_cost = spot
        return hedge_pnl

    def hedge_mtm(self, spot):
        if not self._hedge_init:
            return 0.0
        return self.hedge_units * (spot - self.hedge_cost)

    def record_daily(self, date, mtm):
        self.pnl_history.append({"date": date, "mtm": mtm, "cash": self.cash})

    def pnl_dataframe(self):
        return pd.DataFrame(self.pnl_history).set_index("date")

