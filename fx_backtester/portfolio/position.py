"""Position and cash management for the backtester."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class OptionPosition:
    """Represents a single open option leg."""
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    option_type: str          # 'call' or 'put'
    strike: float
    notional: float           # in domestic currency
    direction: int            # +1 long, -1 short
    premium_paid: float       # positive = cash out
    delta_hedge: float = 0.0  # current delta hedge in FX notional


@dataclass
class Portfolio:
    """Tracks open positions, cash, and daily P&L."""
    cash: float = 0.0
    positions: list[OptionPosition] = field(default_factory=list)
    pnl_history: list[dict] = field(default_factory=list)

    def open_position(self, pos: OptionPosition) -> None:
        self.cash -= pos.premium_paid
        self.positions.append(pos)

    def close_position(self, pos: OptionPosition, close_price: float) -> float:
        pnl = pos.direction * (close_price - pos.premium_paid / abs(pos.notional)) * abs(pos.notional)
        self.cash += close_price * abs(pos.notional) * pos.direction + pos.premium_paid
        self.positions.remove(pos)
        return pnl

    def record_daily(self, date: pd.Timestamp, mtm: float) -> None:
        self.pnl_history.append({"date": date, "mtm": mtm, "cash": self.cash})

    def pnl_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.pnl_history).set_index("date")
