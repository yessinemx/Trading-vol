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
    vol_open: float = 0.0     # implied vol used at inception (for attribution)


@dataclass
class Portfolio:
    """
    Tracks open option positions, the spot delta hedge, and cash.

    The delta hedge is held as a *foreign-currency* spot position
    (``hedge_units``) carried at cost; its P&L is realised through spot
    moves and is booked into ``cash`` whenever the hedge is rebalanced.
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
        """Book the option settlement (signed) into cash and drop the leg."""
        self.cash += pos.direction * settle_value * abs(pos.notional)
        if pos in self.positions:
            self.positions.remove(pos)

    def rebalance_hedge(self, target_units: float, spot: float) -> float:
        """
        Adjust the spot hedge to ``target_units`` foreign-ccy units at ``spot``.

        Returns the hedge P&L realised since the last rebalance (domestic ccy),
        which is also booked into cash. The very first rebalance only
        establishes the hedge (no P&L), avoiding a spurious mark against the
        zero-initialised cost.
        """
        if not self._hedge_init:
            self.hedge_units = target_units
            self.hedge_cost = spot
            self._hedge_init = True
            return 0.0
        # Mark-to-market P&L on the existing hedge since it was struck.
        hedge_pnl = self.hedge_units * (spot - self.hedge_cost)
        self.cash += hedge_pnl
        # Roll the hedge to the new target at the current spot.
        self.hedge_units = target_units
        self.hedge_cost = spot
        return hedge_pnl

    def hedge_mtm(self, spot: float) -> float:
        """Unrealised hedge P&L at ``spot`` (domestic ccy)."""
        if not self._hedge_init:
            return 0.0
        return self.hedge_units * (spot - self.hedge_cost)

    def record_daily(self, date: pd.Timestamp, mtm: float) -> None:
        self.pnl_history.append({"date": date, "mtm": mtm, "cash": self.cash})

    def pnl_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.pnl_history).set_index("date")

