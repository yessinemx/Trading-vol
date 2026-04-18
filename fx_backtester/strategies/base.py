"""Base class for option strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Leg:
    option_type: str        # 'call' or 'put'
    strike_delta: float     # target delta (e.g. 0.25, 0.5 for ATM)
    tenor_days: int         # option tenor in calendar days
    direction: int          # +1 long, -1 short
    ratio: float = 1.0      # size multiplier for ratio spreads


class Strategy(ABC):
    """Abstract base for multi-leg FX option strategies."""

    @property
    @abstractmethod
    def legs(self) -> list[Leg]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def roll_dates(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "W-FRI") -> pd.DatetimeIndex:
        """Generate rebalancing dates between start and end at the given frequency."""
        return pd.date_range(start=start, end=end, freq=freq)
