"""Strategy base class + Leg dataclass"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class Leg:
    option_type: str        # 'call' or 'put'
    strike_delta: float     # target delta; 0.5 = ATM
    tenor_days: int
    direction: int          # +1 long, -1 short
    ratio: float = 1.0


class Strategy(ABC):
    # set to False to skip daily delta hedging
    delta_hedge = True

    @property
    @abstractmethod
    def legs(self):
        ...

    @property
    @abstractmethod
    def name(self):
        ...

    def roll_dates(self, start, end, freq="W-FRI"):
        return pd.date_range(start=start, end=end, freq=freq)

    def size(self, leg, notional, spot):
        # base notional scaled by the leg ratio; override for vega-targeting etc
        return notional * leg.ratio

    def signal(self, date, spot, history):
        # entry filter on roll dates; default = always trade
        return True

