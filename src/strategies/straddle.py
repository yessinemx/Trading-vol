"""ATM straddle strategy: long call + long put at the same strike"""

from collections.abc import Callable

import pandas as pd

from .base import Strategy, Leg


class Straddle(Strategy):
    def __init__(
        self,
        tenor_days: int = 30,
        direction: int = 1,
        signal_fn: Callable[[pd.Series], bool] | None = None,
    ):
        self._tenor = tenor_days
        self._direction = direction
        self._signal_fn = signal_fn

    @property
    def name(self) -> str:
        return "Straddle"

    @property
    def legs(self) -> list[Leg]:
        return [
            Leg("call", strike_delta=0.5, tenor_days=self._tenor, direction=self._direction),
            Leg("put", strike_delta=0.5, tenor_days=self._tenor, direction=self._direction),
        ]

    def signal(self, date: pd.Timestamp, spot: float, history: pd.Series) -> bool:
        if self._signal_fn is None:
            return True
        return bool(self._signal_fn(history))
