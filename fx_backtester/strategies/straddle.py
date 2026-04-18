"""ATM Straddle strategy: long call + long put at the same strike."""

from .base import Strategy, Leg


class Straddle(Strategy):
    def __init__(self, tenor_days: int = 30, direction: int = 1):
        self._tenor = tenor_days
        self._direction = direction

    @property
    def name(self) -> str:
        return "Straddle"

    @property
    def legs(self) -> list[Leg]:
        return [
            Leg("call", strike_delta=0.5, tenor_days=self._tenor, direction=self._direction),
            Leg("put", strike_delta=0.5, tenor_days=self._tenor, direction=self._direction),
        ]
