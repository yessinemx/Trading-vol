"""Calendar spread: long far-dated option, short near-dated option."""

from .base import Strategy, Leg


class CalendarSpread(Strategy):
    def __init__(self, near_tenor_days: int = 30, far_tenor_days: int = 90, option_type: str = "call"):
        self._near = near_tenor_days
        self._far = far_tenor_days
        self._type = option_type

    @property
    def name(self) -> str:
        return "CalendarSpread"

    @property
    def legs(self) -> list[Leg]:
        return [
            Leg(self._type, strike_delta=0.5, tenor_days=self._far, direction=1),
            Leg(self._type, strike_delta=0.5, tenor_days=self._near, direction=-1),
        ]
