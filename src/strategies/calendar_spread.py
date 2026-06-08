"""Calendar spread: long far-dated, short near-dated, same strike"""

from .base import Strategy, Leg


class CalendarSpread(Strategy):
    def __init__(self, near_tenor_days=30, far_tenor_days=90, option_type="call"):
        self._near = near_tenor_days
        self._far = far_tenor_days
        self._type = option_type

    @property
    def name(self):
        return "CalendarSpread"

    @property
    def legs(self):
        return [
            Leg(self._type, strike_delta=0.5, tenor_days=self._far, direction=1),
            Leg(self._type, strike_delta=0.5, tenor_days=self._near, direction=-1),
        ]
