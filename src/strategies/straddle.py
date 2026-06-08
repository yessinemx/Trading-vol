"""ATM straddle: long (or short) 1 call + 1 put at the same strike"""

from .base import Strategy, Leg


class Straddle(Strategy):
    def __init__(self, tenor_days=30, direction=1, signal_fn=None):
        self._tenor = tenor_days
        self._direction = direction
        self._signal_fn = signal_fn

    @property
    def name(self):
        return "Straddle"

    @property
    def legs(self):
        return [
            Leg("call", strike_delta=0.5, tenor_days=self._tenor, direction=self._direction),
            Leg("put",  strike_delta=0.5, tenor_days=self._tenor, direction=self._direction),
        ]

    def signal(self, date, spot, history):
        if self._signal_fn is None:
            return True
        return bool(self._signal_fn(history))
