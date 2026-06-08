"""Call ratio spread: long 1 ATM call, short 2 OTM calls"""

from .base import Strategy, Leg


class CallRatioSpread(Strategy):
    def __init__(self, tenor_days=30, otm_delta=0.25):
        self._tenor = tenor_days
        self._otm_delta = otm_delta

    @property
    def name(self):
        return "CallRatioSpread"

    @property
    def legs(self):
        return [
            Leg("call", strike_delta=0.5, tenor_days=self._tenor, direction=1, ratio=1.0),
            Leg("call", strike_delta=self._otm_delta, tenor_days=self._tenor, direction=-1, ratio=2.0),
        ]
