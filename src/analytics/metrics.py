"""Performance and risk metrics from a daily P&L series"""

import numpy as np

class PerformanceMetrics:
    def __init__(self, daily_pnl, rf=0.0, periods_per_year=252):
        self.pnl = daily_pnl
        self.rf = rf
        self.periods_per_year = periods_per_year

    def total_pnl(self):
        return float(self.pnl.sum())

    def avg_daily_pnl(self):
        return float(self.pnl.mean())

    def win_rate(self):
        return float((self.pnl > 0).mean())

    def volatility_ann(self):
        return float(self.pnl.std() * np.sqrt(self.periods_per_year))

    def sharpe(self):
        excess = self.pnl - self.rf / self.periods_per_year
        sd = excess.std()
        if sd == 0:
            return float("nan")
        return float(np.sqrt(self.periods_per_year) * excess.mean() / sd)

    def sortino(self):
        excess = self.pnl - self.rf / self.periods_per_year
        downside = excess[excess < 0].std()
        if downside == 0:
            return float("nan")
        return float(np.sqrt(self.periods_per_year) * excess.mean() / downside)

    def summary(self):
        return {
            "total_pnl": self.total_pnl(),
            "sharpe": self.sharpe(),
            "sortino": self.sortino(),
            "win_rate": self.win_rate(),
            "avg_daily_pnl": self.avg_daily_pnl(),
            "volatility_ann": self.volatility_ann(),
        }


class RiskMetrics:
    def __init__(self, daily_pnl, confidence=0.95):
        self.pnl = daily_pnl
        self.confidence = confidence

    def max_drawdown(self):
        cum = self.pnl.cumsum()
        return float((cum - cum.cummax()).min())

    def var(self, confidence=None):
        c = self.confidence if confidence is None else confidence
        return float(np.percentile(self.pnl, (1 - c) * 100))

    def cvar(self, confidence=None):
        v = self.var(confidence)
        tail = self.pnl[self.pnl <= v]
        if tail.empty:
            return float("nan")
        return float(tail.mean())

    def summary(self):
        return {
            "max_drawdown": self.max_drawdown(),
            "var_95": self.var(0.95),
            "cvar_95": self.cvar(0.95),
        }


def compute_metrics(daily_pnl, rf=0.0):
    perf = PerformanceMetrics(daily_pnl, rf=rf).summary()
    risk = RiskMetrics(daily_pnl).summary()
    return {
        "total_pnl": perf["total_pnl"],
        "sharpe": perf["sharpe"],
        "sortino": perf["sortino"],
        "max_drawdown": risk["max_drawdown"],
        "var_95": risk["var_95"],
        "cvar_95": risk["cvar_95"],
        "win_rate": perf["win_rate"],
        "avg_daily_pnl": perf["avg_daily_pnl"],
        "volatility_ann": perf["volatility_ann"],
    }
