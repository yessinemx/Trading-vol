"""Entry filters used by Strategy.signal"""


def momentum_signal(history, lookback=20):
    # trade only when spot is above its lookback SMA
    if len(history) < lookback:
        return True
    sma = history.iloc[-lookback:].mean()
    return bool(history.iloc[-1] >= sma)


def realised_vol_signal(history, lookback=20, threshold=0.0):
    # trade only when annualised realised vol exceeds threshold (decimal)
    if len(history) < lookback + 1:
        return True
    rets = history.iloc[-(lookback + 1):].pct_change().dropna()
    realised = float(rets.std() * (252 ** 0.5))
    return realised >= threshold
