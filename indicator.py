import pandas as pd
import numpy as np


# Pct Change (Percentage Change)
def pct_change(series):
    """
    series: Series of close prices
    """
    # Calculate directly; pandas handles null values automatically
    res = series.pct_change() * 100
    return res.to_frame(name='Pct_Change')


# SMA (Simple Moving Average)
def SMA(series, window_list):
    """
    series: Price series (Close)
    window_list: List of periods [5, 10, 20...]
    """
    res = pd.DataFrame(index=series.index)
    for window in window_list:
        res[f'SMA_{window}'] = series.rolling(window).mean()
    return res


# EMA (Exponential Moving Average)
def EMA(series, window_list):
    res = pd.DataFrame(index=series.index)
    for window in window_list:
        res[f'EMA_{window}'] = series.ewm(span=window, adjust=False).mean()
    return res


# RSI (Relative Strength Index)
def RSI(series, window_list):
    res = pd.DataFrame(index=series.index)
    delta = series.diff(1)

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    for window in window_list:
        # Use Wilder's Smoothing (alpha=1/n), which is the standard RSI algorithm
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        res[f'RSI_{window}'] = rsi

    return res


# MACD (Moving Average Convergence Divergence)
def MACD(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    macd_bar = 2 * (dif - dea)

    return pd.DataFrame({
        'DIF': dif,
        'DEA': dea,
        'MACD': macd_bar
    }, index=series.index)


# KDJ (Stochastic Oscillator)
def KDJ(df, n=9, m=3):
    """
    df: Must contain 'High', 'Low', 'Close'
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    low_min = low.rolling(n).min()
    high_max = high.rolling(n).max()

    rsv = (close - low_min) / (high_max - low_min) * 100

    # Use ewm to simulate recursive calculation; faster than for-loops and results are approximate.
    # K = 2/3 * Prev_K + 1/3 * RSV
    # This is equivalent to EMA with alpha=1/3
    k = rsv.ewm(alpha=1 / m, adjust=False).mean()
    d = k.ewm(alpha=1 / m, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({'K': k, 'D': d, 'J': j}, index=df.index)


# BOLL (Bollinger Bands)
def BOLL(series, window=20, k=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)

    upper = mid + k * std
    lower = mid - k * std

    return pd.DataFrame({
        'BOLL_Middle': mid,
        'BOLL_Upper': upper,
        'BOLL_Lower': lower
    }, index=series.index)


# OBV (On-Balance Volume)
def OBV(close, volume):
    # Calculate direction
    price_change = close.diff()
    direction = np.sign(price_change)
    
    # Set the first value to 0
    direction.iloc[0] = 0

    obv = (direction * volume).cumsum()
    return obv.to_frame(name='OBV')


# HV (Historical Volatility)
def HV(series, window_list):
    """
    series: Price series (Close)
    window_list: List of periods [10, 20, 60...]
    """
    # 1. Calculate logarithmic returns
    log_ret = np.log(series / series.shift(1))

    res = pd.DataFrame(index=series.index)

    for window in window_list:
        # 2. Calculate rolling standard deviation
        # ddof=1 is Sample Std Dev, commonly used in finance
        rolling_std = log_ret.rolling(window).std(ddof=1)

        # 3. (Optional) Annualization?
        # For AI training, we usually don't need annualization (values approx 0.01~0.03).
        # If annualization is needed, multiply by sqrt(252).
        # We keep raw values here as they are more neural-network friendly (small and stable).
        res[f'HV_{window}'] = rolling_std

    return res
