import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    rsi = RSIIndicator(close=data["close"], window=window, fillna=True)
    data["rsi"] = rsi.rsi()
    return data

def calculate_macd(data):
    """Calculate MACD indicators."""
    macd = MACD(close=data["close"], fillna=True)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()
    return data

def calculate_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands."""
    bb = BollingerBands(close=data["close"], window=window, fillna=True)
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()
    return data

def calculate_moving_averages(data, window=20):
    """Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)."""
    data[f"sma_{window}"] = data["close"].rolling(window=window, min_periods=1).mean()
    data[f"ema_{window}"] = data["close"].ewm(span=window, adjust=False, min_periods=1).mean()
    return data

def calculate_volume_indicators(data, window=5):
    """Calculate volume surge compared to moving average of volume."""
    data["volume_surge"] = data["volume"] / data["volume"].rolling(window=window, min_periods=1).mean()
    return data

def add_indicators(data):
    """Add all indicators to the DataFrame."""
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_bollinger_bands(data)
    data = calculate_moving_averages(data)
    data = calculate_volume_indicators(data)
    return data