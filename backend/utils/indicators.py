import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, money_flow_index  # ✅ Keep money_flow_index!
from sklearn.preprocessing import StandardScaler

# ✅ Configure Logger
logger = logging.getLogger(__name__)

def preprocess_data_with_indicators(df):
    """
    Add advanced technical indicators and sentiment analysis.
    Returns:
        - Processed DataFrame (df)
        - Scaler (only if required for LSTM)
    """
    try:
        df = df.copy()
        
        # ✅ Ensure `ticker` column is preserved if it exists
        if "ticker" not in df.columns:
            logging.warning("⚠️ Warning: 'ticker' column is missing in preprocess_data_with_indicators!")
        else:
            df["ticker"] = df["ticker"].astype(str)  # Ensure it stays a string

        # ✅ Ensure required columns exist
        required_cols = ["open", "close", "high", "low", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")

        # ✅ Price Change & Volatility
        df["price_change"] = (df["close"] - df["open"]) / df["open"]
        df["volatility"] = (df["high"] - df["low"]) / df["low"]

        # ✅ Volume Surge Calculation
        df["volume_surge"] = df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()

        # ✅ On-Balance Volume (OBV)
        df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"], fillna=True).on_balance_volume()

        # ✅ Williams %R
        df["williams_r"] = WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14, fillna=True).williams_r()

        # ✅ Exponential Moving Averages (EMA 12, 26 for MACD)
        df["ema_12"] = EMAIndicator(close=df["close"], window=12, fillna=True).ema_indicator()
        df["ema_26"] = EMAIndicator(close=df["close"], window=26, fillna=True).ema_indicator()

        # ✅ Bollinger Bands
        bb = BollingerBands(close=df["close"], window=20, fillna=True)
        df["bollinger_upper"] = bb.bollinger_hband()
        df["bollinger_lower"] = bb.bollinger_lband()

        # ✅ MACD Calculation
        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        df["macd_diff"].fillna(0, inplace=True)

        # ✅ RSI Calculation
        df["rsi"] = RSIIndicator(close=df["close"], window=14, fillna=True).rsi()

        # ✅ ADX Calculation
        df["adx"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14, fillna=True).adx()
        df["adx"].fillna(0, inplace=True)

        # ✅ ATR Calculation
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14, fillna=True).average_true_range()
        df["atr"].fillna(0, inplace=True)

        # ✅ Money Flow Index (MFI) (Correct Import!)
        df["mfi"] = money_flow_index(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14)
        df["mfi"].fillna(0, inplace=True)

        # ✅ 🚀 Add `buy_signal` and `sell_signal` if missing
        if "buy_signal" not in df.columns:
            df["buy_signal"] = (
                ((df["rsi"] < 50) & (df["macd_line"] > df["macd_signal"])) |  # RSI less than 50 & MACD crossover
                ((df["adx"] > 15) & (df["macd_diff"] > 0)) |  # ADX confirms trend strength
                ((df["close"] < df["bollinger_lower"]) & (df["volume_surge"] > 1.1))  # Price near lower Bollinger Band & volume spike
            ).astype(int)

        if "sell_signal" not in df.columns:
            df["sell_signal"] = ((df["rsi"] > 70) & (df["macd_diff"] < 0)).astype(int)

        # ✅ Sentiment Score (Placeholder if missing)
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0  # Placeholder, modify if actual sentiment data is available

        # ✅ Debugging Step: Print available columns after processing
        logger.info(f"📌 Final Columns in DataFrame: {df.columns.tolist()}")

        # ✅ Standardizing Feature Scaling for LSTM Compatibility
        scaler = StandardScaler()
        feature_columns = ["price_change", "volatility", "volume", "rsi", "macd_diff", "adx", "atr", "mfi"]
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        logger.info("✅ Data successfully standardized for LSTM model.")
        
         # ✅ Log Buy Signal Distribution
        buy_signal_count = df["buy_signal"].sum()
        logging.info(f"📌 Total Buy Signals Detected: {buy_signal_count}")

        return df, scaler  # 🔥 Ensure this function returns BOTH df and scaler

    except Exception as e:
        logger.error(f"❌ Error in preprocess_data_with_indicators: {e}")
        raise
