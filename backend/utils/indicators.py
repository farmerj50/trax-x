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
    Ensures `ticker` column is preserved throughout processing.
    Returns:
        - Processed DataFrame (df)
        - Scaler (for LSTM feature standardization)
    """
    try:
        df = df.copy()

        # ✅ Debug: Print Initial Columns Before Processing
        logger.info(f"🔍 Initial Columns: {df.columns.tolist()}")

        # ✅ Ensure `ticker` column exists before processing
        if "ticker" not in df.columns:
            logger.warning("⚠️ 'ticker' column is missing in preprocess_data_with_indicators!")
        else:
            df["ticker"] = df["ticker"].astype(str)  # Ensure it stays a string

        # ✅ Backup ticker column before transformations
        tickers_before = df["ticker"].unique()
        logger.info(f"📌 Unique tickers BEFORE preprocessing: {len(tickers_before)}")

        # ✅ Ensure required columns exist
        required_cols = ["open", "close", "high", "low", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")

        # ✅ Feature Engineering - Technical Indicators
        df["price_change"] = (df["close"] - df["open"]) / df["open"]
        df["volatility"] = (df["high"] - df["low"]) / df["low"]
        df["volume_surge"] = df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()

        df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"], fillna=True).on_balance_volume()
        df["williams_r"] = WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14, fillna=True).williams_r()

        df["ema_12"] = EMAIndicator(close=df["close"], window=12, fillna=True).ema_indicator()
        df["ema_26"] = EMAIndicator(close=df["close"], window=26, fillna=True).ema_indicator()

        bb = BollingerBands(close=df["close"], window=20, fillna=True)
        df["bollinger_upper"] = bb.bollinger_hband()
        df["bollinger_lower"] = bb.bollinger_lband()

        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        df["macd_diff"] = df["macd_diff"].fillna(0)

        df["rsi"] = RSIIndicator(close=df["close"], window=14, fillna=True).rsi()

        # ✅ Normalize RSI if necessary
        if df["rsi"].max() < 20 or df["rsi"].min() < -20:
            logger.warning("⚠️ RSI is standardized! Converting back to 0-100 range...")
            df["rsi"] = ((df["rsi"] - df["rsi"].min()) / (df["rsi"].max() - df["rsi"].min())) * 100

        df["adx"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14, fillna=True).adx()
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14, fillna=True).average_true_range()
        df["mfi"] = money_flow_index(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14)

        # ✅ Fill missing values
        df.fillna({"adx": 0, "atr": 0, "mfi": 0}, inplace=True)

        # ✅ Buy/Sell Signals
        df["buy_signal"] = (
            ((df["rsi"] < 50) & (df["macd_line"] > df["macd_signal"])) |
            ((df["adx"] > 15) & (df["macd_diff"] > 0)) |
            ((df["close"] < df["bollinger_lower"]) & (df["volume_surge"] > 1.1))
        ).astype(int)

        df["sell_signal"] = ((df["rsi"] > 70) & (df["macd_diff"] < 0)).astype(int)

        # ✅ Add sentiment score if missing
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0  # Placeholder for future sentiment data integration

        # ✅ Debugging: Print Columns After Processing
        logger.info(f"🔍 Features AFTER processing: {df.columns.tolist()}")

        # ✅ Feature Scaling for LSTM (Exclude `ticker`)
        scaler = StandardScaler()
        feature_columns = ["price_change", "volatility", "volume", "macd_diff", "adx", "atr", "mfi"]

        # ✅ Preserve ticker before scaling
        ticker_col = df["ticker"]

        # ✅ Fix Scaling Issue: Only scale if all required columns exist
        missing_scale_features = [f for f in feature_columns if f not in df.columns]
        if missing_scale_features:
            raise ValueError(f"❌ Missing features for scaling: {missing_scale_features}")

        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        # ✅ Restore ticker column after scaling
        df["ticker"] = ticker_col

        # ✅ Ensure `ticker` column is still present
        if "ticker" not in df.columns:
            raise ValueError("❌ ERROR: `ticker` column disappeared after preprocessing!")

        # ✅ Log ticker count after processing
        tickers_after = df["ticker"].unique()
        logger.info(f"📌 Unique tickers AFTER preprocessing: {len(tickers_after)}")

        logger.info("✅ Data successfully standardized for LSTM model.")

        # ✅ Log Buy Signal Distribution
        buy_signal_count = df["buy_signal"].sum()
        logger.info(f"📌 Total Buy Signals Detected: {buy_signal_count}")

        return df, scaler  # ✅ Ensure this function correctly returns a tuple (DataFrame, Scaler)

    except Exception as e:
        logger.error(f"❌ Error in preprocess_data_with_indicators: {e}")
        print(f"❌ Error in preprocess_data_with_indicators: {e}")  # ✅ Ensure error is visible
        raise

