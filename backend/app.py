from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from threading import Thread

# ✅ WebSocket & Flask SocketIO
import websocket
from flask_socketio import SocketIO

# ✅ Machine Learning & AI
from xgboost import XGBClassifier
from joblib import dump, load
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv1D, BatchNormalization, Dropout, Dense, LSTM, GlobalAveragePooling1D, 
    LeakyReLU, LayerNormalization, MultiHeadAttention, Bidirectional
)
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Model, Sequential  # type: ignore # ✅ FIXED Import Issue

# ✅ Deep Learning (LSTM)
from tensorflow.keras.models import Sequential, save_model, load_model  # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, 
    GlobalAveragePooling1D, LeakyReLU, LayerNormalization, MultiHeadAttention, AdditiveAttention  # type: ignore
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import money_flow_index  # ✅ Correct import from `ta.volume`


# ✅ Technical Indicators (TA)
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# ✅ Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Visualization
import matplotlib.pyplot as plt
import mplfinance as mpf

# ✅ Caching & Utility
from cachetools import TTLCache, cached

# Import utility functions
from utils.scheduler import initialize_scheduler
from utils.fetch_stock_performance import fetch_stock_performance
from utils.fetch_ticker_news import fetch_ticker_news
from utils.sentiment_plot import fetch_sentiment_trend, generate_sentiment_plot
from utils.realtime_tracking import track_stock_event, fetch_live_stock_data
from dotenv import load_dotenv  # ✅ Import dotenv

import logging



# ✅ Load environment variables from .env file
load_dotenv()
print(os.getenv("ALPHA_VANTAGE_API_KEY"))

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# Fix 'NoneType' object error in logging
logging.raiseExceptions = False  # Disable logging-related exceptions
logging.basicConfig(level=logging.INFO)  # Set default log level

# If using Flask logging, make sure it's initialized properly:
gunicorn_error_handlers = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_error_handlers.handlers
app.logger.setLevel(logging.INFO)

# ✅ Fix Gevent and Logging Conflict
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

tickers = set()
# Ensure models/ directory exists
if not os.path.exists("models"):
    os.makedirs("models")
# Cache for models
lstm_cache = {"model": None, "scaler": None}


# Polygon.io WebSocket URL (Delayed by 15 minutes)
POLYGON_WS_URL = "wss://delayed.polygon.io/stocks"
# Polygon.io API Key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "swpC4ge5_aGqdJll3gplZ6a40ADuwhzG")
if not POLYGON_API_KEY:
    raise ValueError("Polygon.io API key not found. Set POLYGON_API_KEY environment variable.")

# Get Alpha Vantage API Key from Environment
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "3R7BUV52GH1MOHNO")

if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("⚠️ Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY in .env file.")
latest_stock_prices = {}  # Store the latest stock prices
# Function to subscribe to tickers in WebSocket connection
def fetch_and_process_sentiment_data(ticker):
    """
    Fetch sentiment data for the given ticker from news sources and apply VADER sentiment analysis.
    """
    try:
        # Fetch news articles for the ticker
        news_data = fetch_ticker_news(ticker)

        # Analyze sentiment
        sentiment_scores = [analyzer.polarity_scores(article["title"])["compound"] for article in news_data]

        # Compute average sentiment score
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

        return avg_sentiment
    except Exception as e:
        print(f"❌ Error fetching sentiment data: {e}")
        return 0  # Default to neutral sentiment

def subscribe_to_tickers(ws):
    if tickers:
        tickers_list = ",".join(tickers)
        message = json.dumps({"action": "subscribe", "params": f"AM.{tickers_list}"})
        ws.send(message)
        print(f"📡 Subscribed to: {tickers_list}")

# WebSocket event handlers
def on_message(ws, message):
    data = json.loads(message)
    if isinstance(data, list):
        for event in data:
            if "sym" in event and "c" in event:
                stock_data = {"ticker": event["sym"], "price": event["c"]}
                latest_stock_prices[event["sym"]] = event["c"]  # Store the latest price
                socketio.emit("stock_update", stock_data)
                print(f"📊 Live Update: {stock_data}")

def on_error(ws, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed, reconnecting in 5 seconds...")
    threading.Timer(5, start_websocket_thread).start()

def on_open(ws):
    subscribe_to_tickers(ws)

# Start WebSocket connection
def start_websocket_thread():
    ws = websocket.WebSocketApp(
        f"{POLYGON_WS_URL}?apiKey={POLYGON_API_KEY}",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

# API to dynamically add tickers for live tracking
@app.route('/api/add_ticker', methods=['POST'])
def add_ticker():
    data = request.get_json()
    ticker = data.get("ticker")
    if ticker:
        global tickers
        tickers.add(ticker.upper())
        subscribe_to_tickers(start_websocket_thread())  # Ensure real-time updates
        return jsonify({"message": f"{ticker} added to live updates."}), 200
    return jsonify({"error": "Ticker not provided."}), 400

# API Route for Real-Time Stock Data
@app.route('/api/live-data', methods=['GET'])
def live_data():
    try:
        ticker = request.args.get("ticker")
        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400
        price = latest_stock_prices.get(ticker.upper(), "No data yet")
        return jsonify({"ticker": ticker, "price": price}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route('/api/historical-data', methods=['GET'])
def historical_data():
    """
    Fetch detailed historical intraday data for a selected stock using Alpha Vantage.
    """
    ticker = request.args.get("ticker")
    interval = request.args.get("interval", "5min")  # Default to 5-minute intervals

    if not ticker:
        return jsonify({"error": "Ticker parameter is missing"}), 400

    print(f"📊 Fetching detailed historical data for: {ticker}")

    # Fetch intraday data from Alpha Vantage
    df = fetch_historical_data(ticker, interval=interval, output_size="full")

    if df.empty:
        return jsonify({"error": "No historical data found"}), 404

    # Apply technical indicators
    df = preprocess_data_with_indicators(df)

    # Predict buy/sell signals
    df["buy_signal"] = (df["rsi"] < 30) & (df["macd_diff"] > 0)  # Example Buy Signal
    df["sell_signal"] = (df["rsi"] > 70) & (df["macd_diff"] < 0)  # Example Sell Signal

    # Format response
    response_data = {
        "dates": df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "buy_signals": df[df["buy_signal"]]["close"].tolist(),
        "sell_signals": df[df["sell_signal"]]["close"].tolist(),
    }

    return jsonify(response_data), 200


# Caching (TTLCache)
historical_data_cache = TTLCache(maxsize=10, ttl=300)

# Function to fetch historical data
def fetch_historical_data():
    """
    Fetch historical stock data from Polygon.io.
    """
    for i in range(14):  # Try fetching data for the last 14 days
        most_recent_date = datetime.utcnow() - timedelta(days=i)
        most_recent_date_str = most_recent_date.strftime("%Y-%m-%d")
        print(f"🔍 Attempting to fetch stock data for: {most_recent_date_str}")

        url = (
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
            f"{most_recent_date_str}?adjusted=true&apiKey={POLYGON_API_KEY}"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])  # Convert JSON to DataFrame

                # ✅ Rename columns to match expected format
                rename_mapping = {
                    "v": "volume",  # Fix volume column
                    "o": "o",       # Open price
                    "c": "c",       # Close price
                    "h": "h",       # High price
                    "l": "l",       # Low price
                }
                df.rename(columns=rename_mapping, inplace=True)

                # ✅ Debugging Output
                print("📌 Raw Data Fetched from API:")
                print(df.head(5))  # Print the first 5 rows
                print("📌 Columns in DataFrame:", df.columns.tolist())  # Print column names

                return df  # Return DataFrame

            print(f"⚠️ No stock data found for {most_recent_date_str}")

        except requests.exceptions.Timeout:
            print(f"❌ Timeout error while fetching data for {most_recent_date_str}")

        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error: {http_err}")

        except requests.exceptions.RequestException as req_err:
            print(f"❌ Request error: {req_err}")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    print("❌ Unable to fetch stock data. Returning empty DataFrame.")
    return pd.DataFrame()


def fetch_alpha_historical_data(ticker, interval="5min", output_size="full"):
    """
    Fetch historical stock data from Alpha Vantage and ensure data consistency.
    """
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not ALPHA_VANTAGE_API_KEY:
        print("❌ ERROR: Alpha Vantage API key is missing.")
        return pd.DataFrame()

    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}&interval={interval}&outputsize={output_size}&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        print(f"📊 Raw API Response Keys: {list(data.keys())}")  # Debugging

        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            print(f"⚠️ No historical data found for {ticker}. Response: {data}")
            return pd.DataFrame()

        records = data[time_series_key]

        # ✅ Convert JSON to DataFrame
        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.to_datetime(df.index)

        print(f"📊 Raw DataFrame Columns Before Renaming: {df.columns.tolist()}")  # Debugging

        # ✅ Rename columns correctly
        rename_mapping = {
            "1. open": "o",
            "2. high": "h",
            "3. low": "l",
            "4. close": "c",
            "5. volume": "volume"
        }

        df.rename(columns=rename_mapping, inplace=True)

        # ✅ Debugging: Check if volume column exists
        if "v" not in df.columns:
            print("❌ ERROR: Volume column ('5. volume') was not correctly renamed!")
            print(f"Current columns: {df.columns.tolist()}")  # Print column names for debugging

        print(f"📊 Sample Row After Renaming:\n{df.head(1)}")  # Print one row for validation

        # ✅ Convert data types to float
        try:
            df = df.astype(float)
        except ValueError as e:
            print(f"❌ Data type conversion error: {e}")
            print(f"📌 Current DataFrame:\n{df.head()}")  # Debugging output

        print(f"✅ {ticker} historical data fetched and formatted successfully.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching Alpha Vantage data for {ticker}: {e}")
        return pd.DataFrame()
# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]
# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def money_flow_index(high, low, close, volume, window=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()

    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def preprocess_data_with_indicators(df):
    df = df.copy()

    # ✅ Ensure required columns exist
    required_cols = ["o", "c", "h", "l", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"❌ Missing required columns in dataset: {required_cols}")

    # ✅ Price Change & Volatility
    df["price_change"] = (df["c"] - df["o"]) / df["o"]
    df["volatility"] = (df["h"] - df["l"]) / df["l"]

    # ✅ Volume Surge Calculation
    df["volume_surge"] = df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()

    # ✅ On-Balance Volume (OBV)
    df["obv"] = OnBalanceVolumeIndicator(close=df["c"], volume=df["volume"], fillna=True).on_balance_volume()

    # ✅ Williams %R
    df["williams_r"] = WilliamsRIndicator(high=df["h"], low=df["l"], close=df["c"], lbp=14, fillna=True).williams_r()

    # ✅ Exponential Moving Averages (EMA 12, 26 for MACD)
    df["ema_12"] = EMAIndicator(close=df["c"], window=12, fillna=True).ema_indicator()
    df["ema_26"] = EMAIndicator(close=df["c"], window=26, fillna=True).ema_indicator()

    # ✅ Bollinger Bands
    bb = BollingerBands(close=df["c"], window=20, fillna=True)
    df["bollinger_upper"] = bb.bollinger_hband()
    df["bollinger_lower"] = bb.bollinger_lband()

    # ✅ MACD Calculation
    macd = MACD(close=df["c"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_diff"] = df["macd_line"] - df["macd_signal"]
    df["macd_diff"] = df["macd_diff"].fillna(0)

    # ✅ RSI Calculation
    df["rsi"] = RSIIndicator(close=df["c"], window=14, fillna=True).rsi()

    # ✅ Volume Weighted Average Price (VWAP)
    df["vwap"] = (df["volume"] * (df["h"] + df["l"] + df["c"]) / 3).cumsum() / df["volume"].cumsum()

    # ✅ ADX Calculation
    df["adx"] = ADXIndicator(high=df["h"], low=df["l"], close=df["c"], window=14, fillna=True).adx()

    # ✅ ATR Calculation
    df["atr"] = AverageTrueRange(high=df["h"], low=df["l"], close=df["c"], window=14, fillna=True).average_true_range()

    # ✅ Money Flow Index (MFI)
    df["mfi"] = money_flow_index(high=df["h"], low=df["l"], close=df["c"], volume=df["volume"], window=14)
    df["mfi"].fillna(0, inplace=True)  # Ensure no NaN values

    # ✅ Buy/Sell Signals
    df["buy_signal"] = ((df["rsi"] < 30) & (df["macd_line"] > df["macd_signal"]) & (df["adx"] > 20)).astype(int)
    df["sell_signal"] = ((df["rsi"] > 70) & (df["macd_line"] < df["macd_signal"]) & (df["adx"] > 20)).astype(int)

    # ✅ Ensure Sentiment Score is included and properly computed
    if "news_headline" in df.columns:
        df["sentiment_score"] = df["news_headline"].apply(analyze_sentiment)
    else:
        df["sentiment_score"] = 0  # ✅ Default to 0 if news data is missing

    # ✅ Final Check: Ensure all required columns exist
    required_features = ["adx", "atr", "mfi", "buy_signal", "sell_signal", "sentiment_score"]
    for feature in required_features:
        if feature not in df.columns:
            print(f"⚠️ Warning: {feature} not found. Adding default values.")
            df[feature] = 0  # Default value for missing indicators

    print("📌 Columns After Processing:", df.columns.tolist())
    return df


def analyze_sentiment(text):
    """
    Extract sentiment score from text (Financial News, Twitter, Reddit).
    """
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]
import numpy as np

def detect_breakouts(data, window=20, threshold=1.02):
    """
    Identify breakout trading opportunities based on price action and volume.
    
    - Looks for price breaking above recent highs.
    - Uses volume surge to confirm breakouts.
    
    Params:
    - data (DataFrame): Stock data with OHLC & indicators.
    - window (int): Number of previous candles for resistance.
    - threshold (float): Percentage above resistance for breakout confirmation.

    Returns:
    - DataFrame with "breakout" signals (1 for breakout, 0 otherwise)
    """
    data["prev_high"] = data["h"].rolling(window=window).max().shift(1)
    data["breakout"] = np.where(
        (data["c"] > data["prev_high"] * threshold) & (data["volume"] > data["volume"].rolling(window=5).mean()),
        1, 0
    )

    return data


def generate_trade_signals(data):
    """
    Generate buy/sell signals using a combination of indicators.
    
    - Buy when: RSI < 30, MACD crosses up, Volume Surge, Breakout detected
    - Sell when: RSI > 70, MACD crosses down, ATR shows high volatility
    
    Returns:
    - DataFrame with "buy_signal" & "sell_signal"
    """
    data["buy_signal"] = (
        (data["rsi"] < 30) &  # Oversold
        (data["macd_line"] > data["macd_signal"]) &  # Bullish MACD crossover
        (data["volume_surge"] > 1.2) &  # High volume move
        (data["breakout"] == 1)  # Confirmed breakout
    ).astype(int)

    data["sell_signal"] = (
        (data["rsi"] > 70) &  # Overbought
        (data["macd_line"] < data["macd_signal"]) &  # Bearish MACD crossover
        (data["atr"] > data["atr"].rolling(14).mean())  # Volatility surge
    ).astype(int)

    return data
def plot_candlestick_chart(data, ticker):
    """
    Plot candlestick chart with AI buy/sell signals.
    """
    buy_signals = data[data["buy_signal"] == 1]
    sell_signals = data[data["sell_signal"] == 1]

    fig, ax = plt.subplots(figsize=(12, 6))

    # ✅ Candlestick Chart
    mpf.plot(data, type="candle", ax=ax, volume=True)

    # ✅ Highlight Buy Signals
    ax.scatter(buy_signals.index, buy_signals["c"], color="green", label="BUY", marker="^", alpha=1, s=100)

    # ✅ Highlight Sell Signals
    ax.scatter(sell_signals.index, sell_signals["c"], color="red", label="SELL", marker="v", alpha=1, s=100)

    # ✅ Display Trendlines
    ax.set_title(f"{ticker} - AI Trading Signals")
    ax.legend()
    plt.show()

# Train XGBoost model
def train_xgboost_model():
    try:
        # Fetch and preprocess data
        data = fetch_historical_data()
        data = preprocess_data_with_indicators(data)

        # Define the feature set dynamically
        features = [
            "price_change", "volatility", "volume", "volume_surge",
            "rsi", "macd_line", "macd_signal", "stochastic",
            "adx", "atr", "mfi", "obv", "ema_12", "ema_26",
            "bollinger_upper", "bollinger_lower", "vwap",
            "breakout", "buy_signal", "sell_signal"
        ]

        # Filter only valid features
        features = [col for col in features if col in data.columns]

        if not features:
            raise ValueError("❌ No valid features available for training the model.")

        # Define target
        data["target"] = (data["h"] >= data["c"] * 1.05).astype(int)  # Example target condition

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data["target"], test_size=0.2, random_state=42
        )

        # ✅ Train XGBoost Model with Feature Selection
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # ✅ Evaluate Feature Importance
        feature_importance = model.feature_importances_
        feature_ranking = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)
        print("📊 Feature Importance Ranking:")
        for feature, importance in feature_ranking:
            print(f"{feature}: {importance:.4f}")

        # ✅ Print Model Evaluation Metrics
        print("📊 Model Evaluation Report:")
        print(classification_report(y_test, model.predict(X_test)))

        # ✅ Save Model
        dump(model, "models/xgb_model.joblib")
        print("✅ XGBoost model trained and saved successfully.")

        return model, features

    except Exception as e:
        print(f"❌ Error in train_xgboost_model: {e}")
        raise
# Function to preprocess data with enhanced indicators
# After fetch_historical_data
def train_and_cache_lstm_model():
    """
    Train the LSTM model and cache it for future use.
    """
    try:
        # Fetch and preprocess historical data
        data = fetch_historical_data()
        data = preprocess_data_with_indicators(data)
        
        # ✅ Debug: Check available columns before training
        print("📌 Columns Available in Data Before LSTM Training:", data.columns.tolist())

        if "macd_diff" not in data.columns:
            raise ValueError(f"❌ ERROR: 'macd_diff' column is missing before LSTM training! Available columns: {data.columns.tolist()}")

        # Define features and target
        features = ["price_change", "volatility", "volume", "sentiment_score", "macd_diff"]  # Ensure macd_diff is included

        # Define features and target
        features = ["price_change", "volatility", "volume", "rsi", "macd_line", "macd_signal", "macd_diff"]

        target = "c"  # Target column (e.g., closing price)

        # Train the LSTM model
        model, scaler = train_cnn_lstm_model(data, features, target)

        # Ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # ✅ Save Model & Scaler
        lstm_model_path = os.path.join(models_dir, "lstm_model.keras")
        scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")

        save_model(model, lstm_model_path)  # Save model in new Keras format
        joblib.dump(scaler, scaler_path)

        # ✅ Debug: Print Paths
        print(f"✅ Model saved at: {lstm_model_path}")
        print(f"✅ Scaler saved at: {scaler_path}")

        # ✅ Verify If Files Exist
        if os.path.exists(lstm_model_path) and os.path.exists(scaler_path):
            print("✅ LSTM model and scaler successfully saved in the models/ directory.")
        else:
            print("❌ ERROR: Model files are missing even after saving!")

        # ✅ Cache Model
        lstm_cache["model"] = model
        lstm_cache["scaler"] = scaler

        return model, scaler

    except Exception as e:
        print(f"❌ Error training and saving LSTM model: {e}")
        raise
# Load XGBoost model if it exists, otherwise train it
# Load LSTM model if it exists, otherwise train it
lstm_model_path = "C:\\Users\\gabby\\trax-x\\backend\\models\\lstm_model.keras"
scaler_path = "C:\\Users\\gabby\\trax-x\\backend\\models\\lstm_scaler.pkl"

if os.path.exists(lstm_model_path) and os.path.exists(scaler_path):
    try:
        lstm_cache["model"] = load_model(lstm_model_path)
        lstm_cache["scaler"] = joblib.load(scaler_path)
        print("✅ LSTM model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR loading LSTM model: {e}")
        lstm_cache["model"], lstm_cache["scaler"] = train_and_cache_lstm_model()
else:
    print("⚠️ LSTM model not found. Training a new one...")
    lstm_cache["model"], lstm_cache["scaler"] = train_and_cache_lstm_model()
@app.route('/api/alpha-historical-data', methods=['GET'])
def alpha_historical_data():
    """
    Fetch historical data for a selected stock from Alpha Vantage.
    """
    ticker = request.args.get("ticker")
    interval = request.args.get("interval", "5min")  # Default to 5-minute intervals

    if not ticker:
        return jsonify({"error": "Ticker parameter is missing"}), 400

    print(f"📊 Fetching historical data from Alpha Vantage for: {ticker}")

    # Fetch intraday data from Alpha Vantage
    df = fetch_alpha_historical_data(ticker, interval=interval, output_size="full")

    if df.empty:
        return jsonify({"error": "No historical data found"}), 404

    # Debugging: Check before processing
    print(f"📊 Columns in DataFrame Before Processing: {df.columns.tolist()}")

    # Apply technical indicators
    df = preprocess_data_with_indicators(df)

    # Debugging: Check after processing
    print(f"📊 Columns in DataFrame After Processing: {df.columns.tolist()}")

    # ✅ Fix: Use "volume" instead of "v"
    response_data = {
        "dates": df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        "open": df["o"].tolist(),
        "high": df["h"].tolist(),
        "low": df["l"].tolist(),
        "close": df["c"].tolist(),
        "volume": df["volume"].tolist()  # Fix applied here
    }

    return jsonify(response_data), 200

# Define API routes below
@app.route('/api/train-xgboost', methods=['POST'])
def train_xgboost_endpoint():
    """
    API endpoint to manually train the XGBoost model.
    """
    try:
        global xgb_model, feature_columns
        xgb_model, feature_columns = train_xgboost_model()
        return jsonify({"message": "XGBoost model trained and saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/scan-stocks', methods=['GET'])
def scan_stocks():
    try:
        # ✅ Parse user inputs
        min_price = float(request.args.get("min_price", 0))
        max_price = float(request.args.get("max_price", float("inf")))
        volume_surge = float(request.args.get("volume_surge", 1.2))
        min_rsi = float(request.args.get("min_rsi", 0))
        max_rsi = float(request.args.get("max_rsi", 100))

        print(f"📌 Scan Stocks Params: min_price={min_price}, max_price={max_price}, volume_surge={volume_surge}, min_rsi={min_rsi}, max_rsi={max_rsi}")

        # ✅ Fetch historical stock data
        data = fetch_historical_data()
        if data.empty:
            print("⚠️ No stock data available!")
            return jsonify({"error": "No stock data available"}), 404
        
        print("📌 Raw Data Before Preprocessing:", data.head())
        print("📌 Columns Before Indicators:", data.columns.tolist())

        # ✅ Apply technical indicators (Bollinger, MACD, RSI, ATR, ADX, Sentiment Score)
        data = preprocess_data_with_indicators(data)

        print("📌 Data After Applying Indicators:", data.head())
        print("📌 Columns After Indicators:", data.columns.tolist())

        # ✅ Ensure all required columns exist (handling missing columns)
        required_columns = [
            "adx", "atr", "mfi", "buy_signal", "sell_signal", "c", "volume_surge", 
            "rsi", "sentiment_score", "bollinger_upper", "bollinger_lower", "macd_hist"
        ]
        for col in required_columns:
            if col not in data.columns:
                print(f"⚠️ WARNING: {col} missing, adding default values.")
                data[col] = 0  # Fill missing columns with neutral values

        print("📌 Columns Before Filtering:", data.columns.tolist())

        # ✅ Apply user-defined filters
        filtered_data = data[
            (data["c"] >= min_price) & 
            (data["c"] <= max_price) & 
            (data["volume_surge"] > volume_surge) & 
            (data["rsi"] >= min_rsi) & 
            (data["rsi"] <= max_rsi)
        ].copy()  # Avoid SettingWithCopyWarning by explicitly copying data

        print("📌 Data After Filtering:", filtered_data.head())

        if filtered_data.empty:
            print("⚠️ No data matching filters!")
            return jsonify({"candidates": []}), 200

        # ✅ Step 1: Apply XGBoost Predictions
        filtered_data["xgboost_prediction"] = xgb_model.predict(filtered_data[feature_columns])
        xgb_filtered_data = filtered_data[filtered_data["xgboost_prediction"] == 1].copy()
        print("📌 Data After XGBoost Filtering:", xgb_filtered_data.head())

        if xgb_filtered_data.empty:
            print("⚠️ No data matching XGBoost predictions!")
            return jsonify({"candidates": []}), 200

        # ✅ Ensure LSTM is trained and cached
        if not lstm_cache["model"] or not lstm_cache["scaler"]:
            print("❌ ERROR: LSTM model is not trained. Please train using /api/train-lstm before scanning stocks.")
            return jsonify({"error": "LSTM model is not trained. Please train using /api/train-lstm before scanning stocks."}), 500

        # ✅ Ensure we have enough data for LSTM predictions
        if len(xgb_filtered_data) < 50:
            print(f"⚠️ Insufficient data for LSTM prediction: {len(xgb_filtered_data)} rows.")
            return jsonify({"candidates": xgb_filtered_data.head(20).to_dict(orient="records")}), 200

        # ✅ Step 3: Apply LSTM Predictions
        lstm_features = [
            "price_change", "volatility", "volume", "sentiment_score"
        ]
        
        xgb_filtered_data["next_day_prediction"] = xgb_filtered_data.apply(
            lambda row: predict_next_day(
                model=lstm_cache["model"],
                recent_data=xgb_filtered_data,
                scaler=lstm_cache["scaler"],
                features=lstm_features
            ),
            axis=1
        )
        print("📌 Data with LSTM Predictions:", xgb_filtered_data.head())

        # ✅ Cap Predictions to Prevent Extreme Spikes
        max_increase = 1.5  # Max 50% increase
        xgb_filtered_data["next_day_prediction"] = xgb_filtered_data["next_day_prediction"].clip(
            upper=xgb_filtered_data["c"] * max_increase
        )
        
        print("📌 Capped Next Day Predictions:", xgb_filtered_data[["T", "c", "next_day_prediction"]].head())

        # ✅ Step 4: Combine Predictions with Weighted Scores
        xgb_weight = 0.7  # Weight for XGBoost
        lstm_weight = 0.3  # Weight for LSTM
        xgb_filtered_data["combined_score"] = (
            (xgb_weight * xgb_filtered_data["xgboost_prediction"]) +
            (lstm_weight * (xgb_filtered_data["next_day_prediction"] / xgb_filtered_data["c"]))
        )
        print("📌 Data with Combined Scores:", xgb_filtered_data.head())

        # ✅ Step 5: Sort and Limit Results
        top_candidates = xgb_filtered_data.sort_values("combined_score", ascending=False).head(20)
        print("📌 Top 20 Candidates:", top_candidates)

        # ✅ Return filtered candidates
        return jsonify({"candidates": top_candidates.to_dict(orient="records")}), 200

    except Exception as e:
        print(f"❌ ERROR in scan-stocks endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Function to preprocess data for LSTM with additional features
from ta.momentum import RSIIndicator

def preprocess_for_lstm(data, features, target, time_steps=150):
    # ✅ Ensure required features exist in DataFrame
    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        print(f"❌ ERROR: Missing features in data: {missing_features}")
        raise ValueError(f"Missing columns: {missing_features}")

    # ✅ Ensure RSI is computed before smoothing
    if "rsi" not in data.columns:
        rsi_indicator = RSIIndicator(close=data["c"], window=14, fillna=True)
        data["rsi"] = rsi_indicator.rsi()

    # ✅ Ensure MACD is computed before smoothing
    if "macd_diff" not in data.columns:
        print("⚠️ Warning: `macd_diff` not found! Ensure it is included in `preprocess_data_with_indicators()`.")

    # ✅ Compute rolling mean for smoothing
    data["sentiment_score_avg"] = data["sentiment_score"].rolling(window=10).mean().fillna(0)
    data["rsi_smooth"] = data["rsi"].rolling(window=5).mean().fillna(0)

    # ✅ Ensure macd_diff is in DataFrame before computing smoothed version
    if "macd_diff" in data.columns:
        data["macd_smooth"] = data["macd_diff"].rolling(window=5).mean().fillna(0)
    else:
        data["macd_smooth"] = 0  # Default if missing

    # ✅ Print debug information
    print("📌 Features Sent to LSTM:", features)
    print("📌 Columns Available in Data:", data.columns.tolist())
    # ✅ Ensure "macd_diff" exists before proceeding
    if "macd_diff" not in data.columns:
        raise ValueError(f"❌ ERROR: 'macd_diff' column is missing before LSTM training! Available columns: {data.columns.tolist()}")


    # ✅ Normalize features using StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i])
        y.append(data[target].values[i])  # 🔹 Fix `iloc` issue

    return np.array(X), np.array(y), scaler

# Optimized LSTM Model with Attention, CNN, and deeper architecture
def train_cnn_lstm_model(data, features, target, time_steps=150):
    """
    Train a CNN-LSTM model for stock price prediction while keeping all existing functionality.
    Uses CNN for feature extraction & LSTM for sequential learning.
    """

    models_dir = "C:\\Users\\gabby\\trax-x\\models"

    # ✅ Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    print("✅ Models directory ensured.")

    # ✅ Ensure `macd_diff` exists OR remove it if not needed
    if "macd_diff" not in data.columns:
        print("⚠️ Warning: 'macd_diff' column is missing! Proceeding without it.")
        features = [f for f in features if f != "macd_diff"]  # Remove macd_diff safely

    # ✅ Preprocess Data
    X, y, scaler = preprocess_for_lstm(data, features, target, time_steps)

    # ✅ Define Input Layer
    input_layer = Input(shape=(X.shape[1], X.shape[2]))

    # ✅ CNN Feature Extraction
    cnn_layer = Conv1D(filters=128, kernel_size=3, activation=LeakyReLU(alpha=0.1))(input_layer)
    cnn_layer = BatchNormalization()(cnn_layer)
    cnn_layer = Dropout(0.3)(cnn_layer)

    # ✅ Attention Mechanism
    attention_layer = MultiHeadAttention(num_heads=4, key_dim=64)(cnn_layer, cnn_layer)  # Fixed
    attention_layer = LayerNormalization()(attention_layer)

    # ✅ LSTM Layers
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(attention_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Dropout(0.3)(lstm_layer)

    lstm_layer = LSTM(64, return_sequences=True)(lstm_layer)
    lstm_layer = GlobalAveragePooling1D()(lstm_layer)

    # ✅ Fully Connected Dense Layers
    dense_layer = Dense(64, activation=LeakyReLU(alpha=0.1))(lstm_layer)
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(32, activation="swish")(dense_layer)
    output_layer = Dense(1)(dense_layer)

    # ✅ Define & Compile Model (Fixed)
    model = Model(inputs=input_layer, outputs=output_layer)  # ✅ FIXED `Model` issue
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error")

    # ✅ Callbacks for Early Stopping & Learning Rate Reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # ✅ Train Model
    model.fit(X, y, epochs=300, batch_size=128, validation_split=0.2, verbose=1,
              callbacks=[early_stopping, reduce_lr])

    # ✅ Save Model & Scaler
    lstm_model_path = os.path.join(models_dir, "cnn_lstm_model.keras")
    scaler_path = os.path.join(models_dir, "cnn_lstm_scaler.pkl")

    save_model(model, lstm_model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Model saved at: {lstm_model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")

    return model, scaler

# Function to predict the next day using LSTM
def predict_next_day(model, recent_data, scaler, features):
    """
    Predict the next day's value using the LSTM model.
    Args:
    - model: Trained LSTM model.
    - recent_data: DataFrame containing recent historical data.
    - scaler: Fitted MinMaxScaler object.
    - features: List of feature columns required by the model.

    Returns:
    - Predicted value for the next day.
    """
    # Ensure enough data for 50 time steps
    if len(recent_data) < 50:
     print(f"⚠️ Warning: Only {len(recent_data)} rows available. Proceeding with all available data.")
     recent_data = pad_sequences(recent_data, required_length=50)


    # Scale the recent data
    recent_scaled = scaler.transform(recent_data[features].values[-50:])

    # Reshape the data for LSTM (batch size = 1, time steps = 50, features = len(features))
    reshaped_data = recent_scaled.reshape(1, 50, len(features))

    # Make prediction
    prediction = model.predict(reshaped_data)[0][0]
    return prediction


# Function to preprocess data with enhanced indicators
# After fetch_historical_data
def train_and_cache_lstm_model(df):
    """
    Train the LSTM model and cache it for future use.
    """
    try:
        # Fetch and preprocess historical data
        data = fetch_historical_data()
        print(f"📌 Columns After Processing: {df.columns.tolist()}")  # Debug

        data = preprocess_data_with_indicators(data)

        # Define features and target
        features = ["price_change", "volatility", "volume", "rsi", "macd_line", "macd_signal", "macd_diff"]

        target = "c"  # Target column (e.g., closing price)

        # Train the LSTM model
        model, scaler = train_cnn_lstm_model(data, features, target)

        # Ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # ✅ Save Model & Scaler
        lstm_model_path = os.path.join(models_dir, "lstm_model.keras")
        scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")

        save_model(model, lstm_model_path)  # Save model in new Keras format
        joblib.dump(scaler, scaler_path)

        # ✅ Debug: Print Paths
        print(f"✅ Model saved at: {lstm_model_path}")
        print(f"✅ Scaler saved at: {scaler_path}")

        # ✅ Verify If Files Exist
        if os.path.exists(lstm_model_path) and os.path.exists(scaler_path):
            print("✅ LSTM model and scaler successfully saved in the models/ directory.")
        else:
            print("❌ ERROR: Model files are missing even after saving!")

        # ✅ Cache Model
        lstm_cache["model"] = model
        lstm_cache["scaler"] = scaler

        return model, scaler

    except Exception as e:
        print(f"❌ Error training and saving LSTM model: {e}")
        raise
@app.route('/api/train-lstm', methods=['POST'])
def train_lstm_endpoint():
    """
    API endpoint to train the CNN-LSTM model and cache it.
    Ensures proper data preprocessing, model saving, and logging.
    """
    try:
        print("📌 Fetching historical stock data...")
        data = fetch_historical_data()

        # ✅ Compute sentiment scores if needed
        if "T" in data.columns:
            data["sentiment_score"] = data["T"].apply(fetch_and_process_sentiment_data)
        else:
            print("⚠️ 'T' column missing, skipping sentiment analysis.")
            data["sentiment_score"] = 0

        print("📌 Preprocessing data with indicators...")
        data = preprocess_data_with_indicators(data)

        # ✅ Ensure `macd_diff` is explicitly computed
        if "macd_line" in data.columns and "macd_signal" in data.columns:
            data["macd_diff"] = data["macd_line"] - data["macd_signal"]
            data["macd_diff"] = data["macd_diff"].fillna(0)  # Handle missing values

        # ✅ Debugging: Check columns before training
        print("📌 Columns Available Before Training:", data.columns.tolist())

        # ✅ Define features list dynamically based on availability
        available_features = set(data.columns.tolist())
        base_features = ["price_change", "volatility", "volume", "rsi", "macd_line", "macd_signal"]
        if "macd_diff" in available_features:
            base_features.append("macd_diff")

        features = [f for f in base_features if f in available_features]  # Filter missing ones
        target = "c"  # Predict closing price

        print(f"📌 Using Features: {features}")
        print("📌 Training CNN-LSTM model...")

        # ✅ Train the CNN-LSTM model
        model, scaler = train_cnn_lstm_model(data, features, target)

        # ✅ Ensure models directory exists before saving
        models_dir = "C:\\Users\\gabby\\trax-x\\models"
        os.makedirs(models_dir, exist_ok=True)

        # ✅ Save Model & Scaler
        lstm_model_path = os.path.join(models_dir, "cnn_lstm_model.keras")
        scaler_path = os.path.join(models_dir, "cnn_lstm_scaler.pkl")

        save_model(model, lstm_model_path)
        joblib.dump(scaler, scaler_path)

        print(f"✅ Model saved at: {lstm_model_path}")
        print(f"✅ Scaler saved at: {scaler_path}")

        return jsonify({"message": "CNN-LSTM model trained successfully!"}), 200

    except Exception as e:
        print(f"❌ Error in LSTM training: {e}")
        return jsonify({"error": str(e)}), 500

# Train both models
data = fetch_historical_data()
xgb_model, feature_columns = train_xgboost_model()
# lstm_model, lstm_scaler = train_lstm_model(
#     data, 
#     features=["price_change", "volatility", "volume", "sentiment_score"], 
#     target="c"
# )

# API to predict using LSTM
@app.route('/api/lstm-predict', methods=['GET'])
def lstm_predict():
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400

        # Get latest price from WebSocket updates
        price = latest_stock_prices.get(ticker.upper(), None)
        if price is None:
            return jsonify({"error": "No live data available yet for this ticker"}), 404

        # Ensure LSTM model and scaler are loaded
        if not lstm_cache["model"] or not lstm_cache["scaler"]:
            raise ValueError("LSTM model is not initialized. Please initialize the model via the scan-stocks route.")

        # Fetch historical data
        df = fetch_alpha_historical_data(ticker)

        if df.empty:
            return jsonify({"error": "No historical data available"}), 404

        # Preprocess data (Ensure alignment with Trading Charts)
        df = preprocess_data_with_indicators(df)

        # Extract relevant features
        features = ["price_change", "volatility", "volume", "rsi", "macd_line", "macd_signal", "ema_12", "ema_26", "vwap"]
        X = df[features]

        # **LSTM Prediction: Using Last 50 Time Steps**
        recent_data = X.values[-50:].reshape(1, 50, len(features))
        prediction = lstm_cache["model"].predict(recent_data)[0][0]

        return jsonify({"ticker": ticker, "next_day_prediction": prediction}), 200
    except Exception as e:
        print(f"❌ Error in lstm-predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# Start WebSocket thread
websocket_thread = threading.Thread(target=start_websocket_thread, daemon=True)
websocket_thread.start()
def preprocess_data_with_indicators(data):
    """
    Add volume, sentiment score, and advanced technical indicators for better predictions.
    """
    try:
        # Rename volume column for consistency
        data.rename(columns={"v": "volume"}, inplace=True)

        # Price Change & Volatility
        data["price_change"] = (data["c"] - data["o"]) / data["o"]
        data["volatility"] = (data["h"] - data["l"]) / data["l"]

        # Volume Surge Calculation
        data["volume_surge"] = data["volume"] / data["volume"].rolling(window=5).mean()

        # ✅ On-Balance Volume (OBV)
        obv_indicator = OnBalanceVolumeIndicator(close=data["c"], volume=data["volume"], fillna=True)
        data["obv"] = obv_indicator.on_balance_volume()

        # ✅ Williams %R
        williams_r = WilliamsRIndicator(high=data["h"], low=data["l"], close=data["c"], lbp=14, fillna=True)
        data["williams_r"] = williams_r.williams_r()

        # ✅ Exponential Moving Averages (EMA 12, 26 for MACD)
        data["ema_12"] = EMAIndicator(close=data["c"], window=12, fillna=True).ema_indicator()
        data["ema_26"] = EMAIndicator(close=data["c"], window=26, fillna=True).ema_indicator()

        # ✅ Bollinger Bands
        bb = BollingerBands(close=data["c"], window=20, fillna=True)
        data["bollinger_upper"] = bb.bollinger_hband()
        data["bollinger_lower"] = bb.bollinger_lband()

        # ✅ MACD Calculation
        macd = MACD(close=data["c"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        data["macd_line"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        data["macd_hist"] = macd.macd_diff()
        

        # ✅ RSI Calculation
        data["rsi"] = RSIIndicator(close=data["c"], window=14, fillna=True).rsi()

        # ✅ Volume Weighted Average Price (VWAP)
        data["vwap"] = (data["volume"] * (data["h"] + data["l"] + data["c"]) / 3).cumsum() / data["volume"].cumsum()

        # Handle Missing Values
        data.fillna(0, inplace=True)

        return data

    except Exception as e:
        print(f"❌ Error in preprocess_data_with_indicators: {e}")
        raise

    # Debug output
    print("Data after adding indicators:", data.head())
    return data

@app.route('/api/candlestick', methods=['GET'])
def candlestick_chart():
    try:
        # Get ticker parameter
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400

        print(f"Fetching candlestick data for ticker: {ticker}")

        # Define date range (last 180 days)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=180)

        # Construct API request URL
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?"
            f"adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
        )

        # Fetch data from Polygon API
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # If no data available, return an empty response (previous behavior)
        if "results" not in data or not data["results"]:
            print(f"Warning: No candlestick data found for ticker {ticker}. Returning empty response.")
            return jsonify({
                "dates": [],
                "open": [],
                "high": [],
                "low": [],
                "close": []
            }), 200  # Ensures frontend does not break

        # Convert results to DataFrame
        results = pd.DataFrame(data["results"])

        # Ensure required columns exist; if missing, default to empty lists
        return jsonify({
            "dates": results["t"].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d')).tolist() if "t" in results else [],
            "open": results["o"].tolist() if "o" in results else [],
            "high": results["h"].tolist() if "h" in results else [],
            "low": results["l"].tolist() if "l" in results else [],
            "close": results["c"].tolist() if "c" in results else [],
        }), 200

    except requests.exceptions.Timeout:
        print(f"Timeout while fetching data for {ticker}")
        return jsonify({"error": "External API request timed out"}), 504

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return jsonify({"error": "External API error"}), 500

    except Exception as e:
        print(f"Unexpected error processing ticker {ticker}: {e}")
        return jsonify({"error": "Internal server error"}), 500

        # Fill any missing values with defaults
        results.fillna(0, inplace=True)

        # Format and return candlestick data
        return jsonify({
            "dates": results["t"].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d')).tolist(),
            "open": results["o"].tolist(),
            "high": results["h"].tolist(),
            "low": results["l"].tolist(),
            "close": results["c"].tolist(),
        }), 200

    except requests.exceptions.Timeout:
        print(f"Timeout occurred while fetching data for ticker: {ticker}")
        return jsonify({"error": "Request to external API timed out"}), 504

    except requests.exceptions.RequestException as e:
        print(f"Request error for ticker {ticker}: {e}")
        return jsonify({"error": "Error fetching data from external API"}), 500

    except Exception as e:
        print(f"Unexpected error for ticker {ticker}: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/ticker-news", methods=["GET"])
def ticker_news():
    tickers = request.args.get("ticker")  # Expect comma-separated tickers
    if not tickers:
        return jsonify({"error": "Ticker is required"}), 400

    ticker_list = tickers.split(",")  # Split tickers into a list
    all_news = {}

    for ticker in ticker_list:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={POLYGON_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            all_news[ticker] = response.json().get("results", [])
        except requests.exceptions.HTTPError as e:
            all_news[ticker] = {"error": f"Error fetching news for {ticker}: {str(e)}"}
        except Exception as e:
            all_news[ticker] = {"error": f"Unexpected error: {str(e)}"}

    return jsonify(all_news)  # Return news grouped by ticker
@app.route('/api/sentiment-plot', methods=['GET'])
def sentiment_plot():
    """
    API endpoint to fetch sentiment trends and reasoning for a ticker within a date range.
    """
    try:
        ticker = request.args.get("ticker")
        start_date = request.args.get("start_date", (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d"))
        end_date = request.args.get("end_date", datetime.today().strftime("%Y-%m-%d"))

        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400

        # Fetch sentiment data
        sentiment_data = fetch_sentiment_trend(ticker, start_date, end_date)

        if sentiment_data.empty:
            return jsonify({"error": "No sentiment data available for this ticker"}), 404

        # Optional: Extract reasoning from insights
        sentiment_reasons = []
        for day in sentiment_data.itertuples():
            daily_reason = {
                "date": day.date,
                "reasons": []
            }
            # Add sentiment reasoning if available
            for insight in getattr(day, 'insights', []):
                daily_reason["reasons"].append({
                    "sentiment": insight.sentiment,
                    "reasoning": insight.sentiment_reasoning,
                })
            sentiment_reasons.append(daily_reason)

        return jsonify({
            "dates": sentiment_data['date'].tolist(),
            "positive": sentiment_data['positive'].tolist(),
            "negative": sentiment_data['negative'].tolist(),
            "neutral": sentiment_data['neutral'].tolist(),
            "sentiment_reasons": sentiment_reasons,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/ai-predict', methods=['GET'])
def ai_predict():
    """
    AI-powered stock prediction endpoint.
    Uses XGBoost & optionally LSTM to predict trade opportunities.
    """
    try:
        ticker = request.args.get("ticker")
        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400

        # ✅ Fetch data and preprocess
        df = fetch_historical_data(ticker)
        df = preprocess_data_with_indicators(df)
        df = detect_breakouts(df)
        df = generate_trade_signals(df)

        # ✅ Load XGBoost Model & Features
        xgb_model = load("models/xgb_model.joblib")
        features = load("models/xgb_features.pkl")

        # ✅ XGBoost Predictions
        df["xgboost_prediction"] = xgb_model.predict(df[features])

        # ✅ Check if LSTM Model is Available
        lstm_available = lstm_cache["model"] and lstm_cache["scaler"]

        if lstm_available:
            print("✅ LSTM Model Found. Enhancing AI predictions.")

            # Ensure enough data for LSTM (at least 50 rows)
            if len(df) >= 50:
                df["lstm_prediction"] = df.apply(
                    lambda row: predict_next_day(
                        model=lstm_cache["model"],
                        recent_data=df,
                        scaler=lstm_cache["scaler"],
                        features=features
                    ),
                    axis=1
                )

                # ✅ Combine XGBoost & LSTM Predictions
                xgb_weight = 0.7  # Weight for XGBoost
                lstm_weight = 0.3  # Weight for LSTM
                df["ai_prediction"] = (
                    xgb_weight * df["xgboost_prediction"] +
                    lstm_weight * (df["lstm_prediction"] / df["c"])
                )

            else:
                print(f"⚠️ Not enough data for LSTM (Only {len(df)} rows). Using XGBoost only.")
                df["ai_prediction"] = df["xgboost_prediction"]

        else:
            print("⚠️ LSTM model not found. Using XGBoost predictions only.")
            df["ai_prediction"] = df["xgboost_prediction"]

        # ✅ Plot AI Candlestick Chart
        plot_candlestick_chart(df, ticker)

        # ✅ Return AI Predictions
        return jsonify({
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "predictions": df["ai_prediction"].tolist(),
            "buy_signals": df[df["buy_signal"] == 1]["c"].tolist(),
            "sell_signals": df[df["sell_signal"] == 1]["c"].tolist()
        })

    except Exception as e:
        print(f"❌ Error in ai-predict: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    socketio.run(app, port=5000, debug=True)