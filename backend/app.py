from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
import pandas as pd
import numpy as np
import json
import threading
from datetime import datetime, timedelta

# ✅ WebSocket & Flask SocketIO
import websocket
from flask_socketio import SocketIO

# ✅ Machine Learning & AI
from joblib import dump, load
import joblib

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
from utils.fetch_historical_performance import fetch_historical_data
# ✅ Import indicators.py for technical analysis
from utils.indicators import preprocess_data_with_indicators
from utils.train_model import train_and_cache_lstm_model
from utils.train_xgboost import train_xgboost_with_optuna
from utils.lstm_utils import load_lstm_model
from utils.model_loader import load_xgb_model  # ✅ Import the utility function



import logging

MODELS_DIR = r"C:\Users\gabby\trax-x\backend\models"
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_scaler.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "optimized_xgb_model.joblib")
XGB_FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_features.pkl")

# ✅ Load environment variables from .env file
load_dotenv()
print(os.getenv("ALPHA_VANTAGE_API_KEY"))

# ✅ Cache for LSTM Model (Fix the issue)
lstm_cache = {"model": None, "scaler": None}

# ✅ Load LSTM model at startup
if lstm_cache["model"] is None or lstm_cache["scaler"] is None:
    print("✅ Checking for saved LSTM model...")

    model, scaler = load_lstm_model()

    if model is not None and scaler is not None:
        lstm_cache["model"], lstm_cache["scaler"] = model, scaler
        print("✅ Loaded saved LSTM model successfully.")
    else:
        print("⚠️ LSTM model or scaler missing! Skipping retraining. Fix the issue first.")
        lstm_cache["model"], lstm_cache["scaler"] = None, None  # Prevent infinite loop


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
    Fetch detailed historical data for a selected stock ticker.
    """
    ticker = request.args.get("ticker")
    
    if not ticker:
        return jsonify({"error": "Ticker parameter is missing"}), 400

    print(f"📊 Fetching historical data for: {ticker}")

    # Fetch historical data for the given ticker
    df = fetch_historical_data()

    if df.empty:
        return jsonify({"error": "No historical data found"}), 404

    # Apply technical indicators
    df, _ = preprocess_data_with_indicators(df)  

    # Generate buy/sell signals
    df["buy_signal"] = (df["rsi"] < 30) & (df["macd_diff"] > 0)
    df["sell_signal"] = (df["rsi"] > 70) & (df["macd_diff"] < 0)

    # Prepare response data
    response_data = {
        "dates": df["timestamp"].dt.strftime('%Y-%m-%d').tolist(),
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["volume"].tolist(),
        "buy_signals": df[df["buy_signal"]]["close"].tolist(),
        "sell_signals": df[df["sell_signal"]]["close"].tolist()
    }

    return jsonify(response_data), 200

# Caching (TTLCache)
historical_data_cache = TTLCache(maxsize=10, ttl=300)

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
def fetch_sentiment_score_alpha(ticker):
    """Fetch market sentiment score for a given stock ticker using Alpha Vantage API."""
    API_KEY = os.getenv("3R7BUV52GH1MOHNO")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        # ✅ Extract sentiment score from Alpha Vantage response
        if "feed" in data and len(data["feed"]) > 0:
            sentiment_scores = [article["overall_sentiment_score"] for article in data["feed"]]
            return np.mean(sentiment_scores) if sentiment_scores else 0
        else:
            return 0  # Default neutral score if no data available
    except Exception as e:
        print(f"❌ ERROR fetching sentiment for {ticker}: {e}")
        return 0  # Default to 0 on failure


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
    data["prev_high"] = data["high"].rolling(window=window).max().shift(1)
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


# Function to preprocess data with enhanced indicators
# After fetch_historical_data

# Load XGBoost model if it exists, otherwise train it
# Load LSTM model if it exists, otherwise train it

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
    df, _ = preprocess_data_with_indicators(df)  # Extract only the DataFrame


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


def predict_next_day(model, recent_data, scaler, features):
    """
    Predict the next day's value using the LSTM model.
    """
    try:
        if len(recent_data) < 50:
            print(f"⚠️ Warning: Only {len(recent_data)} rows available. LSTM requires at least 50.")
            return recent_data["c"].iloc[-1]  # Default to last close price

        # Ensure required features exist
        missing_features = [f for f in features if f not in recent_data.columns]
        if missing_features:
            print(f"⚠️ Warning: Missing features for LSTM: {missing_features}")
            for feature in missing_features:
                recent_data[feature] = 0  # Default missing features to 0

        # Extract last 50 rows for LSTM
        recent_data = recent_data[features].values[-50:]

        # Scale the data
        recent_scaled = scaler.transform(recent_data)

        # Reshape for LSTM (batch_size=1, time_steps=50, features=len(features))
        reshaped_data = recent_scaled.reshape(1, 50, len(features))

        # Make LSTM Prediction
        prediction = model.predict(reshaped_data)[0][0]

        return prediction

    except Exception as e:
        print(f"❌ ERROR in predict_next_day: {e}")
        return 0  # Default to 0 in case of failure



# Function to scan stocks
from utils.train_xgboost import train_xgboost_with_optuna  # ✅ Import Optuna-trained model

import logging

# ✅ Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.route('/api/scan-stocks', methods=['GET']) 
def scan_stocks():
    try:
        # ✅ Extract filtering parameters
        min_price = float(request.args.get("min_price", 0))
        max_price = float(request.args.get("max_price", float("inf")))
        volume_surge = float(request.args.get("volume_surge", 1.2))
        min_rsi = float(request.args.get("min_rsi", 0))
        max_rsi = float(request.args.get("max_rsi", 100))

        logging.info(f"📌 Scan Params: min_price={min_price}, max_price={max_price}, volume_surge={volume_surge}, min_rsi={min_rsi}, max_rsi={max_rsi}")

        # ✅ Fetch and preprocess historical data
        data = fetch_historical_data()
        if data is None or data.empty:
            logging.warning("⚠️ No stock data available!")
            return jsonify({"error": "No stock data available"}), 404

        # ✅ Ensure 'ticker' column exists
        if "ticker" not in data.columns:
            logging.error("❌ ERROR: 'ticker' column is missing from historical data!")
            return jsonify({"error": "'ticker' column missing from data"}), 500

        logging.info(f"📌 Total stocks before preprocessing: {len(data)}")
        data, _ = preprocess_data_with_indicators(data)  # Extract only DataFrame

        logging.info(f"📌 Total stocks after preprocessing: {len(data)}")
        logging.info(f"📌 Buy Signal Distribution Before Filtering:\n{data['buy_signal'].value_counts()}")
        logging.info(f"📌 RSI Distribution Before Filtering:\n{data['rsi'].describe()}")

        # ✅ Convert RSI Back to 0-100 If Needed
        if data["rsi"].max() < 20 or data["rsi"].min() < -20:  # Check if RSI is standardized
            logging.warning("⚠️ RSI is standardized! Converting back to 0-100 range...")

            rsi_original_min = 0
            rsi_original_max = 100

            # Normalize RSI back to its original range
            data["rsi"] = ((data["rsi"] - data["rsi"].min()) / (data["rsi"].max() - data["rsi"].min())) * \
                          (rsi_original_max - rsi_original_min) + rsi_original_min

        # ✅ Log RSI after correction
        logging.info(f"📌 RSI Distribution After Scaling Fix:\n{data['rsi'].describe()}")

        # ✅ Apply filtering conditions step by step
        filtered_data = data[(data["close"] >= min_price) & (data["close"] <= max_price)]
        logging.info(f"📌 Stocks after price filtering: {len(filtered_data)}")

        filtered_data = filtered_data[filtered_data["volume_surge"] > volume_surge]
        logging.info(f"📌 Stocks after volume filtering: {len(filtered_data)}")

        logging.info(f"📌 RSI Distribution After Volume Filtering:\n{filtered_data['rsi'].describe()}")

        # ✅ Apply RSI Filtering
        filtered_data = filtered_data[(filtered_data["rsi"] >= min_rsi) & (filtered_data["rsi"] <= max_rsi)]
        logging.info(f"📌 Stocks after RSI filtering: {len(filtered_data)}")

        if filtered_data.empty:
            logging.warning("⚠️ No stocks left after filtering!")
            return jsonify({"candidates": []}), 200

        # ✅ Log ticker values before returning
        unique_tickers = filtered_data["ticker"].unique()
        logging.info(f"📌 Ticker Count After Filtering: {len(unique_tickers)}")
        logging.info(f"📌 Sample Tickers: {unique_tickers[:10]}")  # Show first 10 tickers

        return jsonify({"candidates": filtered_data.to_dict(orient="records")}), 200

    except Exception as e:
        logging.error(f"❌ ERROR in scan-stocks: {e}", exc_info=True)  # ✅ Include full traceback
        return jsonify({"error": str(e)}), 500

# Function to predict the next day using LSTM
def predict_next_day(model, recent_data, scaler, features):
    """
    Predict the next day's value using the LSTM model.
    
    - Ensures correct feature selection
    - Scales the input before passing it to the LSTM
    - Handles missing feature errors gracefully
    """
    try:
        # ✅ Ensure enough data for LSTM
        if len(recent_data) < 50:
            print(f"⚠️ Warning: Only {len(recent_data)} rows available. LSTM requires at least 50.")
            return recent_data["c"].iloc[-1]  # Default to last close price

        # ✅ Ensure required features exist
        missing_features = [f for f in features if f not in recent_data.columns]
        if missing_features:
            print(f"⚠️ Warning: Missing features for LSTM: {missing_features}")
            for feature in missing_features:
                recent_data[feature] = 0  # Default missing features to 0

        # ✅ Extract last 50 rows for LSTM
        recent_data = recent_data[features].values[-50:]

        # ✅ Scale the data
        recent_scaled = scaler.transform(recent_data)

        # ✅ Reshape for LSTM (batch_size=1, time_steps=50, features=len(features))
        reshaped_data = recent_scaled.reshape(1, 50, len(features))

        # ✅ Make LSTM Prediction
        prediction = model.predict(reshaped_data)[0][0]

        return prediction

    except Exception as e:
        print(f"❌ ERROR in predict_next_day: {e}")
        return 0  # Default to 0 in case of failure


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
        df, _ = preprocess_data_with_indicators(df)  # Extract only the DataFrame


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
        logging.warning("⚠️ No ticker provided in request.")
        return jsonify({"error": "Ticker is required"}), 400

    ticker_list = tickers.split(",")  # Split tickers into a list
    logging.info(f"📌 Fetching news for tickers: {ticker_list}")

    all_news = {}

    for ticker in ticker_list:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={POLYGON_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            all_news[ticker] = response.json().get("results", [])
        except requests.exceptions.HTTPError as e:
            logging.error(f"❌ Error fetching news for {ticker}: {str(e)}")
            all_news[ticker] = {"error": f"Error fetching news for {ticker}: {str(e)}"}
        except Exception as e:
            logging.error(f"❌ Unexpected error fetching news for {ticker}: {str(e)}")
            all_news[ticker] = {"error": f"Unexpected error: {str(e)}"}

    logging.info(f"📌 News response: {all_news}")  # ✅ Log full response
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

        print(f"📌 AI Prediction Triggered for Ticker: {ticker}")

        # ✅ Fetch and preprocess data
        df = fetch_historical_data(ticker)  # ✅ Pass ticker to fetch only relevant data
        if df is None or df.empty:
            print("⚠️ No data available after fetching!")
            return jsonify({"error": "No data available for the given ticker"}), 404

        df, scaler = preprocess_data_with_indicators(df)
        if df.empty:
            print("⚠️ No data available after preprocessing!")
            return jsonify({"error": "No data available for the given ticker"}), 404

        print(f"📌 Dataframe Size After Preprocessing: {len(df)} rows")

        # ✅ Load XGBoost Model & Features
        try:
            xgb_model = load(XGB_MODEL_PATH)
            features = load(XGB_FEATURES_PATH)  # Ensure feature names are correctly loaded
            print("✅ XGBoost Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading XGBoost model: {e}")
            return jsonify({"error": f"❌ Error loading XGBoost model: {e}"}), 500

        # ✅ Load or Train LSTM Model
        lstm_model, lstm_scaler = lstm_cache.get("model"), lstm_cache.get("scaler")

        if not lstm_model or not lstm_scaler:
            print("⚠️ LSTM model not found in cache. Attempting to load saved model...")

            # Attempt to load the saved model first
            lstm_model, lstm_scaler = load_lstm_model()

            if lstm_model and lstm_scaler:
                print("✅ Loaded saved LSTM model successfully!")
                lstm_cache["model"], lstm_cache["scaler"] = lstm_model, lstm_scaler
            else:
                print("⚠️ LSTM model is missing! Training a new one...")
                lstm_model, lstm_scaler = train_and_cache_lstm_model()
                lstm_cache["model"], lstm_cache["scaler"] = lstm_model, lstm_scaler

        # ✅ Apply XGBoost Predictions
        df["xgboost_prediction"] = xgb_model.predict(df[features])
        print("✅ XGBoost Predictions Applied!")

        # ✅ Apply LSTM Predictions if Available
        time_steps = 50  # FIXED: Ensure consistent LSTM time steps
        if len(df) >= time_steps:
            print(f"📌 Applying LSTM on last {time_steps} rows...")

            # ✅ Ensure correct feature order and format
            df_features = df[features]  # Select only the required features
            df_scaled = pd.DataFrame(lstm_scaler.transform(df_features), columns=df_features.columns)

            # ✅ Ensure correct input shape (1, 50, num_features)
            if len(df_scaled) < time_steps:
                padding = np.zeros((time_steps - len(df_scaled), len(features)))
                df_scaled_padded = np.vstack([padding, df_scaled.values])
            else:
                df_scaled_padded = df_scaled.values[-time_steps:]

            X_lstm = df_scaled_padded.reshape(1, time_steps, len(features))

            # ✅ Make LSTM Prediction
            lstm_prediction = lstm_model.predict(X_lstm)[0][0]
            print(f"📌 LSTM Next-Day Prediction: {lstm_prediction}")
            df["lstm_prediction"] = lstm_prediction

            # ✅ Combine XGBoost & LSTM Predictions
            xgb_weight, lstm_weight = 0.6, 0.4
            df["ai_prediction"] = (
                xgb_weight * df["xgboost_prediction"] +
                lstm_weight * (df["lstm_prediction"] / df["close"])
            )
        else:
            print("⚠️ Not enough data for LSTM. Using XGBoost only.")
            df["ai_prediction"] = df["xgboost_prediction"]

        print("✅ AI Predictions Completed!")

        # ✅ Ensure the index is in datetime format
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        return jsonify({
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "predictions": df["ai_prediction"].tolist(),
            "buy_signals": df[df["buy_signal"] == 1]["close"].tolist() if "buy_signal" in df.columns else [],
            "sell_signals": df[df["sell_signal"] == 1]["close"].tolist() if "sell_signal" in df.columns else []
        })

    except Exception as e:
        print(f"❌ Error in ai-predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train-lstm", methods=["POST"])
def train_lstm():
    """
    API endpoint to train the LSTM model.
    """
    try:
        model, scaler = train_and_cache_lstm_model()

        if model and scaler:
            return jsonify({"message": "✅ LSTM model trained and saved successfully!"}), 200
        else:
            return jsonify({"error": "❌ LSTM training failed."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    socketio.run(app, port=5000, debug=True)