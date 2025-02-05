from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime, timedelta
from cachetools import TTLCache, cached
from flask_socketio import SocketIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D, LeakyReLU, MultiHeadAttention # type: ignore
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import WilliamsRIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from tensorflow.keras.models import save_model, load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, LeakyReLU, LayerNormalization # type: ignore
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam # type: ignore
import time
from threading import Thread
import json
import threading
import websocket
# Import utility functions
from utils.scheduler import initialize_scheduler
from utils.fetch_stock_performance import fetch_stock_performance
from utils.fetch_ticker_news import fetch_ticker_news
from utils.sentiment_plot import fetch_sentiment_trend, generate_sentiment_plot
from utils.realtime_tracking import track_stock_event, fetch_live_stock_data
from dotenv import load_dotenv  # ✅ Import dotenv

# ✅ Load environment variables from .env file
load_dotenv()
print(os.getenv("ALPHA_VANTAGE_API_KEY"))

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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
@cached(historical_data_cache)
@cached(historical_data_cache)
def fetch_historical_data():
    """
    Fetch historical stock data from Polygon.io.
    The function will try fetching data for the last 14 days to ensure availability.
    Caches the result to reduce repeated API calls.
    """
    for i in range(14):  # Try fetching data for the last 14 days (extended from 7 days)
        most_recent_date = datetime.utcnow() - timedelta(days=i)
        most_recent_date_str = most_recent_date.strftime("%Y-%m-%d")
        print(f"🔍 Attempting to fetch stock data for: {most_recent_date_str}")

        url = (
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
            f"{most_recent_date_str}?adjusted=true&apiKey={POLYGON_API_KEY}"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()

            if "results" in data and data["results"]:
                print(f"✅ Data fetched successfully for {most_recent_date_str}")
                return pd.DataFrame(data["results"])  # Convert JSON to DataFrame

            print(f"⚠️ No stock data found for {most_recent_date_str}")

        except requests.exceptions.Timeout:
            print(f"❌ Timeout error while fetching data for {most_recent_date_str}")

        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error: {http_err}")

        except requests.exceptions.RequestException as req_err:
            print(f"❌ Request error: {req_err}")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    # If all attempts fail, return an empty DataFrame instead of raising an error
    print("❌ Unable to fetch stock data for any recent trading day. Returning empty DataFrame.")
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
            "5. volume": "v"
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


def preprocess_data_with_indicators(data):
    """
    Add volume, sentiment score, and enhanced technical indicators.
    """
    try:
        # Rename columns for consistency
        data.rename(columns={"v": "volume"}, inplace=True)

        # Calculate basic metrics
        data["price_change"] = (data["c"] - data["o"]) / data["o"]
        data["volatility"] = (data["h"] - data["l"]) / data["l"]

        # Volume Surge Calculation
        data["volume_surge"] = data["volume"] / data["volume"].rolling(window=5).mean()

        # ✅ **Fix for analyze_sentiment missing error**
        if "T" in data.columns:
            data["sentiment_score"] = data["T"].apply(lambda x: analyze_sentiment(str(x)) if isinstance(x, str) else 0)
        else:
            print("Warning: 'T' column missing. Setting sentiment score to 0.")
            data["sentiment_score"] = 0

        # Add Technical Indicators
        if not data["c"].isnull().all():
            rsi_indicator = RSIIndicator(close=data["c"], window=14, fillna=True)
            data["rsi"] = rsi_indicator.rsi()

            macd = MACD(close=data["c"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            data["macd_diff"] = macd.macd_diff()

            data["sma_20"] = data["c"].rolling(window=20).mean()

            bb = BollingerBands(close=data["c"], window=20, fillna=True)
            data["bollinger_upper"] = bb.bollinger_hband()
            data["bollinger_lower"] = bb.bollinger_lband()

        # Handle Missing Values
        data.fillna(0, inplace=True)

        return data

    except Exception as e:
        print(f"Error in preprocess_data_with_indicators: {e}")
        raise


# Train XGBoost model
def train_xgboost_model():
    try:
        # Fetch and preprocess data
        data = fetch_historical_data()
        data = preprocess_data_with_indicators(data)

        # Define the feature set dynamically
        features = [
            col for col in [
                "price_change", "volatility", "volume", "volume_surge",
                "obv", "williams_r", "ema_50", "ema_200",
                "bollinger_upper", "bollinger_lower", "vwap"
            ] if col in data.columns
        ]

        if not features:
            raise ValueError("No valid features available for training the model.")

        # Define target
        data["target"] = (data["h"] >= data["c"] * 1.05).astype(int)  # Example target condition

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data["target"], test_size=0.2, random_state=42
        )

        # Initialize and train the XGBoost model
        model = XGBClassifier()
        model.fit(X_train, y_train)

        # Print evaluation metrics
        print("Model Evaluation:")
        print(classification_report(y_test, model.predict(X_test)))

        # Save the model
        dump(model, "models/xgb_model.joblib")  # Save in models folder
        print("[DEBUG] XGBoost model trained and saved successfully.")

        return model, features

    except KeyError as e:
        print(f"KeyError in train_xgboost_model: {e}")
        raise

    except Exception as e:
        print(f"Error in train_xgboost_model: {e}")
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

        # Define features and target
        features = ["price_change", "volatility", "volume", "sentiment_score"]
        target = "c"  # Target column (e.g., closing price)

        # Train the LSTM model
        model, scaler = train_lstm_model(data, features, target)

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
        # Parse user inputs
        min_price = float(request.args.get("min_price", 0))
        max_price = float(request.args.get("max_price", float("inf")))
        volume_surge = float(request.args.get("volume_surge", 1.2))
        min_rsi = float(request.args.get("min_rsi", 0))
        max_rsi = float(request.args.get("max_rsi", 100))

        print(f"Scan stocks request params: min_price={min_price}, max_price={max_price}, volume_surge={volume_surge}, min_rsi={min_rsi}, max_rsi={max_rsi}")

        # Fetch and preprocess data
        data = fetch_historical_data()
        print("Fetched historical data:", data.head())

        data = preprocess_data_with_indicators(data)
        print("Processed data with indicators:", data.head())

        # Apply user-defined filters
        filtered_data = data[
            (data["c"] >= min_price) & 
            (data["c"] <= max_price) & 
            (data["volume_surge"] > volume_surge) & 
            (data["rsi"] >= min_rsi) & 
            (data["rsi"] <= max_rsi)
        ]
        print("Data after applying user-defined filters:", filtered_data.head())

        if filtered_data.empty:
            print("No data matching filters.")
            return jsonify({"candidates": []}), 200

        # Step 1: Apply XGBoost Predictions
        filtered_data["xgboost_prediction"] = xgb_model.predict(filtered_data[feature_columns])
        xgb_filtered_data = filtered_data[filtered_data["xgboost_prediction"] == 1]
        print("Data after XGBoost filtering:", xgb_filtered_data.head())

        if xgb_filtered_data.empty:
            print("No data matching XGBoost predictions.")
            return jsonify({"candidates": []}), 200

        # Step 2: Ensure LSTM is Trained and Cached
        if not lstm_cache["model"] or not lstm_cache["scaler"]:
            return jsonify({"error": "LSTM model is not trained. Please train the model using /api/train-lstm before scanning stocks."}), 500

        # Check if sufficient data exists for LSTM prediction
        if len(xgb_filtered_data) < 50:
            print(f"Insufficient data for LSTM prediction: {len(xgb_filtered_data)} rows.")
            return jsonify({"candidates": xgb_filtered_data.head(20).to_dict(orient="records")}), 200

        # Step 3: Apply LSTM Predictions
        xgb_filtered_data["next_day_prediction"] = xgb_filtered_data.apply(
            lambda row: predict_next_day(
                model=lstm_cache["model"],
                recent_data=xgb_filtered_data,
                scaler=lstm_cache["scaler"],
                features=["price_change", "volatility", "volume", "sentiment_score"]
            ),
            axis=1
        )
        print("Data with LSTM predictions:", xgb_filtered_data.head())

        # Step 4: Combine Predictions with Weighted Scores
        xgb_weight = 0.7  # Weight for XGBoost
        lstm_weight = 0.3  # Weight for LSTM
        xgb_filtered_data["combined_score"] = (
            (xgb_weight * xgb_filtered_data["xgboost_prediction"]) +
            (lstm_weight * (xgb_filtered_data["next_day_prediction"] / xgb_filtered_data["c"]))
        )
        print("Data with combined scores:", xgb_filtered_data.head())

        # Step 5: Sort and Limit Results
        top_candidates = xgb_filtered_data.sort_values("combined_score", ascending=False).head(20)
        print("Top 20 candidates:", top_candidates)

        # Return filtered candidates
        return jsonify({"candidates": top_candidates.to_dict(orient="records")}), 200

    except Exception as e:
        print(f"Error in scan-stocks endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Function to preprocess data for LSTM with additional features
from ta.momentum import RSIIndicator

def preprocess_for_lstm(data, features, target, time_steps=150):
    # Ensure RSI is computed before smoothing
    if "rsi" not in data.columns:
        rsi_indicator = RSIIndicator(close=data["c"], window=14, fillna=True)
        data["rsi"] = rsi_indicator.rsi()

    # Compute rolling mean for sentiment score to smooth fluctuations
    data["sentiment_score_avg"] = data["sentiment_score"].rolling(window=10).mean().fillna(0)
    data["rsi_smooth"] = data["rsi"].rolling(window=5).mean().fillna(0)
    data["macd_smooth"] = data["macd_diff"].rolling(window=5).mean().fillna(0)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i])
        y.append(data[target].iloc[i])
    
    return np.array(X), np.array(y), scaler

# Optimized LSTM Model with Attention, CNN, and deeper architecture
def train_lstm_model(data, features, target, time_steps=150):
    X, y, scaler = preprocess_for_lstm(data, features, target, time_steps)
    
    model = Sequential([
        Conv1D(filters=128, kernel_size=5, activation=LeakyReLU(alpha=0.1), input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        MultiHeadAttention(num_heads=4, key_dim=64),
        LayerNormalization(),
        Bidirectional(LSTM(256, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        GlobalAveragePooling1D(),
        Dense(64, activation=LeakyReLU(alpha=0.1)),
        Dropout(0.2),
        Dense(32, activation="swish"),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error")
    
    # Callbacks for early stopping and learning rate decay
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    model.fit(X, y, epochs=500, batch_size=128, validation_split=0.2, verbose=1,
              callbacks=[early_stopping, reduce_lr])
    
    return model, scaler
# Function to pad sequences when data is insufficient
def pad_sequences(data, required_length=150):
    if len(data) < required_length:
        pad_size = required_length - len(data)
        padding = np.zeros((pad_size, data.shape[1]))
        return np.vstack((padding, data))
    return data

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
def train_and_cache_lstm_model():
    """
    Train the LSTM model and cache it for future use.
    """
    try:
        # Fetch and preprocess historical data
        data = fetch_historical_data()
        data = preprocess_data_with_indicators(data)

        # Define features and target
        features = ["price_change", "volatility", "volume", "sentiment_score"]
        target = "c"  # Target column (e.g., closing price)

        # Train the LSTM model
        model, scaler = train_lstm_model(data, features, target)

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
    API endpoint to train the LSTM model and cache it.
    """
    try:
        train_and_cache_lstm_model()
        return jsonify({"message": "LSTM model trained and cached successfully."}), 200
    except Exception as e:
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

        # Placeholder for LSTM model prediction
        if not lstm_cache["model"] or not lstm_cache["scaler"]:
            raise ValueError("LSTM model is not initialized. Please initialize the model via the scan-stocks route.")

        prediction = predict_next_day(
            model=lstm_cache["model"],
            recent_data={"ticker": ticker, "price": price},
            scaler=lstm_cache["scaler"],
            features=["price_change", "volatility", "volume", "sentiment_score"]
        )
        return jsonify({"ticker": ticker, "next_day_prediction": prediction}), 200
    except Exception as e:
        print(f"Error in lstm-predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Start WebSocket thread
websocket_thread = threading.Thread(target=start_websocket_thread, daemon=True)
websocket_thread.start()
def preprocess_data_with_indicators(data):
    """
    Add volume, sentiment score, and advanced technical indicators.
    """
    try:
        # Rename volume column
        data.rename(columns={"v": "volume"}, inplace=True)

        # Basic Calculations
        data["price_change"] = (data["c"] - data["o"]) / data["o"]  
        data["volatility"] = (data["h"] - data["l"]) / data["l"]

        # Volume Surge
        data["volume_surge"] = data["volume"] / data["volume"].rolling(window=5).mean()

        # On-Balance Volume (OBV)
        obv_indicator = OnBalanceVolumeIndicator(close=data["c"], volume=data["volume"], fillna=True)
        data["obv"] = obv_indicator.on_balance_volume()

        # Williams %R
        williams_r = WilliamsRIndicator(high=data["h"], low=data["l"], close=data["c"], lbp=14, fillna=True)
        data["williams_r"] = williams_r.williams_r()

        # Exponential Moving Average (EMA)
        data["ema_50"] = EMAIndicator(close=data["c"], window=50, fillna=True).ema_indicator()
        data["ema_200"] = EMAIndicator(close=data["c"], window=200, fillna=True).ema_indicator()

        # Bollinger Bands
        bb = BollingerBands(close=data["c"], window=20, fillna=True)
        data["bollinger_upper"] = bb.bollinger_hband()
        data["bollinger_lower"] = bb.bollinger_lband()

        # Volume Weighted Average Price (VWAP)
        data["vwap"] = (data["volume"] * (data["h"] + data["l"] + data["c"]) / 3).cumsum() / data["volume"].cumsum()

        # Handle Missing Values
        data.fillna(0, inplace=True)

        return data

    except Exception as e:
        print(f"Error in preprocess_data_with_indicators: {e}")
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

if __name__ == "__main__":
    socketio.run(app, port=5000, debug=True)