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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Import utility functions
from utils.scheduler import initialize_scheduler
from utils.fetch_stock_performance import fetch_stock_performance
from utils.fetch_ticker_news import fetch_ticker_news
from utils.sentiment_plot import fetch_sentiment_trend, generate_sentiment_plot
from utils.realtime_tracking import track_stock_event, fetch_live_stock_data

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# Cache for models
lstm_cache = {"model": None, "scaler": None}


# Polygon.io API Key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "swpC4ge5_aGqdJll3gplZ6a40ADuwhzG")
if not POLYGON_API_KEY:
    raise ValueError("Polygon.io API key not found. Set POLYGON_API_KEY environment variable.")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Caching (TTLCache)
historical_data_cache = TTLCache(maxsize=10, ttl=300)

# Function to preprocess data for LSTM
def preprocess_for_lstm(data, features, target, time_steps=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i])
        y.append(data[target].iloc[i])
    return np.array(X), np.array(y), scaler

# Function to train LSTM model
def train_lstm_model(data, features, target, time_steps=50):
    X, y, scaler = preprocess_for_lstm(data, features, target, time_steps)

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
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
        raise ValueError(f"Insufficient data for LSTM prediction. Expected at least 50 rows, but got {len(recent_data)}.")

    # Scale the recent data
    recent_scaled = scaler.transform(recent_data[features].values[-50:])

    # Reshape the data for LSTM (batch size = 1, time steps = 50, features = len(features))
    reshaped_data = recent_scaled.reshape(1, 50, len(features))

    # Make prediction
    prediction = model.predict(reshaped_data)[0][0]
    return prediction


# Function to fetch historical data
@cached(historical_data_cache)
def fetch_historical_data():
    """
    Fetch historical stock data from Polygon.io.
    The function will try the most recent 7 days and return the first available data.
    Caches the result to reduce repeated API calls.
    """
    for i in range(7):  # Attempt to fetch data for the last 7 days
        most_recent_date = datetime.utcnow() - timedelta(days=i)
        most_recent_date_str = most_recent_date.strftime("%Y-%m-%d")
        print(f"Attempting to fetch stock data for: {most_recent_date_str}")
        
        # Construct API URL
        url = (
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
            f"{most_recent_date_str}?adjusted=true&apiKey={POLYGON_API_KEY}"
        )
        
        try:
            # Make API request
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Validate response content
            if "results" in data and data["results"]:
                print(f"Data fetched successfully for {most_recent_date_str}")
                return pd.DataFrame(data["results"])
            else:
                print(f"No data found for {most_recent_date_str}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {most_recent_date_str}: {e}")

    # If no data was fetched for the past 7 days, raise an error
    raise ValueError("Unable to fetch stock data for any recent trading day.")


# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]

# Function to preprocess data with enhanced indicators
# After `fetch_historical_data`

def preprocess_data_with_indicators(data):
    """
    Add volume, sentiment score, and enhanced technical indicators.
    """
    try:
        # Rename columns for consistency
        data.rename(columns={"v": "volume"}, inplace=True)

        # Calculate basic metrics
        data["price_change"] = (data["c"] - data["o"]) / data["o"]  # Percentage change
        data["volatility"] = (data["h"] - data["l"]) / data["l"]    # High-Low spread

        # Volume Surge Calculation
        if "volume" in data.columns:
            data["volume_surge"] = data["volume"] / data["volume"].rolling(window=5).mean()
        else:
            print("Warning: 'volume' column missing. Defaulting volume_surge to 1.0.")
            data["volume_surge"] = 1.0

        # Sentiment Score Calculation
        if "T" in data.columns:
            data["sentiment_score"] = data["T"].apply(
                lambda x: analyze_sentiment(f"News headline for {x}") if isinstance(x, str) else 0
            )
        else:
            print("Warning: 'T' column missing. Sentiment score set to 0.")
            data["sentiment_score"] = 0

        # Add Technical Indicators
        if not data["c"].isnull().all():
            # Relative Strength Index (RSI)
            rsi_indicator = RSIIndicator(close=data["c"], window=14, fillna=True)
            data["rsi"] = rsi_indicator.rsi()

            # Moving Average Convergence Divergence (MACD)
            macd = MACD(close=data["c"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            data["macd_diff"] = macd.macd_diff()

            # Simple Moving Average (SMA)
            data["sma_20"] = data["c"].rolling(window=20).mean()

            # Bollinger Bands (requires `sma_20`)
            data["bollinger_upper"] = data["sma_20"] + (data["c"].rolling(window=20).std() * 2)
            data["bollinger_lower"] = data["sma_20"] - (data["c"].rolling(window=20).std() * 2)

            # Exponential Moving Average (EMA)
            data["ema_20"] = data["c"].ewm(span=20, adjust=False).mean()

            # Average True Range (ATR)
            data["atr"] = (data["h"] - data["l"]).rolling(window=14).mean()

        # Handle Missing Values
        data.fillna(0, inplace=True)

        # Debug Output
        debug_columns = [
            "price_change", "volatility", "volume_surge",
            "rsi", "macd_diff", "sma_20", "ema_20",
            "bollinger_upper", "bollinger_lower", "atr"
        ]
        print(f"Data shape after preprocessing: {data.shape}")
        print("Preview of added indicators:", data[debug_columns].head())

        return data

    except Exception as e:
        print(f"Error in preprocess_data_with_indicators: {e}")
        raise

# Train XGBoost model
def train_xgboost_model():
    data = fetch_historical_data()
    data = preprocess_data_with_indicators(data)
    data["target"] = (data["h"] >= data["c"] * 1.05).astype(int)
    features = ["price_change", "volatility", "volume", "sentiment_score"]
    X_train, X_test, y_train, y_test = train_test_split(data[features], data["target"], test_size=0.2, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    print("Model Evaluation:", classification_report(y_test, model.predict(X_test)))
    return model, features

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

        live_data = fetch_live_stock_data(ticker)

        # Check if the LSTM model is initialized
        if not lstm_cache["model"] or not lstm_cache["scaler"]:
            raise ValueError("LSTM model is not initialized. Please initialize the model via the scan-stocks route.")

        prediction = predict_next_day(
            model=lstm_cache["model"],
            recent_data=live_data,
            scaler=lstm_cache["scaler"],
            features=["price_change", "volatility", "volume", "sentiment_score"]
        )
        return jsonify({"ticker": ticker, "next_day_prediction": prediction}), 200
    except Exception as e:
        print(f"Error in lstm-predict endpoint: {e}")
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

        # Step 2: Train LSTM if not already cached
        if not lstm_cache["model"] or not lstm_cache["scaler"]:
            print("Training LSTM model...")
            lstm_cache["model"], lstm_cache["scaler"] = train_lstm_model(
                data=data,
                features=["price_change", "volatility", "volume", "sentiment_score"],
                target="c"
            )
        else:
            print("Using cached LSTM model...")

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

def preprocess_data_with_indicators(data):
    """Add volume, sentiment score, and enhanced technical indicators."""
    data.rename(columns={"v": "volume"}, inplace=True)
    data["price_change"] = (data["c"] - data["o"]) / data["o"]
    data["volatility"] = (data["h"] - data["l"]) / data["l"]

    # Ensure volume column exists before calculating volume_surge
    if "volume" in data.columns:
        data["volume_surge"] = data["volume"] / data["volume"].rolling(window=5).mean()
    else:
        print("Warning: 'volume' column is missing. Setting default volume_surge=1.0")
        data["volume_surge"] = 1.0  # Default value if volume is missing

    # Safeguard for missing or invalid 'T' column
    if "T" in data.columns:
        data["sentiment_score"] = data["T"].apply(
            lambda x: analyze_sentiment(f"News headline for {x}") if isinstance(x, str) else 0
        )
    else:
        print("Warning: 'T' column is missing. Sentiment score set to 0.")
        data["sentiment_score"] = 0

    # Add RSI, MACD, and other indicators
    if not data["c"].isnull().all():
        rsi_indicator = RSIIndicator(close=data["c"], window=14, fillna=True)
        data["rsi"] = rsi_indicator.rsi()

        macd = MACD(close=data["c"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        data["macd_diff"] = macd.macd_diff()

        # Add more indicators as required
        data["sma_20"] = data["c"].rolling(window=20).mean()
        data["ema_20"] = data["c"].ewm(span=20).mean()

    # Fill missing values to avoid NaN issues
    data.fillna(0, inplace=True)

    # Debug output
    print("Data after adding indicators:", data.head())
    return data
# API Route for Real-Time Stock Data
@app.route('/api/live-data', methods=['GET'])
def live_data():
    try:
        ticker = request.args.get("ticker")
        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400
        live_data = fetch_live_stock_data(ticker)
        return jsonify(live_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/candlestick', methods=['GET'])
def candlestick_chart():
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400
        print(f"Fetching candlestick data for ticker: {ticker}")
        end_date = datetime.today()
        start_date = end_date - timedelta(days=180)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" not in data or not data["results"]:
            return jsonify({"error": f"No data available for ticker {ticker}"}), 404
        results = pd.DataFrame(data["results"])
        results.fillna(0, inplace=True)
        return jsonify({
            "dates": results["t"].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d')).tolist(),
            "open": results["o"].tolist(),
            "high": results["h"].tolist(),
            "low": results["l"].tolist(),
            "close": results["c"].tolist(),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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
