import requests
import pandas as pd
import numpy as np
from flask_socketio import emit
from xgboost import XGBClassifier
import os

# Polygon.io API Key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "swpC4ge5_aGqdJll3gplZ6a40ADuwhzG")

# Example: Load a pre-trained XGBoost model
xgb_model = XGBClassifier()  # Replace with your trained model
feature_columns = ["price_change", "volatility", "volume", "sentiment_score", "rsi", "macd_diff"]

# Fetch Live Stock Data
def fetch_live_stock_data(ticker):
    """Fetch real-time stock data from Polygon.io."""
    url = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return {
        "ticker": ticker,
        "price": data["results"]["p"],
        "timestamp": data["results"]["t"]
    }

# Preprocess Live Data
def preprocess_live_data(price, volume, sentiment_score):
    """Prepare real-time data for AI/ML prediction."""
    return {
        "price_change": price / 100,  # Dummy calculation
        "volatility": price / 50,     # Dummy calculation
        "volume": volume,
        "sentiment_score": sentiment_score,
        "rsi": np.random.uniform(30, 70),  # Placeholder RSI
        "macd_diff": np.random.uniform(-1, 1),  # Placeholder MACD diff
    }

# Handle Real-Time Stock Tracking
def track_stock_event(data):
    """Real-time stock tracking using WebSocket."""
    ticker = data.get("ticker")
    if not ticker:
        return emit("error", {"message": "Ticker is missing"})

    try:
        live_data = fetch_live_stock_data(ticker)
        processed_data = preprocess_live_data(live_data["price"], 1000, 0.5)  # Replace with actual values
        features = pd.DataFrame([processed_data])[feature_columns]
        prediction = xgb_model.predict(features)[0]

        # Emit real-time data with AI/ML prediction
        emit("stock_update", {
            "ticker": live_data["ticker"],
            "price": live_data["price"],
            "timestamp": live_data["timestamp"],
            "recommendation": "Buy" if prediction == 1 else "Sell" if prediction == -1 else "Hold"
        })
    except Exception as e:
        emit("error", {"message": str(e)})
