import os
import pandas as pd
import numpy as np
from flask_socketio import emit
from xgboost import XGBClassifier
from joblib import load, dump
import requests

# Polygon.io API Key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "swpC4ge5_aGqdJll3gplZ6a40ADuwhzG")

# Model file path
MODEL_DIR = os.path.abspath(r"C:\Users\gabby\trax-x\backend\models")  # Ensure absolute path
if not os.path.exists(MODEL_DIR):
    print(f"📁 Creating missing models directory at: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")

# Feature columns for model training
feature_columns = ["price_change", "volatility", "volume", "sentiment_score", "rsi", "macd_diff"]

# ✅ Ensure the models directory exists before saving/loading the model
if not os.path.exists(MODEL_DIR):
    print(f"📁 Creating missing models directory: {MODEL_DIR}")
    os.makedirs(MODEL_DIR)

# ✅ Load or train the XGBoost model
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}!")
        except Exception as e:
            print(f"❌ Error loading the model: {e}. Retraining the model.")
            model = train_and_save_model()
    else:
        print("⚠️ Model file not found. Training a new model.")
        model = train_and_save_model()
    return model

# ✅ Train and save the model
def train_and_save_model():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("📌 Training new XGBoost model...")

    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_features=len(feature_columns), random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved successfully at {MODEL_PATH}!")

    return model

# ✅ Load the trained model
xgb_model = load_or_train_model()

# ✅ Fetch Live Stock Data
def fetch_live_stock_data(ticker):
    """Fetch real-time stock data from Polygon.io."""
    try:
        url = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "ticker": ticker,
            "price": data["results"]["p"],
            "timestamp": data["results"]["t"]
        }
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching live stock data for {ticker}: {e}")

# ✅ Preprocess Real-Time Data for AI/ML Prediction
def preprocess_live_data(price, volume, sentiment_score):
    """Prepare real-time data for AI/ML prediction."""
    return {
        "price_change": np.random.uniform(-0.05, 0.05),  # Replace with actual price change
        "volatility": np.random.uniform(0.01, 0.05),    # Replace with actual volatility
        "volume": volume,
        "sentiment_score": sentiment_score,
        "rsi": np.random.uniform(30, 70),  # Replace with actual RSI calculation
        "macd_diff": np.random.uniform(-1, 1),  # Replace with actual MACD calculation
    }

# ✅ Real-Time Stock Tracking via WebSocket
def track_stock_event(data):
    """Real-time stock tracking using WebSocket."""
    ticker = data.get("ticker")
    if not ticker:
        return emit("error", {"message": "Ticker is missing"})

    try:
        # Fetch live stock data
        live_data = fetch_live_stock_data(ticker)

        # Preprocess data for prediction
        processed_data = preprocess_live_data(
            live_data["price"], volume=1000, sentiment_score=0.5  # Replace with actual values
        )
        features = pd.DataFrame([processed_data])[feature_columns]

        # Predict using the XGBoost model
        prediction = xgb_model.predict(features)[0]

        # Emit real-time stock update
        emit("stock_update", {
            "ticker": live_data["ticker"],
            "price": live_data["price"],
            "timestamp": live_data["timestamp"],
            "recommendation": "Buy" if prediction == 1 else "Sell" if prediction == -1 else "Hold"
        })
    except Exception as e:
        emit("error", {"message": str(e)})
