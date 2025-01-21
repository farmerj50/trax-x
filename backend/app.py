from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime, timedelta
from utils.scheduler import initialize_scheduler
from cachetools import TTLCache, cached
from utils.fetch_stock_performance import fetch_stock_performance
from utils.fetch_ticker_news import fetch_ticker_news  # Import the utility function
from utils.sentiment_plot import fetch_sentiment_trend, generate_sentiment_plot


# Initialize Flask app and scheduler
app = Flask(__name__)
CORS(app)

# Polygon.io API Key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "swpC4ge5_aGqdJll3gplZ6a40ADuwhzG")
if not POLYGON_API_KEY:
    raise ValueError("Polygon.io API key not found. Set POLYGON_API_KEY environment variable.")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Caching (TTLCache)
historical_data_cache = TTLCache(maxsize=10, ttl=300)  # Cache up to 10 requests for 5 minutes

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]

# Function to fetch the most recent trading day
def get_most_recent_trading_day():
    """Determine the most recent trading day, considering weekends and holidays."""
    today = datetime.utcnow()
    holidays_api_url = f"https://api.polygon.io/v1/marketstatus/holidays?apiKey={POLYGON_API_KEY}"
    closed_dates = set()
    try:
        # Fetch holidays from Polygon API
        response = requests.get(holidays_api_url, timeout=10)
        response.raise_for_status()
        holidays = response.json()
        closed_dates = {
            datetime.strptime(holiday["date"], "%Y-%m-%d").date()
            for holiday in holidays if holiday["status"] == "closed"
        }
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to fetch holiday data. Error: {e}")
    for i in range(7):
        date_to_check = today - timedelta(days=i)
        if date_to_check.weekday() < 5 and date_to_check.date() not in closed_dates:
            return date_to_check.strftime("%Y-%m-%d")
    raise ValueError("Could not determine a recent valid trading day.")

# Function to fetch historical stock data with caching
@cached(historical_data_cache)
def fetch_historical_data():
    """Fetch historical stock data for the most recent valid trading day."""
    for i in range(7):
        most_recent_date = datetime.utcnow() - timedelta(days=i)
        most_recent_date_str = most_recent_date.strftime("%Y-%m-%d")
        print(f"Fetching stock data for: {most_recent_date_str}")
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{most_recent_date_str}?adjusted=true&apiKey={POLYGON_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "results" in data and data["results"]:
                return pd.DataFrame(data["results"])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {most_recent_date_str}: {e}")
    raise ValueError("Unable to fetch stock data for any recent trading day.")

# Function to preprocess data with enhanced indicators
def preprocess_data_with_indicators(data):
    """Add volume, sentiment score, and enhanced technical indicators."""
    data.rename(columns={"v": "volume"}, inplace=True)
    data["price_change"] = (data["c"] - data["o"]) / data["o"]
    data["volatility"] = (data["h"] - data["l"]) / data["l"]
    data["sentiment_score"] = data["T"].apply(lambda x: analyze_sentiment(f"News headline for {x}"))
    if not data["c"].isnull().all():
        rsi_indicator = RSIIndicator(close=data["c"], window=14, fillna=True)
        data["rsi"] = rsi_indicator.rsi()
        macd = MACD(close=data["c"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        data["macd_diff"] = macd.macd_diff()
        data["sma_20"] = data["c"].rolling(window=20).mean()
        data["ema_20"] = data["c"].ewm(span=20).mean()
        data["bb_upper"] = data["sma_20"] + 2 * data["c"].rolling(window=20).std()
        data["bb_lower"] = data["sma_20"] - 2 * data["c"].rolling(window=20).std()
    else:
        data.update({"rsi": 0, "macd_diff": 0, "sma_20": 0, "ema_20": 0, "bb_upper": 0, "bb_lower": 0})
    data["volume_surge"] = data["volume"] / data["volume"].rolling(window=5).mean()
    return data.fillna(0)

# Train XGBoost Model
def train_xgboost_model():
    """Train the XGBoost model using updated features."""
    data = fetch_historical_data()
    data = preprocess_data_with_indicators(data)
    data["target"] = (data["h"] >= data["c"] * 1.05).astype(int)
    features = ["price_change", "volatility", "volume", "sentiment_score", "rsi", "macd_diff", "sma_20", "ema_20", "bb_upper", "bb_lower", "volume_surge"]
    X_train, X_test, y_train, y_test = train_test_split(data[features], data["target"], test_size=0.2, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    print("Model Evaluation:", classification_report(y_test, model.predict(X_test)))
    return model, features

# Initialize model
xgb_model, feature_columns = train_xgboost_model()
@app.route('/api/stock-performance', methods=['GET'])
def stock_performance():
    """API to fetch stock performance details."""
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({"error": "Ticker parameter is missing"}), 400

        # Fetch performance details using the utility function
        stock_data = fetch_stock_performance(ticker, POLYGON_API_KEY)
        return jsonify(stock_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/scan-stocks', methods=['GET'])
def scan_stocks():
    try:
        min_price = float(request.args.get("min_price", 0))
        max_price = float(request.args.get("max_price", float("inf")))
        data = fetch_historical_data()
        data = preprocess_data_with_indicators(data)
        data["prediction"] = xgb_model.predict(data[feature_columns])
        filtered_data = data[(data["prediction"] == 1) & (data["c"] >= min_price) & (data["c"] <= max_price) & (data["volume_surge"] > 1.2)].head(20)
        return jsonify({"candidates": filtered_data.to_dict(orient="records")}), 200
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
    app.run(port=5000, debug=True)