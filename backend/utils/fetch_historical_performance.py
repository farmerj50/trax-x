import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from cachetools import TTLCache
import os

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Caching Mechanism (5-minute TTL)
historical_data_cache = TTLCache(maxsize=10, ttl=300)

# ✅ API Key for Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "swpC4ge5_aGqdJll3gplZ6a40ADuwhzG")  # Replace if needed
print(f"Using API Key: {os.getenv('POLYGON_API_KEY')}")

def get_valid_date():
    """
    Get the most recent valid stock market date (no weekends or future dates).
    """
    today = datetime.utcnow()
    for i in range(7):  # Check last 7 days
        check_date = today - timedelta(days=i)
        if check_date.weekday() < 5:  # ✅ Monday-Friday (0-4)
            return check_date.strftime("%Y-%m-%d")
    return today.strftime("%Y-%m-%d")  # Fallback

def fetch_historical_data():
    """
    Fetch historical stock data from Polygon.io.
    If today's data is missing, it falls back to the most recent available trading day.
    """
    for i in range(360):  # ✅ Try fetching data for the last 360 days
        most_recent_date = datetime.utcnow() - timedelta(days=i)  # ✅ Ensure UTC consistency
        most_recent_date_str = most_recent_date.strftime("%Y-%m-%d")
        logging.info(f"🔍 Attempting to fetch stock data for: {most_recent_date_str}")

        # ✅ Check if data is already cached
        if most_recent_date_str in historical_data_cache:
            logging.info(f"✅ Returning cached data for {most_recent_date_str}")
            return historical_data_cache[most_recent_date_str]

        url = (
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
            f"{most_recent_date_str}?adjusted=true&apiKey={POLYGON_API_KEY}"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # ✅ Ensure 'results' exists and contains data
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])  # ✅ Convert JSON to DataFrame

                # ✅ Rename columns for consistency
                rename_mapping = {
                    "T": "ticker",
                    "v": "volume",
                    "vw": "vwap",
                    "o": "open",
                    "c": "close",
                    "h": "high",
                    "l": "low",
                    "t": "timestamp",
                    "n": "trade_count",
                }
                df.rename(columns=rename_mapping, inplace=True)

                # ✅ Preserve ticker column
                if "ticker" not in df.columns:
                    df["ticker"] = "UNKNOWN"  # Assign default placeholder if missing
                    logging.warning("⚠️ 'ticker' column was missing! Placeholder added.")

                logging.info(f"📌 Columns in DataFrame: {df.columns.tolist()}")

                # ✅ Cache the retrieved data to avoid redundant API calls
                historical_data_cache[most_recent_date_str] = df

                return df  # ✅ Return the first valid data found

            logging.warning(f"⚠️ No stock data found for {most_recent_date_str}")

        except requests.exceptions.Timeout:
            logging.error(f"❌ Timeout error while fetching data for {most_recent_date_str}")

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"❌ HTTP error: {http_err}")

        except requests.exceptions.RequestException as req_err:
            logging.error(f"❌ Request error: {req_err}")

        except Exception as e:
            logging.error(f"❌ Unexpected error: {e}")

    logging.warning("❌ Unable to fetch stock data. Returning empty DataFrame.")

    # ✅ Ensure `df` is always defined
    return pd.DataFrame(columns=["ticker", "volume", "vwap", "open", "close", "high", "low", "timestamp", "trade_count"])
