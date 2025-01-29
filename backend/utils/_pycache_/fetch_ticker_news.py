import requests

# Polygon.io API Key
POLYGON_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key

def fetch_ticker_news(ticker, limit=5):
    """
    Fetch the latest news articles for a specific stock ticker.

    Args:
        ticker (str): The stock ticker symbol.
        limit (int): Number of articles to fetch (default is 5).

    Returns:
        dict: The news data returned by the Polygon API.
    """
    if not ticker:
        raise ValueError("Ticker is required")

    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit={limit}&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching news for {ticker}: {e}")