import pandas as pd

# ✅ Try importing the function
try:
    from utils.sentiment_analysis import fetch_and_process_sentiment_data
    print("✅ Sentiment function imported successfully.")
except ImportError:
    print("❌ ERROR: Could not import `fetch_and_process_sentiment_data`.")

# ✅ Mock data for testing
data = pd.DataFrame({"T": ["AAPL", "TSLA", "AMZN"]})

# ✅ Check if 'T' column exists before applying function
if "T" in data.columns:
    try:
        data["sentiment_score"] = data["T"].apply(fetch_and_process_sentiment_data)
        print("✅ Sentiment scores added successfully:")
        print(data)
    except Exception as e:
        print(f"❌ ERROR: {e}")
else:
    print("⚠️ WARNING: 'T' column missing, skipping sentiment analysis.")
