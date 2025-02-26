import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import get_custom_objects  # type: ignore
from tensorflow.keras.layers import LeakyReLU  # type: ignore
from utils.train_model import train_and_cache_lstm_model  # ✅ Import the function from train_model.py

# Define model paths
MODELS_DIR = r"C:/Users/gabby/trax-x/backend/models"
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_scaler.pkl")

# Cache for models
lstm_cache = {"model": None, "scaler": None}

def load_lstm_model():
    """
    Load the LSTM model and scaler from disk if available.
    """
    try:
        print("✅ Checking for saved LSTM model...")

        # ✅ Verify both files exist before loading
        if not os.path.exists(LSTM_MODEL_PATH):
            print(f"⚠️ LSTM model not found at {LSTM_MODEL_PATH}. Retrain required.")
            return None, None

        if not os.path.exists(SCALER_PATH):
            print(f"⚠️ LSTM scaler not found at {SCALER_PATH}. Retrain required.")
            return None, None

        print("✅ Loading saved LSTM model and scaler...")

        # ✅ Register custom activation functions
        custom_objects = get_custom_objects()
        custom_objects["LeakyReLU"] = LeakyReLU  # Ensure compatibility with saved model

        # ✅ Load Model
        model = load_model(LSTM_MODEL_PATH, custom_objects=custom_objects)

        # ✅ Load Scaler
        scaler = joblib.load(SCALER_PATH)

        print("✅ Successfully loaded LSTM model and scaler.")
        return model, scaler

    except Exception as e:
        print(f"❌ Error in load_lstm_model: {e}")
        return None, None
