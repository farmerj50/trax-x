import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from utils.train_model import train_and_cache_lstm_model  # ✅ Import the function from train_model.py
from tensorflow.keras.utils import get_custom_objects # type: ignore

# Define model paths
MODELS_DIR = r"C:/Users/gabby/trax-x/backend/models"
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_scaler.pkl")

# Cache for models
lstm_cache = {"model": None, "scaler": None}
def load_lstm_model():
    """
    Load the saved LSTM model and scaler, ensuring proper deserialization.
    """
    try:
        print("✅ Checking for saved LSTM model...")

        if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print("⚠️ LSTM model or scaler not found.")
            return None, None

        print("✅ Loading saved LSTM model and scaler...")

        # 🚀 FIX: Load the model with custom objects if necessary
        

        # If your model has custom layers, register them here
        custom_objects = get_custom_objects()

        model = load_model(LSTM_MODEL_PATH, custom_objects=custom_objects, compile=False)

        # 🚀 FIX: Load the scaler correctly
        scaler = joblib.load(SCALER_PATH)

        if model and scaler:
            print("✅ Successfully loaded LSTM model and scaler.")
            return model, scaler
        else:
            print("⚠️ Model or scaler corrupted. Retraining required.")
            return None, None

    except Exception as e:
        print(f"❌ Error in load_lstm_model: {e}")
        return None, None  # Prevent breaking the app

