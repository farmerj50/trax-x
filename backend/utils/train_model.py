import os
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import save_model, load_model  # type: ignore # ✅ Import both save & load model
from tensorflow.keras.layers import (  # type: ignore
    Input, Conv1D, BatchNormalization, Dropout, Dense, LSTM, 
    GlobalAveragePooling1D, LeakyReLU, LayerNormalization, MultiHeadAttention, 
    Bidirectional
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# ✅ Import utilities
from utils.fetch_historical_performance import fetch_historical_data
from utils.indicators import preprocess_data_with_indicators

# ✅ Define paths for saving models
MODELS_DIR = "C:/Users/gabby/trax-x/backend/models"
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "optimized_xgb_model.joblib")
XGB_FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_features.pkl")
LSTM_MODEL_PATH_KERAS = os.path.join(MODELS_DIR, "cnn_lstm_attention_model.keras")  # ✅ Keras Format
LSTM_MODEL_PATH_H5 = os.path.join(MODELS_DIR, "cnn_lstm_attention_model.h5")  # ✅ H5 Format
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "cnn_lstm_attention_scaler.pkl")

# ✅ Ensure directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ✅ Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Cache for LSTM Model
lstm_cache = {"model": None, "scaler": None}


def preprocess_for_lstm(data, features, target, time_steps=50):
    """
    Prepares data for LSTM training.
    
    - Scales feature columns
    - Reshapes into (batch_size, time_steps, features)

    Returns:
    - X (numpy array): Scaled input sequences
    - y (numpy array): Target values
    - scaler (StandardScaler object): Used for feature scaling
    """
    try:
        if len(data) < time_steps:
            logger.warning(f"⚠️ Not enough data for LSTM: {len(data)} rows. Required: {time_steps}.")
            return None, None, None

        # ✅ Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i - time_steps:i])
            y.append(data[target].iloc[i])

        return np.array(X), np.array(y), scaler

    except Exception as e:
        logger.error(f"❌ Error in preprocess_for_lstm: {e}")
        return None, None, None


def train_cnn_lstm_model(data, features, target, time_steps=150):
    """
    Train a CNN-LSTM model with attention for stock price prediction.
    """
    try:
        # ✅ Preprocess data for LSTM
        X, y, scaler = preprocess_for_lstm(data, features, target, time_steps)

        if X is None or y is None:
            raise ValueError("❌ ERROR: Not enough data for LSTM training.")

        # ✅ Define Input Layer
        input_layer = Input(shape=(X.shape[1], X.shape[2]))

        # ✅ CNN Feature Extraction
        cnn_layer = Conv1D(filters=128, kernel_size=3, activation=LeakyReLU(alpha=0.1))(input_layer)
        cnn_layer = BatchNormalization()(cnn_layer)
        cnn_layer = Dropout(0.3)(cnn_layer)

        # ✅ Attention Mechanism
        attention_layer = MultiHeadAttention(num_heads=4, key_dim=64)(cnn_layer, cnn_layer)
        attention_layer = LayerNormalization()(attention_layer)

        # ✅ LSTM Layers
        lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(attention_layer)
        lstm_layer = BatchNormalization()(lstm_layer)
        lstm_layer = Dropout(0.3)(lstm_layer)

        lstm_layer = LSTM(64, return_sequences=True)(lstm_layer)
        lstm_layer = GlobalAveragePooling1D()(lstm_layer)

        # ✅ Fully Connected Dense Layers
        dense_layer = Dense(64, activation=LeakyReLU(alpha=0.1))(lstm_layer)
        dense_layer = Dropout(0.2)(dense_layer)
        dense_layer = Dense(32, activation="swish")(dense_layer)
        output_layer = Dense(1)(dense_layer)

        # ✅ Compile Model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error")

        # ✅ Train Model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        model.fit(X, y, epochs=300, batch_size=128, validation_split=0.2, verbose=1,
                  callbacks=[early_stopping, reduce_lr])

        # ✅ Save Model in Both Formats
        model.save(LSTM_MODEL_PATH_KERAS)
        model.save(LSTM_MODEL_PATH_H5, save_format="h5")  # ✅ H5 Format

        joblib.dump(scaler, LSTM_SCALER_PATH)

        logger.info(f"✅ Model saved at: {LSTM_MODEL_PATH_KERAS} and {LSTM_MODEL_PATH_H5}")
        logger.info(f"✅ Scaler saved at: {LSTM_SCALER_PATH}")

        return model, scaler

    except Exception as e:
        logger.error(f"❌ Error in train_cnn_lstm_model: {e}")
        raise


def load_lstm_model():
    """
    Load the LSTM model and scaler from disk if available.
    Tries loading from `.keras` first, then `.h5` if necessary.
    """
    try:
        if os.path.exists(LSTM_SCALER_PATH):
            scaler = joblib.load(LSTM_SCALER_PATH)
        else:
            print("⚠️ LSTM scaler not found.")
            return None, None

        # ✅ Try Loading `.keras` Model First
        if os.path.exists(LSTM_MODEL_PATH_KERAS):
            print(f"✅ Loading LSTM model from: {LSTM_MODEL_PATH_KERAS}")
            model = load_model(LSTM_MODEL_PATH_KERAS)
            return model, scaler

        # ✅ If `.keras` fails, Try Loading `.h5`
        elif os.path.exists(LSTM_MODEL_PATH_H5):
            print(f"✅ Loading LSTM model from: {LSTM_MODEL_PATH_H5}")
            model = load_model(LSTM_MODEL_PATH_H5)
            return model, scaler

        else:
            print("⚠️ No LSTM model found. Retrain needed.")
            return None, None

    except Exception as e:
        print(f"❌ Error in load_lstm_model: {e}")
        return None, None


def train_and_cache_lstm_model():
    """
    Train the LSTM model and cache it for future use.
    """
    try:
        logger.info("📌 Fetching historical stock data...")
        data = fetch_historical_data()

        if data is None or data.empty:
            raise ValueError("❌ No historical data available for training.")

        # ✅ Preprocess Data
        data, scaler = preprocess_data_with_indicators(data)

        # ✅ Train Model
        model, scaler = train_cnn_lstm_model(data, features=["price_change", "volatility", "volume", "rsi"], target="close")

        # ✅ Cache the model
        lstm_cache["model"], lstm_cache["scaler"] = model, scaler

        return model, scaler

    except Exception as e:
        logger.error(f"❌ Error training and saving LSTM model: {e}")
        return None, None  # Prevent app crash
