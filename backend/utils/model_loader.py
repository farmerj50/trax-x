import os
import joblib
import logging

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Define Model Paths
MODELS_DIR = r"C:\Users\gabby\trax-x\backend\models"
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "optimized_xgb_model.joblib")
XGB_FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_features.pkl")

# ✅ Global Cache for XGBoost
xgb_cache = {"model": None, "features": None}

def load_xgb_model():
    """
    Load the trained XGBoost model and feature list.
    Caches the model in memory to avoid reloading on every API request.
    """
    try:
        if xgb_cache["model"] is None:
            logging.info("📌 Loading trained XGBoost model from file...")
            xgb_cache["model"] = joblib.load(XGB_MODEL_PATH)
            xgb_cache["features"] = joblib.load(XGB_FEATURES_PATH)
            logging.info("✅ XGBoost model loaded successfully!")
        return xgb_cache["model"], xgb_cache["features"]
    except Exception as e:
        logging.error(f"❌ ERROR: Unable to load XGBoost model: {e}")
        return None, None

# ✅ Ensure Model is Loaded at Startup
load_xgb_model()
