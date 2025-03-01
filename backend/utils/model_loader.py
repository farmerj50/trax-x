import os
import joblib
import logging
from utils.train_xgboost import train_xgboost_with_optuna  # ✅ Import training function

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Define Model Paths
MODELS_DIR = r"C:\Users\gabby\trax-x\backend\models"
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "optimized_xgb_model.joblib")
XGB_FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_features.pkl")

# ✅ Global Cache for XGBoost
xgb_cache = {"model": None, "features": None}

def load_xgb_model(force_retrain=False):
    """
    Load the trained XGBoost model and feature list.
    Caches the model in memory to avoid reloading on every API request.
    If the model is missing, it triggers training.
    """
    try:
        if xgb_cache["model"] is None or force_retrain:
            logging.info("📌 Loading trained XGBoost model from file...")

            # ✅ If the model file is missing, train a new one
            if not os.path.exists(XGB_MODEL_PATH) or force_retrain:
                logging.warning("⚠️ XGBoost model not found! Training a new model...")
                train_xgboost_with_optuna()  # ✅ Train new model if missing
          
            # ✅ Load the trained model
            xgb_cache["model"] = joblib.load(XGB_MODEL_PATH)
            xgb_cache["features"] = joblib.load(XGB_FEATURES_PATH)

            logging.info("✅ XGBoost model loaded successfully!")

        return xgb_cache["model"], xgb_cache["features"]

    except Exception as e:
        logging.error(f"❌ ERROR: Unable to load XGBoost model: {e}", exc_info=True)
        return None, None

# ✅ Ensure Model is Loaded at Startup
load_xgb_model()
