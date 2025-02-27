import os
import joblib
import logging
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from utils.fetch_historical_performance import fetch_historical_data
from utils.indicators import preprocess_data_with_indicators

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Define Correct Model Paths
MODELS_DIR = r"C:\Users\gabby\trax-x\backend\models"
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "optimized_xgb_model.joblib")
XGB_FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_features.pkl")

# ✅ Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def objective(trial, X, y):
    try:
        # ✅ Validate scale_pos_weight calculation
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)  # Avoid division by zero
        logging.info(f"📌 Calculated scale_pos_weight: {pos_weight}")

        params = {
            "scale_pos_weight": pos_weight,
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        }

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        return cross_val_score(model, X_val, y_val, cv=3, scoring="accuracy").mean()

    except Exception as e:
        logging.error(f"❌ ERROR in Optuna objective function: {e}")
        return 0
    
def tune_xgboost_hyperparameters(X_train, y_train, n_trials=50):
    """Uses Optuna to find the best hyperparameters for XGBoost."""
    try:
        logging.info("📌 Starting XGBoost hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

        if len(study.trials) == 0:
            logging.error("❌ ERROR: No trials were completed in Optuna.")
            return None, {}

        best_params = study.best_params
        logging.info(f"✅ Best XGBoost Parameters: {best_params}")

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
        best_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return best_model, best_params

    except Exception as e:
        logging.error(f"❌ ERROR in tune_xgboost_hyperparameters: {e}")
        return None, {}
    
def train_xgboost_with_optuna():
    try:
        logging.info("📌 Fetching historical stock data...")
        df = fetch_historical_data()

        if df is None or df.empty:
            raise ValueError("❌ ERROR: No historical stock data available for training.")

        # ✅ Ensure `preprocess_data_with_indicators` always returns a tuple
        processed_result = preprocess_data_with_indicators(df)

        # ✅ Debugging: Print type of returned value
        print(f"🔹 Type of processed_result: {type(processed_result)}")

        # ✅ Ensure it's a valid tuple
        if not isinstance(processed_result, tuple) or len(processed_result) != 2:
            raise TypeError(f"❌ preprocess_data_with_indicators did NOT return a valid tuple! Got: {type(processed_result)}")

        df, _ = processed_result  # ✅ Properly unpack df

        # ✅ Ensure df is a DataFrame before proceeding
        if not isinstance(df, pd.DataFrame):
            print(f"❌ ERROR: df is NOT a DataFrame! Instead, it is {type(df)}")
            raise TypeError("❌ preprocess_data_with_indicators did not return a DataFrame!")

        print(f"🔹 Columns in df: {df.columns.tolist()}")

        # ✅ Ensure required features exist
        required_features = [
            "price_change", "volatility", "volume", "rsi",
            "macd_diff", "adx", "atr", "mfi", "macd_line", "macd_signal"
        ]

        for col in required_features:
            if col not in df.columns:
                df[col] = 0
                logging.warning(f"⚠️ Missing column '{col}' filled with 0.")

        # ✅ Ensure 'buy_signal' column exists
        if "buy_signal" not in df.columns:
            print("❌ ERROR: 'buy_signal' column is missing in DataFrame!")
            print(f"🔹 Available columns: {df.columns.tolist()}")
            raise KeyError("❌ 'buy_signal' column is required but missing!")

        X = df[required_features]
        y = df["buy_signal"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_model, best_params = tune_xgboost_hyperparameters(X_train, y_train)

        if best_model:
            joblib.dump(best_model, XGB_MODEL_PATH)
            joblib.dump(required_features, XGB_FEATURES_PATH)  # ✅ Save feature list
            logging.info(f"✅ XGBoost Model saved at: {XGB_MODEL_PATH}")
            logging.info(f"✅ Features saved at: {XGB_FEATURES_PATH}")
        else:
            logging.error("❌ ERROR: Optuna failed to train a valid model.")

        return best_model, best_params

    except Exception as e:
        logging.error(f"❌ ERROR in train_xgboost_with_optuna: {e}", exc_info=True)
        return None, None


if __name__ == "__main__":
    train_xgboost_with_optuna()
