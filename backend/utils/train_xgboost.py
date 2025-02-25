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

def plot_feature_importance(model, feature_names):
    """Plot the feature importance from the trained XGBoost model."""
    try:
        importance = model.get_booster().get_score(importance_type="weight")
        if not importance:
            logging.warning("⚠️ No feature importance found in model.")
            return

        importance_df = pd.DataFrame(importance.items(), columns=["Feature", "Importance"])
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("XGBoost Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()

    except Exception as e:
        logging.error(f"❌ ERROR in plot_feature_importance: {e}")

def train_xgboost_with_optuna():
    try:
        logging.info("📌 Fetching historical stock data...")
        df = fetch_historical_data()
        if df is None or df.empty:
            raise ValueError("❌ ERROR: No historical stock data available for training.")

        df = preprocess_data_with_indicators(df)

        # ✅ Debugging: Ensure buy_signal exists before training
        logging.info(f"📌 Buy Signal Distribution Before Training:\n{df['buy_signal'].value_counts()}")

        required_features = ["price_change", "volatility", "volume", "rsi", "macd_diff", "adx", "atr", "mfi"]
        missing_columns = [col for col in required_features if col not in df.columns]
        if missing_columns:
            logging.warning(f"⚠️ Missing columns: {missing_columns}. Filling with 0.")
            for col in missing_columns:
                df[col] = 0

        X = df[required_features]
        y = df["buy_signal"]  # Ensure this column exists

        logging.info(f"✅ Total Samples: {len(y)}, Buy Signals: {y.sum()}, No-Buy Signals: {(y == 0).sum()}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_model, best_params = tune_xgboost_hyperparameters(X_train, y_train)

        if not best_model or not best_params:
            logging.error("❌ ERROR: Optuna failed to train a valid model.")
            return None, {}

        plot_feature_importance(best_model, X_train.columns)
        joblib.dump(best_model, XGB_MODEL_PATH)
        joblib.dump(list(X.columns), XGB_FEATURES_PATH)
        logging.info(f"✅ XGBoost Model saved at: {XGB_MODEL_PATH}")

        return best_model, best_params

    except Exception as e:
        logging.error(f"❌ ERROR in train_xgboost_with_optuna: {e}")
        return None, {}


if __name__ == "__main__":
    train_xgboost_with_optuna()
