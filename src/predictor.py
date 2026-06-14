"""
Predictor module for gold price forecasting.
Implements Linear Regression, ARIMA, and a naive baseline.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TROY_OZ_TO_GRAM = 31.1035


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    """Create lagged features for time series regression."""
    result = df.copy()
    for lag in lags:
        result[f"lag_{lag}"] = result[target_col].shift(lag)
    return result


def train_test_split_time(df: pd.DataFrame, test_ratio: float = 0.2):
    """Time-series aware train/test split (no shuffling)."""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ---------------------------------------------------------------------------
# Model A: Linear Regression
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train Linear Regression and return model + metrics."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model_name": "Linear Regression",
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
    }

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_
    }).sort_values("coefficient", key=abs, ascending=False)

    return model, metrics, feature_importance, y_pred_test


# ---------------------------------------------------------------------------
# Model B: ARIMA
# ---------------------------------------------------------------------------

def train_arima(train_series, test_series, order: tuple = (1, 1, 1)):
    """Train ARIMA and return model + metrics."""
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    y_pred_test = fitted.forecast(steps=len(test_series))

    metrics = {
        "model_name": f"ARIMA{order}",
        "train_rmse": np.sqrt(mean_squared_error(train_series, fitted.fittedvalues)),
        "test_rmse": np.sqrt(mean_squared_error(test_series, y_pred_test)),
        "train_mae": mean_absolute_error(train_series, fitted.fittedvalues),
        "test_mae": mean_absolute_error(test_series, y_pred_test),
        "train_r2": r2_score(train_series, fitted.fittedvalues),
        "test_r2": r2_score(test_series, y_pred_test),
        "aic": fitted.aic,
        "bic": fitted.bic,
    }

    return fitted, metrics, y_pred_test


def auto_arima_order(series, max_p: int = 3, max_d: int = 2, max_q: int = 3):
    """Find best ARIMA order by AIC (grid search)."""
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    return best_order, best_aic


# ---------------------------------------------------------------------------
# Naive baseline (random walk)
# ---------------------------------------------------------------------------

def naive_forecast(train_series, test_series):
    """Naive forecast: prediction = last observed value (random walk)."""
    last_val = train_series.iloc[-1]
    y_pred = pd.Series(last_val, index=test_series.index)

    metrics = {
        "model_name": "Naive (Random Walk)",
        "train_rmse": np.nan,
        "test_rmse": np.sqrt(mean_squared_error(test_series, y_pred)),
        "train_mae": np.nan,
        "test_mae": mean_absolute_error(test_series, y_pred),
        "train_r2": np.nan,
        "test_r2": r2_score(test_series, y_pred),
    }

    return metrics, y_pred


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from a list of metric dicts."""
    return pd.DataFrame(metrics_list).set_index("model_name")


def predict_price(model, features: np.ndarray) -> float:
    """Predict gold price using a trained model."""
    return model.predict(features.reshape(1, -1))[0]


def save_model(model, name: str) -> Path:
    """Save a model to disk."""
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


def load_model(name: str):
    """Load a model from disk."""
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)
