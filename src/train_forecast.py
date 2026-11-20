"""Train the 7-day BTC close forecaster and save model artifacts.

Outputs (in --model-dir):
    model.joblib  - dict with the fitted model, version and recent closes
    metrics.json  - MAE / RMSE / MAPE on the latest 7-day test window
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import HORIZON, MIN_HISTORY, build_dataset, clean_prices, forecast_prices

DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/raw/btc_usd.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
TRAIN_WINDOW_DAYS = 1460  # use the most recent four years
RECENT_CLOSES = 60


def load_close(path: str) -> pd.Series:
    df = clean_prices(pd.read_csv(path))
    return df["Close"].tail(TRAIN_WINDOW_DAYS + HORIZON)


def fit_model(close: pd.Series, n_estimators: int = 150) -> GradientBoostingRegressor:
    x_train, y_train = build_dataset(close)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    return model.fit(x_train, y_train)


def backtest(close: pd.Series, n_estimators: int = 150) -> dict:
    """Train on everything but the last 7 days, forecast them, compare to actuals."""
    train, test = close.iloc[:-HORIZON], close.iloc[-HORIZON:]
    model = fit_model(train, n_estimators)
    preds = forecast_prices(model, train.to_numpy(), train.index[-1], HORIZON)
    predicted = np.array([p for _, p in preds])
    actual = test.to_numpy()
    errors = predicted - actual
    naive = np.abs(train.iloc[-1] - actual)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors) / actual) * 100),
        "naive_mae": float(np.mean(naive)),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
    }


def _atomic_write(path: Path, writer) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def train(
    data_path: str = DATA_PATH,
    model_dir: str = MODEL_DIR,
    max_mape: float = None,
    n_estimators: int = 150,
) -> dict:
    """Train, evaluate and (if the quality gate passes) register the model."""
    close = load_close(data_path)
    if len(close) < MIN_HISTORY + 2 * HORIZON + 30:
        raise ValueError(f"not enough data to train: {len(close)} rows")

    metrics = backtest(close, n_estimators)
    passed = max_mape is None or metrics["mape"] <= max_mape
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    metrics.update(
        {
            "model_version": version,
            "trained_rows": int(len(close)),
            "threshold_mape": max_mape,
            "registered": bool(passed),
        }
    )
    print(json.dumps(metrics, indent=2))
    if not passed:
        print(
            f"MAPE {metrics['mape']:.2f}% above threshold {max_mape}; not registering"
        )
        return metrics

    final_model = fit_model(close, n_estimators)
    bundle = {
        "model": final_model,
        "version": version,
        "last_date": str(close.index[-1].date()),
        "recent_close": [float(v) for v in close.tail(RECENT_CLOSES)],
    }
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write(out / "model.joblib", lambda p: joblib.dump(bundle, p))
    _atomic_write(
        out / "metrics.json",
        lambda p: Path(p).write_text(json.dumps(metrics, indent=2)),
    )
    print(f"Saved model {version} to {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--max-mape", type=float, default=None)
    args = parser.parse_args()
    train(args.data, args.model_dir, args.max_mape)
