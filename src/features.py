"""Shared feature engineering and recursive forecasting helpers.

Used by the training script, the Airflow DAG and the Flask API so that the
features seen at training time are exactly the ones used at prediction time.
"""

import numpy as np
import pandas as pd

LAGS = 7
HORIZON = 7
MIN_HISTORY = 40  # 30-day moving average + lags need at least this many closes
FEATURE_COLUMNS = [f"ret_lag_{i}" for i in range(1, LAGS + 1)] + [
    "ma7_ratio",
    "ma30_ratio",
    "vol7",
]


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean Date-indexed frame with a positive, numeric Close column."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"]).sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7/30-day moving averages, daily return and 7-day volatility."""
    out = df.copy()
    out["ma7"] = out["Close"].rolling(7).mean()
    out["ma30"] = out["Close"].rolling(30).mean()
    out["return"] = np.log(out["Close"]).diff()
    out["volatility7"] = out["return"].rolling(7).std()
    return out


def _feature_frame(close: pd.Series) -> pd.DataFrame:
    ret = np.log(close).diff()
    feats = pd.DataFrame(index=close.index)
    for i in range(1, LAGS + 1):
        feats[f"ret_lag_{i}"] = ret.shift(i - 1)
    feats["ma7_ratio"] = close / close.rolling(7).mean() - 1
    feats["ma30_ratio"] = close / close.rolling(30).mean() - 1
    feats["vol7"] = ret.rolling(7).std()
    return feats[FEATURE_COLUMNS]


def build_dataset(close: pd.Series):
    """Features known at the end of day t and the target (next-day log return)."""
    feats = _feature_frame(close)
    target = np.log(close).diff().shift(-1).rename("target")
    data = feats.join(target).dropna()
    return data[FEATURE_COLUMNS], data["target"]


def forecast_prices(model, close: pd.Series, last_date, horizon: int = HORIZON):
    """Recursively forecast `horizon` future closes.

    Returns a list of (date_string, predicted_close).
    """
    if len(close) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} closing prices")
    history = pd.Series(
        np.asarray(close, dtype=float)[-(MIN_HISTORY + 20) :],
    )
    date = pd.Timestamp(last_date)
    results = []
    for _ in range(horizon):
        features = _feature_frame(history).iloc[[-1]]
        daily_return = float(model.predict(features)[0])
        next_close = float(history.iloc[-1] * np.exp(daily_return))
        date = date + pd.Timedelta(days=1)
        results.append((date.strftime("%Y-%m-%d"), round(next_close, 2)))
        history = pd.concat([history, pd.Series([next_close])], ignore_index=True)
    return results
