"""Flask REST API serving the 7-day Bitcoin close-price forecast."""

import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from features import MIN_HISTORY, forecast_prices  # noqa: E402

MAX_FORECAST_DAYS = 30

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("btc-api")

app = Flask(__name__)
START_TIME = time.time()
_lock = threading.Lock()
_state = {"bundle": None, "metrics": {}, "source": None, "mtime": None}


class ValidationError(Exception):
    """Raised for invalid client input (HTTP 400)."""


# ---------------------------------------------------------------- model loading
def _model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", HERE.parent / "models"))


def _seed_dir() -> Path:
    return Path(os.getenv("SEED_MODEL_DIR", HERE.parent / "models" / "seed"))


def _locate_model() -> Path:
    """Prefer the live model directory; fall back to the seed baked into the image."""
    for directory in (_model_dir(), _seed_dir()):
        if (directory / "model.joblib").exists():
            return directory
    raise FileNotFoundError("no model.joblib in MODEL_DIR or SEED_MODEL_DIR")


def _load(directory: Path) -> None:
    model_file = directory / "model.joblib"
    bundle = joblib.load(model_file)
    metrics_file = directory / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    _state.update(
        bundle=bundle,
        metrics=metrics,
        source=str(directory),
        mtime=model_file.stat().st_mtime,
    )
    log.info("Loaded model %s from %s", bundle["version"], directory)


def get_model(force: bool = False) -> dict:
    """Return the loaded model, reloading when model.joblib changed on disk.

    Checking the file's mtime on each request lets every replica pick up a
    retrained model from the shared volume without a restart.
    """
    with _lock:
        directory = _locate_model()
        mtime = (directory / "model.joblib").stat().st_mtime
        stale = _state["source"] != str(directory) or _state["mtime"] != mtime
        if force or _state["bundle"] is None or stale:
            _load(directory)
        return _state


def reset_state() -> None:
    with _lock:
        _state.update(bundle=None, metrics={}, source=None, mtime=None)


# ------------------------------------------------------------------ validation
def _parse_request() -> dict:
    if not request.get_data(cache=True).strip():
        return {}
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValidationError("request body must be valid JSON")
    if not isinstance(payload, dict):
        raise ValidationError("JSON body must be an object")
    return payload


def _validate_days(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("forecast_days must be an integer")
    if not 1 <= value <= MAX_FORECAST_DAYS:
        raise ValidationError(
            f"forecast_days must be between 1 and {MAX_FORECAST_DAYS}"
        )
    return value


def _validate_history(value) -> list:
    if not isinstance(value, list):
        raise ValidationError("history must be a list of closing prices")
    if len(value) < MIN_HISTORY:
        raise ValidationError(f"history needs at least {MIN_HISTORY} closing prices")
    prices = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError("history must contain only numbers")
        if not math.isfinite(item) or item <= 0:
            raise ValidationError("history prices must be positive and finite")
        prices.append(float(item))
    return prices


# --------------------------------------------------------------------- routes
@app.get("/health")
def health():
    state = get_model()
    return jsonify(
        status="ok",
        uptime_seconds=round(time.time() - START_TIME, 1),
        model_version=state["bundle"]["version"],
        model_stage=os.getenv("MODEL_STAGE", "development"),
    )


@app.post("/predict")
def predict():
    payload = _parse_request()
    days = _validate_days(payload.get("forecast_days", 7))
    state = get_model()
    bundle = state["bundle"]

    if "history" in payload and payload["history"] is not None:
        closes = _validate_history(payload["history"])
        last_date = datetime.now(timezone.utc).date()
    else:
        closes = bundle["recent_close"]
        last_date = bundle["last_date"]

    rows = forecast_prices(bundle["model"], closes, last_date, days)
    return jsonify(
        model_version=bundle["version"],
        forecast_days=days,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        forecasts=[{"date": d, "predicted_close": p} for d, p in rows],
    )


@app.get("/metrics")
def metrics():
    state = get_model()
    return jsonify(model_version=state["bundle"]["version"], metrics=state["metrics"])


@app.post("/reload")
def reload_model():
    token = os.getenv("RELOAD_TOKEN")
    if token and request.headers.get("X-Reload-Token") != token:
        return jsonify(error="invalid or missing X-Reload-Token"), 401
    state = get_model(force=True)
    return jsonify(status="reloaded", model_version=state["bundle"]["version"])


# -------------------------------------------------------------- error handling
@app.errorhandler(ValidationError)
def handle_validation(err):
    return jsonify(error=str(err)), 400


@app.errorhandler(HTTPException)
def handle_http(err):
    return jsonify(error=err.name.lower()), err.code


@app.errorhandler(Exception)
def handle_unexpected(err):
    log.exception("Unhandled error: %s", err)
    return jsonify(error="internal server error"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
