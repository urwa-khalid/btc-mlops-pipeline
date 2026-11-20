"""Unit tests for the Flask API. They train a tiny model on synthetic prices,
so no internet access or pre-trained model is needed."""

import numpy as np
import pandas as pd
import pytest

import app as api
import train_forecast


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    prices = 30000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    csv = tmp_path_factory.mktemp("data") / "prices.csv"
    pd.DataFrame({"Date": dates, "Close": prices}).to_csv(csv, index=False)
    out = tmp_path_factory.mktemp("models")
    train_forecast.train(str(csv), str(out), n_estimators=20)
    return out


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.delenv("RELOAD_TOKEN", raising=False)
    api.reset_state()
    api.app.config.update(TESTING=False)  # exercise the real error handlers
    return api.app.test_client()


def test_health(client):
    resp = client.get("/health")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0
    assert body["model_version"]


def test_metrics(client):
    body = client.get("/metrics").get_json()
    for key in ("mae", "rmse", "mape"):
        assert body["metrics"][key] >= 0


def test_predict_default_is_seven_days(client):
    resp = client.post("/predict")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["forecast_days"] == 7
    assert len(body["forecasts"]) == 7
    dates = [row["date"] for row in body["forecasts"]]
    assert dates == sorted(dates)
    assert all(row["predicted_close"] > 0 for row in body["forecasts"])


def test_predict_custom_days(client):
    body = client.post("/predict", json={"forecast_days": 3}).get_json()
    assert len(body["forecasts"]) == 3


def test_predict_custom_history(client):
    history = list(np.linspace(30000, 32000, 60))
    resp = client.post("/predict", json={"history": history})
    assert resp.status_code == 200
    assert len(resp.get_json()["forecasts"]) == 7


@pytest.mark.parametrize("bad_days", [0, -1, 31, "7", 2.5, True, None])
def test_predict_rejects_bad_forecast_days(client, bad_days):
    resp = client.post("/predict", json={"forecast_days": bad_days})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


@pytest.mark.parametrize(
    "bad_history",
    [
        [1, 2, 3],
        "abc",
        [30000] * 39 + ["x"],
        [30000] * 39 + [-5],
        [30000] * 39 + [None],
    ],
)
def test_predict_rejects_bad_history(client, bad_history):
    resp = client.post("/predict", json={"history": bad_history})
    assert resp.status_code == 400


def test_predict_rejects_invalid_json(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400


def test_predict_rejects_non_object_json(client):
    assert client.post("/predict", json=[1, 2, 3]).status_code == 400


def test_unknown_route_returns_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.get_json()["error"]


def test_internal_error_returns_500(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(api, "forecast_prices", boom)
    resp = client.post("/predict")
    assert resp.status_code == 500
    assert resp.get_json()["error"] == "internal server error"


def test_reload_requires_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("RELOAD_TOKEN", "s3cret")
    assert client.post("/reload").status_code == 401
    ok = client.post("/reload", headers={"X-Reload-Token": "s3cret"})
    assert ok.status_code == 200
    assert ok.get_json()["status"] == "reloaded"


def test_falls_back_to_seed_model(model_dir, tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("MODEL_DIR", str(empty))
    monkeypatch.setenv("SEED_MODEL_DIR", str(model_dir))
    api.reset_state()
    assert api.app.test_client().get("/health").status_code == 200
