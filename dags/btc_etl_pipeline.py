"""Daily BTC-USD pipeline: fetch -> preprocess -> retrain/evaluate -> reload API."""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models")
RAW_PATH = f"{DATA_DIR}/raw/btc_usd.csv"
PROCESSED_PATH = f"{DATA_DIR}/processed/btc_features.csv"
API_RELOAD_URL = os.getenv("API_RELOAD_URL", "http://api:5000/reload")
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "")
MAX_MAPE = float(os.getenv("MAX_MAPE", "15"))  # quality gate for the 7-day window


def fetch_data() -> int:
    """Task 1: pull BTC-USD daily data and save it to shared storage."""
    from etl_fetch import main

    return main(RAW_PATH)


def preprocess_data() -> int:
    """Task 2: clean data and add 7/30-day moving averages and volatility."""
    from preprocess import preprocess

    return preprocess(RAW_PATH, PROCESSED_PATH)


def train_and_evaluate() -> dict:
    """Task 3: retrain, evaluate on the latest 7-day window, register if it passes."""
    from train_forecast import train

    return train(PROCESSED_PATH, MODEL_DIR, max_mape=MAX_MAPE)


def metrics_passed(ti) -> bool:
    """Gate: stop the run here (skip the reload) if the new model was rejected."""
    result = ti.xcom_pull(task_ids="train_and_evaluate")
    return bool(result and result.get("registered"))


def reload_api() -> None:
    """Task 4: tell the Flask service to re-read the updated model."""
    import requests

    resp = requests.post(
        API_RELOAD_URL, headers={"X-Reload-Token": RELOAD_TOKEN}, timeout=30
    )
    resp.raise_for_status()
    print(resp.json())


default_args = {"owner": "mlops", "retries": 2, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, model retraining and API reload",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["btc", "mlops"],
) as dag:
    fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    preprocess = PythonOperator(
        task_id="preprocess_features", python_callable=preprocess_data
    )
    train_eval = PythonOperator(
        task_id="train_and_evaluate", python_callable=train_and_evaluate
    )
    gate = ShortCircuitOperator(
        task_id="check_quality_threshold", python_callable=metrics_passed
    )
    reload_task = PythonOperator(task_id="reload_api_model", python_callable=reload_api)

    fetch >> preprocess >> train_eval >> gate >> reload_task
