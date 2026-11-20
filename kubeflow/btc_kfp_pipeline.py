"""Kubeflow Pipelines (KFP v2) definition: ingest -> features -> train.

Compile to a pipeline spec with:
    python kubeflow/btc_kfp_pipeline.py
"""

import os
from pathlib import Path

from kfp import compiler, dsl

# Image that contains the repo's src/ folder (see kubeflow/Dockerfile.trainer)
TRAINER_IMAGE = os.getenv(
    "TRAINER_IMAGE", "docker.io/YOUR_DOCKERHUB_USERNAME/btc-forecast-trainer:latest"
)


@dsl.container_component
def ingest_data(raw_data: dsl.Output[dsl.Dataset]):
    """Pull daily BTC-USD prices into a raw dataset artifact."""
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "/app/src/etl_fetch.py"],
        args=["--output", raw_data.path],
    )


@dsl.container_component
def generate_features(
    raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]
):
    """Clean the data and add moving averages / volatility indicators."""
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "/app/src/preprocess.py"],
        args=["--input", raw_data.path, "--output", features.path],
    )


@dsl.container_component
def train_model(
    features: dsl.Input[dsl.Dataset],
    max_mape: float,
    model: dsl.Output[dsl.Model],
):
    """Train the 7-day forecaster; model.joblib and metrics.json land in the artifact."""
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "/app/src/train_forecast.py"],
        args=[
            "--data",
            features.path,
            "--model-dir",
            model.path,
            "--max-mape",
            max_mape,
        ],
    )


@dsl.pipeline(
    name="btc-forecast-pipeline",
    description="Daily BTC-USD ingestion, feature generation and 7-day model training",
)
def btc_pipeline(max_mape: float = 15.0):
    ingest = ingest_data()
    features = generate_features(raw_data=ingest.outputs["raw_data"])
    train_model(features=features.outputs["features"], max_mape=max_mape)


if __name__ == "__main__":
    out = Path(__file__).with_name("btc_kfp_pipeline.yaml")
    compiler.Compiler().compile(pipeline_func=btc_pipeline, package_path=str(out))
    print(f"Compiled pipeline to {out}")
