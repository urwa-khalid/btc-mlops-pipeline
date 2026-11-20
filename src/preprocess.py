"""Clean raw BTC-USD data and add technical indicators."""

import argparse
import os
from pathlib import Path

import pandas as pd

from features import add_indicators, clean_prices

RAW = os.getenv("RAW_DATA_PATH", "data/raw/btc_usd.csv")
PROCESSED = os.getenv("PROCESSED_DATA_PATH", "data/processed/btc_features.csv")


def preprocess(raw_path: str = RAW, output: str = PROCESSED) -> int:
    df = clean_prices(pd.read_csv(raw_path))
    df = add_indicators(df)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index_label="Date")
    print(f"Wrote {len(df)} processed rows to {output}")
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=RAW)
    parser.add_argument("--output", default=PROCESSED)
    args = parser.parse_args()
    preprocess(args.input, args.output)
