"""Pull daily BTC-USD market data and store it as a raw CSV."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT = os.getenv("RAW_DATA_PATH", "data/raw/btc_usd.csv")


def _fetch_yfinance() -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        "BTC-USD", period="max", interval="1d", auto_adjust=False, progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError("yfinance returned no rows")
    df.index.name = "Date"
    return df


def _fetch_coingecko() -> pd.DataFrame:
    """Fallback source: daily close and volume for the last 365 days."""
    import requests

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    resp = requests.get(
        url,
        params={"vs_currency": "usd", "days": 365, "interval": "daily"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    prices = pd.DataFrame(data["prices"], columns=["ts", "Close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "Volume"])
    df = prices.merge(volumes, on="ts")
    df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    df = df.drop(columns="ts").drop_duplicates("Date", keep="last")
    return df.set_index("Date")


def fetch_btc() -> pd.DataFrame:
    try:
        return _fetch_yfinance()
    except Exception as exc:  # noqa: BLE001 - fall back to a second data source
        print(f"yfinance failed ({exc}); falling back to CoinGecko", file=sys.stderr)
        return _fetch_coingecko()


def main(output: str = DEFAULT_OUTPUT) -> int:
    df = fetch_btc()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Saved {len(df)} rows ({df.index.min()} -> {df.index.max()}) to {path}")
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
