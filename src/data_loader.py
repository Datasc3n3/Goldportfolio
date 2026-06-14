"""
Data loader module for fetching gold prices and macro-economic indicators.
Uses yfinance for market data.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ticker definitions
TICKERS = {
    "gold_spot": "GC=F",        # Gold futures (USD/oz)
    "eur_usd": "EURUSD=X",      # EUR/USD exchange rate
    "gold_etf": "GLD",           # SPDR Gold Shares (sentiment proxy)
}


def fetch_gold_price(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch daily gold spot price (USD per troy ounce) from Yahoo Finance."""
    if end is None:
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching gold price from {start} to {end}...")
    data = yf.download(TICKERS["gold_spot"], start=start, end=end, progress=False)
    # Handle MultiIndex columns from yfinance (Price, Ticker format)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Close"]].rename(columns={"Close": "gold_usd_oz"})
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    data = data.dropna()
    print(f"  -> {len(data)} trading days fetched.")
    return data


def fetch_eur_usd(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch daily EUR/USD exchange rate."""
    if end is None:
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching EUR/USD from {start} to {end}...")
    data = yf.download(TICKERS["eur_usd"], start=start, end=end, progress=False)
    # Handle MultiIndex columns from yfinance (Price, Ticker format)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Close"]].rename(columns={"Close": "eur_usd"})
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    data = data.dropna()
    print(f"  -> {len(data)} trading days fetched.")
    return data


def fetch_gold_etf(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch daily GLD ETF data as market sentiment proxy."""
    if end is None:
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching GLD ETF from {start} to {end}...")
    data = yf.download(TICKERS["gold_etf"], start=start, end=end, progress=False)
    # Handle MultiIndex columns from yfinance (Price, Ticker format)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Close", "Volume"]].rename(columns={"Close": "gld_close", "Volume": "gld_volume"})
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    data = data.dropna()
    print(f"  -> {len(data)} trading days fetched.")
    return data


def fetch_macro_data(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch macro-economic indicators (EUR interest rates via ECB, CPI proxy)."""
    if end is None:
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print("Fetching macro-economic data...")

    # Use FRED tickers via yfinance for EUR zone data
    macro_tickers = {
        "EUR_INFLATION": "UCLS0000EU0",  # EU HICP (inflation proxy)
    }

    frames = {}
    for name, ticker in macro_tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                frames[name] = df[["Close"]].rename(columns={"Close": name})
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    if frames:
        macro = pd.concat(frames.values(), axis=1)
        macro.index = pd.to_datetime(macro.index)
        macro.index.name = "date"
        macro = macro.ffill()  # Forward-fill for missing days
        print(f"  -> Macro data: {macro.shape[1]} indicators, {macro.shape[0]} days")
        return macro
    else:
        print("  -> No macro data fetched (tickers may not be available).")
        return pd.DataFrame()


def fetch_all(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch all data sources and merge into a single DataFrame."""
    gold = fetch_gold_price(start, end)
    eurusd = fetch_eur_usd(start, end)
    etf = fetch_gold_etf(start, end)
    macro = fetch_macro_data(start, end)

    # Merge all on date index
    merged = gold.join(eurusd, how="outer")
    merged = merged.join(etf, how="outer")
    if not macro.empty:
        merged = merged.join(macro, how="outer")

    # Forward-fill then drop remaining NaNs at start
    merged = merged.ffill().dropna(subset=["gold_usd_oz"])

    print(f"\nMerged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


def convert_gold_to_eur_per_gram(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns for gold price in EUR per gram."""
    TROY_OZ_TO_GRAM = 31.1035

    if "gold_usd_oz" in df.columns and "eur_usd" in df.columns:
        df = df.copy()
        df["gold_eur_oz"] = df["gold_usd_oz"] / df["eur_usd"]
        df["gold_eur_gram"] = df["gold_eur_oz"] / TROY_OZ_TO_GRAM
        df["gold_10g_eur"] = df["gold_eur_gram"] * 10  # Price of a 10g bar (spot only, no premium)
        print("Added columns: gold_eur_oz, gold_eur_gram, gold_10g_eur")
    return df


def save_raw(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to data/raw/."""
    path = RAW_DATA_DIR / filename
    df.to_csv(path)
    print(f"Saved: {path}")
    return path


def load_raw(filename: str) -> pd.DataFrame:
    """Load DataFrame from data/raw/."""
    path = RAW_DATA_DIR / filename
    return pd.read_csv(path, index_col=0, parse_dates=True)
