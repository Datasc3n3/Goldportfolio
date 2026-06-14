"""
Utility functions for the gold portfolio project.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Karat purity conversion (fraction of pure gold)
KARAT_PURITY = {
    14: 14 / 24,
    18: 18 / 24,
    21: 21 / 24,
    22: 22 / 24,
    24: 24 / 24,
}


def load_portfolio() -> pd.DataFrame:
    """Load the raw portfolio CSV."""
    path = PROJECT_ROOT / "PORTFOLIO.csv"
    df = pd.read_csv(path)
    df["datum_aankoop"] = pd.to_datetime(df["datum_aankoop"])
    return df


def calculate_pure_gold_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Add column with pure gold weight (gram * karat_purity)."""
    df = df.copy()
    df["karaat_purity"] = df["karaat"].map(KARAT_PURITY)
    df["zuiver_goud_gram"] = df["gram"] * df["karaat_purity"]
    return df


def filter_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Filter only gold bars (staaf) from portfolio."""
    return df[df["type"] == "staaf"].copy()


def filter_jewelry(df: pd.DataFrame) -> pd.DataFrame:
    """Filter only jewelry (juweel) from portfolio."""
    return df[df["type"] == "juweel"].copy()


def get_spot_price_at_date(gold_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Get gold spot price (EUR/gram) closest to a given date."""
    idx = gold_df.index.get_indexer([date], method="nearest")[0]
    if "gold_eur_gram" in gold_df.columns:
        return gold_df.iloc[idx]["gold_eur_gram"]
    elif "gold_usd_oz" in gold_df.columns:
        # Fallback: convert USD/oz to EUR/gram (approximate)
        TROY_OZ_TO_GRAM = 31.1035
        return gold_df.iloc[idx]["gold_usd_oz"] / TROY_OZ_TO_GRAM
    return np.nan


def calculate_portfolio_value(portfolio_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    """Calculate total portfolio statistics."""
    bars = filter_bars(portfolio_df)
    jewelry = filter_jewelry(portfolio_df)

    total_invested = bars["aankoopprijs"].sum()
    total_weight = bars["gram"].sum()
    total_pure_gold = bars["zuiver_goud_gram"].sum() if "zuiver_goud_gram" in bars.columns else total_weight * 0.999

    # Current value (using latest spot price)
    if "gold_eur_gram" in gold_df.columns:
        current_spot = gold_df["gold_eur_gram"].iloc[-1]
    else:
        current_spot = np.nan

    current_value_bars = total_weight * current_spot if not np.isnan(current_spot) else np.nan

    # Jewelry estimated value (pure gold content * spot price)
    jewelry_weight = jewelry["gram"].sum()
    jewelry_pure = jewelry["zuiver_goud_gram"].sum() if "zuiver_goud_gram" in jewelry.columns else 0
    jewelry_value = jewelry_pure * current_spot if not np.isnan(current_spot) else 0

    total_value = current_value_bars + jewelry_value if not np.isnan(current_value_bars) else np.nan
    roi = ((total_value - total_invested) / total_invested * 100) if total_invested > 0 and not np.isnan(total_value) else np.nan

    return {
        "total_bars": len(bars),
        "total_jewelry": len(jewelry),
        "total_weight_gram": total_weight,
        "total_pure_gold_gram": total_pure_gold,
        "total_invested_eur": total_invested,
        "current_spot_eur_gram": current_spot,
        "current_value_bars_eur": current_value_bars,
        "jewelry_weight_gram": jewelry_weight,
        "jewelry_value_eur": jewelry_value,
        "total_portfolio_value_eur": total_value,
        "roi_percent": roi,
    }


def goal_progress(current_value: float, goal: float = 100_000) -> dict:
    """Calculate progress toward the EUR 100,000 goal."""
    pct = (current_value / goal * 100) if goal > 0 else 0
    remaining = goal - current_value
    return {
        "goal_eur": goal,
        "current_eur": current_value,
        "remaining_eur": remaining,
        "progress_percent": pct,
        "reached": current_value >= goal,
    }


def next_purchase_date(last_purchase: pd.Timestamp, interval_months: int = 2) -> pd.Timestamp:
    """Calculate the next purchase date."""
    return last_purchase + pd.DateOffset(months=interval_months)


def format_eur(value: float) -> str:
    """Format a number as EUR."""
    if np.isnan(value):
        return "N/A"
    return f"EUR {value:,.2f}"


def format_gram(value: float) -> str:
    """Format a number as grams."""
    if np.isnan(value):
        return "N/A"
    return f"{value:,.2f} g"
