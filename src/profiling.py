"""
Data profiling module.
Provides functions for data quality checks, distributions, and outlier detection.
"""

import pandas as pd
import numpy as np
from IPython.display import display


def profile_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a complete data profile report."""
    print("=" * 60)
    print(f"DATA PROFILE: {name}")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("\n--- Column Types ---")
    print(df.dtypes.to_string())

    print("\n--- Null Counts ---")
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_df = pd.DataFrame({"count": nulls, "percent": null_pct})
    print(null_df.to_string())

    print("\n--- Duplicates ---")
    n_dup = df.duplicated().sum()
    print(f"Duplicate rows: {n_dup} ({n_dup/len(df)*100:.2f}%)")

    print("\n--- Descriptive Statistics ---")
    display(df.describe(include="all"))

    print("\n--- Unique Value Counts (categorical columns) ---")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values")

    print("\n--- First 5 Rows ---")
    display(df.head())

    print("=" * 60)


def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of null values per column."""
    nulls = df.isnull().sum()
    total = len(df)
    return pd.DataFrame({
        "null_count": nulls,
        "null_percent": (nulls / total * 100).round(2),
        "dtype": df.dtypes
    }).sort_values("null_percent", ascending=False)


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.DataFrame:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outliers = series[(series < lower) | (series > upper)]

    return pd.DataFrame({
        "value": outliers,
        "lower_bound": lower,
        "upper_bound": upper,
        "iqr": iqr
    })


def correlation_report(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)


def distribution_summary(series: pd.Series) -> dict:
    """Return distribution statistics for a numeric series."""
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "median": series.median(),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurtosis(),
        "cv": series.std() / series.mean() if series.mean() != 0 else np.nan,
    }


def validate_portfolio(df: pd.DataFrame) -> dict:
    """Validate portfolio data quality and return issues found."""
    issues = []

    # Check required columns
    required = ["datum_aankoop", "omschrijving", "type", "karaat", "gram", "aankoopprijs"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check date format
    if "datum_aankoop" in df.columns:
        try:
            pd.to_datetime(df["datum_aankoop"])
        except Exception:
            issues.append("datum_aankoop contains invalid dates")

    # Check numeric columns
    for col in ["karaat", "gram", "aankoopprijs"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"{col} is not numeric")

    # Check karat values
    if "karaat" in df.columns:
        valid_karat = [14, 18, 21, 22, 24]
        invalid_karat = df[~df["karaat"].isin(valid_karat)]["karaat"].unique()
        if len(invalid_karat) > 0:
            issues.append(f"Unexpected karat values: {invalid_karat}")

    # Check negative values
    for col in ["gram", "aankoopprijs"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"{col} has {neg} negative values")

    return {"valid": len(issues) == 0, "issues": issues}
