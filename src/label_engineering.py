# predictive_model/src/label_engineering.py

import pandas as pd

def make_labels(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """
    Binary label: 1 if next-day return > threshold, else 0.
    Assumes 'close', 'symbol', 'date' in df and sorted.
    """
    df = df.copy()
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["return"] = (df["next_close"] - df["close"]) / df["close"]
    return (df["return"] > threshold).astype(int)
