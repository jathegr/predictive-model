# predictive_model/src/data_loader.py

from feature_store.online.client import FeatureStoreClient
import pandas as pd
from src.label_engineering import make_labels

def load_dataset(symbols, start, end, fs_url, return_full_df: bool = False):
    """
    Fetch features from the feature store, add binary labels.
    Returns X, y (and optionally the full DataFrame).
    """
    client = FeatureStoreClient(url=fs_url)
    feat_df = client.get_features(symbols=symbols, start=start, end=end)
    
    # Create labels
    feat_df["label"] = make_labels(feat_df)
    feat_df = feat_df.dropna(subset=["label"])
    
    # Split into X, y
    drop_cols = ["date", "symbol", "label", "next_close", "return"]
    X = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])
    y = feat_df["label"].astype(int)
    
    if return_full_df:
        return X, y, feat_df
    return X, y
