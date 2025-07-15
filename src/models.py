# predictive_model/src/models.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def build_pipeline(params: dict) -> Pipeline:
    """
    Build an ML pipeline: standard scaling + XGBoost classifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(**params))
    ])
