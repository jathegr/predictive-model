# predictive_model/src/evaluate.py

import random
from pathlib import Path
import yaml
import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)

from src.data_loader import load_dataset
from src.utils import get_logger

logger = get_logger(__name__)

def sample_symbols(symbols_file: Path, sample_size: int):
    lines = symbols_file.read_text().splitlines()
    all_syms = [s.strip() for s in lines if s.strip()]
    random.seed()
    return random.sample(all_syms, sample_size)

def evaluate_classification(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def backtest_strategy(feat_df, y_proba, threshold=0.5):
    df = feat_df.copy()
    df["signal"] = (y_proba >= threshold).astype(int)
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["daily_ret"] = (df["next_close"] - df["close"]) / df["close"]
    df["strategy_ret"] = df["signal"] * df["daily_ret"]
    
    daily = df.groupby("date")[["daily_ret","strategy_ret"]].mean()
    daily["cum_strategy"] = (1 + daily["strategy_ret"]).cumprod()
    daily["cum_buy_hold"] = (1 + daily["daily_ret"]).cumprod()
    
    ret = daily["strategy_ret"]
    sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else np.nan
    drawdown = daily["cum_strategy"] / daily["cum_strategy"].cummax() - 1
    max_dd = drawdown.min()
    
    return daily, {
        "sharpe_annual": float(sharpe),
        "max_drawdown": float(max_dd),
        "final_return": float(daily["cum_strategy"].iloc[-1])
    }

def plot_roc_curve(y, y_proba, out_dir: Path):
    fpr, tpr, _ = roc_curve(y, y_proba)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_proba):.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

def plot_equity_curve(daily_df, out_dir: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(daily_df.index, daily_df["cum_strategy"], label="Strategy")
    plt.plot(daily_df.index, daily_df["cum_buy_hold"], label="Buy & Hold", alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png")
    plt.close()

def main(config_path: str = "configs/config.yaml"):
    # 1. Load config
    cfg = yaml.safe_load(open(config_path, "r"))
    
    # 2. Sample symbols
    symbols_file = Path(cfg["data"]["symbols_file"])
    syms = sample_symbols(symbols_file, cfg["data"]["sample_size"])
    logger.info(f"Sampled {len(syms)} symbols")
    
    # 3. Load data + full df
    data_cfg = {
        "symbols": syms,
        "start": cfg["data"]["start"],
        "end": cfg["data"]["end"],
        "fs_url": cfg["data"]["fs_url"]
    }
    X, y, feat_df = load_dataset(**data_cfg, return_full_df=True)
    
    # 4. Load model
    model_path = Path(cfg["artifacts"]["dir"]) / f"model_{cfg['model']['version']}.pkl"
    pipe = load(model_path)
    
    # 5. Predict and evaluate
    y_proba = pipe.predict_proba(X)[:,1]
    cls_metrics = evaluate_classification(y, y_proba, cfg["model"]["threshold"])
    daily_df, perf_metrics = backtest_strategy(feat_df, y_proba, cfg["model"]["threshold"])
    
    # 6. Save outputs
    out_dir = Path(cfg["artifacts"]["dir"]) / f"eval_{cfg['model']['version']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"classification": cls_metrics, "performance": perf_metrics}, f, indent=2)
    
    plot_roc_curve(y, y_proba, out_dir)
    plot_equity_curve(daily_df, out_dir)
    
    logger.info(f"[EVAL] Results saved to {out_dir}")

if __name__ == "__main__":
    main()
