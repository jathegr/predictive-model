# predictive_model/src/train.py

import random
from pathlib import Path
import yaml
import json

from joblib import dump
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.data_loader import load_dataset
from src.models import build_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

def sample_symbols(symbols_file: Path, sample_size: int):
    lines = symbols_file.read_text().splitlines()
    all_syms = [s.strip() for s in lines if s.strip()]
    random.seed()  # or seed with date for reproducibility
    return random.sample(all_syms, sample_size)

def main(config_path: str = "configs/config.yaml"):
    # 1. Load config
    cfg = yaml.safe_load(open(config_path, "r"))
    
    # 2. Sample symbols
    symbols_file = Path(cfg["data"]["symbols_file"])
    syms = sample_symbols(symbols_file, cfg["data"]["sample_size"])
    logger.info(f"Sampled {len(syms)} symbols")

    # 3. Prepare data kwargs
    data_cfg = {
        "symbols": syms,
        "start": cfg["data"]["start"],
        "end": cfg["data"]["end"],
        "fs_url": cfg["data"]["fs_url"]
    }
    
    # 4. Load dataset
    X, y = load_dataset(**data_cfg)
    
    # 5. Build and evaluate CV
    pipe = build_pipeline(cfg["model"]["params"])
    tscv = TimeSeriesSplit(n_splits=cfg["model"]["cv_splits"])
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc")
    mean_auc = scores.mean()
    logger.info(f"[TRAIN] CV ROC AUC: {mean_auc:.4f}")
    
    # 6. Fit on full data
    pipe.fit(X, y)
    
    # 7. Persist artifacts
    art_dir = Path(cfg["artifacts"]["dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = art_dir / f"model_{cfg['model']['version']}.pkl"
    dump(pipe, model_path)
    
    metrics = {"cv_auc": mean_auc}
    with open(art_dir / f"metrics_{cfg['model']['version']}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"[TRAIN] Saved model → {model_path}")
    logger.info(f"[TRAIN] Saved metrics → {art_dir}")

if __name__ == "__main__":
    main()
