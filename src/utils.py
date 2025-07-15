# predictive_model/src/utils.py

import logging
import json
from pathlib import Path

def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    return json.loads(Path(path).read_text())
