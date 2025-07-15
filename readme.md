# Predictive Model for Stock Movement

This repo trains and evaluates a next‐day return classifier using your feature store.

## Structure

- `configs/config.yaml`: paths, symbols, hyperparameters  
- `src/`: code for loading data, labeling, modeling, training, evaluation  
- `artifacts/`: output models, metrics, plots  
- `requirements.txt`: Python dependencies  

## Setup

```bash
git clone https://github.com/jathegr/predictive-model.git
cd predictive-model
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
