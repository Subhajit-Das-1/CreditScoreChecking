#!/usr/bin/env python3
import os
import sys
import pandas as pd
import joblib
import argparse
from utils import add_age_bin

OUTPUT_DIR = "outputs"
HIGH_RISK_PROB_THRESHOLD = 0.3

# Make sure utils.py is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model)
    X_new = pd.read_csv(args.input)
    X_new = add_age_bin(X_new)

    y_pred = model.predict(X_new)
    X_new['predicted_class'] = y_pred

    try:
        y_proba = model.predict_proba(X_new)[:, 1]
        X_new['predicted_proba'] = y_proba
    except Exception:
        y_proba = None

    X_new.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    if y_proba is not None:
        high_risk = X_new[(X_new['predicted_class'] == 0) | (X_new['predicted_proba'] < HIGH_RISK_PROB_THRESHOLD)]
    else:
        high_risk = X_new[X_new['predicted_class'] == 0]

    high_risk.to_csv(os.path.join(args.output_dir, "high_risk_customers.csv"), index=False)
    print(f"[+] Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=os.path.join(OUTPUT_DIR, "best_model.joblib"))
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args)
