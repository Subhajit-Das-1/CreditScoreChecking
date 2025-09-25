#!/usr/bin/env python3
"""
predict.py

Load a trained credit scoring model, make predictions on new data,
and flag high-risk customers automatically.
"""
import os
import pandas as pd
import joblib
import argparse

# Custom transformer to add age_bin
def add_age_bin(X):
    X = X.copy()
    if 'age' in X.columns:
        X['age_bin'] = pd.cut(
            X['age'], bins=[0, 25, 35, 50, 65, 120],
            labels=['<25', '25-34', '35-49', '50-64', '65+']
        )
    return X

# Threshold for flagging high-risk customers based on predicted probability
HIGH_RISK_PROB_THRESHOLD = 0.3  # optional: adjust

OUTPUT_DIR = "outputs"

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"[+] Loaded model from {model_path}")
    return model

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained model
    model = load_model(args.model)

    # Load input CSV
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    X_new = pd.read_csv(args.input)
    print(f"[+] Loaded input data ({X_new.shape[0]} rows, {X_new.shape[1]} columns)")

    # Make predictions
    y_pred = model.predict(X_new)
    X_new['predicted_class'] = y_pred

    # Optional: predicted probabilities
    try:
        y_proba = model.predict_proba(X_new)[:, 1]
        X_new['predicted_proba'] = y_proba
    except Exception:
        y_proba = None

    # Save full predictions
    predictions_file = os.path.join(args.output_dir, "predictions.csv")
    X_new.to_csv(predictions_file, index=False)
    print(f"[+] Saved full predictions to {predictions_file}")

    # Flag high-risk customers
    if y_proba is not None:
        high_risk = X_new[(X_new['predicted_class'] == 0) | (X_new['predicted_proba'] < HIGH_RISK_PROB_THRESHOLD)]
    else:
        high_risk = X_new[X_new['predicted_class'] == 0]

    high_risk_file = os.path.join(args.output_dir, "high_risk_customers.csv")
    high_risk.to_csv(high_risk_file, index=False)
    print(f"[+] Saved high-risk customers to {high_risk_file}")
    print(f"[+] Number of high-risk customers: {len(high_risk)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained credit scoring model and flag high-risk customers")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.joblib)')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Folder to save predictions')
    args = parser.parse_args()

    main(args)
