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
from utils import add_age_bin  # import from utils.py

# Threshold for flagging high-risk customers based on predicted probability
HIGH_RISK_PROB_THRESHOLD = 0.3  # adjust as needed

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

    # Apply preprocessing
    X_new = add_age_bin(X_new)

    # Predict
    y_pred = model.predict(X_new)
    X_new['predicted_class'] = y_pred

    # Predict probabilities (optional)
    try:
        y_proba = model.predict_proba(X_new)[:, 1]  # probability of "bad" class
        X_new['predicted_proba'] = y_proba
    except AttributeError:
        y_proba = None

    # Save full predictions
    predictions_file = os.path.join(args.output_dir, "predictions.csv")
    X_new.to_csv(predictions_file, index=False)
    print(f"[+] Saved full predictions to {predictions_file}")

    # Flag high-risk customers
    if y_proba is not None:
        # Adjust logic: consider "bad" class (1) with probability > threshold as high-risk
        high_risk = X_new[(X_new['predicted_class'] == 1) | (X_new['predicted_proba'] > HIGH_RISK_PROB_THRESHOLD)]
    else:
        high_risk = X_new[X_new['predicted_class'] == 1]

    high_risk_file = os.path.join(args.output_dir, "high_risk_customers.csv")
    high_risk.to_csv(high_risk_file, index=False)
    print(f"[+] Saved high-risk customers to {high_risk_file}")
    print(f"[+] Number of high-risk customers: {len(high_risk)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict using trained credit scoring model and flag high-risk customers"
    )
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.joblib)')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Folder to save predictions')
    args = parser.parse_args()

    main(args)
