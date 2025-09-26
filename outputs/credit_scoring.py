#!/usr/bin/env python3
import os
import sys
import argparse
import joblib
import warnings
import numpy as np
import pandas as pd
from time import time

# Make utils importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import add_age_bin

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional imports
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")
SEED = 42
OUTPUT_DIR = "outputs"

# ----------------- Helper Functions -----------------
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        target_col = df.columns[-1]  # default: last column
        y = df[target_col]
        X = df.drop(columns=[target_col])
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_unique = y.unique()
            if len(y_unique) == 2:
                mapping = {y_unique[0]: 0, y_unique[1]: 1}
                y = y.map(mapping)
            else:
                y, _ = pd.factorize(y)
        return X, y
    else:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml('credit-g', version=1, as_frame=True)
        df = ds.frame.copy()
        y = df['class'].map({'good': 1, 'bad': 0})
        X = df.drop(columns=['class'])
        return X, y

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    return preprocessor

def evaluate_model(pipe, X_test, y_test, name):
    y_pred = pipe.predict(X_test)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))
    return f1_score(y_test, y_pred, zero_division=0)

# ----------------- Main -----------------
def main(args):
    ensure_output_dir()
    X, y = load_data(args.data)
    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED),
        'DecisionTree': DecisionTreeClassifier(random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    }

    best_f1 = -1
    best_pipeline = None

    for name, clf in models.items():
        print(f"\n=== Training: {name} ===")
        pipe = Pipeline([
            ('feature_eng', FunctionTransformer(add_age_bin, validate=False)),
            ('preprocessor', preprocessor),
            ('model', clf)
        ])
        pipe.fit(X_train, y_train)
        f1 = evaluate_model(pipe, X_test, y_test, name)

        model_file = os.path.join(OUTPUT_DIR, f"model_{name}.joblib")
        joblib.dump(pipe, model_file)
        print(f"[+] Saved {name} model to {model_file}")

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe

    if best_pipeline:
        best_file = os.path.join(OUTPUT_DIR, "best_model.joblib")
        joblib.dump(best_pipeline, best_file)
        print(f"[+] Best model saved to {best_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit scoring models")
    parser.add_argument('--data', type=str, default=None, help="Path to CSV file")
    args = parser.parse_args()
    main(args)
