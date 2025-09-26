#!/usr/bin/env python3
"""
credit_scoring.py

Full credit scoring project pipeline:
- load dataset (CSV path or fallback to OpenML 'credit-g')
- basic EDA prints
- feature engineering (best-effort)
- preprocessing pipeline (numeric impute+scale, categorical impute+one-hot)
- train LogisticRegression, DecisionTree, RandomForest
- evaluate with Precision, Recall, F1, ROC-AUC, confusion matrix, ROC plot
- optional SMOTE for imbalance, GridSearch for RandomForest
- save best model (joblib) and outputs in ./outputs/
"""

import os
import sys
import argparse
import joblib
import warnings
from time import time

# Add parent directory to path to import utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import add_age_bin  # import add_age_bin from utils.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)

# Optional imports
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
SEED = 42
OUTPUT_DIR = "outputs"

# ---------- Helper Functions ----------
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"[+] Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        candidates = ['target', 'class', 'y', 'default', 'loan_status', 'label', 'is_default']
        target_col = None
        for c in candidates:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            target_col = df.columns[-1]
            print(f"[!] No canonical target column found — using last column '{target_col}' as target.")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y_unique = y.unique()
            if len(y_unique) == 2:
                mapping = {y_unique[0]: 0, y_unique[1]: 1}
                print(f"[+] Mapping target categories {mapping}")
                y = y.map(mapping)
            else:
                print("[!] Target has more than 2 classes — factorizing labels (may not be binary!).")
                y, _ = pd.factorize(y)
        return X, y, target_col
    else:
        print("[+] No CSV provided or file not found — fetching OpenML 'credit-g'")
        from sklearn.datasets import fetch_openml
        ds = fetch_openml('credit-g', version=1, as_frame=True)
        df = ds.frame.copy()
        y = df['class'].map({'good': 1, 'bad': 0})
        X = df.drop(columns=['class'])
        return X, y, 'class'

def basic_eda(X, y, n=5):
    print("\n=== BASIC EDA ===")
    print("Shape:", X.shape)
    print("Target distribution:\n", pd.Series(y).value_counts(dropna=False))
    print("\nSample rows:")
    display = pd.concat([X.head(n), pd.Series(y.head(n), name='target')], axis=1)
    print(display)
    print("Columns and dtypes:")
    print(X.dtypes.value_counts())
    print("=================\n")

def feature_engineering(X):
    X = X.copy()
    if 'debt' in X.columns and 'income' in X.columns:
        X['debt_to_income'] = X['debt'] / (X['income'].replace({0: np.nan}))
    if 'balance' in X.columns and 'credit_limit' in X.columns:
        X['credit_utilization'] = X['balance'] / (X['credit_limit'].replace({0: np.nan}))
    return X

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
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
    return preprocessor, numeric_cols, categorical_cols

def evaluate_model(pipe, X_test, y_test, model_name):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['model'], 'predict_proba') else None
    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))
    return {}

# ---------- Main ----------
def main(args):
    ensure_output_dir()
    X, y, target_name = load_data(args.data)
    basic_eda(X, y)
    X = feature_engineering(X)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    use_smote = args.smote and IMBLEARN_AVAILABLE
    if args.smote and not IMBLEARN_AVAILABLE:
        print("[!] imbalanced-learn (SMOTE) not available.")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED),
        'DecisionTree': DecisionTreeClassifier(random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)

    results = {}
    best_f1 = -1
    best_pipeline = None

    for name, clf in models.items():
        print(f"\n=== Training: {name} ===")
        pipe = Pipeline([
            ('feature_eng', FunctionTransformer(add_age_bin, validate=False)),
            ('preprocessor', preprocessor),
            ('model', clf)
        ])
        if use_smote:
            preprocessor.fit(X_train)
            X_train_trans = preprocessor.transform(X_train)
            X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train_trans, y_train)
            clf.fit(X_res, y_res)
            pipe = Pipeline([('preprocessor', preprocessor), ('model', clf)])
        else:
            pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test, name)
        model_file = os.path.join(OUTPUT_DIR, f'model_{name}.joblib')
        joblib.dump(pipe, model_file)
        if metrics.get('f1', 0) > best_f1:
            best_f1 = metrics.get('f1', 0)
            best_pipeline = pipe

    if best_pipeline is not None:
        best_file = os.path.join(OUTPUT_DIR, f'best_model.joblib')
        joblib.dump(best_pipeline, best_file)
        print(f"[+] Best model saved to {best_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring Model pipeline")
    parser.add_argument('--data', type=str, default=None, help='Path to CSV file (optional).')
    parser.add_argument('--smote', action='store_true', help='Use SMOTE for class imbalance.')
    args = parser.parse_args()
    main(args)
