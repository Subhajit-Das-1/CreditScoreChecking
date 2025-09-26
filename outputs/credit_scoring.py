#!/usr/bin/env python3
"""
credit_scoring.py

Full credit scoring project pipeline with feature engineering, model training, evaluation,
and saving outputs. 'add_age_bin' is imported from utils.py.
"""
import os
import argparse
import joblib
import warnings
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-GUI backend for saving plots
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
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

from utils import add_age_bin  # Import age_bin function

warnings.filterwarnings("ignore")
SEED = 42
OUTPUT_DIR = "outputs"

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- load_data, basic_eda, feature_engineering, build_preprocessor ---
# (keep these functions as in your previous code; no change needed)

# Use FunctionTransformer with imported add_age_bin
def main(args):
    ensure_output_dir()
    X, y, target_name = load_data(args.data)
    basic_eda(X, y)
    X = feature_engineering(X)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"[+] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    use_smote = args.smote and IMBLEARN_AVAILABLE
    if args.smote and not IMBLEARN_AVAILABLE:
        print("[!] imbalanced-learn (SMOTE) not available; install 'imbalanced-learn' to use SMOTE.")
    if use_smote:
        print("[+] Using SMOTE to oversample minority class in training set.")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED),
        'DecisionTree': DecisionTreeClassifier(random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)

    results = {}
    best_model_name = None
    best_f1 = -1
    best_pipeline = None

    for name, clf in models.items():
        print(f"\n=== Training: {name} ===")
        pipe = Pipeline(steps=[
            ('feature_eng', FunctionTransformer(add_age_bin, validate=False)),
            ('preprocessor', preprocessor),
            ('model', clf)
        ])
        t0 = time()
        if use_smote:
            preprocessor.fit(X_train)
            X_train_trans = preprocessor.transform(X_train)
            X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train_trans, y_train)
            clf.fit(X_res, y_res)
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
        else:
            pipe.fit(X_train, y_train)
        t1 = time()
        print(f"[+] Training completed in {t1 - t0:.2f}s")
        metrics = evaluate_model(pipe, X_test, y_test, name)
        results[name] = metrics
        model_file = os.path.join(OUTPUT_DIR, f'model_{name}.joblib')
        joblib.dump(pipe, model_file)
        print(f"[+] Saved model pipeline to {model_file}")
        if metrics['f1'] is not None and metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            best_pipeline = pipe
        try:
            if hasattr(pipe.named_steps['model'], 'feature_importances_'):
                feature_names = get_feature_names_from_preprocessor(pipe.named_steps['preprocessor'], numeric_cols, categorical_cols)
                importances = pipe.named_steps['model'].feature_importances_
                imp_idx = np.argsort(importances)[::-1][:20]
                top_feats = [(feature_names[i], importances[i]) for i in imp_idx if i < len(feature_names)]
                print(f"[+] Top features for {name}:")
                for feat, val in top_feats[:15]:
                    print(f"   {feat}: {val:.4f}")
        except Exception:
            pass

    print(f"\n=== Summary of results ===")
    for name, metrics in results.items():
        print(f"{name}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, ROC_AUC={metrics['roc_auc']}")

    if args.tune:
        print("\n[+] Starting quick GridSearchCV on RandomForest (this may take time)...")
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf)])
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        gs = GridSearchCV(pipe, param_grid, scoring='f1', n_jobs=-1, cv=cv, verbose=1)
        gs.fit(X_train, y_train)
        print("[+] Best params:", gs.best_params_)
        best_gs = gs.best_estimator_
        metrics = evaluate_model(best_gs, X_test, y_test, "RandomForest_GridSearch")
        joblib.dump(best_gs, os.path.join(OUTPUT_DIR, 'model_RandomForest_GridSearch.joblib'))
        print("[+] Saved tuned RandomForest pipeline.")

    if best_pipeline is not None:
        best_file = os.path.join(OUTPUT_DIR, f'best_model_{best_model_name}.joblib')
        joblib.dump(best_pipeline, best_file)
        print(f"[+] Best model ({best_model_name}) saved to {best_file}")

    print("\nAll done. Check outputs/ for saved models and plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring Model pipeline")
    parser.add_argument('--data', type=str, default=None, help='Path to CSV file (optional). If not provided, script fetches OpenML credit-g.')
    parser.add_argument('--smote', action='store_true', help='Use SMOTE for class imbalance (requires imbalanced-learn).')
    parser.add_argument('--tune', action='store_true', help='Run quick GridSearchCV for RandomForest (takes longer).')
    args = parser.parse_args()
    main(args)
