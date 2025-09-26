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
import argparse
import joblib
import warnings
from time import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Use non-GUI backend (for saving only)
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

from utils import add_age_bin  # import add_age_bin from utils.py

warnings.filterwarnings("ignore")
SEED = 42
OUTPUT_DIR = "outputs"

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
                print("[!] Target has more than 2 classes — factorizing labels.")
                y, _ = pd.factorize(y)
        return X, y, target_col
    else:
        print("[+] No CSV provided or file not found — fetching OpenML 'credit-g' dataset")
        ds = fetch_openml('credit-g', version=1, as_frame=True)
        df = ds.frame.copy()
        if 'class' in df.columns:
            y = df['class'].map({'good': 1, 'bad': 0})
            X = df.drop(columns=['class'])
            return X, y, 'class'
        else:
            raise RuntimeError("OpenML dataset loaded but expected 'class' column missing.")

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
        print("[+] Created feature: debt_to_income")
    if 'balance' in X.columns and 'credit_limit' in X.columns:
        X['credit_utilization'] = X['balance'] / (X['credit_limit'].replace({0: np.nan}))
        print("[+] Created feature: credit_utilization")
    if 'payment_history' in X.columns:
        try:
            X['late_payments_count'] = X['payment_history'].apply(
                lambda v: sum(int(x) for x in str(v).split(',') if x.strip().isdigit()))
            print("[+] Created feature: late_payments_count")
        except Exception:
            pass
    for c in ['previous', 'previous_defaults', 'num_defaults']:
        if c in X.columns:
            X['has_previous_default'] = (X[c] > 0).astype(int)
            print(f"[+] Created feature: has_previous_default from column {c}")
            break
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    print(f"[+] Numeric cols ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
    print(f"[+] Categorical cols ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols)>10 else ''}")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')
    return preprocessor, numeric_cols, categorical_cols

def evaluate_model(pipe, X_test, y_test, model_name):
    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps['model'], 'predict_proba'):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

    print(f"\n--- Evaluation: {model_name} ---")
    print("Accuracy:  ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall:    ", metrics['recall'])
    print("F1-score:  ", metrics['f1'])
    print("ROC-AUC:   ", metrics['roc_auc'])
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f'Confusion matrix: {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="w")
    plt.colorbar(im, ax=ax)
    cm_file = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png')
    fig.savefig(cm_file)
    plt.close(fig)
    print(f"[+] Saved confusion matrix to {cm_file}")

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc_val:.3f})')
        ax2.plot([0, 1], [0, 1], linestyle='--', lw=1)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'Receiver operating characteristic: {model_name}')
        ax2.legend(loc="lower right")
        roc_file = os.path.join(OUTPUT_DIR, f'roc_{model_name}.png')
        fig2.savefig(roc_file)
        plt.close(fig2)
        print(f"[+] Saved ROC curve to {roc_file}")

    return metrics

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
        print("[!] imbalanced-learn (SMOTE) not available; install to use SMOTE.")
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
            ('feature_eng', FunctionTransformer(add_age_bin, validate=False)),  # uses utils.py
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

    if best_pipeline is not None:
        best_file = os.path.join(OUTPUT_DIR, f'best_model_{best_model_name}.joblib')
        joblib.dump(best_pipeline, best_file)
        print(f"[+] Best model ({best_model_name}) saved to {best_file}")

    print("\nAll done. Check outputs/ for saved models and plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring Model pipeline")
    parser.add_argument('--data', type=str, default=None, help='Path to CSV file (optional).')
    parser.add_argument('--smote', action='store_true', help='Use SMOTE for class imbalance.')
    parser.add_argument('--tune', action='store_true', help='Run GridSearchCV (optional).')
    args = parser.parse_args()
    main(args)
