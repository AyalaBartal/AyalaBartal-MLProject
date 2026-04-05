#!/usr/bin/env python3
"""
Train all 7 malware detection models end-to-end.

Usage:
    python train.py --data /path/to/brazilian-malware.csv

The script:
  1. Loads and preprocesses the dataset
  2. Performs an 80/20 stratified train/test split (seed 42)
  3. Trains each model with 10-fold stratified cross-validation
  4. Saves models + metrics to models/<name>/
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def load_and_preprocess(data_path: str):
    from src.common.preprocessor import DtPePreprocessorProvider

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    y = df["Label"].values
    print(f"  {len(df)} rows — class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("Preprocessing features ...")
    mapper = DtPePreprocessorProvider.get_mapper()
    X = mapper.map(df.drop("Label", axis=1))

    # Remove duplicate columns produced by pipeline
    seen, keep = {}, []
    for i, col in enumerate(X.columns):
        if col not in seen:
            seen[col] = i
            keep.append(i)
    X = X.iloc[:, keep]
    print(f"  {X.shape[1]} features after deduplication")
    return X, y


def cv_evaluate(model_factory, X_train, y_train, scaler=None, n_splits=10):
    """Run n-fold stratified CV and return (auc_scores, acc_scores)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    auc_scores, acc_scores = [], []
    for fold, (tr, val) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr], X_train[val]
        y_tr, y_val = y_train[tr], y_train[val]
        if scaler is not None:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)
        model = model_factory()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        auc_scores.append(roc_auc_score(y_val, proba))
        acc_scores.append(accuracy_score(y_val, pred))
        print(f"    fold {fold:2d}: AUC={auc_scores[-1]:.4f}  Acc={acc_scores[-1]:.4f}")
    return auc_scores, acc_scores


def save_metrics(path, cv_auc, cv_acc, test_auc, test_acc, cm, **extra):
    metrics = {
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
        "cv_accuracy_mean": float(np.mean(cv_acc)),
        "cv_accuracy_std": float(np.std(cv_acc)),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm,
        **extra,
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def train_logistic_regression(X_train, y_train, X_test, y_test, out_dir):
    print("\n=== Logistic Regression ===")
    os.makedirs(out_dir, exist_ok=True)
    auc, acc = cv_evaluate(
        lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        X_train, y_train, scaler=True,
    )
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_tr_s, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_te_s)[:, 1])
    test_acc = accuracy_score(y_test, model.predict(X_te_s))
    cm = confusion_matrix(y_test, model.predict(X_te_s)).tolist()
    joblib.dump(model, os.path.join(out_dir, "logistic_regression_model.joblib"))
    joblib.dump(sc, os.path.join(out_dir, "logistic_regression_scaler.joblib"))
    save_metrics(os.path.join(out_dir, "lr_cv_metrics.json"), auc, acc, test_auc, test_acc, cm)
    print(f"  CV  AUC={np.mean(auc):.4f}±{np.std(auc):.4f}  Acc={np.mean(acc):.4f}±{np.std(acc):.4f}")
    print(f"  Test AUC={test_auc:.4f}  Acc={test_acc:.4f}")


def train_decision_tree(X_train, y_train, X_test, y_test, out_dir, feature_names):
    print("\n=== Decision Tree ===")
    os.makedirs(out_dir, exist_ok=True)
    auc, acc = cv_evaluate(
        lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        X_train, y_train,
    )
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test)).tolist()
    joblib.dump(model, os.path.join(out_dir, "decision_tree_model.joblib"))
    with open(os.path.join(out_dir, "dt_feature_schema.json"), "w") as f:
        json.dump({"feature_order": feature_names}, f)
    save_metrics(os.path.join(out_dir, "dt_cv_metrics.json"), auc, acc, test_auc, test_acc, cm)
    print(f"  CV  AUC={np.mean(auc):.4f}±{np.std(auc):.4f}  Acc={np.mean(acc):.4f}±{np.std(acc):.4f}")
    print(f"  Test AUC={test_auc:.4f}  Acc={test_acc:.4f}")


def train_random_forest(X_train, y_train, X_test, y_test, out_dir, feature_names):
    print("\n=== Random Forest ===")
    os.makedirs(out_dir, exist_ok=True)
    auc, acc = cv_evaluate(
        lambda: RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE),
        X_train, y_train,
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test)).tolist()
    joblib.dump(model, os.path.join(out_dir, "random_forest_model.joblib"))
    with open(os.path.join(out_dir, "rf_feature_schema.json"), "w") as f:
        json.dump({"feature_order": feature_names}, f)
    save_metrics(os.path.join(out_dir, "rf_cv_metrics.json"), auc, acc, test_auc, test_acc, cm)
    print(f"  CV  AUC={np.mean(auc):.4f}±{np.std(auc):.4f}  Acc={np.mean(acc):.4f}±{np.std(acc):.4f}")
    print(f"  Test AUC={test_auc:.4f}  Acc={test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train all malware detection models")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset (must have a 'Label' column)")
    args = parser.parse_args()

    X, y = load_and_preprocess(args.data)
    feature_names = list(X.columns)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    train_logistic_regression(X_train, y_train, X_test, y_test, "models/logistic_regression")
    train_decision_tree(X_train, y_train, X_test, y_test, "models/decision_tree", feature_names)
    train_random_forest(X_train, y_train, X_test, y_test, "models/random_forest", feature_names)

    print("\nFor XGBoost, LightGBM, CatBoost, and PyTorch MLP run the individual train_*.py scripts.")
    print("All baseline models trained successfully.")


if __name__ == "__main__":
    main()
