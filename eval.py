#!/usr/bin/env python3
"""
Evaluate trained malware detection models on a CSV file.

Usage:
    python eval.py --data /path/to/test.csv --model rf
    python eval.py --data /path/to/test.csv --model all

Models: lr, dt, rf, mlp, xgb, lgb, cbst
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

RANDOM_STATE = 42

MODEL_CONFIGS = {
    "lr": {
        "model_path": "models/logistic_regression/logistic_regression_model.joblib",
        "scaler_path": "models/logistic_regression/logistic_regression_scaler.joblib",
        "schema_path": "models/logistic_regression/lr_feature_schema.json",
        "schema_key": "feature_order",
    },
    "dt": {
        "model_path": "models/decision_tree/decision_tree_model.joblib",
        "scaler_path": None,
        "schema_path": "models/decision_tree/dt_feature_schema.json",
        "schema_key": "feature_order",
    },
    "rf": {
        "model_path": "models/random_forest/random_forest_model.joblib",
        "scaler_path": None,
        "schema_path": "models/random_forest/rf_feature_schema.json",
        "schema_key": "feature_order",
    },
    "xgb": {
        "model_path": "models/xgboost/xgboost_model.joblib",
        "scaler_path": "models/xgboost/xgboost_scaler.joblib",
        "schema_path": "models/xgboost/xgb_feature_schema.json",
        "schema_key": "feature_order",
    },
    "lgb": {
        "model_path": "models/lightgbm/lightgbm_model.joblib",
        "scaler_path": "models/lightgbm/lightgbm_scaler.joblib",
        "schema_path": "models/lightgbm/lgb_feature_schema.json",
        "schema_key": "feature_order",
    },
    "cbst": {
        "model_path": "models/catboost/catboost_model.joblib",
        "scaler_path": "models/catboost/catboost_scaler.joblib",
        "schema_path": "models/catboost/cbst_feature_schema.json",
        "schema_key": "features",
    },
}


def load_and_preprocess(data_path: str):
    from src.common.preprocessor import DtPePreprocessorProvider

    df = pd.read_csv(data_path)
    has_labels = "Label" in df.columns
    y = df["Label"].values if has_labels else None

    mapper = DtPePreprocessorProvider.get_mapper()
    X = mapper.map(df.drop("Label", axis=1) if has_labels else df)

    seen, keep = {}, []
    for i, col in enumerate(X.columns):
        if col not in seen:
            seen[col] = i
            keep.append(i)
    X = X.iloc[:, keep]
    return X, y


def evaluate_model(name: str, X: pd.DataFrame, y):
    cfg = MODEL_CONFIGS[name]
    if not os.path.exists(cfg["model_path"]):
        print(f"  [{name}] Model file not found: {cfg['model_path']}")
        return

    model = joblib.load(cfg["model_path"])
    scaler = joblib.load(cfg["scaler_path"]) if cfg["scaler_path"] and os.path.exists(cfg["scaler_path"]) else None

    with open(cfg["schema_path"]) as f:
        schema = json.load(f)
    feature_order = schema[cfg["schema_key"]]
    X_aligned = X.reindex(columns=feature_order, fill_value=0)

    X_input = scaler.transform(X_aligned) if scaler else X_aligned.values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[:, 1]
    else:
        proba = model.predict(X_input)

    pred = (proba >= 0.5).astype(int)

    print(f"\n  [{name.upper()}]")
    if y is not None:
        auc = roc_auc_score(y, proba)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        print(f"    AUC      : {auc:.4f}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Confusion Matrix:\n{cm}")
    else:
        malware_count = int(pred.sum())
        print(f"    Predictions: {len(pred)} samples — {malware_count} malware, {len(pred)-malware_count} goodware")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained malware detection models")
    parser.add_argument("--data", required=True, help="Path to CSV file (with or without Label column)")
    parser.add_argument("--model", default="rf", help="Model to evaluate: lr|dt|rf|mlp|xgb|lgb|cbst|all")
    args = parser.parse_args()

    print(f"Loading and preprocessing {args.data} ...")
    X, y = load_and_preprocess(args.data)
    print(f"  {len(X)} samples, {X.shape[1]} features, labels={'yes' if y is not None else 'no'}")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    for m in models_to_run:
        if m not in MODEL_CONFIGS:
            print(f"Unknown model '{m}'. Choose from: {', '.join(MODEL_CONFIGS)}")
            continue
        evaluate_model(m, X, y)


if __name__ == "__main__":
    main()
