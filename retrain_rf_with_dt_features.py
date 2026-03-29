#!/usr/bin/env python3
"""Retrain Random Forest using DT feature set for consistency"""

import os
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib

def main():
    print("=" * 60)
    print("RETRAINING RF WITH DT FEATURES")
    print("=" * 60)
    
    # Load combined data
    print("\nLoading data...")
    df = pd.read_csv('uploads/combined_pe_data.csv', low_memory=False)
    print(f"✓ Loaded {len(df)} samples")
    
    # Load DT feature schema
    print("Loading DT feature schema...")
    with open('models/decision_tree/dt_feature_schema.json', 'r') as f:
        dt_schema = json.load(f)
        dt_features = dt_schema['feature_order']
    print(f"✓ DT features: {len(dt_features)}")
    
    # Extract features and label
    y = df['Label']
    X = df.drop('Label', axis=1)
    
    # Use preprocessor to get DT features
    print("\nApplying DT preprocessing...")
    from src.common.preprocessor import DtPePreprocessorProvider
    
    preprocessor = DtPePreprocessorProvider.get_mapper()
    X_transformed = preprocessor.map(X)
    
    # Align to DT features
    for col in set(dt_features) - set(X_transformed.columns):
        X_transformed[col] = 0
    X_transformed = X_transformed[dt_features]
    
    print(f"✓ Preprocessed shape: {X_transformed.shape}")
    
    # Train RF with cross-validation
    print("\nTraining Random Forest with 10-fold CV...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_auc_scores = []
    cv_acc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_transformed, y)):
        print(f"  Fold {fold + 1}/10...", end=" ")
        X_train, X_test = X_transformed.iloc[train_idx], X_transformed.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_fold = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        rf_fold.fit(X_train, y_train)
        
        y_pred = rf_fold.predict(X_test)
        y_pred_proba = rf_fold.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        cv_auc_scores.append(auc)
        cv_acc_scores.append(acc)
        print(f"AUC: {auc:.4f}, Acc: {acc:.4f}")
    
    print(f"\n✓ Mean AUC: {sum(cv_auc_scores)/len(cv_auc_scores):.4f}")
    print(f"✓ Mean Accuracy: {sum(cv_acc_scores)/len(cv_acc_scores):.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_final.fit(X_transformed, y)
    print("✓ Final model trained")
    
    # Get confusion matrix on all data
    y_pred = rf_final.predict(X_transformed)
    cm = confusion_matrix(y, y_pred)
    print(f"✓ Confusion matrix:\n{cm}")
    
    # Save model
    models_dir = 'models/random_forest'
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'random_forest_model.joblib')
    joblib.dump(rf_final, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': dt_features,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(models_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to {importance_path}")
    
    # Save feature schema (now same as DT)
    schema = {'feature_order': dt_features}
    schema_path = os.path.join(models_dir, 'rf_feature_schema.json')
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Feature schema saved to {schema_path}")
    
    # Save metrics
    metrics = {
        'cv_splits': 10,
        'cm': cm.tolist(),
        'auc_mean': sum(cv_auc_scores) / len(cv_auc_scores),
        'auc_std': 0,
        'acc_mean': sum(cv_acc_scores) / len(cv_acc_scores),
        'acc_std': 0,
        'n_features': len(dt_features),
        'n_samples': len(X_transformed)
    }
    
    metrics_path = os.path.join(models_dir, 'rf_cv_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
