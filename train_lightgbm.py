#!/usr/bin/env python3
"""
LightGBM training script with 5-fold cross-validation
Trains a LightGBM classifier on the Brazilian malware dataset
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import lightgbm as lgb

from src.common.preprocessor import DtPePreprocessorProvider


def load_and_preprocess_data():
    """Load and preprocess the Brazilian malware dataset"""
    print("Loading dataset...")
    df = pd.read_csv('data/brazilian-malware.csv')
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Apply preprocessing
    print("Applying preprocessing...")
    preprocessor = DtPePreprocessorProvider.get_transformer()
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"Transformed shape: {X_transformed.shape}")
    
    return X_transformed, y, preprocessor


def train_with_cv(X, y, n_splits=5):
    """Train LightGBM with 5-fold stratified cross-validation"""
    print(f"\n{'='*80}")
    print("LIGHTGBM TRAINING WITH 5-FOLD STRATIFIED CV")
    print(f"{'='*80}\n")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        'fold': [],
        'train_auc': [],
        'val_auc': [],
        'train_acc': [],
        'val_acc': []
    }
    
    fold_num = 0
    best_val_auc = 0
    best_model = None
    
    for train_idx, val_idx in skf.split(X, y):
        fold_num += 1
        print(f"Fold {fold_num}/{n_splits}")
        print("-" * 40)
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred_proba = lgb_model.predict_proba(X_train_scaled)[:, 1]
        val_pred_proba = lgb_model.predict_proba(X_val_scaled)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred_proba)
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        train_pred = lgb_model.predict(X_train_scaled)
        val_pred = lgb_model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        cv_results['fold'].append(fold_num)
        cv_results['train_auc'].append(train_auc)
        cv_results['val_auc'].append(val_auc)
        cv_results['train_acc'].append(train_acc)
        cv_results['val_acc'].append(val_acc)
        
        print(f"  Train AUC: {train_auc:.4f}  |  Val AUC: {val_auc:.4f}")
        print(f"  Train Acc: {train_acc:.4f}  |  Val Acc: {val_acc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = lgb_model
            best_scaler = scaler
        
        print()
    
    # Print summary
    print(f"\n{'='*80}")
    print("CV SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Val AUC:  {np.mean(cv_results['val_auc']):.4f} ± {np.std(cv_results['val_auc']):.4f}")
    print(f"Mean Val Acc:  {np.mean(cv_results['val_acc']):.4f} ± {np.std(cv_results['val_acc']):.4f}")
    print(f"Best Val AUC:  {best_val_auc:.4f}")
    
    return best_model, best_scaler, cv_results


def evaluate_on_test_set(model, scaler, X_test, y_test):
    """Evaluate model on test set"""
    print(f"\n{'='*80}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*80}\n")
    
    X_test_scaled = scaler.transform(X_test)
    
    test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_pred = model.predict(X_test_scaled)
    
    test_auc = roc_auc_score(y_test, test_pred_proba)
    test_acc = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Pred Safe  Pred Malware")
    print(f"Act Safe     {cm[0, 0]:6d}      {cm[0, 1]:6d}")
    print(f"Act Malware  {cm[1, 0]:6d}      {cm[1, 1]:6d}")
    
    return test_auc, test_acc, test_pred_proba, cm


def save_model_and_metadata(model, scaler, feature_schema, cv_results, test_metrics):
    """Save model, scaler, and metadata"""
    print(f"\n{'='*80}")
    print("SAVING MODEL")
    print(f"{'='*80}\n")
    
    os.makedirs('models/lightgbm', exist_ok=True)
    
    # Save model
    model.booster_.save_model('models/lightgbm/lightgbm_model.txt')
    model_reloaded = lgb.Booster(model_file='models/lightgbm/lightgbm_model.txt')
    joblib.dump(model, 'models/lightgbm/lightgbm_model.joblib')
    print("✓ Saved: models/lightgbm/lightgbm_model.joblib")
    
    # Save scaler
    joblib.dump(scaler, 'models/lightgbm/lightgbm_scaler.joblib')
    print("✓ Saved: models/lightgbm/lightgbm_scaler.joblib")
    
    # Save feature schema
    with open('models/lightgbm/lgb_feature_schema.json', 'w') as f:
        json.dump(feature_schema, f, indent=2)
    print("✓ Saved: models/lightgbm/lgb_feature_schema.json")
    
    # Save CV metrics
    cv_metrics = {
        'cv_mean_auc': float(np.mean(cv_results['val_auc'])),
        'cv_std_auc': float(np.std(cv_results['val_auc'])),
        'cv_mean_acc': float(np.mean(cv_results['val_acc'])),
        'cv_std_acc': float(np.std(cv_results['val_acc'])),
        'test_auc': float(test_metrics['auc']),
        'test_accuracy': float(test_metrics['accuracy']),
        'confusion_matrix': test_metrics['cm'].tolist()
    }
    
    with open('models/lightgbm/lgb_cv_metrics.json', 'w') as f:
        json.dump(cv_metrics, f, indent=2)
    print("✓ Saved: models/lightgbm/lgb_cv_metrics.json")


def main():
    """Main training pipeline"""
    # Load data
    X, y, preprocessor = load_and_preprocess_data()
    
    # Split into train+val (for CV) and test (80/20)
    n_samples = len(X)
    n_test = int(0.2 * n_samples)
    
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train_val = X[train_indices]
    y_train_val = y.iloc[train_indices]
    X_test = X[test_indices]
    y_test = y.iloc[test_indices]
    
    print(f"\nTrain+Val: {len(X_train_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Train with CV
    best_model, best_scaler, cv_results = train_with_cv(X_train_val, y_train_val, n_splits=5)
    
    # Evaluate on test set
    test_auc, test_acc, test_pred_proba, cm = evaluate_on_test_set(
        best_model, best_scaler, X_test, y_test
    )
    
    # Get feature schema
    feature_schema = {
        'feature_count': X.shape[1],
        'feature_order': list(preprocessor.get_feature_names_out()),
        'model_type': 'lightgbm'
    }
    
    # Save everything
    test_metrics = {
        'auc': test_auc,
        'accuracy': test_acc,
        'cm': cm
    }
    
    save_model_and_metadata(best_model, best_scaler, feature_schema, cv_results, test_metrics)
    
    print(f"\n{'='*80}")
    print("✅ LIGHTGBM TRAINING COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
