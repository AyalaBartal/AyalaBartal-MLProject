"""
Train CatBoost model for malware detection.
Uses 5-fold stratified cross-validation and saves model to models/catboost/
"""

import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib

from src.common.preprocessor.pe_dt_preprocessor_provider import DtPePreprocessorProvider


def load_and_preprocess_data():
    """Load data and apply preprocessing."""
    df = pd.read_csv('uploads/brazilian-malware.csv')
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    provider = DtPePreprocessorProvider()
    mapper = provider.get_mapper()
    X_mapped = mapper.map(X)
    
    return X_mapped, y


def train_with_cv(X, y):
    """Train model with 5-fold cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = CatBoostClassifier(
        iterations=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_scaled, y, cv=skf,
        scoring=['roc_auc', 'accuracy', 'precision', 'recall'],
        return_train_score=True
    )
    
    print(f"✅ Cross-Validation Results (5-fold):")
    print(f"  AUC:       {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
    
    return model, scaler, X_scaled, y, cv_results


def train_final_model(model, X_scaled, y):
    """Train final model on full data."""
    model.fit(X_scaled, y, verbose=0)
    return model


def save_model_and_metadata(model, scaler, X, y, cv_results):
    """Save model, scaler, and metadata to disk."""
    os.makedirs('models/catboost', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, 'models/catboost/catboost_model.joblib')
    joblib.dump(scaler, 'models/catboost/catboost_scaler.joblib')
    
    # Save feature schema
    with open('models/catboost/cbst_feature_schema.json', 'w') as f:
        json.dump({'features': list(X.columns)}, f)
    
    # Save CV metrics
    with open('models/catboost/cbst_cv_metrics.json', 'w') as f:
        json.dump({
            'auc_mean': float(cv_results['test_roc_auc'].mean()),
            'auc_std': float(cv_results['test_roc_auc'].std()),
            'accuracy_mean': float(cv_results['test_accuracy'].mean()),
            'accuracy_std': float(cv_results['test_accuracy'].std()),
            'precision_mean': float(cv_results['test_precision'].mean()),
            'precision_std': float(cv_results['test_precision'].std()),
            'recall_mean': float(cv_results['test_recall'].mean()),
            'recall_std': float(cv_results['test_recall'].std()),
        }, f)
    
    print(f"\n✅ Model saved to models/catboost/")


if __name__ == '__main__':
    print("Loading data...")
    X, y = load_and_preprocess_data()
    print(f"Data shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}\n")
    
    print("Training with 5-fold CV...")
    model, scaler, X_scaled, y, cv_results = train_with_cv(X, y)
    
    print("\nTraining final model...")
    model = train_final_model(model, X_scaled, y)
    
    print("Saving model and metadata...")
    save_model_and_metadata(model, scaler, X, y, cv_results)
