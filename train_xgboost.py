#!/usr/bin/env python3
"""
XGBoost training script with 5-fold cross-validation and comprehensive evaluation.

Trains XGBoost classifier on malware detection dataset, evaluates performance,
and saves the trained model for production use.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import xgboost as xgb
from joblib import dump

from src.common.preprocessor.pe_dt_preprocessor_provider import DtPePreprocessorProvider

print("=" * 70)
print("Training XGBoost for Malware Detection")
print("=" * 70)

# 1. Load data
print("\n1. Loading brazilian-malware.csv...")
df = pd.read_csv('/Users/ayalabartal/Downloads/brazilian-malware.csv')
print(f"   Total samples: {len(df)}")
print(f"   Label distribution:")
print(df['Label'].value_counts())

# Split into train and test
y = df['Label'].values
train_indices, test_indices = train_test_split(
    np.arange(len(df)), 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

df_train = df.iloc[train_indices].reset_index(drop=True)
df_test = df.iloc[test_indices].reset_index(drop=True)

print(f"   Train set: {len(df_train)} samples")
print(f"   Test set: {len(df_test)} samples")

# 2. Preprocess training data
print("\n2. Preprocessing training data...")
mapper = DtPePreprocessorProvider.get_mapper()
X_train = mapper.map(df_train.drop('Label', axis=1))
y_train = df_train['Label'].values

print(f"   After preprocessing: {X_train.shape}")
feature_order = list(X_train.columns)

# Fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Cross-validation training
print("\n3. Training with 5-fold Stratified Cross-Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_scores = []
cv_accuracy_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
    print(f"\n   Fold {fold}/5...")
    
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train[val_idx]
    
    # Train fold model
    fold_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    fold_model.fit(X_fold_train, y_fold_train)
    
    # Evaluate fold
    y_fold_pred = fold_model.predict(X_fold_val)
    y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
    
    fold_auc = roc_auc_score(y_fold_val, y_fold_proba)
    fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
    
    cv_auc_scores.append(fold_auc)
    cv_accuracy_scores.append(fold_accuracy)
    
    print(f"      AUC: {fold_auc:.4f}, Accuracy: {fold_accuracy:.4f}")

print(f"\n   5-Fold CV Results:")
print(f"   AUC:      {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
print(f"   Accuracy: {np.mean(cv_accuracy_scores):.4f} ± {np.std(cv_accuracy_scores):.4f}")

# 4. Train final model on full training set
print("\n4. Training final model on full training set...")
final_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
final_model.fit(X_train_scaled, y_train)
print("   ✓ Model trained")

# 5. Evaluate on test set
print("\n5. Evaluating on hold-out test set...")
X_test = mapper.map(df_test.drop('Label', axis=1))
y_test = df_test['Label'].values

# Match feature order
col_to_idx = {col: i for i, col in enumerate(X_test.columns)}
X_test_array = np.zeros((X_test.shape[0], len(feature_order)))
for i, feature_name in enumerate(feature_order):
    if feature_name in col_to_idx:
        X_test_array[:, i] = X_test.iloc[:, col_to_idx[feature_name]].values

X_test_scaled = scaler.transform(X_test_array)

y_test_pred = final_model.predict(X_test_scaled)
y_test_proba = final_model.predict_proba(X_test_scaled)[:, 1]

test_auc = roc_auc_score(y_test, y_test_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print(f"   AUC:       {test_auc:.4f}")
print(f"   Accuracy:  {test_accuracy:.4f}")
print(f"   Confusion Matrix:")
print(f"     True Neg:  {cm[0,0]:6d}  |  False Pos: {cm[0,1]:6d}")
print(f"     False Neg: {cm[1,0]:6d}  |  True Pos:  {cm[1,1]:6d}")

# 6. Save model and metadata
print("\n6. Saving model and metadata...")
os.makedirs('models/xgboost', exist_ok=True)

# Save model
dump(final_model, 'models/xgboost/xgboost_model.joblib')
print("   ✓ Saved model to models/xgboost/xgboost_model.joblib")

# Save scaler
dump(scaler, 'models/xgboost/xgboost_scaler.joblib')
print("   ✓ Saved scaler to models/xgboost/xgboost_scaler.joblib")

# Save feature schema
feature_schema = {
    'feature_order': feature_order,
    'n_features': len(feature_order)
}
with open('models/xgboost/xgb_feature_schema.json', 'w') as f:
    json.dump(feature_schema, f, indent=2)
print("   ✓ Saved feature schema")

# Save CV metrics
cv_metrics = {
    'cv_auc_mean': float(np.mean(cv_auc_scores)),
    'cv_auc_std': float(np.std(cv_auc_scores)),
    'cv_accuracy_mean': float(np.mean(cv_accuracy_scores)),
    'cv_accuracy_std': float(np.std(cv_accuracy_scores)),
    'test_auc': float(test_auc),
    'test_accuracy': float(test_accuracy),
    'confusion_matrix': cm.tolist(),
    'architecture': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
}
with open('models/xgboost/xgb_cv_metrics.json', 'w') as f:
    json.dump(cv_metrics, f, indent=2)
print("   ✓ Saved metrics")

# Summary
print("\n" + "=" * 70)
print("XGBOOST TRAINING COMPLETE")
print("=" * 70)
print("\nCV Results (5-fold):")
print(f"  AUC:      {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
print(f"  Accuracy: {np.mean(cv_accuracy_scores):.4f} ± {np.std(cv_accuracy_scores):.4f}")
print("\nTest Set Results:")
print(f"  AUC:      {test_auc:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")
print("=" * 70)
