#!/usr/bin/env python3
"""Retrain RF with deduplicated features"""

import pandas as pd
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np

# Load combined data
print("Loading data...")
df = pd.read_csv('uploads/combined_pe_data.csv', low_memory=False)
y = df['Label']
X = df.drop('Label', axis=1)

# Preprocess using mapper
print("Preprocessing...")
from src.common.preprocessor import DtPePreprocessorProvider
mapper = DtPePreprocessorProvider.get_mapper()
X_transformed = mapper.map(X)

# Deduplicate columns
print("Deduplicating features...")
original_cols = list(X_transformed.columns)
seen = {}
keep_indices = []
for i, col in enumerate(original_cols):
    if col not in seen:
        seen[col] = i
        keep_indices.append(i)

unique_cols = [original_cols[i] for i in keep_indices]
X_clean = X_transformed.iloc[:, keep_indices]

print(f"Original: {len(original_cols)} features")
print(f"Unique: {len(unique_cols)} features")
print(f"Removed {len(original_cols) - len(unique_cols)} duplicates")

# Train RF
print("\nTraining RF with 10-fold CV...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_auc = []
cv_acc = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_clean, y)):
    print(f"  Fold {fold + 1}/10...", end=" ")
    X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf_fold = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_fold.fit(X_train, y_train)
    
    y_pred = rf_fold.predict(X_test)
    y_pred_proba = rf_fold.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    cv_auc.append(auc)
    cv_acc.append(acc)
    print(f"AUC: {auc:.4f}, Acc: {acc:.4f}")

print(f"\n✓ Mean AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
print(f"✓ Mean Acc: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

# Train final model
print("\nTraining final model...")
rf_final = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_final.fit(X_clean, y)

y_pred = rf_final.predict(X_clean)
cm = confusion_matrix(y, y_pred)
print(f"✓ Confusion matrix:\n{cm}")

# Save
models_dir = 'models/random_forest'
os.makedirs(models_dir, exist_ok=True)

joblib.dump(rf_final, os.path.join(models_dir, 'random_forest_model.joblib'))
print(f"✓ Model saved")

# Save feature schema
schema = {'feature_order': unique_cols}
with open(os.path.join(models_dir, 'rf_feature_schema.json'), 'w') as f:
    json.dump(schema, f, indent=2)
print(f"✓ Schema saved with {len(unique_cols)} features")

# Save metrics
metrics = {
    'cv_splits': 10,
    'cm': cm.tolist(),
    'auc_mean': float(np.mean(cv_auc)),
    'auc_std': float(np.std(cv_auc)),
    'acc_mean': float(np.mean(cv_acc)),
    'acc_std': float(np.std(cv_acc)),
    'n_features': len(unique_cols),
    'n_samples': len(X_clean)
}

with open(os.path.join(models_dir, 'rf_cv_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved")

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': unique_cols,
    'importance': rf_final.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv(os.path.join(models_dir, 'feature_importance.csv'), index=False)
print(f"✓ Feature importance saved")

print(f"\n✓✓✓ Retraining complete!")

