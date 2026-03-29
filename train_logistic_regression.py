#!/usr/bin/env python3
"""Train Logistic Regression model for malware detection"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from joblib import dump
from src.common.preprocessor.pe_dt_preprocessor_provider import DtPePreprocessorProvider

print("=" * 70)
print("Training Logistic Regression for Malware Detection")
print("=" * 70)

# 1. Load and split data
print("\n1. Loading brazilian-malware.csv...")
df = pd.read_csv('/Users/ayalabartal/Downloads/brazilian-malware.csv')
print(f"   Total samples: {len(df)}")
print(f"   Label distribution:\n{df['Label'].value_counts()}")

# Hold out 20% test set BEFORE any preprocessing
y = df['Label'].values
train_indices, test_indices = train_test_split(
    np.arange(len(df)), 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

df_train = df.iloc[train_indices].reset_index(drop=True)
df_test = df.iloc[test_indices].reset_index(drop=True)

y_train = df_train['Label'].values
y_test = df_test['Label'].values

print(f"\n   Train set: {len(df_train)} samples")
print(f"   Test set: {len(df_test)} samples")

# 2. Preprocess training data
print("\n2. Preprocessing training data...")
mapper = DtPePreprocessorProvider.get_mapper()
X_train = mapper.map(df_train.drop('Label', axis=1))
print(f"   After preprocessing: {X_train.shape}")

# Store feature order for inference
feature_order = list(X_train.columns)

# 3. Train with 5-fold Stratified CV
print("\n3. Training with 5-fold Stratified Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train model on full training set (will use CV scoring)
lr_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    solver='lbfgs',
    class_weight='balanced'  # Handle class imbalance
)

cv_results = cross_validate(
    lr_model,
    X_train,
    y_train,
    cv=skf,
    scoring=['roc_auc', 'accuracy'],
    return_train_score=False
)

cv_auc_scores = cv_results['test_roc_auc']
cv_accuracy_scores = cv_results['test_accuracy']

print(f"\n   5-Fold CV Results:")
print(f"   AUC:      {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print(f"   Accuracy: {cv_accuracy_scores.mean():.4f} ± {cv_accuracy_scores.std():.4f}")

# 4. Train final model on entire training set
print("\n4. Training final model on full training set...")
lr_model.fit(X_train, y_train)

# 5. Evaluate on test set
print("\n5. Evaluating on hold-out test set...")
X_test = mapper.map(df_test.drop('Label', axis=1))

# Align test features with training features (use numpy array to handle duplicates)
col_to_idx = {col: i for i, col in enumerate(X_test.columns)}
X_test_array = np.zeros((X_test.shape[0], len(feature_order)))
for i, feature_name in enumerate(feature_order):
    if feature_name in col_to_idx:
        X_test_array[:, i] = X_test.iloc[:, col_to_idx[feature_name]].values

X_test = pd.DataFrame(X_test_array, columns=feature_order)

y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)

test_auc = roc_auc_score(y_test, y_proba[:, 1])
test_accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"   AUC:       {test_auc:.4f}")
print(f"   Accuracy:  {test_accuracy:.4f}")
print(f"   Confusion Matrix:")
print(f"     True Neg:  {cm[0,0]:6d}  |  False Pos: {cm[0,1]:6d}")
print(f"     False Neg: {cm[1,0]:6d}  |  True Pos:  {cm[1,1]:6d}")

# 6. Save model and metadata
print("\n6. Saving model and metadata...")
os.makedirs('models/logistic_regression', exist_ok=True)

# Save model
dump(lr_model, 'models/logistic_regression/logistic_regression_model.joblib')
print("   ✓ Saved model to models/logistic_regression/logistic_regression_model.joblib")

# Save feature schema
feature_schema = {
    'feature_order': feature_order,
    'n_features': len(feature_order)
}
with open('models/logistic_regression/lr_feature_schema.json', 'w') as f:
    json.dump(feature_schema, f, indent=2)
print("   ✓ Saved feature schema")

# Save CV metrics
cv_metrics = {
    'cv_auc_mean': float(cv_auc_scores.mean()),
    'cv_auc_std': float(cv_auc_scores.std()),
    'cv_accuracy_mean': float(cv_accuracy_scores.mean()),
    'cv_accuracy_std': float(cv_accuracy_scores.std()),
    'test_auc': float(test_auc),
    'test_accuracy': float(test_accuracy),
    'confusion_matrix': cm.tolist()
}
with open('models/logistic_regression/lr_cv_metrics.json', 'w') as f:
    json.dump(cv_metrics, f, indent=2)
print("   ✓ Saved metrics")

# 7. Summary
print("\n" + "=" * 70)
print("LOGISTIC REGRESSION TRAINING COMPLETE")
print("=" * 70)
print(f"\nCV Results (5-fold):")
print(f"  AUC:      {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print(f"  Accuracy: {cv_accuracy_scores.mean():.4f} ± {cv_accuracy_scores.std():.4f}")
print(f"\nTest Set Results:")
print(f"  AUC:      {test_auc:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")
print("=" * 70)
