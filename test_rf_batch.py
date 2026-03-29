#!/usr/bin/env python3
"""Test RF batch predictions directly"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from src.common.preprocessor import DtPePreprocessorProvider
from src.model_wrapper import MalwareDetector

print("=" * 70)
print("Testing RF Batch Predictions")
print("=" * 70)

# Load test data
print("\n1. Loading data...")
df = pd.read_csv('uploads/combined_pe_data.csv', nrows=100, low_memory=False)
y = df['Label'].values
df_features = df.drop('Label', axis=1)
print(f"   Loaded {len(df)} samples, {len(df_features.columns)} features")

# Preprocess using mapper
print("\n2. Preprocessing...")
mapper = DtPePreprocessorProvider.get_mapper()
X_transformed = mapper.map(df_features)
print(f"   After mapper: {X_transformed.shape}")

# Load RF schema and align
print("\n3. Loading RF schema...")
with open('models/random_forest/rf_feature_schema.json') as f:
    model_features = json.load(f)['feature_order']
print(f"   Schema has {len(model_features)} features")

# Align features
missing_cols = set(model_features) - set(X_transformed.columns)
for col in missing_cols:
    X_transformed[col] = 0

X_aligned = X_transformed[model_features]
print(f"   After alignment: {X_aligned.shape}")

# Make predictions
print("\n4. Making RF predictions...")
rf_detector = MalwareDetector('models/random_forest/random_forest_model.joblib', transformer_path=None)
predictions = rf_detector.predict_batch(X_aligned.values.tolist())

preds = np.array([p['prediction'] for p in predictions])
proba = np.array([p['probability']['malware'] for p in predictions])

print(f"   Predictions: {len(preds)} samples")

# Calculate metrics
acc = accuracy_score(y, preds)
auc = roc_auc_score(y, proba)
cm = confusion_matrix(y, preds)

print("\n5. Results:")
print(f"   Accuracy:  {acc:.1%}")
print(f"   AUC:       {auc:.4f}")
print(f"   Confusion Matrix:")
print(f"     True Neg:  {cm[0,0]:6d}  |  False Pos: {cm[0,1]:6d}")
print(f"     False Neg: {cm[1,0]:6d}  |  True Pos:  {cm[1,1]:6d}")

if acc > 0.9 and auc > 0.9:
    print("\n✓✓✓ RF BATCH PREDICTIONS WORKING CORRECTLY! ✓✓✓")
else:
    print("\n✗ RF predictions may have issues")
    print(f"  Expected: Accuracy > 0.9, AUC > 0.9")
    print(f"  Got: Accuracy {acc:.1%}, AUC {auc:.4f}")

print("=" * 70)
