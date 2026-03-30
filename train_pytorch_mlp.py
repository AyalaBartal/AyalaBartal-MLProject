#!/usr/bin/env python3
"""Train PyTorch MLP model for malware detection"""

import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.common.preprocessor.pe_dt_preprocessor_provider import DtPePreprocessorProvider

print("=" * 70)
print("Training PyTorch MLP for Malware Detection")
print("=" * 70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

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
n_features = len(feature_order)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_order)

# 3. Define MLP Model
class MalwareMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(MalwareMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 4. Train with 5-fold Stratified CV
print("\n3. Training with 5-fold Stratified Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = MalwareMLP(n_features, hidden_sizes=[128, 64, 32], dropout=0.3)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

cv_auc_scores = []
cv_accuracy_scores = []
all_confusion_matrices = []

fold = 1
for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    print(f"\n   Fold {fold}/5...")
    
    X_fold_train = torch.FloatTensor(X_train_scaled.iloc[train_idx].values).to(device)
    y_fold_train = torch.FloatTensor(y_train[train_idx]).reshape(-1, 1).to(device)
    
    X_fold_val = torch.FloatTensor(X_train_scaled.iloc[val_idx].values).to(device)
    y_fold_val = torch.FloatTensor(y_train[val_idx]).reshape(-1, 1).to(device)
    
    # Create fold-specific model
    fold_model = MalwareMLP(n_features, hidden_sizes=[128, 64, 32], dropout=0.3)
    fold_model = fold_model.to(device)
    fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train for 10 epochs
    for epoch in range(10):
        fold_model.train()
        
        # Training
        fold_optimizer.zero_grad()
        train_output = fold_model(X_fold_train)
        train_loss = criterion(train_output, y_fold_train)
        train_loss.backward()
        fold_optimizer.step()
        
        if epoch == 9:  # Last epoch
            fold_model.eval()
            with torch.no_grad():
                val_output = fold_model(X_fold_val)
                val_loss = criterion(val_output, y_fold_val)
    
    # Validation
    fold_model.eval()
    with torch.no_grad():
        val_pred_prob = fold_model(X_fold_val).cpu().numpy().flatten()
        val_pred = (val_pred_prob > 0.5).astype(int)
        y_fold_val_np = y_train[val_idx]
        
        fold_auc = roc_auc_score(y_fold_val_np, val_pred_prob)
        fold_acc = accuracy_score(y_fold_val_np, val_pred)
        fold_cm = confusion_matrix(y_fold_val_np, val_pred)
        
        cv_auc_scores.append(fold_auc)
        cv_accuracy_scores.append(fold_acc)
        all_confusion_matrices.append(fold_cm)
        
        print(f"      AUC: {fold_auc:.4f}, Accuracy: {fold_acc:.4f}")
    
    fold += 1

cv_auc_scores = np.array(cv_auc_scores)
cv_accuracy_scores = np.array(cv_accuracy_scores)

print(f"\n   5-Fold CV Results:")
print(f"   AUC:      {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print(f"   Accuracy: {cv_accuracy_scores.mean():.4f} ± {cv_accuracy_scores.std():.4f}")

# 5. Train final model on entire training set
print("\n4. Training final model on full training set...")
final_model = MalwareMLP(n_features, hidden_sizes=[128, 64, 32], dropout=0.3)
final_model = final_model.to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)

X_train_tensor = torch.FloatTensor(X_train_scaled.values).to(device)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

for epoch in range(20):
    final_model.train()
    final_optimizer.zero_grad()
    output = final_model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    final_optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch + 1}/20 - Loss: {loss.item():.4f}")

# 6. Evaluate on test set
print("\n5. Evaluating on hold-out test set...")
X_test = mapper.map(df_test.drop('Label', axis=1))

# Align test features with training features
col_to_idx = {col: i for i, col in enumerate(X_test.columns)}
X_test_array = np.zeros((X_test.shape[0], len(feature_order)))
for i, feature_name in enumerate(feature_order):
    if feature_name in col_to_idx:
        X_test_array[:, i] = X_test.iloc[:, col_to_idx[feature_name]].values

X_test = pd.DataFrame(X_test_array, columns=feature_order)

# Normalize test data using training scaler
X_test_scaled = scaler.transform(X_test)

final_model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_pred_prob = final_model(X_test_tensor).cpu().numpy().flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

test_auc = roc_auc_score(y_test, y_pred_prob)
test_accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"   AUC:       {test_auc:.4f}")
print(f"   Accuracy:  {test_accuracy:.4f}")
print(f"   Confusion Matrix:")
print(f"     True Neg:  {cm[0,0]:6d}  |  False Pos: {cm[0,1]:6d}")
print(f"     False Neg: {cm[1,0]:6d}  |  True Pos:  {cm[1,1]:6d}")

# 7. Save model and metadata
print("\n6. Saving model and metadata...")
os.makedirs('models/pytorch_mlp', exist_ok=True)

# Save model with joblib for compatibility with MalwareDetector
dump(final_model, 'models/pytorch_mlp/pytorch_mlp_model.joblib')
print("   ✓ Saved model to models/pytorch_mlp/pytorch_mlp_model.joblib")

# Also save scaler
dump(scaler, 'models/pytorch_mlp/pytorch_mlp_scaler.joblib')
print("   ✓ Saved scaler to models/pytorch_mlp/pytorch_mlp_scaler.joblib")

# Save feature schema
feature_schema = {
    'feature_order': feature_order,
    'n_features': len(feature_order)
}
with open('models/pytorch_mlp/ml_feature_schema.json', 'w') as f:
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
    'confusion_matrix': cm.tolist(),
    'architecture': {
        'input_size': n_features,
        'hidden_sizes': [128, 64, 32],
        'dropout': 0.3,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss_function': 'BCELoss'
    }
}
with open('models/pytorch_mlp/ml_cv_metrics.json', 'w') as f:
    json.dump(cv_metrics, f, indent=2)
print("   ✓ Saved metrics")

# 8. Summary
print("\n" + "=" * 70)
print("PYTORCH MLP TRAINING COMPLETE")
print("=" * 70)
print(f"\nCV Results (5-fold):")
print(f"  AUC:      {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print(f"  Accuracy: {cv_accuracy_scores.mean():.4f} ± {cv_accuracy_scores.std():.4f}")
print(f"\nTest Set Results:")
print(f"  AUC:      {test_auc:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")
print("=" * 70)
