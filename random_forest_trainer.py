#!/usr/bin/env python3
"""Random Forest Model Training for Malware Detection

This script:
1. Loads goodware (label=0) and malware (label=1) datasets
2. Splits data 80/20 stratified BEFORE preprocessing (prevents data leakage)
3. Trains Random Forest with 10-fold stratified cross-validation
4. Saves model, scaler, and results
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class RandomForestTrainer:
    """Trains and evaluates Random Forest for malware detection"""
    
    def __init__(self, goodware_path='uploads/goodware.csv', 
                 malware_path='uploads/brazilian-malware.csv',
                 random_state=RANDOM_STATE):
        self.goodware_path = goodware_path
        self.malware_path = malware_path
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cv_results = {}
        self.test_results = {}
        
    def load_data(self):
        """STEP 1: Load both datasets and combine with labels"""
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        # Load goodware (label=0)
        print(f"Loading goodware from {self.goodware_path}...")
        goodware = pd.read_csv(self.goodware_path)
        goodware['Label'] = 0
        print(f"  ✓ Goodware: {goodware.shape[0]} samples")
        
        # Load malware (label=1)
        print(f"Loading malware from {self.malware_path}...")
        malware = pd.read_csv(self.malware_path)
        malware['Label'] = 1
        print(f"  ✓ Malware: {malware.shape[0]} samples")
        
        # Combine datasets
        self.data = pd.concat([goodware, malware], ignore_index=True)
        print(f"\n✓ Combined dataset: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
        # Check class distribution
        class_dist = self.data['Label'].value_counts()
        print(f"\nClass distribution:")
        print(f"  Goodware (0): {class_dist[0]} ({class_dist[0]/len(self.data)*100:.1f}%)")
        print(f"  Malware (1):  {class_dist[1]} ({class_dist[1]/len(self.data)*100:.1f}%)")
        
        return self.data
    
    def preprocess_data(self):
        """STEP 2: Split data BEFORE preprocessing (prevent data leakage)"""
        print("\n" + "=" * 60)
        print("STEP 2: Stratified 80/20 Train/Test Split")
        print("=" * 60)
        print("⚠️  CRITICAL: Split BEFORE preprocessing to prevent data leakage")
        
        # Separate features and labels
        X = self.data.drop('Label', axis=1)
        y = self.data['Label']
        
        # Stratified 80/20 split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples (80%)")
        print(f"Test set:     {self.X_test.shape[0]} samples (20%)")
        print(f"Features:     {self.X_train.shape[1]}")
        
        # Verify stratification
        train_dist = self.y_train.value_counts()
        test_dist = self.y_test.value_counts()
        print(f"\nClass distribution preserved:")
        print(f"  Train - Goodware: {train_dist[0]} ({train_dist[0]/len(self.y_train)*100:.1f}%)")
        print(f"  Train - Malware:  {train_dist[1]} ({train_dist[1]/len(self.y_train)*100:.1f}%)")
        print(f"  Test  - Goodware: {test_dist[0]} ({test_dist[0]/len(self.y_test)*100:.1f}%)")
        print(f"  Test  - Malware:  {test_dist[1]} ({test_dist[1]/len(self.y_test)*100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_data(self):
        """STEP 3: Fit StandardScaler on training data ONLY"""
        print("\n" + "=" * 60)
        print("STEP 3: Feature Scaling (fit on training set only)")
        print("=" * 60)
        
        self.scaler = StandardScaler()
        
        # FIT scaler on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        print(f"✓ Scaler fit on training data")
        
        # TRANSFORM test data using training scaler
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✓ Training & test data scaled")
        
        # Check scaling statistics
        print(f"\nScaling statistics (from training data):")
        print(f"  Mean: {self.scaler.mean_[:3]}... (first 3 features)")
        print(f"  Std:  {self.scaler.scale_[:3]}... (first 3 features)")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_with_cross_validation(self):
        """STEP 4: Train Random Forest with 10-fold stratified cross-validation"""
        print("\n" + "=" * 60)
        print("STEP 4: Random Forest Training (10-fold Stratified CV)")
        print("=" * 60)
        
        # Create Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,  # Parallel processing
            verbose=0
        )
        
        print(f"Random Forest parameters:")
        print(f"  n_estimators: 100")
        print(f"  max_depth: 20")
        print(f"  min_samples_split: 5")
        print(f"  min_samples_leaf: 2")
        print(f"  n_jobs: -1 (all cores)")
        
        # Create 10-fold stratified CV splitter
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        print(f"\nRunning 10-fold stratified cross-validation...")
        print(f"This may take 3-5 minutes...")
        
        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            self.model,
            self.X_train_scaled,
            self.y_train,
            cv=skf,
            scoring=['roc_auc', 'accuracy', 'precision', 'recall', 'f1'],
            n_jobs=-1,
            verbose=1
        )
        
        # Store results
        self.cv_results = {
            'auc': cv_results['test_roc_auc'],
            'accuracy': cv_results['test_accuracy'],
            'precision': cv_results['test_precision'],
            'recall': cv_results['test_recall'],
            'f1': cv_results['test_f1']
        }
        
        # Print cross-validation results
        print("\n" + "─" * 60)
        print("CROSS-VALIDATION RESULTS (10 folds):")
        print("─" * 60)
        
        auc_scores = self.cv_results['auc']
        acc_scores = self.cv_results['accuracy']
        
        print(f"\nAUC Scores (10 folds):")
        for i, score in enumerate(auc_scores, 1):
            print(f"  Fold {i:2d}: {score:.4f}")
        
        print(f"\nAccuracy Scores (10 folds):")
        for i, score in enumerate(acc_scores, 1):
            print(f"  Fold {i:2d}: {score:.4f}")
        
        # Calculate mean ± std
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        acc_mean = np.mean(acc_scores)
        acc_std = np.std(acc_scores)
        
        print(f"\n{'═' * 60}")
        print(f"SUMMARY:")
        print(f"{'═' * 60}")
        print(f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"Precision: {np.mean(self.cv_results['precision']):.4f} ± {np.std(self.cv_results['precision']):.4f}")
        print(f"Recall:    {np.mean(self.cv_results['recall']):.4f} ± {np.std(self.cv_results['recall']):.4f}")
        print(f"F1-Score:  {np.mean(self.cv_results['f1']):.4f} ± {np.std(self.cv_results['f1']):.4f}")
        
        return self.cv_results
    
    def evaluate_on_test_set(self):
        """STEP 5: Final evaluation on hold-out test set"""
        print("\n" + "=" * 60)
        print("STEP 5: Final Evaluation on Hold-out Test Set")
        print("=" * 60)
        print("⚠️  CRITICAL: This test set was NOT touched during training/CV")
        
        # Train final model on full training set
        print(f"\nTraining final model on all training data...")
        self.model.fit(self.X_train_scaled, self.y_train)
        print(f"✓ Model trained")
        
        # Predict on test set
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_acc = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.test_results = {
            'auc': float(test_auc),
            'accuracy': float(test_acc),
            'confusion_matrix': cm.tolist(),
            'samples': len(self.y_test)
        }
        
        print(f"\n{'═' * 60}")
        print(f"TEST SET RESULTS:")
        print(f"{'═' * 60}")
        print(f"AUC:      {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Goodware  Malware")
        print(f"Actual Goodware  {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"       Malware   {cm[1,0]:6d}    {cm[1,1]:6d}")
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print(f"\nSensitivity (Malware detection): {sensitivity:.4f}")
        print(f"Specificity (Goodware detection): {specificity:.4f}")
        
        return self.test_results
    
    def save_model(self, output_dir='models/random_forest'):
        """STEP 6: Save model, scaler, and results"""
        print("\n" + "=" * 60)
        print("STEP 6: Saving Model & Results")
        print("=" * 60)
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'model.joblib')
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        # Save CV results
        cv_path = os.path.join(output_dir, 'cv_results.json')
        cv_data = {
            'auc_mean': float(np.mean(self.cv_results['auc'])),
            'auc_std': float(np.std(self.cv_results['auc'])),
            'accuracy_mean': float(np.mean(self.cv_results['accuracy'])),
            'accuracy_std': float(np.std(self.cv_results['accuracy'])),
            'auc_scores': [float(x) for x in self.cv_results['auc']],
            'accuracy_scores': [float(x) for x in self.cv_results['accuracy']]
        }
        with open(cv_path, 'w') as f:
            json.dump(cv_data, f, indent=2)
        print(f"✓ CV results saved to {cv_path}")
        
        # Save test results
        test_path = os.path.join(output_dir, 'test_results.json')
        with open(test_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"✓ Test results saved to {test_path}")
        
        # Save feature schema
        schema = {
            'n_features': self.X_train.shape[1],
            'feature_names': list(self.X_train.columns),
            'trained_on_samples': len(self.X_train),
            'random_state': self.random_state
        }
        schema_path = os.path.join(output_dir, 'schema.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"✓ Feature schema saved to {schema_path}")
        
        print(f"\n✓ All files saved to {output_dir}/")
        return output_dir


def main():
    """Main training pipeline"""
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "   RANDOM FOREST MALWARE DETECTION MODEL TRAINING".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    # Initialize trainer
    trainer = RandomForestTrainer()
    
    # Execute pipeline
    trainer.load_data()
    trainer.preprocess_data()
    trainer.scale_data()
    trainer.train_with_cross_validation()
    trainer.evaluate_on_test_set()
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review results in models/random_forest/")
    print(f"  2. Run unit tests: pytest tests/unit/test_random_forest_trainer.py -v")
    print(f"  3. Run integration tests: pytest tests/integration/test_random_forest_api.py -v")
    print(f"  4. Update app.py to use this model")


if __name__ == '__main__':
    main()
