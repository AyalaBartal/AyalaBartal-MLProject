"""
Model wrapper for production inference
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
import os

from src.specific.dt.preprocess import DtPePreprocessorProvider

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

class MalwareDetector:
    def __init__(self, model_path=None, scaler_path=None, transformer_path=None):
        if model_path:
            self.model_type = 'pytorch' if model_path.endswith('.pt') else 'sklearn'
            
            if self.model_type == 'pytorch':
                # Load PyTorch model
                try:
                    import torch
                    import torch.nn as nn
                    
                    # Define model class
                    class MalwareMLP(nn.Module):
                        def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
                            super(MalwareMLP, self).__init__()
                            layers = []
                            prev_size = input_size
                            for hidden_size in hidden_sizes:
                                layers.append(nn.Linear(prev_size, hidden_size))
                                layers.append(nn.ReLU())
                                layers.append(nn.Dropout(dropout))
                                prev_size = hidden_size
                            layers.append(nn.Linear(prev_size, 1))
                            layers.append(nn.Sigmoid())
                            self.network = nn.Sequential(*layers)
                        
                        def forward(self, x):
                            return self.network(x)
                    
                    # Load model state
                    self.model = MalwareMLP(423)
                    self.model.load_state_dict(torch.load(model_path))
                    self.model.eval()
                    self.device = 'cpu'
                except Exception as e:
                    raise RuntimeError(f"Failed to load PyTorch model: {e}")
                
                # Load scaler
                self.scaler = joblib.load(scaler_path) if scaler_path else None
            else:
                # Load sklearn model
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path) if scaler_path else None
            
            # Load transformer if path provided, otherwise create new one
            if transformer_path and os.path.exists(transformer_path):
                self.transformer = joblib.load(transformer_path)
            else:
                self.transformer = DtPePreprocessorProvider.get_transformer()
        else:
            # Create dummy model for development
            self.model_type = 'sklearn'
            self.model = RandomForestClassifier(random_state=42)
            self.scaler = StandardScaler()
            self.transformer = None
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy trained model for development"""
        # Generate dummy training data (27 features)
        X_dummy = np.random.randn(1000, 27)
        y_dummy = np.random.randint(0, 2, 1000)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X_dummy)
        self.model.fit(X_scaled, y_dummy)
    
    def preprocess_raw_features(self, raw_data_dict):
        """Transform raw PE file features to model features"""
        if not self.transformer:
            raise ValueError("Transformer not initialized")
        
        # Convert dict to DataFrame
        df = pd.DataFrame([raw_data_dict])
        
        # Apply transformation
        X_transformed = self.transformer.transform(df)
        
        return X_transformed.values[0]
    
    def predict(self, features):
        """Make prediction on features"""
        features = np.array(features).reshape(1, -1)
        
        # Apply scaling if scaler exists
        if self.scaler:
            features = self.scaler.transform(features)
        
        if self.model_type == 'pytorch':
            import torch
            with torch.no_grad():
                X = torch.FloatTensor(features).to(self.device)
                output = self.model(X).cpu().numpy().flatten()
                probability = output[0]
                prediction = int(probability > 0.5)
            
            return {
                'prediction': prediction,
                'label': 'Malware' if prediction == 1 else 'Goodware',
                'probability': {
                    'goodware': float(1 - probability),
                    'malware': float(probability)
                }
            }
        else:
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            return {
                'prediction': int(prediction),
                'label': 'Malware' if prediction == 1 else 'Goodware',
                'probability': {
                    'goodware': float(probability[0]),
                    'malware': float(probability[1])
                }
            }
    
    def predict_raw(self, raw_data_dict):
        """Make prediction from raw PE file features"""
        features = self.preprocess_raw_features(raw_data_dict)
        return self.predict(features)
    
    def predict_batch(self, features_list):
        """Make predictions on multiple feature sets"""
        features = np.array(features_list)
        
        # Apply scaling if scaler exists
        if self.scaler:
            features = self.scaler.transform(features)
        
        if self.model_type == 'pytorch':
            import torch
            with torch.no_grad():
                X = torch.FloatTensor(features).to(self.device)
                outputs = self.model(X).cpu().numpy().flatten()
            
            results = []
            for prob in outputs:
                pred = int(prob > 0.5)
                results.append({
                    'prediction': pred,
                    'label': 'Malware' if pred == 1 else 'Goodware',
                    'probability': {
                        'goodware': float(1 - prob),
                        'malware': float(prob)
                    }
                })
            return results
        else:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'prediction': int(pred),
                    'label': 'Malware' if pred == 1 else 'Goodware',
                    'probability': {
                        'goodware': float(probabilities[i][0]),
                        'malware': float(probabilities[i][1])
                    }
                })
            return results
    
    def predict_batch_raw(self, raw_data_list):
        """Make predictions from raw PE file features"""
        df = pd.DataFrame(raw_data_list)
        X_transformed = self.transformer.transform(df)
        return self.predict_batch(X_transformed.values.tolist())

    def save(self, model_path, scaler_path):
        """Save model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
