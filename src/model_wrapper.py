"""
Model wrapper for production inference
"""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class MalwareDetector:
    def __init__(self, model_path=None, scaler_path=None):
        if model_path and scaler_path:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            # Create dummy model for development
            self.model = RandomForestClassifier(random_state=42)
            self.scaler = StandardScaler()
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy trained model for development"""
        # Generate dummy training data (27 features)
        X_dummy = np.random.randn(1000, 27)
        y_dummy = np.random.randint(0, 2, 1000)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X_dummy)
        self.model.fit(X_scaled, y_dummy)
    
    def predict(self, features):
        """Make prediction on features"""
        features = np.array(features).reshape(1, -1)
        
        # Validate feature count
        if features.shape[1] != 27:
            raise ValueError(f"Expected 27 features, got {features.shape[1]}")
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Malware' if prediction == 1 else 'Goodware',
            'probability': {
                'goodware': float(probability[0]),
                'malware': float(probability[1])
            }
        }
    
    def predict_batch(self, features_list):
        """Make predictions on multiple feature sets"""
        features = np.array(features_list)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
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
    
    def save(self, model_path, scaler_path):
        """Save model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
