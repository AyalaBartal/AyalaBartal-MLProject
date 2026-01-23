"""
Unit tests for preprocessing and model functions
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_wrapper import MalwareDetector

class TestMalwareDetector:
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        detector = MalwareDetector()
        assert detector.model is not None
        assert detector.scaler is not None
    
    def test_single_prediction(self):
        """Test single prediction functionality"""
        detector = MalwareDetector()
        features = [0.5] * 27  # 27 dummy features
        result = detector.predict(features)
        
        assert 'prediction' in result
        assert 'label' in result
        assert 'probability' in result
        assert result['prediction'] in [0, 1]
        assert result['label'] in ['Malware', 'Goodware']
        assert 'goodware' in result['probability']
        assert 'malware' in result['probability']
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        detector = MalwareDetector()
        features_list = [[0.5] * 27, [0.3] * 27]  # 2 samples
        results = detector.predict_batch(features_list)
        
        assert len(results) == 2
        for result in results:
            assert 'prediction' in result
            assert 'label' in result
            assert 'probability' in result
    
    def test_feature_validation(self):
        """Test that model handles correct number of features"""
        detector = MalwareDetector()
        
        # Test with wrong number of features
        with pytest.raises(ValueError):
            detector.predict([0.5] * 26)  # Wrong number of features
    
    def test_probability_sum(self):
        """Test that probabilities sum to 1"""
        detector = MalwareDetector()
        features = [0.5] * 27
        result = detector.predict(features)
        
        prob_sum = result['probability']['goodware'] + result['probability']['malware']
        assert abs(prob_sum - 1.0) < 1e-6  # Should sum to 1
