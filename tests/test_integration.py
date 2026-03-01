"""
Integration tests for Flask API endpoints
"""
import pytest
import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the Flask app from app.py (not app/__init__.py)
import app as app_module

@pytest.fixture
def client():
    """Create test client"""
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_home_page(client):
    """Test home page loads"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Malware Detection System' in response.data

def test_api_predict_endpoint(client):
    """Test API prediction endpoint"""
    test_features = [0.0] * 473  # Updated to match model's expected features
    response = client.post('/api/predict',
                          data=json.dumps({'features': test_features}),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'label' in data
    assert 'probability' in data

def test_api_predict_invalid_features(client):
    """Test API with invalid number of features"""
    test_features = [0.5] * 26  # Wrong number
    response = client.post('/api/predict',
                          data=json.dumps({'features': test_features}),
                          content_type='application/json')
    
    assert response.status_code == 400

def test_predict_form_submission(client):
    """Test form-based prediction"""
    form_data = {f'feature_{i}': '0.5' for i in range(27)}
    response = client.post('/predict', data=form_data)
    
    # Should redirect or show result page
    assert response.status_code in [200, 302]
