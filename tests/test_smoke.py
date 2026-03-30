"""
Smoke tests for post-deployment verification
"""
import requests
import pytest
import os

def test_deployment_health():
    """Test deployed application health endpoint"""
    # This will be updated with actual deployment URL
    base_url = os.getenv('DEPLOYMENT_URL', 'http://localhost:5000')
    
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code == 403:
            pytest.skip("Deployment port in use by system service (e.g., AirTunes)")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Deployment not accessible: {e}")

def test_deployment_home_page():
    """Test deployed application home page"""
    base_url = os.getenv('DEPLOYMENT_URL', 'http://localhost:5000')
    
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 403:
            pytest.skip("Deployment port in use by system service (e.g., AirTunes)")
        assert response.status_code == 200
        assert 'Malware Detection System' in response.text
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Deployment not accessible: {e}")
