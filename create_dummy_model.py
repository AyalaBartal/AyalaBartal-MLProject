#!/usr/bin/env python3
"""
Create dummy model for development
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_wrapper import MalwareDetector

def create_dummy_model():
    """Create and save dummy model for development"""
    detector = MalwareDetector()
    
    # Save model and scaler
    detector.save('models/model.pkl', 'models/scaler.pkl')
    print("Dummy model created and saved to models/")

if __name__ == "__main__":
    create_dummy_model()
