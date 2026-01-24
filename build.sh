#!/usr/bin/env bash
# Build script for Render

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating models directory..."
mkdir -p models

echo "Creating dummy model..."
python create_dummy_model.py

echo "Build completed!"
