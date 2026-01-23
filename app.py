from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import os
import sys
import json
from werkzeug.utils import secure_filename
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model_wrapper import MalwareDetector

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add JSON filter
@app.template_filter('tojson')
def to_json(value):
    return json.dumps(value)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
detector = MalwareDetector('models/model.pkl', 'models/scaler.pkl')

# Demo data (27 features) with realistic feature names
FEATURE_NAMES = [
    'file_size', 'entropy', 'num_sections', 'num_imports', 'num_exports',
    'has_debug', 'has_relocations', 'has_resources', 'has_tls', 'timestamp',
    'dll_characteristics', 'subsystem', 'machine_type', 'checksum', 'size_of_code',
    'size_of_headers', 'size_of_heap_reserve', 'size_of_heap_commit', 'size_of_stack_reserve',
    'size_of_stack_commit', 'loader_flags', 'num_rva_and_sizes', 'export_size',
    'import_size', 'resource_size', 'exception_size', 'security_size'
]

DEMO_DATA = [0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.3, 
             0.8, 0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.6, 0.3, 0.8,
             0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.6]

@app.route('/')
def index():
    return render_template('index.html', demo_data=DEMO_DATA, feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features = []
        for i in range(27):
            feature = request.form.get(f'feature_{i}', 0)
            features.append(float(feature))
        
        # Make prediction
        result = detector.predict(features)
        
        return render_template('result.html', 
                             prediction=result['label'],
                             probability=result['probability'])
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Check if labels exist
            has_labels = 'Label' in df.columns
            
            if has_labels:
                X = df.drop('Label', axis=1)
                y = df['Label']
            else:
                X = df
                y = None
            
            # Make predictions
            predictions = detector.predict_batch(X.values.tolist())
            
            # Calculate metrics if labels exist
            metrics = None
            if has_labels:
                y_pred = [p['prediction'] for p in predictions]
                y_prob = [p['probability']['malware'] for p in predictions]
                
                auc = roc_auc_score(y, y_prob)
                accuracy = accuracy_score(y, y_pred)
                cm = confusion_matrix(y, y_pred)
                
                metrics = {
                    'auc': auc,
                    'accuracy': accuracy,
                    'confusion_matrix': cm.tolist()
                }
            
            return render_template('batch_result.html', 
                                 predictions=predictions,
                                 metrics=metrics,
                                 has_labels=has_labels)
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Please upload a CSV file')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        features = data['features']
        result = detector.predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)