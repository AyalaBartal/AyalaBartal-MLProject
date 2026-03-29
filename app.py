from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import os
import sys
import json
from werkzeug.utils import secure_filename
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.common.preprocessor import (
    DtPeStringConverter,
    DtPeListConverter,
    DtPePreprocessorProvider,
    ColumnTransformerRegistry,
    DtPeDataPreprocessMapArgs,
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model_wrapper import MalwareDetector

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Add JSON filter
@app.template_filter('tojson')
def to_json(value):
    return json.dumps(value)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load real models with transformers
dt_detector = MalwareDetector(
    'models/decision_tree/decision_tree_model.joblib',
    transformer_path='models/decision_tree/decision_tree_transformer.joblib'
)

rf_detector = None
rf_available = False
try:
    rf_detector = MalwareDetector(
        'models/random_forest/random_forest_model.joblib',
        transformer_path=None  # RF doesn't use sklearn transformer
    )
    rf_available = True
except Exception as e:
    print(f"Warning: RF model not available: {e}")

# Default to DT, will be overridden by model parameter in routes
detector = dt_detector

# Load feature names from DT schema
with open('models/decision_tree/dt_feature_schema.json', 'r') as f:
    schema = json.load(f)
    DT_FEATURE_NAMES = schema['feature_order']

# Load RF feature names if available
RF_FEATURE_NAMES = None
if rf_available:
    try:
        with open('models/random_forest/rf_feature_schema.json', 'r') as f:
            rf_schema = json.load(f)
            RF_FEATURE_NAMES = rf_schema['feature_order']
    except Exception as e:
        print(f"Warning: RF feature schema not found: {e}")

# Default to DT features
FEATURE_NAMES = DT_FEATURE_NAMES

# Generate demo data (all zeros as placeholder)
DEMO_DATA = [0.0] * len(FEATURE_NAMES)

# Raw PE file fields (original input format)
RAW_FIELDS = [
    'Size', 'SizeOfCode', 'SizeOfHeaders', 'SizeOfImage', 'SizeOfInitializedData',
    'SizeOfUninitializedData', 'FileAlignment', 'ImageBase', 'BaseOfCode', 'BaseOfData',
    'NumberOfSections', 'NumberOfRvaAndSizes', 'Entropy', 'SizeOfOptionalHeader',
    'PointerToSymbolTable', 'NumberOfSymbols', 'Characteristics', 'DllCharacteristics',
    'Machine', 'PE_TYPE', 'Identify', 'ImportedDlls', 'ImportedSymbols',
    'FirstSeenDate', 'TimeDateStamp'
]

# Demo values for raw PE fields (real goodware sample from training data)
RAW_DEMO_DATA = {
    'Size': '76288',
    'SizeOfCode': '64855',
    'SizeOfHeaders': '1024',
    'SizeOfImage': '86016',
    'SizeOfInitializedData': '2560',
    'SizeOfUninitializedData': '1500',
    'FileAlignment': '512',
    'ImageBase': '4194304',
    'BaseOfCode': '4096',
    'BaseOfData': '69632',
    'NumberOfSections': '5',
    'NumberOfRvaAndSizes': '16',
    'Entropy': '5.981248597',
    'SizeOfOptionalHeader': '224',
    'PointerToSymbolTable': '0',
    'NumberOfSymbols': '0',
    'Characteristics': '783',
    'DllCharacteristics': '0',
    'Machine': '332',
    'PE_TYPE': '267',
    'Identify': 'powerbasic/win 8.00',
    'ImportedDlls': 'comdlg32.dll gdi32.dll kernel32.dll ole32.dll oleaut32.dll user32.dll comctl32.dll libnodave.dll',
    'ImportedSymbols': 'printdlga getopenfilenamea getsavefilenamea bitblt createcompatiblebitmap createcompatibledc',
    'FirstSeenDate': '1970-01-01',
    'TimeDateStamp': '12345'
}

@app.route('/')
def index():
    return render_template('index.html', 
                         demo_data=DEMO_DATA, 
                         feature_names=FEATURE_NAMES,
                         raw_fields=RAW_FIELDS,
                         raw_demo_data=RAW_DEMO_DATA,
                         rf_available=rf_available)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get model selection from form
        model_param = request.form.get('model', 'dt')
        
        # Select detector and feature names based on model
        if model_param == 'rf' and rf_available:
            selected_detector = rf_detector
            selected_features = RF_FEATURE_NAMES
        else:
            selected_detector = dt_detector
            selected_features = DT_FEATURE_NAMES
        
        # Get features from form
        features = []
        for i in range(len(selected_features)):
            feature = request.form.get(f'feature_{i}', 0)
            features.append(float(feature))
        
        # Make prediction
        result = selected_detector.predict(features)
        
        return render_template('result.html', 
                             prediction=result['label'],
                             probability=result['probability'],
                             model_used=model_param.upper())
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/predict_raw', methods=['POST'])
def predict_raw():
    print("=== SINGLE SAMPLE PREDICTION ===")
    try:
        # Get model selection
        model_param = request.form.get('model', 'dt')
        
        # Get raw features from form
        raw_data = {}
        for field in RAW_FIELDS:
            value = request.form.get(field, '')
            if value:
                # Try to convert to appropriate type
                if field in ['Identify', 'ImportedDlls', 'ImportedSymbols', 'FirstSeenDate', 'TimeDateStamp']:
                    raw_data[field] = value
                else:
                    try:
                        if '.' in value:
                            raw_data[field] = float(value)
                        else:
                            raw_data[field] = int(value)
                    except ValueError:
                        raw_data[field] = value
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([raw_data])

        # Preprocess
        transformer = DtPePreprocessorProvider.get_transformer()
        X_transformed = transformer.transform(df)

        # transformer returns a list of DataFrames, so combine them into one DataFrame
        X_transformed = pd.concat(X_transformed, axis=1)

        # Select model and features
        if model_param == 'rf' and rf_available:
            selected_detector = rf_detector
            model_features = RF_FEATURE_NAMES
        else:
            selected_detector = dt_detector
            model_features = DT_FEATURE_NAMES
        
        # Align with model features
        for col in set(model_features) - set(X_transformed.columns):
            X_transformed[col] = 0
        X_transformed = X_transformed[model_features]
        
        # Make prediction
        result = selected_detector.predict(X_transformed.values[0].tolist())
        
        return render_template('result.html', 
                             prediction=result['label'],
                             probability=result['probability'],
                             model_used=model_param.upper())
    except Exception as e:
        import traceback
        print(f"Single sample error: {traceback.format_exc()}")
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    print("=== UPLOAD STARTED ===", flush=True)
    with open('/tmp/batch_debug.log', 'a') as f:
        f.write("\n=== UPLOAD STARTED ===\n")
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    # Get model selection
    model_param = request.form.get('model', 'dt')
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV with error handling
            try:
                df = pd.read_csv(filepath)
            except pd.errors.ParserError:
                df = pd.read_csv(filepath, on_bad_lines='skip')
                flash('⚠️ Warning: Some malformed lines were skipped during CSV parsing')
            
            # Basic validation
            if df.empty:
                flash('❌ Empty file. Please upload a valid CSV file.')
                return redirect(url_for('index'))
            
            # Check if labels exist
            has_labels = 'Label' in df.columns
            
            # Detect if raw data (has raw field names) or preprocessed
            is_raw = any(field in df.columns for field in RAW_FIELDS[:5])
            
            # Select model and features
            if model_param == 'rf' and rf_available:
                selected_detector = rf_detector
                model_features = RF_FEATURE_NAMES
            else:
                selected_detector = dt_detector
                model_features = DT_FEATURE_NAMES
            
            print(f"=== BATCH UPLOAD DEBUG ===")
            print(f"File shape: {df.shape}")
            print(f"Has Label: {has_labels}")
            print(f"Is raw: {is_raw}")
            print(f"Model: {model_param}")
            print(f"Expected features: {len(model_features)}")
            
            with open('/tmp/batch_debug.log', 'a') as f:
                f.write(f"\n=== BATCH UPLOAD ===\n")
                f.write(f"Shape: {df.shape}\n")
                f.write(f"Has Label: {has_labels}\n")
                f.write(f"Is raw: {is_raw}\n")
                f.write(f"Model: {model_param}\n")
            
            if is_raw:
                # Raw data - use preprocessing pipeline
                if has_labels:
                    y = df['Label']
                    df_features = df.drop('Label', axis=1)
                else:
                    y = None
                    df_features = df
                
                try:
                    # Use the mapper to preprocess (same as training)
                    mapper = DtPePreprocessorProvider.get_mapper()
                    X_transformed = mapper.map(df_features)
                    
                    # IMPORTANT: Mapper sometimes produces duplicate columns (e.g., 'dll__' appears twice)
                    # Remove duplicate columns, keeping only the first occurrence
                    X_transformed = X_transformed.loc[:, ~X_transformed.columns.duplicated()]
                    
                    # Update y to match if preprocessing changed row count
                    if y is not None and len(X_transformed) < len(y):
                        y = y.iloc[:len(X_transformed)].reset_index(drop=True)
                    
                    # Align columns with model's expected features
                    missing_cols = set(model_features) - set(X_transformed.columns)
                    
                    # Add missing columns with zeros
                    for col in missing_cols:
                        X_transformed[col] = 0.0
                    
                    # Remove extra columns and reorder to match model
                    X_transformed = X_transformed[model_features]
                    
                    # Debug: Check feature count
                    print(f"Transformed shape: {X_transformed.shape}")
                    print(f"Expected features: {len(model_features)}")
                    print(f"Actual features: {X_transformed.shape[1]}")
                    
                    predictions = selected_detector.predict_batch(X_transformed.values.tolist())
                except Exception as e:
                    import traceback
                    print(f"Full error: {traceback.format_exc()}")
                    flash(f'Preprocessing error: {str(e)}')
                    return redirect(url_for('index'))
            else:
                # Preprocessed data
                if has_labels:
                    X = df.drop('Label', axis=1)
                    y = df['Label']
                else:
                    X = df
                    y = None
                
                predictions = selected_detector.predict_batch(X.values.tolist())
            
            # Calculate metrics if labels exist
            metrics = None
            if has_labels:
                y_pred = [p['prediction'] for p in predictions]
                y_prob = [p['probability']['malware'] for p in predictions]
                
                # Check if we have both classes
                unique_labels = set(y) | set(y_pred)
                
                if len(unique_labels) > 1:
                    auc = roc_auc_score(y, y_prob)
                    accuracy = accuracy_score(y, y_pred)
                    cm = confusion_matrix(y, y_pred, labels=[0, 1])
                    
                    metrics = {
                        'auc': auc,
                        'accuracy': accuracy,
                        'confusion_matrix': cm.tolist()
                    }
                else:
                    # Only one class present
                    accuracy = accuracy_score(y, y_pred)
                    metrics = {
                        'auc': None,
                        'accuracy': accuracy,
                        'confusion_matrix': None
                    }
                    flash('⚠️ Note: Only one class present in data, AUC and confusion matrix not available')
            
            return render_template('batch_result.html', 
                                 predictions=predictions,
                                 metrics=metrics,
                                 has_labels=has_labels)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"=== UPLOAD ERROR ===")
            print(error_details)
            print(f"===================")
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

@app.route('/model/dt/visualization')
def dt_visualization():
    """Serve decision tree visualization image"""
    from flask import send_file
    return send_file('models/decision_tree/decision_tree_model.jpg', mimetype='image/jpeg')

@app.route('/model/rf/visualization')
def rf_visualization():
    """Serve pre-generated random forest feature importance visualization"""
    from flask import send_file
    try:
        return send_file('models/random_forest/feature_importance.png', mimetype='image/png')
    except Exception as e:
        print(f"Error serving RF visualization: {e}")
        return f"Error serving visualization: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5555)