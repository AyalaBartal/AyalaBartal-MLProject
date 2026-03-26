from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import os
import sys
import json
from werkzeug.utils import secure_filename
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.specific.dt.preprocess import DtPeStringConverter, DtPeListConverter, DtPePreprocessorProvider
from src.specific.dt.preprocess.column_transformer_registry import ColumnTransformerRegistry
from src.specific.dt.preprocess.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs

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

# Load real model with transformer
detector = MalwareDetector(
    'models/decision_tree/decision_tree_model.joblib',
    transformer_path='models/decision_tree/decision_tree_transformer.joblib'
)

# Load feature names from schema
with open('models/decision_tree/dt_feature_schema.json', 'r') as f:
    schema = json.load(f)
    FEATURE_NAMES = schema['feature_order']

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
                         raw_demo_data=RAW_DEMO_DATA)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features = []
        for i in range(len(FEATURE_NAMES)):
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

@app.route('/predict_raw', methods=['POST'])
def predict_raw():
    print("=== SINGLE SAMPLE PREDICTION ===")
    try:
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
        
        # Align with model features
        model_features = FEATURE_NAMES
        for col in set(model_features) - set(X_transformed.columns):
            X_transformed[col] = 0
        X_transformed = X_transformed[model_features]
        
        # Make prediction
        result = detector.predict(X_transformed.values[0].tolist())
        
        return render_template('result.html', 
                             prediction=result['label'],
                             probability=result['probability'])
    except Exception as e:
        import traceback
        print(f"Single sample error: {traceback.format_exc()}")
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    print("=== UPLOAD STARTED ===")
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
            
            if is_raw:
                # Raw data - use preprocessing pipeline
                if has_labels:
                    y = df['Label']
                    df_features = df.drop('Label', axis=1)
                else:
                    y = None
                    df_features = df
                
                try:
                    # Import and use the preprocessing classes
                    transformer = DtPePreprocessorProvider.get_transformer()

                    # Transform with same parameters as training
                    # Process in batches and skip rows that fail
                    successful_rows = []
                    failed_indices = []
                    
                    for idx in range(len(df_features)):
                        try:
                            row_df = df_features.iloc[idx:idx+1]
                            X_row = transformer.transform(row_df)
                            successful_rows.append(X_row)
                        except Exception as row_error:
                            failed_indices.append(idx)
                            print(f"Row {idx} failed: {str(row_error)}")
                            continue
                    
                    if not successful_rows:
                        flash('❌ All rows failed preprocessing. Please check your data format.')
                        return redirect(url_for('index'))
                    
                    if failed_indices:
                        flash(f'⚠️ Skipped {len(failed_indices)} rows that failed preprocessing')

                    # Combine successful rows
                    row_dfs = []

                    for i, row_parts in enumerate(successful_rows):
                        if (i + 1) % 100 == 0:
                            print(f"Building row_dfs: {i + 1}/{len(successful_rows)}")

                        row_df = pd.concat(row_parts, axis=1)
                        row_dfs.append(row_df)
                    X_transformed = pd.concat(row_dfs, ignore_index=True)
                    
                    # Update y to match successful rows
                    if y is not None:
                        y = y.drop(failed_indices).reset_index(drop=True)
                    
                    # Align columns with model's expected features
                    model_features = FEATURE_NAMES
                    missing_cols = set(model_features) - set(X_transformed.columns)
                    
                    # Add missing columns with zeros
                    for col in missing_cols:
                        X_transformed[col] = 0
                    
                    # Remove extra columns and reorder to match model
                    X_transformed = X_transformed[model_features]
                    
                    # Debug: Check feature variance
                    print(f"Transformed shape: {X_transformed.shape}")
                    print(f"Non-zero features: {(X_transformed != 0).sum().sum()}")
                    print(f"Feature variance sample: {X_transformed.var().head(10).to_dict()}")
                    
                    predictions = detector.predict_batch(X_transformed.values.tolist())
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
                
                predictions = detector.predict_batch(X.values.tolist())
            
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

if __name__ == '__main__':
    app.run(debug=True)