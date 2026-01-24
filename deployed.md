# Deployment URL

The live web application is deployed at: **https://malware-detection-app-ayala-bar6tal.onrender.com**

## Features
- Manual feature entry with demo data (27 malware analysis features)
- Batch file upload for multiple predictions
- Model evaluation with metrics display (AUC, accuracy, confusion matrix)
- Real-time malware/goodware classification
- Beautiful responsive design with professional UI
- CI/CD pipeline with automated testing

## Usage
1. **Single Prediction**: Enter 27 feature values or click "Load Demo Data"
2. **Batch Analysis**: Upload CSV file with 27 feature columns
3. **Evaluation**: Include 'Label' column in CSV for performance metrics

## Technology Stack
- **Backend**: Flask with Python 3.11
- **ML Model**: Random Forest with StandardScaler preprocessing
- **Frontend**: Responsive HTML/CSS with modern design
- **Deployment**: Render free tier with automatic CI/CD
- **Testing**: Comprehensive test suite with pytest
