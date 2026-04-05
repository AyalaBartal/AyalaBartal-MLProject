# Malware Detection ML Project

**Student:** Ayala Bartal  
**Course:** MSSE Intro to Machine Learning — Quantic School of Business and Technology

A machine learning web application that classifies Windows executable (PE) files as **malware or goodware** using static analysis — without running the file.

## 🌐 Live Demo

**Deployed Application:** https://malware-detection-app-ayala-bar6tal.onrender.com

## 📊 Models Trained

| Model | CV AUC | CV Accuracy | Algorithm Family |
|-------|--------|-------------|------------------|
| Logistic Regression | 0.8610 ± 0.0032 | 0.7981 ± 0.0052 | Linear |
| Decision Tree | 0.9915 ± 0.0025 | 0.9918 ± 0.0024 | Tree-based |
| **Random Forest** ⭐ | **0.9996 ± 0.0002** | **0.9937 ± 0.0012** | Ensemble |
| PyTorch MLP | 0.9318 ± 0.0050 | 0.8660 ± 0.0304 | Neural Network |
| XGBoost | 0.9998 ± 0.0001 | 0.9953 ± 0.0006 | Gradient Boosting |
| LightGBM | 0.9998 ± 0.0001 | 0.9955 ± 0.0009 | Gradient Boosting |
| CatBoost | 0.9996 ± 0.0001 | 0.9948 ± 0.0004 | Gradient Boosting |

**Production model:** Random Forest — Test AUC = 1.0000, Test Accuracy = 99.98%

## 🔧 Features

- **Single prediction** — enter raw PE file features manually or load a demo sample
- **Batch upload** — upload a CSV file for bulk predictions
- **Evaluation mode** — include a `Label` column to see AUC, accuracy, and confusion matrix
- **Model switching** — compare all 7 trained models in the UI
- **Wrong prediction highlighting** — batch results show which predictions were incorrect

## 🗂️ Project Structure

```
AyalaBartal-MLProject/
├── app.py                    # Flask web application
├── train.py                  # Train baseline models (LR, DT, RF)
├── eval.py                   # Evaluate any model on a CSV
├── train_xgboost.py          # XGBoost training script
├── train_lightgbm.py         # LightGBM training script
├── train_catboost.py         # CatBoost training script
├── train_pytorch_mlp.py      # PyTorch MLP training script
├── models/                   # Trained model files
│   ├── decision_tree/
│   ├── random_forest/
│   ├── logistic_regression/
│   ├── xgboost/
│   ├── lightgbm/
│   ├── catboost/
│   └── pytorch_mlp/
├── src/
│   ├── common/preprocessor/  # Shared PE feature preprocessing pipeline
│   └── specific/             # Per-model trainer classes
├── tests/
│   ├── unit/                 # Unit tests (model + preprocessing logic)
│   ├── integration/          # Integration tests (Flask API endpoints)
│   └── test_smoke.py         # Smoke test (live deployment health check)
├── templates/                # HTML templates
├── docs/
│   ├── evaluation-and-design.md
│   ├── deployed.md
│   └── ai-tooling.md
├── .github/workflows/ci-cd.yml
└── requirements.txt
```

## 🚀 Setup & Run

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
python app.py
```

### 4. Train Models (requires dataset)
```bash
python train.py --data /path/to/brazilian-malware.csv
```

### 5. Evaluate a Model
```bash
python eval.py --data /path/to/test.csv --model rf
python eval.py --data /path/to/test.csv --model all
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/test_integration.py -v
```

**944 tests pass, 17 intentionally skipped** (end-to-end pipeline tests requiring dataset, and smoke tests requiring live deployment URL).

## ⚙️ CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):
1. **Unit tests** — model and preprocessing logic
2. **Integration tests** — Flask API endpoints
3. **Smoke test** — live `/health` endpoint
4. **Auto-deploy to Render** — only if all tests pass

## 📄 Documentation

- [`docs/evaluation-and-design.md`](docs/evaluation-and-design.md) — CV results for all 7 models, final test set evaluation, design decisions
- [`docs/deployed.md`](docs/deployed.md) — Live deployment URL
- [`docs/ai-tooling.md`](docs/ai-tooling.md) — AI tools used (Kiro, ChatGPT, GitHub Copilot)

