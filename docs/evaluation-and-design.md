# Model Evaluation and Design Decisions

## Dataset

- **Source**: Brazilian Malware Dataset (PE static analysis features)
- **Size**: 50,181 instances (29,065 malware, 21,116 goodware)
- **Target**: `Label` column — 0 = goodware, 1 = malware
- **Split**: 80% training (40,144) / 20% hold-out test (10,037), stratified by class
- **Features**: 410 engineered features after preprocessing (from raw PE header fields, imported DLLs/APIs, timestamps, section sizes, and entropy)

## Cross-Validation Results

All models were evaluated using **stratified 10-fold cross-validation** on the training set (random seed 42). Preprocessing (scaling where applicable) was fit only on training folds and applied to validation folds to prevent leakage.

### Baseline Models

| Model | CV AUC (mean ± std) | CV Accuracy (mean ± std) | Algorithm Family |
|-------|---------------------|--------------------------|------------------|
| Logistic Regression | 0.8610 ± 0.0032 | 0.7981 ± 0.0052 | Linear |
| Decision Tree | 0.9915 ± 0.0025 | 0.9918 ± 0.0024 | Tree-based |
| Random Forest | **0.9996 ± 0.0002** | 0.9937 ± 0.0012 | Ensemble (Bagging) |
| PyTorch MLP | 0.9318 ± 0.0050 | 0.8660 ± 0.0304 | Neural Network |

### Additional Models

| Model | CV AUC (mean ± std) | CV Accuracy (mean ± std) | Algorithm Family |
|-------|---------------------|--------------------------|------------------|
| XGBoost | 0.9998 ± 0.0001 | 0.9953 ± 0.0006 | Gradient Boosting |
| LightGBM | 0.9998 ± 0.0001 | 0.9955 ± 0.0009 | Gradient Boosting |
| CatBoost | 0.9996 ± 0.0001 | 0.9948 ± 0.0004 | Gradient Boosting |

## Final Test Set Evaluation

The hold-out test set (20%) was kept untouched until final evaluation. The **Random Forest** was selected as the production model based on its highest cross-validation AUC among all models.

| Model | Test AUC | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 0.8597 | 0.7927 |
| Decision Tree | 0.9912 | 0.9913 |
| **Random Forest (Production)** | **1.0000** | **0.9998** |
| PyTorch MLP | 0.8969 | 0.8830 |
| XGBoost | 0.9998 | 0.9951 |
| LightGBM | 0.9995 | 0.9939 |
| CatBoost | 0.9997 | 0.9963 |

### Production Model (Random Forest) — Confusion Matrix

```
                 Predicted Goodware   Predicted Malware
Actual Goodware         4222                  2
Actual Malware             0               5813
```

- **False Positive Rate**: 2 / 4224 = 0.047% (goodware misclassified as malware)
- **False Negative Rate**: 0 / 5813 = 0.00% (zero malware missed)

## Design Decisions

### Data Preprocessing

The preprocessing pipeline (`DtPePreprocessorProvider`) transforms raw PE file fields into 410 engineered binary and numeric features:

- **PE Header Scalars**: `Size`, `SizeOfCode`, `SizeOfHeaders`, `SizeOfImage`, `FileAlignment`, `ImageBase`, `BaseOfCode`, `BaseOfData`, `NumberOfSections`, `NumberOfRvaAndSizes`, `Entropy`
- **Characteristics Bit Flags**: `char_b0`–`char_b15` and `dllc_b0`–`dllc_b15` (from `Characteristics` and `DllCharacteristics` integer fields)
- **Machine / PE Type**: One-hot encoded (`Machine_332`, `Machine_450`, `Machine_452`, `Machine_nan`, `PE_TYPE_267`, `PE_TYPE_nan`)
- **Compiler / Packer Identity**: 90+ binary flags from `Identify` string (e.g., `id_upx`, `id_msvc`, `id_net`, `id_inno`)
- **Imported DLLs**: 100+ binary flags for presence of common DLLs (e.g., `dll_kernel32`, `dll_user32`, `dll_ntdll`, `dll_ws2_32`)
- **Imported API Symbols**: ~100 binary flags for high-signal API calls (e.g., `api_virtualalloc`, `api_writeprocessmemory`, `api_regopenkeyexa`)
- **Timestamp Features**: Year, month, day-of-week derived from `FirstSeenDate` and `TimeDateStamp`, plus anomaly flag
- **Derived Ratios**: `ratio_Code_Image`, `ratio_InitData_Size`, `ratio_Headers_Size`
- **Missing Value Indicators**: `BaseOfData_missing`, `FirstSeen_missing`

**Scaling**: StandardScaler applied (fit on training data only) for Logistic Regression, XGBoost, LightGBM, CatBoost, and PyTorch MLP. Decision Tree and Random Forest do not require scaling.

**Missing Values**: Handled per-column during preprocessing (missing numeric values imputed as 0; missing categorical values produce dedicated indicator flags).

### Model Selection Criteria

- **Primary Metric**: AUC-ROC — more robust to class imbalance and threshold choice than accuracy
- **Secondary Metric**: Accuracy
- **Cross-Validation**: Stratified 10-fold CV on the 80% training set
- **Final Evaluation**: 20% hold-out test set, untouched until model was fully selected

**Random Forest was selected as the production model because:**
- Highest CV AUC across all baseline models (0.9996 ± 0.0002)
- Near-perfect test set performance (AUC = 1.0000, Accuracy = 99.98%)
- No false negatives on the test set — zero malware samples missed
- Robust with low variance across folds (std = 0.0002 AUC)
- No scaling required, simplifying the inference pipeline
- Feature importance available for security analysis and interpretability

Although XGBoost and LightGBM achieved marginally higher CV AUC (0.9998), the RF achieved perfect separation on the hold-out test, making it the strongest final model.

### Cross-Validation Strategy

- **Method**: Stratified 10-fold cross-validation (preserves class distribution per fold)
- **Seed**: Fixed random seed = 42 across all models and splits
- **Leakage Prevention**: Scalers are fit on training folds only, then applied to validation folds

### Web Application Design

- **Architecture**: Flask with a shared `MalwareDetector` model wrapper
- **Input modes**: (1) Manual form entry of raw PE features, (2) CSV batch upload
- **Evaluation mode**: When uploaded CSV includes a `Label` column, the app computes AUC, accuracy, and confusion matrix
- **Model switching**: All 7 trained models are available in the UI for comparison
- **Demo row**: Pre-filled demo data from a real malware sample for easy demonstration

### Deployment Strategy

- **Platform**: Render free tier (`https://malware-detection-app-ayala-bar6tal.onrender.com`)
- **CI/CD**: GitHub Actions — tests run on push and pull requests to `main`; deployment triggered on passing tests
- **Testing**: Unit tests (model/preprocessing logic), integration tests (Flask endpoints), smoke test (live `/health`)
- **Server**: Gunicorn WSGI server

## Limitations and Future Work

- **Feature Engineering**: No automated feature selection (e.g., mutual information, LASSO). All 410 features used as-is; dimensionality reduction may improve MLP and LR performance.
- **MLP Architecture**: A simple 3-layer MLP (128→64→32). Deeper architectures or attention-based models may close the gap with tree-based methods.
- **No Ensemble Stacking**: A stacked ensemble of RF + XGBoost + LightGBM could further reduce the already minimal error rate.
- **Dataset Scope**: Single publicly available Brazilian PE dataset; performance on other malware families or PE variants is unknown.

## Reproducibility

- **Environment**: Python 3.9, all dependencies pinned in `requirements.txt`
- **Random Seeds**: Fixed at 42 for `numpy`, `sklearn`, and `torch`
- **Training Scripts**: Separate `train_*.py` scripts per model; `train.py` trains all models end-to-end
- **Evaluation Script**: `eval.py` loads trained models and evaluates on a supplied CSV
- **Data Split**: `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` — identical across all models
