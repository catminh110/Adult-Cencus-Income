# Income Prediction >50K/year - Adult Census Income

A comprehensive Data Science project for predicting income based on census demographics using the UCI Adult Census Income dataset.

## Overview

This project implements:
- **Data Preprocessing:** Handling missing values ('?'), one-hot encoding, scaling
- **EDA:** Analysis of income rates by education, work hours, and demographics
- **Modeling:** Baselines (Majority), Logistic Regression, Linear SVC
- **Ablation Study:** L1 vs L2 comparison, LogReg vs Linear SVC
- **Evaluation:** Stratified 5-Fold CV with ROC-AUC, PR-AUC, F1
- **Web App:** Flask application with prediction, explanation, and ethics warnings
- **Chatbot:** Virtual assistant for explaining predictions and limitations

## Project Structure

```
adult_income_project/
├── data/
│   ├── raw/                    # Raw data files
│   │   ├── adult.csv           # Training data (32,561 samples)
│   │   └── adult_test.csv      # Test data (16,281 samples)
│   └── processed/              # Processed data
├── notebooks/
│   └── 01_complete_pipeline.ipynb  # Jupyter notebook
├── src/
│   ├── config.py               # Configuration
│   ├── data_preprocessing.py   # Preprocessing module
│   ├── eda.py                  # EDA module
│   └── models.py               # Model training module
├── reports/
│   ├── figures/                # Visualizations
│   │   ├── income_by_education.png
│   │   ├── income_by_hours.png
│   │   ├── demographics_analysis.png
│   │   ├── correlation_matrix.png
│   │   ├── roc_curves.png
│   │   ├── pr_curves.png
│   │   ├── coefficient_plot.png
│   │   └── model_comparison.csv
│   └── scientific_report.md    # Research paper
├── app/
│   ├── app.py                  # Flask application
│   └── templates/
│       └── index.html          # Web interface
├── logs/                       # Log files
├── models/                     # Trained models
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── feature_names.txt
└── requirements.txt            # Dependencies
```

## Installation

```bash
# Clone or download the project
cd adult_income_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run Complete Pipeline

```bash
# Preprocess data
python src/data_preprocessing.py

# Run EDA
python src/eda.py

# Train models
python src/models.py
```

### 2. Run Web Application

```bash
python app/app.py
```

Access at: http://localhost:5000

## Model Results

### Model Comparison (Stratified 5-Fold CV)

| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| Majority | 0.5000 ± 0.0000 | 0.2408 ± 0.0000 | 0.0000 ± 0.0000 |
| **LogReg_L1_C1** | **0.9068 ± 0.0017** | **0.7672 ± 0.0088** | **0.6626 ± 0.0070** |
| LogReg_L2_C1 | 0.9067 ± 0.0017 | 0.7670 ± 0.0088 | 0.6624 ± 0.0070 |
| LinearSVC_C1 | 0.9065 ± 0.0015 | 0.7668 ± 0.0087 | 0.6591 ± 0.0067 |

### Best Model
- **Model:** Logistic Regression (L1, C=1)
- **ROC-AUC:** 0.9068 ± 0.0017
- **PR-AUC:** 0.7672 ± 0.0088
- **F1-Score:** 0.6626 ± 0.0070

## Research Questions

### RQ1: How does encoding affect performance?
One-hot encoding enables linear models to learn non-linear relationships, achieving ROC-AUC ~0.907.

### RQ2: Subgroup metrics and responsible reporting
Analysis reveals income rate disparities between demographic groups, reflecting historical biases in the data.

### RQ3: LogReg vs Linear SVC trade-off
Logistic Regression offers better interpretability (probability output) with comparable performance.

## Web App Features

### 1. Income Prediction
- Enter demographic information
- Get income prediction (>50K or ≤50K)
- View prediction confidence

### 2. Prediction Explanation
- Analysis of key influencing factors
- Explanation based on model coefficients

### 3. Ethics Warning
- Data bias notification
- Model limitations
- Usage recommendations

### 4. Chatbot
- Q&A about predictions
- Factor explanations
- Ethics and limitations information

## Limitations and Risks

### Data Limitations
1. **Outdated (1994):** May not reflect current economic conditions
2. **Historical bias:** Contains gender and race biases
3. **Missing values:** ~5% missing data

### Deployment Risks
1. **Wrong inference:** Not for hiring/lending decisions
2. **Discrimination:** May perpetuate bias
3. **Context shift:** Performance degrades on new data

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
3. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks.
4. Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers.

## License

MIT License
