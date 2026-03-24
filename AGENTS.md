# Adult Census Income Prediction - AI Agent Guide

## Project Overview

This is a comprehensive Data Science project for predicting income (>50K/year vs ≤50K/year) based on census demographics using the UCI Adult Census Income dataset. The project implements a complete ML pipeline from data preprocessing to deployment via a Flask web application.

### Key Features
- **Data Pipeline:** Handles missing values ('?'), one-hot encoding, standard scaling
- **EDA Module:** Analysis of income rates by education, work hours, and demographics
- **Modeling:** Baselines, Logistic Regression (L1/L2), Linear SVC with ablation study
- **Evaluation:** Stratified 5-Fold Cross-Validation with ROC-AUC, PR-AUC, F1 metrics
- **Web Application:** Flask app with prediction, explanation, ethics warnings, and chatbot

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Core Data Science | numpy==1.24.3, pandas==2.0.3, scipy==1.11.1 |
| Machine Learning | scikit-learn==1.3.0 |
| Visualization | matplotlib==3.7.2, seaborn==0.12.2 |
| Web Framework | flask==2.3.3, flask-cors==4.0.0 |
| Development | jupyter==1.0.0, ipython==8.14.0 |
| Utilities | pyyaml==6.0.1, joblib==1.3.2 |

## Project Structure

```
.
├── app/                          # Flask web application
│   ├── app.py                    # Main Flask app with prediction API
│   └── templates/
│       └── index.html            # Web UI with chatbot
├── data/
│   ├── raw/                      # Original dataset
│   │   ├── adult.csv             # Training data (32,561 samples)
│   │   └── adult_test.csv        # Test data (16,281 samples)
│   └── processed/                # Preprocessed data
│       └── adult_processed.csv
├── logs/                         # Application logs
│   ├── preprocessing.log
│   └── training.log
├── models/                       # Trained artifacts
│   ├── best_model.joblib         # Best performing model
│   ├── preprocessor.joblib       # Fitted preprocessor pipeline
│   └── feature_names.txt         # Feature names after encoding
├── reports/
│   ├── figures/                  # Generated visualizations
│   │   ├── coefficient_plot.png
│   │   ├── correlation_matrix.png
│   │   ├── demographics_analysis.png
│   │   ├── income_by_education.png
│   │   ├── income_by_hours.png
│   │   ├── model_comparison.csv
│   │   ├── pr_curves.png
│   │   └── roc_curves.png
│   └── scientific_report.md      # Research paper
├── src/                          # Source code modules
│   ├── config.py                 # Configuration constants
│   ├── data_preprocessing.py     # Data preprocessing class
│   ├── eda.py                    # Exploratory data analysis
│   └── models.py                 # Model training & evaluation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Build and Run Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Complete ML Pipeline
Execute modules in sequence from the project root:

```bash
# 1. Preprocess data
python src/data_preprocessing.py

# 2. Run Exploratory Data Analysis
python src/eda.py

# 3. Train and evaluate models
python src/models.py
```

### Run Web Application
```bash
python app/app.py
```
Access at: http://localhost:5000

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Make income prediction |
| `/api/features` | GET | Get feature definitions |
| `/health` | GET | Health check |

## Code Organization

### Module Responsibilities

**`src/config.py`**
- Defines paths, column specifications, model configurations
- Sets random seed (42) for reproducibility
- Contains hyperparameter grids for model tuning

**`src/data_preprocessing.py`**
- `DataPreprocessor` class: Handles missing values ('?' → 'Unknown')
- Creates sklearn pipeline: StandardScaler (numerical) + OneHotEncoder (categorical)
- Target encoding: `<=50K` → 0, `>50K` → 1
- Saves/loads preprocessor via joblib

**`src/eda.py`**
- `EDAAnalyzer` class: Generates visualizations
- Analyzes income by education, hours worked, demographics
- Creates correlation matrix
- Outputs saved to `reports/figures/`

**`src/models.py`**
- `ModelTrainer` class: Implements model training pipeline
- Models: Majority baseline, Logistic Regression (L1/L2), Linear SVC
- Evaluation: Stratified 5-Fold CV with multiple metrics
- Saves best model and generates comparison tables

**`app/app.py`**
- Flask application for interactive predictions
- Rule-based explanation generation
- Ethics and limitations warnings
- Client-side chatbot with keyword-based responses

## Data Schema

### Categorical Columns (8)
- `workclass`: Private, Self-emp, Gov, etc.
- `education`: Preschool to Doctorate
- `marital-status`: Married, Divorced, etc.
- `occupation`: Tech-support, Sales, etc.
- `relationship`: Wife, Husband, etc.
- `race`: White, Black, Asian, etc.
- `sex`: Male, Female
- `native-country`: 42 countries + Unknown

### Numerical Columns (6)
- `age`: 17-90 years
- `fnlwgt`: Census weight
- `education-num`: 1-16 years
- `capital-gain`: 0-99999 USD
- `capital-loss`: 0-5000 USD
- `hours-per-week`: 1-99 hours

### Target
- `income`: `<=50K` or `>50K`

## Code Style Guidelines

### Python Code Style
- **Docstrings:** Module-level and class-level docstrings required
- **Comments:** Inline comments for complex logic
- **Logging:** Use Python `logging` module with timestamp and level
- **Naming:** 
  - Classes: `PascalCase` (e.g., `DataPreprocessor`)
  - Functions/Variables: `snake_case` (e.g., `handle_missing_values`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `RANDOM_SEED`)

### Example Pattern
```python
"""
Module description
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/file.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MyClass:
    """Class description"""
    
    def method(self):
        """Method description"""
        logger.info("Processing...")
```

## Testing Strategy

**Note:** This project does not have automated unit tests. Testing is done through:

1. **Manual Module Execution:** Run each module individually to verify functionality
2. **Web Application Testing:** Use the Flask app to test predictions
3. **Health Check:** Use `/health` endpoint to verify model loading
4. **Log Inspection:** Check `logs/preprocessing.log` and `logs/training.log`

## Important Implementation Notes

### Path Configuration
**CRITICAL:** The `src/config.py` file contains hardcoded absolute paths:
```python
BASE_DIR = '/mnt/okcomputer/output/adult_income_project'
```

When running in different environments, you may need to update these paths or set up symlinks.

### Model Artifacts
- Models are saved using `joblib` (`.joblib` files)
- The best model is selected based on ROC-AUC score
- Preprocessor must be loaded alongside the model for predictions

### Missing Value Handling
- Missing values are represented as `'?'` in the raw data
- Strategy: Replace with `'Unknown'` for categorical variables
- Approximately 5% of data has missing values

## Security and Ethics Considerations

### Data Limitations (Documented in App)
1. **Outdated (1994):** May not reflect current economic conditions
2. **Historical bias:** Contains potential gender and race biases
3. **Missing values:** ~5% missing data in raw dataset

### Usage Restrictions
- **NOT for employment decisions:** Do not use for hiring/lending
- **Discrimination risk:** May perpetuate historical biases
- **Individual inference:** Statistical model only, not individual capability

### Ethics Warning (Displayed in Web App)
The web application displays ethics warnings covering:
- No social inference (not individual worth)
- Bias in data (gender, race, origin biases)
- Prediction limitations (demographic features only)
- Not for employment decisions
- Temporal context (1994 data)

## Best Practices for Modifications

1. **Reproducibility:** Always set `random_state=42` (defined in `config.py`)
2. **Logging:** Add logging to all new modules for debugging
3. **Path Handling:** Use `os.path.join()` for cross-platform compatibility
4. **Error Handling:** Wrap model inference in try-except blocks
5. **Feature Engineering:** Update `CATEGORICAL_COLUMNS` and `NUMERICAL_COLUMNS` in config if adding features

## Deployment Notes

- Flask app runs with `debug=True` by default (disable for production)
- Default port: 5000
- No authentication/authorization implemented
- Model files must be present in `models/` directory for web app to function
