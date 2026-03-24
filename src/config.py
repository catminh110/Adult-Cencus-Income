"""
Configuration file for Adult Census Income Prediction Project
"""
import os
import numpy as np

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths - Use relative path from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Data files
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, 'adult.csv')
TEST_DATA_FILE = os.path.join(DATA_RAW_DIR, 'adult_test.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_PROCESSED_DIR, 'adult_processed.csv')

# Column definitions
CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

NUMERICAL_COLUMNS = [
    'age', 'fnlwgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

TARGET_COLUMN = 'income'

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': 'liblinear',
        'max_iter': 1000
    },
    'linear_svc': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': 2000
    }
}

# Cross-validation
CV_FOLDS = 5

# Metrics
METRICS = ['roc_auc', 'average_precision', 'f1']
