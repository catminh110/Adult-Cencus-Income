"""
Data Preprocessing Module for Adult Census Income Prediction
Handles missing values, encoding, and scaling
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessor for Adult Census Income dataset.
    Handles missing values ('?'), one-hot encoding, and scaling.
    """
    
    def __init__(self, categorical_columns, numerical_columns, target_column):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.preprocessor = None
        self.target_mapping = {'<=50K': 0, '>50K': 1}
        
    def load_data(self, filepath):
        """Load raw data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values represented as '?' in the dataset.
        Strategy: Replace '?' with 'Unknown' for categorical variables
        """
        logger.info("Handling missing values...")
        df_clean = df.copy()
        
        # Count missing values before
        missing_before = (df_clean == '?').sum().sum()
        logger.info(f"Missing values ('?') before processing: {missing_before}")
        
        # Replace '?' with 'Unknown' for categorical columns
        for col in self.categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace('?', 'Unknown')
        
        # Verify no '?' remains
        missing_after = (df_clean == '?').sum().sum()
        logger.info(f"Missing values ('?') after processing: {missing_after}")
        
        return df_clean
    
    def create_preprocessor(self):
        """
        Create sklearn preprocessor pipeline with:
        - One-hot encoding for categorical variables
        - Standard scaling for numerical variables
        """
        logger.info("Creating preprocessing pipeline...")
        
        # Numerical pipeline: Standard scaling
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: One-hot encoding
        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_columns),
            ('cat', categorical_pipeline, self.categorical_columns)
        ], remainder='drop')
        
        logger.info("Preprocessor created successfully")
        return self.preprocessor
    
    def prepare_target(self, y):
        """Convert target to binary (0/1)"""
        # Handle both <=50K and <=50K. formats
        y_clean = y.str.replace('.', '', regex=False)
        return y_clean.map(self.target_mapping)
    
    def fit_transform(self, df):
        """
        Fit preprocessor and transform data
        Returns: X (features), y (target), feature_names
        """
        logger.info("Fitting and transforming data...")
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Separate features and target
        X = df_clean.drop(columns=[self.target_column])
        y = self.prepare_target(df_clean[self.target_column])
        
        # Create and fit preprocessor
        if self.preprocessor is None:
            self.create_preprocessor()
        
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        logger.info(f"Transformed data shape: {X_transformed.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X_transformed, y.values, feature_names
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet. Call fit_transform first.")
        
        df_clean = self.handle_missing_values(df)
        
        if self.target_column in df_clean.columns:
            X = df_clean.drop(columns=[self.target_column])
            y = self.prepare_target(df_clean[self.target_column])
            return self.preprocessor.transform(X), y.values
        else:
            X = df_clean
            return self.preprocessor.transform(X)
    
    def get_feature_names(self):
        """Get feature names after transformation"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet.")
        
        # Get numerical feature names
        num_features = self.numerical_columns
        
        # Get categorical feature names from one-hot encoder
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(self.categorical_columns)
        
        # Combine
        all_features = list(num_features) + list(cat_features)
        return all_features
    
    def save(self, filepath):
        """Save preprocessor to file"""
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from file"""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def main():
    """Main preprocessing function"""
    from config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE, 
        CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN
    )
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        categorical_columns=CATEGORICAL_COLUMNS,
        numerical_columns=NUMERICAL_COLUMNS,
        target_column=TARGET_COLUMN
    )
    
    # Load data
    df = preprocessor.load_data(RAW_DATA_FILE)
    
    # Fit and transform
    X, y, feature_names = preprocessor.fit_transform(df)
    
    # Save processed data
    processed_df = pd.DataFrame(X, columns=feature_names)
    processed_df['target'] = y
    processed_df.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info(f"Processed data saved to {PROCESSED_DATA_FILE}")
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.joblib')
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    return X, y, feature_names


if __name__ == '__main__':
    main()
