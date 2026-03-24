"""
Model Training and Evaluation Module
Implements baselines, Logistic Regression, Linear SVC with ablation studies
"""
import warnings
# Suppress sklearn deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model trainer for Adult Census Income classification.
    Implements baselines and ablation studies.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.cv_folds = 5
        
    def create_baselines(self):
        """Create baseline models"""
        logger.info("Creating baseline models...")
        
        baselines = {
            'Majority': DummyClassifier(strategy='most_frequent', random_state=self.random_state),
            'Stratified': DummyClassifier(strategy='stratified', random_state=self.random_state)
        }
        
        return baselines
    
    def create_logistic_regression_models(self):
        """
        Create Logistic Regression models for ablation study
        - L1 vs L2 regularization
        - Different C values
        """
        logger.info("Creating Logistic Regression models...")
        
        models = {}
        
        # L1 Regularization
        for C in [0.01, 0.1, 1, 10]:
            models[f'LogReg_L1_C{C}'] = LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=1000
            )
        
        # L2 Regularization
        for C in [0.01, 0.1, 1, 10]:
            models[f'LogReg_L2_C{C}'] = LogisticRegression(
                penalty='l2', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=1000
            )
        
        return models
    
    def create_linear_svc_models(self):
        """
        Create Linear SVC models for ablation study
        - Different C values
        """
        logger.info("Creating Linear SVC models...")
        
        models = {}
        
        for C in [0.01, 0.1, 1, 10]:
            models[f'LinearSVC_C{C}'] = LinearSVC(
                C=C, random_state=self.random_state, max_iter=2000,
                dual=False
            )
        
        return models
    
    def evaluate_model_cv(self, model, X, y, model_name):
        """
        Evaluate model using Stratified Cross-Validation
        Returns mean +- std for each metric
        """
        logger.info(f"Evaluating {model_name} with Stratified {self.cv_folds}-Fold CV...")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Metrics to evaluate
        scoring = ['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'accuracy']
        
        # Cross-validation
        cv_results = cross_validate(
            model, X, y, cv=skf, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )
        
        # Compile results
        results = {}
        for metric in scoring:
            mean_val = cv_results[f'test_{metric}'].mean()
            std_val = cv_results[f'test_{metric}'].std()
            results[metric] = {'mean': mean_val, 'std': std_val}
        
        return results
    
    def train_and_evaluate_all(self, X, y):
        """Train and evaluate all models"""
        logger.info("Starting model training and evaluation...")
        
        # Combine all models
        all_models = {}
        all_models.update(self.create_baselines())
        all_models.update(self.create_logistic_regression_models())
        all_models.update(self.create_linear_svc_models())
        
        # Evaluate each model
        for name, model in all_models.items():
            try:
                results = self.evaluate_model_cv(model, X, y, name)
                self.results[name] = results
                self.models[name] = model
                
                # Log results
                logger.info(f"{name}:")
                logger.info(f"  ROC-AUC: {results['roc_auc']['mean']:.4f} +- {results['roc_auc']['std']:.4f}")
                logger.info(f"  PR-AUC: {results['average_precision']['mean']:.4f} +- {results['average_precision']['std']:.4f}")
                logger.info(f"  F1: {results['f1']['mean']:.4f} +- {results['f1']['std']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        return self.results
    
    def plot_roc_curves(self, X, y, output_dir):
        """Plot ROC curves for all models"""
        logger.info("Plotting ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Stratified K-Fold for consistent evaluation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Select key models to plot
        key_models = [
            'LogReg_L2_C1', 'LogReg_L1_C1',
            'LinearSVC_C1', 'Majority'
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
        
        for (name, color) in zip(key_models, colors):
            if name not in self.models:
                continue
                
            model = self.models[name]
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X_val)
                else:
                    continue
                
                fpr, tpr, _ = roc_curve(y_val, y_score)
                roc_auc = roc_auc_score(y_val, y_score)
                
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
            
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            
            ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                   label=f'{name} (AUC = {mean_auc:.3f} +- {std_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Cross-Validation', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_dir}/roc_curves.png")
    
    def plot_pr_curves(self, X, y, output_dir):
        """Plot Precision-Recall curves for all models"""
        logger.info("Plotting PR curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Select key models
        key_models = [
            'LogReg_L2_C1', 'LogReg_L1_C1',
            'LinearSVC_C1', 'Majority'
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
        
        for (name, color) in zip(key_models, colors):
            if name not in self.models:
                continue
                
            model = self.models[name]
            precisions = []
            aucs = []
            mean_recall = np.linspace(0, 1, 100)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X_val)
                else:
                    continue
                
                precision, recall, _ = precision_recall_curve(y_val, y_score)
                pr_auc = average_precision_score(y_val, y_score)
                
                # Interpolate precision values
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                precisions.append(interp_precision)
                aucs.append(pr_auc)
            
            mean_precision = np.mean(precisions, axis=0)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            
            ax.plot(mean_recall, mean_precision, color=color, lw=2,
                   label=f'{name} (AP = {mean_auc:.3f} +- {std_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = y.sum() / len(y)
        ax.axhline(y=baseline, color='gray', linestyle='--', 
                  label=f'Baseline (random) = {baseline:.3f}')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - Cross-Validation', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_dir}/pr_curves.png")
    
    def plot_coefficients(self, feature_names, output_dir):
        """Plot coefficient analysis for interpretable models"""
        logger.info("Plotting coefficient analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # Logistic Regression Coefficients
        if 'LogReg_L2_C1' in self.models:
            lr_model = self.models['LogReg_L2_C1']
            if hasattr(lr_model, 'coef_'):
                coefs = lr_model.coef_[0]
                
                # Get top 20 features by absolute coefficient value
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefs
                })
                coef_df['abs_coef'] = coef_df['coefficient'].abs()
                coef_df = coef_df.nlargest(20, 'abs_coef')
                
                ax1 = axes[0]
                colors = ['#27ae60' if c > 0 else '#e74c3c' for c in coef_df['coefficient']]
                bars = ax1.barh(range(len(coef_df)), coef_df['coefficient'], color=colors)
                ax1.set_yticks(range(len(coef_df)))
                ax1.set_yticklabels(coef_df['feature'], fontsize=9)
                ax1.set_xlabel('Coefficient Value', fontsize=12)
                ax1.set_title('Top 20 Features - Logistic Regression (L2)', fontsize=12, fontweight='bold')
                ax1.axvline(x=0, color='black', linewidth=0.5)
        
        # Linear SVC Coefficients
        if 'LinearSVC_C1' in self.models:
            svc_model = self.models['LinearSVC_C1']
            if hasattr(svc_model, 'coef_'):
                coefs = svc_model.coef_[0]
                
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefs
                })
                coef_df['abs_coef'] = coef_df['coefficient'].abs()
                coef_df = coef_df.nlargest(20, 'abs_coef')
                
                ax2 = axes[1]
                colors = ['#27ae60' if c > 0 else '#e74c3c' for c in coef_df['coefficient']]
                bars = ax2.barh(range(len(coef_df)), coef_df['coefficient'], color=colors)
                ax2.set_yticks(range(len(coef_df)))
                ax2.set_yticklabels(coef_df['feature'], fontsize=9)
                ax2.set_xlabel('Coefficient Value', fontsize=12)
                ax2.set_title('Top 20 Features - Linear SVC', fontsize=12, fontweight='bold')
                ax2.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/coefficient_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_dir}/coefficient_plot.png")
    
    def save_results_table(self, output_dir):
        """Save results as a formatted table"""
        logger.info("Saving results table...")
        
        # Create results dataframe
        results_list = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            for metric_name, values in metrics.items():
                row[f'{metric_name}_mean'] = values['mean']
                row[f'{metric_name}_std'] = values['std']
            results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        
        # Save to CSV
        results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        
        # Create formatted table for report
        formatted_results = []
        for model_name, metrics in self.results.items():
            formatted_results.append({
                'Model': model_name,
                'ROC-AUC': f"{metrics['roc_auc']['mean']:.4f} +- {metrics['roc_auc']['std']:.4f}",
                'PR-AUC': f"{metrics['average_precision']['mean']:.4f} +- {metrics['average_precision']['std']:.4f}",
                'F1-Score': f"{metrics['f1']['mean']:.4f} +- {metrics['f1']['std']:.4f}",
                'Precision': f"{metrics['precision']['mean']:.4f} +- {metrics['precision']['std']:.4f}",
                'Recall': f"{metrics['recall']['mean']:.4f} +- {metrics['recall']['std']:.4f}",
                'Accuracy': f"{metrics['accuracy']['mean']:.4f} +- {metrics['accuracy']['std']:.4f}"
            })
        
        formatted_df = pd.DataFrame(formatted_results)
        formatted_df.to_csv(f'{output_dir}/model_comparison_formatted.csv', index=False)
        
        logger.info(f"Saved: {output_dir}/model_comparison.csv")
        return formatted_df
    
    def save_best_model(self, X, y, output_dir):
        """Save the best performing model"""
        logger.info("Saving best model...")
        
        # Find best model by ROC-AUC
        best_model_name = None
        best_score = 0
        
        for name, metrics in self.results.items():
            if metrics['roc_auc']['mean'] > best_score and name != 'Majority':
                best_score = metrics['roc_auc']['mean']
                best_model_name = name
        
        if best_model_name:
            best_model = self.models[best_model_name]
            best_model.fit(X, y)
            
            joblib.dump(best_model, f'{output_dir}/best_model.joblib')
            logger.info(f"Best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
            
            # Save model info
            model_info = {
                'model_name': best_model_name,
                'roc_auc': best_score,
                'timestamp': datetime.now().isoformat()
            }
            with open(f'{output_dir}/best_model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return best_model_name
        
        return None


def main():
    """Main training function"""
    from data_preprocessing import DataPreprocessor
    from config import (
        RAW_DATA_FILE, CATEGORICAL_COLUMNS, 
        NUMERICAL_COLUMNS, TARGET_COLUMN, RANDOM_SEED
    )
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(
        categorical_columns=CATEGORICAL_COLUMNS,
        numerical_columns=NUMERICAL_COLUMNS,
        target_column=TARGET_COLUMN
    )
    
    df = preprocessor.load_data(RAW_DATA_FILE)
    X, y, feature_names = preprocessor.fit_transform(df)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=RANDOM_SEED)
    
    # Train and evaluate all models
    results = trainer.train_and_evaluate_all(X, y)
    
    # Generate visualizations
    output_dir = 'reports/figures'
    trainer.plot_roc_curves(X, y, output_dir)
    trainer.plot_pr_curves(X, y, output_dir)
    trainer.plot_coefficients(feature_names, output_dir)
    
    # Save results
    results_df = trainer.save_results_table(output_dir)
    
    # Save best model
    models_dir = 'models'
    best_model = trainer.save_best_model(X, y, models_dir)
    
    # Save preprocessor
    preprocessor.save(f'{models_dir}/preprocessor.joblib')
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest Model: {best_model}")
    
    return trainer, results_df


if __name__ == '__main__':
    main()
