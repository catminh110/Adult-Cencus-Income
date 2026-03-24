"""
Exploratory Data Analysis Module for Adult Census Income
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EDAAnalyzer:
    """Exploratory Data Analysis for Adult Census Income Dataset"""
    
    def __init__(self, df, output_dir='reports/figures'):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def data_overview(self):
        """Generate basic data overview"""
        logger.info("Generating data overview...")
        
        overview = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': (self.df == '?').sum().to_dict(),
            'missing_percentage': ((self.df == '?').sum() / len(self.df) * 100).to_dict()
        }
        
        return overview
    
    def analyze_income_by_education(self):
        """
        Analyze income rate by education level
        Research Question: How does education affect income?
        """
        logger.info("Analyzing income by education...")
        
        # Calculate income rate by education
        education_income = self.df.groupby('education')['income'].apply(
            lambda x: (x.str.contains('>50K')).mean() * 100
        ).reset_index()
        education_income.columns = ['education', 'income_rate']
        education_income = education_income.sort_values('income_rate', ascending=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(education_income['education'], education_income['income_rate'])
        
        # Color bars based on income rate
        for bar, rate in zip(bars, education_income['income_rate']):
            if rate < 20:
                bar.set_color('#e74c3c')
            elif rate < 50:
                bar.set_color('#f39c12')
            else:
                bar.set_color('#27ae60')
        
        ax.set_xlabel('Income >50K Rate (%)', fontsize=12)
        ax.set_ylabel('Education Level', fontsize=12)
        ax.set_title('Income Rate by Education Level', fontsize=14, fontweight='bold')
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        
        # Add value labels
        for i, (idx, row) in enumerate(education_income.iterrows()):
            ax.text(row['income_rate'] + 1, i, f"{row['income_rate']:.1f}%", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/income_by_education.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {self.output_dir}/income_by_education.png")
        return education_income
    
    def analyze_income_by_hours(self):
        """
        Analyze income rate by hours worked per week
        Research Question: How does working hours affect income?
        """
        logger.info("Analyzing income by hours per week...")
        
        # Create hour bins
        self.df['hours_category'] = pd.cut(
            self.df['hours-per-week'],
            bins=[0, 20, 35, 40, 50, 100],
            labels=['Part-time (<20)', 'Half-time (20-35)', 'Standard (35-40)', 
                   'Overtime (40-50)', 'Heavy (>50)']
        )
        
        # Calculate income rate by hours category
        hours_income = self.df.groupby('hours_category', observed=False)['income'].apply(
            lambda x: (x.str.contains('>50K')).mean() * 100
        ).reset_index()
        hours_income.columns = ['hours_category', 'income_rate']
        
        # Also calculate distribution
        hours_dist = self.df['hours_category'].value_counts().reset_index()
        hours_dist.columns = ['hours_category', 'count']
        hours_dist['percentage'] = hours_dist['count'] / len(self.df) * 100
        
        # Merge
        hours_analysis = hours_income.merge(hours_dist, on='hours_category')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Income rate by hours
        bars1 = ax1.bar(range(len(hours_income)), hours_income['income_rate'], 
                       color=['#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#e74c3c'])
        ax1.set_xticks(range(len(hours_income)))
        ax1.set_xticklabels(hours_income['hours_category'], rotation=45, ha='right')
        ax1.set_ylabel('Income >50K Rate (%)', fontsize=12)
        ax1.set_title('Income Rate by Hours per Week', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, rate in zip(bars1, hours_income['income_rate']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{rate:.1f}%", ha='center', fontsize=10)
        
        # Distribution of hours
        bars2 = ax2.bar(range(len(hours_dist)), hours_dist['percentage'],
                       color=['#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#e74c3c'])
        ax2.set_xticks(range(len(hours_dist)))
        ax2.set_xticklabels(hours_dist['hours_category'], rotation=45, ha='right')
        ax2.set_ylabel('Distribution (%)', fontsize=12)
        ax2.set_title('Hours per Week Distribution', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, pct in zip(bars2, hours_dist['percentage']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{pct:.1f}%", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/income_by_hours.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {self.output_dir}/income_by_hours.png")
        return hours_analysis
    
    def analyze_demographics(self):
        """Analyze income by demographic factors"""
        logger.info("Analyzing demographics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Income by Sex
        sex_income = self.df.groupby('sex')['income'].apply(
            lambda x: (x.str.contains('>50K')).mean() * 100
        )
        ax1 = axes[0, 0]
        bars = ax1.bar(sex_income.index, sex_income.values, color=['#3498db', '#e91e63'])
        ax1.set_ylabel('Income >50K Rate (%)')
        ax1.set_title('Income by Sex')
        for bar, val in zip(bars, sex_income.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{val:.1f}%", ha='center')
        
        # Income by Race
        race_income = self.df.groupby('race')['income'].apply(
            lambda x: (x.str.contains('>50K')).mean() * 100
        ).sort_values(ascending=True)
        ax2 = axes[0, 1]
        bars = ax2.barh(race_income.index, race_income.values)
        ax2.set_xlabel('Income >50K Rate (%)')
        ax2.set_title('Income by Race')
        
        # Income by Workclass
        workclass_income = self.df.groupby('workclass')['income'].apply(
            lambda x: (x.str.contains('>50K')).mean() * 100
        ).sort_values(ascending=True)
        ax3 = axes[1, 0]
        bars = ax3.barh(workclass_income.index, workclass_income.values)
        ax3.set_xlabel('Income >50K Rate (%)')
        ax3.set_title('Income by Workclass')
        
        # Income by Occupation
        occupation_income = self.df.groupby('occupation')['income'].apply(
            lambda x: (x.str.contains('>50K')).mean() * 100
        ).sort_values(ascending=True)
        ax4 = axes[1, 1]
        bars = ax4.barh(occupation_income.index, occupation_income.values)
        ax4.set_xlabel('Income >50K Rate (%)')
        ax4.set_title('Income by Occupation')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/demographics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {self.output_dir}/demographics_analysis.png")
        
        return {
            'sex': sex_income.to_dict(),
            'race': race_income.to_dict(),
            'workclass': workclass_income.to_dict(),
            'occupation': occupation_income.to_dict()
        }
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        logger.info("Analyzing correlations...")
        
        # Select numerical columns
        num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                   'capital-loss', 'hours-per-week']
        
        # Add binary income
        self.df['income_binary'] = self.df['income'].str.contains('>50K').astype(int)
        
        # Calculate correlation matrix
        corr_matrix = self.df[num_cols + ['income_binary']].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {self.output_dir}/correlation_matrix.png")
        return corr_matrix
    
    def generate_full_report(self):
        """Generate complete EDA report with all analyses"""
        logger.info("Generating full EDA report...")
        
        # Run all analyses
        overview = self.data_overview()
        education_analysis = self.analyze_income_by_education()
        hours_analysis = self.analyze_income_by_hours()
        demographics = self.analyze_demographics()
        correlation = self.correlation_analysis()
        
        # Compile report
        report = {
            'overview': overview,
            'education_analysis': education_analysis.to_dict(),
            'hours_analysis': hours_analysis.to_dict(),
            'demographics': demographics,
            'correlation': correlation.to_dict()
        }
        
        logger.info("EDA report generated successfully!")
        return report


def main():
    """Main EDA function"""
    from config import RAW_DATA_FILE
    
    # Load data
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Initialize analyzer
    analyzer = EDAAnalyzer(df, output_dir='reports/figures')
    
    # Generate report
    report = analyzer.generate_full_report()
    
    print("EDA Complete! Generated visualizations:")
    print("  - income_by_education.png")
    print("  - income_by_hours.png")
    print("  - demographics_analysis.png")
    print("  - correlation_matrix.png")
    
    return report


if __name__ == '__main__':
    main()
