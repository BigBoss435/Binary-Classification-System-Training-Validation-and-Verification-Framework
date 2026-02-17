"""
Comprehensive Exploratory Data Analysis (EDA) Module for Melanoma Classification

This module provides detailed statistical analysis and visualization 
of the melanoma dataset including both image metadata and class distributions.

This module includes:
- Class distribution analysis with imbalance detection
- Numerical feature analysis with outlier detection
- Categorical feature analysis with frequency distributions
- Feature-target relationship analysis using statistical tests
- Comprehensive visualization generation
- Automated report generation in multiple formats

Author: Deividas Kalvelis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os
from typing import Dict, Optional
import warnings
import json
from datetime import datetime
from utils import setup_logging

# Suppress warnings for cleaner output during analysis
warnings.filterwarnings('ignore')

# Configure plotting style for consistent, professional visualizations
plt.style.use('default')
sns.set_palette("husl")

class MelanomaEDA:
    """
    Comprehensive Exploratory Data Analysis for melanoma dataset.
    
    This class provides a complete suite of EDA tools specifically designed
    for medical image classification datasets, with particular focus on:
    - Class imbalance analysis (critical for medical datasets)
    - Statistical significance testing for feature importance
    - Medical context-aware visualizations
    - Automated report generation for reproducibility

    Attributes:
        logger: Logger instance for logging analysis steps
        results: Dictionary to store EDA results
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, output_dir: Optional[str] = None):
        """
        Initialize the EDA analyzer.
        
        Args:
            logger: Logger instance to use. If None, creates a default logger.
            output_dir: Output directory for logging (used only if logger is None)
        """
        # Setup logging with fallback hierarchy for different use cases
        if logger is not None:
            # Use provided logger (preferred for integration with existing systems)
            self.logger = logger
        elif output_dir is not None:
            # Use utils logging setup for consistency
            self.logger = setup_logging(output_dir)
        else:
            # Fallback to basic logger
            self.logger = logging.getLogger(__name__)

        # Initialize results storage for analysis outputs
        self.results = {}
        
    def analyze_class_distribution(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        """
        Analyze target class distribution with detailed statistics.
        
        This method is crucial for medical datasets where class imbalance
        can significantly impac model performance and clinical relevance.

        Args:
            df: Input dataframe containing the dataset
            target_col: Name of the target column (default: 'target')
                        Expected to contain binary values (0: benign, 1: melanoma)
            
        Returns:
            Dictionary containing:
            - total_samples: Total number of samples in dataset
            - class_counts: Raw counts of each class
            - class_percentages: Percentage distribution of each class
            - imbalance_ratio: Ratio of majority to minority class
            - minority_class_percentage: Percentage of minority class (melanoma)
        """
        # Calculate basic class distribution statistics
        class_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        # Build comprehensive analysis dictionary
        analysis = {
            'total_samples': total_samples,
            'class_counts': class_counts.to_dict(),
            'class_percentages': (class_counts / total_samples * 100).to_dict(),
            # Imbalance ration: critical metric for understanding dataset bias
            'imbalance_ratio': class_counts[0] / class_counts[1] if 1 in class_counts else 0,
            # Minority class percentage: imporant for sampling strategy decisions
            'minority_class_percentage': (class_counts[1] / total_samples * 100) if 1 in class_counts else 0
        }
        
        # Log results for monitoring and debugging
        if self.logger:
            self.logger.info(f"Class Distribution Analysis:")
            self.logger.info(f"  Total samples: {total_samples}")
            self.logger.info(f"  Benign (0): {class_counts.get(0, 0)} ({analysis['class_percentages'].get(0, 0):.2f}%)")
            self.logger.info(f"  Melanoma (1): {class_counts.get(1, 0)} ({analysis['class_percentages'].get(1, 0):.2f}%)")
            self.logger.info(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}:1")
        
        return analysis
    
    def analyze_numerical_features(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis of numerical features with statistical summaries.
        
        Focuses on age analysis whichs is a critical demoraphic factor
        in melanoma risk assessment. Includes outlier detection using
        the IQR method for robust statistical analysis.

        Args:
            df: Input dataframe containing numerical features
            
        Returns:
            Dictionary with detailed statistics for each numerical feature:
            - Basic statistics (mean, median, std, min, max, quartiles)
            - Distribution properties (skewness, kurtosis)
            - Missing value analysis
            - Outlier detection results
        """
        # Define numerical columns of interest
        # Currently focusing on age as it's the primary numerical demographic factor
        numerical_cols = ['age_approx']
        # Filter to only existing columns to handle different dataset versions
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        analysis = {}
        
        # Analyze each numerical feature independently
        for col in numerical_cols:
            # Remove missing values for statistical calculations
            col_data = df[col].dropna()
            
            # Skip if no valid data available
            if len(col_data) == 0:
                continue
            
            # Calculate comprehensive statistics
            analysis[col] = {
                # Basic counts and missing value analysis
                'count': len(col_data),
                'missing_values': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,

                # Central tendency measures
                'mean': col_data.mean(),
                'median': col_data.median(),

                # Dispersion measures
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),

                # Quartile analysis for analysing distribution
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),

                # Distribution shape analysis
                'skewness': stats.skew(col_data),  # Measures asymmetry
                'kurtosis': stats.kurtosis(col_data)  # Measures tail heaviness
            }
            
            # Outlier detection using IQR method (robust to extreme values)
            Q1 = analysis[col]['q25']
            Q3 = analysis[col]['q75']
            IQR = analysis[col]['iqr']

            # Standard outlier boundaries: Q1 - 1.5*IQR and Q3 + 1.5*IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Store outlier analysis results
            analysis[col]['outliers_count'] = len(outliers)
            analysis[col]['outliers_percentage'] = (len(outliers) / len(col_data)) * 100
        
        # Log completion status
        if self.logger:
            self.logger.info(f"Numerical Features Analysis completed for {len(numerical_cols)} features")
            
        return analysis
    
    def analyze_categorical_features(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis of categorical features in the melanoma dataset.
        
        Analyze key demographic and anatomical features that are clinically
        relevant for melanoma risk assessment:
        - sex: Gender-based risk differences
        - anatom_site_general_challenge: Body location of lesions
        - diagnosis: Specific diagnosis categories
        - benign_malignant: Primary classification

        Args:
            df: Input dataframe contaning categorical features
            
        Returns:
            Dictionary with analysis for each categorical feature:
            - Unique value counts and distributions
            - Missing value analysis
            - Most frequent categories
            - Percentage distributions
        """
        # Define categorical columns of clinical interest
        categorical_cols = ['sex', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant']
        # Filter to existing columns for dataset compatibility
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        analysis = {}
        
        # Analyze each categorical feature
        for col in categorical_cols:
            # Calculate value distributions (including NaN for missing data analysis)
            value_counts = df[col].value_counts(dropna=False)
            total_count = len(df)
            
            # Build comprehensive feature analysis
            analysis[col] = {
                # Diversity metrics
                'unique_values': df[col].nunique(),

                # Missing data analysis (critical for data quality assessment)
                'missing_values': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / total_count) * 100,

                # Distribution analysis
                'value_counts': value_counts.to_dict(),
                'value_percentages': (value_counts / total_count * 100).to_dict(),

                # Dominant category identification
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_percentage': (value_counts.iloc[0] / total_count * 100) if len(value_counts) > 0 else 0
            }
        
        # Log completion status
        if self.logger:
            self.logger.info(f"Categorical Features Analysis completed for {len(categorical_cols)} features")
            
        return analysis
    
    def analyze_feature_target_relationships(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        """
        Analyze statistical relationships between features and target variable.
        
        This method performs appropriate statistical tests to determine
        which features have significant associations with melanoma risk:
        - T-tests for numerical features (comparing means between classes)
        - Chi-square tests for categorical features (testing independence)

        Results help identify the most predictive features for model development.

        Args:
            df: Input dataframe
            target_col: Target column name (binary: 0 - benign, 1 - melanoma)
            
        Returns:
            Dictionary with statistical test results and effect sizes for each feature
        """
        relationships = {}
        
        # Numerical features vs target
        # Analyze continouns variables using t-tests to compare group means
        numerical_cols = ['age_approx']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        for col in numerical_cols:
            # Separate data by target class for comparison
            benign_data = df[df[target_col] == 0][col].dropna()
            malignant_data = df[df[target_col] == 1][col].dropna()
            
            # Perform independent t-test if both groups have data
            if len(benign_data) > 0 and len(malignant_data) > 0:
                try:
                    # Two-sample t-test: tests if means are significantly different
                    t_stat, p_value = stats.ttest_ind(benign_data, malignant_data)
                    
                    # Store comprehensive statistical results
                    relationships[col] = {
                        'type': 'numerical',
                        # Group statistics for effect size interpretation
                        'benign_mean': benign_data.mean(),
                        'malignant_mean': malignant_data.mean(),
                        'benign_std': benign_data.std(),
                        'malignant_std': malignant_data.std(),
                        # Statistical test results
                        't_statistic': t_stat,
                        'p_value': p_value,
                        # Significance flag (α = 0.05)
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    # Log any statistical test failures for debugging
                    if self.logger:
                        self.logger.warning(f"Could not perform t-test for {col}: {e}")
        
        # Categorical features vs target
        # Analyze discrete variables using chi-square tests for independence
        categorical_cols = ['sex', 'anatom_site_general_challenge', 'diagnosis']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            try:
                # Create contingency table for chi-square test
                contingency_table = pd.crosstab(df[col], df[target_col])
                
                # Perform chi-square test if table has sufficient dimensions
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    # Chi-square test: tests independence between categorical variables
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    # Calculate melanoma rates for each category (clinical interpretation)
                    melanoma_rates = {}
                    for category in df[col].unique():
                        if pd.notna(category):
                            subset = df[df[col] == category]
                            if len(subset) > 0:
                                # Calculate percentage of melanoma cases in each category
                                melanoma_rates[category] = subset[target_col].mean() * 100
                    
                    # Store comprehensive categorical analysis results
                    relationships[col] = {
                        'type': 'categorical',
                        # Raw contingency data for detailed analysis
                        'contingency_table': contingency_table.to_dict(),
                        # Statistical test results
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'significant': p_value < 0.05,
                        # Clinical interpretation: melanoma risk by category
                        'melanoma_rates_by_category': melanoma_rates
                    }
            except Exception as e:
                # Log any statistical test failures for debugging
                if self.logger:
                    self.logger.warning(f"Could not perform chi-square test for {col}: {e}")
        
        return relationships
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str, target_col: str = 'target'):
        """
        Create comprehensive visualizations for the melanoma dataset.
        
        Generates publication-ready plots that provide clinical insights:
        - Class distribution plots (critical for understanding dataset balance)
        - Age distribution analysis (key demographic risk factor)
        - Categorical feature analysis (anatomical and demographic factors)

        All plots are saved as high-resolution PNG files for reports and presentations.

        Args:
            df: Input dataframe
            output_dir: Directory to save plots
            target_col: Target column name
        """

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Class Distribution Plot
        # Essential for understanding dataset characteristics and potential bias
        plt.figure(figsize=(12, 5))
        class_counts = df[target_col].value_counts()
        colors = ['lightblue', 'lightcoral']  # Distingishable colors for classes
        
        # Bar plot showing absolute counts
        plt.subplot(1, 2, 1)
        class_counts.plot(kind='bar', color=colors)
        plt.title('Class Distribution (Count)', fontsize=14)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks([0, 1], ['Benign', 'Melanoma'], rotation=0)
        
        # Pie chart showing relative proportions
        plt.subplot(1, 2, 2)
        class_percentages = class_counts / len(df) * 100
        plt.pie(class_percentages, labels=['Benign', 'Melanoma'], colors=colors, autopct='%1.2f%%')
        plt.title('Class Distribution (Percentage)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Age Distribution (if available)
        # Age is a critical risk factor for melanoma development
        if 'age_approx' in df.columns:
            plt.figure(figsize=(15, 5))
            
            # Overall age distribution histogram
            plt.subplot(1, 3, 1)
            df['age_approx'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Age Distribution (Overall)', fontsize=14)
            plt.xlabel('Age', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
            # Age distribution by class (overlaid histograms)
            plt.subplot(1, 3, 2)
            benign_ages = df[df[target_col] == 0]['age_approx'].dropna()
            malignant_ages = df[df[target_col] == 1]['age_approx'].dropna()
            
            plt.hist(benign_ages, bins=20, alpha=0.7, label='Benign', color='lightblue')
            plt.hist(malignant_ages, bins=20, alpha=0.7, label='Melanoma', color='lightcoral')
            plt.title('Age Distribution by Class', fontsize=14)
            plt.xlabel('Age', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            
            # Box plot for statistical comparison
            plt.subplot(1, 3, 3)
            data_for_box = [benign_ages.values, malignant_ages.values]
            plt.boxplot(data_for_box, labels=['Benign', 'Melanoma'])
            plt.title('Age Distribution Box Plot by Class', fontsize=14)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Age', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'age_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Categorical Features Analysis
        # Analyze key demographic and anatomical features
        categorical_cols = ['sex', 'anatom_site_general_challenge', 'diagnosis']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            plt.figure(figsize=(15, 10))
            
            # Overall distribution of categories
            plt.subplot(2, 2, 1)
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar', color='lightgreen')
            plt.title(f'{col.replace("_", " ").title()} - Overall Distribution', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Stacked bar chart: distribution by class
            plt.subplot(2, 2, 2)
            pd.crosstab(df[col], df[target_col]).plot(kind='bar', stacked=True, 
                                                     color=['lightblue', 'lightcoral'])
            plt.title(f'{col.replace("_", " ").title()} - Distribution by Class', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(['Benign', 'Melanoma'])
            
            # Melanoma rate by category (clinical risk assessment)
            plt.subplot(2, 2, 3)
            melanoma_rates = df.groupby(col)[target_col].mean() * 100
            melanoma_rates.plot(kind='bar', color='orange')
            plt.title(f'Melanoma Rate by {col.replace("_", " ").title()}', fontsize=12)
            plt.ylabel('Melanoma Rate (%)', fontsize=10)
            plt.xticks(rotation=45, ha='right')
            
            # Proportional distribution (normalized view)
            plt.subplot(2, 2, 4)
            prop_table = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            prop_table.plot(kind='bar', stacked=True, color=['lightblue', 'lightcoral'])
            plt.title(f'{col.replace("_", " ").title()} - Proportional Distribution', fontsize=12)
            plt.ylabel('Percentage', fontsize=10)
            plt.xticks(rotation=45, ha='right')
            plt.legend(['Benign', 'Melanoma'])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Log completion status
        if self.logger:
            self.logger.info(f"Visualizations saved to {output_dir}")
    
    def run_comprehensive_eda(self, df: pd.DataFrame, output_dir: str, 
                            target_col: str = 'target') -> Dict:
        """
        Execute complete EDA pipeline and generate comprehensive reports.

        This is the main method that orchestrates all analysis components:
        - Dataset overview and basic statistics
        - Class distribution analysis
        - Numerical feature analysis
        - Categorical feature analysis
        - Feature-target relationship analysis
        - Visualization generation
        - Report generation in multiple formats
        
        Args:
            df: Input dataframe containing the melanoma dataset
            output_dir: Output directory for all results and reports
            target_col: Target column name (default: 'target')
            
        Returns:
            Complete EDA results dictionary containing all analysis outputs
        """
        if self.logger:
            self.logger.info("Starting comprehensive EDA analysis...")
        
        # Initialize results structure with metadata
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(df),
                'total_features': len(df.columns),
                # Memory usage analysis and performance considerations
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values_total': df.isnull().sum().sum()
            }
        }
        
        # Execute all analysis components sequentially

        # Class distribution analysis (critical for medical datasets)
        results['class_analysis'] = self.analyze_class_distribution(df, target_col)
        
        # Numerical features analysis (age demographics)
        results['numerical_analysis'] = self.analyze_numerical_features(df)
        
        # Categorical features analysis (clinical factors)
        results['categorical_analysis'] = self.analyze_categorical_features(df)
        
        # Feature-target relationships (statistical significance testing)
        results['feature_target_relationships'] = self.analyze_feature_target_relationships(df, target_col)
        
        # Generate comprehensive visualizations
        viz_dir = os.path.join(output_dir, 'eda_visualizations')
        self.create_visualizations(df, viz_dir, target_col)
        
        # Save machine-readable results (JSON format for programmatic access)
        with open(os.path.join(output_dir, 'eda_report.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate human-readable summary report
        self._generate_text_summary(results, output_dir)
        
        if self.logger:
            self.logger.info("Comprehensive EDA analysis completed")
        
        return results
    
    def _generate_text_summary(self, results: Dict, output_dir: str):
        """
        Generate human-readable EDA summary.

        Creates a formatted text file with key insights and findings
        that can be easily shared with stakeholders, included in reports,
        or used for quick reference during model development.

        Args:
            results: Complete EDA results dictionary
            output_dir: Output directory for the summary file
        """
        summary_path = os.path.join(output_dir, 'eda_summary.txt')
        
        with open(summary_path, 'w') as f:
            # Header section
            f.write("=" * 60 + "\n")
            f.write("MELANOMA DATASET - EXPLORATORY DATA ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Samples: {results['dataset_info']['total_samples']:,}\n")
            f.write(f"Total Features: {results['dataset_info']['total_features']}\n")
            f.write(f"Memory Usage: {results['dataset_info']['memory_usage'] / 1024 / 1024:.2f} MB\n")
            f.write(f"Total Missing Values: {results['dataset_info']['missing_values_total']:,}\n\n")
            
            # Class distribution
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            class_info = results['class_analysis']
            f.write(f"Benign Cases: {class_info['class_counts'].get(0, 0):,} ({class_info['class_percentages'].get(0, 0):.2f}%)\n")
            f.write(f"Melanoma Cases: {class_info['class_counts'].get(1, 0):,} ({class_info['class_percentages'].get(1, 0):.2f}%)\n")
            f.write(f"Imbalance Ratio: {class_info.get('imbalance_ratio', 0):.1f}:1\n\n")
            
            # Feature insights
            # Focus on statistically significant findings for clinical relevance
            if 'feature_target_relationships' in results:
                f.write("KEY INSIGHTS:\n")
                f.write("-" * 20 + "\n")
                relationships = results['feature_target_relationships']
                
                # Report significant relationships with clinical interpretation
                for feature, analysis in relationships.items():
                    if analysis.get('significant', False):
                        if analysis['type'] == 'numerical':
                            # Report numerical feature findings (e.g., age differences)
                            f.write(f"• {feature}: Significant difference between classes (p={analysis['p_value']:.4f})\n")
                            f.write(f"  Benign mean: {analysis['benign_mean']:.1f}, Melanoma mean: {analysis['malignant_mean']:.1f}\n")
                        elif analysis['type'] == 'categorical':
                            # Report categorical feature findings (e.g., risk by category)
                            f.write(f"• {feature}: Significant association with melanoma risk (p={analysis['p_value']:.4f})\n")
                            rates = analysis.get('melanoma_rates_by_category', {})
                            # List melanoma rates for each category for clinical interpretation
                            for category, rate in rates.items():
                                f.write(f"  {category}: {rate:.2f}% melanoma rate\n")
                        f.write("\n")

        # Log summary generation completion
        if self.logger:
            self.logger.info(f"EDA summary saved to {summary_path}")


def run_eda_analysis(df: pd.DataFrame, output_dir: str, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Convenience function to run complete EDA analysis.
    
    This function provides a simple interface for external modules
    to perform comprehensive EDA without needing to instantiate
    the MelanomaEDA class directly.

    Args:
        df: Input dataframe containing melanoma datase
        output_dir: Output directory for all analysis results
        logger: Optional logger instance for consistent logging
        
    Returns:
        Complete EDA results dictionary with all analysis outputs
    """
    eda_analyzer = MelanomaEDA(logger)
    return eda_analyzer.run_comprehensive_eda(df, output_dir)
