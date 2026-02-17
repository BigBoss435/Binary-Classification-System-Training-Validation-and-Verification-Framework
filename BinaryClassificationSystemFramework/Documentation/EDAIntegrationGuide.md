# Comprehensive Exploratory Data Analysis (EDA) Integration

## Overview

The melanoma classification system now includes comprehensive EDA capabilities that provide detailed statistical analysis and visualization of your dataset. The EDA module helps you understand data distributions, feature relationships, and potential issues before model training.

## Features

### Statistical Analysis
- **Class Distribution**: Detailed analysis of melanoma vs benign cases with imbalance metrics
- **Numerical Features**: Statistical summaries (mean, median, std, outliers) for age and other numerical features
- **Categorical Features**: Distribution analysis for sex, anatomical location, and diagnosis categories
- **Feature-Target Relationships**: Statistical tests (t-tests, chi-square) to identify significant relationships

### Visualizations
- Class distribution plots (bar charts and pie charts)
- Age distribution analysis by class
- Categorical feature analysis with melanoma rates
- Box plots and histograms for numerical features

### Reports
- **JSON Report**: `eda_report.json` - Detailed machine-readable analysis
- **Text Summary**: `eda_summary.txt` - Human-readable executive summary
- **Visualizations**: PNG files in `eda_visualizations/` directory

## Integration Methods

### 1. Automatic EDA (Default)
EDA runs automatically during the normal training pipeline:

```python
# In main.py, EDA is controlled by config
from config import RUN_EDA

# EDA runs automatically when dataset is loaded
dataset_components = setup_dataset_pipeline(
    csv_path=CSV_PATH,
    data_dir=DATA_DIR,
    run_eda=RUN_EDA,  # True by default
    logger=logger
)
```

### 2. Manual EDA Integration
Add EDA to your existing data loading:

```python
from data_validation import DataValidationPipeline
from exploratory_data_analysis import MelanomaEDA

# Load your data
df = pd.read_csv("data/ISIC_2020_Training_GroundTruth_v2.csv")
df['filepath'] = df['image_name'].apply(lambda x: f"data/jpeg/train/{x}.jpg")

# Run EDA
eda_analyzer = MelanomaEDA(logger)
eda_results = eda_analyzer.run_comprehensive_eda(df, "results/eda_analysis")
```

### 3. Enhanced Data Validation
EDA is integrated into the validation pipeline:

```python
from data_validation import DataValidationPipeline

validator = DataValidationPipeline(logger)

# This now includes EDA analysis automatically
validation_passed, results = validator.run_comprehensive_validation(df)

# Access EDA results
if "eda_analysis" in results:
    eda_data = results["eda_analysis"]
    class_info = eda_data["class_analysis"]
    feature_relationships = eda_data["feature_target_relationships"]
```

### 4. Standalone EDA Script
Run EDA independently without training:

```bash
# Use default paths
python run_eda.py

# Specify custom paths
python run_eda.py data/custom.csv data/images/ results/my_eda/
```

## Configuration Options

### In config.py
```python
# Control EDA execution
RUN_EDA = True  # Set to False to skip EDA during training
```

### In dataset loading
```python
# Control EDA per function call
df = load_and_validate_data(
    csv_path="data.csv",
    images_dir="images/",
    logger=logger,
    run_validation=True,  # Run validation pipeline
    run_eda=True         # Run EDA analysis
)
```

## Output Structure

```
results/
├── eda_analysis/
│   ├── eda_report.json           # Detailed analysis results
│   ├── eda_summary.txt           # Human-readable summary
│   └── eda_visualizations/       # Generated plots
│       ├── class_distribution.png
│       ├── age_analysis.png
│       ├── sex_analysis.png
│       ├── anatom_site_general_challenge_analysis.png
│       └── diagnosis_analysis.png
├── data_validation_report.json   # Validation results
└── validation_summary.txt        # Validation summary
```

## Key Insights Generated

### Dataset Overview
- Total samples and features
- Memory usage
- Missing values analysis

### Class Balance Analysis
- Melanoma vs benign distribution
- Imbalance ratio
- Prevalence statistics

### Feature Analysis
- Age distribution differences between classes
- Gender-based melanoma risk
- Anatomical location risk factors
- Diagnosis category analysis

### Statistical Relationships
- Features with significant associations to melanoma risk
- P-values and effect sizes
- Melanoma rates by category

## Performance Considerations

- **Automatic Mode**: EDA runs only if data validation passes (saves time on bad data)
- **Visualization Generation**: Creates high-quality plots but may take time for large datasets
- **Memory Usage**: Calculates and reports actual memory consumption
- **Sampling**: Some analyses use sampling for performance on very large datasets

## Medical Context

The EDA module is designed specifically for medical datasets and includes:
- **Clinical Relevance**: Focus on medically relevant patterns
- **Risk Stratification**: Analysis of melanoma risk by demographic and clinical factors
- **Quality Assurance**: Integration with data validation for comprehensive quality checks
- **Interpretability**: Human-readable summaries for clinical stakeholders

## Integration with Existing Workflow

Your existing code will continue to work unchanged. EDA is added as an optional enhancement that:
1. Runs automatically during normal data loading (controlled by `RUN_EDA` config)
2. Integrates seamlessly with existing validation pipeline
3. Provides additional insights without affecting model training
4. Can be disabled for faster iteration during development

The EDA analysis complements your existing outlier detection and data validation features, providing a comprehensive view of your dataset before model training begins.
