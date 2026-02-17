# Binary-Classification-System-Training-Validation-and-Verification-Framework
# Melanoma Classification System: A Comprehensive Framework for Medical Binary Classification

A production-ready, research-oriented deep learning framework for melanoma detection that demonstrates best practices in medical AI development. This system provides a complete pipeline from data validation to model deployment, specifically designed for binary classification tasks with severe class imbalance.

## Main Goals

This framework addresses the unique challenges of medical image classification:
- Clinical-Grade Validation: Comprehensive data quality assurance using automated validation pipelines
- Robust Performance Estimation: Cross-validation and ensemble methods for unbiased evaluation
- Class Imbalance Handling: Multiple strategies for dealing with rare positive cases (~2% of the dataset is melanoma)
- Reproducibility: Complete configuration management and detailed logging for regulatory compliance
- Medical Context: Statistical analysis and interpretability features for clinical deployment

## Target Application  
**Current Use Case**: ISIC 2020 Melanoma Classification  
- Dataset: 33,126 dermoscopic images (98% benign, 2% melanoma)  
- Task: Binary classification (melanoma vs. benign)  
- Architecture: EfficientNet-B3 with transfer learning  
- Performance Focus: High sensitivity (recall) for melanoma detection  

## Key Features
### 1. Comprehensive Data Pipeline

**Automated Data Validation (data_validation.py)** 
```
validator = DataValidationPipeline(logger)
validation_passed, results = validator.run_comprehensive_validation(df)
```

**Key Features:**
- Great Expectations Integration: Schema validation, data type checks, null value detection
- Image Integrity Verification: Corruption detection, format validation, size checks
- Distribution Analysis: Class balance monitoring, duplicate detection, outlier identification
- Statistical Quality Control: IQR method, Modified Z-sciore, Isolation Forest for anomaly detection

**Outputs**:
- data_validation_report.json: Machine-readable validation results
- validation_summary.txt: Human-readable quality report

**Exploratory Data Analysis (exploratory_data_analysis.py)**
```
eda_analyzer = MelanomaEDA(logger)
results = eda_analyzer.run_comprehensive_eda(df, output_dir)
```

**Analyses Performed:**
- Class Distribution: Imbalance ratios, prevelance statistics
- Numerical Features: Age analysis with outlier detection (IQR, Modified Z-score)
- Categorical Features: Sex, anatomical location, diagnosis distributions
- Statistical Testing: T-tests for numerical features, Chi-square for categorical relationships
- Clinical Insights: Risk stratification by demographic and anatomical factors

**Visualizations Generated:**
- Class distribution plots (bar chart)
- Age distribution by diagnosis (histograms, box plots)
- Categorical feature analysis with melanoma rates
- High-resolution PNG exports for publication

**Outputs:**
- eda_report.json: Complete statistical analysis
- eda_summary.txt: Text format summary for human readers
- eda_visualizations/: Publication-ready plots

### 2. Advanced Data Augmentation

**Dual Augmentation Support (dataset.py)**

- Albumentations Pipeline (Medical Image Optimized)
```
USE_ALBUMENTATIONS = True  # config.py

# Augmentations include:
# - Geometric: Rotation (±20°), flips, shifts, elastic transforms
# - Color: HSV adjustments, contrast, brightness, Clahe
# - Medical-specific: GridDistortion, CoarseDropout, RandomShadow
```

- PyTorch Transforms (Alternative):
```
USE_ALBUMENTATIONS = False  # Use classical PyTorch pipeline
```

**Key Features:**
- Dermatology-specific augmentations (skin texture, lighting variations)
- Probability-controlled application (p=0.3-0.5 for realistic variations)
- ImageNet normalization for transfer learning compatibility
- Validation/test transforms (normalization only, no augmenation)

### 3. Class Imbalance Solutions

**Multi-Strategy Approach**

- 1. Weighted Random Sampling (dataset.py)
```
sampler = create_weighted_sampler(train_df, logger)
# Melanoma samples ~50x more likely to be selected
# Ensures balanced batches during training
```

- 2. Focal Loss (models.py)
```
criterion = FocalLoss(alpha=0.75, gamma=2.0)
# Down-weights easy examples, focuses on hard cases
# Alpha balances positive/negative classes
# Gamma emphasizes misclassified examples
```

- 3. Dynamic Class Weighting (cross_validation.py)
```
pos_weight = (benign_count / melanoma_count) * 2.0
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Fold-specific weighting for cross-validation
```

- 4. Threshold Optimization
```
optimal_threshold = find_optimal_threshold(y_true, y_probs)
# Finds threshold maximizing F1-score or sensitivity
# Accounts for clinical cost of false negatives
```

### 4. Robust Training Strategies

**Cross-Validation Pipeline (cross_validation.py)**
```
USE_CROSS_VALIDATION = True  # config.py
N_FOLDS = 5

cv_stats, cv_models = run_cross_validation(
    train_val_df, train_transform, val_transform, ...
)
```

**Key Features:**
- Stratified K-Fold: Maintains class balance across folds
- Independent Models: Fresh initilization per fold (no infroamtion leakage)
- Fold-Specific Optimization: Dynamic class weighting per fold
- Statistical Analysis: Mean ± std across folds for all metrics
- Ensemble Evaluation: Combined predictions from all folds on test set

**Outputs per Fold:**
results/cross_validation/fold_1/
├── best_model.pth
├── roc_val.png
├── pr_curve_val.png
└── confusion_matrix_val.png

**Comprehensive Statistics:**
- AUC, Accuracy, Precision, Recall, F1-Score
- Mean, std, min, max, across all folds
- Individual fold performance breakdown
- Training time analysis

**Single Split Training (main.py)**
```
USE_CROSS_VALIDATION = False  # Faster for experimentation

final_results = run_single_split_training(
    model, train_loader, val_loader, test_loader, ...
)
```

**Key Features:**
- Early stopping (patience=15 epochs)
- Model Checkpointing (Based on the best validation AUC)
- Learning rate scheduling (ReduceLROnPlateau)
- Tensorboard logging for real-time monitoring
- Comprehensive validation and test evaluation

### 5. Advanced Evaluation

**Test-Time Augmentation (evaluation.py)**
```
TTA_N = 4  # Number of augmented versions

test_probs, test_true = predict_proba(
    model, test_loader, device, 
    use_tta=True, tta_n=TTA_N, tta_transform=tta_transform
)
# Averages predictions across multiple augmented versions
# Improves robustness and confidence calibration
```

**Ensemble Predictions (cross_validation.py)**
```
ensemble_probs = np.mean([
    model.predict(test_data) for model in cv_models
], axis=0)
# Combines predictions from all CV folds
# Reduces variance, improves generalization
```

**Comprehensive Metrics (evaluation.py)**
```
metrics = compute_classification_metrics(y_true, y_probs, threshold)
# Returns: ROC-AUC, Accuracy, Precision, Recall, F1-Score
```

**Visualizations:**
- ROC Curve: TPR vs FPR, AUC score
- Precision-Recall Curve: Critical for imbalanced data
- Confusion Matrix: With counts and percentages
- All plots saved as high-resolution PNG files

### 6. Mixed Precision Training
```
USE_AMP = torch.cuda.is_available()  # config.py

with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- Significant speed improvement on modern GPUs (for example NVIDIA RTX series of graphics cards)
- Memory reduction (enables larger batch sizes)
- Maintained Accuracy through gradient scaling
- Automatic fallback to FP32 on CPU if no graphics device is available

### 7. Production-Ready Infrastructure

**Comprehensive Logging (utils.py)**
```
logger = setup_logging(results_dir)
# Creates timestamped log files with detailed execution trace
# Logs system stats (CPU, RAM, GPU) for resource monitoring
```

**Configuration Management (config.py)**
```
# All hyperparameters in one place
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
TTA_N = 4
USE_CROSS_VALIDATION = False
N_FOLDS = 5
RUN_EDA = True
```

**Checkpoint Management**
```
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_auc": best_auc,
    "train_loss": train_loss,
    "val_auc": val_auc,
    "val_detailed_metrics": metrics
}
torch.save(checkpoint, checkpoint_path)
```

**Key Features:**
- Training resumption after interruption
- Best model preservation
- Complete state recovery (optimizer, scheduler, metrics)

## Project Structure
BinaryClassificationSystemFramework/
├── config.py                          # Configuration management
├── main.py                           # Main training pipeline
├── models.py                         # Model architectures and losses
├── training.py                       # Training loop implementation
├── evaluation.py                     # Evaluation metrics and plots
├── dataset.py                        # Data pipeline and augmentation
├── cross_validation.py               # K-fold CV implementation
├── data_validation.py                # Automated quality assurance
├── exploratory_data_analysis.py      # Statistical analysis and EDA
├── pearson_correlation_analysis.py   # Feature correlation analysis
├── utils.py                          # Logging and system utilities
├── run_eda.py                        # Standalone EDA script
├── EDA_INTEGRATION_GUIDE.md          # EDA usage documentation
│
├── data/
│   ├── ISIC_2020_Training_GroundTruth_v2.csv
│   └── jpeg/train/                   # Image files
│
└── results/
    ├── checkpoints/                  # Model checkpoints
    ├── metrics/                      # Evaluation plots
    ├── cross_validation/             # CV fold results
    ├── eda_analysis/                 # EDA outputs
    ├── logs/                         # TensorBoard logs
    ├── data_validation_report.json
    ├── validation_summary.txt
    └── metrics_summary.json

## Quick Start
```
git clone <repository-url>
cd BinaryClassificationSystemFramework

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Basic Usage

1. Run Complete Pipeline (Single Split)
```
python main.py
```
2. Run with Cross-Validation
```
# Edit config.py
USE_CROSS_VALIDATION = True
N_FOLDS = 5
```
```
python main.py
```

3. Run Standalone EDA
```
# Default paths
python run_eda.py
```

4. Disable EDA for Faster Iteration
```
# Edit config.py
RUN_EDA = False
```

**Expected Outputs**
After running `python main.py`, you will find:
results/
├── checkpoints/
│   ├── best_model.pth               # Best validation model
│   └── checkpoint_epoch_*.pth       # Epoch checkpoints
│
├── metrics/
│   ├── roc_val.png                  # Validation ROC curve
│   ├── roc_test.png                 # Test ROC curve
│   ├── pr_curve_val.png             # Validation PR curve
│   ├── pr_curve_test.png            # Test PR curve
│   ├── confusion_matrix.png         # Confusion matrix
│   └── confusion_matrix_test.png
│
├── eda_analysis/
│   ├── eda_report.json              # Complete analysis
│   ├── eda_summary.txt              # Executive summary
│   └── eda_visualizations/          # Generated plots
│
├── data_validation_report.json      # Validation details
├── validation_summary.txt           # Quality assurance
├── metrics_summary.json             # Final results
├── logs/                            # TensorBoard logs
└── training_YYYYMMDD_HHMMSS.log     # Execution log

## Adapting to a specific Use Case

### 1. Change Dataset Source

**Modify config.py:**
```
CSV_PATH = "data/your_dataset.csv"
DATA_DIR = "data/your_images/"
```

**CSV Format Required:**
```
image_name,target
sample_001,0
sample_002,1
...
```

- image_name: Filename without extension
- target: 0 (negative) or 1 (positive)

### 2. Adjust Class Imbalance Strategy

**For Different Imbalance Ratios:**
```
# In cross_validation.py or main.py
pos_weight = (negative_count / positive_count) * multiplier
# Adjust multiplier based on your clinical priorities:
# - 1.0: Balanced weighting
# - 2.0: Moderate emphasis on positives (current)
# - 5.0: Strong emphasis on positives (rare diseases)
```

**Focal Loss Tuning:**
```
# In models.py
criterion = FocalLoss(
    alpha=0.75,  # 0.5-0.9 for rare positive class
    gamma=2.0    # 1.0-5.0 (higher = more focus on hard examples)
)
```

### 3. Modify Image Size and Preprocessing

**For Different Image Resolutions:**
```
# In dataset.py - create_albumentations_transforms()
A.Resize(height=384, width=384),  # Change from 224x224

# Update normalization for your dataset
A.Normalize(
    mean=[0.485, 0.456, 0.406],  # Your dataset mean
    std=[0.229, 0.224, 0.225],   # Your dataset std
)
```

### 4. Add Custom Data Validation Rules

**Extend data_validation.py:**
```
def create_expectation_suite(self, df):
    validator = super().create_expectation_suite(df)
    
    # Add your custom expectations
    validator.expect_column_values_to_be_in_set(
        "diagnosis_category",
        value_set=["benign", "malignant", "premalignant"]
    )
    
    validator.expect_column_values_to_be_between(
        "patient_age",
        min_value=0,
        max_value=120
    )
    
    return validator
```

### 5. Customize EDA for Your Features

**Modify exploratory_data_analysis.py:**
```
def analyze_categorical_features(self, df):
    # Add your specific categorical features
    categorical_cols = [
        'patient_gender',
        'lesion_location',
        'your_custom_feature'
    ]
    # Analysis automatically adapts
```

Adapting to Non-Medical Binary Classification

**Example: Fraud Detection**
```
# 1. Update config.py
CSV_PATH = "data/transactions.csv"
DATA_DIR = None  # No images

# 2. Replace MelanomaDataset with custom dataset
class TransactionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.features = dataframe.drop('is_fraud', axis=1).values
        self.labels = dataframe['is_fraud'].values
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

# 3. Use tabular model instead of EfficientNet
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)
```

**Example: Spam Detection**
```
# Similar approach but with text preprocessing
class SpamDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['email_text'].values
        self.labels = dataframe['is_spam'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
```

### 6. Hyperparameter Tuning

**Key Parameters to Tune:**
```
# config.py - Recommended Ranges

# Learning Rate (most important)
LEARNING_RATE = 3e-4  # Try: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

# Batch Size (memory dependent)
BATCH_SIZE = 32  # Try: [16, 32, 64] based on GPU memory

# Weight Decay (regularization)
WEIGHT_DECAY = 1e-4  # Try: [0, 1e-5, 1e-4, 1e-3]

# Focal Loss Gamma (imbalance handling)
gamma = 2.0  # Try: [1.0, 2.0, 3.0, 5.0]

# TTA Iterations (inference time vs accuracy)
TTA_N = 4  # Try: [0, 2, 4, 8, 16]
```

**Systematic Tuning Approach:**
```
# 1. Grid search script
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        # Update config
        LEARNING_RATE = lr
        BATCH_SIZE = bs
        
        # Run training
        results = main()
        
        # Log results
        save_hyperparameter_results(lr, bs, results)

# 2. Analyze results
best_config = find_best_configuration(all_results)
```

### 7. Fairness Assessment

**Fairness Evaluation (fairness_assessment.py)**
```
fairness_results = evaluate_fairness(
    y_true, y_pred, y_probs, sensitive_features_df, results_dir, logger
)
```

**Key Features:**
- Demographic Parity Difference: Measures prediction rate equality across groups
- Equalized Odds Difference: Measures error rate equality (TPR, FPR) across groups
- Per-group metrics: Accuracy, Selection Rate, TPR, FPR, FNR for each demographic group
- Sensitive Features: Age groups, Sex, Anatomical site

**Outputs:**
- fairness_report.txt: Text report of fairness metrics
- fairness_metrics_*.png: Visualizations by group
- confusion_matrices_*.png: Confusion matrices per group

**Interpretation:**
- Values < 0.1 indicate good fairness
- Values > 0.1 suggest potential bias

---

### 8. Adversarial Robustness Testing

**Adversarial Testing (adversarial_testing.py)**
```
adversarial_results = evaluate_adversarial_robustness(
    model, test_loader, device, results_dir, logger, num_samples=500
)
```

**Key Features:**
- Attack Methods: FGSM (eps=0.01, 0.03), PGD (eps=0.03, 10 iterations), DeepFool (10 iterations)
- Metrics: Adversarial Accuracy, Attack Success Rate, L2/L∞ Perturbation, Robustness Score
- Visualizations: adversarial_examples_*.png (attack examples)
- Comprehensive text report: adversarial_report.txt

**Interpretation:**
- Robustness Score > 0.8: STRONG
- 0.6-0.8: MODERATE
- < 0.6: WEAK (consider adversarial training)


## Educational Use
This framework is ideal for:

1. Academic Research
    - Demonstrates publication-quality methodology
    - Provides complete reproducibility pipeline
    - Includes statistical rigor (cross-validation, significance testing)
2. Teaching Medical AI
    - Shows proper train/val/test splitting
    - Demonstrates class imbalance handling
    - Illustrates data validation importance
    - Provides clinical context for decision
3. Industry Prototyping
    - Production-ready code structure
    - Comprehensive logging for auditing
    - Automated quality assurance
    - Easy adaptation to new datasets

## Best Practices Implemented

1. Data Science
    - Stratified Splitting (maintans class balance)
    - Independent test set (never used during development)
    - Cross-validation (robust performance estimation)
    - Threshold optimization (not fixed at 0.5)
    - Multiple evaluation metrics (not just accurary)
2. Software Engineering
    - Modular design (separation of concerns)
    - Configuration Management (single source of truth)
    - Comprehensive logging (debugging and auditing)
    - Error handling (graceful failures)
    - Code documentation (docstring, comments)
3. Medical AI
    - Data quality assurance (automated validation)
    - Explainable decisions (statistical testing, visualizations)
    - Clinical context (sensitivity prioritization)
    - Uncertainty quantification (ensemble predictions, TTA)
    - Reproducibility (random seeds, version control)