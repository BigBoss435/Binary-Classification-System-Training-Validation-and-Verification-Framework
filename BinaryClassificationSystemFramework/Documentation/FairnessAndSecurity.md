# Fairness Assessment and Adversarial Robustness Testing

## Overview

This implementation adds two critical evaluation components to your melanoma classification system:

1. **Fairness Assessment** using Fairlearn
2. **Adversarial Robustness Testing** using the Adversarial Robustness Toolbox (ART)

## Installation

The required packages have been installed:

```bash
pip install fairlearn adversarial-robustness-toolbox
```

Or use the requirements file:
```bash
pip install -r requirements_fairness_security.txt
```

## Configuration

In `config.py`, you can enable/disable these assessments:

```python
# Fairness and Security Assessment Configuration
RUN_FAIRNESS_ASSESSMENT = True  # Set to False to skip fairness evaluation
RUN_ADVERSARIAL_TESTING = True  # Set to False to skip adversarial robustness testing
ADVERSARIAL_NUM_SAMPLES = 500  # Number of samples for adversarial testing
```

## Features

### 1. Fairness Assessment (`fairness_assessment.py`)

Evaluates model performance across different demographic groups to identify potential biases:

**Metrics Computed:**
- **Demographic Parity Difference**: Measures whether the model's positive prediction rate is similar across groups
- **Equalized Odds Difference**: Measures whether error rates (TPR, FPR) are similar across groups
- **Per-group metrics**: Accuracy, Selection Rate, TPR, FPR, FNR for each demographic group

**Sensitive Features Analyzed:**
- Age groups: <30, 30-50, 50-70, 70+
- Sex: Male, Female
- Anatomical site: Different body locations

**Outputs:**
- `results/fairness_analysis/fairness_report.txt` - Comprehensive text report
- `results/fairness_analysis/fairness_metrics_*.png` - Visualization of metrics by group
- `results/fairness_analysis/confusion_matrices_*.png` - Confusion matrices per group

**Interpretation:**
- Values < 0.1 indicate good fairness
- Values > 0.1 suggest potential bias requiring investigation

### 2. Adversarial Robustness Testing (`adversarial_testing.py`)

Tests model resilience against adversarial attacks:

**Attack Methods:**
1. **FGSM (Fast Gradient Sign Method)**
   - `eps=0.01`: Small perturbations
   - `eps=0.03`: Moderate perturbations
   
2. **PGD (Projected Gradient Descent)**
   - Multi-step iterative attack
   - `eps=0.03`, 10 iterations
   
3. **DeepFool**
   - Finds minimal perturbations to fool the model
   - 10 iterations

**Metrics Computed:**
- **Adversarial Accuracy**: Model accuracy on adversarial examples
- **Attack Success Rate**: Percentage of successful adversarial attacks
- **L2 Perturbation**: Average L2 norm of perturbations
- **L∞ Perturbation**: Average infinity norm of perturbations
- **Robustness Score**: Overall model resilience (0-1 scale)

**Outputs:**
- `results/adversarial_analysis/adversarial_report.txt` - Comprehensive text report
- `results/adversarial_analysis/adversarial_examples_*.png` - Visual examples of attacks

**Interpretation:**
- Robustness Score > 0.8: STRONG robustness
- Robustness Score 0.6-0.8: MODERATE robustness
- Robustness Score < 0.6: WEAK robustness (consider adversarial training)

## Usage

### Basic Usage

Simply run your training script as usual:

```bash
python main.py
```

The fairness and adversarial testing will automatically run after training completes (if enabled in config).

### Customizing Sensitive Features

To customize which demographic features are analyzed, modify the `prepare_sensitive_features()` function in `main.py`:

```python
def prepare_sensitive_features(test_df):
    sensitive_features = pd.DataFrame()
    
    # Add your custom features
    if 'your_feature' in test_df.columns:
        sensitive_features['your_feature'] = test_df['your_feature']
    
    return sensitive_features
```

### Adjusting Adversarial Testing

Modify attack parameters in `adversarial_testing.py`:

```python
attacks = {
    'FGSM_eps0.01': FastGradientMethod(estimator=art_classifier, eps=0.01),
    'FGSM_eps0.05': FastGradientMethod(estimator=art_classifier, eps=0.05),  # More aggressive
    # Add more attacks...
}
```

## Expected Output Structure

```
results/
├── fairness_analysis/
│   ├── fairness_report.txt
│   ├── fairness_metrics_age_group.png
│   ├── fairness_metrics_sex.png
│   ├── confusion_matrices_age_group.png
│   └── confusion_matrices_sex.png
├── adversarial_analysis/
│   ├── adversarial_report.txt
│   ├── adversarial_examples_FGSM_eps0.01.png
│   ├── adversarial_examples_FGSM_eps0.03.png
│   ├── adversarial_examples_PGD_eps0.03.png
│   └── adversarial_examples_DeepFool.png
└── metrics_summary.json (includes fairness and adversarial results)
```

## Understanding Results

### Fairness Assessment

Check `fairness_report.txt` for:
- Group-wise performance metrics
- Fairness indicators (demographic parity, equalized odds)
- Interpretation and warnings

**Red flags:**
- Large differences in accuracy between groups (>10%)
- High demographic parity difference (>0.1)
- Disparate error rates (FPR, FNR) across groups

### Adversarial Robustness

Check `adversarial_report.txt` for:
- Clean model accuracy vs adversarial accuracy
- Attack success rates
- Perturbation magnitudes

**Red flags:**
- Adversarial accuracy << clean accuracy (>20% drop)
- High attack success rates (>50%)
- Low robustness scores (<0.6)

## Clinical Implications

### Fairness
- Ensures equitable performance across patient demographics
- Critical for regulatory approval (FDA, CE marking)
- Helps identify subpopulations requiring additional data or model refinement

### Adversarial Robustness
- Ensures model reliability in clinical deployment
- Protects against malicious attacks in production systems
- Validates model behavior under perturbations similar to image artifacts

## Troubleshooting

### Missing Demographic Features

If you see "No sensitive features available":
- Check that your CSV file includes demographic columns (age_approx, sex, anatom_site_general_challenge)
- Verify column names match those expected in `prepare_sensitive_features()`

### Adversarial Testing Failures

If adversarial testing fails:
- Reduce `ADVERSARIAL_NUM_SAMPLES` for faster testing (e.g., 100-200)
- Check GPU memory availability
- Ensure test_loader has sufficient samples

### Memory Issues

If you encounter OOM errors:
- Reduce `ADVERSARIAL_NUM_SAMPLES` in config.py
- Run fairness and adversarial testing separately by disabling one
- Use smaller batch sizes

## References

- **Fairlearn**: https://fairlearn.org/
- **ART**: https://adversarial-robustness-toolbox.readthedocs.io/
- **Demographic Parity**: https://fairlearn.org/v0.10/user_guide/fairness_in_machine_learning.html
- **Adversarial Examples**: https://arxiv.org/abs/1412.6572

## Citation

If you use these tools in your research, please cite:

```bibtex
@article{bird2020fairlearn,
  title={Fairlearn: A toolkit for assessing and improving fairness in AI},
  author={Bird, Sarah and others},
  journal={Microsoft Research},
  year={2020}
}

@article{nicolae2018adversarial,
  title={Adversarial Robustness Toolbox v1.0.0},
  author={Nicolae, Maria-Irina and others},
  journal={arXiv preprint arXiv:1807.01069},
  year={2018}
}
```
