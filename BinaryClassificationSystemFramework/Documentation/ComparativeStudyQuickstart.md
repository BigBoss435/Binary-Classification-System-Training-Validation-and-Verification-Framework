# Quick Start Guide: Running the Comparative Study

## Overview
This guide walks you through running a comparative V&V study in 5 simple steps.

## Prerequisites

### 1. Install Required Packages
```bash
pip install fairlearn adversarial-robustness-toolbox great-expectations
```

### 2. Verify Your Dataset
Ensure you have:
- `data/ISIC_2020_Training_GroundTruth_v2.csv`
- `data/jpeg/train/` (directory with images)

## Running the Study

### Option 1: Run Full Automated Study (Recommended)

```bash
# Single command to run complete comparative study
python comparative_study.py
```

**What happens:**
1. Loads and prepares identical data splits for both groups
2. Trains control group with minimal V&V
3. Trains enhanced group with comprehensive V&V
4. Generates comparative analysis
5. Saves all results to `comparative_study_results/`

**Expected runtime:**
- Control group: ~2-4 hours
- Enhanced group: ~4-6 hours
- Total: ~6-10 hours (depending on hardware)

### Option 2: Run Groups Separately (For Testing)

```python
from comparative_study import ComparativeStudyFramework

# Initialize framework
study = ComparativeStudyFramework()

# Prepare data (shared by both groups)
data_dict = study.load_and_prepare_data()

# Run just control group
control_results = study.run_control_group(data_dict)

# Run just enhanced group
enhanced_results = study.run_enhanced_group(data_dict)

# Compare results
comparison = study.compare_results(control_results, enhanced_results)
```

## Understanding the Results

### Directory Structure
```
comparative_study_results/
├── control_group/                  # Control group results
│   ├── checkpoints/
│   │   └── best_model.pth         # Best control model
│   ├── metrics/
│   │   ├── roc_curve.png          # ROC curve
│   │   └── confusion_matrix.png   # Confusion matrix
│   ├── results.json               # Control metrics
│   └── training.log               # Training logs
│
├── enhanced_group/                 # Enhanced group results
│   ├── checkpoints/
│   │   └── best_model.pth         # Best enhanced model
│   ├── metrics/
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   └── confusion_matrix.png
│   ├── eda_analysis/              # EDA results
│   │   ├── eda_report.json
│   │   └── eda_visualizations/
│   ├── fairness_analysis/         # Fairness results
│   │   ├── fairness_report.txt
│   │   └── fairness_metrics_*.png
│   ├── adversarial_analysis/      # Adversarial testing
│   │   ├── adversarial_report.txt
│   │   └── adversarial_examples_*.png
│   ├── data_validation_report.json
│   ├── results.json
│   └── training.log
│
└── comparison/                     # Comparative analysis
    ├── comparative_analysis.json   # Main comparison results
    ├── data_splits.json            # Confirms identical splits
    └── comparison.log              # Comparison logs
```

### Key Files to Check

#### 1. `comparison/comparative_analysis.json`
Main results file with performance comparison:
```json
{
  "improvements": {
    "roc_auc": {
      "absolute": 0.0234,
      "relative_percent": 2.56
    },
    "f1_score": {
      "absolute": 0.0512,
      "relative_percent": 5.87
    }
  }
}
```

#### 2. `control_group/results.json`
Control group performance:
```json
{
  "group": "control",
  "vv_components": ["basic_training", "standard_metrics"],
  "test_metrics": {
    "roc_auc": 0.9140,
    "accuracy": 0.8523,
    "f1_score": 0.8723
  }
}
```

#### 3. `enhanced_group/results.json`
Enhanced group performance + additional insights:
```json
{
  "group": "enhanced",
  "vv_components": [
    "data_validation",
    "exploratory_data_analysis",
    "advanced_augmentation",
    "weighted_sampling",
    "focal_loss",
    "test_time_augmentation",
    "fairness_assessment",
    "adversarial_testing"
  ],
  "test_metrics": {
    "roc_auc": 0.9374,
    "accuracy": 0.8812,
    "f1_score": 0.9235
  },
  "adversarial_summary": {
    "robustness_score": 0.7823,
    "interpretation": "MODERATE"
  }
}
```

## Analyzing Results

### Quick Analysis Script

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('comparative_study_results/comparison/comparative_analysis.json') as f:
    results = json.load(f)

# Extract metrics
metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
control_vals = [results['control']['test_metrics'][m] for m in metrics]
enhanced_vals = [results['enhanced']['test_metrics'][m] for m in metrics]

# Create comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, control_vals, width, label='Control', alpha=0.8)
bars2 = ax.bar(x + width/2, enhanced_vals, width, label='Enhanced', alpha=0.8)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Control vs Enhanced V&V Framework')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics], rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparative_study_results/comparison/performance_comparison.png', dpi=300)
plt.show()

# Print improvements
print("\nPerformance Improvements:")
print("=" * 50)
for metric in metrics:
    improvement = results['improvements'][metric]['relative_percent']
    print(f"{metric.upper():<15}: {improvement:>+7.2f}%")
```

### Interpretation Guide

**ROC-AUC Improvement:**
- < 1%: Minimal impact
- 1-3%: Small but meaningful
- 3-5%: Moderate impact ✓
- > 5%: Large impact

**F1 Score Improvement:**
- < 2%: Minimal
- 2-5%: Small to moderate
- 5-8%: Moderate impact ✓
- > 8%: Large impact

**Robustness Score (Enhanced Only):**
- > 0.8: Strong robustness ✓
- 0.6-0.8: Moderate robustness
- < 0.6: Weak robustness (needs improvement)

## Common Questions

### Q: How long does it take?
**A:** 6-10 hours total on a modern GPU (RTX 3080 or better). CPU-only will take significantly longer.

### Q: Can I run just one group to test?
**A:** Yes! Use Option 2 above to run groups separately. Good for debugging.

### Q: What if I get out-of-memory errors?
**A:** Reduce `BATCH_SIZE` in `config.py` (e.g., from 32 to 16):
```python
BATCH_SIZE = 16  # Reduced for memory constraints
```

### Q: Can I run multiple times for statistical confidence?
**A:** Yes! Recommended for publications:
```python
for run_id in range(5):
    study = ComparativeStudyFramework(
        base_results_dir=f"comparative_study_run_{run_id}"
    )
    study.run_full_study()
```

### Q: What if the enhanced group performs worse?
**A:** This is possible and valuable! It could indicate:
- Overfitting from aggressive augmentation
- Threshold optimization overfitting to validation set
- Need for hyperparameter tuning

Document this honestly—negative results are publishable!

### Q: How do I use this in my thesis/paper?
**A:** See the methodology document for paper structure. Key sections:
1. **Methods**: Describe the comparative design
2. **Results**: Present the comparison table
3. **Discussion**: Interpret the improvements (or lack thereof)
4. **Conclusion**: Recommendations based on findings

## Troubleshooting

### Issue: Data validation fails in enhanced group
**Solution:** Check your CSV file format and image integrity. You can temporarily disable validation:
```python
# In comparative_study.py, comment out validation:
# validation_passed, validation_results = validator.run_comprehensive_validation(...)
```

### Issue: Adversarial testing is too slow
**Solution:** Reduce the number of test samples in `config.py`:
```python
ADVERSARIAL_NUM_SAMPLES = 100  # Reduced from 500
```

### Issue: Fairness assessment fails
**Solution:** Ensure your dataset has the required demographic features:
- `age_approx`
- `sex`
- `anatom_site_general`

If missing, comment out fairness assessment in `comparative_study.py`.

## Next Steps

1. **Run the study** using the commands above
2. **Review results** in `comparative_study_results/`
3. **Analyze improvements** using the analysis script
4. **Document findings** for your thesis/paper
5. **Consider multiple runs** for statistical robustness

## Need Help?

Check these resources:
- `COMPARATIVE_STUDY_METHODOLOGY.md` - Detailed methodology
- `FAIRNESS_AND_SECURITY_README.md` - V&V component details
- `readme.md` - Overall system documentation

## Example Output

When the study completes, you'll see:

```
================================================================================
COMPARATIVE STUDY COMPLETE
================================================================================

--- Performance Metrics Comparison ---
Metric               Control         Enhanced        Improvement    
----------------------------------------------------------------------
roc_auc              0.9140          0.9374          +2.56%
accuracy             0.8523          0.8812          +3.39%
precision            0.8234          0.8756          +6.34%
recall               0.7892          0.8423          +6.73%
f1_score             0.8723          0.9235          +5.87%

✓ Comparative analysis complete!
Results saved to: comparative_study_results/comparison
```

## Conclusion

You're now ready to run your comparative V&V study! This design provides rigorous, scientifically sound evidence for the value (or limitations) of comprehensive V&V frameworks in medical AI development.

Good luck with your research! 🚀
