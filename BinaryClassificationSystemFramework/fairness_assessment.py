"""
Fairness Assessment Module for Melanoma Classification
======================================================
This module implements comprehensive fairness evaluation using Fairlearn
to assess model performance across different patient demographic groups.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference
)
import os


def evaluate_fairness(y_true, y_pred, y_probs, sensitive_features_df, results_dir, logger):
    """
    Comprehensive fairness evaluation across demographic groups.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        sensitive_features_df: DataFrame with demographic features (age_group, sex, etc.)
        results_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        dict: Fairness metrics and statistics
    """
    
    fairness_dir = os.path.join(results_dir, "fairness_analysis")
    os.makedirs(fairness_dir, exist_ok=True)
    
    fairness_results = {}
    
    # Analyze each sensitive attribute
    for feature in sensitive_features_df.columns:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fairness Analysis for: {feature}")
        logger.info(f"{'='*60}")
        
        sensitive_feature = sensitive_features_df[feature]
        
        # Create MetricFrame for comprehensive analysis
        metric_frame = MetricFrame(
            metrics={
                'accuracy': lambda y_t, y_p: np.mean(y_t == y_p),
                'selection_rate': selection_rate,
                'tpr': true_positive_rate,
                'fpr': false_positive_rate,
                'fnr': false_negative_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_feature
        )
        
        # Log detailed metrics per group
        logger.info(f"\nMetrics by {feature}:")
        logger.info("\n" + str(metric_frame.by_group))
        
        # Calculate fairness metrics
        dp_diff = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_feature
        )
        eo_diff = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_feature
        )
        
        logger.info(f"\nFairness Metrics for {feature}:")
        logger.info(f"  Demographic Parity Difference: {dp_diff:.4f}")
        logger.info(f"  Equalized Odds Difference: {eo_diff:.4f}")
        
        # Store results
        fairness_results[feature] = {
            'metrics_by_group': metric_frame.by_group.to_dict(),
            'overall_metrics': metric_frame.overall.to_dict(),
            'demographic_parity_difference': float(dp_diff),
            'equalized_odds_difference': float(eo_diff),
            'max_difference': metric_frame.difference().to_dict(),
            'ratio': metric_frame.ratio().to_dict()
        }
        
        # Visualizations
        _plot_fairness_metrics(metric_frame, feature, fairness_dir)
        _plot_confusion_matrices_by_group(y_true, y_pred, sensitive_feature, feature, fairness_dir)
    
    # Save fairness report
    _save_fairness_report(fairness_results, fairness_dir, logger)
    
    return fairness_results


def _plot_fairness_metrics(metric_frame, feature_name, output_dir):
    """Plot fairness metrics by group."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Fairness Metrics by {feature_name}', fontsize=16)
    
    metrics = ['accuracy', 'selection_rate', 'tpr', 'fpr']
    titles = ['Accuracy', 'Selection Rate', 'True Positive Rate', 'False Positive Rate']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        data = metric_frame.by_group[metric]
        
        bars = ax.bar(range(len(data)), data.values)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.axhline(y=data.mean(), color='r', linestyle='--', label='Mean')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Color bars based on deviation from mean
        mean_val = data.mean()
        for bar, val in zip(bars, data.values):
            if abs(val - mean_val) > 0.1:
                bar.set_color('orange')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fairness_metrics_{feature_name}.png'), dpi=300)
    plt.close()


def _plot_confusion_matrices_by_group(y_true, y_pred, sensitive_feature, feature_name, output_dir):
    """Plot confusion matrices for each demographic group."""
    
    unique_groups = sensitive_feature.unique()
    n_groups = len(unique_groups)
    
    fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
    if n_groups == 1:
        axes = [axes]
    
    fig.suptitle(f'Confusion Matrices by {feature_name}', fontsize=16)
    
    for idx, group in enumerate(unique_groups):
        mask = sensitive_feature == group
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{group} (n={mask.sum()})')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrices_{feature_name}.png'), dpi=300)
    plt.close()


def _save_fairness_report(fairness_results, output_dir, logger):
    """Save comprehensive fairness report."""
    
    report_path = os.path.join(output_dir, 'fairness_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FAIRNESS ASSESSMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        for feature, results in fairness_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Feature: {feature}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("Metrics by Group:\n")
            f.write("-" * 40 + "\n")
            for metric, values in results['metrics_by_group'].items():
                f.write(f"\n{metric}:\n")
                for group, value in values.items():
                    f.write(f"  {group}: {value:.4f}\n")
            
            f.write("\nFairness Indicators:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Demographic Parity Difference: {results['demographic_parity_difference']:.4f}\n")
            f.write(f"Equalized Odds Difference: {results['equalized_odds_difference']:.4f}\n")
            
            f.write("\nInterpretation:\n")
            f.write("-" * 40 + "\n")
            if abs(results['demographic_parity_difference']) < 0.1:
                f.write("Model shows good demographic parity (< 0.1 difference)\n")
            else:
                f.write("Model shows potential bias in selection rates\n")
            
            if abs(results['equalized_odds_difference']) < 0.1:
                f.write("Model shows good equalized odds (< 0.1 difference)\n")
            else:
                f.write("Model shows disparate error rates across groups\n")
    
    logger.info(f"Fairness report saved to: {report_path}")
