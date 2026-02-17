"""
Calibration and Reliability Checking Module

This module implements model calibration and reliability assessment for medical AI,
following the BPMN workflow for "Calibrating and checking reliability".

Key features:
- Calibration assessment (ECE, MCE, Brier Score)
- Temperature scaling for probability calibration
- Reliability assessment with confidence thresholds
- Selective prediction analysis for human-in-the-loop deployment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, accuracy_score
from tqdm import tqdm
import os
import json


class TemperatureScalingModel(nn.Module):
    """
    Wrapper model that applies temperature scaling to logits.
    
    Temperature scaling is a post-hoc calibration method that learns a single
    parameter to scale model logits before converting to probabilities.
    """
    
    def __init__(self, model, temperature=1.0):
        """
        Initialize temperature scaling wrapper.
        
        Args:
            model: Base PyTorch model
            temperature: Initial temperature value (default=1.0, no scaling)
        """
        super(TemperatureScalingModel, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x):
        """
        Forward pass with temperature scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Temperature-scaled logits
        """
        logits = self.model(x)
        return logits / self.temperature
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.temperature.item()


def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Evaluate model calibration using reliability diagram and calibration metrics.
    
    Implements Algorithm 23: Calibration Evaluation from documentation.
    
    Args:
        y_true: True labels (numpy array)
        y_prob: Predicted probabilities (numpy array)
        n_bins: Number of bins for calibration curve (default=10)
        
    Returns:
        dict: Calibration metrics (ECE, MCE, Brier Score) and calibration data
    """
    # Bin predictions into probability ranges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1  # 0-indexed
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases
    
    calibration_data = []
    
    # Calculate mean predicted vs actual for each bin
    for bin_id in range(n_bins):
        bin_mask = (bin_indices == bin_id)
        
        if np.sum(bin_mask) > 0:
            mean_predicted = np.mean(y_prob[bin_mask])
            actual_positive_rate = np.mean(y_true[bin_mask])
            bin_size = np.sum(bin_mask)
            
            calibration_data.append({
                'bin': bin_id,
                'mean_predicted': float(mean_predicted),
                'actual_rate': float(actual_positive_rate),
                'size': int(bin_size)
            })
    
    # Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(y_prob)
    
    for bin_data in calibration_data:
        ece += (bin_data['size'] / total_samples) * abs(
            bin_data['actual_rate'] - bin_data['mean_predicted']
        )
    
    # Maximum Calibration Error (MCE)
    if calibration_data:
        mce = max(abs(bin_data['actual_rate'] - bin_data['mean_predicted']) 
                  for bin_data in calibration_data)
    else:
        mce = 0.0
    
    # Brier Score (proper scoring rule)
    brier_score = float(brier_score_loss(y_true, y_prob))
    
    calibration_metrics = {
        'ECE': float(ece),
        'MCE': float(mce),
        'Brier_score': brier_score
    }
    
    return calibration_metrics, calibration_data


def calibrate_with_temperature_scaling(model, val_loader, device, max_iterations=100, 
                                       learning_rate=0.01, logger=None):
    """
    Calibrate model using temperature scaling on validation set.
    
    Implements Algorithm 24: Platt Scaling Calibration from documentation.
    
    Args:
        model: Trained PyTorch model
        val_loader: Validation DataLoader
        device: Device to run on (CPU or GPU)
        max_iterations: Maximum optimization iterations (default=100)
        learning_rate: Learning rate for temperature optimization (default=0.01)
        logger: Optional logger for status messages
        
    Returns:
        TemperatureScalingModel: Calibrated model wrapper
    """
    if logger:
        logger.info("Starting temperature scaling calibration...")
    
    model.eval()
    
    # Collect model logits on validation set
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Collecting validation logits", leave=False):
            images = images.to(device, non_blocking=True)
            
            # Get logits (before sigmoid)
            outputs = model(images)
            logits_list.append(outputs.cpu())
            labels_list.append(labels)
    
    # Stack all logits and labels
    logits = torch.cat(logits_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device).float()
    
    # Create temperature scaling model
    calibrated_model = TemperatureScalingModel(model, temperature=1.0)
    calibrated_model = calibrated_model.to(device)
    
    # Optimize temperature parameter
    optimizer = optim.LBFGS([calibrated_model.temperature], lr=learning_rate, max_iter=max_iterations)
    
    criterion = nn.BCEWithLogitsLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = logits / calibrated_model.temperature
        loss = criterion(scaled_logits.squeeze(), labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    optimal_temperature = calibrated_model.get_temperature()
    
    if logger:
        logger.info(f"Optimal temperature: {optimal_temperature:.4f}")
    
    return calibrated_model


def assess_reliability(model, test_loader, device, confidence_thresholds=None, 
                      rejection_rates=None, logger=None):
    """
    Assess model reliability and confidence-accuracy relationship.
    
    Implements Algorithm 25: Clinical Reliability Assessment from documentation.
    
    Args:
        model: Calibrated model
        test_loader: Test DataLoader
        device: Device to run on
        confidence_thresholds: List of confidence thresholds to evaluate (default=[0.6, 0.7, 0.8, 0.9])
        rejection_rates: List of rejection rates for selective prediction (default=[0.1, 0.2, 0.3])
        logger: Optional logger
        
    Returns:
        dict: Reliability report with confidence-based metrics
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.6, 0.7, 0.8, 0.9]
    
    if rejection_rates is None:
        rejection_rates = [0.1, 0.2, 0.3]
    
    if logger:
        logger.info("Starting reliability assessment...")
    
    model.eval()
    
    # Collect all predictions and labels
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Reliability assessment", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(images))
            
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Get maximum probability (confidence) for each prediction
    max_probabilities = np.maximum(all_probs, 1 - all_probs)
    
    reliability_report = {
        'confidence_thresholds': {},
        'selective_prediction': {}
    }
    
    # Evaluate at different confidence levels
    for threshold in confidence_thresholds:
        high_confidence_mask = max_probabilities > threshold
        
        if np.sum(high_confidence_mask) > 0:
            confident_accuracy = accuracy_score(
                all_labels[high_confidence_mask],
                all_preds[high_confidence_mask]
            )
            coverage = np.sum(high_confidence_mask) / len(all_labels)
            
            reliability_report['confidence_thresholds'][threshold] = {
                'accuracy': float(confident_accuracy),
                'coverage': float(coverage),
                'samples': int(np.sum(high_confidence_mask))
            }
            
            if logger:
                logger.info(f"Confidence >{threshold:.2f}: Accuracy={confident_accuracy:.4f}, "
                          f"Coverage={coverage:.2%}, Samples={np.sum(high_confidence_mask)}")
    
    # Selective prediction analysis
    for rejection_rate in rejection_rates:
        confidence_threshold = np.percentile(max_probabilities, rejection_rate * 100)
        accepted_mask = max_probabilities > confidence_threshold
        
        if np.sum(accepted_mask) > 0:
            accepted_accuracy = accuracy_score(
                all_labels[accepted_mask],
                all_preds[accepted_mask]
            )
            
            reliability_report['selective_prediction'][rejection_rate] = {
                'threshold': float(confidence_threshold),
                'accuracy_on_accepted': float(accepted_accuracy),
                'rejection_rate': rejection_rate,
                'accepted_samples': int(np.sum(accepted_mask))
            }
            
            if logger:
                logger.info(f"Rejecting {rejection_rate:.1%} lowest confidence: "
                          f"Accuracy={accepted_accuracy:.4f} on remaining samples")
    
    return reliability_report


def plot_reliability_diagram(y_true, y_prob, n_bins=10, output_path=None):
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        output_path: Path to save plot (if None, displays plot)
    """
    # Calculate calibration data
    _, calibration_data = evaluate_calibration(y_true, y_prob, n_bins=n_bins)
    
    if not calibration_data:
        print("No calibration data available for plotting")
        return
    
    # Extract data for plotting
    mean_predicted = [bin_data['mean_predicted'] for bin_data in calibration_data]
    actual_rate = [bin_data['actual_rate'] for bin_data in calibration_data]
    bin_sizes = [bin_data['size'] for bin_data in calibration_data]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax1.scatter(mean_predicted, actual_rate, s=np.array(bin_sizes) / 10, 
                alpha=0.6, c='blue', label='Binned predictions')
    ax1.plot(mean_predicted, actual_rate, 'b-', alpha=0.5)
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Actual Positive Rate', fontsize=12)
    ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_accuracy_curve(model, test_loader, device, output_path=None):
    """
    Plot accuracy as a function of confidence threshold.
    
    Args:
        model: Calibrated model
        test_loader: Test DataLoader
        device: Device to run on
        output_path: Path to save plot (if None, displays plot)
    """
    model.eval()
    
    # Collect predictions
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting predictions", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(images))
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Get confidence (max probability)
    max_probabilities = np.maximum(all_probs, 1 - all_probs)
    
    # Evaluate at multiple thresholds
    thresholds = np.linspace(0.5, 1.0, 50)
    accuracies = []
    coverages = []
    
    for threshold in thresholds:
        mask = max_probabilities > threshold
        if np.sum(mask) > 0:
            acc = accuracy_score(all_labels[mask], all_preds[mask])
            cov = np.sum(mask) / len(all_labels)
            accuracies.append(acc)
            coverages.append(cov)
        else:
            accuracies.append(np.nan)
            coverages.append(0)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(thresholds, accuracies, 'b-', linewidth=2, label='Accuracy')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverages, 'r--', linewidth=2, label='Coverage')
    ax2.set_ylabel('Coverage (Proportion of Samples)', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Accuracy vs Confidence Threshold\n(with Coverage)', 
              fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confidence-accuracy curve saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def run_full_calibration_pipeline(model, val_loader, test_loader, device, 
                                  output_dir, logger=None):
    """
    Run complete calibration and reliability assessment pipeline.
    
    This is the main function to use for integrating calibration into your workflow.
    
    Args:
        model: Trained PyTorch model
        val_loader: Validation DataLoader (for temperature scaling)
        test_loader: Test DataLoader (for final evaluation)
        device: Device to run on
        output_dir: Directory to save results and plots
        logger: Optional logger
        
    Returns:
        dict: Complete calibration and reliability results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("Starting Calibration and Reliability Assessment")
        logger.info("=" * 60)
    
    # Step 1: Evaluate pre-calibration performance
    if logger:
        logger.info("\n[Step 1/4] Evaluating pre-calibration performance...")
    
    model.eval()
    pre_calib_probs = []
    pre_calib_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Pre-calibration evaluation", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(images))
            pre_calib_probs.append(outputs.cpu().numpy())
            pre_calib_labels.append(labels.numpy())
    
    pre_calib_probs = np.concatenate(pre_calib_probs, axis=0).squeeze()
    pre_calib_labels = np.concatenate(pre_calib_labels, axis=0)
    
    pre_calib_metrics, pre_calib_data = evaluate_calibration(
        pre_calib_labels, pre_calib_probs, n_bins=10
    )
    
    if logger:
        logger.info("Pre-calibration metrics:")
        logger.info(f"  ECE: {pre_calib_metrics['ECE']:.4f}")
        logger.info(f"  MCE: {pre_calib_metrics['MCE']:.4f}")
        logger.info(f"  Brier Score: {pre_calib_metrics['Brier_score']:.4f}")
    
    # Plot pre-calibration reliability diagram
    plot_reliability_diagram(
        pre_calib_labels, pre_calib_probs, n_bins=10,
        output_path=os.path.join(output_dir, 'reliability_diagram_pre_calibration.png')
    )
    
    # Step 2: Apply temperature scaling
    if logger:
        logger.info("\n[Step 2/4] Applying temperature scaling...")
    
    calibrated_model = calibrate_with_temperature_scaling(
        model, val_loader, device, max_iterations=100, learning_rate=0.01, logger=logger
    )
    
    # Step 3: Evaluate post-calibration performance
    if logger:
        logger.info("\n[Step 3/4] Evaluating post-calibration performance...")
    
    calibrated_model.eval()
    post_calib_probs = []
    post_calib_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Post-calibration evaluation", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = torch.sigmoid(calibrated_model(images))
            post_calib_probs.append(outputs.cpu().numpy())
            post_calib_labels.append(labels.numpy())
    
    post_calib_probs = np.concatenate(post_calib_probs, axis=0).squeeze()
    post_calib_labels = np.concatenate(post_calib_labels, axis=0)
    
    post_calib_metrics, post_calib_data = evaluate_calibration(
        post_calib_labels, post_calib_probs, n_bins=10
    )
    
    if logger:
        logger.info("Post-calibration metrics:")
        logger.info(f"  ECE: {post_calib_metrics['ECE']:.4f} "
                   f"(improvement: {pre_calib_metrics['ECE'] - post_calib_metrics['ECE']:.4f})")
        logger.info(f"  MCE: {post_calib_metrics['MCE']:.4f} "
                   f"(improvement: {pre_calib_metrics['MCE'] - post_calib_metrics['MCE']:.4f})")
        logger.info(f"  Brier Score: {post_calib_metrics['Brier_score']:.4f} "
                   f"(improvement: {pre_calib_metrics['Brier_score'] - post_calib_metrics['Brier_score']:.4f})")
    
    # Plot post-calibration reliability diagram
    plot_reliability_diagram(
        post_calib_labels, post_calib_probs, n_bins=10,
        output_path=os.path.join(output_dir, 'reliability_diagram_post_calibration.png')
    )
    
    # Step 4: Reliability assessment
    if logger:
        logger.info("\n[Step 4/4] Performing reliability assessment...")
    
    reliability_report = assess_reliability(
        calibrated_model, test_loader, device,
        confidence_thresholds=[0.6, 0.7, 0.8, 0.9],
        rejection_rates=[0.1, 0.2, 0.3],
        logger=logger
    )
    
    # Plot confidence-accuracy curve
    plot_confidence_accuracy_curve(
        calibrated_model, test_loader, device,
        output_path=os.path.join(output_dir, 'confidence_accuracy_curve.png')
    )
    
    # Compile comprehensive results
    results = {
        'pre_calibration': {
            'metrics': pre_calib_metrics,
            'calibration_data': pre_calib_data
        },
        'post_calibration': {
            'metrics': post_calib_metrics,
            'calibration_data': post_calib_data,
            'temperature': calibrated_model.get_temperature()
        },
        'improvement': {
            'ECE_reduction': float(pre_calib_metrics['ECE'] - post_calib_metrics['ECE']),
            'MCE_reduction': float(pre_calib_metrics['MCE'] - post_calib_metrics['MCE']),
            'Brier_improvement': float(pre_calib_metrics['Brier_score'] - post_calib_metrics['Brier_score'])
        },
        'reliability': reliability_report
    }
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'calibration_report.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    if logger:
        logger.info(f"\nCalibration report saved to {os.path.join(output_dir, 'calibration_report.json')}")
        logger.info("=" * 60)
        logger.info("Calibration and Reliability Assessment Complete")
        logger.info("=" * 60)
    
    return results, calibrated_model
