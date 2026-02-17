"""
Comparative Study Framework: Control vs Enhanced V&V
====================================================
This module implements a rigorous comparative study design to evaluate the impact
of comprehensive V&V (Verification & Validation) methodologies on CNN model performance.

Study Design:
- Independent Variable: V&V methodology (Control vs Enhanced)
- Dependent Variables: Model performance metrics, robustness, fairness
- Control: Ensures only V&V methodology differs between groups

Author: Deividas Kalvelis
"""

import json
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from config import *
from utils import setup_logging, get_device
from models import create_melanoma_model, create_advanced_melanoma_model, FocalLoss
from training import train_one_epoch, evaluate_with_threshold_search
from evaluation import (
    predict_proba, compute_classification_metrics, find_optimal_threshold,
    plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
)
from dataset import MelanomaDataset, create_transforms, create_albumentations_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Import V&V components (only used in enhanced group)
from data_validation import DataValidationPipeline
from exploratory_data_analysis import MelanomaEDA
from fairness_assessment import evaluate_fairness
from adversarial_testing import evaluate_adversarial_robustness


class ComparativeStudyFramework:
    """
    Framework for conducting comparative V&V studies.
    
    This class orchestrates a controlled experiment comparing:
    - Control Group: Minimal V&V (basic train/val/test split, standard metrics)
    - Enhanced Group: Comprehensive V&V (data validation, EDA, fairness, adversarial testing)
    """
    
    def __init__(self, base_results_dir="comparative_study_results"):
        """
        Initialize the comparative study framework.
        
        Args:
            base_results_dir: Base directory for all study results
        """
        self.base_results_dir = base_results_dir
        self.control_dir = os.path.join(base_results_dir, "control_group")
        self.enhanced_dir = os.path.join(base_results_dir, "enhanced_group")
        self.comparison_dir = os.path.join(base_results_dir, "comparison")
        
        # Create directories
        for dir_path in [self.control_dir, self.enhanced_dir, self.comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)
            os.makedirs(os.path.join(dir_path, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(dir_path, "metrics"), exist_ok=True)
        
        self.device = get_device()
        
        # Initialize loggers for both groups
        self.control_logger = setup_logging(self.control_dir)
        self.enhanced_logger = setup_logging(self.enhanced_dir)
        self.comparison_logger = setup_logging(self.comparison_dir)
        
        self.comparison_logger.info("="*80)
        self.comparison_logger.info("COMPARATIVE STUDY: Control vs Enhanced V&V Framework")
        self.comparison_logger.info("="*80)
        
    def load_and_prepare_data(self):
        """
        Load and prepare data for both control and enhanced groups.
        
        CRITICAL: Both groups use IDENTICAL data splits to ensure fair comparison.
        The only difference is the V&V process applied.
        
        Returns:
            dict: Contains all data components for both groups
        """
        self.comparison_logger.info("\n" + "="*60)
        self.comparison_logger.info("PHASE 1: Data Loading and Preparation")
        self.comparison_logger.info("="*60)
        
        # Load raw data
        df = pd.read_csv(CSV_PATH)
        self.comparison_logger.info(f"Loaded dataset: {len(df)} samples")
        self.comparison_logger.info(f"  Benign: {len(df[df['target']==0])} ({len(df[df['target']==0])/len(df)*100:.2f}%)")
        self.comparison_logger.info(f"  Melanoma: {len(df[df['target']==1])} ({len(df[df['target']==1])/len(df)*100:.2f}%)")
        
        # Add filepath column (required by MelanomaDataset)
        df['filepath'] = df['image_name'].apply(lambda x: os.path.join(DATA_DIR, f"{x}.jpg"))
        self.comparison_logger.info(f"Added filepath column using DATA_DIR: {DATA_DIR}")
        
        # Create stratified splits (IDENTICAL for both groups)
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=TEST_SIZE, 
            stratify=df['target'],
            random_state=RANDOM_STATE
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=VAL_SIZE,
            stratify=train_val_df['target'],
            random_state=RANDOM_STATE
        )
        
        self.comparison_logger.info(f"\nData Splits (Identical for both groups):")
        self.comparison_logger.info(f"  Train: {len(train_df)} samples")
        self.comparison_logger.info(f"  Validation: {len(val_df)} samples")
        self.comparison_logger.info(f"  Test: {len(test_df)} samples")
        
        # Save split information
        split_info = {
            'train_indices': train_df.index.tolist(),
            'val_indices': val_df.index.tolist(),
            'test_indices': test_df.index.tolist(),
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE,
            'val_size': VAL_SIZE
        }
        
        with open(os.path.join(self.comparison_dir, 'data_splits.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return {
            'full_df': df,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'split_info': split_info
        }
    
    def run_control_group(self, data_dict):
        """
        Run CONTROL group with MINIMAL V&V.
        
        Control V&V Process:
        - Basic data loading (no validation)
        - Standard train/val/test split
        - Basic augmentation (geometric only)
        - Standard training loop
        - Basic metrics (accuracy, AUC, F1)
        - NO fairness assessment
        - NO adversarial testing
        - NO comprehensive data validation
        
        Args:
            data_dict: Dictionary containing data splits
            
        Returns:
            dict: Control group results
        """
        self.control_logger.info("\n" + "="*80)
        self.control_logger.info("CONTROL GROUP: Minimal V&V Framework")
        self.control_logger.info("="*80)
        self.control_logger.info("V&V Components: Basic training, standard metrics only")
        
        # Create minimal dataloaders (no weighted sampling, basic augmentation)
        self.control_logger.info("\nCreating minimal data pipeline...")
        
        # Get PyTorch transforms for control group
        train_transform, val_transform, _ = create_transforms()
        
        train_dataset = MelanomaDataset(
            data_dict['train_df'],
            transform=train_transform,  # Basic PyTorch transforms
            use_albumentations=False
        )
        
        val_dataset = MelanomaDataset(
            data_dict['val_df'],
            transform=val_transform,
            use_albumentations=False
        )
        
        test_dataset = MelanomaDataset(
            data_dict['test_df'],
            transform=val_transform,
            use_albumentations=False
        )
        
        # Basic DataLoaders (no weighted sampling)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,  # Simple random shuffle
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        
        # Create model
        model = create_melanoma_model(num_classes=1).to(self.device)
        
        # Basic loss function (no class weights)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        scaler = torch.amp.GradScaler() if USE_AMP else None
        
        # Training loop
        self.control_logger.info("\nStarting CONTROL training...")
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch, self.control_logger
            )
            
            val_auc, val_metrics = evaluate_with_threshold_search(
                model, val_loader, epoch, self.control_logger
            )
            
            scheduler.step(val_auc)
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                }, os.path.join(self.control_dir, 'checkpoints', 'best_model.pth'))
                self.control_logger.info(f"Saved best model (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                self.control_logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model for testing
        checkpoint = torch.load(os.path.join(self.control_dir, 'checkpoints', 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Basic evaluation (no TTA)
        self.control_logger.info("\n" + "="*60)
        self.control_logger.info("CONTROL GROUP: Test Evaluation (Basic Metrics Only)")
        self.control_logger.info("="*60)
        
        test_probs, test_targets = predict_proba(
            model, test_loader, self.device, use_tta=False
        )
        
        optimal_threshold = find_optimal_threshold(test_targets, test_probs, self.control_logger)
        test_metrics, test_preds = compute_classification_metrics(
            test_targets, test_probs, threshold=optimal_threshold
        )
        
        self.control_logger.info("\nTest Set Performance:")
        for metric, value in test_metrics.items():
            self.control_logger.info(f"  {metric}: {value:.4f}")
        
        # Save basic visualizations
        plot_roc_curve(test_targets, test_probs, 
                      os.path.join(self.control_dir, 'metrics', 'roc_curve.png'))
        plot_confusion_matrix(test_targets, test_preds,
                            os.path.join(self.control_dir, 'metrics', 'confusion_matrix.png'))
        
        # Save results
        control_results = {
            'group': 'control',
            'vv_components': ['basic_training', 'standard_metrics'],
            'test_metrics': test_metrics,
            'best_val_auc': float(best_val_auc),
            'training_epochs': epoch + 1
        }
        
        with open(os.path.join(self.control_dir, 'results.json'), 'w') as f:
            json.dump(control_results, f, indent=2)
        
        return control_results
    
    def run_enhanced_group(self, data_dict):
        """
        Run ENHANCED group with COMPREHENSIVE V&V.
        
        Enhanced V&V Process:
        - Comprehensive data validation (Great Expectations)
        - Exploratory Data Analysis (statistical testing)
        - Advanced augmentation (Albumentations)
        - Weighted sampling for class imbalance
        - Focal Loss
        - Test-Time Augmentation
        - Fairness assessment across demographics
        - Adversarial robustness testing
        - Comprehensive metrics and visualizations

        Args:
            data_dict: Dictionary containing data splits

        Returns:
            dict: Enhanced group results
        """
        self.enhanced_logger.info("\n" + "="*80)
        self.enhanced_logger.info("ENHANCED GROUP: Comprehensive V&V Framework")
        self.enhanced_logger.info("="*80)
        self.enhanced_logger.info("V&V Components: Full data validation, EDA, fairness, adversarial testing")

        # PHASE 1: Comprehensive Data Validation
        self.enhanced_logger.info("\n--- V&V Component 1: Data Validation ---")
        validator = DataValidationPipeline(self.enhanced_logger)
        validation_passed, validation_results = validator.run_comprehensive_validation(
            data_dict['full_df']
        )

        if not validation_passed:
            self.enhanced_logger.warning("Data validation found issues (proceeding with caution)")

        # PHASE 2: Exploratory Data Analysis
        self.enhanced_logger.info("\n--- V&V Component 2: Exploratory Data Analysis ---")
        eda_analyzer = MelanomaEDA(self.enhanced_logger)
        eda_results = eda_analyzer.run_comprehensive_eda(
            data_dict['full_df'],
            output_dir=os.path.join(self.enhanced_dir, 'eda_analysis')
        )

        # PHASE 3: Advanced Data Pipeline
        self.enhanced_logger.info("\n--- V&V Component 3: Advanced Data Pipeline ---")

        # Calculate class weights for weighted sampling
        train_targets = data_dict['train_df']['target'].values
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_targets]

        self.enhanced_logger.info(f"Class weights: Benign={class_weights[0]:.4f}, Melanoma={class_weights[1]:.4f}")

        # Get Albumentations transforms for enhanced group with 300x300 image size
        # EfficientNet-B3 uses 300x300 input (vs 224x224 for ResNet-50)
        train_transform, val_transform, tta_transform = create_albumentations_transforms(image_size=300)

        train_dataset = MelanomaDataset(
            data_dict['train_df'],
            transform=train_transform,  # Advanced augmentations
            use_albumentations=True
        )

        val_dataset = MelanomaDataset(
            data_dict['val_df'],
            transform=val_transform,
            use_albumentations=True
        )

        test_dataset = MelanomaDataset(
            data_dict['test_df'],
            transform=val_transform,
            use_albumentations=True
        )

        # Weighted sampler for class imbalance
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        # Create model - ADVANCED architecture with EfficientNet-B3
        # Features: multi-dropout, better backbone, swish activation
        model = create_advanced_melanoma_model(
            out_dim=2,
            pretrained=True,
            dropout_rate=0.5,
            num_dropouts=5
        ).to(self.device)
        
        self.enhanced_logger.info("\nModel Architecture: Advanced EfficientNet-B3")
        self.enhanced_logger.info("  - Backbone: EfficientNet-B3 (~12M parameters)")
        self.enhanced_logger.info("  - Multi-dropout TTA: 5 dropout layers")
        self.enhanced_logger.info("  - Activation: Swish")
        self.enhanced_logger.info("  - Input size: 300x300 (vs 224x224 for control)")

        # Focal Loss for class imbalance
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        scaler = torch.amp.GradScaler() if USE_AMP else None

        # Training loop
        self.enhanced_logger.info("\nStarting ENHANCED training...")
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch, self.enhanced_logger
            )

            val_auc, val_metrics = evaluate_with_threshold_search(
                model, val_loader, epoch, self.enhanced_logger
            )

            scheduler.step(val_auc)

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                }, os.path.join(self.enhanced_dir, 'checkpoints', 'best_model.pth'))
                self.enhanced_logger.info(f"Saved best model (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                self.enhanced_logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model for testing
        checkpoint = torch.load(os.path.join(self.enhanced_dir, 'checkpoints', 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])

        # PHASE 4: Enhanced Evaluation with TTA
        self.enhanced_logger.info("\n--- V&V Component 4: Enhanced Evaluation (with TTA) ---")

        # Use the TTA transform from Albumentations
        test_probs, test_targets = predict_proba(
            model, test_loader, self.device,
            use_tta=True, tta_n=TTA_N,
            tta_transform=tta_transform
        )

        optimal_threshold = find_optimal_threshold(test_targets, test_probs, self.enhanced_logger)
        test_metrics, test_preds = compute_classification_metrics(
            test_targets, test_probs, threshold=optimal_threshold
        )

        self.enhanced_logger.info("\nTest Set Performance (with TTA):")
        for metric, value in test_metrics.items():
            self.enhanced_logger.info(f"  {metric}: {value:.4f}")

        # Save visualizations
        plot_roc_curve(test_targets, test_probs,
                      os.path.join(self.enhanced_dir, 'metrics', 'roc_curve.png'))
        plot_precision_recall_curve(test_targets, test_probs,
                                   os.path.join(self.enhanced_dir, 'metrics', 'pr_curve.png'))
        plot_confusion_matrix(test_targets, test_preds,
                            os.path.join(self.enhanced_dir, 'metrics', 'confusion_matrix.png'))

        # PHASE 5: Fairness Assessment
        self.enhanced_logger.info("\n--- V&V Component 5: Fairness Assessment ---")

        # Prepare sensitive features
        test_df_with_features = data_dict['test_df'].copy()
        test_df_with_features['age_group'] = pd.cut(
            test_df_with_features['age_approx'],
            bins=[0, 30, 50, 70, 100],
            labels=['<30', '30-50', '50-70', '70+']
        )

        sensitive_features = test_df_with_features[['age_group', 'sex', 'anatom_site_general_challenge']].copy()
        # Ensure categorical columns include "Unknown"
        for col in sensitive_features.columns:
            if pd.api.types.is_categorical_dtype(sensitive_features[col]):
                sensitive_features[col] = sensitive_features[col].cat.add_categories(["Unknown"])

        # Now safe
        sensitive_features = sensitive_features.fillna("Unknown")

        fairness_results = evaluate_fairness(
            test_targets, test_preds, test_probs,
            sensitive_features, self.enhanced_dir, self.enhanced_logger
        )

        # PHASE 6: Adversarial Robustness Testing
        self.enhanced_logger.info("\n--- V&V Component 6: Adversarial Robustness Testing ---")

        adversarial_results = evaluate_adversarial_robustness(
            model, test_loader, self.device, self.enhanced_dir,
            self.enhanced_logger, num_samples=ADVERSARIAL_NUM_SAMPLES
        )

        # Save comprehensive results
        enhanced_results = {
            'group': 'enhanced',
            'vv_components': [
                'data_validation',
                'exploratory_data_analysis',
                'advanced_augmentation',
                'weighted_sampling',
                'focal_loss',
                'test_time_augmentation',
                'fairness_assessment',
                'adversarial_testing'
            ],
            'test_metrics': test_metrics,
            'best_val_auc': float(best_val_auc),
            'training_epochs': epoch + 1,
            'fairness_summary': {
                k: v for k, v in fairness_results.items()
                if k in ['age_group', 'sex']  # Summary only
            },
            'adversarial_summary': adversarial_results
        }

        with open(os.path.join(self.enhanced_dir, 'results.json'), 'w') as f:
            json.dump(enhanced_results, f, indent=2)

        return enhanced_results

    def compare_results(self, control_results, enhanced_results):
        """
        Generate comprehensive comparison between control and enhanced groups.

        Args:
            control_results: Results from control group
            enhanced_results: Results from enhanced group
        """
        self.comparison_logger.info("\n" + "="*80)
        self.comparison_logger.info("COMPARATIVE ANALYSIS: Control vs Enhanced V&V")
        self.comparison_logger.info("="*80)

        # Performance comparison
        self.comparison_logger.info("\n--- Performance Metrics Comparison ---")
        self.comparison_logger.info(f"{'Metric':<20} {'Control':<15} {'Enhanced':<15} {'Improvement':<15}")
        self.comparison_logger.info("-" * 70)

        for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']:
            control_val = control_results['test_metrics'][metric]
            enhanced_val = enhanced_results['test_metrics'][metric]
            improvement = (
                None if control_val == 0
                else ((enhanced_val - control_val) / control_val) * 100
            )

            self.comparison_logger.info(
                f"{metric:<20} {control_val:<15.4f} {enhanced_val:<15.4f} {improvement:>+14.2f}%"
            )

        # V&V Components comparison
        self.comparison_logger.info("\n--- V&V Components Comparison ---")
        self.comparison_logger.info(f"Control V&V Components: {len(control_results['vv_components'])}")
        for comp in control_results['vv_components']:
            self.comparison_logger.info(f"  - {comp}")

        self.comparison_logger.info(f"\nEnhanced V&V Components: {len(enhanced_results['vv_components'])}")
        for comp in enhanced_results['vv_components']:
            self.comparison_logger.info(f"  - {comp}")

        # Additional insights from enhanced group
        if 'adversarial_summary' in enhanced_results:
            self.comparison_logger.info("\n--- Additional Insights (Enhanced Only) ---")
            adv = enhanced_results['adversarial_summary']

            # Compute top-level robustness score as average over all attacks
            if 'attacks' in adv and adv['attacks']:
                avg_robustness = np.mean([r['robustness_score'] for r in adv['attacks'].values()])
                self.comparison_logger.info(f"Adversarial Robustness Score: {avg_robustness:.4f}")
            else:
                self.comparison_logger.info("Adversarial Robustness Score: N/A")
        # Save comparison
        comparison = {
            'study_date': datetime.now().isoformat(),
            'control': control_results,
            'enhanced': enhanced_results,
            'improvements': {
                metric: {
                    'absolute': enhanced_results['test_metrics'][metric] - control_results['test_metrics'][metric],
                    'relative_percent': ((enhanced_results['test_metrics'][metric] - control_results['test_metrics'][metric]) / control_results['test_metrics'][metric]) * 100
                }
                for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
            }
        }
        
        with open(os.path.join(self.comparison_dir, 'comparative_analysis.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Generate comparison visualizations
        self.comparison_logger.info("\n--- Generating Comparison Visualizations ---")
        self._create_comparison_plots(control_results, enhanced_results, comparison)
        
        self.comparison_logger.info("\nComparative analysis complete!")
        self.comparison_logger.info(f"Results saved to: {self.comparison_dir}")
        
        return comparison
    
    def _create_comparison_plots(self, control_results, enhanced_results, comparison):
        """
        Create comprehensive comparison visualizations.
        
        Args:
            control_results: Results from control group
            enhanced_results: Results from enhanced group
            comparison: Comparison dictionary with improvements
        """
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        # 1. Side-by-side bar chart comparison
        self._plot_metrics_comparison(control_results, enhanced_results)
        
        # 2. Improvement percentage chart
        self._plot_improvement_percentages(comparison)
        
        # 3. V&V components comparison
        self._plot_vv_components(control_results, enhanced_results)
        
        # 4. Comprehensive summary figure
        self._plot_comprehensive_summary(control_results, enhanced_results, comparison)
        
        self.comparison_logger.info("All comparison plots saved")
    
    def _plot_metrics_comparison(self, control_results, enhanced_results):
        """Create side-by-side bar chart comparing metrics."""
        metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_keys = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
        
        control_vals = [control_results['test_metrics'][k] for k in metric_keys]
        enhanced_vals = [enhanced_results['test_metrics'][k] for k in metric_keys]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, control_vals, width, label='Control (Minimal V&V)', 
                       color='#FF6B6B', alpha=0.8, edgecolor='darkred', linewidth=1.5)
        bars2 = ax.bar(x + width/2, enhanced_vals, width, label='Enhanced (Comprehensive V&V)', 
                       color='#4ECDC4', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Comparison: Control vs Enhanced V&V Framework', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.comparison_logger.info("Saved: metrics_comparison.png")
    
    def _plot_improvement_percentages(self, comparison):
        """Create horizontal bar chart showing improvement percentages."""
        metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_keys = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
        
        improvements = [comparison['improvements'][k]['relative_percent'] for k in metric_keys]
        colors = ['#4ECDC4' if imp > 0 else '#FF6B6B' for imp in improvements]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(metrics, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('Relative Improvement: Enhanced vs Control V&V Framework', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            x_pos = val + (0.3 if val > 0 else -0.3)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{val:+.2f}%', 
                   ha=ha, va='center', fontsize=10, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ECDC4', edgecolor='black', label='Positive Improvement'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Negative Change')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'improvement_percentages.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.comparison_logger.info("Saved: improvement_percentages.png")
    
    def _plot_vv_components(self, control_results, enhanced_results):
        """Create visualization comparing V&V components."""
        # Define all possible components
        all_components = [
            'Data Validation',
            'Exploratory Data Analysis',
            'Advanced Augmentation',
            'Weighted Sampling',
            'Focal Loss',
            'Test-Time Augmentation',
            'Fairness Assessment',
            'Adversarial Testing'
        ]
        
        # Map component keys to display names
        component_map = {
            'data_validation': 'Data Validation',
            'exploratory_data_analysis': 'Exploratory Data Analysis',
            'advanced_augmentation': 'Advanced Augmentation',
            'weighted_sampling': 'Weighted Sampling',
            'focal_loss': 'Focal Loss',
            'test_time_augmentation': 'Test-Time Augmentation',
            'fairness_assessment': 'Fairness Assessment',
            'adversarial_testing': 'Adversarial Testing'
        }
        
        enhanced_components = [component_map.get(c, c.replace('_', ' ').title()) 
                              for c in enhanced_results['vv_components']]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(all_components))
        
        # Create bars showing implementation status
        control_status = [0] * len(all_components)  # Control has none
        enhanced_status = [1 if comp in enhanced_components else 0 for comp in all_components]
        
        # Plot
        width = 0.35
        ax.barh(y_pos - width/2, control_status, width, label='Control Group', 
               color='#FF6B6B', alpha=0.6, edgecolor='darkred', linewidth=1.5)
        ax.barh(y_pos + width/2, enhanced_status, width, label='Enhanced Group', 
               color='#4ECDC4', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_components, fontsize=10)
        ax.set_xlabel('Implementation Status', fontsize=11, fontweight='bold')
        ax.set_title('V&V Components Comparison: Control vs Enhanced', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-0.2, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Implemented', 'Implemented'])
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add checkmarks and X marks
        for i, status in enumerate(enhanced_status):
            if status == 1:
                ax.text(1.05, i + width/2, '✓', fontsize=16, color='darkgreen', 
                       ha='left', va='center', fontweight='bold')
            else:
                ax.text(0.05, i + width/2, '✗', fontsize=16, color='darkred', 
                       ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'vv_components_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.comparison_logger.info("Saved: vv_components_comparison.png")
    
    def _plot_comprehensive_summary(self, control_results, enhanced_results, comparison):
        """Create a comprehensive 2x2 summary figure."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Metrics comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['AUC', 'Acc', 'Prec', 'Rec', 'F1']
        metric_keys = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
        control_vals = [control_results['test_metrics'][k] for k in metric_keys]
        enhanced_vals = [enhanced_results['test_metrics'][k] for k in metric_keys]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width/2, control_vals, width, label='Control', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, enhanced_vals, width, label='Enhanced', color='#4ECDC4', alpha=0.8)
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Performance Metrics', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # 2. Improvements (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        improvements = [comparison['improvements'][k]['relative_percent'] for k in metric_keys]
        colors = ['#4ECDC4' if imp > 0 else '#FF6B6B' for imp in improvements]
        ax2.barh(metrics, improvements, color=colors, alpha=0.8)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Improvement (%)', fontweight='bold')
        ax2.set_title('Relative Improvement', fontweight='bold', fontsize=12)
        ax2.grid(axis='x', alpha=0.3)
        
        for i, val in enumerate(improvements):
            ax2.text(val + (0.2 if val > 0 else -0.2), i, f'{val:+.1f}%', 
                    ha='left' if val > 0 else 'right', va='center', fontweight='bold')
        
        # 3. Training comparison (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        training_data = {
            'Validation AUC': [control_results['best_val_auc'], enhanced_results['best_val_auc']],
            'Training Epochs': [control_results['training_epochs'], enhanced_results['training_epochs']],
            'Test AUC': [control_results['test_metrics']['roc_auc'], enhanced_results['test_metrics']['roc_auc']]
        }
        
        x = np.arange(len(training_data))
        width = 0.35
        control_training = [training_data[k][0] for k in training_data.keys()]
        enhanced_training = [training_data[k][1] for k in training_data.keys()]
        
        # Normalize epochs to 0-1 scale for visualization
        control_training[1] = control_training[1] / MAX_EPOCHS  # Assuming MAX_EPOCHS=50
        enhanced_training[1] = enhanced_training[1] / MAX_EPOCHS
        
        ax3.bar(x - width/2, control_training, width, label='Control', color='#FF6B6B', alpha=0.8)
        ax3.bar(x + width/2, enhanced_training, width, label='Enhanced', color='#4ECDC4', alpha=0.8)
        ax3.set_ylabel('Normalized Score', fontweight='bold')
        ax3.set_title('Training Summary', fontweight='bold', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Val AUC', 'Epochs\n(norm)', 'Test AUC'], fontsize=9)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Summary statistics (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"""
COMPARATIVE STUDY SUMMARY
{'='*40}

Control Group (Minimal V&V):
  • Components: {len(control_results['vv_components'])}
  • Test AUC: {control_results['test_metrics']['roc_auc']:.4f}
  • Test F1: {control_results['test_metrics']['f1_score']:.4f}
  • Training Epochs: {control_results['training_epochs']}

Enhanced Group (Comprehensive V&V):
  • Components: {len(enhanced_results['vv_components'])}
  • Test AUC: {enhanced_results['test_metrics']['roc_auc']:.4f}
  • Test F1: {enhanced_results['test_metrics']['f1_score']:.4f}
  • Training Epochs: {enhanced_results['training_epochs']}

Overall Improvements:
  • AUC: {comparison['improvements']['roc_auc']['relative_percent']:+.2f}%
  • F1 Score: {comparison['improvements']['f1_score']['relative_percent']:+.2f}%
  • Precision: {comparison['improvements']['precision']['relative_percent']:+.2f}%
  • Recall: {comparison['improvements']['recall']['relative_percent']:+.2f}%
"""
        if 'adversarial_summary' in enhanced_results:
            adv = enhanced_results['adversarial_summary']
            if 'attacks' in adv and adv['attacks']:
                avg_robustness = np.mean([r['robustness_score'] for r in adv['attacks'].values()])
                self.comparison_logger.info(f"Adversarial Robustness Score: {avg_robustness:.4f}")
                summary_text += f"\nAdditional Insights (Enhanced Only):\n  • Robustness Score: {adv.get('robustness_score')}"
            else:
                self.comparison_logger.info("Adversarial Robustness Score: N/A")
                summary_text += f"\nAdditional Insights (Enhanced Only):\n  • Robustness Score: N/A"

        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle('Comparative Study: Control vs Enhanced V&V Framework', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.comparison_dir, 'comprehensive_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.comparison_logger.info("Saved: comprehensive_summary.png")
    
    def run_full_study(self):
        """
        Execute the complete comparative study.
        
        Returns:
            dict: Complete study results
        """
        self.comparison_logger.info("Starting Comparative Study...")
        self.comparison_logger.info(f"Start time: {datetime.now()}")
        
        # Phase 1: Load data
        data_dict = self.load_and_prepare_data()
        
        # Phase 2: Run control group
        self.comparison_logger.info("\n" + "="*80)
        self.comparison_logger.info("Running CONTROL Group...")
        self.comparison_logger.info("="*80)
        control_results = self.run_control_group(data_dict)
        
        # Phase 3: Run enhanced group
        self.comparison_logger.info("\n" + "="*80)
        self.comparison_logger.info("Running ENHANCED Group...")
        self.comparison_logger.info("="*80)
        enhanced_results = self.run_enhanced_group(data_dict)
        
        # Phase 4: Compare results
        comparison = self.compare_results(control_results, enhanced_results)
        
        self.comparison_logger.info(f"\nEnd time: {datetime.now()}")
        self.comparison_logger.info("=" * 80)
        self.comparison_logger.info("COMPARATIVE STUDY COMPLETE")
        self.comparison_logger.info("=" * 80)
        
        return comparison


def main():
    """Main entry point for comparative study."""
    study = ComparativeStudyFramework()
    results = study.run_full_study()
    print("\n✓ Comparative study completed successfully!")
    print(f"Results available in: {study.comparison_dir}")


if __name__ == "__main__":
    main()
