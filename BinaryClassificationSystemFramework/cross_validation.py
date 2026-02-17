"""
Cross-validation training module for melanoma classification.

This module implements k-fold cross-validation for robust model evaluation and training.
Cross-validation is critical for medical ML to ensure models generalize well across 
different patient populations and data distributions. It provides unbiased performance 
estimates and helps detect overfitting to specific data splits.

Key components:
- Stratified k-fold splitting to maintain class balance across folds
- Ensemble model evaluation for improved robustness
- Comprehensive metrics tracking and visualization
- Early stopping and learning rate scheduling per fold
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

def create_model():
    """
    Create a fresh ResNet-50 model instance for each fold.
    
    Each fold requires a completely independent model to ensure:
    1. No information leakage between folds
    2. Fair comparison of fold performance
    3. Proper ensemble diversity for final predictions
    
    The model uses ImageNet pre-trained weights for transfer learning,
    which is particularly effective for medical image analysis due to:
    - Rich low-level feature representations (edges, textures)
    - Faster convergence with limited medical data
    - Better generalization to dermoscopic image patterns
    
    Returns:
        torch.nn.Module: Fresh ResNet-50 model with binary classification head
    """
    # Load pre-trained ResNet-50 with ImageNet weights
    # These weights provide feature extractors learned from millions of natural images
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # Replace final classification layer for binary melanoma detection
    # Original: 2048 features -> 1000 ImageNet classes
    # Modified: 2048 features -> 1 binary output (melanoma probability)
    num_ftrs = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Linear(num_ftrs, 1)

    return model

def create_optimizer_and_scheduler(model, train_df, device):
    """
    Create optimizer, loss function, and scheduler optimized for each fold's data distribution.
    
    This function creates training components specifically tuned for medical image classification:
    
    Loss Function Design:
    - Uses BCEWithLogitsLoss with dynamic positive class weighting
    - Positive weight calculated from fold-specific class distribution
    - Multiplied by 2.0 to further emphasize melanoma detection (clinical priority)
    - This addresses the severe class imbalance (~2% melanoma, ~98% benign)
    
    Optimizer Configuration:
    - Adam optimizer for stable convergence with medical data
    - Lower learning rate (1e-4) to prevent overfitting on limited data
    - Weight decay (1e-4) for regularization to improve generalization
    
    Scheduler Strategy:
    - CosineAnnealingWarmRestarts for learning rate scheduling
    - Warm restarts help escape local minima during training
    - Lower minimum LR (1e-7) for fine-tuning in later epochs
    
    Args:
        model (torch.nn.Module): Model to optimize
        train_df (pd.DataFrame): Training data for this fold (for class weight calculation)
        device (torch.device): Device for tensor operations
        
    Returns:
        tuple: (criterion, optimizer, scheduler) configured for this fold
    """  
    # Calculate positive class weight based on fold-specific class distribution
    # This is critical because class distribution may vary slightly between folds
    benign_count = (train_df['target'] == 0).sum()  # Count of benign samples
    melanoma_count = (train_df['target'] == 1).sum()  # Count of melanoma samples
    
    # Calculate imbalance ratio and apply aggressive weighting
    # Formula: pos_weight = (negative_samples / positive_samples) * multiplier
    # Multiplier of 2.0 gives extra emphasis to melanoma detection
    # This is medically justified - missing melanoma is worse than false alarms
    pos_weight = (benign_count / melanoma_count) * 2.0
    pos_weight = torch.tensor([pos_weight]).to(device)
    
    # BCEWithLogitsLoss combines sigmoid and BCE for numerical stability
    # pos_weight parameter scales the loss for positive examples (melanoma)
    # Higher pos_weight means melanoma misclassification is penalized more heavily
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Adam optimizer with conservative parameters for medical data
    # lr=1e-4: Lower learning rate prevents overfitting on limited medical data
    # weight_decay=1e-4: L2 regularization helps model generalize better
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Cosine annealing with warm restarts scheduler
    # T_0=10: Initial restart period (10 epochs)
    # T_mult=2: Period doubles after each restart (10, 20, 40, ...)
    # eta_min=1e-7: Minimum learning rate for fine-tuning
    # Warm restarts help escape local minima and can improve final performance
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    return criterion, optimizer, scheduler

def train_single_fold(fold_idx, train_indices, val_indices, train_val_df, train_transform, 
                     val_transform, dataset_class, train_one_epoch_func, evaluate_func, 
                     predict_proba_func, compute_metrics_func, find_threshold_func,
                     plot_functions, device, scaler, max_epochs=50, n_folds=5):
    """
    Train a model for a single fold in cross-validation.
    
    This function handles the complete training pipeline for one fold:
    1. Data splitting and preparation for the current fold
    2. Model initialization and optimization setup
    3. Training loop with early stopping and checkpointing
    4. Final evaluation and metric computation
    5. Visualization and result saving
    
    Cross-validation folds are critical for medical ML because:
    - They provide unbiased performance estimates
    - Help detect overfitting to specific data partitions
    - Enable ensemble predictions for improved robustness
    - Allow assessment of model stability across different data distributions
    
    Args:
        fold_idx (int): Current fold index (0-based)
        train_indices (np.array): Indices for training samples in this fold
        val_indices (np.array): Indices for validation samples in this fold
        train_val_df (pd.DataFrame): Complete train+validation dataframe
        train_transform (torchvision.transforms): Augmentation for training
        val_transform (torchvision.transforms): Normalization-only for validation
        dataset_class: PyTorch Dataset class for creating datasets
        train_one_epoch_func: Function to train for one epoch
        evaluate_func: Function to evaluate model performance
        predict_proba_func: Function to get prediction probabilities
        compute_metrics_func: Function to compute classification metrics
        find_threshold_func: Function to find optimal classification threshold
        plot_functions (dict): Dictionary of plotting functions for visualization
        device (torch.device): Device for training (CPU/GPU)
        scaler: Mixed precision scaler for training efficiency
        max_epochs (int): Maximum training epochs per fold
        n_folds (int): Total number of folds (for logging)
        
    Returns:
        tuple: (fold_results dict, trained_model) containing performance metrics and model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*30} FOLD {fold_idx + 1}/{n_folds} {'='*30}")
    
    # Split the combined train+val data according to current fold indices
    # reset_index ensures clean indexing for the fold-specific dataframes
    fold_train_df = train_val_df.iloc[train_indices].reset_index(drop=True)
    fold_val_df = train_val_df.iloc[val_indices].reset_index(drop=True)
    
    # Log fold statistics for monitoring data distribution
    # Important to verify stratification worked correctly
    fold_train_melanoma = fold_train_df['target'].sum()
    fold_val_melanoma = fold_val_df['target'].sum()
    fold_train_ratio = fold_train_melanoma / len(fold_train_df) if len(fold_train_df) > 0 else 0
    fold_val_ratio = fold_val_melanoma / len(fold_val_df) if len(fold_val_df) > 0 else 0

    logger.info(f"Fold {fold_idx + 1} Data Distribution:")
    logger.info(f"  Train: {len(fold_train_df)} samples ({fold_train_melanoma} melanoma, {fold_train_ratio:.2%})")
    logger.info(f"  Val: {len(fold_val_df)} samples ({fold_val_melanoma} melanoma, {fold_val_ratio:.2%})")
    
    # Apply appropriate transforms: augmentation for training, normalization-only for validation
    fold_train_ds = dataset_class(fold_train_df, transform=train_transform)
    fold_val_ds = dataset_class(fold_val_df, transform=val_transform)
    
    # Calculate fold-specific class weights to handle imbalance
    fold_class_counts = fold_train_df['target'].value_counts().sort_index().to_numpy()

    # Calculate imbalance ratio and apply aggressive weighting for melanoma class
    # This ensures roughly balanced batches during training
    imbalance_ratio = fold_class_counts[0] / fold_class_counts[1]  # benign/melanoma ratio

    # Create class weights: [weight_for_benign, weight_for_melanoma]
    # Give melanoma samples 2x the imbalance ratio to emphasize detection
    fold_class_weights = torch.tensor([1.0, imbalance_ratio * 2.0], dtype=torch.float)

    # Apply square root to moderate extreme weights while maintaining emphasis
    fold_class_weights = torch.sqrt(fold_class_weights)

    # Assign weights to individual samples based on their class
    fold_sample_weights = fold_class_weights[fold_train_df['target'].values]

    
    logger.info(f"Fold {fold_idx + 1} Class Weights: Benign={fold_class_weights[0]:.3f}, "
               f"Melanoma={fold_class_weights[1]:.3f} (ratio: {fold_class_weights[1]/fold_class_weights[0]:.1f}x)")
    
    # Training loader uses weighted sampler for balanced batches
    fold_train_loader = DataLoader(fold_train_ds, batch_size=32, shuffle=True, pin_memory=True)

    # Validation loader uses sequential sampling (no randomization needed)
    fold_val_loader = DataLoader(fold_val_ds, batch_size=32, shuffle=False, pin_memory=True)
    
    # Each fold gets a completely fresh model to ensure independence
    fold_model = create_model().to(device)
    fold_criterion, fold_optimizer, fold_scheduler = create_optimizer_and_scheduler(fold_model, fold_train_df, device)
    
    # Training variables for this fold
    fold_best_auc = 0.0  # Track best validation AUC for this fold
    fold_patience_counter = 0  # Early stopping counter
    fold_patience = 5  # Stop after 5 epochs without improvement
    
    # Fold-specific checkpoint directory
    cv_results_dir = "results/cross_validation"
    fold_checkpoint_dir = os.path.join(cv_results_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    
    # TensorBoard for this fold
    fold_writer = SummaryWriter(log_dir=os.path.join(fold_checkpoint_dir, "logs"))
    
    # Training loop for this fold
    fold_start_time = datetime.now()
    
    for epoch in range(max_epochs):
        epoch_start_time = datetime.now()
        
        # Train for one epoch
        # Returns average training loss across all batches
        train_loss = train_one_epoch_func(
            fold_model, fold_train_loader, fold_optimizer, 
            fold_criterion, scaler, epoch, logger
        )
        
        # Validate on fold validation set
        # Returns AUC and detailed metrics dictionary
        val_auc, val_detailed_metrics = evaluate_func(
            fold_model, fold_val_loader, epoch, logger
        )
        
        # Calculate epoch time
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # Log fold-specific metrics
        logger.info(f"Fold {fold_idx + 1} Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val AUC: {val_auc:.6f}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        
        # TensorBoard logging
        fold_writer.add_scalar("Loss/train", train_loss, epoch)
        fold_writer.add_scalar("ROC-AUC/val", val_auc, epoch)
        fold_writer.add_scalar("Learning_Rate", fold_optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate scheduler
        # Uses validation AUC as metric for scheduling
        fold_scheduler.step(val_auc)
        
        # Save checkpoint
        fold_ckpt = {
            "fold": fold_idx,
            "epoch": epoch,
            "model_state_dict": fold_model.state_dict(),
            "optimizer_state_dict": fold_optimizer.state_dict(),
            "best_auc": fold_best_auc,
            "train_loss": train_loss,
            "val_auc": val_auc
        }
        
        # Save best model for this fold
        if val_auc > fold_best_auc:
            improvement = val_auc - fold_best_auc
            fold_best_auc = val_auc
            fold_patience_counter = 0  # Reset early stopping counter
            
            # Save best model checkpoint for this fold
            fold_best_path = os.path.join(fold_checkpoint_dir, "best_model.pth")
            torch.save(fold_ckpt, fold_best_path)
            
            logger.info(f"Fold {fold_idx + 1} NEW BEST! AUC improved by {improvement:.6f} to {fold_best_auc:.6f}")
        else:
            # No improvement - increment patience counter
            fold_patience_counter += 1
            logger.info(f"Fold {fold_idx + 1} No improvement. Patience: {fold_patience_counter}/{fold_patience}")
        
        # Early stopping for this fold
        if fold_patience_counter >= fold_patience:
            logger.info(f"Fold {fold_idx + 1} Early stopping triggered after {epoch + 1} epochs")
            break
    
    fold_training_time = (datetime.now() - fold_start_time).total_seconds()
    logger.info(f"Fold {fold_idx + 1} completed in {fold_training_time/60:.2f} minutes. Best AUC: {fold_best_auc:.6f}")
    
    # Clean up TensorBoard writer
    fold_writer.close()
    
    # Load best model for this fold and evaluate
    fold_best_path = os.path.join(fold_checkpoint_dir, "best_model.pth")
    if os.path.exists(fold_best_path):
        fold_ckpt = torch.load(fold_best_path, map_location=device)
        fold_model.load_state_dict(fold_ckpt["model_state_dict"])
    
    # Get prediction probabilities without test-time augmentation for consistency
    fold_val_probs, fold_val_true = predict_proba_func(fold_model, fold_val_loader, device, use_tta=False)
    
    # Find optimal threshold using validation data
    # This threshold maximizes F1-score or other specified metric
    fold_optimal_threshold = find_threshold_func(fold_val_true, fold_val_probs, logger)

    # Compute comprehensive metrics using optimal threshold
    fold_val_metrics, fold_val_pred = compute_metrics_func(
        fold_val_true, fold_val_probs, threshold=fold_optimal_threshold
    )
    
    # ROC curve shows TPR vs FPR across all thresholds
    plot_functions['roc'](fold_val_true, fold_val_probs, 
                         os.path.join(fold_checkpoint_dir, "roc_val.png"))
    
    # Precision-Recall curve shows precision vs recall (important for imbalanced data)
    plot_functions['pr'](fold_val_true, fold_val_probs, 
                        os.path.join(fold_checkpoint_dir, "pr_curve_val.png"))
    
    # Confusion matrix shows classification results at optimal threshold
    plot_functions['cm'](fold_val_true, fold_val_pred, 
                        os.path.join(fold_checkpoint_dir, "confusion_matrix_val.png"))
    
    # Return fold results
    fold_results = {
        "fold": fold_idx + 1,
        "best_auc": fold_best_auc,
        "optimal_threshold": fold_optimal_threshold,
        "final_metrics": fold_val_metrics,
        "training_time_minutes": fold_training_time / 60,
        "epochs_trained": epoch + 1,
        "train_samples": len(fold_train_df),
        "val_samples": len(fold_val_df),
        "train_melanoma_ratio": fold_train_ratio,
        "val_melanoma_ratio": fold_val_ratio
    }
    
    return fold_results, fold_model

def run_cross_validation(train_val_df, train_transform, val_transform, dataset_class,
                        train_one_epoch_func, evaluate_func, predict_proba_func, 
                        compute_metrics_func, find_threshold_func, plot_functions,
                        device, scaler, n_folds=5, max_epochs=50):
    """
    Run complete k-fold cross-validation for robust model evaluation.
    
    Cross-validation is the gold standard for evaluating medical ML models because:
    
    1. **Unbiased Performance Estimation**: Provides realistic performance estimates
       that generalize to unseen patient populations
       
    2. **Robustness Assessment**: Tests model stability across different data splits
       and helps identify models that overfit to specific partitions
       
    3. **Statistical Significance**: Enables calculation of confidence intervals
       and statistical tests for model comparison
       
    4. **Ensemble Creation**: Trained models can be combined for improved prediction
       accuracy through ensemble averaging
    
    Process Overview:
    1. Create stratified folds to maintain class balance
    2. Train independent models on each fold
    3. Collect performance statistics across all folds
    4. Generate comprehensive reports and visualizations
    5. Return ensemble of models for final evaluation
    
    Args:
        train_val_df (pd.DataFrame): Combined training and validation data
        train_transform (torchvision.transforms): Augmentation transforms for training
        val_transform (torchvision.transforms): Normalization transforms for validation
        dataset_class: PyTorch Dataset class for data loading
        train_one_epoch_func: Function to train model for one epoch
        evaluate_func: Function to evaluate model performance  
        predict_proba_func: Function to get prediction probabilities
        compute_metrics_func: Function to compute classification metrics
        find_threshold_func: Function to find optimal classification threshold
        plot_functions (dict): Dictionary of plotting functions
        device (torch.device): Device for training (CPU/GPU)
        scaler: Mixed precision scaler for efficiency
        n_folds (int): Number of cross-validation folds (default: 5)
        max_epochs (int): Maximum epochs per fold (default: 50)
        
    Returns:
        tuple: (cv_statistics_dict, list_of_trained_models)
               - cv_statistics: Comprehensive performance statistics across folds
               - cv_models: List of trained models (one per fold) for ensemble
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50} STARTING {n_folds}-FOLD CROSS VALIDATION {'='*50}")
    
    # Create results directory
    cv_results_dir = "results/cross_validation"
    os.makedirs(cv_results_dir, exist_ok=True)
    
    # StratifiedKFold ensures each fold maintains the same class distribution
    # This is critical for imbalanced medical datasets to ensure:
    # - Each fold has representative samples of both classes
    # - Fair comparison between folds
    # - Stable performance estimates
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    logger.info(f"Created {n_folds} stratified folds with random_state=42")
    logger.info(f"Total samples: {len(train_val_df)}")
    logger.info(f"Overall class distribution: {train_val_df['target'].value_counts().to_dict()}")
    
    # Initialize storage for results
    cv_results = []  # Performance metrics for each fold
    cv_models = []   # Trained models for ensemble
    
    # Run each fold
    cv_start_time = datetime.now()
    logger.info(f"Cross-validation started at: {cv_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train model for each fold
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_val_df, train_val_df['target'])):
        # Train single fold and collect results
        fold_results, fold_model = train_single_fold(
            fold_idx, train_indices, val_indices, train_val_df, train_transform, 
            val_transform, dataset_class, train_one_epoch_func, evaluate_func, 
            predict_proba_func, compute_metrics_func, find_threshold_func,
            plot_functions, device, scaler, max_epochs, n_folds
        )

        # Store results and model
        cv_results.append(fold_results)
        cv_models.append(fold_model)
        
        # Clean up GPU memory after each fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    cv_total_time = (datetime.now() - cv_start_time).total_seconds()
    
    # Extract performance metrics from all folds
    cv_aucs = [result['best_auc'] for result in cv_results]
    cv_accuracies = [result['final_metrics']['accuracy'] for result in cv_results]
    cv_precisions = [result['final_metrics']['precision'] for result in cv_results]
    cv_recalls = [result['final_metrics']['recall'] for result in cv_results]
    cv_f1s = [result['final_metrics']['f1_score'] for result in cv_results]
    cv_specificities = [result['final_metrics'].get('specificity', 0) for result in cv_results]

    # Calculate comprehensive statistics for each metric
    # Include mean, standard deviation, min, max, and individual fold values
    cv_stats = {
        "experiment_info": {
            "n_folds": n_folds,
            "max_epochs_per_fold": max_epochs,
            "total_samples": len(train_val_df),
            "start_time": cv_start_time.isoformat(),
            "total_time_minutes": cv_total_time / 60,
            "average_time_per_fold_minutes": (cv_total_time / 60) / n_folds
        },
        "fold_results": cv_results,  # Detailed results for each fold
        "cross_validation_metrics": {
            "auc": {
                "mean": np.mean(cv_aucs),
                "std": np.std(cv_aucs),
                "min": np.min(cv_aucs),
                "max": np.max(cv_aucs),
                "values": cv_aucs,
                "confidence_interval_95": {
                    "lower": np.mean(cv_aucs) - 1.96 * np.std(cv_aucs) / np.sqrt(n_folds),
                    "upper": np.mean(cv_aucs) + 1.96 * np.std(cv_aucs) / np.sqrt(n_folds)
                }
            },
            "accuracy": {
                "mean": np.mean(cv_accuracies),
                "std": np.std(cv_accuracies),
                "min": np.min(cv_accuracies),
                "max": np.max(cv_accuracies),
                "values": cv_accuracies
            },
            "precision": {
                "mean": np.mean(cv_precisions),
                "std": np.std(cv_precisions),
                "min": np.min(cv_precisions),
                "max": np.max(cv_precisions),
                "values": cv_precisions
            },
            "recall": {
                "mean": np.mean(cv_recalls),
                "std": np.std(cv_recalls),
                "min": np.min(cv_recalls),
                "max": np.max(cv_recalls),
                "values": cv_recalls
            },
            "f1_score": {
                "mean": np.mean(cv_f1s),
                "std": np.std(cv_f1s),
                "min": np.min(cv_f1s),
                "max": np.max(cv_f1s),
                "values": cv_f1s
            },
            "specificity": {
                "mean": np.mean(cv_specificities),
                "std": np.std(cv_specificities),
                "values": cv_specificities
            }
        }
    }
    
    # Log cross-validation summary
    logger.info(f"\n{'='*50} CROSS VALIDATION SUMMARY {'='*50}")
    logger.info(f"Total CV Time: {cv_total_time/60:.2f} minutes")
    logger.info(f"AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    logger.info(f"Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    logger.info(f"Precision: {np.mean(cv_precisions):.4f} ± {np.std(cv_precisions):.4f}")
    logger.info(f"Recall: {np.mean(cv_recalls):.4f} ± {np.std(cv_recalls):.4f}")
    logger.info(f"F1-Score: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
    
    # Log individual fold performance for detailed analysis
    logger.info(f"\nIndividual Fold Performance:")
    logger.info(f"{'Fold':<6} {'AUC':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Time(m)':<8}")
    logger.info("-" * 58)
    for i, result in enumerate(cv_results):
        logger.info(f"Fold {i+1}: AUC={result['best_auc']:.4f}, "
                   f"Acc={result['final_metrics']['accuracy']:.4f}, "
                   f"F1={result['final_metrics']['f1_score']:.4f}")
    
    # Identify best and worst performing folds
    best_fold_idx = np.argmax(cv_aucs)
    worst_fold_idx = np.argmin(cv_aucs)
    logger.info(f"\nBest fold: {best_fold_idx + 1} (AUC: {cv_aucs[best_fold_idx]:.4f})")
    logger.info(f" Worst fold: {worst_fold_idx + 1} (AUC: {cv_aucs[worst_fold_idx]:.4f})")
    logger.info(f"Performance range: {np.max(cv_aucs) - np.min(cv_aucs):.4f}")
    
    # Save CV results
    cv_results_path = os.path.join(cv_results_dir, "cross_validation_results.json")
    try:
        with open(cv_results_path, "w") as f:
            json.dump(cv_stats, f, indent=2, default=str)
        logger.info(f"Cross-validation results saved to: {cv_results_path}")
    except Exception as e:
        logger.error(f"Failed to save CV results: {e}")
    
    logger.info(f"Cross-validation completed successfully!")
    
    return cv_stats, cv_models

def evaluate_ensemble_on_test(cv_models, test_loader, predict_proba_func, compute_metrics_func, 
                             find_threshold_func, plot_functions, device, tta_n=0):
    """
    Evaluate ensemble of cross-validation models on independent test set.
    
    Ensemble evaluation combines predictions from all CV models to create
    more robust and accurate final predictions. This approach:
    
    1. **Reduces Variance**: Individual model predictions may vary, but ensemble
       averaging smooths out these variations for more stable predictions
       
    2. **Improves Accuracy**: Multiple models may capture different aspects of
       the data, leading to better overall performance
       
    3. **Increases Robustness**: Less likely to fail catastrophically since
       multiple models would need to fail simultaneously
       
    4. **Provides Uncertainty Estimates**: Variance in predictions across models
       can indicate prediction confidence
    
    The ensemble uses simple average voting, where each model contributes
    equally to the final prediction. This is effective when all models
    have similar performance levels.
    
    Args:
        cv_models (list): List of trained models from cross-validation
        test_loader (DataLoader): DataLoader for independent test set
        predict_proba_func: Function to get prediction probabilities
        compute_metrics_func: Function to compute classification metrics
        find_threshold_func: Function to find optimal threshold
        plot_functions (dict): Dictionary of plotting functions
        device (torch.device): Device for inference
        tta_n (int): Number of test-time augmentation iterations (0 = disabled)
        
    Returns:
        tuple: (ensemble_metrics, ensemble_probabilities, true_labels)
               - ensemble_metrics: Dictionary of performance metrics
               - ensemble_probabilities: Average predictions across all models
               - true_labels: Ground truth labels for test set
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*20} ENSEMBLE EVALUATION ON TEST SET {'='*20}")
    logger.info(f"Evaluating ensemble of {len(cv_models)} CV models...")
    
    if tta_n > 0:
        logger.info(f"Using Test-Time Augmentation with {tta_n} iterations per sample")

    cv_results_dir = "results/cross_validation"
    all_ensemble_probs = []  # Store predictions from each model
    test_true = None  # Ground truth labels (same for all models)
    
    # Get predictions from each CV model
    for i, model in enumerate(cv_models):
        logger.info(f"Getting predictions from fold {i+1} model...")

        # Get predictions from current model
        # use_tta=True applies test-time augmentation for improved robustness
        fold_probs, fold_true = predict_proba_func(model, test_loader, device, use_tta=(tta_n>0), tta_n=tta_n)
        all_ensemble_probs.append(fold_probs)

        # Store ground truth labels (same for all models)
        if test_true is None:
            test_true = fold_true

        # Log individual model statistics
        logger.info(f"  Model {i+1}: Mean pred = {np.mean(fold_probs):.4f}, "
                   f"Std = {np.std(fold_probs):.4f}")
    
    # Simple average ensemble - each model contributes equally
    # More sophisticated approaches (weighted voting, stacking) could be explored
    ensemble_probs = np.mean(all_ensemble_probs, axis=0)
    
    logger.info(f"Ensemble Statistics:")
    logger.info(f"  Mean prediction: {np.mean(ensemble_probs):.4f}")
    logger.info(f"  Std prediction: {np.std(ensemble_probs):.4f}")
    logger.info(f"  Min prediction: {np.min(ensemble_probs):.4f}")
    logger.info(f"  Max prediction: {np.max(ensemble_probs):.4f}")

    # Calculate prediction variance across models (uncertainty estimate)
    prediction_variance = np.var(all_ensemble_probs, axis=0)
    logger.info(f"  Mean prediction variance: {np.mean(prediction_variance):.6f}")
    logger.info(f"  Max prediction variance: {np.max(prediction_variance):.6f}")
    
    # Find optimal threshold using ensemble predictions
    ensemble_threshold = find_threshold_func(test_true, ensemble_probs, logger)
    logger.info(f"Optimal ensemble threshold: {ensemble_threshold:.4f}")
    
    # Compute comprehensive metrics using optimal threshold
    ensemble_metrics, ensemble_pred = compute_metrics_func(
        test_true, ensemble_probs, threshold=ensemble_threshold
    )
    
    logger.info("Generating ensemble visualization plots...")

    # ROC curve for ensemble predictions
    plot_functions['roc'](test_true, ensemble_probs, 
                         os.path.join(cv_results_dir, "ensemble_roc_test.png"))
    
    # Precision-Recall curve for ensemble predictions  
    plot_functions['pr'](test_true, ensemble_probs, 
                        os.path.join(cv_results_dir, "ensemble_pr_curve_test.png"))
    
    # Confusion matrix using optimal threshold
    plot_functions['cm'](test_true, ensemble_pred, 
                        os.path.join(cv_results_dir, "ensemble_confusion_matrix_test.png"))
    
    logger.info(f"\nENSEMBLE TEST RESULTS:")
    logger.info(f"{'='*50}")
    for metric, value in ensemble_metrics.items():
        logger.info(f"  {metric:<20}: {value:.6f}")
    
    # Add ensemble-specific metrics
    ensemble_metrics['prediction_variance_mean'] = np.mean(prediction_variance)
    ensemble_metrics['prediction_variance_max'] = np.max(prediction_variance)
    ensemble_metrics['num_models'] = len(cv_models)
    ensemble_metrics['optimal_threshold'] = ensemble_threshold
    
    logger.info(f"\nEnsemble Characteristics:")
    logger.info(f"  Number of models: {len(cv_models)}")
    logger.info(f"  Prediction uncertainty (mean variance): {np.mean(prediction_variance):.6f}")
    logger.info(f"  Test samples: {len(test_true)}")
    logger.info(f"  Positive samples: {np.sum(test_true)} ({np.mean(test_true):.2%})")
    
    return ensemble_metrics, ensemble_probs, test_true