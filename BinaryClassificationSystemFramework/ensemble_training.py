"""
Ensemble Training Module for Enhanced Performance

This module implements ensemble training strategies to improve model robustness
and accuracy through diversity in model initialization, architecture variations,
or training procedures.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import os
from models import create_melanoma_model, create_advanced_melanoma_model
from training import train_one_epoch, evaluate_with_threshold_search
from evaluation import predict_proba, compute_classification_metrics


def create_ensemble_models(n_models: int, device: torch.device, 
                          diversity_strategy: str = "varied_init") -> List[nn.Module]:
    """
    Create diverse models for ensemble.
    
    Args:
        n_models: Number of models to create
        device: Training device
        diversity_strategy: How to introduce diversity
        
    Returns:
        List of initialized models
    """
    models = []
    
    for i in range(n_models):
        if diversity_strategy == "varied_init":
            # Use different random seeds for initialization
            torch.manual_seed(42 + i * 100)
            model = create_advanced_melanoma_model(
                out_dim=2,
                pretrained=True,
                dropout_rate=0.5,
                num_dropouts=5
            ).to(device)
            
        elif diversity_strategy == "dropout_variation":
            # Vary dropout rates for each model
            dropout_rate = 0.3 + (i * 0.1)  # 0.3, 0.4, 0.5, etc.
            model = create_advanced_melanoma_model(
                out_dim=2,
                pretrained=True,
                dropout_rate=dropout_rate,
                num_dropouts=5
            ).to(device)
        else:
            # Default: use standard advanced model
            model = create_advanced_melanoma_model(
                out_dim=2,
                pretrained=True,
                dropout_rate=0.5,
                num_dropouts=5
            ).to(device)
            
        models.append(model)
    
    return models


def train_ensemble(models: List[nn.Module], train_loader, val_loader, 
                  optimizers: List, schedulers: List, criterion,
                  scaler, device, max_epochs: int, patience: int,
                  checkpoint_dir: str, logger) -> Tuple[List[nn.Module], List[Dict]]:
    """
    Train multiple models independently as an ensemble.
    
    Args:
        models: List of models to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizers: List of optimizers (one per model)
        schedulers: List of schedulers (one per model)
        criterion: Loss function
        scaler: Mixed precision scaler
        device: Training device
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        checkpoint_dir: Directory for checkpoints
        logger: Logger instance
        
    Returns:
        Tuple of (trained_models, training_histories)
    """
    training_histories = []
    
    for idx, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING ENSEMBLE MODEL {idx + 1}/{len(models)}")
        logger.info(f"{'='*60}")
        
        best_auc = 0.0
        patience_counter = 0
        model_history = {
            'train_losses': [],
            'val_aucs': [],
            'best_auc': 0.0,
            'best_epoch': 0
        }
        
        for epoch in range(max_epochs):
            # Training phase
            train_loss = train_one_epoch(model, train_loader, optimizer, 
                                        criterion, scaler, epoch, logger)
            
            # Validation phase
            val_auc, val_metrics = evaluate_with_threshold_search(model, val_loader, 
                                                                  epoch, logger)
            
            # Update learning rate
            scheduler.step(val_auc)
            
            # Track history
            model_history['train_losses'].append(train_loss)
            model_history['val_aucs'].append(val_auc)
            
            logger.info(f"Model {idx+1} - Epoch {epoch+1}: "
                       f"Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                improvement = val_auc - best_auc
                best_auc = val_auc
                patience_counter = 0
                
                model_history['best_auc'] = best_auc
                model_history['best_epoch'] = epoch + 1
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, 
                                              f"ensemble_model_{idx+1}_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_auc': best_auc
                }, checkpoint_path)
                
                logger.info(f"Model {idx+1}: NEW BEST! AUC improved by "
                          f"{improvement:.6f} to {best_auc:.6f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Model {idx+1}: Early stopping at epoch {epoch+1}")
                break
        
        training_histories.append(model_history)
        logger.info(f"Model {idx+1} training completed. Best AUC: {best_auc:.6f}")
    
    return models, training_histories


def ensemble_predict(models: List[nn.Module], loader, device: torch.device,
                     aggregation: str = "average", use_tta: bool = False,
                     tta_n: int = 0, tta_transform=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ensemble predictions.
    
    Args:
        models: List of trained models
        loader: Data loader
        device: Inference device
        aggregation: How to combine predictions ("average", "weighted", "voting")
        use_tta: Whether to use test-time augmentation
        tta_n: Number of TTA iterations
        tta_transform: TTA transform
        
    Returns:
        Tuple of (ensemble_probabilities, true_labels)
    """
    all_predictions = []
    true_labels = None
    
    # Get predictions from each model
    for idx, model in enumerate(models):
        probs, labels = predict_proba(
            model=model,
            loader=loader,
            device=device,
            use_tta=use_tta,
            tta_n=tta_n,
            tta_transform=tta_transform
        )
        all_predictions.append(probs)
        
        if true_labels is None:
            true_labels = labels
    
    # Stack predictions: (n_models, n_samples)
    predictions_stack = np.stack(all_predictions, axis=0)
    
    # Aggregate predictions
    if aggregation == "average":
        ensemble_probs = np.mean(predictions_stack, axis=0)
    elif aggregation == "weighted":
        # Weight by validation AUC (would need to pass weights)
        ensemble_probs = np.mean(predictions_stack, axis=0)
    elif aggregation == "voting":
        # Hard voting: majority vote
        binary_preds = (predictions_stack > 0.5).astype(int)
        ensemble_probs = np.mean(binary_preds, axis=0)
    else:
        ensemble_probs = np.mean(predictions_stack, axis=0)
    
    return ensemble_probs, true_labels


def evaluate_ensemble(models: List[nn.Module], test_loader, device: torch.device,
                     optimal_threshold: float, aggregation: str = "average",
                     use_tta: bool = False, tta_n: int = 0, tta_transform=None,
                     logger=None) -> Dict:
    """
    Evaluate ensemble on test set.
    
    Args:
        models: List of trained models
        test_loader: Test data loader
        device: Inference device
        optimal_threshold: Threshold for binary classification
        aggregation: Aggregation strategy
        use_tta: Whether to use TTA
        tta_n: Number of TTA iterations
        tta_transform: TTA transform
        logger: Logger instance
        
    Returns:
        Dictionary with ensemble metrics
    """
    if logger:
        logger.info("Evaluating ensemble on test set...")
    
    # Get ensemble predictions
    ensemble_probs, test_true = ensemble_predict(
        models=models,
        loader=test_loader,
        device=device,
        aggregation=aggregation,
        use_tta=use_tta,
        tta_n=tta_n,
        tta_transform=tta_transform
    )
    
    # Compute metrics
    ensemble_metrics, ensemble_pred = compute_classification_metrics(
        test_true, ensemble_probs, threshold=optimal_threshold
    )
    
    if logger:
        logger.info("Ensemble Test Metrics:")
        for metric, value in ensemble_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    return {
        'metrics': ensemble_metrics,
        'predictions': ensemble_pred,
        'probabilities': ensemble_probs,
        'true_labels': test_true
    }