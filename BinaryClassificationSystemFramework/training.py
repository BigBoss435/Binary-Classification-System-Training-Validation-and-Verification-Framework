import torch
import numpy as np
from tqdm import tqdm

import config
from utils import get_device
from datetime import datetime
from evaluation import compute_classification_metrics, find_optimal_threshold
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from config import USE_AMP, ADV_TRAINING, ADV_EPS

def fgsm_attack(model, images, labels, epsilon, criterion):
    images = images.clone().detach().requires_grad_(True)

    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, logger):
    """
    Train model for one epoch using mixed precision.

    Args:
        model: PyTorch model to train
        loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        scaler: GradScaler for mixed precision
        epoch: Current epoch number (for logging)
        logger: Logger instance

    Returns:
        float: Average training loss for the epoch
    """
    device = get_device()
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss = 0  # Accumulator for total loss across all batches
    step_losses = []  # List to store individual batch losses for statistics
    progress = tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=False)  # Progress bar for visual feedback

    # Log training start
    logger.info(f"Starting training epoch {epoch+1} with {len(loader)} batches.")

    for step, (imgs, labels) in enumerate(progress):
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        # ----------------------------
        # ADVERSARIAL TRAINING (FGSM)
        # ----------------------------
        if ADV_TRAINING:
            # Generate adversarial batch (clean → adversarial)
            adv_imgs = fgsm_attack(model, imgs, labels, ADV_EPS, criterion)

            # Combine clean + adversarial
            imgs = torch.cat([imgs, adv_imgs], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        try:
            # Forward pass with automatic mixed precision
            if USE_AMP:
                # Mixed precision forward pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs)  # Model predictions
                    loss = criterion(outputs, labels)  # Calculate loss
                # Backward pass with gradient scaling for mixed precision
                scaler.scale(loss).backward()  # Scale loss to prevent underflow
                scaler.step(optimizer)  # Update weights with scaled gradients
                scaler.update()  # Update scaler for next iteration
            else:
                # Standard precision training (CPU or when AMP is disabled)
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

            # Track loss statistics
            batch_loss = loss.item()
            running_loss += batch_loss * imgs.size(0)  # Accumulate los
            step_losses.append(batch_loss)

            # Update progress bar with current metrics
            progress.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{np.mean(step_losses[-10:]):.4f}',  # Moving average
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Detailed logging every N steps
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                monitor_training_step(epoch, step, batch_loss, current_lr, logger, log_interval=100)

            # Memory monitoring for potential issues
            if step % 200 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear unused GPU memory periodically
                
        except RuntimeError as e:
            logger.error(f"Training error at epoch {epoch+1}, step {step}: {e}")
            if "out of memory" in str(e):
                logger.error("GPU out of memory. Consider reducing batch size or model size.")
                torch.cuda.empty_cache()
            raise e

    # Calculate epoch statistics
    avg_loss = running_loss / len(loader.dataset)  # Average loss per sample
    loss_std = np.std(step_losses)  # Standard deviation of batch losses
    
    # Log epoch completion summary
    logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.6f} ± {loss_std:.6f}, "
               f"Steps: {len(step_losses)}")
    
    return avg_loss

def evaluate_with_threshold_search(model, loader, epoch, logger):
    """Enhanced evaluation with better metrics for imbalanced data."""
    device = get_device()
    model.eval()
    preds, targets = [], []
    
    # Collect all predictions and targets
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            batch_preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds.extend(batch_preds)
            targets.extend(labels.numpy())
    
    # Calculate class distribution
    melanoma_count = sum(targets)
    benign_count = len(targets) - melanoma_count
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(targets, preds, logger)
    
    # Calculate metrics with optimal threshold
    metrics, y_pred = compute_classification_metrics(targets, preds, threshold=optimal_threshold)
    
    # Additional imbalanced data metrics
    avg_precision = average_precision_score(targets, preds)
    balanced_acc = balanced_accuracy_score(targets, y_pred)
    
    # Enhanced logging
    logger.info(f"Epoch {epoch+1} Validation:")
    logger.info(f"  Dataset: {benign_count} benign, {melanoma_count} melanoma ({melanoma_count/(melanoma_count+benign_count)*100:.1f}% melanoma)")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.6f}")
    logger.info(f"  Average Precision: {avg_precision:.6f}")
    logger.info(f"  Balanced Accuracy: {balanced_acc:.6f}")
    logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.6f}")
    logger.info(f"  Precision: {metrics['precision']:.6f}")
    logger.info(f"  Recall: {metrics['recall']:.6f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.6f}")
    
    # Prediction distribution analysis
    melanoma_preds = sum(y_pred)
    logger.info(f"  Predictions: {len(y_pred)-melanoma_preds} benign, {melanoma_preds} melanoma")
    
    return metrics['roc_auc'], metrics

def evaluate(model, loader, epoch, logger):
    """
    Evaluate model performance on validation set.
    
    Performs inference on validation data without gradient computation
    for memory efficiency and speed. Calculates ROC-AUC score.

    Args:
        model: PyTorch model to evaluate
        loader: DataLoader for validation data
        epoch: Current epoch number (for logging)
        logger: Logger instance for monitoring

    Returns:
        float: ROC-AUC score (0.5 = random, 1.0 = perfect)
    """
    device = get_device()
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm in eval mode)
    preds, targets = [], []  # Store predictions and ground truth labels
    total_samples = 0  # Counter for porcessed samples

    # Progress bar for validation feedback
    progress = tqdm(loader, desc=f"Validating Epoch {epoch+1}", leave=False)

    logger.info(f"Starting validation for epoch {epoch+1}")

    # Time validation process
    start_time = datetime.now()

    # Disable gradient computation for faster evaluation and memory efficiency
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(progress):
            try:
                # Move images to GPU
                imgs = imgs.to(device, non_blocking=True)
                outputs = model(imgs)
                batch_preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                batch_targets = labels.numpy()
                
                # Accumulate predictions and targets
                preds.extend(batch_preds)
                targets.extend(batch_targets)
                total_samples += len(labels)
                
                # Update progress
                progress.set_postfix({'samples': total_samples})
                
            except RuntimeError as e:
                logger.error(f"Validation error at batch {batch_idx}: {e}")
                raise e

    # Calculate metrics
    try:
        # Calculate ROC-AUC score (standard metric for binary classification)
        # AUC measures model's ability to distinguish between classes
        auc = roc_auc_score(targets, preds)
        eval_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Validation completed - AUC: {auc:.6f}, "
                   f"Samples: {total_samples}, Time: {eval_time:.2f}s")
        
        return auc

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return 0.0
    
def monitor_training_step(epoch, step, loss, lr, logger, log_interval=50):
    """
    Monitor training progress with detailed logging at specified intervals.
    
    Provides real-time feedback on training progress, loss trends, and resource usage
    to help identify training issues early.
    
    Args:
        epoch: Current epoch number
        step: Current step/batch number
        loss: Current batch loss value
        lr: Current learning rate
        logger: Logger instance for output
        log_interval: How often to log (every N steps)
    """
    if step % log_interval == 0:
        logger.info(f"Epoch {epoch+1} Step {step}: Loss={loss:.6f}, LR={lr:.2e}")
        
        # Log GPU memory usage during training for memory leak detection
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # Currently allocated GPU memory in GB
            gpu_cached = torch.cuda.memory_reserved() / (1024**3)  # Cached memory
            logger.debug(f"GPU Memory - Allocated: {gpu_memory:.2f}GB, Cached: {gpu_cached:.2f}GB")
