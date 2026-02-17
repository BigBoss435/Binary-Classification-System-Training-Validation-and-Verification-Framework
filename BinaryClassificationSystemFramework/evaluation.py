import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from config import USE_AMP
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)

# ---------------------------
# Metric and plot functions
# ---------------------------
def predict_proba(model, loader, device, use_tta=False, tta_n=0, tta_transform=None):
    """
    Generate probability predictions with optional Test Time Augmentation (TTA).

    TTA improves prediction reliability by averaging predictions from multiple 
    augmented version of each image.

    Args:
        model: Trained PyTorch model
        loader: DataLoader for prediction
        device: Device to run inference on (CPU or GPU)
        use_tta: Whether to use Test Time Augmentation
        tta_n: Number of TTA iterations
        tta_transform: Transform to apply for TTA (required if use_tta=True)

    Returns:
        tuple: (predicted_probabilities, true_labels) as numpy arrays
    """
    model.eval()  # Set model to evaluation mode
    all_probs, all_targets =[], []  # Storage for predictions and labels

    # Validate TTA parameters
    if use_tta and tta_n > 0 and tta_transform is None:
        raise ValueError("tta_transform must be provided when use_tta=True")

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Predicting", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.numpy()
            if use_tta and tta_n > 0:
                # Test Time Augmentation: multiple predictions per image
                batch_preds = None
                for _ in range(tta_n):
                    # Convert tensors back to PIL for TTA transforms
                    # This is computationally expensive but improves accuracy
                    imgs_cpu = imgs.cpu()
                    tta_batch = []
                    for t in imgs_cpu:
                        # Denormalize tensor back to [0,255] range
                        arr = (t * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))
                        arr = (arr.numpy().transpose(1,2,0) * 255).astype("uint8")
                        pil = Image.fromarray(arr)
                        img_np = np.array(pil)
                        augmented = tta_transform(image=img_np)
                        aug = augmented["image"].unsqueeze(0)
                        # Apply TTA transform and add to batch
                        tta_batch.append(aug)

                    # Combine augmented images into batch tensor
                    tta_batch = torch.cat(tta_batch, dim=0).to(device)

                    # Forward pass with optional mixed precision
                    if USE_AMP:
                        with torch.amp.autocast(device_type="cuda"):
                            logits = model(tta_batch)
                    else:
                        logits = model(tta_batch)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                    # Accumulate predictions for averaging
                    if batch_preds is None:
                        batch_preds = probs
                    else:
                        batch_preds += probs

                # Average all TTA predictions
                batch_preds /= float(tta_n)
                all_probs.extend(batch_preds.tolist())
            else:
                # Standard single inference without TTA
                if USE_AMP:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(imgs)
                else:
                    logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())

            all_targets.extend(labels.tolist())

    return np.array(all_probs), np.array(all_targets, dtype=int)

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute comprehensive classification metrics for binary classification.
    
    Calculates standard metrics used to evaluate binary classifiers including
    ROC-AUC, accuracy, precision, recall, and F1-score.

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (0.0 to 1.0)
        threshold: Decision threshold for binary classification (default 0.5)

    Returns:
        tuple: (metrics_dict, binary_predictions)
    """
    # Convert probabilities to binary predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate all standard classification metrics
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),  # Area under ROC curve
        "accuracy": accuracy_score(y_true, y_pred),  # Overall correctness
        "precision": precision_score(y_true, y_pred, zero_division=0),  # True positive rate
        "recall": recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        "f1_score": f1_score(y_true, y_pred, zero_division=0),  # Harmonic mean of precision and recall
        "threshold": threshold
    }
    return metrics, y_pred

def find_optimal_threshold(y_true, y_prob, logger=None):
    """Find threshold that balances precision and recall for imbalanced data"""
    
    # Use precision-recall curve for imbalanced data
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Ensure we're not using an extreme threshold
    optimal_threshold = np.clip(optimal_threshold, 0.1, 0.9)
    
    logger.info(f"Optimal F1 threshold: {optimal_threshold:.4f}")
    logger.info(f"At this threshold - Precision: {precision[optimal_idx]:.4f}, Recall: {recall[optimal_idx]:.4f}")
    
    return optimal_threshold

def plot_roc_curve(y_true, y_prob, out_path):
    """
    Generate and save ROC curve plot.

    ROC curves show the trade-off between sensitivity (true positive rate)
    and specificity (1 - false positive rate) across all classification thresholds.
    The area under the curve (AUC) summarizes the model's discriminative ability.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        out_path: File path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_precision_recall_curve(y_true, y_prob, out_path):
    """Generate and save Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, out_path):
    """
    Generate and save confusion matrix plot.

    Confusion matrix shows counts of true/false positives/negatives,
    helping understand specific types of classification errors.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (after threshold)
        out_path: File path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar()

    # Set tick marks and labels
    tick_marks = [0,1]
    plt.xticks(tick_marks, ['Benign (0)', 'Melanoma (1)'])
    plt.yticks(tick_marks, ['Benign (0)', 'Melanoma (1)'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Annotate cells with counts and percentages
    total_samples = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i,j]
            percentage = (count / total_samples) * 100
            plt.text(j, i, f'{count}\n({percentage:.1f}%)', 
                    ha="center", va="center", fontsize=12,
                    color="white" if count > cm.max()/2 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')  # High DPI for publication quality
    plt.close()  # Close figure to free memory
